# rlhf_ppo/train_ppo.py
"""
PPO fine-tuning of a pretrained NanoGPT with an EXACT reward model (the true HMM).

Setup (a token-level contextual bandit):
  * context = a real data prefix z_0..z_t   (so the context does not depend on the policy)
  * action  = the next token a_t
  * reward  = log p*(a_t | context) - log pi_theta(a_t | context)
              where p* is the exact Bayes-optimal next-token distribution (hmm_oracle.py).
E_{a~pi}[reward] = -KL(pi || p*), maximised only by pi = p*. One-step decisions => advantage = r - baseline.

Two ways to evaluate the PPO objective (set MODE below):
  "exact"   Because the oracle gives the reward of EVERY token, the expectation over actions is computed
            exactly:  L = -sum_a pi_old(a) * min(rho_a A_a, clip(rho_a) A_a).  Same clipped PPO objective,
            but no action-sampling noise and no critic (the exact baseline is -KL(pi_old || p*)).
  "sampled" Classic PPO: draw ONE action per position from the policy, use a learned critic as baseline.
"""
import csv
import math
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim

from actor_critic import ActorCriticNanoGPT
from dataset import HDF5HMMDataset
from hmm_oracle import HMMOracle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'NanoGPT')))
from model import GPTConfig

# ----------------------------------------------------------------------------- settings
MODE = "exact"                  # "exact" | "sampled"

num_epochs = 50
num_batches_per_epoch = 10
batch_size = 32
ppo_epochs = 4                  # clipped passes per rollout
clip_eps = 0.2
target_kl = 0.01                # stop the passes early if the policy moves further than this (per token)
lr_policy = 5e-5                # raise/lower until the clip fraction sits around 0.02-0.2
lr_min_frac = 0.1               # cosine decay down to this fraction of lr_policy
weight_decay = 0.0              # no decay: we are fine-tuning a pretrained model
# sampled mode only
lr_value = 1e-3
value_coef = 0.5
normalize_advantages = False    # std-normalising removes the signal that tells us we are close to optimal

# ----------------------------------------------------------------------------- setup
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device} | mode: {MODE}")

# IMPORTANT: point at ONE explicit data file (prepare_data/ holds several .h5 files).
data_file = os.path.abspath("../prepare_data/data_d64_m128_lamda0.9_example.h5")
ckpt_path = os.path.abspath("../NanoGPT/m128_d64_4l4h/ckpt.pt")

checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])

policy = ActorCriticNanoGPT(gptconf).to(device)
state_dict = checkpoint['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
policy.gpt.load_state_dict(state_dict)

oracle = HMMOracle.from_h5(data_file, device=device)
assert gptconf.vocab_size == oracle.m, (
    f"Model vocab_size={gptconf.vocab_size} but the data file has m={oracle.m} observation symbols: "
    f"checkpoint and data file do not belong together.")

train_set = HDF5HMMDataset(h5_path=data_file, seq_len=gptconf.block_size, key='z_train')
val_set = HDF5HMMDataset(h5_path=data_file, seq_len=gptconf.block_size, key='z_validation')
assert train_set.key == 'z_train' and val_set.key == 'z_validation', "unexpected dataset keys"

backbone_params = [p for n, p in policy.named_parameters() if not n.startswith("value_head")]
value_params = list(policy.value_head.parameters())
optimizer = optim.AdamW([
    {"params": backbone_params, "lr": lr_policy, "weight_decay": weight_decay},
    {"params": value_params, "lr": lr_value, "weight_decay": 0.0},
])
total_steps = num_epochs * num_batches_per_epoch * ppo_epochs
cosine = lambda s: lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[cosine, lambda s: 1.0])

# ----------------------------------------------------------------------------- fixed validation metrics
val_tokens = val_set.data.to(device)                       # the ENTIRE validation split, identical every time
val_inputs, val_targets = val_tokens[:, :-1].contiguous(), val_tokens[:, 1:].contiguous()
val_p_star = torch.cat([oracle.next_token_probs(val_inputs[i:i + 100]) for i in range(0, len(val_inputs), 100)])
val_logp_star = val_p_star.gather(-1, val_targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-30).log()
OPTIMAL_CE = (-val_logp_star).mean().item()                # computed here, no hard-coded constant
print(f"Optimal validation CE (exact HMM filter): {OPTIMAL_CE:.4f}")


@torch.no_grad()
def evaluate(model):
    """Deterministic metrics on the full validation split."""
    model.eval()
    ce_sum, kl_pi_p, kl_p_pi, n = 0.0, 0.0, 0.0, 0
    for i in range(0, len(val_inputs), 100):
        logits, _, _ = model(val_inputs[i:i + 100])
        logp = F.log_softmax(logits, dim=-1)
        p_star = val_p_star[i:i + 100]
        log_p_star = p_star.clamp_min(1e-30).log()
        tgt = val_targets[i:i + 100]
        ce_sum += -logp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        kl_pi_p += (logp.exp() * (logp - log_p_star)).sum(-1).sum().item()      # KL(pi || p*): the PPO objective
        kl_p_pi += (p_star * (log_p_star - logp)).sum(-1).sum().item()          # KL(p* || pi): what the CE gap measures
        n += tgt.numel()
    return {"val_ce": ce_sum / n, "kl_pi_to_oracle": kl_pi_p / n, "kl_oracle_to_pi": kl_p_pi / n}


m0 = evaluate(policy)
print(f"Before PPO: val CE {m0['val_ce']:.4f} (optimal {OPTIMAL_CE:.4f}, gap {m0['val_ce'] - OPTIMAL_CE:+.4f}) | "
      f"KL(pi||p*) {m0['kl_pi_to_oracle']:.4f} | KL(p*||pi) {m0['kl_oracle_to_pi']:.4f}")
print("\n--- Starting PPO Loop ---")

# ----------------------------------------------------------------------------- PPO
policy.eval()   # PPO needs a deterministic probability ratio: no dropout
log_rows = [{"epoch": 0, **m0, "optimal_ce": OPTIMAL_CE}]

for epoch in range(num_epochs):
    stats = defaultdict(list)

    for _ in range(num_batches_per_epoch):
        batch = train_set.get_batch(batch_size=batch_size).to(device)
        inputs = batch[:, :-1].contiguous()

        # ---- 1. Rollout: freeze the old policy and score with the exact oracle (no gradients) ----
        with torch.no_grad():
            old_logits, old_values, _ = policy(inputs)
            old_logp_all = F.log_softmax(old_logits, dim=-1)           # (b, t, V)
            old_p = old_logp_all.exp()
            log_p_star = oracle.next_token_probs(inputs).clamp_min(1e-12).log()
            r_all = log_p_star - old_logp_all                          # reward of EVERY action

            if MODE == "exact":
                baseline = (old_p * r_all).sum(-1, keepdim=True)       # exactly -KL(pi_old || p*) per position
                adv_all = r_all - baseline
                batch_reward = baseline.mean().item()
            else:
                b, t, V = old_p.shape
                actions = torch.multinomial(old_p.reshape(-1, V), 1).reshape(b, t)
                old_logp = old_logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                rewards = r_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                advantages = rewards - old_values                      # one-step decisions: gamma = 0
                if normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                batch_reward = rewards.mean().item()

        # ---- 2. Several clipped PPO passes on the same rollout ----
        for _ in range(ppo_epochs):
            logits, values, _ = policy(inputs)
            logp_all = F.log_softmax(logits, dim=-1)
            ratio_all = torch.exp(logp_all - old_logp_all)             # pi_theta(a) / pi_old(a) for every action

            if MODE == "exact":
                surr = torch.min(ratio_all * adv_all, ratio_all.clamp(1 - clip_eps, 1 + clip_eps) * adv_all)
                policy_loss = -(old_p * surr).sum(-1).mean()           # exact expectation over actions
                value_loss = torch.zeros((), device=device)
                loss = policy_loss
                clip_frac = (old_p * ((ratio_all - 1).abs() > clip_eps).float()).sum(-1).mean().item()
            else:
                ratio = ratio_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                surr = torch.min(ratio * advantages, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages)
                policy_loss = -surr.mean()
                value_loss = F.mse_loss(values, rewards)
                loss = policy_loss + value_coef * value_loss
                clip_frac = ((ratio - 1).abs() > clip_eps).float().mean().item()

            # how far has the policy moved from the rollout policy? KL(pi_old || pi_theta), exact, per token
            approx_kl = (old_p * (old_logp_all - logp_all)).sum(-1).mean().item()
            if approx_kl > 1.5 * target_kl:
                stats["early_stops"].append(1.0)
                break                                                  # trust region exceeded: new rollout

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(value_params, max_norm=1.0)  # no-op in exact mode (no critic grads)
            optimizer.step()
            scheduler.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["clip_frac"].append(clip_frac)
            stats["approx_kl"].append(approx_kl)
        stats["reward"].append(batch_reward)

    avg = {k: sum(v) / len(v) for k, v in stats.items() if k != "early_stops"}
    avg["early_stops"] = sum(stats["early_stops"])
    avg["lr"] = optimizer.param_groups[0]["lr"]
    m = evaluate(policy)
    policy.eval()
    log_rows.append({"epoch": epoch + 1, **m, "optimal_ce": OPTIMAL_CE, **avg})
    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Val CE {m['val_ce']:.4f} (gap {m['val_ce'] - OPTIMAL_CE:+.4f}) | "
          f"KL(pi||p*) {m['kl_pi_to_oracle']:.4f} | KL(p*||pi) {m['kl_oracle_to_pi']:.4f} | "
          f"reward {avg['reward']:+.4f} | clip {avg.get('clip_frac', float('nan')):.2f} | "
          f"step-KL {avg.get('approx_kl', float('nan')):.4f} | early-stops {int(avg['early_stops'])} | lr {avg['lr']:.1e}")

with open(f"ppo_oracle_{MODE}_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted({k for r in log_rows for k in r}))
    writer.writeheader()
    writer.writerows(log_rows)
print(f"\nSaved per-epoch metrics to ppo_oracle_{MODE}_metrics.csv")