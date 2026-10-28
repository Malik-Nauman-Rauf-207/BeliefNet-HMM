# rlhf_ppo/train_ppo_beliefnet.py
"""
PPO fine-tuning of a pretrained NanoGPT with a LEARNED reward model (BeliefNet), judged against the TRUE HMM.

Same contextual-bandit setup as train_ppo.py, but the reward uses the BeliefNet predictive q_phi instead
of the exact p*:
    r_phi(c, a) = log q_phi(a | c) - log pi_theta(a | c)          E_a~pi[r_phi] = -KL(pi || q_phi)

The exact HMM (hmm_oracle.py) is the "gold" reward model: it is used ONLY to evaluate the policy, never to
train it. Each epoch you therefore see both
    KL(pi || q_phi)   what PPO is optimising (the proxy objective), and
    KL(pi || p*)      what we actually care about (the gold objective).
If the proxy improves while the gold stalls or worsens, the policy is exploiting reward-model errors.

MODE "exact"   : exact expectation over all actions (no sampling noise, no critic)
MODE "sampled" : one sampled action per position + learned critic (classic PPO)
Reward model = the BeliefNet trained by the repo's own BeliefNet/training.py (see REWARD_MODEL_PATH).
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
from belief_net import load_reward_model
from dataset import HDF5HMMDataset
from hmm_oracle import HMMOracle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'NanoGPT')))
from model import GPTConfig

# ----------------------------------------------------------------------------- settings
MODE = "exact"                  # "exact" | "sampled"
# The BeliefNet you trained with the repo's training.py (config_cpu_d64_m128.json). Either file works:
#   learning_filter.pt  -> read directly with the `ss` library (no export step; needs the `ss` env you trained in)
#   learning_hmm_...h5  -> the file written by `training.py ... --mode get_matricies`
# BELIEFNET_FOLDER = os.path.abspath("../BeliefNet/Model_m128/full_matrix/cpu_char_i2000_b10_dr0.1_d64")
BELIEFNET_FOLDER = os.path.abspath("../BeliefNet/Model_m128/full_matrix/cpu_char_i10000_b10_dr0.1_d64")
REWARD_MODEL_PATH = os.path.join(BELIEFNET_FOLDER, "learning_filter.pt")

num_epochs = 50
num_batches_per_epoch = 10
batch_size = 32
ppo_epochs = 4
clip_eps = 0.2
target_kl = 0.01                # stop the passes early if the policy moves further than this (per token)
lr_policy = 5e-5
lr_min_frac = 0.1
weight_decay = 0.0
# sampled mode only
lr_value = 1e-3
value_coef = 0.5
normalize_advantages = False

# ----------------------------------------------------------------------------- setup
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device} | mode: {MODE}")

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

assert os.path.exists(REWARD_MODEL_PATH), f"BeliefNet file not found: {REWARD_MODEL_PATH}"
rm = load_reward_model(REWARD_MODEL_PATH, device=device)          # PROXY reward model (BeliefNet), frozen
gold = HMMOracle.from_h5(data_file, device=device)                # GOLD reward model: evaluation only
assert gptconf.vocab_size == rm.m == gold.m, (
    f"vocab mismatch: policy {gptconf.vocab_size}, BeliefNet {rm.m}, data file {gold.m}")

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

# ----------------------------------------------------------------------------- fixed validation set + reward-model quality
val_tokens = val_set.data.to(device)
val_inputs, val_targets = val_tokens[:, :-1].contiguous(), val_tokens[:, 1:].contiguous()
chunks = range(0, len(val_inputs), 100)
val_p_star = torch.cat([gold.next_token_probs(val_inputs[i:i + 100]) for i in chunks])    # gold distributions
val_q = torch.cat([rm.next_token_probs(val_inputs[i:i + 100]) for i in chunks])           # proxy distributions
val_log_p_star = val_p_star.clamp_min(1e-12).log()
val_log_q = val_q.clamp_min(1e-12).log()

nll = lambda probs: -probs.gather(-1, val_targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-30).log().mean().item()
OPTIMAL_CE, PROXY_CE = nll(val_p_star), nll(val_q)
KL_GOLD_TO_RM = (val_p_star * (val_log_p_star - val_log_q)).sum(-1).mean().item()
KL_RM_TO_GOLD = (val_q * (val_log_q - val_log_p_star)).sum(-1).mean().item()
print("\n=== Reward-model quality on the validation split ===")
print(f"optimal CE (gold)         : {OPTIMAL_CE:.4f}")
print(f"BeliefNet CE (proxy)      : {PROXY_CE:.4f}   (gap {PROXY_CE - OPTIMAL_CE:+.4f})")
print(f"KL(p* || q)               : {KL_GOLD_TO_RM:.4f}   proxy error")
print(f"KL(q || p*)               : {KL_RM_TO_GOLD:.4f}   gold KL(pi||p*) if the policy matched the proxy exactly")


@torch.no_grad()
def evaluate(model):
    """Deterministic metrics on the full validation split, against BOTH the gold and the proxy."""
    model.eval()
    ce_sum, kl_gold, kl_proxy, kl_gold_fwd, n = 0.0, 0.0, 0.0, 0.0, 0
    for i in range(0, len(val_inputs), 100):
        logits, _, _ = model(val_inputs[i:i + 100])
        logp = F.log_softmax(logits, dim=-1)
        sl = slice(i, i + 100)
        tgt = val_targets[sl]
        ce_sum += -logp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        kl_gold += (logp.exp() * (logp - val_log_p_star[sl])).sum(-1).sum().item()            # KL(pi || p*)
        kl_proxy += (logp.exp() * (logp - val_log_q[sl])).sum(-1).sum().item()               # KL(pi || q): PPO objective
        kl_gold_fwd += (val_p_star[sl] * (val_log_p_star[sl] - logp)).sum(-1).sum().item()   # KL(p* || pi)
        n += tgt.numel()
    return {"val_ce": ce_sum / n, "kl_pi_to_gold": kl_gold / n, "kl_pi_to_proxy": kl_proxy / n,
            "kl_gold_to_pi": kl_gold_fwd / n}


m0 = evaluate(policy)
print(f"\nBefore PPO: val CE {m0['val_ce']:.4f} (gap {m0['val_ce'] - OPTIMAL_CE:+.4f}) | "
      f"KL(pi||gold) {m0['kl_pi_to_gold']:.4f} | KL(pi||proxy) {m0['kl_pi_to_proxy']:.4f}")
print("\n--- Starting PPO Loop ---")

# ----------------------------------------------------------------------------- PPO
policy.eval()
log_rows = [{"epoch": 0, **m0, "optimal_ce": OPTIMAL_CE, "proxy_ce": PROXY_CE,
             "kl_gold_to_rm": KL_GOLD_TO_RM, "kl_rm_to_gold": KL_RM_TO_GOLD}]

for epoch in range(num_epochs):
    stats = defaultdict(list)

    for _ in range(num_batches_per_epoch):
        batch = train_set.get_batch(batch_size=batch_size).to(device)
        inputs = batch[:, :-1].contiguous()

        # ---- 1. Rollout: freeze the old policy and score with the BeliefNet reward model ----
        with torch.no_grad():
            old_logits, old_values, _ = policy(inputs)
            old_logp_all = F.log_softmax(old_logits, dim=-1)           # (b, t, V)
            old_p = old_logp_all.exp()
            log_q = rm.next_token_probs(inputs).clamp_min(1e-12).log()
            r_all = log_q - old_logp_all                               # proxy reward of EVERY action

            if MODE == "exact":
                baseline = (old_p * r_all).sum(-1, keepdim=True)       # exactly -KL(pi_old || q) per position
                adv_all = r_all - baseline
                batch_reward = baseline.mean().item()
            else:
                b, t, V = old_p.shape
                actions = torch.multinomial(old_p.reshape(-1, V), 1).reshape(b, t)
                old_logp = old_logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                rewards = r_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                advantages = rewards - old_values
                if normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                batch_reward = rewards.mean().item()

        # ---- 2. Several clipped PPO passes on the same rollout ----
        for _ in range(ppo_epochs):
            logits, values, _ = policy(inputs)
            logp_all = F.log_softmax(logits, dim=-1)
            ratio_all = torch.exp(logp_all - old_logp_all)

            if MODE == "exact":
                surr = torch.min(ratio_all * adv_all, ratio_all.clamp(1 - clip_eps, 1 + clip_eps) * adv_all)
                policy_loss = -(old_p * surr).sum(-1).mean()
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

            approx_kl = (old_p * (old_logp_all - logp_all)).sum(-1).mean().item()
            if approx_kl > 1.5 * target_kl:
                stats["early_stops"].append(1.0)
                break

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(value_params, max_norm=1.0)
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
    log_rows.append({"epoch": epoch + 1, **m, "optimal_ce": OPTIMAL_CE, "proxy_ce": PROXY_CE,
                     "kl_gold_to_rm": KL_GOLD_TO_RM, "kl_rm_to_gold": KL_RM_TO_GOLD, **avg})
    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Val CE {m['val_ce']:.4f} (gap {m['val_ce'] - OPTIMAL_CE:+.4f}) | "
          f"KL(pi||gold) {m['kl_pi_to_gold']:.4f} | KL(pi||proxy) {m['kl_pi_to_proxy']:.4f} | "
          f"reward {avg['reward']:+.4f} | clip {avg.get('clip_frac', float('nan')):.2f} | "
          f"step-KL {avg.get('approx_kl', float('nan')):.4f} | early-stops {int(avg['early_stops'])}")

out_csv = f"ppo_beliefnet_{MODE}_metrics.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted({k for r in log_rows for k in r}))
    writer.writeheader()
    writer.writerows(log_rows)
print(f"\nSaved per-epoch metrics to {out_csv}")
print(f"Reference points: policy that matches the proxy exactly would reach gold KL(pi||p*) = {KL_RM_TO_GOLD:.4f} "
      f"and val CE = {PROXY_CE:.4f}.")
