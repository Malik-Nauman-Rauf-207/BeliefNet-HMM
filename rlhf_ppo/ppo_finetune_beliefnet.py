# rlhf_ppo/ppo_finetune_beliefnet.py
"""
PPO fine-tuning of the BELIEFNET itself, with the EXACT oracle (true HMM) as the reward model.

  policy  pi_theta(a | context) = the BeliefNet's predictive next-token distribution q_theta(a | z_<=t)
                                  (trainable logits for A, C, pi; initialised from the BeliefNet you trained)
  reward  r(c, a) = log p*(a | c) - log pi_theta(a | c)       p* = exact HMM filter (hmm_oracle.py)

The oracle gives the reward of EVERY token, so the expectation over actions is computed exactly
(same "exact" PPO as train_ppo.py): L = -sum_a pi_old(a) * min(rho_a A_a, clip(rho_a) A_a).

OBJECTIVE
  "reverse"  A_a = r_a - E_pi_old[r]      optimises KL(pi || p*)   (the literal RL problem with this reward)
  "forward"  A_a = p*(a)/pi_old(a) - 1    optimises KL(p* || pi)   (what validation cross-entropy measures);
             its gradient at theta_old equals the gradient of the soft-label cross-entropy, but it stays
             inside the PPO clipped trust region.
Validation CE = H(p*) + KL(p* || pi), so only the forward objective is guaranteed to push CE down.
Try "reverse" first (as asked), then "forward" and compare the curves.

Run from rlhf_ppo/:   python ppo_finetune_beliefnet.py
"""
import csv
import math
import os
from collections import defaultdict

import h5py
import torch
import torch.optim as optim

from belief_net import BeliefNetFilter, load_reward_model
from hmm_oracle import HMMOracle

# ----------------------------------------------------------------------------- settings
OBJECTIVE = "forward"           # "reverse" | "forward"
BELIEFNET_FOLDER = os.path.abspath("../BeliefNet/Model_m128/full_matrix/cpu_char_i10000_b10_dr0.1_d64")
START_PATH = os.path.join(BELIEFNET_FOLDER, "learning_filter.pt")   # or an exported .h5 / our .pt
data_file = os.path.abspath("../prepare_data/data_d64_m128_lamda0.9_example.h5")

num_epochs = 300
num_batches_per_epoch = 2
batch_size = 32                 # whole sequences: the filter must start from pi at t=0, like the oracle
ppo_epochs = 4
clip_eps = 0.2
target_kl = 0.01                # per-token KL(pi_old || pi_new) trust region; stop the passes early beyond 1.5x
lr = 2e-4                        # BeliefNet params are logits: keep small. Raise if clip_frac stays ~0, lower if it explodes
lr_min_frac = 0.1
device = "cpu"                  # tiny matmuls in a 255-step loop: CPU is as fast as a GPU here
seed = 0

torch.manual_seed(seed)
print(f"Objective: {OBJECTIVE} | start model: {START_PATH}")

# ----------------------------------------------------------------------------- models and data
start = load_reward_model(START_PATH, device=device)               # frozen copy, only used to read the matrices
policy = BeliefNetFilter.from_matrices(start.A.cpu().numpy(), start.C.cpu().numpy(), start.pi.cpu().numpy()).to(device)
for p in policy.parameters():
    p.requires_grad_(True)
oracle = HMMOracle.from_h5(data_file, device=device)
assert policy.m == oracle.m and policy.d == oracle.d, f"BeliefNet (d={policy.d}, m={policy.m}) vs data (d={oracle.d}, m={oracle.m})"

with h5py.File(data_file, "r") as f:
    z_train = torch.tensor(f["z_train"][:], dtype=torch.long, device=device)
    z_val = torch.tensor(f["z_validation"][:], dtype=torch.long, device=device)

val_inputs, val_targets = z_val[:, :-1].contiguous(), z_val[:, 1:].contiguous()
chunks = range(0, len(val_inputs), 100)
val_p_star = torch.cat([oracle.next_token_probs(val_inputs[i:i + 100]) for i in chunks])
val_log_p_star = val_p_star.clamp_min(1e-12).log()
OPTIMAL_CE = -val_log_p_star.gather(-1, val_targets.unsqueeze(-1)).squeeze(-1).mean().item()
print(f"Optimal validation CE (exact HMM): {OPTIMAL_CE:.4f}")

optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
total_steps = num_epochs * num_batches_per_epoch * ppo_epochs
cosine = lambda s: lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine)


@torch.no_grad()
def evaluate(model):
    """Deterministic metrics on the full validation split."""
    ce, kl_rev, kl_fwd, n = 0.0, 0.0, 0.0, 0
    for i in range(0, len(val_inputs), 100):
        sl = slice(i, i + 100)
        logp = model.predictive_probs(val_inputs[sl]).clamp_min(1e-12).log()
        ce += -logp.gather(-1, val_targets[sl].unsqueeze(-1)).sum().item()
        kl_rev += (logp.exp() * (logp - val_log_p_star[sl])).sum(-1).sum().item()           # KL(pi || p*)
        kl_fwd += (val_p_star[sl] * (val_log_p_star[sl] - logp)).sum(-1).sum().item()       # KL(p* || pi)
        n += val_targets[sl].numel()
    return {"val_ce": ce / n, "kl_pi_to_gold": kl_rev / n, "kl_gold_to_pi": kl_fwd / n}


m0 = evaluate(policy)
print(f"Before PPO: val CE {m0['val_ce']:.4f} (optimal {OPTIMAL_CE:.4f}, gap {m0['val_ce'] - OPTIMAL_CE:+.4f}) | "
      f"KL(pi||p*) {m0['kl_pi_to_gold']:.4f} | KL(p*||pi) {m0['kl_gold_to_pi']:.4f}")
print("\n--- Starting PPO Loop ---")

# ----------------------------------------------------------------------------- PPO
log_rows = [{"epoch": 0, **m0, "optimal_ce": OPTIMAL_CE}]
best_ce, tag = m0["val_ce"], f"beliefnet_ppo_{OBJECTIVE}"
policy.save(f"{tag}_best.pt", val_ce=best_ce, objective=OBJECTIVE, epoch=0)

for epoch in range(num_epochs):
    stats = defaultdict(list)

    for _ in range(num_batches_per_epoch):
        batch = z_train[torch.randint(len(z_train), (batch_size,), device=device)]
        inputs = batch[:, :-1].contiguous()

        # ---- 1. Rollout: freeze the old policy, score every token with the exact oracle ----
        with torch.no_grad():
            old_p = policy.predictive_probs(inputs)                       # (b, T, m)
            old_logp_all = old_p.clamp_min(1e-30).log()
            p_star = oracle.next_token_probs(inputs)
            log_p_star = p_star.clamp_min(1e-12).log()
            if OBJECTIVE == "reverse":
                r_all = log_p_star - old_logp_all
                baseline = (old_p * r_all).sum(-1, keepdim=True)          # exactly -KL(pi_old || p*)
                adv_all = r_all - baseline
                batch_reward = baseline.mean().item()
            else:
                adv_all = p_star / old_p.clamp_min(1e-8) - 1.0            # E_pi_old[adv] = 0
                batch_reward = (old_p * (log_p_star - old_logp_all)).sum(-1).mean().item()

        # ---- 2. Several clipped PPO passes on the same rollout ----
        for _ in range(ppo_epochs):
            logp_all = policy.predictive_probs(inputs).clamp_min(1e-30).log()
            ratio_all = torch.exp(logp_all - old_logp_all)
            surr = torch.min(ratio_all * adv_all, ratio_all.clamp(1 - clip_eps, 1 + clip_eps) * adv_all)
            loss = -(old_p * surr).sum(-1).mean()
            clip_frac = (old_p * ((ratio_all - 1).abs() > clip_eps).float()).sum(-1).mean().item()

            approx_kl = (old_p * (old_logp_all - logp_all)).sum(-1).mean().item()   # KL(pi_old || pi_theta)
            if approx_kl > 1.5 * target_kl:
                stats["early_stops"].append(1.0)
                break

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            stats["policy_loss"].append(loss.item())
            stats["clip_frac"].append(clip_frac)
            stats["approx_kl"].append(approx_kl)
        stats["reward"].append(batch_reward)

    avg = {k: sum(v) / len(v) for k, v in stats.items() if k != "early_stops"}
    avg["early_stops"] = sum(stats["early_stops"])
    avg["lr"] = optimizer.param_groups[0]["lr"]
    m = evaluate(policy)
    log_rows.append({"epoch": epoch + 1, **m, "optimal_ce": OPTIMAL_CE, **avg})
    if m["val_ce"] < best_ce:
        best_ce = m["val_ce"]
        policy.save(f"{tag}_best.pt", val_ce=best_ce, objective=OBJECTIVE, epoch=epoch + 1)
    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Val CE {m['val_ce']:.4f} (gap {m['val_ce'] - OPTIMAL_CE:+.4f}) | "
          f"KL(pi||p*) {m['kl_pi_to_gold']:.4f} | KL(p*||pi) {m['kl_gold_to_pi']:.4f} | "
          f"reward {avg.get('reward', float('nan')):+.4f} | clip {avg.get('clip_frac', float('nan')):.2f} | "
          f"step-KL {avg.get('approx_kl', float('nan')):.4f} | early-stops {int(avg['early_stops'])}")

policy.save(f"{tag}_final.pt", val_ce=m["val_ce"], objective=OBJECTIVE, epoch=num_epochs)
out_csv = f"{tag}_metrics.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted({k for r in log_rows for k in r}))
    writer.writeheader()
    writer.writerows(log_rows)
print(f"\nBest validation CE {best_ce:.4f} (started {m0['val_ce']:.4f}, optimal {OPTIMAL_CE:.4f}).")
print(f"Saved {tag}_best.pt, {tag}_final.pt (our .pt format, loadable with load_reward_model) and {out_csv}")
