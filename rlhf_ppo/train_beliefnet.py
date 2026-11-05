# rlhf_ppo/train_beliefnet.py
"""
Train the BeliefNet reward model: a learnable HMM filter fitted with next-token cross-entropy on z_train.

Same model and loss as the authors' training.py (d=64, AdamW, next-token CE) but plain PyTorch, no `ss`.
The authors' run (batch 10, constant lr 0.1, 2000 iterations) stalled at val CE ~1.7, so this uses more
iterations, a larger batch and a lower lr with cosine decay (a guess, tune it). The exact HMM (hmm_oracle.py) is used ONLY to report how far the
learned reward model is from the truth; it never influences training.

Outputs: beliefnet_m128.pt  (best-validation matrices A, C, pi)  and  beliefnet_training_log.csv
Run:     python train_beliefnet.py
"""
import csv
import math
import os
import time

import h5py
import torch

from belief_net import BeliefNetFilter
from hmm_oracle import HMMOracle

# ----------------------------------------------------------------------------- settings
DATA_FILE = os.path.abspath("../prepare_data/data_d64_m128_lamda0.9_example.h5")
OUT_PATH = "beliefnet_m128.pt"
d = 64                  # hidden-state dimension of the learned filter
iters = 10000
batch_size = 20
lr = 0.03
lr_min_frac = 0.05
eval_every = 50
seed = 0
device = "cpu"          # tiny matmuls in a 255-step loop: CPU is usually as fast as a GPU here

torch.manual_seed(seed)
with h5py.File(DATA_FILE, "r") as f:
    z_train = torch.tensor(f["z_train"][:], dtype=torch.long)
    z_val = torch.tensor(f["z_validation"][:], dtype=torch.long)
    m = int(f["C"].shape[1])
z_train, z_val = z_train.to(device), z_val.to(device)
print(f"Data: {len(z_train)} train / {len(z_val)} validation sequences of length {z_train.shape[1]}, m={m} symbols")

oracle = HMMOracle.from_h5(DATA_FILE, device=device)
OPTIMAL_CE = oracle.optimal_ce(z_val)
print(f"Optimal validation CE (exact HMM, used only for reporting): {OPTIMAL_CE:.4f}")

model = BeliefNetFilter(d=d, m=m).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
cosine = lambda s: lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(s, iters) / iters))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine)


@torch.no_grad()
def val_ce(chunk=100):
    total, count = 0.0, 0
    for i in range(0, len(z_val), chunk):
        zz = z_val[i:i + chunk]
        total += model.nll(zz).item() * zz[:, 1:].numel()
        count += zz[:, 1:].numel()
    return total / count


best_ce, best_state, rows, run_loss, t0 = float("inf"), None, [], [], time.time()
print(f"\n--- Training BeliefNet (d={d}) for {iters} iterations ---")
for it in range(1, iters + 1):
    batch = z_train[torch.randint(len(z_train), (batch_size,))]
    loss = model.nll(batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    run_loss.append(loss.item())

    if it % eval_every == 0 or it == iters:
        ce = val_ce()
        if ce < best_ce:
            best_ce, best_state = ce, {k: v.detach().clone() for k, v in model.state_dict().items()}
        rows.append({"iter": it, "train_loss": sum(run_loss) / len(run_loss), "val_ce": ce,
                     "gap_to_optimal": ce - OPTIMAL_CE, "lr": optimizer.param_groups[0]["lr"]})
        print(f"iter {it:4d} | train {rows[-1]['train_loss']:.4f} | val CE {ce:.4f} "
              f"(gap {ce - OPTIMAL_CE:+.4f}, best {best_ce:.4f}) | lr {rows[-1]['lr']:.3f} | {time.time() - t0:.0f}s")
        run_loss = []

# ----------------------------------------------------------------------------- reward-model quality report
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    kl_gold_to_rm, kl_rm_to_gold, n = 0.0, 0.0, 0
    for i in range(0, len(z_val), 100):
        zz = z_val[i:i + 100, :-1]
        p_star = oracle.next_token_probs(zz).clamp_min(1e-12)
        q = model.next_token_probs(zz).clamp_min(1e-12)
        kl_gold_to_rm += (p_star * (p_star.log() - q.log())).sum(-1).sum().item()
        kl_rm_to_gold += (q * (q.log() - p_star.log())).sum(-1).sum().item()
        n += zz.numel()
kl_gold_to_rm, kl_rm_to_gold = kl_gold_to_rm / n, kl_rm_to_gold / n

print("\n=== BeliefNet reward-model quality (validation contexts) ===")
print(f"best validation CE          : {best_ce:.4f}   (exact optimum {OPTIMAL_CE:.4f}, gap {best_ce - OPTIMAL_CE:+.4f})")
print(f"KL(p* || q_phi)  forward    : {kl_gold_to_rm:.4f}   <- how much the proxy reward model is wrong")
print(f"KL(q_phi || p*)  reverse    : {kl_rm_to_gold:.4f}   <- gold KL a policy would have if it matched q_phi exactly")

model.save(OUT_PATH, best_val_ce=best_ce, optimal_ce=OPTIMAL_CE, d=d, m=m, data_file=DATA_FILE)
with open("beliefnet_training_log.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved reward model to {OUT_PATH} and the log to beliefnet_training_log.csv")
