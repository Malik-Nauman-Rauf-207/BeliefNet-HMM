# rlhf_ppo/check_reward_model.py
"""
Sanity-check a BeliefNet reward model (the authors' exported .h5, or our .pt) BEFORE running PPO.

Run:  python check_reward_model.py <path to learning_hmm_d64_m128_...h5>

Expected: validation CE a little above the exact optimum (1.3809). If it is far above (e.g. ~ln(128)=4.85
or > 2), the matrices are probably mis-oriented or from an undertrained/mismatched model.
Conventions confirmed from the `ss` library (v0.0.2): transition.matrix is row-stochastic (state @ A),
emission.matrix is (d, m) row-stochastic, initial_state is the prior at t=0 -- identical to belief_net.py.
"""
import os
import sys

import h5py
import torch

from belief_net import load_reward_model
from hmm_oracle import HMMOracle

DATA_FILE = os.path.abspath("../prepare_data/data_d64_m128_lamda0.9_example.h5")
path = sys.argv[1]

rm = load_reward_model(path)
oracle = HMMOracle.from_h5(DATA_FILE)
with h5py.File(DATA_FILE, "r") as f:
    z_val = torch.tensor(f["z_validation"][:], dtype=torch.long)

ce_rm = ce_gold = kl_fwd = n = 0.0
with torch.no_grad():
    for i in range(0, len(z_val), 100):
        zz = z_val[i:i + 100]
        ctx, tgt = zz[:, :-1], zz[:, 1:]
        q, p = rm.next_token_probs(ctx), oracle.next_token_probs(ctx)
        ce_rm += -q.gather(-1, tgt.unsqueeze(-1)).clamp_min(1e-30).log().sum().item()
        ce_gold += -p.gather(-1, tgt.unsqueeze(-1)).clamp_min(1e-30).log().sum().item()
        kl_fwd += (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum().item()
        n += tgt.numel()
print(f"reward model d={rm.d}, m={rm.m}")
print(f"validation CE  reward model: {ce_rm / n:.4f} | exact optimum: {ce_gold / n:.4f} | gap {(ce_rm - ce_gold) / n:+.4f}")
print(f"KL(p* || q)  (proxy-reward error floor): {kl_fwd / n:.4f}")
