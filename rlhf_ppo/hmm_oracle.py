# rlhf_ppo/hmm_oracle.py
"""
Exact "oracle" reward model for the HMM data in BeliefNet-HMM.

The data file stores the true HMM:  A (d x d transitions), C (d x m emissions), pi (initial state).
Observations z_t are what NanoGPT models; hidden states x_t are never shown to it.
The Bayes-optimal next-token distribution given a context is computed exactly by the HMM
forward (filtering) recursion:

    prior   P(x_0)                   = pi
    joint   P(x_t, z_t | z_<t)       = prior(x_t) * C[x_t, z_t]
    post    P(x_t | z_<=t)           = joint / sum(joint)
    prior   P(x_{t+1} | z_<=t)       = post @ A
    p*      P(z_{t+1} | z_<=t)       = prior(x_{t+1}) @ C

Average -log p*(z_{t+1} | z_<=t) over real data is the best cross-entropy ANY model can reach.
For data_d64_m128_lamda0.9_example.h5 it is (checked against the file):

    train 1.3732 | validation 1.3809 | test 1.3700

Run this file directly to re-verify on your machine:
    python hmm_oracle.py ../prepare_data/data_d64_m128_lamda0.9_example.h5
"""
import sys

import h5py
import torch
import torch.nn as nn


class HMMOracle(nn.Module):
    def __init__(self, A, C, pi, device="cpu"):
        super().__init__()
        # float64 where supported (exact to ~1e-12); Apple MPS has no float64
        dtype = torch.float32 if str(device).startswith("mps") else torch.float64
        self.register_buffer("A", torch.as_tensor(A, dtype=dtype))
        self.register_buffer("C", torch.as_tensor(C, dtype=dtype))
        self.register_buffer("pi", torch.as_tensor(pi, dtype=dtype))
        self.d, self.m = self.C.shape
        self.to(device)

    @classmethod
    def from_h5(cls, path, device="cpu"):
        with h5py.File(path, "r") as f:
            return cls(f["A"][:], f["C"][:], f["pi"][:], device=device)

    @torch.no_grad()
    def next_token_probs(self, z):
        """
        z: (b, T) long tensor of observations z_0..z_{T-1}.
        Returns (b, T, m) float32 with entry [:, t] = P(z_{t+1} | z_0..z_t),
        i.e. exactly what a language model should output at position t.
        """
        b, T = z.shape
        tiny = torch.finfo(self.A.dtype).tiny
        prior = self.pi.expand(b, -1)                                  # P(x_0)
        out = []
        for t in range(T):
            joint = prior * self.C[:, z[:, t]].t()                     # (b, d)
            post = joint / joint.sum(-1, keepdim=True).clamp_min(tiny)  # P(x_t | z_<=t)
            prior = post @ self.A                                      # P(x_{t+1} | z_<=t)
            out.append(prior @ self.C)                                 # P(z_{t+1} | z_<=t)
        return torch.stack(out, dim=1).float()

    @torch.no_grad()
    def log_prob_of(self, z_context, tokens):
        """log p*(tokens[b, t] | z_context[b, :t+1]) for arbitrary tokens (e.g. policy samples)."""
        probs = self.next_token_probs(z_context)
        return probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).clamp_min(1e-30).log()

    @torch.no_grad()
    def optimal_ce(self, z, chunk=100):
        """Exact best-possible next-token cross-entropy (nats) on real sequences z: (n, L)."""
        total, count = 0.0, 0
        for i in range(0, len(z), chunk):
            zz = z[i : i + chunk].to(self.A.device)
            logp = self.log_prob_of(zz[:, :-1], zz[:, 1:])
            total += (-logp).sum().item()
            count += logp.numel()
        return total / count


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../prepare_data/data_d64_m128_lamda0.9_example.h5"
    oracle = HMMOracle.from_h5(path)
    print(f"HMM from {path}: d={oracle.d} hidden states, m={oracle.m} observation symbols")
    with h5py.File(path, "r") as f:
        for split in ("z_train", "z_validation", "z_test"):
            z = torch.tensor(f[split][:], dtype=torch.long)
            probs = oracle.next_token_probs(z[:20, :-1])
            assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-5), "probabilities must sum to 1"
            print(f"{split:13s} optimal CE = {oracle.optimal_ce(z):.4f}")
    print("Expected for the shipped m=128 file: train 1.3732 | validation 1.3809 | test 1.3700")
