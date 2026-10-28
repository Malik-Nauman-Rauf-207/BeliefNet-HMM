import h5py
import numpy as np
import torch
import torch.nn.functional as F

data_file = "data_d64_m128_lamda0.9_example.h5"

with h5py.File(data_file, "r") as f:
    # 1. Load Ground Truth HMM Parameters
    A = torch.tensor(f["A"][:], dtype=torch.float64)       # (64, 64)
    C = torch.tensor(f["C"][:], dtype=torch.float64)       # (64, 128)
    pi = torch.tensor(f["pi"][:], dtype=torch.float64)     # (64,)
    
    # 2. Load Validation Sequences (400 sequences of length 256)
    z_val = torch.tensor(f["z_validation"][:], dtype=torch.long)

B, T = z_val.shape
d, m = C.shape
print(f"Loaded {B} validation sequences of length {T} (d={d}, m={m}).")

# Ensure valid row-stochastic probability matrices
if not torch.allclose(A.sum(dim=-1), torch.ones(d, dtype=torch.float64), atol=1e-4):
    A = F.softmax(A, dim=-1)
if not torch.allclose(C.sum(dim=-1), torch.ones(d, dtype=torch.float64), atol=1e-4):
    C = F.softmax(C, dim=-1)
if not torch.allclose(pi.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-4):
    pi = F.softmax(pi, dim=-1)

# 3. Simulate Exact Recursive Bayesian Forward Filter
prior = pi.unsqueeze(0).expand(B, -1)  # Prior mu_{0|-1} shape: (B, d)
total_loss = 0.0

for t in range(T - 1):
    z_curr = z_val[:, t]      # Current observation Z_t
    z_next = z_val[:, t + 1]  # Target next token Z_{t+1}

    # Likelihood: e_t = C[:, Z_t]
    e_t = C.t()[z_curr]       # (B, d)

    # Correction: posterior mu_{t|t} = (e_t * prior) / sum(e_t * prior)
    unnorm_post = e_t * prior + 1e-15
    posterior = unnorm_post / unnorm_post.sum(dim=-1, keepdim=True)

    # Transition: mu_{t+1|t} = posterior @ A
    next_prior = torch.matmul(posterior, A)

    # Prediction: p_{t+1} = next_prior @ C
    p_next = torch.matmul(next_prior, C)  # (B, m)

    # Negative log-likelihood (cross-entropy) of true next token
    nll = F.nll_loss(torch.log(p_next.clamp(min=1e-15)), z_next, reduction="mean")
    total_loss += nll.item()

    prior = next_prior

optimal_loss = total_loss / (T - 1)

print("\n" + "=" * 55)
print(f"Optimal HMM Filter Loss (Empirical):  {optimal_loss:.4f}")
print(f"Table 1 Paper Benchmark:             1.368")
print(f"Uniform Random Guess Baseline ln(m): {np.log(m):.4f}")
print("=" * 55)