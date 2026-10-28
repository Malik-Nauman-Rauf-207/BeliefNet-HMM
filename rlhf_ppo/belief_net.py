# rlhf_ppo/belief_net.py
"""
BeliefNet as a learned HMM filter, in plain PyTorch (no dependency on the `ss` library).

Parameters  phi = (A, C, pi)   with every row a softmax over learnable logits:
    A[i, j] = P(x_{t+1}=j | x_t=i)      d x d   transitions
    C[i, k] = P(z_t=k   | x_t=i)        d x m   emissions
    pi[i]   = P(x_0=i)                  d       initial state

Filtering recursion (identical to the exact oracle, but with learned matrices):
    joint_t   = prior_t * C[:, z_t]
    belief_t  = joint_t / sum(joint_t)                  P_phi(x_t | z_<=t)
    prior_t+1 = belief_t @ A
    q_phi(. | z_<=t) = prior_t+1 @ C                    predictive next-token distribution

It is trained by next-token cross-entropy (train_beliefnet.py) and then frozen as a reward model:
    r_phi(c, a) = log q_phi(a | c) - log pi_theta(a | c).

`load_reward_model` also reads the matrices exported by the authors' code
(`python training.py <config> --mode get_matricies`, an .h5 file with datasets A, C, pi).
"""
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefNetFilter(nn.Module):
    def __init__(self, d: int, m: int, init_std: float = 1.0):
        super().__init__()
        self.d, self.m = d, m
        self.A_logits = nn.Parameter(init_std * torch.randn(d, d))
        self.C_logits = nn.Parameter(init_std * torch.randn(d, m))
        self.pi_logits = nn.Parameter(torch.zeros(d))

    @staticmethod
    def _stochastic(logits, floor: float = 1e-8):
        """Row softmax with a tiny probability floor (renormalised). Without it, large logits can
        underflow to exact zeros in float32 and an observed token could become impossible under every
        state, which collapses the belief to 0 and kills the gradient."""
        p = F.softmax(logits, dim=-1) + floor
        return p / p.sum(-1, keepdim=True)

    @property
    def A(self):
        return self._stochastic(self.A_logits)

    @property
    def C(self):
        return self._stochastic(self.C_logits)

    @property
    def pi(self):
        return self._stochastic(self.pi_logits)

    def predictive_probs(self, z):
        """
        Differentiable. z: (b, T) long tensor of observations.
        Returns (b, T, m): entry [:, t] = q_phi(z_{t+1} | z_0..z_t).
        """
        A, C, pi = self.A, self.C, self.pi
        b, T = z.shape
        prior = pi.expand(b, -1)
        out = []
        for t in range(T):
            joint = prior * C[:, z[:, t]].t()                                  # (b, d)
            post = joint / joint.sum(-1, keepdim=True).clamp_min(1e-30)
            prior = post @ A
            out.append(prior @ C)
        return torch.stack(out, dim=1)

    def nll(self, z):
        """Mean next-token cross-entropy (nats) on sequences z: (n, L). This is the training loss."""
        probs = self.predictive_probs(z[:, :-1])
        p = probs.gather(-1, z[:, 1:].unsqueeze(-1)).squeeze(-1)
        return -p.clamp_min(1e-30).log().mean()

    @torch.no_grad()
    def next_token_probs(self, z):
        """Reward-model interface (same as HMMOracle): (b, T) -> (b, T, m), no gradients."""
        return self.predictive_probs(z)

    @torch.no_grad()
    def log_prob_of(self, z_context, tokens):
        probs = self.predictive_probs(z_context)
        return probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).clamp_min(1e-30).log()

    @classmethod
    def from_matrices(cls, A, C, pi):
        A, C, pi = (np.asarray(x, dtype=np.float64) for x in (A, C, pi))
        model = cls(d=A.shape[0], m=C.shape[1])
        with torch.no_grad():
            for param, mat in ((model.A_logits, A), (model.C_logits, C), (model.pi_logits, pi)):
                param.copy_(torch.log(torch.as_tensor(mat, dtype=torch.float32).clamp_min(1e-30)))
        return model

    def save(self, path, **meta):
        torch.save({"A": self.A.detach().cpu(), "C": self.C.detach().cpu(), "pi": self.pi.detach().cpu(),
                    "meta": meta}, path)


def _row_stochastic(M: np.ndarray, name: str) -> np.ndarray:
    """The authors' export may store a transposed (column-stochastic) matrix; detect and fix it."""
    if np.allclose(M.sum(axis=-1), 1.0, atol=1e-4):
        return M
    if M.ndim == 2 and np.allclose(M.sum(axis=0), 1.0, atol=1e-4):
        print(f"[belief_net] {name} is column-stochastic in the file; transposing to row-stochastic.")
        return M.T
    raise ValueError(f"{name} is neither row- nor column-stochastic (row sums {M.sum(-1)[:3]}...)")


def _matrices_from_authors_checkpoint(pt_path):
    """
    Read A, C, pi straight from the authors' trained checkpoint (learning_filter.pt) with the `ss` library --
    exactly what `training.py --mode get_matricies` does, but in memory (no .h5 export step needed).
    Needs the BeliefNet/ folder (for utils.py) and the `ss` package, i.e. the environment you trained in.
    """
    import os
    import sys
    belief_dir = os.path.dirname(os.path.abspath(pt_path))
    while belief_dir != os.path.dirname(belief_dir) and not os.path.exists(os.path.join(belief_dir, "utils.py")):
        belief_dir = os.path.dirname(belief_dir)                     # walk up to the folder that holds utils.py
    sys.path.insert(0, belief_dir)
    import tiktoken
    from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
    from utils import CharTokenizer, UserConfig
    learning_filter, _ = LearningHmmFilter.load(
        pt_path,
        safe_callables={torch.nn.functional.cross_entropy, torch.optim.AdamW, UserConfig, CharTokenizer,
                        tiktoken.core.Encoding},
        strict=False,
    )
    with learning_filter.evaluation_mode():                          # dropout off
        A = learning_filter.transition.matrix.cpu().detach().numpy()
        C = learning_filter.emission.matrix.cpu().detach().numpy()
        pi = learning_filter.transition.initial_state.cpu().detach().numpy()
    return A, C, pi


def load_reward_model(path, device="cpu") -> BeliefNetFilter:
    """Load a frozen BeliefNet from our .pt file or from the authors' exported .h5/.hdf5 (A, C, pi)."""
    path = str(path)
    if path.endswith("learning_filter.pt") or os.path.basename(path).startswith("learning_filter"):
        A, C, pi = _matrices_from_authors_checkpoint(path)           # the authors' trained BeliefNet checkpoint
    elif path.endswith((".h5", ".hdf5")):
        with h5py.File(path, "r") as f:
            A, C, pi = f["A"][:], f["C"][:], f["pi"][:]
    else:
        ckpt = torch.load(path, map_location="cpu")                  # our own train_beliefnet.py output
        A, C, pi = ckpt["A"].numpy(), ckpt["C"].numpy(), ckpt["pi"].numpy()
    A, C, pi = _row_stochastic(np.asarray(A), "A"), _row_stochastic(np.asarray(C), "C"), np.asarray(pi).reshape(-1)
    if C.shape[0] != A.shape[0]:
        C = C.T
    assert A.shape[0] == A.shape[1] == C.shape[0] == len(pi), f"inconsistent shapes A{A.shape} C{C.shape} pi{pi.shape}"
    model = BeliefNetFilter.from_matrices(A, C, pi).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
