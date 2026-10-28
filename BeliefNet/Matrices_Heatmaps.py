import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

model_dir = Path("Model_m128/full_matrix/cpu_char_i2000_b10_dr0.1_d64")
pt_path = model_dir / "learning_filter.pt"
h5_path = model_dir / "learning_filter.hdf5"
ground_truth_path = Path("../prepare_data/data_d64_m128_lamda0.9_example.h5")

# 1. Load the checkpoint object
ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

def find_all_tensors(root):
    """Recursively extracts all torch.Tensor and nn.Parameter objects."""
    extracted = {}
    visited = set()

    def _traverse(obj, prefix=""):
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if torch.is_tensor(obj):
            extracted[prefix] = obj
            return

        # Check torch nn.Module state
        if hasattr(obj, "named_parameters"):
            try:
                for n, p in obj.named_parameters():
                    extracted[f"{prefix}.{n}".strip(".")] = p.data
            except Exception:
                pass

        if hasattr(obj, "state_dict"):
            try:
                sd = obj.state_dict()
                if isinstance(sd, dict):
                    for k, v in sd.items():
                        if torch.is_tensor(v):
                            extracted[f"{prefix}.{k}".strip(".")] = v
            except Exception:
                pass

        # Check object attributes
        if hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                if not k.startswith("__"):
                    _traverse(v, f"{prefix}.{k}".strip("."))

        # Check dictionary keys
        if isinstance(obj, dict):
            for k, v in obj.items():
                _traverse(v, f"{prefix}.{k}".strip("."))

    _traverse(root)
    return extracted

tensors = find_all_tensors(ckpt)

print(f"Found {len(tensors)} tensors inside checkpoint:")
for name, t in tensors.items():
    print(f"  - {name:35s} | shape {tuple(t.shape)}")

# Match 64x64 (A) and 64x128 (C)
A_tensor, C_tensor = None, None
for name, t in tensors.items():
    if t.ndim == 2:
        if t.shape == (64, 64):
            A_tensor = t
        elif t.shape == (64, 128):
            C_tensor = t

if A_tensor is None or C_tensor is None:
    raise RuntimeError("Could not find (64, 64) and (64, 128) tensors among the identified parameters.")

# 2. Row-wise softmax normalization (Belief Net parameterization)
A_hat = F.softmax(A_tensor.float(), dim=-1).detach().cpu().numpy()
C_hat = F.softmax(C_tensor.float(), dim=-1).detach().cpu().numpy()

# 3. Load Ground Truth HMM Data
A_true = None
if ground_truth_path.exists():
    with h5py.File(ground_truth_path, "r") as f:
        if "A" in f:
            A_true = f["A"][:]

# 4. Generate Figure 8 Matrix Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im0 = axes[0].imshow(A_hat, cmap="Blues", aspect="auto")
axes[0].set_title(r"Learned Transition Matrix $\hat{A}$ ($64 \times 64$)", fontsize=13, fontweight="bold")
axes[0].set_xlabel(r"Next State $X_{t+1}$", fontsize=11)
axes[0].set_ylabel(r"Current State $X_t$", fontsize=11)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(C_hat, cmap="Blues", aspect="auto")
axes[1].set_title(r"Learned Emission Matrix $\hat{C}$ ($64 \times 128$)", fontsize=13, fontweight="bold")
axes[1].set_xlabel(r"Observation Alphabet $Z_t$", fontsize=11)
axes[1].set_ylabel(r"Hidden State $X_t$", fontsize=11)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
heatmap_out = model_dir / "learned_matrices_heatmaps.png"
plt.savefig(heatmap_out, dpi=300)
print(f"\n[+] Saved Matrix Heatmaps to: {heatmap_out}")

# 5. Generate Figure 6 Eigenvalue Spectrum
eigvals_hat = np.sort(np.abs(np.linalg.eigvals(A_hat)))[::-1]

plt.figure(figsize=(7, 4.5))
plt.plot(range(1, len(eigvals_hat) + 1), eigvals_hat, "ro-", label=r"Belief Net $|\lambda(\hat{A})|$", markersize=4)

if A_true is not None:
    eigvals_true = np.sort(np.abs(np.linalg.eigvals(A_true)))[::-1]
    plt.plot(range(1, len(eigvals_true) + 1), eigvals_true, "k--", label=r"True HMM $|\lambda(A)|$")

plt.xlabel("Eigenvalue Index (Sorted)", fontsize=11)
plt.ylabel(r"Magnitude $|\lambda|$", fontsize=11)
plt.title(r"Transition Matrix Eigenvalue Spectrum", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

spectrum_out = model_dir / "eigenvalue_spectrum.png"
plt.savefig(spectrum_out, dpi=300)
print(f"[+] Saved Eigenvalue Spectrum to: {spectrum_out}")