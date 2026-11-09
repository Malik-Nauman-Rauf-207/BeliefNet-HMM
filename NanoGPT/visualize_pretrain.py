"""Plot pretraining loss and evaluate/visualize multiscale binary addition checkpoints.

Run from the NanoGPT directory, for example:
    python3 visualize_pretrain.py --out-dirs out_binary_addition_multiscale out_binary_addition_multiscale_embd128_mps

Outputs per run: loss_curves.png, accuracy_by_scale.png, carry_run_accuracy.png,
attention_maps.png, evaluation_metrics.csv/.json, and rollout_examples.csv.
Also writes loss_comparison.png when multiple runs are supplied.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from model import GPT, GPTConfig


TOKENS = ["0", "1", "+", "=", "<EOS>", "<PAD>"]
STOI = {ch: i for i, ch in enumerate(TOKENS)}
ITOS = {i: ch for i, ch in enumerate(TOKENS)}
TOK_EOS = STOI["<EOS>"]
SCALES = [2, 4, 8, 16]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dirs", nargs="+", default=["out_binary_addition_multiscale_embd128_mps"],
        help="Pretraining output directories containing ckpt.pt and loss_trajectory_*.h5.",
    )
    parser.add_argument("--output-dir", default="pretraining_visualizations", help="Root directory for generated artifacts.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--samples-per-scale", type=int, default=300, help="Random evaluation problems per scale.")
    parser.add_argument("--hard-samples", type=int, default=100, help="Long-carry evaluation problems for 8/16-bit scales.")
    parser.add_argument("--max-examples", type=int, default=40, help="Maximum rollout examples saved per run.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(requested):
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def find_loss_file(run_dir):
    candidates = glob.glob(os.path.join(run_dir, "loss_trajectory_*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No loss_trajectory_*.h5 found in {run_dir}")
    return max(candidates, key=os.path.getmtime)


def load_loss_history(path):
    with h5py.File(path, "r") as f:
        required = ("training_loss", "training_iters", "validation_loss", "validation_iters")
        missing = [key for key in required if key not in f]
        if missing:
            raise ValueError(f"{path} is missing datasets: {missing}")
        return {key: np.asarray(f[key][:]) for key in required}


def moving_average(values, window=25):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def format_prompt(a, b, bits):
    a_str, b_str = format(a, f"0{bits}b"), format(b, f"0{bits}b")
    prompt = [STOI[c] for c in a_str] + [STOI["+"]] + [STOI[c] for c in b_str] + [STOI["="]]
    expected = format(a + b, f"0{bits + 1}b")[::-1]
    return prompt, expected


def carry_run(a, b, bits):
    """Longest run of carry-outs while adding from least significant bit upward."""
    carry = run = best = 0
    for j in range(bits):
        total = ((a >> j) & 1) + ((b >> j) & 1) + carry
        carry = int(total >= 2)
        run = run + 1 if carry else 0
        best = max(best, run)
    return best


def sample_problem(bits, rng, require_hard=False):
    limit = (1 << bits) - 1
    threshold = {8: 6, 16: 10}.get(bits, bits + 1)
    while True:
        a, b = rng.randint(0, limit), rng.randint(0, limit)
        run = carry_run(a, b, bits)
        if not require_hard or run >= threshold:
            return a, b, run


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    state = checkpoint["model"]
    for key in list(state):
        if key.startswith("_orig_mod."):
            state[key[len("_orig_mod."):]] = state.pop(key)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint, config


@torch.no_grad()
def generate_batch(model, prompts, max_new_tokens, device):
    idx = torch.tensor(prompts, dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, nxt), dim=1)
    return idx[:, len(prompts[0]):].cpu().tolist()


def score_response(response, expected):
    eos_pos = response.index(TOK_EOS) if TOK_EOS in response else -1
    bit_ids = response[:eos_pos] if eos_pos >= 0 else response
    predicted = "".join(ITOS.get(tok, "?") for tok in bit_ids if tok != STOI["<PAD>"])
    bit_matches = sum(i < len(predicted) and predicted[i] == bit for i, bit in enumerate(expected))
    exact = predicted == expected and eos_pos >= 0
    return predicted, exact, bit_matches, len(expected)


def evaluate_model(model, device, samples_per_scale, hard_samples, seed):
    rng = random.Random(seed)
    metrics, examples = [], []
    for bits in SCALES:
        cases = []
        for _ in range(samples_per_scale):
            a, b, run = sample_problem(bits, rng)
            cases.append(("random", a, b, run))
        if bits in (8, 16):
            for _ in range(hard_samples):
                a, b, run = sample_problem(bits, rng, require_hard=True)
                cases.append(("long_carry", a, b, run))

        prompts, expected_by_case = [], []
        for _, a, b, _ in cases:
            prompt, expected = format_prompt(a, b, bits)
            prompts.append(prompt)
            expected_by_case.append(expected)
        responses = generate_batch(model, prompts, bits + 2, device)

        agg = defaultdict(lambda: {"n": 0, "exact": 0, "bit_matches": 0, "bit_total": 0})
        carry_agg = defaultdict(lambda: {"n": 0, "exact": 0})
        for case, prompt, expected, response in zip(cases, prompts, expected_by_case, responses):
            split, a, b, run = case
            predicted, exact, bit_matches, bit_total = score_response(response, expected)
            row = agg[split]
            row["n"] += 1
            row["exact"] += int(exact)
            row["bit_matches"] += bit_matches
            row["bit_total"] += bit_total
            if split == "random":
                carry_row = carry_agg[run]
                carry_row["n"] += 1
                carry_row["exact"] += int(exact)
            examples.append({
                "split": split, "scale": bits, "a": a, "b": b, "carry_run": run,
                "prompt": "".join(ITOS[tok] for tok in prompt), "expected_lsb": expected,
                "predicted_lsb": predicted, "emitted_eos": TOK_EOS in response,
                "exact": exact,
            })
        for split, row in agg.items():
            metrics.append({
                "scale": bits, "split": split, "n": row["n"],
                "exact_accuracy_pct": 100 * row["exact"] / row["n"],
                "bit_accuracy_pct": 100 * row["bit_matches"] / max(1, row["bit_total"]),
            })
        for run, row in carry_agg.items():
            metrics.append({
                "scale": bits, "split": "carry_run", "carry_run": run, "n": row["n"],
                "exact_accuracy_pct": 100 * row["exact"] / row["n"],
                "bit_accuracy_pct": "",
            })
    return metrics, examples


def select_examples(examples, limit):
    """Select a scale-balanced set of failures and successes for the diagnostic CSV."""
    selected = []
    for target in (False, True):
        pool = [row for row in examples if row["exact"] is target]
        while pool and len(selected) < limit:
            for scale in SCALES:
                match = next((i for i, row in enumerate(pool) if row["scale"] == scale), None)
                if match is not None:
                    selected.append(pool.pop(match))
                    if len(selected) >= limit:
                        break
            else:
                break
    return selected


def plot_loss(history, run_label, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    train_x = history["training_iters"]
    train_y = history["training_loss"]
    val_x = history["validation_iters"]
    val_y = history["validation_loss"]
    if len(train_x):
        ax.plot(train_x, train_y, color="#9ecae1", alpha=0.45, lw=0.8, label="Training loss (raw)")
        ma = moving_average(train_y)
        ma_x = train_x[len(train_x) - len(ma):]
        ax.plot(ma_x, ma, color="#2171b5", lw=2, label="Training loss (25-step moving average)")
    if len(val_x):
        ax.plot(val_x, val_y, marker="o", ms=4, color="#d62728", lw=1.8, label="Validation loss")
        best = int(np.argmin(val_y))
        ax.scatter([val_x[best]], [val_y[best]], color="#2ca02c", zorder=5,
                   label=f"Best validation: {val_y[best]:.4f} at iter {val_x[best]}")
    ax.set_title(f"Pretraining loss — {run_label}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-entropy loss (nats)")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_accuracy(metrics, run_label, out_path):
    rows = [r for r in metrics if r["split"] in ("random", "long_carry")]
    scales = [s for s in SCALES if any(r["scale"] == s for r in rows)]
    x = np.arange(len(scales))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 6))
    series = [
        ("random", "exact_accuracy_pct", "Random exact", -1.5 * width, "#1f77b4"),
        ("random", "bit_accuracy_pct", "Random bitwise", -0.5 * width, "#6baed6"),
        ("long_carry", "exact_accuracy_pct", "Long-carry exact", 0.5 * width, "#d62728"),
        ("long_carry", "bit_accuracy_pct", "Long-carry bitwise", 1.5 * width, "#ff9896"),
    ]
    for split, key, label, offset, color in series:
        values = []
        for scale in scales:
            row = next((r for r in rows if r["scale"] == scale and r["split"] == split), None)
            values.append(float(row[key]) if row and row[key] != "" else np.nan)
        if np.isfinite(values).any():
            ax.bar(x + offset, values, width, label=label, color=color)
    ax.set_xticks(x, [f"{s}-bit" for s in scales])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Greedy generation accuracy — {run_label}")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_carry_accuracy(metrics, run_label, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    found = False
    for scale, color in ((8, "#1f77b4"), (16, "#d62728")):
        rows = sorted((r for r in metrics if r["scale"] == scale and r["split"] == "carry_run"),
                      key=lambda row: row["carry_run"])
        rows = [r for r in rows if r["n"] >= 3]
        if rows:
            found = True
            ax.plot([r["carry_run"] for r in rows], [r["exact_accuracy_pct"] for r in rows],
                    marker="o", lw=2, color=color, label=f"{scale}-bit")
    if found:
        ax.set_xlabel("Longest consecutive carry-out run")
        ax.set_ylabel("Exact-match accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Accuracy versus carry-chain length — {run_label}")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not enough carry-run samples to plot", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def attention_maps(model, device, out_path, scale=16, a=0x7FFF, b=1):
    prompt, expected = format_prompt(a, b, scale)
    full_seq = prompt + [STOI[c] for c in expected] + [TOK_EOS]
    x = torch.tensor([full_seq], dtype=torch.long, device=device)
    attn_by_layer = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            hidden = inputs[0]
            batch, length, channels = hidden.shape
            q, k, _ = module.c_attn(hidden).split(module.n_embd, dim=2)
            q = q.view(batch, length, module.n_head, channels // module.n_head).transpose(1, 2)
            k = k.view(batch, length, module.n_head, channels // module.n_head).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / np.sqrt(k.size(-1))
            causal = torch.tril(torch.ones(length, length, device=hidden.device, dtype=torch.bool))
            weights = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
            attn_by_layer[layer_idx] = weights[0].mean(dim=0).detach().cpu().numpy()
        return hook

    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        model(x)
    for hook in hooks:
        hook.remove()

    fig, axes = plt.subplots(1, len(attn_by_layer), figsize=(5 * len(attn_by_layer), 5), squeeze=False)
    labels = [ITOS[t] for t in full_seq]
    for i, ax in enumerate(axes[0]):
        image = ax.imshow(attn_by_layer[i], cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"Layer {i}, mean over heads")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        tick_positions = list(range(0, len(labels), 4))
        ax.set_xticks(tick_positions, [labels[j] for j in tick_positions], rotation=0)
        ax.set_yticks(tick_positions, [labels[j] for j in tick_positions])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Attention maps on {scale}-bit carry-chain example: {a} + {b}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    """Recursively convert checkpoint/config values to JSON-serializable Python values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def process_run(run_path, output_root, device, args):
    run_dir = resolve_path(run_path)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    loss_path = find_loss_file(run_dir)
    history = load_loss_history(loss_path)
    model, checkpoint, config = load_model(ckpt_path, device)
    label = os.path.basename(os.path.normpath(run_dir))
    out_dir = os.path.join(output_root, label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{label}] evaluating checkpoint from iteration {checkpoint.get('iter_num', 'unknown')} on {device}")
    metrics, examples = evaluate_model(model, device, args.samples_per_scale, args.hard_samples, args.seed)
    plot_loss(history, label, os.path.join(out_dir, "loss_curves.png"))
    plot_accuracy(metrics, label, os.path.join(out_dir, "accuracy_by_scale.png"))
    plot_carry_accuracy(metrics, label, os.path.join(out_dir, "carry_run_accuracy.png"))
    attention_maps(model, device, os.path.join(out_dir, "attention_maps.png"))
    save_csv(os.path.join(out_dir, "evaluation_metrics.csv"), metrics)
    save_csv(os.path.join(out_dir, "rollout_examples.csv"), select_examples(examples, args.max_examples))
    report = {
        "run": label,
        "checkpoint": os.path.abspath(ckpt_path),
        "loss_history": os.path.abspath(loss_path),
        "device": str(device),
        "iteration": checkpoint.get("iter_num"),
        "best_validation_loss": checkpoint.get("best_val_loss"),
        "model_args": checkpoint["model_args"],
        "metrics": metrics,
    }
    with open(os.path.join(out_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(json_safe(report), f, indent=2)
    print(f"[{label}] wrote plots and metrics to {out_dir}")
    return {"label": label, "history": history, "out_dir": out_dir}


def plot_loss_comparison(runs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for run in runs:
        history, label = run["history"], run["label"]
        train_x, train_y = history["training_iters"], history["training_loss"]
        val_x, val_y = history["validation_iters"], history["validation_loss"]
        if len(train_y):
            ma = moving_average(train_y)
            axes[0].plot(train_x[len(train_x) - len(ma):], ma, lw=2, label=label)
        if len(val_y):
            axes[1].plot(val_x, val_y, marker="o", ms=3, lw=1.5, label=label)
    axes[0].set_title("Training loss (moving average)")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cross-entropy loss (nats)")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.45)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    if args.samples_per_scale < 1 or args.hard_samples < 0 or args.max_examples < 0:
        raise ValueError("samples-per-scale must be >=1; hard-samples and max-examples must be >=0")
    device = choose_device(args.device)
    output_root = resolve_path(args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    runs = [process_run(path, output_root, device, args) for path in args.out_dirs]
    if len(runs) > 1:
        plot_loss_comparison(runs, os.path.join(output_root, "loss_comparison.png"))
    print(f"All visualization artifacts are under: {output_root}")


if __name__ == "__main__":
    main()
