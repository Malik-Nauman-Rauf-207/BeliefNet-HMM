"""Compare visible generations from pretrained, SFT, and PPO binary-addition checkpoints.

Run from rlhf_ppo/:
    python3 compare_binary_addition_inference.py --bits 2 4 8 16 --random-cases 10 --hard-cases 10

The default is to evaluate all four scales. Pass ``--bits 16`` only when you want a 16-bit-only run.

The script prints generations and writes a detailed CSV and text report under
inference_comparison/. It evaluates identical prompts with every selected checkpoint.
"""

import argparse
import csv
import os
import random
import sys
import tempfile

import torch
import torch.nn.functional as F

_mpl_config_dir = os.path.join(tempfile.gettempdir(), "binary-addition-mplconfig")
_cache_dir = os.path.join(tempfile.gettempdir(), "binary-addition-cache")
os.makedirs(_mpl_config_dir, exist_ok=True)
os.makedirs(_cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_config_dir)
os.environ.setdefault("XDG_CACHE_HOME", _cache_dir)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NANOGPT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "NanoGPT"))
if NANOGPT_DIR not in sys.path:
    sys.path.insert(0, NANOGPT_DIR)
from model import GPT, GPTConfig


TOKENS = ["0", "1", "+", "=", "<EOS>", "<PAD>"]
STOI = {token: i for i, token in enumerate(TOKENS)}
ITOS = {i: token for i, token in enumerate(TOKENS)}
EOS_ID = STOI["<EOS>"]
SCALES = [2, 4, 8, 16]
HARD_MIN_RUN = {8: 6, 16: 10}


MODEL_PATHS = {
    "pretrained": os.path.join(NANOGPT_DIR, "out_binary_addition_multiscale_embd128_mps", "ckpt.pt"),
    "sft_nonconservative": os.path.join(CURRENT_DIR, "out_binary_addition_sft_ppo_embd128_mps", "ckpt_sft.pt"),
    "ppo_nonconservative_final": os.path.join(CURRENT_DIR, "out_binary_addition_sft_ppo_embd128_mps", "ckpt_sft_ppo.pt"),
    "ppo_nonconservative_best": os.path.join(CURRENT_DIR, "out_binary_addition_sft_ppo_embd128_mps", "ckpt_sft_ppo_best.pt"),
    "sft_klguard": os.path.join(CURRENT_DIR, "out_binary_addition_sft_ppo_embd128_mps_klguard", "ckpt_sft.pt"),
    "ppo_klguard_selected": os.path.join(CURRENT_DIR, "out_binary_addition_sft_ppo_embd128_mps_klguard", "ckpt_sft_ppo.pt"),
}

MODEL_LABELS = {
    "pretrained": "Pretrained 128d",
    "sft_nonconservative": "SFT, first 128d run",
    "ppo_nonconservative_final": "PPO, first run final",
    "ppo_nonconservative_best": "PPO, first run best saved",
    "sft_klguard": "SFT, KL-guard run",
    "ppo_klguard_selected": "PPO, KL-guard selected",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", nargs="+", type=int, default=SCALES, choices=SCALES,
                        help="Operand widths to test; defaults to all scales.")
    parser.add_argument("--random-cases", type=int, default=10, help="Random cases per scale.")
    parser.add_argument("--hard-cases", type=int, default=10,
                        help="Long-carry cases per 8/16-bit scale (0 disables them).")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_PATHS), default=list(MODEL_PATHS),
                        help="Checkpoint variants to compare.")
    parser.add_argument("--mode", choices=["both", "greedy", "sample"], default="both")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus cutoff used only for sampled inference.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; positive values only.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=os.path.join(CURRENT_DIR, "inference_comparison"))
    return parser.parse_args()


def choose_device(requested):
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but PyTorch reports that MPS is unavailable")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but PyTorch reports that CUDA is unavailable")
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state = checkpoint["model"]
    for key in list(state):
        if key.startswith("_orig_mod."):
            state[key[len("_orig_mod."):]] = state.pop(key)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def carry_run(a, b, bits):
    carry = run = best = 0
    for j in range(bits):
        total = ((a >> j) & 1) + ((b >> j) & 1) + carry
        carry = int(total >= 2)
        run = run + 1 if carry else 0
        best = max(best, run)
    return best


def sample_operands(bits, rng, hard=False):
    limit = (1 << bits) - 1
    min_run = HARD_MIN_RUN.get(bits, bits + 1)
    while True:
        a, b = rng.randint(0, limit), rng.randint(0, limit)
        run = carry_run(a, b, bits)
        if not hard or run >= min_run:
            return a, b, run


def make_prompt(a, b, bits):
    a_text, b_text = format(a, f"0{bits}b"), format(b, f"0{bits}b")
    prompt = [STOI[c] for c in a_text] + [STOI["+"]] + [STOI[c] for c in b_text] + [STOI["="]]
    expected = format(a + b, f"0{bits + 1}b")[::-1]
    return prompt, expected, f"{a_text}+{b_text}="


@torch.no_grad()
def generate(model, prompt, max_new_tokens, device, mode, top_p, temperature):
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    emitted = []
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :]
        if mode == "greedy":
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            scaled = logits / temperature
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(scaled, float("-inf"))
            filtered.scatter_(1, sorted_indices, sorted_logits)
            next_token = torch.multinomial(F.softmax(filtered, dim=-1), 1)
        token_id = int(next_token.item())
        emitted.append(token_id)
        idx = torch.cat((idx, next_token), dim=1)
        if token_id == EOS_ID:
            break
    return emitted


def decode(ids):
    return "".join(ITOS.get(int(token), f"<{token}>") for token in ids)


def score(ids, expected):
    eos = EOS_ID in ids
    first_eos = ids.index(EOS_ID) if eos else len(ids)
    predicted = decode(ids[:first_eos])
    matches = sum(i < len(predicted) and predicted[i] == bit for i, bit in enumerate(expected))
    return {
        "predicted_bits": predicted,
        "emitted_eos": eos,
        "exact": predicted == expected and eos,
        "bit_accuracy_pct": 100.0 * matches / len(expected),
    }


def make_cases(bits_list, random_count, hard_count, seed):
    rng = random.Random(seed)
    cases = []
    for bits in bits_list:
        for i in range(random_count):
            a, b, run = sample_operands(bits, rng)
            cases.append({"scale": bits, "case": f"random_{i + 1}", "a": a, "b": b, "carry_run": run})
        if bits in HARD_MIN_RUN:
            for i in range(hard_count):
                a, b, run = sample_operands(bits, rng, hard=True)
                cases.append({"scale": bits, "case": f"long_carry_{i + 1}", "a": a, "b": b, "carry_run": run})
    return cases


def write_csv(path, rows):
    fields = ["scale", "case", "a", "b", "carry_run", "model", "mode", "prompt", "expected_bits",
              "generated", "predicted_bits", "emitted_eos", "exact", "bit_accuracy_pct"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows):
    grouped = {}
    for row in rows:
        scenario = "long_carry" if row["case"].startswith("long_carry") else "random"
        key = (row["model"], row["scale"], row["mode"], scenario)
        item = grouped.setdefault(key, {"n": 0, "exact": 0, "bit_acc": 0.0, "eos": 0})
        item["n"] += 1
        item["exact"] += int(row["exact"])
        item["bit_acc"] += float(row["bit_accuracy_pct"])
        item["eos"] += int(row["emitted_eos"])
    return [
        {"model": model, "scale": scale, "mode": mode, "scenario": scenario, "n": v["n"],
         "exact_accuracy_pct": 100.0 * v["exact"] / v["n"],
         "mean_bit_accuracy_pct": v["bit_acc"] / v["n"],
         "eos_rate_pct": 100.0 * v["eos"] / v["n"]}
        for (model, scale, mode, scenario), v in sorted(grouped.items())
    ]


def short_label(label):
    return {
        "Pretrained 128d": "Pretrained",
        "SFT, first 128d run": "SFT 1",
        "PPO, first run final": "PPO 1 final",
        "PPO, first run best saved": "PPO 1 best",
        "SFT, KL-guard run": "SFT guard",
        "PPO, KL-guard selected": "PPO guard",
    }.get(label, label)


def save_summary_csv(path, summary_rows):
    fields = ["model", "scale", "mode", "scenario", "n", "exact_accuracy_pct",
              "mean_bit_accuracy_pct", "eos_rate_pct"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_accuracy_heatmaps(summary_rows, model_labels, bits, output_dir):
    lookup = {(r["model"], r["scale"], r["mode"], r["scenario"]): r for r in summary_rows}
    model_order = [MODEL_LABELS[key] for key in model_labels]
    short_names = [short_label(label) for label in model_order]
    fig, axes = plt.subplots(2, 2, figsize=(max(12, 2.2 * len(bits)), 10), squeeze=False)
    panels = [
        (0, 0, "greedy", "exact_accuracy_pct", "Greedy exact match"),
        (0, 1, "sample", "exact_accuracy_pct", "Sampled exact match"),
        (1, 0, "greedy", "mean_bit_accuracy_pct", "Greedy bit accuracy"),
        (1, 1, "sample", "mean_bit_accuracy_pct", "Sampled bit accuracy"),
    ]
    for row_idx, col_idx, mode, metric, title in panels:
        matrix = np.full((len(model_order), len(bits)), np.nan)
        for i, model in enumerate(model_order):
            for j, scale in enumerate(bits):
                value = lookup.get((model, scale, mode, "random"))
                if value:
                    matrix[i, j] = value[metric]
        ax = axes[row_idx, col_idx]
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(bits)), [f"{b}-bit" for b in bits])
        ax.set_yticks(range(len(model_order)), short_names)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    color = "white" if matrix[i, j] < 55 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Binary addition inference accuracy by model and scale (random prompts)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(output_dir, "accuracy_heatmaps.png"), dpi=220)
    plt.close(fig)


def plot_carry_heatmap(summary_rows, model_labels, output_dir):
    lookup = {(r["model"], r["scale"], r["mode"], r["scenario"]): r for r in summary_rows}
    model_order = [MODEL_LABELS[key] for key in model_labels]
    short_names = [short_label(label) for label in model_order]
    scales = [scale for scale in (8, 16) if any(r["scale"] == scale and r["scenario"] == "long_carry" for r in summary_rows)]
    if not scales:
        return
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 2.5 * len(scales)), 6), squeeze=False)
    for col, mode in enumerate(("greedy", "sample")):
        matrix = np.full((len(model_order), len(scales)), np.nan)
        for i, model in enumerate(model_order):
            for j, scale in enumerate(scales):
                value = lookup.get((model, scale, mode, "long_carry"))
                if value:
                    matrix[i, j] = value["exact_accuracy_pct"]
        ax = axes[0, col]
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="magma", aspect="auto")
        ax.set_title(f"{mode.capitalize()} exact accuracy on long-carry prompts")
        ax.set_xticks(range(len(scales)), [f"{b}-bit" for b in scales])
        ax.set_yticks(range(len(model_order)), short_names)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    color = "white" if matrix[i, j] < 55 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Long carry-chain stress test", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(output_dir, "long_carry_heatmap.png"), dpi=220)
    plt.close(fig)


def plot_prompt_correctness(rows, model_labels, bits, output_dir):
    model_order = [MODEL_LABELS[key] for key in model_labels]
    short_names = [short_label(label) for label in model_order]
    for mode in ("greedy", "sample"):
        fig, axes = plt.subplots(len(bits), 1, figsize=(15, max(4, 3.1 * len(bits))), squeeze=False)
        for axis_idx, scale in enumerate(bits):
            cases = []
            for row in rows:
                if row["scale"] == scale and row["mode"] == mode:
                    case_id = (row["case"], row["a"], row["b"])
                    if case_id not in cases:
                        cases.append(case_id)
            matrix = np.full((len(model_order), len(cases)), np.nan)
            for i, model in enumerate(model_order):
                for j, (case_name, a, b) in enumerate(cases):
                    match = next((r for r in rows if r["model"] == model and r["mode"] == mode and
                                  r["scale"] == scale and r["case"] == case_name and r["a"] == a and r["b"] == b), None)
                    if match is not None:
                        matrix[i, j] = int(match["exact"])
            ax = axes[axis_idx, 0]
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap=matplotlib.colors.ListedColormap(["#c94747", "#3ca66b"]),
                              aspect="auto", interpolation="nearest")
            ax.set_title(f"{scale}-bit: red = incorrect, green = exact")
            ax.set_yticks(range(len(model_order)), short_names)
            tick_positions = list(range(0, len(cases), max(1, len(cases) // 10)))
            ax.set_xticks(tick_positions, [f"{cases[j][0]}\n{cases[j][1]}+{cases[j][2]}" for j in tick_positions],
                          rotation=0, fontsize=7)
            ax.set_xlabel("Shared prompts (case label and operands)")
        fig.suptitle(f"Prompt-by-prompt exact-match map — {mode}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(output_dir, f"prompt_correctness_{mode}.png"), dpi=220)
        plt.close(fig)


def plot_carry_run_curves(rows, model_labels, output_dir):
    model_order = [MODEL_LABELS[key] for key in model_labels]
    short_names = [short_label(label) for label in model_order]
    scales = [8, 16]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    colors = plt.get_cmap("tab10").colors
    for r_idx, scale in enumerate(scales):
        for c_idx, mode in enumerate(("greedy", "sample")):
            ax = axes[r_idx, c_idx]
            for m_idx, (model, short) in enumerate(zip([MODEL_LABELS[k] for k in model_labels], short_names)):
                grouped = {}
                for row in rows:
                    if row["scale"] == scale and row["mode"] == mode and row["model"] == model:
                        item = grouped.setdefault(row["carry_run"], [0, 0])
                        item[0] += int(row["exact"])
                        item[1] += 1
                xs = sorted(grouped)
                if xs:
                    ys = [100.0 * grouped[x][0] / grouped[x][1] for x in xs]
                    ax.plot(xs, ys, marker="o", lw=1.5, label=short, color=colors[m_idx % len(colors)])
            ax.set_title(f"{scale}-bit {mode} inference")
            ax.set_xlabel("Longest carry-out run")
            ax.set_ylabel("Exact accuracy (%)")
            ax.set_ylim(-5, 105)
            ax.grid(True, ls=":", alpha=0.45)
    axes[0, 1].legend(fontsize=8, ncol=2)
    fig.suptitle("Exact-match accuracy by carry-chain length (random and stress prompts)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(output_dir, "carry_run_accuracy.png"), dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    if args.random_cases < 0 or args.hard_cases < 0 or args.random_cases + args.hard_cases == 0:
        raise ValueError("Choose at least one random or long-carry case.")
    if args.temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")

    device = choose_device(args.device)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cases = make_cases(args.bits, args.random_cases, args.hard_cases, args.seed)
    modes = ["greedy", "sample"] if args.mode == "both" else [args.mode]

    print(f"Inference device: {device}")
    print(f"Sample settings: top-p={args.top_p}, temperature={args.temperature}")
    print(f"Prompts: {len(cases)} | Models: {len(args.models)} | Modes: {', '.join(modes)}")
    print(f"Output directory: {output_dir}\n")

    loaded = {}
    for key in args.models:
        loaded[key] = load_model(MODEL_PATHS[key], device)

    rows, report_lines = [], []
    for case_index, case in enumerate(cases):
        prompt_ids, expected, prompt_text = make_prompt(case["a"], case["b"], case["scale"])
        report_lines.append(
            f"\n{'=' * 96}\n{case['scale']}-bit {case['case']} | carry run={case['carry_run']}\n"
            f"Prompt:   {prompt_text}\nExpected: {expected}<EOS>\n"
        )
        print(f"{case['scale']}-bit {case['case']} (carry run {case['carry_run']}): {prompt_text} -> {expected}<EOS>")
        for model_key in args.models:
            model = loaded[model_key]
            for mode in modes:
                # Reset the random stream per prompt/mode so every checkpoint gets a comparable seed.
                torch.manual_seed(args.seed + case_index * 101 + (0 if mode == "greedy" else 1))
                ids = generate(model, prompt_ids, case["scale"] + 2, device, mode, args.top_p, args.temperature)
                result = score(ids, expected)
                generated = decode(ids)
                label = MODEL_LABELS[model_key]
                mark = "EXACT" if result["exact"] else "wrong"
                line = (f"{label:32s} | {mode:6s} | {generated:26s} | "
                        f"bit={result['bit_accuracy_pct']:5.1f}% | {mark}")
                print("  " + line)
                report_lines.append(line + "\n")
                rows.append({
                    "scale": case["scale"], "case": case["case"], "a": case["a"], "b": case["b"],
                    "carry_run": case["carry_run"], "model": label, "mode": mode,
                    "prompt": prompt_text, "expected_bits": expected, "generated": generated,
                    **result,
                })

    csv_path = os.path.join(output_dir, "inference_results.csv")
    summary_csv_path = os.path.join(output_dir, "summary_metrics.csv")
    txt_path = os.path.join(output_dir, "inference_outputs.txt")
    write_csv(csv_path, rows)
    summary_rows = summarize_rows(rows)
    save_summary_csv(summary_csv_path, summary_rows)
    plot_accuracy_heatmaps(summary_rows, args.models, args.bits, output_dir)
    plot_carry_heatmap(summary_rows, args.models, output_dir)
    plot_prompt_correctness(rows, args.models, args.bits, output_dir)
    if any(scale in (8, 16) for scale in args.bits):
        plot_carry_run_curves(rows, args.models, output_dir)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"\nSaved inference outputs and visualizations under {output_dir}:\n"
          f"  {txt_path}\n  {csv_path}\n  {summary_csv_path}\n"
          "  accuracy_heatmaps.png\n  long_carry_heatmap.png\n"
          "  prompt_correctness_greedy.png\n  prompt_correctness_sample.png\n"
          "  carry_run_accuracy.png (when 8/16-bit scales are selected)")


if __name__ == "__main__":
    main()
