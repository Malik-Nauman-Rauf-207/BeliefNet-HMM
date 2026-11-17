"""Plot causal attention for every layer and head across binary-addition checkpoints.

By default, creates a 4-by-4 panel for each of the pretrained, SFT, PPO, and
GRPO checkpoints under attention_visualizations/. Each cell is one head in one
transformer layer. The model sees the same teacher-forced 16-bit long-carry
example so that the maps are comparable.

Run from this directory:
    python3 visualize_attention_checkpoints.py --device auto
"""

import argparse
import os
import sys
import tempfile

_mpl_dir = os.path.join(tempfile.gettempdir(), "binary-addition-attention-mplconfig")
os.makedirs(_mpl_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_dir)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NANOGPT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "NanoGPT"))
if NANOGPT_DIR not in sys.path:
    sys.path.insert(0, NANOGPT_DIR)
from model import GPT, GPTConfig

CHECKPOINTS = {
    "pretrained": os.path.join(NANOGPT_DIR, "out_binary_addition_multiscale_embd128_mps", "ckpt.pt"),
    "sft_first_run": os.path.join(CURRENT_DIR, "out_binary_addition_sft_ppo_embd128_mps", "ckpt_sft.pt"),
    "ppo_first_run_best": os.path.join(CURRENT_DIR, "out_binary_addition_sft_ppo_embd128_mps", "ckpt_sft_ppo_best.pt"),
    "sft_klguard_reference": os.path.join(CURRENT_DIR, "out_binary_addition_sft_ppo_embd128_mps_klguard", "ckpt_sft.pt"),
    "ppo_klguard_selected": os.path.join(CURRENT_DIR, "out_binary_addition_sft_ppo_embd128_mps_klguard", "ckpt_sft_ppo.pt"),
    "grpo_selected": os.path.join(CURRENT_DIR, "out_binary_addition_grpo_klguard", "ckpt_grpo_best.pt"),
}

LABELS = {
    "pretrained": "Pretraining",
    "sft_first_run": "SFT (first run)",
    "ppo_first_run_best": "PPO (first run, selected)",
    "sft_klguard_reference": "SFT checkpoint used as PPO KL reference",
    "ppo_klguard_selected": "PPO (KL-guard run, selected)",
    "grpo_selected": "GRPO (selected)",
}

TOKENS = ["0", "1", "+", "=", "<EOS>", "<PAD>"]
ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    "soft_sage_teal", ["#f1f4f2", "#d9e8e3", "#aecfc6", "#7eaaa6", "#4c7883"]
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", default=os.path.join(CURRENT_DIR, "attention_visualizations"))
    parser.add_argument("--models", nargs="+", choices=CHECKPOINTS, default=list(CHECKPOINTS))
    parser.add_argument("--bits", type=int, default=16, choices=(2, 4, 8, 16))
    parser.add_argument("--a", type=lambda x: int(x, 0), default=0x7FFF,
                        help="First operand (default: 0x7fff, a long-carry input for 16 bits).")
    parser.add_argument("--b", type=lambda x: int(x, 0), default=1, help="Second operand (default: 1).")
    return parser.parse_args()


def choose_device(name):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def make_example(bits, a, b):
    if not (0 <= a < (1 << bits) and 0 <= b < (1 << bits)):
        raise ValueError(f"Operands must fit in {bits} bits.")
    a_text, b_text = format(a, f"0{bits}b"), format(b, f"0{bits}b")
    answer_msb = format(a + b, f"0{bits + 1}b")
    answer_lsb = answer_msb[::-1]
    prompt = a_text + "+" + b_text + "="
    sequence = prompt + answer_lsb + "<EOS>"
    token_to_id = {token: i for i, token in enumerate(TOKENS)}
    # <EOS> is represented as one token, not as a string of five characters.
    ids = [token_to_id[ch] for ch in prompt + answer_lsb] + [token_to_id["<EOS>"]]
    labels = list(prompt + answer_lsb) + ["<EOS>"]
    return torch.tensor([ids], dtype=torch.long), labels, prompt, answer_lsb, sequence


def load_model(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state = dict(checkpoint["model"])
    for key in list(state):
        if key.startswith("_orig_mod."):
            state[key[len("_orig_mod."):]] = state.pop(key)
    model.load_state_dict(state)
    return model.to(device).eval()


def collect_attention(model, token_ids):
    """Recompute per-head softmax(QK^T/sqrt(d)) at each attention layer."""
    attention = {}
    hooks = []

    def make_hook(layer_index):
        def hook(module, inputs):
            hidden = inputs[0]
            batch, length, channels = hidden.shape
            query, key, _ = module.c_attn(hidden).split(module.n_embd, dim=-1)
            head_dim = channels // module.n_head
            query = query.view(batch, length, module.n_head, head_dim).transpose(1, 2)
            key = key.view(batch, length, module.n_head, head_dim).transpose(1, 2)
            if module.rotary is not None:
                positions = torch.arange(length, device=hidden.device)
                cos, sin = module.rotary(positions, query.dtype)
                query, key = module.apply_rotary(query, key, cos, sin)
            scores = (query @ key.transpose(-2, -1)) / np.sqrt(head_dim)
            causal_mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=hidden.device))
            weights = F.softmax(scores.masked_fill(~causal_mask, float("-inf")), dim=-1)
            attention[layer_index] = weights[0].detach().float().cpu().numpy()
        return hook

    for layer_index, block in enumerate(model.transformer.h):
        hooks.append(block.attn.register_forward_pre_hook(make_hook(layer_index)))
    try:
        with torch.no_grad():
            model(token_ids.to(next(model.parameters()).device))
    finally:
        for hook in hooks:
            hook.remove()
    return attention


def save_heatmap(attention, labels, title, output_path):
    layers = sorted(attention)
    n_layers = len(layers)
    n_heads = attention[layers[0]].shape[0]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4.2 * n_heads, 3.6 * n_layers),
                             squeeze=False, sharex=True, sharey=True)
    image = None
    for layer in layers:
        for head in range(n_heads):
            ax = axes[layer, head]
            image = ax.imshow(attention[layer][head], cmap=ATTENTION_CMAP, vmin=0.0, vmax=1.0,
                              interpolation="nearest", aspect="auto")
            ax.set_title(f"Layer {layer + 1}, Head {head + 1}", fontsize=10)
            if layer == n_layers - 1:
                ax.set_xlabel("Key token position")
            if head == 0:
                ax.set_ylabel("Query token position")

    # Sparse position labels keep long sequences legible; every position is still
    # represented in the heatmap cells. Add punctuation/EOS locations explicitly.
    length = len(labels)
    ticks = sorted(set(range(0, length, max(1, length // 10))) | {length - 1})
    tick_labels = [str(i) for i in ticks]
    for ax in axes[-1, :]:
        ax.set_xticks(ticks, tick_labels, rotation=60, ha="right", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_yticks(ticks, tick_labels, fontsize=7)
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.12, top=0.95, hspace=0.24, wspace=0.12)
    colorbar_axis = fig.add_axes([0.955, 0.16, 0.012, 0.68])
    fig.colorbar(image, cax=colorbar_axis, label="Attention probability")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = choose_device(args.device)
    token_ids, labels, prompt, answer, sequence = make_example(args.bits, args.a, args.b)
    os.makedirs(args.output_dir, exist_ok=True)
    for model_key in args.models:
        checkpoint_path = CHECKPOINTS[model_key]
        print(f"[{LABELS[model_key]}] loading {checkpoint_path}")
        model = load_model(checkpoint_path, device)
        if len(token_ids[0]) > model.config.block_size:
            raise ValueError(f"Sequence length {len(token_ids[0])} exceeds checkpoint context {model.config.block_size}.")
        maps = collect_attention(model, token_ids)
        filename = f"attention_{model_key}_{args.bits}bit.png"
        output_path = os.path.join(args.output_dir, filename)
        title = (f"{LABELS[model_key]} attention — {args.bits}-bit addition\n"
                 f"Prompt {prompt}  |  teacher-forced target {answer}<EOS>")
        save_heatmap(maps, labels, title, output_path)
        print(f"  saved {output_path} ({len(maps)} layers x {maps[0].shape[0]} heads)")
        del model

    with open(os.path.join(args.output_dir, "attention_example.txt"), "w") as f:
        f.write(f"Bits: {args.bits}\nPrompt: {prompt}\nTeacher-forced target: {answer}<EOS>\n")
        f.write(f"Full sequence: {sequence}\nSequence token count: {len(labels)}\n")
        f.write("Each heatmap is causal attention over this same complete sequence.\n")


if __name__ == "__main__":
    main()
