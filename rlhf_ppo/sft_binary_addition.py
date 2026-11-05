# rlhf_ppo/sft_binary_addition.py
"""
SFT (supervised fine-tuning) of the pretrained binary-addition NanoGPT -- the standalone first stage.

Trains on sft_data/sft_train.jsonl with cross-entropy on the RESPONSE tokens only (the prompt is context),
scale-balanced batches (2/4/8-bit equally often), validates on sft_data/sft_val.jsonl, and keeps the best-validation
weights. Prints the exact-match accuracy of the pretrained base and of the SFT model on the same fixed eval set the
PPO scripts use, so the SFT gain is directly comparable.

Output (out_binary_addition_sft/): ckpt_sft.pt, sft_metrics.csv, sft_training_curves.png
(train CE, validation CE per scale, greedy accuracy vs the pretrained baseline, KL to the pretrained model,
entropy, token accuracy).
ckpt_sft.pt has the same layout as the pretraining checkpoint, so PPO can start from it: in
train_ppo_binary_addition.py set  base_ckpt_path = os.path.join(current_dir, 'out_binary_addition_sft', 'ckpt_sft.pt')
(the SFT model is then also the KL reference).

Run from rlhf_ppo/ (needs sft_data/ from make_sft_dataset.py):   python sft_binary_addition.py
"""
# >>> COMMON-BEGIN
import copy
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- connect to NanoGPT
current_dir = os.path.dirname(os.path.abspath(__file__))
nanogpt_dir = os.path.abspath(os.path.join(current_dir, '..', 'NanoGPT'))
if nanogpt_dir not in sys.path:
    sys.path.insert(0, nanogpt_dir)
from model import GPT, GPTConfig

# ----------------------------------------------------------------------------- common settings
device = 'cpu'
base_ckpt_path = os.path.join(nanogpt_dir, 'out_binary_addition_multiscale', 'ckpt.pt')
SFT_DATA_DIR = os.path.join(current_dir, 'sft_data')
SCALES = [2, 4, 8]              # operand bit-widths used for training and evaluation
MAX_NEW_TOKENS = 12             # must cover the longest answer: (max scale + 1) bits + <EOS>
EVAL_PROMPTS_PER_SCALE = 200
SEED = 0

random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------------- token alphabet
TOKENS = ['0', '1', '+', '=', '<EOS>', '<PAD>']
stoi = {ch: i for i, ch in enumerate(TOKENS)}
itos = {i: ch for i, ch in enumerate(TOKENS)}
TOK_ADD, TOK_EQ, TOK_EOS, TOK_PAD = stoi['+'], stoi['='], stoi['<EOS>'], stoi['<PAD>']


def forward_with_hidden(gpt, idx):
    """Full-sequence logits (B, T, V) and final hidden states (B, T, n_embd) of a NanoGPT."""
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
    x = gpt.transformer.drop(gpt.transformer.wte(idx) + gpt.transformer.wpe(pos))
    for block in gpt.transformer.h:
        x = block(x)
    h = gpt.transformer.ln_f(x)
    return gpt.lm_head(h), h


# ----------------------------------------------------------------------------- task
def format_prompt(a, b, bits):
    a_str, b_str = format(a, f'0{bits}b'), format(b, f'0{bits}b')
    ans_lsb = format(a + b, f'0{bits + 1}b')[::-1]
    return [stoi[c] for c in a_str] + [TOK_ADD] + [stoi[c] for c in b_str] + [TOK_EQ], ans_lsb


def task_verifier(prompt_toks, gen_toks):
    """Returns (score, is_correct). +2 exact + <EOS>, +0.5 right bits without <EOS>, -1 otherwise."""
    prompt_str = "".join(itos[t] for t in prompt_toks if t != TOK_PAD)
    try:
        a_bin, rest = prompt_str.split('+')
        b_bin = rest.replace('=', '')
        expected_lsb = format(int(a_bin, 2) + int(b_bin, 2), f'0{len(a_bin) + 1}b')[::-1]
    except Exception:
        return -1.0, False
    resp = "".join(itos[t] for t in gen_toks if t != TOK_PAD)
    eos_idx = resp.find('<EOS>')
    pred_bits = resp[:eos_idx] if eos_idx != -1 else resp
    if pred_bits == expected_lsb:
        return (2.0, True) if eos_idx != -1 else (0.5, True)
    return -1.0, False


def make_prompts(scale, n, rng):
    """n prompts of identical length (same scale) -> no padding needed."""
    toks = []
    for _ in range(n):
        a, b = rng.randint(0, (1 << scale) - 1), rng.randint(0, (1 << scale) - 1)
        toks.append(format_prompt(a, b, scale)[0])
    return torch.tensor(toks, dtype=torch.long, device=device)


@torch.no_grad()
def generate(model, prompt, max_new_tokens, greedy=False):
    """Sample from the model's own distribution (temperature 1) or take the argmax."""
    model.eval()
    idx = prompt
    for _ in range(max_new_tokens):
        logits = forward_with_hidden(model, idx)[0][:, -1, :]
        nxt = logits.argmax(-1, keepdim=True) if greedy else torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def response_mask(resp):
    """(B, N) bool: True for response tokens up to AND INCLUDING the first <EOS> (all N if there is none)."""
    is_eos = (resp == TOK_EOS).long()
    return (torch.cumsum(is_eos, dim=1) - is_eos) == 0


# ----------------------------------------------------------------------------- load the pretrained model
def load_base():
    if not os.path.exists(base_ckpt_path):
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_path}. Run the multi-scale pretraining first.")
    print(f"[*] Loading base checkpoint: {base_ckpt_path}")
    ckpt = torch.load(base_ckpt_path, map_location=device)
    conf = GPTConfig(**ckpt['model_args'])
    assert conf.vocab_size >= len(TOKENS), f"model vocab {conf.vocab_size} < token alphabet {len(TOKENS)}"
    assert MAX_NEW_TOKENS >= max(SCALES) + 2, "MAX_NEW_TOKENS must fit (max scale + 1) answer bits plus <EOS>"
    assert 2 * max(SCALES) + 2 + MAX_NEW_TOKENS <= conf.block_size, "prompt + response exceeds the model's block_size"
    sd = ckpt['model']
    for k in list(sd.keys()):
        if k.startswith('_orig_mod.'):
            sd[k[len('_orig_mod.'):]] = sd.pop(k)
    model = GPT(conf).to(device)
    model.load_state_dict(sd)
    return model.eval(), ckpt, conf


def save_ckpt(path, model, base_ckpt, **extra):
    """Same layout as the pretraining checkpoint (model + model_args + any extra keys), so other scripts can load it."""
    out = {k: v for k, v in base_ckpt.items() if k not in ("model", "optimizer")}
    out.update(model=model.state_dict(), **extra)
    torch.save(out, path)


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model):
    """Exact-match accuracy per scale on a FIXED prompt set (same prompts every call): greedy and sampled."""
    out = {}
    rng = random.Random(1234)
    for scale in SCALES:
        prompt = make_prompts(scale, EVAL_PROMPTS_PER_SCALE, rng)
        P = prompt.size(1)
        for name, greedy in (("greedy", True), ("sample", False)):
            seq = generate(model, prompt, MAX_NEW_TOKENS, greedy=greedy)
            resp = seq[:, P:]
            rmask = response_mask(resp)
            ok = 0
            for b in range(len(prompt)):
                score, _ = task_verifier(prompt[b].tolist(), resp[b][rmask[b]].tolist())
                ok += score == 2.0
            out[f"eval_{name}_acc_s{scale}"] = 100.0 * ok / len(prompt)
    out["eval_greedy_acc_mean"] = sum(out[f"eval_greedy_acc_s{s}"] for s in SCALES) / len(SCALES)
    return out


def fmt_eval(e, mode="greedy"):
    return " | ".join(f"{s}-bit {e[f'eval_{mode}_acc_s{s}']:.1f}%" for s in SCALES)
# <<< COMMON-END


# >>> SFT-BEGIN
# ----------------------------------------------------------------------------- SFT settings
SFT_STEPS = 600
SFT_BATCH_PER_SCALE = 16        # every step uses this many examples of EACH scale (scale-balanced)
SFT_LR = 5e-5
SFT_LR_MIN_FRAC = 0.1           # cosine decay to this fraction of SFT_LR
SFT_WARMUP_STEPS = 30
SFT_WEIGHT_DECAY = 0.01         # applied to matrices only (not biases / layer norms)
SFT_GRAD_CLIP = 1.0
SFT_EVAL_EVERY = 50


def load_sft_split(path):
    """{scale: (token tensor (n, L), prompt_len)}. All examples of a scale have identical lengths -> no padding."""
    by_scale, plen = defaultdict(list), {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_scale[r["scale"]].append(r["prompt"] + r["response"])
            plen[r["scale"]] = len(r["prompt"])
    missing = [s for s in SCALES if s not in by_scale]
    assert not missing, f"{path} has no examples for scales {missing}"
    return {s: (torch.tensor(by_scale[s], dtype=torch.long, device=device), plen[s]) for s in SCALES}


def response_ce(model, seq, P):
    """Cross-entropy and teacher-forced token accuracy on the RESPONSE tokens of seq (B, L); prompt = first P tokens."""
    logits, _ = forward_with_hidden(model, seq[:, :-1])        # logits[:, j] predicts seq[:, j + 1]
    lg, tgt = logits[:, P - 1:, :], seq[:, P:]                 # the response targets seq[:, P:]
    ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1))
    acc = (lg.argmax(-1) == tgt).float().mean()
    return ce, acc


@torch.no_grad()
def sft_validate(model, val):
    model.eval()
    out = {}
    for s in SCALES:
        seq, P = val[s]
        ce, acc = response_ce(model, seq, P)
        out[f"val_ce_s{s}"], out[f"val_tokacc_s{s}"] = ce.item(), 100 * acc.item()
    out["val_ce_mean"] = sum(out[f"val_ce_s{s}"] for s in SCALES) / len(SCALES)
    return out


@torch.no_grad()
def sft_drift(model, ref, val):
    """KL(pi || pi_pretrained) and entropy per response token on the validation prompts (teacher-forced)."""
    model.eval()
    kl, ent = [], []
    for s in SCALES:
        seq, P = val[s]
        lp = F.log_softmax(forward_with_hidden(model, seq[:, :-1])[0][:, P - 1:], dim=-1)
        lr = F.log_softmax(forward_with_hidden(ref, seq[:, :-1])[0][:, P - 1:], dim=-1)
        kl.append((lp.exp() * (lp - lr)).sum(-1).mean().item())
        ent.append((-(lp.exp() * lp).sum(-1)).mean().item())
    return {"kl_base": sum(kl) / len(kl), "entropy": sum(ent) / len(ent)}


def run_sft(model):
    """SFT on the response tokens. Returns (log rows, best step); `model` ends with the best-validation weights."""
    train = load_sft_split(os.path.join(SFT_DATA_DIR, 'sft_train.jsonl'))
    val = load_sft_split(os.path.join(SFT_DATA_DIR, 'sft_val.jsonl'))
    print(f"[SFT] train examples per scale: " + ", ".join(f"{s}-bit {len(train[s][0])}" for s in SCALES) +
          " | val: " + ", ".join(f"{s}-bit {len(val[s][0])}" for s in SCALES))
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": SFT_WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}], lr=SFT_LR, betas=(0.9, 0.95))

    def lr_at(step):
        if step <= SFT_WARMUP_STEPS:
            return SFT_LR * step / SFT_WARMUP_STEPS
        frac = (step - SFT_WARMUP_STEPS) / max(1, SFT_STEPS - SFT_WARMUP_STEPS)
        return SFT_LR * (SFT_LR_MIN_FRAC + (1 - SFT_LR_MIN_FRAC) * 0.5 * (1 + math.cos(math.pi * frac)))

    ref = copy.deepcopy(model).eval()                           # frozen pretrained model, for the KL-drift curve
    for p in ref.parameters():
        p.requires_grad_(False)
    rows, run_loss, best_val, best_state, best_step = [], [], float("inf"), None, 0
    print(f"{'step':>5} | {'train CE':>8} | {'val CE':>8} | val CE per scale | greedy exact-match (fixed eval set) | KL(base) | note")
    for step in range(1, SFT_STEPS + 1):
        model.train()                                           # dropout on during SFT (if the model has any)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = 0.0
        for s in SCALES:                                        # scale-balanced: equal weight per scale
            seq, P = train[s]
            idx = torch.randint(len(seq), (SFT_BATCH_PER_SCALE,), device=device)
            ce, _ = response_ce(model, seq[idx], P)
            loss = loss + ce / len(SCALES)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), SFT_GRAD_CLIP)
        opt.step()
        run_loss.append(loss.item())

        if step == 1 or step % SFT_EVAL_EVERY == 0 or step == SFT_STEPS:
            v, e = sft_validate(model, val), evaluate(model)
            note = ""
            if v["val_ce_mean"] < best_val:
                best_val, best_step = v["val_ce_mean"], step
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                note = "*best*"
            rows.append({"step": step, "train_ce": sum(run_loss) / len(run_loss), "lr": lr_at(step),
                         **v, **e, **sft_drift(model, ref, val)})
            print(f"{step:>5} | {rows[-1]['train_ce']:8.4f} | {v['val_ce_mean']:8.4f} | " +
                  " ".join(f"{v[f'val_ce_s{s}']:.4f}" for s in SCALES) + " | " +
                  " ".join(f"{e[f'eval_greedy_acc_s{s}']:5.1f}%" for s in SCALES) +
                  f" | {rows[-1]['kl_base']:8.4f} | {note}")
            run_loss = []
    model.load_state_dict(best_state)
    model.eval()
    print(f"[SFT] kept the best-validation weights from step {best_step} (val CE {best_val:.4f})")
    return rows, best_step


def plot_sft(rows, path, base_eval):
    """Same panels as the PPO trajectory plot, adapted to SFT: loss, accuracy vs the pretrained baseline, drift, entropy."""
    x = [r["step"] for r in rows]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    ax[0, 0].plot(x, [r["train_ce"] for r in rows], color='#2ca02c', lw=2)
    ax[0, 0].set_title("SFT train CE (response tokens, scale-averaged)"); ax[0, 0].set_yscale("log")
    for s in SCALES:
        ax[0, 1].plot(x, [r[f"val_ce_s{s}"] for r in rows], marker='o', label=f"{s}-bit")
        line, = ax[0, 2].plot(x, [r[f"eval_greedy_acc_s{s}"] for r in rows], marker='o', label=f"{s}-bit")
        ax[0, 2].axhline(base_eval[f"eval_greedy_acc_s{s}"], color=line.get_color(), ls='--', alpha=0.6,
                         label=f"{s}-bit pretrained base")
        ax[1, 2].plot(x, [r[f"val_tokacc_s{s}"] for r in rows], marker='o', label=f"{s}-bit")
    ax[0, 1].set_title("SFT validation CE per scale"); ax[0, 1].set_yscale("log"); ax[0, 1].legend()
    ax[0, 2].set_title("Greedy exact-match on the fixed eval set (%)"); ax[0, 2].legend(fontsize=7)
    ax[1, 0].plot(x, [r["kl_base"] for r in rows], color='#d62728', lw=2)
    ax[1, 0].set_title("KL(pi || pretrained model) per response token (validation)")
    ax[1, 1].plot(x, [r["entropy"] for r in rows], color='#1f77b4', lw=2)
    ax[1, 1].set_title("Policy entropy per response token (validation)")
    ax[1, 2].set_title("Validation token accuracy, teacher-forced (%)"); ax[1, 2].legend()
    for a in ax.flat:
        a.set_xlabel("SFT step"); a.grid(True, ls=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
# <<< SFT-END

# ----------------------------------------------------------------------------- main: base -> SFT
if __name__ == "__main__":
    out_dir = os.path.join(current_dir, 'out_binary_addition_sft')
    os.makedirs(out_dir, exist_ok=True)

    model, base_ckpt, gptconf = load_base()
    base_eval = evaluate(model)
    print("\n[pretrained base] greedy exact-match: " + fmt_eval(base_eval) + "   sampled: " + fmt_eval(base_eval, "sample"))

    print("\n=== SFT (loss on response tokens only) ===")
    rows, best_step = run_sft(model)
    sft_eval = evaluate(model)
    save_ckpt(os.path.join(out_dir, 'ckpt_sft.pt'), model, base_ckpt, sft_step=best_step)
    write_csv(os.path.join(out_dir, 'sft_metrics.csv'), rows)
    plot_sft(rows, os.path.join(out_dir, 'sft_training_curves.png'), base_eval)

    for mode in ("greedy", "sample"):
        print(f"\n--- exact-match accuracy, {mode} (%) on the fixed eval set ---")
        print(f"{'':<12}" + "".join(f"{str(s) + '-bit':>10}" for s in SCALES))
        print(f"{'pretrained':<12}" + "".join(f"{base_eval[f'eval_{mode}_acc_s{s}']:10.1f}" for s in SCALES))
        print(f"{'+ SFT':<12}" + "".join(f"{sft_eval[f'eval_{mode}_acc_s{s}']:10.1f}" for s in SCALES))
        print(f"{'SFT gain':<12}" + "".join(f"{sft_eval[f'eval_{mode}_acc_s{s}'] - base_eval[f'eval_{mode}_acc_s{s}']:+10.1f}" for s in SCALES))
    print(f"\n[*] Saved {os.path.join(out_dir, 'ckpt_sft.pt')} (best-validation weights from SFT step {best_step}), "
          f"sft_metrics.csv and sft_training_curves.png")
    print("Note: 2- and 4-bit have only 16 / 256 possible problems, so those columns mostly measure memorisation; "
          "8-bit is the generalisation signal.")