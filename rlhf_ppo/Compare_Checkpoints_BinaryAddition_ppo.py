# rlhf_ppo/compare_checkpoints.py
"""
Before/after comparison on a LARGER fresh prompt set: pretrained base vs PPO checkpoint(s).
Run from rlhf_ppo/:   python compare_checkpoints.py
Reports exact-match accuracy (answer bits correct AND <EOS> emitted) per scale, greedy and sampled,
with a binomial standard error, plus the paired change (PPO right / base wrong, and the reverse).
"""
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

here = os.path.dirname(os.path.abspath(__file__))
nanogpt_dir = os.path.abspath(os.path.join(here, '..', 'NanoGPT'))
sys.path.insert(0, nanogpt_dir)
from model import GPT, GPTConfig

SCALES = [2, 4, 8]
N_PER_SCALE = 1000
MAX_NEW_TOKENS = 12
SEED = 4321                                  # different from the 1234 used inside training
CKPTS = {
    "base (pretrained)": os.path.join(nanogpt_dir, 'out_binary_addition_multiscale', 'ckpt.pt'),
    "PPO final": os.path.join(here, 'out_binary_addition_ppo', 'ckpt_ppo.pt'),
    "PPO best": os.path.join(here, 'out_binary_addition_ppo', 'ckpt_ppo_best.pt'),
}
TOKENS = ['0', '1', '+', '=', '<EOS>', '<PAD>']
stoi = {c: i for i, c in enumerate(TOKENS)}
TOK_EOS = stoi['<EOS>']


def load(path):
    ck = torch.load(path, map_location='cpu')
    sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()}
    m = GPT(GPTConfig(**ck['model_args']))
    m.load_state_dict(sd)
    return m.eval()


def prompts(scale, n, rng):
    rows, ans = [], []
    for _ in range(n):
        a, b = rng.randint(0, (1 << scale) - 1), rng.randint(0, (1 << scale) - 1)
        rows.append([stoi[c] for c in format(a, f'0{scale}b')] + [stoi['+']] +
                    [stoi[c] for c in format(b, f'0{scale}b')] + [stoi['=']])
        ans.append(format(a + b, f'0{scale + 1}b')[::-1])
    return torch.tensor(rows), ans


@torch.no_grad()
def run(model, prompt, greedy, seed):
    torch.manual_seed(seed)
    idx = prompt
    for _ in range(MAX_NEW_TOKENS):
        x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(torch.arange(idx.size(1))))
        for blk in model.transformer.h:
            x = blk(x)
        logits = model.lm_head(model.transformer.ln_f(x))[:, -1, :]
        nxt = logits.argmax(-1, keepdim=True) if greedy else torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, nxt], 1)
    return idx[:, prompt.size(1):]


def correct(resp, ans):
    out = []
    for r, a in zip(resp.tolist(), ans):
        s = "".join(TOKENS[t] for t in r if t != stoi['<PAD>'])
        e = s.find('<EOS>')
        out.append(e != -1 and s[:e] == a)
    return out


models = {name: load(p) for name, p in CKPTS.items() if os.path.exists(p)}
rng = random.Random(SEED)
data = {s: prompts(s, N_PER_SCALE, rng) for s in SCALES}
res = {}
for name, m in models.items():
    for mode, greedy in (("greedy", True), ("sample", False)):
        for s in SCALES:
            p, ans = data[s]
            res[(name, mode, s)] = correct(run(m, p, greedy, SEED), ans)

print(f"{N_PER_SCALE} fresh prompts per scale; exact match = right bits AND <EOS>.  +/- is one standard error.\n")
for mode in ("greedy", "sample"):
    print(f"--- {mode} ---")
    print(f"{'model':<20}" + "".join(f"{str(s) + '-bit':>16}" for s in SCALES))
    for name in models:
        cells = []
        for s in SCALES:
            c = res[(name, mode, s)]
            p = sum(c) / len(c)
            cells.append(f"{100 * p:6.1f}% +/-{100 * math.sqrt(p * (1 - p) / len(c)):3.1f}")
        print(f"{name:<20}" + "".join(f"{c:>16}" for c in cells))
    base = "base (pretrained)"
    for name in models:
        if name == base or base not in models:
            continue
        for s in SCALES:
            b, n = res[(base, mode, s)], res[(name, mode, s)]
            gained = sum(1 for x, y in zip(b, n) if y and not x)
            lost = sum(1 for x, y in zip(b, n) if x and not y)
            print(f"   {name} vs base, {s}-bit: fixed {gained}, broke {lost}")
    print()