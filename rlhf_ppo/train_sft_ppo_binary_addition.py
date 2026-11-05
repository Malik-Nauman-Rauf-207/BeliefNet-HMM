# rlhf_ppo/train_sft_ppo_binary_addition.py
"""
Pretrained NanoGPT  ->  SFT (supervised fine-tuning)  ->  PPO, in one run, for binary addition.

Stage 0  measure the pretrained model (baseline).
Stage 1  SFT on sft_data/sft_train.jsonl: cross-entropy on the RESPONSE tokens only (the prompt is context),
         scale-balanced batches (2/4/8-bit equally often), validation on sft_data/sft_val.jsonl, best-validation
         weights kept. Saves out_binary_addition_sft_ppo/ckpt_sft.pt.
Stage 2  PPO starting from the SFT model, with the SFT model as the frozen KL reference (standard RLHF), using the
         same verifier reward as train_ppo_binary_addition.py (+2 exact with <EOS>, +0.5 right bits no <EOS>, -1 wrong).
End      table: base -> SFT -> SFT+PPO exact-match accuracy per scale (greedy and sampled) on the fixed eval set,
         i.e. how much SFT gained and how much PPO added on top.

PPO details are the corrected ones from train_ppo_binary_addition.py: temperature-1 sampling, eval-mode actor,
response-token masking, terminal reward on <EOS>, masked GAE, per-scale batches (no padding), several clipped epochs
with a KL early stop, detached critic with its own optimizer and a short warm-up.

Run from rlhf_ppo/ (needs sft_data/ from make_sft_dataset.py):   python train_sft_ppo_binary_addition.py
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


# >>> PPO-BEGIN
# ----------------------------------------------------------------------------- PPO settings
PROMPTS_PER_SCALE = 32          # rollout batch = len(SCALES) * PROMPTS_PER_SCALE sequences
NUM_STEPS = 150                 # PPO iterations (rollout + several clipped epochs)
PPO_EPOCHS = 4
CLIP_EPS = 0.2
TARGET_KL = 0.02                # skip the remaining epochs once KL(old || new) per token exceeds 1.5x this
LR_ACTOR = 2e-5
LR_CRITIC = 1e-3
KL_BETA = 0.02                  # per-token KL penalty to the frozen reference (= the SFT model)
GAMMA = 1.0                     # episodic task
GAE_LAMBDA = 0.95
ENT_COEF = 0.0                  # entropy bonus; raise (e.g. 0.01) if entropy collapses too fast
CRITIC_WARMUP_STEPS = 5         # iterations that train only the critic
EVAL_EVERY = 10


class ValueHead(nn.Module):
    """Critic: maps NanoGPT hidden states to a scalar V(s). Receives DETACHED hidden states."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.v_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, hidden):
        return self.v_proj(hidden).squeeze(-1)


@torch.no_grad()
def collect_group(actor, ref_model, critic, scale, rng):
    """One rollout group (all prompts share a length). Returns tensors + logging stats."""
    prompt = make_prompts(scale, PROMPTS_PER_SCALE, rng)
    B, P = prompt.shape
    seq = generate(actor, prompt, MAX_NEW_TOKENS)                       # (B, P + N), sampled at temperature 1
    targets = seq[:, 1:]

    logits, h = forward_with_hidden(actor, seq)
    logp_all = F.log_softmax(logits[:, :-1], dim=-1)                    # (B, P+N-1, V): index j predicts seq[:, j+1]
    old_lp = logp_all.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    ref_logits, _ = forward_with_hidden(ref_model, seq)
    ref_lp = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    values = critic(h[:, :-1])
    entropy = -(logp_all.exp() * logp_all).sum(-1)

    resp = seq[:, P:]
    rmask = response_mask(resp)                                         # (B, N)
    am = torch.zeros_like(old_lp, dtype=torch.bool)                     # action mask over the log-prob axis
    am[:, P - 1:] = rmask
    last_idx = P - 1 + rmask.sum(1) - 1                                 # index of the last action (the <EOS> if emitted)

    scores = torch.zeros(B, device=device)
    for b in range(B):
        scores[b], _ = task_verifier(prompt[b].tolist(), resp[b][rmask[b]].tolist())

    # per-token reward = -beta * (log pi - log pi_ref) on response tokens, plus the verifier score at the end
    rewards = -KL_BETA * (old_lp - ref_lp) * am
    rewards[torch.arange(B), last_idx] += scores

    # GAE over response tokens only; the episode ends at last_idx (bootstrap value 0 afterwards)
    T = old_lp.size(1)
    adv = torch.zeros_like(rewards)
    last = torch.zeros(B, device=device)
    for t in reversed(range(T)):
        if t + 1 < T:
            nxt_m, nxt_v = am[:, t + 1].float(), values[:, t + 1]
        else:
            nxt_m, nxt_v = torch.zeros(B, device=device), torch.zeros(B, device=device)
        delta = rewards[:, t] + GAMMA * nxt_v * nxt_m - values[:, t]
        last = delta + GAMMA * GAE_LAMBDA * nxt_m * last
        adv[:, t] = last
    adv = adv * am
    returns = (adv + values) * am

    d = ref_lp - old_lp
    stats = {
        "scores": scores, "exact": (scores == 2.0).float(), "resp_len": rmask.sum(1).float(),
        "kl_ref": ((torch.exp(d) - d - 1) * am).sum() / am.sum(),       # k3 estimate of KL(pi || pi_ref), >= 0
        "entropy": (entropy * am).sum() / am.sum(),
    }
    return {"seq": seq, "old_lp": old_lp, "am": am, "adv": adv, "returns": returns, "stats": stats, "scale": scale}


def run_ppo(actor, ref_model, critic, start_eval, save_best, save_final):
    """PPO from the current `actor`. Returns the metric rows and the best greedy mean accuracy."""
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=LR_ACTOR, weight_decay=0.0)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=LR_CRITIC, weight_decay=0.0)
    rng = random.Random(SEED)
    rows = [{"step": 0, **start_eval}]
    best_acc = start_eval["eval_greedy_acc_mean"]
    save_best(0)
    print(f"\n{'step':>4} | {'reward':>7} | {'exact%':>6} | {'KL_ref':>7} | {'entropy':>7} | {'len':>4} | {'clip':>5} | "
          f"{'stepKL':>7} | {'v_loss':>7} | note")

    for step in range(1, NUM_STEPS + 1):
        warmup = step <= CRITIC_WARMUP_STEPS
        actor.eval()                                                    # PPO needs a deterministic ratio

        # ---- 1. rollout (one group per scale) ----
        groups = [collect_group(actor, ref_model, critic, s, rng) for s in SCALES]
        all_adv = torch.cat([g["adv"][g["am"]] for g in groups])
        mean, std = all_adv.mean(), all_adv.std() + 1e-8
        for g in groups:                                                # whiten over response tokens only
            g["adv"] = (g["adv"] - mean) / std * g["am"]
        n_tok = sum(g["am"].sum() for g in groups).float()

        # ---- 2. several clipped PPO epochs on this rollout ----
        ep_stats = defaultdict(list)
        early = False
        for epoch in range(PPO_EPOCHS):
            pol_loss = val_loss = ent_sum = clip_sum = kl_sum = 0.0
            for g in groups:
                logits, h = forward_with_hidden(actor, g["seq"])
                logp_all = F.log_softmax(logits[:, :-1], dim=-1)
                new_lp = logp_all.gather(2, g["seq"][:, 1:].unsqueeze(-1)).squeeze(-1)
                new_v = critic(h[:, :-1].detach())                      # critic cannot change the backbone
                am = g["am"]
                log_ratio = new_lp - g["old_lp"]
                ratio = torch.exp(log_ratio)
                surr = torch.min(ratio * g["adv"], ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * g["adv"])
                pol_loss = pol_loss - (surr * am).sum() / n_tok
                val_loss = val_loss + 0.5 * (((new_v - g["returns"]) ** 2) * am).sum() / n_tok
                ent_sum = ent_sum + (-(logp_all.exp() * logp_all).sum(-1) * am).sum() / n_tok
                clip_sum += (((ratio - 1).abs() > CLIP_EPS).float() * am).sum().item() / n_tok.item()
                kl_sum += (((ratio - 1) - log_ratio) * am).sum().item() / n_tok.item()   # k3 estimate of KL(old || new)

            if kl_sum > 1.5 * TARGET_KL and not warmup:
                early = True
                break                                                   # trust region exceeded: new rollout

            loss = val_loss if warmup else pol_loss - ENT_COEF * ent_sum + val_loss
            actor_opt.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()
            if not warmup:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step()
            ep_stats["policy_loss"].append(pol_loss.item())
            ep_stats["value_loss"].append(val_loss.item())
            ep_stats["clip_frac"].append(clip_sum)
            ep_stats["approx_kl"].append(kl_sum)

        # ---- 3. logging ----
        scores = torch.cat([g["stats"]["scores"] for g in groups])
        exact = torch.cat([g["stats"]["exact"] for g in groups])
        row = {
            "step": step, "reward": scores.mean().item(), "exact_acc": 100 * exact.mean().item(),
            "kl_ref": sum(g["stats"]["kl_ref"].item() for g in groups) / len(groups),
            "entropy": sum(g["stats"]["entropy"].item() for g in groups) / len(groups),
            "resp_len": torch.cat([g["stats"]["resp_len"] for g in groups]).mean().item(),
            "early_stop": int(early), "warmup": int(warmup),
            **{k: sum(v) / len(v) for k, v in ep_stats.items()},
            **{f"train_acc_s{g['scale']}": 100 * g["stats"]["exact"].mean().item() for g in groups},
        }
        note = "critic warm-up" if warmup else ("early stop" if early else "")
        if step % EVAL_EVERY == 0 or step == NUM_STEPS:
            row.update(evaluate(actor))
            note = (note + " ").lstrip() + "eval greedy: " + " ".join(
                f"{s}b {row[f'eval_greedy_acc_s{s}']:.0f}%" for s in SCALES)
            if row["eval_greedy_acc_mean"] > best_acc:
                best_acc = row["eval_greedy_acc_mean"]
                save_best(step)
                note += " *best*"
        rows.append(row)
        print(f"{step:>4} | {row['reward']:+7.3f} | {row['exact_acc']:6.1f} | {row['kl_ref']:7.4f} | {row['entropy']:7.3f} | "
              f"{row['resp_len']:4.1f} | {row.get('clip_frac', float('nan')):5.2f} | {row.get('approx_kl', float('nan')):7.4f} | "
              f"{row.get('value_loss', float('nan')):7.4f} | {note}")
    save_final(NUM_STEPS)
    return rows, best_acc


def plot_ppo(rows, base_eval, path):
    train = [r for r in rows if r["step"] > 0]
    evals = [r for r in rows if "eval_greedy_acc_mean" in r]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot([r["step"] for r in train], [r["reward"] for r in train], color='#2ca02c', lw=2)
    ax[0, 0].axhline(2.0, color='gray', ls=':', label='max verifier score (+2)')
    ax[0, 0].set_title("Mean verifier score (training rollouts)"); ax[0, 0].legend()
    for i, s in enumerate(SCALES):
        line, = ax[0, 1].plot([r["step"] for r in evals], [r[f"eval_greedy_acc_s{s}"] for r in evals],
                              marker='o', label=f"{s}-bit (step 0 = after SFT)")
        ax[0, 1].axhline(base_eval[f"eval_greedy_acc_s{s}"], color=line.get_color(), ls='--', alpha=0.6,
                         label=f"{s}-bit pretrained base")
    ax[0, 1].set_title("Greedy exact-match on the fixed eval set (%)"); ax[0, 1].legend(fontsize=7)
    ax[1, 0].plot([r["step"] for r in train], [r["kl_ref"] for r in train], color='#d62728', lw=2)
    ax[1, 0].set_title("KL(pi || SFT reference) per response token")
    ax[1, 1].plot([r["step"] for r in train], [r["entropy"] for r in train], color='#1f77b4', lw=2)
    ax[1, 1].set_title("Policy entropy per response token")
    for a in ax.flat:
        a.set_xlabel("PPO step"); a.grid(True, ls=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)
# <<< PPO-END



# ----------------------------------------------------------------------------- combined plot (SFT then PPO on one axis)
def plot_pipeline(base_eval, sft_rows, ppo_rows, path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for j, mode in enumerate(("greedy", "sample")):
        for s in SCALES:
            key = f"eval_{mode}_acc_s{s}"
            xs = [0] + [r["step"] for r in sft_rows] + [SFT_STEPS + r["step"] for r in ppo_rows if key in r and r["step"] > 0]
            ys = [base_eval[key]] + [r[key] for r in sft_rows] + [r[key] for r in ppo_rows if key in r and r["step"] > 0]
            ax[j].plot(xs, ys, marker='o', ms=3, label=f"{s}-bit")
        ax[j].axvline(SFT_STEPS, color='gray', ls=':')
        ax[j].text(SFT_STEPS, ax[j].get_ylim()[0], " SFT | PPO", va='bottom', fontsize=9)
        ax[j].set_title(f"{mode.capitalize()} exact-match (%) across the pipeline  (x=0: pretrained base)")
        ax[j].set_xlabel("SFT steps, then PPO steps"); ax[j].grid(True, ls=':', alpha=0.6); ax[j].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main: base -> SFT -> PPO
if __name__ == "__main__":
    out_dir = os.path.join(current_dir, 'out_binary_addition_sft_ppo')
    os.makedirs(out_dir, exist_ok=True)

    actor, base_ckpt, gptconf = load_base()
    base_eval = evaluate(actor)
    print("\n[stage 0: pretrained base] greedy exact-match: " + fmt_eval(base_eval) + "   sampled: " + fmt_eval(base_eval, "sample"))

    # ---- stage 1: SFT ----
    print("\n=== Stage 1: SFT (loss on response tokens only) ===")
    sft_rows, sft_best_step = run_sft(actor)
    sft_eval = evaluate(actor)
    save_ckpt(os.path.join(out_dir, 'ckpt_sft.pt'), actor, base_ckpt, sft_step=sft_best_step)
    write_csv(os.path.join(out_dir, 'sft_metrics.csv'), sft_rows)
    plot_sft(sft_rows, os.path.join(out_dir, 'sft_training_curves.png'), base_eval)
    print("[stage 1: after SFT] greedy exact-match: " + fmt_eval(sft_eval) + "   sampled: " + fmt_eval(sft_eval, "sample"))
    if sft_eval["eval_greedy_acc_mean"] > 99.0:
        print("NOTE: the SFT model is already ~100% on the eval set, so PPO has almost nothing left to improve.")

    # ---- stage 2: PPO from the SFT model; the SFT model is the frozen KL reference ----
    print("\n=== Stage 2: PPO from the SFT model ===")
    actor.eval()
    ref_model = copy.deepcopy(actor).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    critic = ValueHead(gptconf.n_embd).to(device)
    save_best = lambda step: save_ckpt(os.path.join(out_dir, 'ckpt_sft_ppo_best.pt'), actor, base_ckpt,
                                       critic=critic.state_dict(), ppo_step=step)
    save_final = lambda step: save_ckpt(os.path.join(out_dir, 'ckpt_sft_ppo.pt'), actor, base_ckpt,
                                        critic=critic.state_dict(), ppo_step=step)
    ppo_rows, best_acc = run_ppo(actor, ref_model, critic, sft_eval, save_best, save_final)
    write_csv(os.path.join(out_dir, 'ppo_metrics.csv'), ppo_rows)
    plot_ppo(ppo_rows, base_eval, os.path.join(out_dir, 'ppo_training_trajectory.png'))
    plot_pipeline(base_eval, sft_rows, ppo_rows, os.path.join(out_dir, 'pipeline_accuracy.png'))

    # ---- summary: what SFT added and what PPO added on top ----
    final_eval = [r for r in ppo_rows if "eval_greedy_acc_mean" in r][-1]
    for mode in ("greedy", "sample"):
        print(f"\n--- exact-match accuracy, {mode} (%) on the fixed eval set ---")
        print(f"{'':<14}" + "".join(f"{str(s) + '-bit':>10}" for s in SCALES))
        for name, e in (("pretrained", base_eval), ("+ SFT", sft_eval), ("+ SFT + PPO", final_eval)):
            print(f"{name:<14}" + "".join(f"{e[f'eval_{mode}_acc_s{s}']:10.1f}" for s in SCALES))
        print(f"{'SFT gain':<14}" + "".join(f"{sft_eval[f'eval_{mode}_acc_s{s}'] - base_eval[f'eval_{mode}_acc_s{s}']:+10.1f}" for s in SCALES))
        print(f"{'PPO gain':<14}" + "".join(f"{final_eval[f'eval_{mode}_acc_s{s}'] - sft_eval[f'eval_{mode}_acc_s{s}']:+10.1f}" for s in SCALES))
    print(f"\n[*] Outputs in {out_dir}: ckpt_sft.pt, ckpt_sft_ppo.pt, ckpt_sft_ppo_best.pt (best greedy eval {best_acc:.1f}%), "
          f"sft_metrics.csv, ppo_metrics.csv, sft_training_curves.png, ppo_training_trajectory.png, pipeline_accuracy.png")
    print("Note: 2- and 4-bit have only 16 / 256 possible problems, so those columns mostly measure memorisation; "
          "8-bit is the generalisation signal.")