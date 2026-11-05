# rlhf_ppo/train_ppo_binary_addition.py
"""
PPO fine-tuning of a pretrained NanoGPT on binary addition, with a programmatic verifier as the reward.

Task: prompt  "a + b ="  (a, b are `scale`-bit binary numbers, MSB first), answer = (a+b) written LSB first
followed by <EOS>.  Reward = verifier score at the end of the answer (+2 exact with <EOS>, +0.5 right bits but no
<EOS>, -1 wrong), minus a per-token KL penalty to the frozen pretrained reference model.

What this implements (and what the previous version got wrong)
  1. Sampling and training use the SAME distribution. The old code sampled at temperature 0.7 but computed
     log-probs at temperature 1, so the importance ratio was meaningless. Now: temperature 1.0 everywhere.
  2. Deterministic ratio. The old code rolled out in eval() mode and updated in train() mode, so dropout made
     the ratio != 1 before any update. The actor stays in eval() for the whole run.
  3. Response masking. Only tokens up to and including the first <EOS> are actions. The old code applied
     reward, KL, GAE, advantage normalisation and the loss to prompt tokens and to junk generated after <EOS>.
  4. Terminal reward on the LAST RESPONSE TOKEN (the <EOS>), not on the last position of the 12-token buffer.
  5. GAE restricted to the response (terminal bootstrap = 0, no leakage from masked positions).
  6. Advantages whitened over response tokens only.
  7. No padding: each batch is built per scale, so every prompt in a group has the same length (the old code
     left-padded with <PAD>, which the model never saw in pretraining and which had no attention mask).
  8. Several clipped epochs per rollout, with a KL early stop (trust region). The old code did one gradient step
     per rollout, where clipping never matters.
  9. The critic gets DETACHED hidden states and its own optimizer/lr, so a randomly initialised value head cannot
     push gradients into the pretrained backbone. A short critic warm-up runs before the policy is updated.
 10. No weight decay on the pretrained actor.
 11. Measurement: baseline + periodic evaluation (greedy and sampled exact-match per scale on a FIXED prompt set),
     KL to reference, entropy, response length, clip fraction. Best and final checkpoints are saved.

Run from rlhf_ppo/:   python train_ppo_binary_addition.py
"""
import csv
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

# ----------------------------------------------------------------------------- settings
device = 'cpu'
base_ckpt_path = os.path.join(nanogpt_dir, 'out_binary_addition_multiscale', 'ckpt.pt')
out_dir = os.path.join(current_dir, 'out_binary_addition_ppo')
os.makedirs(out_dir, exist_ok=True)

SCALES = [2, 4, 8]              # operand bit-widths used for training and evaluation
PROMPTS_PER_SCALE = 32          # rollout batch = len(SCALES) * PROMPTS_PER_SCALE sequences
MAX_NEW_TOKENS = 12             # must cover the longest answer: (max scale + 1) bits + <EOS>
NUM_STEPS = 150                 # PPO iterations (rollout + several clipped epochs)
PPO_EPOCHS = 4
CLIP_EPS = 0.2
TARGET_KL = 0.02                # skip the remaining epochs once KL(old || new) per token exceeds 1.5x this
LR_ACTOR = 2e-5
LR_CRITIC = 1e-3
KL_BETA = 0.02                  # per-token KL penalty to the frozen reference model
GAMMA = 1.0                     # episodic task
GAE_LAMBDA = 0.95
ENT_COEF = 0.0                  # entropy bonus; raise (e.g. 0.01) if entropy collapses too fast
CRITIC_WARMUP_STEPS = 5         # iterations that train only the critic
EVAL_EVERY = 10
EVAL_PROMPTS_PER_SCALE = 200
SEED = 0

random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------------- token alphabet
TOKENS = ['0', '1', '+', '=', '<EOS>', '<PAD>']
stoi = {ch: i for i, ch in enumerate(TOKENS)}
itos = {i: ch for i, ch in enumerate(TOKENS)}
TOK_ADD, TOK_EQ, TOK_EOS, TOK_PAD = stoi['+'], stoi['='], stoi['<EOS>'], stoi['<PAD>']


# ----------------------------------------------------------------------------- critic
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


# ----------------------------------------------------------------------------- load pretrained model
if not os.path.exists(base_ckpt_path):
    raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_path}. Run the multi-scale pretraining first.")
print(f"[*] Loading base checkpoint: {base_ckpt_path}")
checkpoint = torch.load(base_ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
assert gptconf.vocab_size >= len(TOKENS), f"model vocab {gptconf.vocab_size} < token alphabet {len(TOKENS)}"
assert MAX_NEW_TOKENS >= max(SCALES) + 2, "MAX_NEW_TOKENS must fit (max scale + 1) answer bits plus <EOS>"
assert 2 * max(SCALES) + 2 + MAX_NEW_TOKENS <= gptconf.block_size, "prompt + response exceeds the model's block_size"

state_dict = checkpoint['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

actor = GPT(gptconf).to(device)
actor.load_state_dict(state_dict)
actor.eval()                                    # stays in eval() ALL run: PPO needs a deterministic ratio
ref_model = GPT(gptconf).to(device)
ref_model.load_state_dict(state_dict)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)
critic = ValueHead(gptconf.n_embd).to(device)

actor_opt = torch.optim.AdamW(actor.parameters(), lr=LR_ACTOR, weight_decay=0.0)
critic_opt = torch.optim.AdamW(critic.parameters(), lr=LR_CRITIC, weight_decay=0.0)


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


# ----------------------------------------------------------------------------- rollout
@torch.no_grad()
def collect_group(scale, rng):
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


# ----------------------------------------------------------------------------- training
base_eval = evaluate(actor)
print("\n[baseline, before PPO] greedy exact-match: " +
      " | ".join(f"{s}-bit {base_eval[f'eval_greedy_acc_s{s}']:.1f}%" for s in SCALES) +
      "   sampled: " + " | ".join(f"{s}-bit {base_eval[f'eval_sample_acc_s{s}']:.1f}%" for s in SCALES))
print("If the baseline is ~100% there is nothing for PPO to improve; if it is ~0% the +2 reward is too sparse to learn from.\n")

rng = random.Random(SEED)
rows = [{"step": 0, **base_eval}]
best_acc = base_eval["eval_greedy_acc_mean"]


def save_ckpt(path, step):
    out = {k: v for k, v in checkpoint.items() if k not in ("model", "optimizer")}
    out.update(model=actor.state_dict(), critic=critic.state_dict(), ppo_step=step)
    torch.save(out, path)


save_ckpt(os.path.join(out_dir, 'ckpt_ppo_best.pt'), 0)
print(f"{'step':>4} | {'reward':>7} | {'exact%':>6} | {'KL_ref':>7} | {'entropy':>7} | {'len':>4} | {'clip':>5} | "
      f"{'stepKL':>7} | {'v_loss':>7} | note")

for step in range(1, NUM_STEPS + 1):
    warmup = step <= CRITIC_WARMUP_STEPS

    # ---- 1. rollout (one group per scale) ----
    groups = [collect_group(s, rng) for s in SCALES]
    all_adv = torch.cat([g["adv"][g["am"]] for g in groups])
    mean, std = all_adv.mean(), all_adv.std() + 1e-8
    for g in groups:                                                    # whiten over response tokens only
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
            new_v = critic(h[:, :-1].detach())                          # critic cannot change the backbone
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
            break                                                       # trust region exceeded: new rollout

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
        note = (note + " ").lstrip() + "eval greedy: " + " ".join(f"{s}b {row[f'eval_greedy_acc_s{s}']:.0f}%" for s in SCALES)
        if row["eval_greedy_acc_mean"] > best_acc:
            best_acc = row["eval_greedy_acc_mean"]
            save_ckpt(os.path.join(out_dir, 'ckpt_ppo_best.pt'), step)
            note += " *best*"
    rows.append(row)
    print(f"{step:>4} | {row['reward']:+7.3f} | {row['exact_acc']:6.1f} | {row['kl_ref']:7.4f} | {row['entropy']:7.3f} | "
          f"{row['resp_len']:4.1f} | {row.get('clip_frac', float('nan')):5.2f} | {row.get('approx_kl', float('nan')):7.4f} | "
          f"{row.get('value_loss', float('nan')):7.4f} | {note}")

# ----------------------------------------------------------------------------- save + plots
save_ckpt(os.path.join(out_dir, 'ckpt_ppo.pt'), NUM_STEPS)
fields = sorted({k for r in rows for k in r})
with open(os.path.join(out_dir, 'ppo_metrics.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

train = [r for r in rows if r["step"] > 0]
evals = [r for r in rows if "eval_greedy_acc_mean" in r]
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot([r["step"] for r in train], [r["reward"] for r in train], color='#2ca02c', lw=2)
ax[0, 0].axhline(2.0, color='gray', ls=':', label='max verifier score (+2)')
ax[0, 0].set_title("Mean verifier score (training rollouts)"); ax[0, 0].legend()
for s in SCALES:
    ax[0, 1].plot([r["step"] for r in evals], [r[f"eval_greedy_acc_s{s}"] for r in evals], marker='o', label=f"{s}-bit")
ax[0, 1].set_title("Greedy exact-match on the fixed eval set (%)"); ax[0, 1].legend()
ax[1, 0].plot([r["step"] for r in train], [r["kl_ref"] for r in train], color='#d62728', lw=2)
ax[1, 0].set_title("KL(pi || reference) per response token")
ax[1, 1].plot([r["step"] for r in train], [r["entropy"] for r in train], color='#1f77b4', lw=2)
ax[1, 1].set_title("Policy entropy per response token")
for a in ax.flat:
    a.set_xlabel("PPO step"); a.grid(True, ls=':', alpha=0.6)
plt.tight_layout()
plot_path = os.path.join(current_dir, "ppo_training_trajectory.png")
plt.savefig(plot_path, dpi=200)

print(f"\n[*] Final checkpoint: {os.path.join(out_dir, 'ckpt_ppo.pt')}   best (greedy eval): ckpt_ppo_best.pt")
print(f"[*] Metrics: {os.path.join(out_dir, 'ppo_metrics.csv')}   plot: {plot_path}")
print(f"[*] Baseline greedy accuracy {base_eval['eval_greedy_acc_mean']:.1f}% -> final {evals[-1]['eval_greedy_acc_mean']:.1f}% "
      f"(best {best_acc:.1f}%)")