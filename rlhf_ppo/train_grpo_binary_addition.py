"""Train and compare GRPO with PPO for multi-scale binary addition.

GRPO starts from the KL-guard SFT checkpoint used by the conservative PPO run.
It samples several completions for each prompt and normalizes verifier rewards
within each prompt group. It has no learned value critic. The script saves
checkpoints, rollout/evaluation CSVs, comparison plots, and sample generations.

Run from this directory, for example:
    python3 train_grpo_binary_addition.py --device auto --steps 150
"""

import argparse
import copy
import csv
import json
import os
import random
import sys
import tempfile

_mpl_dir = os.path.join(tempfile.gettempdir(), "binary-addition-grpo-mplconfig")
_cache_dir = os.path.join(tempfile.gettempdir(), "binary-addition-grpo-cache")
os.makedirs(_mpl_dir, exist_ok=True)
os.makedirs(_cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_dir)
os.environ.setdefault("XDG_CACHE_HOME", _cache_dir)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NANOGPT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "NanoGPT"))
if NANOGPT_DIR not in sys.path:
    sys.path.insert(0, NANOGPT_DIR)
from model import GPT, GPTConfig

SFT_DIR = os.path.join(CURRENT_DIR, "out_binary_addition_sft_ppo_embd128_mps_klguard")
SFT_CHECKPOINT = os.path.join(SFT_DIR, "ckpt_sft.pt")
PPO_CHECKPOINT = os.path.join(SFT_DIR, "ckpt_sft_ppo.pt")
SFT_DATA_DIR = os.path.join(CURRENT_DIR, "sft_data")
DEFAULT_OUT_DIR = os.path.join(CURRENT_DIR, "out_binary_addition_grpo_klguard")

TOKENS = ["0", "1", "+", "=", "<EOS>", "<PAD>"]
STOI = {token: i for i, token in enumerate(TOKENS)}
ITOS = {i: token for i, token in enumerate(TOKENS)}
EOS_ID = STOI["<EOS>"]
PAD_ID = STOI["<PAD>"]
SCALES = [2, 4, 8, 16]
HARD_MIN_RUN = {8: 6, 16: 10}
MAX_NEW_TOKENS = 18
EVAL_PROMPTS_PER_SCALE = 200
EVAL_TOP_P = 0.9
EVAL_TEMPERATURE = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--prompts-per-scale", type=int, default=16,
                        help="Number of distinct prompts per scale per update; completions = prompts x group-size.")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Number of sampled completions per prompt for relative reward normalization.")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-prompts-per-scale", type=int, default=EVAL_PROMPTS_PER_SCALE)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--kl-beta", type=float, default=0.1)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.005,
                        help="Stop the GRPO update if sampled old-to-new KL exceeds 1.5x this value.")
    parser.add_argument("--max-kl-ref", type=float, default=0.05,
                        help="Freeze actor updates when exact sampled-state KL to the SFT reference exceeds this.")
    parser.add_argument("--hard-prompt-frac", type=float, default=0.3)
    parser.add_argument("--adv-std-floor", type=float, default=0.1)
    parser.add_argument("--comparison-only", action="store_true",
                        help="Skip GRPO training and compare SFT/PPO with an existing GRPO checkpoint.")
    return parser.parse_args()


def choose_device(name):
    if name != "auto":
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def forward_logits(model, ids):
    # NanoGPT's public forward returns only the final-position logits when no
    # target tensor is supplied. GRPO needs logits at every sampled action.
    batch, length = ids.shape
    positions = torch.arange(length, dtype=torch.long, device=ids.device)
    x = model.transformer.drop(model.transformer.wte(ids) + model.transformer.wpe(positions))
    for block in model.transformer.h:
        x = block(x)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = dict(ckpt["model"])
    for key in list(state):
        if key.startswith("_orig_mod."):
            state[key[len("_orig_mod."):]] = state.pop(key)
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt


def save_checkpoint(path, model, source_ckpt, **extra):
    out = {k: v for k, v in source_ckpt.items() if k not in ("model", "optimizer")}
    out.update(model=model.state_dict(), **extra)
    torch.save(out, path)


def carry_run(a, b, bits):
    carry = run = best = 0
    for j in range(bits):
        total = ((a >> j) & 1) + ((b >> j) & 1) + carry
        carry = int(total >= 2)
        run = run + 1 if carry else 0
        best = max(best, run)
    return best


def prompt_for(a, b, bits):
    aa, bb = format(a, f"0{bits}b"), format(b, f"0{bits}b")
    prompt = [STOI[c] for c in aa] + [STOI["+"]] + [STOI[c] for c in bb] + [STOI["="]]
    expected = format(a + b, f"0{bits + 1}b")[::-1]
    return prompt, expected, f"{aa}+{bb}="


def sample_prompts(bits, n, rng, hard_fraction=0.0):
    n_hard = round(n * hard_fraction) if bits in HARD_MIN_RUN else 0
    rows, meta = [], []
    while len(rows) < n:
        a, b = rng.randint(0, (1 << bits) - 1), rng.randint(0, (1 << bits) - 1)
        crun = carry_run(a, b, bits)
        if len(rows) < n_hard and crun < HARD_MIN_RUN[bits]:
            continue
        prompt, expected, prompt_text = prompt_for(a, b, bits)
        rows.append(prompt)
        meta.append({"a": a, "b": b, "expected": expected, "prompt": prompt_text, "carry_run": crun})
    return torch.tensor(rows, dtype=torch.long), meta


def expected_from_prompt(prompt):
    text = "".join(ITOS[int(t)] for t in prompt if int(t) != PAD_ID)
    left, right = text.split("+")
    right = right.rstrip("=")
    n = len(left)
    return format(int(left, 2) + int(right, 2), f"0{n + 1}b")[::-1]


def response_mask(response):
    eos_before = torch.cumsum((response == EOS_ID).long(), dim=1) - (response == EOS_ID).long()
    return eos_before == 0


@torch.no_grad()
def sample_completions(actor, prompts, max_new_tokens):
    """Categorical temperature-1 sampling; pad rows after EOS while other rows finish."""
    actor.eval()
    ids = prompts
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
    generated = []
    for _ in range(max_new_tokens):
        logits = forward_logits(actor, ids)[:, -1, :]
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
        nxt = torch.where(finished, torch.full_like(nxt, PAD_ID), nxt)
        generated.append(nxt)
        finished |= nxt == EOS_ID
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        if bool(finished.all()):
            break
    if generated:
        resp = torch.stack(generated, dim=1)
    else:
        resp = torch.empty((prompts.size(0), 0), dtype=torch.long, device=prompts.device)
    return torch.cat([prompts, resp], dim=1), resp


def score_response(response, expected):
    ids = [int(t) for t in response]
    eos = EOS_ID in ids
    first_eos = ids.index(EOS_ID) if eos else len(ids)
    pred = "".join(ITOS.get(t, "") for t in ids[:first_eos] if t != PAD_ID)
    matches = sum(i < len(pred) and pred[i] == bit for i, bit in enumerate(expected))
    exact = pred == expected and eos
    L = len(expected)
    task_reward = -1.0 + 2.0 * matches / L + float(exact)
    return task_reward, exact, matches / L, pred, eos


def make_eval_prompts(n, seed, device):
    rng = random.Random(seed)
    prompts, meta = {}, {}
    for bits in SCALES:
        x, m = sample_prompts(bits, n, rng)
        prompts[bits] = x.to(device)
        meta[bits] = m
    return prompts, meta


@torch.no_grad()
def generate_eval(model, prompts, max_new_tokens, mode, top_p, temperature):
    model.eval()
    ids = prompts
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        logits = forward_logits(model, ids)[:, -1, :]
        if mode == "greedy":
            nxt = logits.argmax(dim=-1)
        else:
            logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, sorted_indices, sorted_logits)
            nxt = torch.multinomial(F.softmax(filtered, dim=-1), 1).squeeze(1)
        nxt = torch.where(finished, torch.full_like(nxt, PAD_ID), nxt)
        finished |= nxt == EOS_ID
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        if bool(finished.all()):
            break
    return ids[:, prompts.size(1):]


@torch.no_grad()
def evaluate_model(model, eval_prompts, eval_meta, seed):
    rows, summary = [], {}
    for bits in SCALES:
        prompts = eval_prompts[bits]
        for mode in ("greedy", "sample"):
            torch.manual_seed(seed + bits * 100 + (mode == "sample"))
            responses = generate_eval(model, prompts, MAX_NEW_TOKENS, mode, EVAL_TOP_P, EVAL_TEMPERATURE)
            n_exact = n_bits = n_eos = 0
            for i, meta in enumerate(eval_meta[bits]):
                _, exact, bit_acc, pred, eos = score_response(responses[i].tolist(), meta["expected"])
                n_exact += int(exact)
                n_bits += bit_acc
                n_eos += int(eos)
                rows.append({"scale": bits, "mode": mode, "case": i, "a": meta["a"], "b": meta["b"],
                             "carry_run": meta["carry_run"], "prompt": meta["prompt"],
                             "expected_bits": meta["expected"], "predicted_bits": pred,
                             "emitted_eos": eos, "exact": exact, "bit_accuracy_pct": 100 * bit_acc})
            summary[(bits, mode)] = {
                "n": len(eval_meta[bits]), "exact_accuracy_pct": 100 * n_exact / len(eval_meta[bits]),
                "mean_bit_accuracy_pct": 100 * n_bits / len(eval_meta[bits]),
                "eos_rate_pct": 100 * n_eos / len(eval_meta[bits]),
            }
    return summary, rows


def load_validation_data(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"SFT validation set not found: {path}")
    by_scale = {s: [] for s in SCALES}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            by_scale[rec["scale"]].append((rec["prompt"], rec["response"]))
    return {s: [(torch.tensor(p, dtype=torch.long, device=device),
                 "".join(ITOS[int(t)] for t in r[:-1])) for p, r in examples]
            for s, examples in by_scale.items()}


@torch.no_grad()
def validation_accuracy(model, data, scale=16):
    correct = 0
    examples = data[scale]
    prompts = torch.stack([prompt for prompt, _ in examples])
    responses = generate_eval(model, prompts, MAX_NEW_TOKENS, "greedy", EVAL_TOP_P, EVAL_TEMPERATURE)
    for i, (_, expected) in enumerate(examples):
        response = responses[i]
        _, exact, _, _, _ = score_response(response.tolist(), expected)
        correct += int(exact)
    return 100.0 * correct / max(1, len(examples))


def rollout_one_scale(actor, reference, bits, args, rng, device):
    prompts_cpu, meta = sample_prompts(bits, args.prompts_per_scale, rng, args.hard_prompt_frac)
    prompts = prompts_cpu.to(device).repeat_interleave(args.group_size, dim=0)
    seq, responses = sample_completions(actor, prompts, MAX_NEW_TOKENS)
    response_mask_ = response_mask(responses)
    B, P = prompts.shape
    logits = forward_logits(actor, seq[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    old_lp = logp.gather(-1, seq[:, 1:, None]).squeeze(-1).detach()
    ref_logits = forward_logits(reference, seq[:, :-1])
    ref_logp = F.log_softmax(ref_logits, dim=-1).detach()
    ref_token_lp = ref_logp.gather(-1, seq[:, 1:, None]).squeeze(-1)
    action_mask = torch.zeros_like(old_lp, dtype=torch.bool)
    action_mask[:, P - 1:P - 1 + responses.size(1)] = response_mask_
    probs = logp.exp()
    full_kl = (probs * (logp - ref_logp)).sum(-1)
    kl_ref = (full_kl * action_mask).sum() / action_mask.sum().clamp_min(1)

    rewards, exact_flags, bit_accs = [], [], []
    for j, expected in enumerate([m["expected"] for m in meta]):
        for g in range(args.group_size):
            reward, exact, bit_acc, _, _ = score_response(responses[j * args.group_size + g].tolist(), expected)
            rewards.append(reward)
            exact_flags.append(float(exact))
            bit_accs.append(bit_acc)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).view(len(meta), args.group_size)
    mean = reward_tensor.mean(dim=1, keepdim=True)
    std = reward_tensor.std(dim=1, keepdim=True, unbiased=False)
    advantages = (reward_tensor - mean) / std.clamp_min(args.adv_std_floor)
    advantages = torch.where(std > 1e-8, advantages, torch.zeros_like(advantages)).reshape(-1).detach()
    return {
        "seq": seq, "old_lp": old_lp, "ref_logp": ref_logp, "ref_token_lp": ref_token_lp,
        "mask": action_mask, "adv": advantages, "kl_ref": kl_ref.detach(), "rewards": reward_tensor.detach(),
        "exact": torch.tensor(exact_flags, dtype=torch.float32, device=device),
        "bit_acc": torch.tensor(bit_accs, dtype=torch.float32, device=device), "meta": meta, "bits": bits,
    }


def grpo_update(actor, groups, optimizer, args, actor_frozen):
    total_tokens = sum(g["mask"].sum() for g in groups).float().clamp_min(1)
    policy_loss = torch.zeros((), device=next(actor.parameters()).device)
    kl_loss = torch.zeros_like(policy_loss)
    approx_kl = 0.0
    for g in groups:
        logits = forward_logits(actor, g["seq"][:, :-1])
        logp = F.log_softmax(logits, dim=-1)
        new_lp = logp.gather(-1, g["seq"][:, 1:, None]).squeeze(-1)
        log_ratio = new_lp - g["old_lp"]
        ratio = log_ratio.exp()
        adv = g["adv"][:, None]
        surrogate = torch.minimum(ratio * adv,
                                  ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * adv)
        policy_loss = policy_loss - (surrogate * g["mask"]).sum() / total_tokens
        probs = logp.exp()
        token_kl = (probs * (logp - g["ref_logp"])).sum(-1)
        kl_loss = kl_loss + (token_kl * g["mask"]).sum() / total_tokens
        sampled_kl = ((ratio - 1) - log_ratio) * g["mask"]
        approx_kl += sampled_kl.sum().detach().item() / total_tokens.item()
    if actor_frozen or approx_kl > 1.5 * args.target_kl:
        return {"policy_loss": float(policy_loss.detach()), "kl_loss": float(kl_loss.detach()),
                "approx_kl": approx_kl, "updated": False}
    loss = policy_loss + args.kl_beta * kl_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()
    return {"policy_loss": float(policy_loss.detach()), "kl_loss": float(kl_loss.detach()),
            "approx_kl": approx_kl, "updated": True}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison(models, eval_prompts, eval_meta, out_dir, seed):
    all_rows, raw_rows = [], []
    for name, model in models.items():
        summary, cases = evaluate_model(model, eval_prompts, eval_meta, seed)
        for (bits, mode), values in summary.items():
            all_rows.append({"model": name, "scale": bits, "mode": mode, **values})
        for row in cases:
            raw_rows.append({"model": name, **row})
    write_csv(os.path.join(out_dir, "comparison_metrics.csv"), all_rows)
    write_csv(os.path.join(out_dir, "comparison_cases.csv"), raw_rows)
    return all_rows


def plot_comparison(rows, path):
    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mode in zip(axes, ("greedy", "sample")):
        for name in models:
            data = [next(r for r in rows if r["model"] == name and r["scale"] == s and r["mode"] == mode)
                    for s in SCALES]
            ax.plot(SCALES, [r["exact_accuracy_pct"] for r in data], marker="o", label=name)
        ax.set_xticks(SCALES, [f"{s}-bit" for s in SCALES])
        ax.set_ylim(-2, 102)
        ax.set_ylabel("Exact-match accuracy (%)")
        ax.set_title(f"{mode.title()} evaluation")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)
    fig.suptitle("Binary addition: SFT, PPO, and GRPO checkpoints")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training(rows, path):
    train = [r for r in rows if r["step"] > 0]
    evals = [r for r in rows if "eval_greedy_acc_s16" in r]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot([r["step"] for r in train], [r["rollout_reward"] for r in train])
    axes[0, 0].set_title("Mean verifier reward (GRPO rollouts)")
    axes[0, 1].plot([r["step"] for r in evals], [r["eval_greedy_acc_s16"] for r in evals], marker="o")
    axes[0, 1].set_title("16-bit greedy exact accuracy (%)")
    axes[1, 0].plot([r["step"] for r in train], [r["kl_ref"] for r in train], color="tab:red")
    axes[1, 0].axhline(0.05, color="black", linestyle="--", label="actor freeze cap")
    axes[1, 0].set_title("Full-distribution KL to SFT reference")
    axes[1, 0].legend()
    axes[1, 1].plot([r["step"] for r in train], [r["group_reward_std"] for r in train], color="tab:green")
    axes[1, 1].set_title("Within-prompt group reward spread")
    for ax in axes.flat:
        ax.set_xlabel("GRPO update")
        ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def train_grpo(actor, reference, source_ckpt, args, device, eval_prompts, eval_meta, validation):
    os.makedirs(args.output_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    initial_val = validation_accuracy(actor, validation)
    best_val, best_step = initial_val, 0
    best_state = copy.deepcopy(actor.state_dict())
    save_checkpoint(os.path.join(args.output_dir, "ckpt_grpo_best.pt"), actor, source_ckpt,
                    grpo_step=0, validation_16bit_greedy=initial_val)
    initial_summary, _ = evaluate_model(actor, eval_prompts, eval_meta, args.seed)
    rows = [{"step": 0, "rollout_reward": None, "rollout_exact_pct": None,
             "group_reward_std": None, "kl_ref": 0.0, "actor_frozen": 0,
             "eval_greedy_acc_s16": initial_summary[(16, "greedy")]["exact_accuracy_pct"],
             "eval_greedy_acc_mean": sum(initial_summary[(s, "greedy")]["exact_accuracy_pct"] for s in SCALES) / len(SCALES),
             "val_greedy_acc_s16": initial_val}]
    print(f"[GRPO] SFT init: validation 16-bit greedy={initial_val:.1f}%")

    for step in range(1, args.steps + 1):
        actor.eval()
        groups = [rollout_one_scale(actor, reference, bits, args, rng, device) for bits in SCALES]
        kl_ref = sum(float(g["kl_ref"]) for g in groups) / len(groups)
        actor_frozen = kl_ref > args.max_kl_ref
        update = grpo_update(actor, groups, optimizer, args, actor_frozen)
        rewards = torch.cat([g["rewards"].reshape(-1) for g in groups])
        exact = torch.cat([g["exact"] for g in groups])
        bitacc = torch.cat([g["bit_acc"] for g in groups])
        group_stds = torch.cat([g["rewards"].std(dim=1, unbiased=False) for g in groups])
        row = {
            "step": step,
            "rollout_reward": rewards.mean().item(),
            "rollout_exact_pct": 100 * exact.mean().item(),
            "rollout_bit_accuracy_pct": 100 * bitacc.mean().item(),
            "group_reward_std": group_stds.mean().item(),
            "kl_ref": kl_ref,
            "actor_frozen": int(actor_frozen),
            **update,
            **{f"rollout_exact_s{g['bits']}": 100 * g["exact"].mean().item() for g in groups},
        }
        if step % args.eval_every == 0 or step == args.steps:
            eval_summary, _ = evaluate_model(actor, eval_prompts, eval_meta, args.seed)
            row.update({f"eval_{mode}_acc_s{bits}": eval_summary[(bits, mode)]["exact_accuracy_pct"]
                        for bits in SCALES for mode in ("greedy", "sample")})
            row["eval_greedy_acc_mean"] = sum(eval_summary[(s, "greedy")]["exact_accuracy_pct"] for s in SCALES) / len(SCALES)
            row["val_greedy_acc_s16"] = validation_accuracy(actor, validation)
            if row["val_greedy_acc_s16"] > best_val:
                best_val = row["val_greedy_acc_s16"]
                best_step = step
                best_state = copy.deepcopy(actor.state_dict())
                save_checkpoint(os.path.join(args.output_dir, "ckpt_grpo_best.pt"), actor, source_ckpt,
                                grpo_step=step, validation_16bit_greedy=best_val)
        rows.append(row)
        print(f"[GRPO] {step:4d}/{args.steps} reward={row['rollout_reward']:+.3f} "
              f"rollout_exact={row['rollout_exact_pct']:5.1f}% KL_ref={kl_ref:.5f} "
              f"group_std={row['group_reward_std']:.3f} updated={update['updated']} "
              f"frozen={actor_frozen} val16={row.get('val_greedy_acc_s16', float('nan')):.1f}%")
    actor.load_state_dict(best_state)
    save_checkpoint(os.path.join(args.output_dir, "ckpt_grpo.pt"), actor, source_ckpt,
                    grpo_step=best_step, validation_16bit_greedy=best_val, selected_by="sft_val_16bit_greedy")
    write_csv(os.path.join(args.output_dir, "grpo_metrics.csv"), rows)
    plot_training(rows, os.path.join(args.output_dir, "grpo_training_trajectory.png"))
    print(f"[GRPO] selected step {best_step}, validation 16-bit greedy={best_val:.1f}%")
    return rows


def write_sample_generations(models, eval_prompts, eval_meta, output_path, seed):
    lines = []
    for bits in SCALES:
        indices = list(range(min(5, len(eval_meta[bits]))))
        for mode in ("greedy", "sample"):
            for name, model in models.items():
                torch.manual_seed(seed + bits * 100 + (mode == "sample"))
                responses = generate_eval(model, eval_prompts[bits][:max(indices) + 1], MAX_NEW_TOKENS,
                                          mode, EVAL_TOP_P, EVAL_TEMPERATURE)
                lines.append(f"\n[{bits}-bit | {mode} | {name}]")
                for i in indices:
                    m = eval_meta[bits][i]
                    _, exact, bitacc, pred, eos = score_response(responses[i].tolist(), m["expected"])
                    lines.append(f"{m['prompt']} -> {pred}{'<EOS>' if eos else ''} | expected={m['expected']} "
                                 f"| exact={exact} | bit_acc={bitacc:.3f}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.steps < 1 or args.prompts_per_scale < 1 or args.group_size < 2:
        raise ValueError("steps and prompts-per-scale must be positive; group-size must be at least 2")
    if args.eval_every < 1 or args.eval_prompts_per_scale < 1:
        raise ValueError("evaluation intervals and prompt count must be positive")
    device = choose_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"GRPO device={device}; output={args.output_dir}")
    print(f"SFT initialization={SFT_CHECKPOINT}")

    sft_model, sft_ckpt = load_model(SFT_CHECKPOINT, device)
    reference = copy.deepcopy(sft_model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    validation = load_validation_data(os.path.join(SFT_DATA_DIR, "sft_val.jsonl"), device)
    eval_prompts, eval_meta = make_eval_prompts(args.eval_prompts_per_scale, args.seed, device)

    grpo_path = os.path.join(args.output_dir, "ckpt_grpo.pt")
    if args.comparison_only:
        if not os.path.exists(grpo_path):
            raise FileNotFoundError(f"GRPO checkpoint not found for comparison: {grpo_path}")
        grpo_model, _ = load_model(grpo_path, device)
    else:
        actor = copy.deepcopy(sft_model).train()
        # The generation/evaluation path disables dropout; the training forward is deterministic too,
        # matching the eval-mode PPO actor used for likelihood ratios.
        actor.eval()
        train_grpo(actor, reference, sft_ckpt, args, device, eval_prompts, eval_meta, validation)
        grpo_model, _ = load_model(grpo_path, device)

    models = {"SFT (KL-guard run)": sft_model, "GRPO (selected)": grpo_model}
    if os.path.exists(PPO_CHECKPOINT):
        ppo_model, _ = load_model(PPO_CHECKPOINT, device)
        models["PPO (selected)"] = ppo_model
    comparison_rows = save_comparison(models, eval_prompts, eval_meta, args.output_dir, args.seed)
    plot_comparison(comparison_rows, os.path.join(args.output_dir, "model_comparison.png"))
    write_sample_generations(models, eval_prompts, eval_meta,
                             os.path.join(args.output_dir, "sample_generations.txt"), args.seed)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "device_resolved": str(device), "sft_checkpoint": SFT_CHECKPOINT,
                   "ppo_checkpoint": PPO_CHECKPOINT if os.path.exists(PPO_CHECKPOINT) else None,
                   "group_completions": args.group_size, "sft_reference": True}, f, indent=2)
    print(f"Saved GRPO/PPO/SFT metrics and plots under {args.output_dir}")
    for row in comparison_rows:
        print(f"{row['model']:<22} {row['scale']:>2}-bit {row['mode']:<6} "
              f"exact={row['exact_accuracy_pct']:5.1f}% bit={row['mean_bit_accuracy_pct']:5.1f}% "
              f"(n={row['n']})")


if __name__ == "__main__":
    main()
