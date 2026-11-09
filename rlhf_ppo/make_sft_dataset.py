# rlhf_ppo/make_sft_dataset.py
"""
Build the SFT dataset for binary addition (prompt/response pairs, same format as the pretraining data and PPO task).

  prompt   : a (MSB first) + b (MSB first) =             e.g. 0101+0011=
  response : (a+b) written LSB FIRST, bits+1 digits, then <EOS>   e.g. 5+3=8=01000 -> 00010<EOS>

Scales: 2, 4, 8 AND 16 bits. (Your pretraining file is 6% 2-bit, 17% 4-bit, 33% 8-bit and 44% 16-bit, so an SFT set
without 16-bit would leave the biggest part of the pretraining distribution untouched and unmeasured.)

How big is each problem space?  An n-bit operand pair has 2^(2n) possibilities:
    2-bit: 16      4-bit: 256      8-bit: 65,536      16-bit: ~4.3 billion
so 2- and 4-bit are tiny enough to enumerate (a model can memorise them); 8- and 16-bit test generalisation.

Design
  * 2- and 4-bit: EXHAUSTIVE. Every pair is used; a few are held out as SFT validation (2 and 32 pairs).
  * 8-bit and 16-bit: 20,000 train + 500 validation unique pairs each, 30% "hard" (a long carry chain:
    8-bit run >= 6, natural rate 14%;  16-bit run >= 10, natural rate 8%). Train and validation are disjoint.
  * Pairs drawn by the evaluation sets in train_sft_ppo_binary_addition.py / sft_binary_addition.py (seed 1234,
    200 per scale) and compare_checkpoints.py (seed 4321, 1000 per scale) are EXCLUDED at 8 and 16 bit, so evaluation
    accuracy there cannot come from having trained on those prompts. At 2/4 bit the eval sets cover the whole space,
    so those scales cannot be disjoint: read only the 8- and 16-bit numbers as generalisation.
  * Every record is re-verified with independent arithmetic before writing.
  * Scale counts are unbalanced (14 / 224 / 20,000 / 20,000); the SFT stage weights scales per step instead of
    sampling by count (see SFT_BATCH in the SFT scripts).

Run:  python make_sft_dataset.py        (pure Python, no torch; deterministic given SEED)
Out:  sft_data/sft_train.jsonl, sft_data/sft_val.jsonl, sft_data/sft_meta.json
"""
import json
import os
import random

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_data")

SCALES = [2, 4, 8, 16]                      # keep identical (and in this order) to SCALES in the training scripts
EXHAUSTIVE = {2, 4}                         # enumerate every pair at these scales
VAL_N = {2: 2, 4: 32, 8: 500, 16: 500}
TRAIN_N = {8: 20000, 16: 20000}             # sampled scales only
HARD_FRAC = {8: 0.3, 16: 0.3}
HARD_MIN_RUN = {8: 6, 16: 10}               # "hard" = longest run of consecutive carries >= this
SEED = 0
# (seed, prompts per scale) of the evaluation sets used by the training / comparison scripts. Keep in sync with them.
EVAL_SETS = [(1234, 200), (4321, 1000)]

TOKENS = ['0', '1', '+', '=', '<EOS>', '<PAD>']
stoi = {c: i for i, c in enumerate(TOKENS)}
TOK_ADD, TOK_EQ, TOK_EOS = stoi['+'], stoi['='], stoi['<EOS>']


def carry_run(a, b, bits):
    """Longest run of consecutive carry-outs when adding a + b from the least significant bit."""
    carry, run, best = 0, 0, 0
    for j in range(bits):
        s = ((a >> j) & 1) + ((b >> j) & 1) + carry
        carry = 1 if s >= 2 else 0
        run = run + 1 if carry else 0
        best = max(best, run)
    return best


def make_record(scale, a, b):
    a_str, b_str = format(a, f'0{scale}b'), format(b, f'0{scale}b')
    ans = format(a + b, f'0{scale + 1}b')[::-1]
    return {
        "scale": scale, "a": a, "b": b, "carry_run": carry_run(a, b, scale),
        "prompt": [stoi[c] for c in a_str] + [TOK_ADD] + [stoi[c] for c in b_str] + [TOK_EQ],
        "response": [stoi[c] for c in ans] + [TOK_EOS],
        "text": f"{a_str}+{b_str}={ans}<EOS>",
    }


def eval_pairs():
    """Re-create the exact (scale, a, b) prompts the evaluation scripts draw (same RNG call order)."""
    reserved = set()
    for seed, n in EVAL_SETS:
        rng = random.Random(seed)
        for scale in SCALES:                              # one shared rng, scales in order, a then b
            for _ in range(n):
                a, b = rng.randint(0, (1 << scale) - 1), rng.randint(0, (1 << scale) - 1)
                reserved.add((scale, a, b))
    return reserved


def verify(rec):
    s = rec["scale"]
    assert len(rec["prompt"]) == 2 * s + 2 and len(rec["response"]) == s + 2
    assert rec["response"][-1] == TOK_EOS and TOK_EOS not in rec["response"][:-1]
    bits = "".join(TOKENS[t] for t in rec["response"][:-1])
    assert int(bits[::-1], 2) == rec["a"] + rec["b"], rec
    left, right = "".join(TOKENS[t] for t in rec["prompt"]).split('+')
    assert int(left, 2) == rec["a"] and int(right.rstrip('='), 2) == rec["b"], rec


def sample_pairs(scale, n_hard, n_easy, used, rng):
    """Unique random pairs not in `used`: n_hard with a long carry chain, n_easy with a short one."""
    mx, out = (1 << scale) - 1, []
    need = {"hard": n_hard, "easy": n_easy}
    tries = 0
    while need["hard"] or need["easy"]:
        tries += 1
        assert tries < 50_000_000, "sampling is not converging"
        a, b = rng.randint(0, mx), rng.randint(0, mx)
        if (a, b) in used:
            continue
        kind = "hard" if carry_run(a, b, scale) >= HARD_MIN_RUN[scale] else "easy"
        if not need[kind]:
            continue
        need[kind] -= 1
        used.add((a, b))
        out.append((a, b))
    return out


rng = random.Random(SEED)
reserved = eval_pairs()
train, val = [], []
for s in SCALES:
    if s in EXHAUSTIVE:
        all_pairs = [(a, b) for a in range(1 << s) for b in range(1 << s)]
        rng.shuffle(all_pairs)
        v_pairs, t_pairs = all_pairs[:VAL_N[s]], all_pairs[VAL_N[s]:]
    else:
        used = {(a, b) for (sc, a, b) in reserved if sc == s}            # never use eval prompts
        hv, ht = round(VAL_N[s] * HARD_FRAC[s]), round(TRAIN_N[s] * HARD_FRAC[s])
        v_pairs = sample_pairs(s, hv, VAL_N[s] - hv, used, rng)
        t_pairs = sample_pairs(s, ht, TRAIN_N[s] - ht, used, rng)       # `used` already holds the validation pairs
    val += [make_record(s, a, b) for a, b in v_pairs]
    train += [make_record(s, a, b) for a, b in t_pairs]

for rec in train + val:
    verify(rec)
key = lambda r: (r["scale"], r["a"], r["b"])
tk, vk = {key(r) for r in train}, {key(r) for r in val}
assert len(tk) == len(train) and len(vk) == len(val), "duplicate pairs inside a split"
assert not (tk & vk), "train/validation overlap"
assert not {k for k in tk | vk if k[0] not in EXHAUSTIVE and k in reserved}, "evaluation prompt leaked into SFT data"

rng.shuffle(train)
rng.shuffle(val)
os.makedirs(OUT_DIR, exist_ok=True)
for name, rows in (("sft_train.jsonl", train), ("sft_val.jsonl", val)):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def summarize(rows):
    out = {}
    for s in SCALES:
        rs = [r for r in rows if r["scale"] == s]
        thr = HARD_MIN_RUN.get(s)
        out[str(s)] = {"examples": len(rs),
                       "hard_carry_examples": sum(r["carry_run"] >= thr for r in rs) if thr else None,
                       "mean_carry_run": round(sum(r["carry_run"] for r in rs) / max(1, len(rs)), 3)}
    return out


meta = {
    "format": {
        "prompt": "a (MSB first) + b (MSB first) =", "response": "a+b, LSB first, scale+1 bits, then <EOS>",
        "tokens": TOKENS, "fields": ["scale", "a", "b", "carry_run", "prompt", "response", "text"],
        "sft_loss": "cross-entropy on `response` tokens only; `prompt` tokens are context (masked)",
    },
    "seed": SEED, "scales": SCALES, "train_total": len(train), "val_total": len(val),
    "train": summarize(train), "val": summarize(val),
    "problem_space_pairs": {str(s): 1 << (2 * s) for s in SCALES},
    "exhaustive_scales": sorted(EXHAUSTIVE),
    "hard_definition": f"longest carry chain >= {HARD_MIN_RUN} (8- and 16-bit only)",
    "excluded_eval_sets": [{"seed": sd, "prompts_per_scale": n, "scales_excluded": [s for s in SCALES if s not in EXHAUSTIVE]}
                           for sd, n in EVAL_SETS],
    "pretraining_mix_for_reference": {"2": "6%", "4": "17%", "8": "33%", "16": "44%"},
    "notes": ["2- and 4-bit use every pair (16 and 256 in total); the eval sets cover these spaces, so only 8- and "
              "16-bit accuracy measures generalisation.",
              "Scale counts are very unbalanced; the SFT stage weights scales per step (SFT_BATCH)."],
}
with open(os.path.join(OUT_DIR, "sft_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"train {len(train)} | val {len(val)} | written to {OUT_DIR}")
print(json.dumps({"train": meta["train"], "val": meta["val"]}, indent=2))
print("\nsamples:")
for r in train[:5]:
    print("  ", r["text"], f"(scale {r['scale']}, carry run {r['carry_run']})")