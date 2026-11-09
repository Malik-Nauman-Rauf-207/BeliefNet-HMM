import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import GPT, GPTConfig

device = 'cpu'
ckpt_path = 'out_binary_addition_multiscale_embd128_mps/ckpt.pt'

print(f"Loading checkpoint from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Vocabulary mapping
TOKENS = ['0', '1', '+', '=', '<EOS>', '<PAD>']
stoi = {ch: i for i, ch in enumerate(TOKENS)}
itos = {i: ch for i, ch in enumerate(TOKENS)}
EOS_ID = stoi['<EOS>']
EQ_ID = stoi['=']
ADD_ID = stoi['+']

def format_eq(a: int, b: int, bits: int):
    a_str = format(a, f'0{bits}b')
    b_str = format(b, f'0{bits}b')
    ans_int = a + b
    ans_bin = format(ans_int, f'0{bits + 1}b')
    ans_lsb = ans_bin[::-1]
    prompt_str = f"{a_str}+{b_str}="
    prompt_tokens = [stoi[c] for c in a_str] + [ADD_ID] + [stoi[c] for c in b_str] + [EQ_ID]
    return prompt_str, ans_lsb, ans_int, prompt_tokens

@torch.no_grad()
def greedy_generate(prompt_tokens, max_new_tokens):
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, next_tok], dim=1)
        if next_tok.item() == EOS_ID:
            break
    resp_tokens = x[0, len(prompt_tokens):].tolist()
    return resp_tokens

# =============================================================================
# 1. Multi-Scale Quantitative Benchmark
# =============================================================================
scales = [2, 4, 8, 16]
samples_per_scale = 50
scale_results = {}

print("\n" + "=" * 80)
print(f"{'Scale':<8} | {'Correct/Total':<14} | {'Exact Accuracy (%)':<20} | {'Carry-Flip Accuracy (%)':<22}")
print("=" * 80)

random.seed(42)

for bits in scales:
    max_val = (1 << bits) - 1
    exact_success = 0
    bit_matches = 0
    total_bits = 0
    diagnostic_samples = []
    
    for _ in range(samples_per_scale):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        prompt_str, expected_lsb, ans_int, prompt_tokens = format_eq(a, b, bits)
        
        # Generation: expected len is bits + 1 answer tokens + 1 EOS
        resp_toks = greedy_generate(prompt_tokens, max_new_tokens=bits + 4)
        resp_str = "".join([itos.get(t, '?') for t in resp_toks])
        
        cleaned_resp = resp_str.replace('<EOS>', '').replace('<PAD>', '')
        is_exact = (cleaned_resp == expected_lsb)
        if is_exact:
            exact_success += 1
            
        # Per-bit accuracy evaluation
        for i in range(min(len(cleaned_resp), len(expected_lsb))):
            if cleaned_resp[i] == expected_lsb[i]:
                bit_matches += 1
        total_bits += len(expected_lsb)
        
        if len(diagnostic_samples) < 3:
            diagnostic_samples.append({
                "prompt": prompt_str,
                "expected": expected_lsb,
                "generated": resp_str,
                "exact": is_exact
            })
            
    acc = (exact_success / samples_per_scale) * 100.0
    bit_acc = (bit_matches / total_bits) * 100.0 if total_bits > 0 else 0.0
    scale_results[bits] = {
        "acc": acc,
        "bit_acc": bit_acc,
        "samples": diagnostic_samples
    }
    print(f"{f'{bits}-bit':<8} | {f'{exact_success}/{samples_per_scale}':<14} | {f'{acc:5.1f}%':<20} | {f'{bit_acc:5.1f}%':<22}")

print("=" * 80)

# Print Detailed Rollouts
print("\n" + "=" * 80)
print("  ACTUAL ROLLOUT EXAMPLES ACROSS BIT SCALES")
print("=" * 80)
for bits in scales:
    print(f"\n--- [{bits}-bit Additions] ---")
    for s in scale_results[bits]["samples"]:
        status = "PASSED" if s["exact"] else "FAILED"
        print(f"Prompt: {s['prompt']:<38} | Generated: {s['generated']:<22} | Expected: {s['expected']:<18} | [{status}]")

# =============================================================================
# 2. Attention Multi-Head Extraction (8-bit Case Study)
# =============================================================================
demo_a, demo_b, demo_bits = 13, 27, 8  # 00001101 + 00011011
p_str, exp_lsb, _, p_toks = format_eq(demo_a, demo_b, demo_bits)
ans_toks = [stoi[c] for c in exp_lsb] + [EOS_ID]
full_eval_seq = p_toks + ans_toks
seq_tensor = torch.tensor([full_eval_seq], dtype=torch.long, device=device)

all_heads_attn = []
def hook_all_heads(module, input, output):
    B, T, C = input[0].size()
    q, k, v = module.c_attn(input[0]).split(module.n_embd, dim=2)
    k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
    q = q.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
    causal_mask = torch.tril(torch.ones(T, T, device=input[0].device)).view(1, 1, T, T)
    att = att.masked_fill(causal_mask == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    all_heads_attn.append(att.detach().cpu())

hook = model.transformer.h[0].attn.register_forward_hook(hook_all_heads)
with torch.no_grad():
    model(seq_tensor)
hook.remove()

# =============================================================================
# 3. Comprehensive 4-Panel Visualization Plot
# =============================================================================
fig = plt.figure(figsize=(16, 10))

# Subplot 1: Accuracy Degradation Curve vs Bit Scale
ax1 = plt.subplot(2, 2, 1)
accs = [scale_results[b]["acc"] for b in scales]
bit_accs = [scale_results[b]["bit_acc"] for b in scales]
x_indices = np.arange(len(scales))
ax1.plot(scales, accs, marker='o', lw=2.5, color='#1f77b4', label='Exact Sequence Accuracy (%)')
ax1.plot(scales, bit_accs, marker='s', lw=2.0, linestyle='--', color='#2ca02c', label='Per-Bit Arithmetic Accuracy (%)')
ax1.set_xticks(scales)
ax1.set_xticklabels([f"{b}-bit" for b in scales])
ax1.set_xlabel("Bit Scale Width (Operand Horizon)", fontweight="bold")
ax1.set_ylabel("Accuracy (%)", fontweight="bold")
ax1.set_ylim(-5, 105)
ax1.set_title("Multi-Scale Arithmetic Generalization", fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()
for i, b in enumerate(scales):
    ax1.text(b, accs[i] + 3, f"{accs[i]:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#1f77b4')

# Subplot 2: Carry Length Distribution Failure Histogram
ax2 = plt.subplot(2, 2, 2)
# Evaluate ripple carry complexity on 100 randomly sampled 8-bit problems
ripple_lengths = []
success_by_ripple = []
for _ in range(120):
    a = random.randint(0, 255)
    b = random.randint(0, 255)
    # Count consecutive carry cascades
    a_bin = [int(c) for c in format(a, '08b')[::-1]]
    b_bin = [int(c) for c in format(b, '08b')[::-1]]
    carry = 0
    max_c_len = 0
    curr_c = 0
    for ba, bb in zip(a_bin, b_bin):
        if ba + bb + carry >= 2:
            carry = 1
            curr_c += 1
            max_c_len = max(max_c_len, curr_c)
        else:
            carry = 0
            curr_c = 0
    _, exp_lsb, _, p_toks = format_eq(a, b, 8)
    resp = greedy_generate(p_toks, max_new_tokens=12)
    clean = "".join([itos.get(t, '') for t in resp]).replace('<EOS>', '').replace('<PAD>', '')
    ripple_lengths.append(max_c_len)
    success_by_ripple.append(1 if clean == exp_lsb else 0)

max_r = max(ripple_lengths) if ripple_lengths else 1
r_bins = np.arange(0, max_r + 2) - 0.5
ax2.hist(
    [np.array(ripple_lengths)[np.array(success_by_ripple) == 1],
     np.array(ripple_lengths)[np.array(success_by_ripple) == 0]],
    bins=r_bins, stacked=True, color=['#2ca02c', '#d62728'],
    label=['Exact Success', 'Carry Failure'], edgecolor='black', rwidth=0.7
)
ax2.set_xlabel("Max Consecutive Ripple Carry Length (8-bit)", fontweight="bold")
ax2.set_ylabel("Number of Equations", fontweight="bold")
ax2.set_title("Performance Breakdown by Carry Cascade Depth", fontweight="bold")
ax2.set_xticks(range(max_r + 1))
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend()

# Subplot 3 & 4: Multi-Head Attention Maps (Head 0 vs Head 1)
tok_labels = [itos[t] for t in full_eval_seq]
attn_head_0 = all_heads_attn[0][0, 0].numpy()
attn_head_1 = all_heads_attn[0][0, 1].numpy()

ax3 = plt.subplot(2, 2, 3)
im3 = ax3.imshow(attn_head_0, cmap='Blues', aspect='auto')
ax3.set_title("Layer 0 - Head 0 (Positional / Delimiter Tracking)", fontweight="bold")
ax3.set_xlabel("Key Position")
ax3.set_ylabel("Query Position")
fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

ax4 = plt.subplot(2, 2, 4)
im4 = ax4.imshow(attn_head_1, cmap='Purples', aspect='auto')
ax4.set_title("Layer 0 - Head 1 (Operand Bit / Carry Routing)", fontweight="bold")
ax4.set_xlabel("Key Position")
ax4.set_ylabel("Query Position")
fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

plt.tight_layout()
out_img = "multiscale_evaluation_analysis.png"
plt.savefig(out_img, dpi=300)
print("\n" + "=" * 80)
print(f"Comprehensive analysis plot saved to: {os.path.abspath(out_img)}")
print("=" * 80)
