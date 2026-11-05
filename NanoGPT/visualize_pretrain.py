import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import GPT, GPTConfig

device = 'cpu'
ckpt_path = 'out_binary_addition/ckpt.pt'

print(f"Loading checkpoint from {ckpt_path}...")
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

# Vocabulary mappings
TOKENS = ['0', '1', '+', '=', '<EOS>', '<PAD>']
stoi = {ch: i for i, ch in enumerate(TOKENS)}
itos = {i: ch for i, ch in enumerate(TOKENS)}
bits = 4

def make_equation_str(a, b):
    a_str = format(a, f'0{bits}b')
    b_str = format(b, f'0{bits}b')
    ans_int = a + b
    ans_bin = format(ans_int, f'0{bits + 1}b')
    ans_lsb = ans_bin[::-1]
    prompt_str = f"{a_str}+{b_str}="
    full_toks = [stoi[c] for c in a_str] + [stoi['+']] + [stoi[c] for c in b_str] + [stoi['=']]
    return prompt_str, ans_lsb, full_toks

# 1. Qualitative Evaluation on Unseen Problems
print("\n" + "="*50)
print("  PRETRAINED BASELINE: EVALUATION SAMPLES")
print("="*50)
test_cases = [(3, 5), (7, 8), (12, 14), (1, 2), (15, 15)]
for a, b in test_cases:
    prompt_str, expected_lsb, prompt_toks = make_equation_str(a, b)
    x = torch.tensor([prompt_toks], dtype=torch.long, device=device)
    
    with torch.no_grad():
        out = x
        for _ in range(bits + 2):
            logits, _ = model(out)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
            if next_tok.item() == stoi['<EOS>']:
                break
                
    generated_toks = out[0, len(prompt_toks):].tolist()
    gen_str = "".join([itos.get(t, '?') for t in generated_toks])
    status = "SUCCESS" if gen_str.replace('<EOS>', '') == expected_lsb else "FAIL"
    print(f"Prompt: {prompt_str:<12} | Generated: {gen_str:<10} | Expected (LSB): {expected_lsb} | [{status}]")

# 2. Extract Attention Map via Dynamic Causal Mask (No dependency on module.bias)
sample_prompt, sample_exp, sample_seq = make_equation_str(7, 8)
seq_tensor = torch.tensor([sample_seq], dtype=torch.long, device=device)

attn_weights = []
def hook_fn(module, input, output):
    B, T, C = input[0].size()
    q, k, v = module.c_attn(input[0]).split(module.n_embd, dim=2)
    k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
    q = q.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
    # Dynamic causal mask
    causal_mask = torch.tril(torch.ones(T, T, device=input[0].device)).view(1, 1, T, T)
    att = att.masked_fill(causal_mask == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    attn_weights.append(att.detach().cpu())

hook = model.transformer.h[0].attn.register_forward_hook(hook_fn)
with torch.no_grad():
    model(seq_tensor)
hook.remove()

# Extract Layer 0, Head 0 attention
attn_map = attn_weights[0][0, 0].numpy()
tok_labels = [itos[t] for t in sample_seq]

# 3. Generate Visualization Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot A: Attention Weights
im = ax1.imshow(attn_map, cmap='Blues', aspect='auto')
ax1.set_xticks(range(len(tok_labels)))
ax1.set_yticks(range(len(tok_labels)))
ax1.set_xticklabels(tok_labels, fontsize=9)
ax1.set_yticklabels(tok_labels, fontsize=9)
ax1.set_title("Self-Attention Pattern (Layer 0, Head 0)", fontweight="bold")
ax1.set_xlabel("Key Position (Input Context)")
ax1.set_ylabel("Query Position (Generation Step)")
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

# Plot B: Benchmark Loss Horizon
val_loss = checkpoint.get('best_val_loss', 0.10)
bars = ax2.bar(
    ["Random Guess\nln(m=6)", "Trained Base\nNanoGPT", "Optimal Filter\nGoal"], 
    [np.log(6), val_loss, 0.0],
    color=["#7f7f7f", "#1f77b4", "#2ca02c"],
    edgecolor="black",
    width=0.5
)
ax2.set_ylabel("Cross-Entropy Loss (nats)")
ax2.set_title("Base Model Loss vs. Theoretical Bounds", fontweight="bold")
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
out_fig = "pretrain_results.png"
plt.savefig(out_fig, dpi=300)
print("\n" + "="*50)
print(f"Visualization saved to: {os.path.abspath(out_fig)}")
print("="*50)
