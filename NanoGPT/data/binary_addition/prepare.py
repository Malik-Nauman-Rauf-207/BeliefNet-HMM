import os
import pickle
import numpy as np

# 1. Define Token Space (m = 6)
TOKENS = ['0', '1', '+', '=', '<EOS>', '<PAD>']
vocab_size = len(TOKENS)
stoi = {ch: i for i, ch in enumerate(TOKENS)}
itos = {i: ch for i, ch in enumerate(TOKENS)}

TOK_0   = stoi['0']
TOK_1   = stoi['1']
TOK_ADD = stoi['+']
TOK_EQ  = stoi['=']
TOK_EOS = stoi['<EOS>']
TOK_PAD = stoi['<PAD>']

def int_to_bin_tokens(n: int, bits: int = 4):
    """Converts an integer to a fixed-width binary token list."""
    s = format(n, f'0{bits}b')
    return [stoi[c] for c in s]

def make_equation(a: int, b: int, bits: int = 4):
    """
    Constructs: [A] + [B] = [Sum_LSB] <EOS>
    Sum is emitted Little-Endian (LSB first) to enable step-by-step carry evaluation.
    """
    a_toks = int_to_bin_tokens(a, bits=bits)
    b_toks = int_to_bin_tokens(b, bits=bits)
    
    # Calculate sum and reverse the bits for LSB-first autoregressive generation
    ans_int = a + b
    ans_bin = format(ans_int, f'0{bits + 1}b')
    ans_toks = [stoi[c] for c in ans_bin[::-1]]  # LSB first
    
    seq = a_toks + [TOK_ADD] + b_toks + [TOK_EQ] + ans_toks + [TOK_EOS]
    return seq

# 2. Build All Combinations for 4-bit addition (0..15 + 0..15 = 256 pairs)
bits = 4
all_pairs = [(a, b) for a in range(2**bits) for b in range(2**bits)]
np.random.seed(42)
np.random.shuffle(all_pairs)

# Split: 80% train (204 equations), 20% validation (52 equations)
split_idx = int(0.8 * len(all_pairs))
train_pairs = all_pairs[:split_idx]
val_pairs = all_pairs[split_idx:]

# Repeat sequences to provide sufficient tokens for pre-training steps
train_seqs = [make_equation(a, b, bits=bits) for a, b in train_pairs] * 200
val_seqs   = [make_equation(a, b, bits=bits) for a, b in val_pairs] * 50

train_tokens = [tok for seq in train_seqs for tok in seq]
val_tokens   = [tok for seq in val_seqs for tok in seq]

train_arr = np.array(train_tokens, dtype=np.uint8)
val_arr   = np.array(val_tokens, dtype=np.uint8)

# 3. Save Binary Files and Meta Dictionary
out_dir = os.path.dirname(__file__)
train_arr.tofile(os.path.join(out_dir, 'train.bin'))
val_arr.tofile(os.path.join(out_dir, 'val.bin'))

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Dataset successfully created in {out_dir}:")
print(f" - Train tokens: {len(train_arr):,} (saved to train.bin)")
print(f" - Val tokens:   {len(val_arr):,} (saved to val.bin)")
print(f" - Vocab size:   {vocab_size} (saved to meta.pkl)")
