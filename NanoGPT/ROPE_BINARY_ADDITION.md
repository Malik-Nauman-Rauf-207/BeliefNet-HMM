# RoPE pretraining for binary addition

This adds rotary position embeddings (RoPE) to the NanoGPT binary-addition
pretraining model while keeping the existing absolute-position run intact.
Both variants use the same data, architecture, training settings, and seed as
the train_binary_addition.py configuration. They train from scratch and write
to separate output directories.

The rotary operation is applied to each attention head's query and key vectors
before the attention dot product. For position \(p\) and frequency \(\omega_i\),
the angle is \(p\omega_i\), with the standard rotation applied to each adjacent
even/odd coordinate pair.

## Variants

- Fixed-frequency RoPE uses the standard inverse-frequency schedule
  \(\omega_i=10000^{-2i/d_h}\).
- Learnable-frequency RoPE initializes the same schedule, then learns positive
  inverse frequencies independently for each attention head in each layer.
  Frequencies are parameterized in log space and excluded from weight decay.

Both versions retain the four-layer, four-head, 128-dimensional architecture.
The RoPE checkpoints contain no trained absolute position table. A temporary
table is initialized and discarded so shared model weights retain the same
seed-based initialization as the absolute-position baseline.

## Run commands

From this NanoGPT directory, run either or both:

    python3 train.py config/train_binary_addition_rope_fixed.py
    python3 train.py config/train_binary_addition_rope_learnable.py

Each command runs the full 4,000-iteration pretraining configuration on MPS
when available. To run one variant only, run just its command. The fixed and
learnable runs save independently under:

- out_binary_addition_rope_fixed_embd128_mps/
- out_binary_addition_rope_learnable_embd128_mps/

Each directory contains its checkpoint and loss-trajectory HDF5 file. These
runs start from scratch because the existing absolute-position checkpoint has
an incompatible parameterization.
