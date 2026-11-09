# NanoGPT Configuration for Multi-Scale Binary Addition (2, 4, 8, 16 bits)

out_dir = 'out_binary_addition_multiscale_embd128_mps'
eval_interval = 250
eval_iters = 50
log_interval = 25

# Multi-scale dataset path
data_file_path = '../prepare_data/data_binary_addition_multiscale.h5'

always_save_checkpoint = False

wandb_log = False
wandb_project = 'binary-addition'
wandb_run_name = 'nanogpt-multiscale'

batch_size = 64
block_size = 64      # Context length to accommodate 16-bit addition sequences

# Architecture
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# Optimization hyperparameters
learning_rate = 1e-3
max_iters = 4000     # Increased to account for higher complexity
lr_decay_iters = 4000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 150

device = 'mps'
dtype = 'float32'       # use full precision for the MPS run
compile = False
