# nanoGPT-m for undercomplete synthetic HMM benchmark (d=64, m=128)
lamda = 0.9

out_dir = 'm128_d64_4l4h'
eval_interval = 50
eval_iters = 20
log_interval = 50
always_save_checkpoint = False

wandb_log = False
wandb_project = 'hmm-benchmark'
wandb_run_name = 'nanogpt-m-m128'

data_file_path = '../prepare_data/data_d64_m128_lamda0.9_example.h5'
gradient_accumulation_steps = 1
batch_size = 10
block_size = 256

# nanoGPT-m architecture (4 layers, 4 heads)
n_layer = 4
n_head = 4
n_embd = 64
dropout = 0.0

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = learning_rate / 10
beta2 = 0.99
warmup_iters = 0

device = 'cpu'
compile = False
