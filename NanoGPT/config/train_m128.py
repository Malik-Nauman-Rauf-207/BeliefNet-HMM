# nanoGPT-s for undercomplete synthetic HMM benchmark (d=64, m=128)
lamda = 0.9

out_dir = 'm128_d64_1l1h'
eval_interval = 50
eval_iters = 20
log_interval = 50
always_save_checkpoint = False

wandb_log = False
wandb_project = 'hmm-benchmark'
wandb_run_name = 'nanogpt-s-m128'

data_file_path = '../prepare_data/data_d64_m128_lamda0.9_example.h5'
gradient_accumulation_steps = 1
batch_size = 10
block_size = 256

# nanoGPT-s architecture (Table 1: 73,920 parameters)
n_layer = 1
n_head = 1
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
