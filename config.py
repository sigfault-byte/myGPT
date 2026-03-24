import torch

"""
Config file
"""

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# training
batch_size = 64  # B
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

# model
block_size = 256  # T
n_embd = 384  # C
n_head = 6
n_layer = 6
dropout = 0.2
