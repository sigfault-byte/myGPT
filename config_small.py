import torch

"""
Config file small computer
"""

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# training
batch_size = 16
max_iters = 200
eval_interval = 50
learning_rate = 1e-3
eval_iters = 20

# model
block_size = 64
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.1
