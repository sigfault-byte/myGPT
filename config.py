import torch

"""
Config file
"""

# config.py

CONFIGS = {
    "small": {
        "batch_size": 16,
        "block_size": 128,
        "n_embd": 128,
        "n_head": 4,
        "n_layer": 4,
        "dropout": 0.2,
        "learning_rate": 3e-4,
        "eval_interval": 100,
        "eval_iters": 20,
        "max_iters": 1000,
    },
    "big": {
        "batch_size": 16,
        "block_size": 128,
        "n_embd": 256,
        "n_head": 4,
        "n_layer": 2,
        "dropout": 0.25,
        "learning_rate": 1e-4,
        "eval_interval": 100,
        "eval_iters": 20,
        "max_iters": 5000,
    },
}

DEFAULT_PROFILE = "small"


def get_config(profile: str):
    if profile not in CONFIGS:
        raise ValueError(f"Unknown config profile: {profile}")
    return CONFIGS[profile].copy()


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")

    return device
