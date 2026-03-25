import argparse
import os

import torch

from config import DEFAULT_PROFILE, get_config, get_device
from generate_loss_plot import generate_loss_plot
from generate_sample import generate_sample
from logger_manager import LoggerManager
from models.myGPT import GPTLanguageModel

# tokeninzer
from my_tokenizers.bigram_tokenizer import CharLevelTokenizer
from my_tokenizers.bpe_tokenizer import BPETokenizer

# tokenizer var
parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", required=True, choices=["char", "bpe"])
parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=["small", "big"])
args = parser.parse_args()

# config var
run_config = get_config(args.profile)
device = get_device()
batch_size = run_config["batch_size"]
block_size = run_config["block_size"]
n_embd = run_config["n_embd"]
n_head = run_config["n_head"]
n_layer = run_config["n_layer"]
dropout = run_config["dropout"]
learning_rate = run_config["learning_rate"]
eval_interval = run_config["eval_interval"]
eval_iters = run_config["eval_iters"]
max_iters = run_config["max_iters"]

TOKENIZER_NAME = args.tokenizer

# ... just in case
if TOKENIZER_NAME is None:
    raise ValueError("Choose a tokenizer: 'char' or 'bpe'")

# file input
file = "data/rousseau_pol_and_emile_vol1-4-5.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()


# Char tokenizer
def build_tokenizer(tokenizer_name: str, text: str):
    if tokenizer_name == "char":
        tokenizer = CharLevelTokenizer.from_text(text)
        tokenizer_config = {
            "name": "char",
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        }
        return tokenizer, tokenizer_config
    # BPT tokenizer
    if tokenizer_name == "bpe":
        tokenizer_path = "rousseau_bpe2048_vol1-4-5.json"
        tokenizer = BPETokenizer(tokenizer_path)
        tokenizer_config = {
            "name": "bpe",
            "tokenizer_path": tokenizer_path,
        }
        return tokenizer, tokenizer_config

    raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


# Build tokenizer
tokenizer, tokenizer_config = build_tokenizer(TOKENIZER_NAME, text)

torch.manual_seed(1337)

# Variable for specific data
text_size = len(text)

# Train and test splits
tokens = tokenizer.encode(text)
vocab_size = tokenizer.vocab_size
data = torch.tensor(tokens, dtype=torch.long)

n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    # selec a random window across all the corpus
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    # send to cpu / gpu
    x, y = x.to(device), y.to(device)
    return x, y


# prevent some memory optimization keeping this in memory for gradient descent
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)

            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# model and apssing all arguments
model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
).to(device)

# print the number of parameters from the model
params = sum(p.numel() for p in model.parameters()) / 1e6
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

config = {
    "batch_size": batch_size,
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "params_millions": params,
    "learning_rate": learning_rate,
    "dataset": {
        "path": file,
        "chars": text_size,
        "millions": text_size / 1e6,
    },
    "tokenizer": tokenizer_config,
    "vocab_size": vocab_size,
}
logger = LoggerManager(config)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        # losses = estimate_loss()
        losses = estimate_loss()
        train_loss = losses["train"]
        val_loss = losses["val"]

        logger.log_metrics(iter, train_loss, val_loss)

        sample = generate_sample(
            model=model,
            tokenizer=tokenizer,
            prompt="La liberté",
            max_new_tokens=block_size,
        )
        logger.save_sample(iter, sample)

    # sample a batch of data
    xb, yb = get_batch("train")

    # bit of Paranoia
    assert xb.device == next(model.parameters()).device
    assert yb.device == next(model.parameters()).device

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

os.makedirs("checkpoints", exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step": iter,
    "text_trained": file,
    "vocab_size": tokenizer.vocab_size,
    "tokenizer": tokenizer_config,
}

# Save checkpoint in the run directory
checkpoint_path = logger.run_dir / "checkpoint_last.pt"
torch.save(checkpoint, checkpoint_path)
# Nerd print
print(f"Checkpoint saved to {checkpoint_path}")
# plot loss and save in run dir
generate_loss_plot(logger.run_dir)
