import os

import torch

from config import (
    batch_size,
    block_size,
    device,
    eval_interval,
    eval_iters,
    learning_rate,
    max_iters,
    n_embd,
    n_head,
    n_layer,
)
from generate_loss_plot import generate_loss_plot
from generate_sample import generate_sample
from logger_manager import LoggerManager
from models.myGPT import GPTLanguageModel
from tokenizers.bigram_tokenizer import CharTokenizer

torch.manual_seed(1337)

# with open("data/shakespear-1M.txt", "r", encoding="utf-8") as f:
file = "data/rousseau_ouvrage_pol_Vol1.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()

text_size = len(text)
tokenizer = CharTokenizer.from_text(text)
vocab_size = tokenizer.vocab_size

# Train and test splits
tokens = tokenizer.encode(text)
data = torch.tensor(tokens, dtype=torch.long)

n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


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


model = GPTLanguageModel(vocab_size).to(device)
# print the number of parameters in the model
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
    "params_millions": params,
    "learning_rate": learning_rate,
    "dataset": {
        "path": file,
        "chars": text_size,
        "millions": text_size / 1e6,
    },
    "tokenizer": tokenizer.name,
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
            device=device,
        )
        logger.save_sample(iter, sample)
        # print(
        #     f"step {iter}: train loss {train_loss['train']:.4f}, val loss {val_loss['val']:.4f}"
        # )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
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
    "stoi": tokenizer.stoi,
    "itos": tokenizer.itos,
}


checkpoint_path = logger.run_dir / "checkpoint_last.pt"
torch.save(checkpoint, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")
generate_loss_plot(logger.run_dir)
