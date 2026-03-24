# import torch

# from config import device
# from models.myGPT import GPTLanguageModel
# from my_tokenizers.bigram_tokenizer import CharTokenizer

# checkpoint_path = "runs/20260324-133222_bs64_blk256_emb384_h6_l6/checkpoint_last.pt"

# checkpoint = torch.load(checkpoint_path, map_location=device)

# tokenizer = CharTokenizer(
#     stoi=checkpoint["stoi"],
#     itos=checkpoint["itos"],
# )

# model = GPTLanguageModel(checkpoint["vocab_size"]).to(device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# prompt = "la religion est"
# context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).view(
#     1, -1
# )

# with torch.no_grad():
#     out = model.generate(context, max_new_tokens=500)[0].tolist()

# print(tokenizer.decode(out))

import torch

from models.myGPT import GPTLanguageModel
from my_tokenizers.bpe_tokenizer import BPETokenizer

checkpoint_path = "runs/your_bpe_run/checkpoint_last.pt"

checkpoint = torch.load(checkpoint_path, map_location="cuda")

tokenizer = BPETokenizer(checkpoint["rousseau_bpe.json"])

model = GPTLanguageModel(checkpoint["vocab_size"]).to("cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

device = next(model.parameters()).device

prompt = "la religion est"

context = torch.tensor(
    tokenizer.encode(prompt),
    dtype=torch.long,
    device=device,
).view(1, -1)

with torch.no_grad():
    out = model.generate(context, max_new_tokens=500)[0].tolist()

print(tokenizer.decode(out))
