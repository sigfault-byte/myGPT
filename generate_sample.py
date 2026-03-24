import torch


def generate_sample(
    model,
    tokenizer,
    prompt: str = "La liberté",
    max_new_tokens: int = 128,
    device: str = "cpu",
) -> str:
    was_training = model.training
    model.eval()

    context = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=device,
    ).view(1, -1)

    with torch.no_grad():
        out = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()

    if was_training:
        model.train()

    return tokenizer.decode(out)
