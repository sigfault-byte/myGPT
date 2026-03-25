import torch


def generate_sample(
    model,
    tokenizer,
    prompt: str = "La liberté",
    max_new_tokens: int = 128,
) -> str:
    device = next(model.parameters()).device

    context = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=device,
    ).view(1, -1)

    # sliding window for generation is handle in the Model.generate method when max_token > T
    generated = model.generate(context, max_new_tokens=max_new_tokens)
    out = tokenizer.decode(generated[0].tolist())

    return out
