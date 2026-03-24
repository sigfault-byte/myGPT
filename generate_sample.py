import torch


def generate_sample(
    model,
    tokenizer,
    prompt: str = "La liberté",
    max_new_tokens: int = 128,
    block_size: int = 128,
) -> str:
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device

    context = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=device,
    ).view(1, -1)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = context[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)

    out = tokenizer.decode(context[0].tolist())

    if was_training:
        model.train()

    return out
