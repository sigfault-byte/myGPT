import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape

        # hs => headsize !
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)

        # attention scores: (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # Mask future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # normalize attention weights
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted aggregation of values
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, n_embd: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        # important ! head_size = int -> must be % must be 0
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [
                Head(
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """simple linear layer followed by a non-linearity."""

    # for each token:
    # process its vectors with aMulti Layer Positron,
    # works on single token, does not communicate with others !

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            # * 4 is straigh from karpathy explaining self attention
            # expand the dimension 4 times
            nn.Linear(n_embd, 4 * n_embd),
            # breaking linearity
            nn.ReLU(),
            # compress back to the required dimensions
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=n_head,
            block_size=block_size,
            dropout=dropout,
        )
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        # first pass of layer norm before projection on lm head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video,
        # but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape

        if T > self.block_size:
            # !!
            raise ValueError(
                f"Input sequence length {T} exceeds block_size {self.block_size}"
            )

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)  # (T, C)

        # raw representation (not normalized !)
        x = tok_emb + pos_emb  # (B, T, C)

        # repeated refinement:
        # each block:
        #   - normalizes input locally
        #   - computes update
        #   - adds it back
        #   - x is not normalized at all yet
        x = self.blocks(x)  # (B, T, C)

        # first time the **WHOLE** representation is normalized
        x = self.ln_f(x)  # (B, T, C)

        # projection to vocab
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
            return logits, loss

        B, T, C = logits.shape
        # using reshape instead of view
        logits_flat = logits.reshape(B * T, C)
        targets_flat = targets.reshape(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            # model was trained with sequences of length `block_size`.
            # during generation the context keeps growing, so only feed
            # the last `block_size` tokens (sliding window !)
            # stop crashing when tweaking max token generation exeeding T
            idx_cond = idx[:, -self.block_size :]

            logits, _ = self(idx_cond)  # no need for loss
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()

        return idx
