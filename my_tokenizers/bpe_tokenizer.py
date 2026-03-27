from tokenizers import Tokenizer


class BPETokenizer:
    def __init__(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        self.name = "bpe"
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def tokens(self, text: str) -> list[str]:
        return self.tokenizer.encode(text).tokens

    def encode_with_tokens(self, text: str) -> tuple[list[int], list[str]]:
        enc = self.tokenizer.encode(text)
        return enc.ids, enc.tokens
