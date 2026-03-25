from tokenizers import Tokenizer


class BPETokenizer:
    def __init__(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        self.name = "bpe"
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    # def decode(self, ids: list[int]) -> str:
    #     return self.tokenizer.decode(ids)

    def decode(self, ids: list[int]) -> str:
        toks = [self.tokenizer.id_to_token(i) for i in ids]
        text = "".join(toks)

        # clean spacing artifacts
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" ;", ";")
        text = text.replace(" :", ":")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" '", "'")

        return text
