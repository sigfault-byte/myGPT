class CharTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @property
    def name(self):
        return "char"

    def get_vocab(self):
        return {
            "stoi": self.stoi,
            "itos": self.itos,
        }

    @classmethod
    def from_text(cls, text):
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, tokens):
        return "".join([self.itos[i] for i in tokens])
