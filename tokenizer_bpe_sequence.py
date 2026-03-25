from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.trainers import BpeTrainer

from tokenizers import Tokenizer

# Path to corpus
files = ["data/rousseau_pol_and_emile_vol1-4-5.txt"]

# Create tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])

# Trainer
trainer = BpeTrainer(
    vocab_size=1024,
    min_frequency=3,
    special_tokens=["[UNK]"],
)

# Train
tokenizer.train(files, trainer)

# Save
tokenizer.save("rousseau_bpe_3_vol1-4-5.json")

print("Tokenizer trained and saved!")
print("Vocab size:", tokenizer.get_vocab_size())
