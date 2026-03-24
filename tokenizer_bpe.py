from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from tokenizers import Tokenizer

# Path to corpus
files = ["data/rousseau_ouvrage_pol_Vol1.txt"]

# Create tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# byte-level handles accents, punctuation, etc.
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

# Trainer
trainer = BpeTrainer(
    vocab_size=512,
    min_frequency=2,
    special_tokens=["[UNK]"],
)

# Train
tokenizer.train(files, trainer)

# Save
tokenizer.save("rousseau_bpe.json")

print("Tokenizer trained and saved!")
print("Vocab size:", tokenizer.get_vocab_size())
