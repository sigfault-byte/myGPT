from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

from tokenizers import Tokenizer

files = ["data/Test-bad-vol1-2-3-4-5-6-8-10-12-14-17.txt"]

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")

trainer = BpeTrainer(
    vocab_size=1024,
    min_frequency=3,
    special_tokens=["[UNK]"],
)

tokenizer.train(files, trainer)
tokenizer.save("rousseau_test_bpe_metaspace_1024.json")

print("Tokenizer trained and saved!")
print("Vocab size:", tokenizer.get_vocab_size())
