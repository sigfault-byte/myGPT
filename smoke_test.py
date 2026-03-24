from my_tokenizers.bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer("rousseau_bpe.json")

text = "La liberté politique dépend de la volonté générale."
ids = tokenizer.encode(text)

print(ids)
print(tokenizer.decode(ids))
print(tokenizer.vocab_size)
