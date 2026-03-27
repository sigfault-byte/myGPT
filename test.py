from my_tokenizers.bpe_tokenizer import BPETokenizer

tok = BPETokenizer("rousseau_bpe_metaspace_1024.json")

text = "La liberté est onéreuse"
(
    ids,
    token,
) = tok.encode_with_tokens(text)

print(token)
print(ids)
print(tok.decode(ids))
