# Run 20260324-133222_bs64_blk256_emb384_h6_l6
> seed `1337`
> [run.json](run.json))
- [metrics.csv](metrics.csv)
- [loss plot](loss.png)

## Loss at 5k iter
- train: `0.560688`
- val: `1.302736`

> Validation degradation appears around step ~ 2500.  
> Model had `10.814306M` parameters.  
> Processing `64 * 256 = 16,384` tokens per iter.  
> Training split was `0.9 * 815,846 = 734,261` characters.  
> A 5k-step run processed `81,920,000 / 734,261 ~= 111` corpus length equivalents.


# Sample at 5k iter - prompt: *La liberté*:
```text
La liberté de ce que de l’Etat en qualité d’un mal.
Ces différens mots est très conditionnée, il n’y avoit point d’homme à gens d’un peuple, d’autre moyen de sa supporter une multitude de partie qui lui est simple, il n’y en a plus autre; mais puissant surprenant ce
```


## Notes

  -	The model clearly captures a large part of the prose texture of the corpus.
	-	The longer 5000-character sample shows that paragraph flow and rhetorical cadence are often coherent at a surface level.
	
	However grammatical control remains weak:
	-	agreement errors: *mots **est** très*
	-	pronoun / reflexive confusion: *moyen de **sa** supporter*
	
	The model appears much better at imitating style and vocabulary distribution than at maintaining correct grammar, conjugation, and syntactic constraint satisfaction.
	
	> In other words, it has learned to *sound plausibly Rousseau-like* before it has learned to remain consistently correct.
	
	The current model remains far from the target:
  generating text that combines the corpus’ prose, vocabulary, and rhetorical structure with reliably correct grammar and conjugation, especially around rarer or more constrained word sequences...

---

This was probably the sixth run on char level tokenizer. 
Increasing the T gradually, until going on a another machine with a 4080 to increase to 256 T.

Persuaded that I cannot even start to comprehand the *geometric* gymnastic the model has to do to keep track of every token for letter, to group of token to possible word to group of group of token for a sentence and so on and so forth, I decided to change the tokenizer to a BPE.
