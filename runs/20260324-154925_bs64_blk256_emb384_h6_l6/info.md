# Run 20260324-154925_bs64_blk256_emb384_h6_l6
> First bpe tokenizer run

> seed `1337`
> [run.json](run.json)
- [metrics.csv](metrics.csv)
- [loss plot](loss.png)

## Loss at 5k iter
> max entropy log2(1024) = 10

- train: `0.091359`
- val: `6.629541`

> Validation degradation appears around step ~800.
> Because BPE tokens often represent more than one character, direct comparison with char-level loss is not straightforward.
> The model likely processed well over 100 corpus length equivalents of the training data.



# Sample at 5k iter - prompt: *La liberté*:
```text
La liberté, sans lesquelles soient eux-mêmes ne jouissent des biens, surtout en vie même, & le défaut de ceux qui en sont les plus dangereuses, périlleux pour pouvoir être: en sorte que, si elles le Souverain ne s’exerçoivent qu’à sa faveur de cette maxime de politique, que la force publique, & qu’il tient de son propre intérêt à la leur, il fut plus nécessaire des maximes de Corps sanction à l’Etat, on trouveroit que pour que l’établissement des loix. Alors le Corps social se tire le Souverain n’est pas un contrat entre les sujets d’un côté & envers le peuple d’agir de l’autre.
```

## Notes

  The sample quality is clearly stronger than in the comparable char-level run.
  - Clause structure, rhetorical flow, and overall prose texture are substantially better.
  -	Local morphology and conjugation also appear improved in several places.
  
  -	However, grammatical control is still unreliable, especially across longer dependencies and agreement chains.
  
  -	The gap between train and validation loss suggests very early overfitting or a rapid breakdown in generalization.
  > But the model does produce a somehow *good* output. 
  
BPE appears to be a better direction for this corpus, likely because it provides more meaningful units than raw characters and increases the effective semantic span of the context window.
Of course it also "ease" the *geometric* gymnastic the model has to do to keep track of the context. 

A better subword vocabulary or tokenizer design may improve this further.
