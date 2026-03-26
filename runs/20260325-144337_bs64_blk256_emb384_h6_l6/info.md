# Run 20260325-144337_bs64_blk256_emb384_h6_l6
> First BPE run on the larger corpus.
> ~10M parameters, limited to `3k` steps smoke test run.

> seed `1337`
- [run.json](run.json)
- [metrics.csv](metrics.csv)
- [loss plot](loss.png)

## Loss at 3k iter (vocab_size: 512)
- train: `1.918170`
- val: `2.651846`

> Model is clearly learning.
> Validation loss remains broadly stable and continues to decrease overall, though with fluctuations.
> Last 500 steps:
> `2.640007 -> 2.651107 -> 2.620239 -> 2.636732 -> 2.651846`

## Sample at 3k iter — prompt: *La liberté*

```text
La liberté, j’appartient l’hospitorité; &, jusqu’à quel point j’étudié dans l’ordre qu’il n’y a point de véritable éclaircissement qui ne peut plus extravagance qui auront ni suivit.
Il est aisé de accoutumer un instant à sa dernière matière, & d’amuser ici les premières sensations de l’homme, sans attaquer l’un de la femme aux Gouvernemens de toute espèce, à se compter d’un autre animal ces membres qui sont à la gêne des révolutions par la sagesse, que la nature ne la dépend point.
```

## Notes

  - The model reproduces prose rhythm, punctuation, and surface rhetorical structure fairly well.
  - Grammar, conjugation, and lexical precision remain weak.

This run looked promising early because training was clearly progressing without the sharp degradation seen in some earlier experiments.
However, the tokenizer likely remains a limiting fact.

  
---

Knowing the outcome of later runs, and digging more into the *tonenization* and how it affects the model, this looks like a case where BPE was viable on the larger corpus, but vocab_size=512 was probably too coarse for the dataset.

A larger BPE vocabulary is worth testing:
  - 512 may merge too many heterogeneous fragments
  - 1024 will not seem sufficient to clearly unlock the hoped-for gains
A **substantially** larger vocabulary may better preserve useful lexical and morphological structure

The same training configuration may still be worth keeping, but combined with:
	•	a larger tokenizer vocabulary
	•	and a learning-rate schedule such as cosine decay to stabilize later training if the pattenr matches the usual *nice shape, and suddent divergence*
