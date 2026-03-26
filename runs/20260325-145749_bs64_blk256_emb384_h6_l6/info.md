# Run 20260325-145749_bs64_blk256_emb384_h6_l6
> Same config as the previous run, but extended to `5k` steps.

> seed `1337`
- [run.json](run.json)
- [metrics.csv](metrics.csv)
- [loss plot](loss.png)

## Loss at 5k iter (vocab_size: 512)
- train: `1.491304`
- val: `2.747775`

> Validation degradation begins around `3k`.
> Previous run at `3k`: train `1.918170`, val `2.651846`.
> The model continued learning the training set, but validation stagnated and then slightly worsened.

## Sample at 5k iter — prompt: *La liberté*
```text
La liberté ne peut subsister sans loi qu’il peut résoudre et pour laquelle il ne doit pas excellent avoir ouet son propre avantage; au lieu que les combats mêmes se gardent en accordant avec les deniers du monde le nom de
capitae
, joindre à cela que plus, plus, plus ils deviennent uniquement de traiterie, plus également contre les délibérations des mêmes employées, plus proportionnés au Corps politique, plus ils n’en sont quinsensibles, plusieurs auteurs extrêmes, mais moins le Gouvernement se détruit; ils sont remplis de même dans le lieu dans un Etat, un Etat.
```

## Notes

  - The model reproduces surface prose structure very well here: punctuation, clause rhythm, and even the Rousseau-like typographic gesture of **isolating an important term**.
  - Looking at the text without seeing the words looks like a very legitimate sample from a book.

However, lexical and semantic control remain weak.
The sample contains:
  - hallucinated or malformed words (*capitae*, *ouet*, *quinsensibles*)
  - repetition (plus, plus, plus)
  - plausible-looking syntax that does not remain semantically stable
  - Grammar, agreement, and conjugation appear somewhat improved locally, but are still unreliable overall.

---

Extending training from 3k to 5k improved training loss but not validation or generation quality...
This suggests that the earlier 3k run was already near the useful limit for this configuration.

The lexical errors and invented forms support the hypothesis that vocab_size=512 may be too restrictive for this corpus, though this run alone does not prove anything...
