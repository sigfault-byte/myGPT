# Run 20260325-130314_bs64_blk512_emb384_h6_l6
> New corpus: ~800k → ~ 2.34M chars
> Char-level, ~10M params


> seed `1337`
- [run.json](run.json)
- [metrics.csv](metrics.csv)
- [loss plot](loss.png)

## Loss at 10k iter (vocab_size: 101)
- train: `0.841352`
- val: `1.092314`

> Last 500 steps:
> `1.091452 -> 1.081430 -> 1.091459 -> 1.090804 -> 1.093435`

> Training appears stable with no strong divergence.
> Char-level loss still tends to look "healthy", but ... !

> Tokens per iteration: `32,768`
> Total processed: `163,840,000`
> ≈ `69` corpus-length equivalents

## Sample at 5k iter — prompt: *La liberté*

```text
La liberté seule fait une politesse sans inconvénient. Quelle seconde est la femme naturelle de ces hommes? Est-ce dans les grandes femmes que l’éducation publique convienne? Non seulement il n’y en a point des animaux où il suit dans la ville primitive, mais comme les voyageurs ne restent plus lieu qu’eux.
```

## Notes

  - Increasing dataset size significantly improved both **sample quality** and especially **training stability**.
  - The model reproduces French-like syntax and morphology more consistently than previous runs.
  - Errors are less frequent and less catastrophic, though semantic coherence remains garbage.
  - Some sentences remain syntactically fragile or semantically unclear.
  - The larger dataset appears to reduce overfitting pressure compared to earlier experiments.
  - 5000 char sample confirms that errors still occur (malformed words and agreement issues), though less frequently.
