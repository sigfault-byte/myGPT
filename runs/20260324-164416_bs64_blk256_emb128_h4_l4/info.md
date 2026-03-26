# Run 20260324-164416_bs64_blk256_emb128_h4_l4
> Previous comparable run with vocab `512` is missing.
> Smaller BPE model intended to reduce the rapid overfitting seen in the earlier larger run.

> seed `1337`
- [run.json](run.json)
- [metrics.csv](metrics.csv)
- [loss plot](loss.png)

## Loss at 5k iter
> max entropy for a uniform distribution over `512` tokens: `log2(512) = 9`

- train: `2.145592`
- val: `2.832775`

> First run where there was not too much divergence by the end.
> Validation loss kept decreasing throughout training, though the slope was clearly flattening near 5k.
> Model capacity was reduced from roughly `10M` parameters to about `0.9M`.

## Sample at 5k iter — prompt: *La liberté*

```text
La liberté, il dispose de lui-même.
Mon croir, je dis que j’appelle entonne qu’un peuple qui préféré comme contraire, dans un peuple un des positions, d’une sorte demeure déine ingénible, d’une autre propre respecte dans cette matière quatre mécontente, & suffite bien tout.
```
# Note

This run appears much more stable than the earlier larger BPE configuration.

  -	The sample still shows some structural and rhetorical shape from the corpus.
  -	Lexical choice and semantic coherence remain very weak, and large parts of the text are still effectively garbage.
  -	Grammar, conjugation, and agreement are not completely collapsing, but they are far from reliable.
  -	This looks less like severe overfitting and more like an underpowered or still-undertrained model.

The next experiment was to extend training and see whether this smaller model could keep improving without the sharp validation degradation seen in the larger run.
