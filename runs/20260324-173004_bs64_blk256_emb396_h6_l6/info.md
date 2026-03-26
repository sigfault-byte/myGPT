# Run 20260324-173004_bs64_blk256_emb396_h6_l6
> I was luck of typo 396 / 6 = 66
> Char-level run after disappointment with BPE.
> Largely a random-params experiment.
> Ended again around `10M` parameters, which is likely too large for this corpus.

> seed `1337`
- [run.json](run.json)
- [metrics.csv](metrics.csv)
- [loss plot](loss.png)

## Loss at 10k iter
- train: `0.144461`
- val: `1.909731`

> Char-level loss continues to *look* healthier than BPE loss.
> > But loss curves are only comparable within the same tokenization regime. 
> The first training steps show a very steep improvement.

## Loss at 2.5k iter
- train: `0.882604`
- val: `1.168010`

> Despite the relatively good metrics, generated text remains weak semantically.
> Local syntax and grammatical shape are noticeably stronger than in earlier runs.
> Validation loss starts degrading from around this point.

## Sample at 10k iter — prompt: *La liberté*
```text
La liberté sont celle des particuliers, comme homme pourroit être libre, sont toujours plus propres à décrire, mais pourquoi plus ceux qui sont accoutumés à ne regarder que les autres peuples.
```


## Sample at 2.5K iter — prompt: *La liberté*:

```text
La liberté naturelle, quelqu’un senti à la portée & le feu exclut de trop, dans leur fantage, dont on dit conventionner à ce désobéir après que ce porte autorité conserve notre maxime dont il s’en charge; en mutuelle il soit contraire d’borner l’Etat, & que l’on ne 
```
 
## Notes

  - The model reproduces local French-looking syntax and morphology *much better* than many earlier runs.
  - However, semantic coherence remains weak.
  - Even at 10k, grammatical correctness is not reliable (*La liberté **sont**...*).
  - The 2.5k sample is less controlled, but the 10k sample is not clearly more meaningful despite lower trai loss. 
  - Maybe char-level modeling is improving local fluency without being able to solve deeper semantic and agreement problems.
  - 5000 char sample from the end of the run demonstrates the same pattern.
  - Char-level modeling likely remains limited by context efficiency: the model spends context budget on characters rather than larger linguistic units.
  - Increasing context length may still be worth testing, though the attention cost will rise way too quickly.
