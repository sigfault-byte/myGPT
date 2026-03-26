# Run 20260325-154426_bs64_blk256_emb384_h6_l6
> Attempt to modify tokenizer split logic.
> Failed run due to incorrect encoding/decoding (loss of whitespace information).

> seed `1337`
- [run.json](run.json)
- [metrics.csv](metrics.csv)
- [loss plot](loss.png)

## Loss at 5k iter (vocab_size: 512)
- train: `1.510001`
- val: `3.113218`

> Lowest validation loss was around ~3k steps.

## Sample at 5k iter — prompt: *La liberté*

```text
La lib er té est on é re u se , la lé g is la tion s ’ é v an ou it ; & les G re c ques sont comme les ma î tres des C am pa g nes . Qu el homme non moins lib re que par don né e ? 
```
## Manually reconstructed:

```text
La liberté est onéreuse, la législation s’ évanouit; & les Grecques sont comme les maîtres des Campagnes. Quel homme non moins libre que pardonnée ? 
```

## Notes

Lesson learnt, there is an, almost fun albeit complicated, relationship with the **0x20** aka **whitespace** and tokens.

  - The tokenizer modification removed or failed to encode whitespace explicitly.
  - This demonstrates that tokenization must preserve all structural information needed for decoding.
  - The *La liberté est onéreuse* is stricking, because *onereuse* appears only 3 times in the whole corpus, and never next to *liberté*!
  - This suggests that the model is learning distributional relationships beyond exact memorization.
  - Splitting on spaces is not inherently incorrect, but removing whitespace as a token leads to loss of structural information.

The model appears qualitatively different, likely because tokens align more closely with words, but output is degraded due to decoding issues.

---
