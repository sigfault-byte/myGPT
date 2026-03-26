# Run 20260324-165748_bs64_blk256_emb128_h4_l4
> Same params as before only doubled the steps.
> Did not start from a previous checkpoint. 

> seed `1337`
- [run.json](run.json)
- [metrics.csv](metrics.csv)
- [loss plot](loss.png)

## Loss at 10k iter
  - train: `1.836662`
  - val: `2.883449`

> At 5k steps, loss was almost identical to the previous run.
> Over the last ~3000 steps, validation loss remained mostly flat / oscillating and slightly increased:
> `2.870250 → 2.872107 → 2.882326 → 2.883449`

## Sample at 10k iter — prompt: *La liberté*

```text
 La liberté & du dernier passage de la maître. Ceci ne nous apprend que par la forme, sur de confédérer les mains. Or, elle fasse de nouveau tout ce qui est arbitraire, mais elle et constante de l’autorité de la raison, qu’il soit contre ce qu’il y ait fouter à force de la cause des guerres civiles & à maintenir la nature. Ce n’est pas courir; mais je vous réponds que, je n’a point en disoit point, en sorte que si différent, & que toutes les forces qu’un Etat se développe.
```


# Note

Quite the opposite of what I was hoping to see.

  - Doubling training steps did not improve validation loss or sample quality.
  - This suggests the model is likely capacity-limited rather than undertrained.
  - The model continues to reproduce surface-level prose structure (punctuation, clause rhythm, stylistic markers).
  - Semantic coherence and lexical correctness remain weak.
  - Some local patterns -> pronoun usage such as **elle** appear occasionally correct, but are not consistently reliable. No Idea if this is pattern recognition or grammatical intuition.
  - Confusion between tokens such as “et” and “est” persists, indicating weak grammatical constraint enforcement.

Also, curious to how the model determine when to chose *&* and *et*.

---

Increasing training alone did not improve validation or sample quality, rather almost the opposite,  indicating that the model is likely capacity limited rather than undertrained.
