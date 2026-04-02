# Small-Scale Language Model Experiments (Rousseau-style corpus)

# Overview

This repository contains a series of quick experiments training small GPT-style language models from scratch on a limited corpus (~2MB → ~10MB) of 18th-century French texts (Rousseau and later contemporaries).

The goal is not to achieve production quality text generation, but to explore: 

	- the impact of dataset size
	- tokenizer choice (char vs BPE)
	- model capacity vs data scale
	- training dynamics (loss vs sample quality)

⸻
# Observations

Most runs observtion may be found in `runs/experimentID/info.md`
