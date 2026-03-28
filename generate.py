import json
from pathlib import Path

import torch

from config import get_device
from generate_sample import generate_sample
from models.myGPT import GPTLanguageModel
from my_tokenizers.bigram_tokenizer import CharLevelTokenizer
from my_tokenizers.bpe_tokenizer import BPETokenizer

RUN_DIR = Path("runs/20260328-124239_bs32_blk256_emb768_h12_l6")
CHECKPOINT_PATH = RUN_DIR / "checkpoint_last.pt"
CONFIG_PATH = RUN_DIR / "run.json"

PROMPT = "La liberté de l’enfans est"
MAX_NEW_TOKENS = 500


def load_run_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(tokenizer_config: dict):
    tokenizer_name = tokenizer_config["name"]

    if tokenizer_name == "char":
        return CharLevelTokenizer(
            stoi=tokenizer_config["stoi"],
            itos=tokenizer_config["itos"],
        )

    if tokenizer_name == "bpe":
        return BPETokenizer(tokenizer_config["tokenizer_path"])

    raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


def build_model(run_config: dict) -> GPTLanguageModel:
    return GPTLanguageModel(
        vocab_size=run_config["vocab_size"],
        block_size=run_config["block_size"],
        n_embd=run_config["n_embd"],
        n_head=run_config["n_head"],
        n_layer=run_config["n_layer"],
        dropout=run_config["dropout"],
    )


def warn_if_suspicious(device, run_config: dict):
    params_millions = run_config.get("params_millions")

    if device.type == "cpu" and params_millions is not None and params_millions > 2:
        print(
            f"[Warning] Generating on CPU with a model of "
            f"{params_millions:.2f}M parameters."
        )

        try:
            answer = input("Continue anyway? [y/N]: ").strip().lower()
        except EOFError:
            # !non-interactive environment (e.g. pipe, script)
            print("[Abort] No input available. Abort.")
            raise SystemExit(1)

        if answer not in ("y", "yes"):
            print("[Abort] Generation cancelled.")
            raise SystemExit(1)


def main():
    device = get_device()
    run_config = load_run_config(CONFIG_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    warn_if_suspicious(device, run_config)

    tokenizer = load_tokenizer(checkpoint["tokenizer"])

    model = build_model(run_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = generate_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print(sample)


if __name__ == "__main__":
    main()
