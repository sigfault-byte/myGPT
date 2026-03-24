import csv
from pathlib import Path

import matplotlib.pyplot as plt


def generate_loss_plot(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.csv"
    output_path = run_dir / "loss.png"

    steps: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    if not steps:
        raise ValueError(f"No metric rows found in {metrics_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, label="train loss")
    plt.plot(steps, val_losses, label="val loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved plot to {output_path}")
