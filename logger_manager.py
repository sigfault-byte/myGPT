import json
import time
from datetime import datetime
from pathlib import Path


class LoggerManager:
    def __init__(self, config: dict, base_dir="runs"):
        self.config = config

        # create run_id
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = (
            f"{timestamp}_bs{config['batch_size']}"
            f"_blk{config['block_size']}"
            f"_emb{config['n_embd']}"
            f"_h{config['n_head']}"
            f"_l{config['n_layer']}"
        )

        # create directory
        self.run_dir = Path(base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # paths
        self.metrics_path = self.run_dir / "metrics.csv"
        self.config_path = self.run_dir / "run.json"
        self.info_path = self.run_dir / "info.md"

        # write config
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        # init metrics file
        with open(self.metrics_path, "w") as f:
            f.write("step,train_loss,val_loss,elapsed_sec\n")

        # init info.md
        with open(self.info_path, "w") as f:
            f.write(f"# Run {self.run_id}\n\n## Notes\n\n")

        self.start_time = time.time()

    def log_metrics(self, step, train_loss, val_loss):
        # Validate types early
        assert isinstance(step, int), f"step must be int, got {type(step)}"

        train_loss = self._to_float(train_loss, "train_loss")
        val_loss = self._to_float(val_loss, "val_loss")

        elapsed = time.time() - self.start_time

        print(
            f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f}s"
        )

        with open(self.metrics_path, "a") as f:
            f.write(f"{step},{train_loss:.6f},{val_loss:.6f},{elapsed:.2f}\n")

    def save_sample(self, step, text):
        samples_dir = self.run_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        path = samples_dir / f"sample_step{step:04d}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def append_info(self, text):
        with open(self.info_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    @staticmethod
    def _to_float(x, name="value"):
        if hasattr(x, "item"):  # torch tensor
            return float(x.item())
        try:
            return float(x)
        except Exception:
            raise TypeError(f"{name} must be numeric, got {type(x)}: {x}")
