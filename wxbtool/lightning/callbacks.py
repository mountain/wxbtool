import os
import lightning.pytorch as pl

from typing import Any, Dict
from wxbtool.util.plotter import plot


class UniversalLoggingCallback(pl.Callback):
    def _flush_newline(self, trainer: pl.Trainer) -> None:
        # Flush a newline to separate logs
        if hasattr(trainer, "is_global_zero") and trainer.is_global_zero:
            print(flush=True)

    def _flush_artifacts(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        artifacts: Dict[str, Dict[str, Any]] = getattr(pl_module, "artifacts", None)
        if not artifacts:
            return

        # Rank-0 only writes/logs artifacts
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            pl_module.artifacts = {}
            return

        logger = getattr(trainer, "logger", None)
        log_dir = getattr(logger, "log_dir", None) or os.getcwd()
        out_dir = os.path.join(log_dir, "plots")
        os.makedirs(out_dir, exist_ok=True)

        # Current implementation: persist PNGs to disk using util.plotter.plot
        # Tags become filenames.
        for tag, payload in artifacts.items():
            try:
                var = payload["var"]
                data = payload["data"]
                file_path = os.path.join(out_dir, var, f"{tag}.png")
                parent_dir = os.path.dirname(file_path)
                os.makedirs(parent_dir, exist_ok=True)
                with open(file_path, mode="wb") as f:
                    plot(var, f, data)
            except Exception as ex:  # pragma: no cover - best-effort logging
                print(f"Warning: failed to log artifact {tag}: {ex}")

        # Clear after emission
        pl_module.artifacts = {}

    # Flush at key moments
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        self._flush_artifacts(trainer, pl_module)
        self._flush_newline(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        self._flush_artifacts(trainer, pl_module)
        self._flush_newline(trainer)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        self._flush_artifacts(trainer, pl_module)
        self._flush_newline(trainer)
