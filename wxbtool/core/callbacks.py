import os
import lightning.pytorch as pl

from typing import Any, Dict
from wxbtool.util.plot import plot,plot_image


class UniversalLoggingCallback(pl.Callback):
    def _flush_newline(self, trainer: pl.Trainer) -> None:
        if hasattr(trainer, "is_global_zero") and trainer.is_global_zero:
            print(flush=True)

    def _flush_artifacts(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        artifacts: Dict[str, Dict[str, Any]] = getattr(pl_module, "artifacts", None)
        if not artifacts:
            return

        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            pl_module.artifacts = {}
            return

        logger = getattr(trainer, "logger", None)
        log_dir = getattr(logger, "log_dir", None) or os.getcwd()
        out_dir = os.path.join(log_dir, "plots")
        os.makedirs(out_dir, exist_ok=True)

        for tag, payload in artifacts.items():
            try:
                var = payload["var"]
                data = payload["data"]
                data_type = payload["type"]
                data_kind = payload["kind"]
                if data_type == "data":
                    file_path = os.path.join(out_dir, data_type, var, f"{tag}.png")
                    parent_dir = os.path.dirname(file_path)
                    os.makedirs(parent_dir, exist_ok=True)
                    with open(file_path, mode="wb") as f:
                            plot(var, f, data)
                elif data_type == "map" and data_kind == "input":
                    file_path = os.path.join(out_dir, data_type, var, f"{tag}.png")
                    parent_dir = os.path.dirname(file_path)
                    os.makedirs(parent_dir, exist_ok=True)

                    input_data = data
                    truth_data = artifacts[tag.replace("_input", "_truth")]["data"]
                    forecast_data = artifacts[tag.replace("_input", "_forecast")]["data"]
                    year = payload.get("year", None)
                    doy = payload.get("doy", None)
                    plot_image(
                        var,
                        input_data=input_data,
                        truth=truth_data,
                        forecast=forecast_data,
                        title=var,
                        year=year,
                        doy=doy,
                        save_path=file_path,
                    )
            except Exception as ex:  # pragma: no cover - best-effort logging
                print(f"Warning: failed to log artifact {tag}: {ex}")

        pl_module.artifacts = {}

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._flush_artifacts(trainer, pl_module)
        self._flush_newline(trainer)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int=0) -> None:
        self._flush_artifacts(trainer, pl_module)
        self._flush_newline(trainer)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int=0) -> None:
        self._flush_artifacts(trainer, pl_module)
        self._flush_newline(trainer)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        self._flush_artifacts(trainer, pl_module)
        self._flush_newline(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        self._flush_artifacts(trainer, pl_module)
        self._flush_newline(trainer)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        self._flush_artifacts(trainer, pl_module)
        self._flush_newline(trainer)
