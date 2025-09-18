# -*- coding: utf-8 -*-
"""
Universal logging callback to adapt artifacts_to_log to different loggers.

Producer (LightningModule) should populate:
    self.artifacts_to_log: Dict[str, Dict[str, Any]]
where each value is a dict containing:
    - "var": variable name/code (used by util.plotter.plot)
    - "data": numpy array shaped [H, W] or [1, H, W] (will be reshaped)

This callback will:
    - On rank-0 only, flush artifacts to the configured logger backend.
    - Prefer saving PNG files under <logger.log_dir>/plots.
    - Clear artifacts_to_log after flushing to avoid memory leaks.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import lightning.pytorch as pl

try:
    from lightning.pytorch.loggers import TensorBoardLogger
except Exception:  # pragma: no cover
    TensorBoardLogger = object  # type: ignore

from wxbtool.util.plotter import plot


class UniversalLoggingCallback(pl.Callback):
    def _flush_artifacts(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        artifacts: Dict[str, Dict[str, Any]] = getattr(pl_module, "artifacts_to_log", None)
        if not artifacts:
            return

        # Rank-0 only writes/logs artifacts
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            pl_module.artifacts_to_log = {}
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
                file_path = os.path.join(out_dir, f"{tag}.png")
                with open(file_path, mode="wb") as f:
                    plot(var, f, data)
            except Exception:  # pragma: no cover - best-effort logging
                # Fail silently on plotting errors to avoid breaking training
                pass

        # Clear after emission
        pl_module.artifacts_to_log = {}

    # Flush at key moments
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        self._flush_artifacts(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        self._flush_artifacts(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        self._flush_artifacts(trainer, pl_module)
