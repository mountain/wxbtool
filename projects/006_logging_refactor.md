# Project 006: Metrics, CRPS, and Plotting Refactor with Logger-Agnostic Interface

Document ID: WXB-2025-09-18-IO  
Date: 2025-09-18  
Author: Mingli  
Status: Design – Ready for Implementation (Supersedes prior draft)

1. Executive Summary

This document refines and completes the design for a pluggable, logger-agnostic metrics and plotting system centered around nn/lightning.py. It specifies:

- Deterministic metrics: RMSE and ACC (anomaly correlation coefficient) as mandatory.
- GAN ensemble metric: CRPS as mandatory when -G/--gan is enabled.
- Plot generation that can be toggled by CLI -p/--plot.
- Logger-agnostic artifact handling via a universal callback.
- Exact shapes, weighting, denormalization, climatology use, file outputs (JSON/PNGs), and distributed constraints.

This design is the prerequisite for implementation; no code changes are included in this phase.

Core Pattern — Producer / Adapter / Configurator (Restored)

Why this matters: the essence of this refactor is a clean separation-of-concerns that keeps model science code independent from logging backends, while still enabling rich artifacts and metrics.

- Producer (LightningModule)
  - Responsibilities:
    - Compute and log scalar metrics via self.log/self.log_dict (RMSE/ACC/CRPS scalars).
    - Generate complex artifacts (matplotlib figures, images, maps) only when -p=true.
    - Do not call any backend-specific API. Instead, populate a transient dict artifacts_to_log: dict[str, Figure|np.ndarray].
  - Invariants:
    - Produces consistent shapes for metrics ([B,1,P,H,W] normalization).
    - Applies denormalization before metric computations.
    - Avoids file I/O directly for plots; defers to Adapter.
  - Minimal example:
    ```python
    class WxbModule(pl.LightningModule):
        def validation_step(self, batch, batch_idx):
            # ... compute metrics rmse/acc/crps and log scalars ...
            self.log("val_rmse", avg_rmse, sync_dist=True, prog_bar=True)
            if self.opt.plot == "true" and batch_idx == 0:
                fig = self._plot_fcst_vs_tgt(...)
                # Backend-agnostic handoff
                if not hasattr(self, "artifacts_to_log"):
                    self.artifacts_to_log = {}
                self.artifacts_to_log[f"fcst_{var}_d{day}"] = fig
    ```

- Adapter (UniversalLoggingCallback)
  - Responsibilities:
    - Inspect trainer.logger type and dispatch artifacts to the appropriate backend.
    - Handle rank-0-only writes and safe cleanup (plt.close).
    - Fallback to saving under logger.log_dir/plots (or cwd/plots) if backend offers no native figure logging.
  - Invariants:
    - Pure adapter layer; no science/metric logic.
    - Clears pl_module.artifacts_to_log after logging to avoid leaks.
  - Minimal example:
    ```python
    class UniversalLoggingCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            if not getattr(pl_module, "artifacts_to_log", None):
                return
            if not trainer.is_global_zero:
                pl_module.artifacts = {}
                return
            logger = trainer.logger
            out_dir = getattr(logger, "log_dir", "plots")
            os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
            for tag, fig in pl_module.artifacts.items():
                if isinstance(logger, WandbLogger):
                    logger.experiment.log({tag: wandb.Image(fig)})
                elif isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_figure(tag, fig, global_step=trainer.global_step)
                else:
                    fig.savefig(os.path.join(out_dir, "plots", f"{tag}.png"))
                plt.close(fig)
            pl_module.artifacts = {}
    ```

- Configurator (Training Script / CLI Entrypoint)
  - Responsibilities:
    - Parse --logger and -p flags and instantiate the correct logger.
    - Always attach UniversalLoggingCallback to keep logging pluggable.
    - Ensure DDP-friendly behavior (rank-0 writes handled by PL + callback).
  - Minimal example:
    ```python
    def build_trainer(args):
        if args.logger == "wandb":
            logger = WandbLogger(project="wxb", name=args.run_name)
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger("tb_logs", name=args.run_name)
        else:
            logger = CSVLogger("logs", name=args.run_name)
        return Trainer(logger=logger, callbacks=[UniversalLoggingCallback()], ...)
    ```

Interaction flow

1) Producer computes metrics (RMSE/ACC/CRPS) and logs scalars via self.log.
2) When -p=true, Producer creates figures/images and stashes them in artifacts_to_log.
3) Adapter (callback) runs after step/epoch, detects artifacts, dispatches to logger backend, and writes files only on rank 0.
4) Configurator selects backend via --logger and mounts the adapter, keeping core code logger-agnostic.

How metrics/plots map onto the pattern

- Metrics (RMSE, ACC, CRPS): computed and logged by Producer as scalars (self.log).
- Per-day values (RMSE/ACC JSON): still serialized by Producer (rank-0 gated).
- Plots: produced by Producer when -p=true; actual emission delegated to Adapter.
- Backend selection and callback wiring: handled by Configurator via CLI (e.g., --logger wandb|tensorboard|csv).

This restored pattern is authoritative; all subsequent sections (metrics definitions, plotting scheme, DDP safety, CLI) conform to it.

2. Goals and Non-Goals

2.1 Goals

- Formalize how RMSE, ACC, and CRPS are computed, aggregated, and logged in Lightning validation/test loops.
- Define plotting scope, filenames, directories, and CLI toggle (-p).
- Keep logic logger-agnostic and DDP-safe (rank-0 writes only).
- Preserve and clarify existing JSON artifacts (val_rmse.json, val_acc.json, test_* counterparts).
- Support both deterministic and GAN modes consistently.

2.2 Non-Goals

- Alter model architectures or dataset interfaces.
- Change training loss definitions.
- Implement code; this is a design document only.

3. Context (Current State in nn/lightning.py)

- Weighted RMSE: compute_rmse and compute_rmse_by_time (per-day and overall) using model.weight (area weights), pred_span, and denormalizors.
- ACC: calculate_acc computes anomaly correlation using ClimatologyAccessor and writes anomaly plots anomaly_{var}_fcs_{day}.png and anomaly_{var}_obs_{day}.png into cwd/plots/.
- CRPS (GAN): compute_crps computes a simplified per-pixel CRPS and an “absorb” ratio, logging crps and absb scalars; forecast/test loaders use ensemble_loader.
- JSON outputs: val_rmse.json/test_rmse.json and val_acc.json/test_acc.json are written in logger.log_dir.
- Plotting: Plotting occurs conditionally based on opt.plot == "true"; CI mode reduces work.
- Distributed: PyTorch Lightning typically runs logging hooks on rank 0, but explicit rank-zero enforcement for file I/O should be clarified and standardized.

4. Unified Metric Definitions and Shapes

4.1 Notation and Shapes

- Variables: vars_out (list of strings).
- Temporal:
  - input_span: model.setting.input_span
  - pred_span: model.setting.pred_span
- Spatial:
  - lat_size × lon_size = H × W (from model.setting)
  - model.weight: area weight array broadcastable to [B, C, P, H, W] or [1,1,1,H,W].
- Tensors (deterministic):
  - targets[var]: shape either [B, P, H, W] or [B, C, P, H, W] (C=1). We normalize to [B, 1, P, H, W] for metric compute.
  - results[var]: same normalization to [B, 1, P, H, W].
- Tensors (GAN ensemble):
  - predictions: ensemble samples. Canonical design shape: [B, S, C, P?, H, W] where:
    - B: batch size
    - S: ensemble size (samples)
    - C: channels (1 for single variable map)
    - P?: optional pred_span dimension (day index); for daily CRPS, compute per day.
  - targets: [B, C, P?, H, W].

4.2 Weighted RMSE (Mandatory, Deterministic and GAN)

- Denormalize before metric computation via denormalizors[var].
- Broadcast weights to [B, 1, P, H, W].
- Per-day RMSE:
  - For day d: RMSE_d = sqrt( sum(weight * (forecast_d - target_d)^2) / sum(weight) ).
  - Store float values (Python float) for JSON serialization.
- Overall RMSE across days: combine weighted sums over all d before sqrt.
- Outputs:
  - val_rmse.json / test_rmse.json with schema:
    {
      "<var>": {
        "<epoch>": {
          "<day>": <rmse_float>
        }
      }
    }
  - Scalars logged: val_rmse/test_rmse = average over vars_out (scalar tensors logged via self.log).

4.3 ACC – Anomaly Correlation Coefficient (Mandatory, Deterministic)

- Climatology: Use ClimatologyAccessor with years chosen by mode (train/eval/test). Map indexes (sample time origins) to forecast times using step and pred_shift; retrieve per-day climatology.
- Anomalies:
  - f_anomaly = forecast - climatology
  - o_anomaly = observation - climatology
  - All shapes converted to [B, 1, P, H, W] (Numpy for current code; PyTorch acceptable too).
- Area-weighted ACC per day:
  - prod = sum(weight * f_anomaly_d * o_anomaly_d)
  - fsum = sum(weight * f_anomaly_d^2)
  - osum = sum(weight * o_anomaly_d^2)
  - acc_d = prod / sqrt(fsum * osum)
- Aggregation:
  - Persist per-day ACC per variable in accByVar for JSON outputs:
    val_acc.json / test_acc.json schema mirrors RMSE JSON.
- Scalars logged:
  - We will log per-variable epoch ACC as val_acc_<var> averaged over days. An overall val_acc (over vars) is optional; if logged, ensure it is clearly named.

4.4 CRPS – Continuous Ranked Probability Score (Mandatory for GAN)

- Applies only when -G/--gan is enabled and ensemble samples are produced.
- Per-pixel CRPS for ensemble E(y):
  - CRPS(x) = E|Y - x| - 0.5 E|Y - Y'|
  - In practice, compute per pixel across samples S; then average over pixels with area weights.
- Shapes:
  - predictions: [B, S, 1, P, H, W] or [B, S, 1, H, W] if no day dimension.
  - targets: [B, 1, P, H, W] or [B, 1, H, W].
- Day handling:
  - If predictions include P, compute CRPS per day. Else, compute for current day only.
- Weighting:
  - Weight per-pixel CRPS maps with model.weight then average.
- Additional diagnostic: absorb = 0.5 * E|Y - Y'| / (E|Y - x| + 1e-7). Log as absb as currently done.
- Scalars logged: crps and absb. If per-day, log crps_d and absb_d, and optionally average to crps/absb.

5. Plotting Plan and -p/--plot Toggle

5.1 CLI Semantics

- -p/--plot: boolean toggle to enable image generation.
  - Default: false (no plots).
  - Accepts values true/false. If provided without value, treat as true.
- Applies to: train (selected steps), validation, test for both deterministic and GAN modes.
- Plot frequency controls (defer implementation but reserve options):
  - --plot_every_n_steps (int, default 10 for train in GAN, off otherwise)
  - --plot_max_batches (int, default 1 when CI mode to limit output)

5.2 What to Plot (when -p=true)

- Inputs:
  - For each var in vars_in and t in [0..input_span-1]:
    - plots/<var>/var_inp_{t}.png
- Forecast vs Target (deterministic and GAN):
  - For each var in vars_out and day d in [0..pred_span-1]:
    - plots/<var>/var_fcst_{d}.png
    - plots/<var>/var_tgt_{d}.png
- ACC anomalies (deterministic only):
  - For each var and day d:
    - plots/anomaly_{var}_fcs_{d}.png (forecast anomaly)
    - plots/anomaly_{var}_obs_{d}.png (obs anomaly)
- GAN CRPS/absorb maps (optional but recommended):
  - For each var and day d (if P present; else current horizon):
    - plots/gan_crps_{var}_{d}.png (CRPS map)
    - plots/gan_absorb_{var}_{d}.png (absorb map)
  - Note: Requires retaining per-pixel CRPS/absorb tensors; omit if memory/time prohibitive.

5.3 Output Location and DDP Safety

- Plots directory:
  - Prefer logger.log_dir/plots to keep artifacts under the run directory.
  - Backward-compatible: if logger.log_dir not available, fallback to cwd/plots (current behavior).
- JSON files remain in logger.log_dir.
- DDP/rank-zero:
  - Only rank 0 writes images and JSON to avoid clobbering. Enforce via rank_zero_only or trainer.is_global_zero checks.
- CI mode:
  - CI reduces plotting to batch_idx == 0 (or per existing logic) and may skip anomalies to speed up runs.

6. Data and Pre/Post-Processing Invariants

6.1 Denormalization

- All deterministic metric computations (RMSE, ACC) must use denormalized data. Use denormalizors[var] on both forecast and target before any metric operations.
- For CRPS, compute on denormalized maps as well.

6.2 Device/DType

- Align device and dtype prior to compute:
  - Move tensors to the same device (results’ device).
  - Cast weights to target dtype.
- Shapes must be made explicit:
  - Convert [B, P, H, W] to [B, 1, P, H, W] (add channel dim) for consistency.

6.3 Area Weights

- model.weight must broadcast to [B, 1, P, H, W]; reshape to [1,1,1,H,W] and repeat/broadcast as needed.
- Provide explicit error messages if model.weight is missing or has an incompatible shape.

6.4 Climatology

- ClimatologyAccessor home inferred from WXBHOME or falls back to data/climatology.
- Index mapping:
  - For each batch indices tensor (indexes), apply step and pred_shift to compute forecast time indices per day.
- CI mode:
  - May bypass expensive climatology work and return zeros with shape [B, |vars_out|, P, H, W] to keep tests fast.

7. Logger-Agnostic Artifact Adapter

7.1 UniversalLoggingCallback

- Always attach a UniversalLoggingCallback to the Trainer.
- Responsibilities:
  - Consume pl_module.artifacts_to_log (dict tag->matplotlib figures or image arrays) and dispatch to logger-specific APIs.
  - Supported loggers:
    - WandbLogger: log images as wandb.Image
    - TensorBoardLogger: add_figure per tag
    - CSVLogger or others: save to logger.log_dir/plots as files
  - Close figures to avoid leaks; clear artifacts_to_log after logging.
  - Enforce rank-0-only execution for any file/log writes.

7.2 Producing Artifacts in LightningModule

- LightningModule does NOT directly call logger-specific methods.
- It populates artifacts_to_log when -p is enabled and at selected hooks:
  - on_train_batch_end (GAN every N steps)
  - validation_step/test_step (batch_idx gating)
  - on_validation_epoch_end/on_test_epoch_end (optional summaries)
- Artifacts may be either matplotlib Figures or raw numpy arrays to be wrapped for the backend.

7.3 JSON Emission

- JSON files (val_rmse.json, val_acc.json, test_rmse.json, test_acc.json) remain written by LightningModule, but gated by rank 0.
- Schema specified in 4.2 and 4.3.

8. CLI and Configuration

- -p/--plot: bool; default false; applies across train/test/forecast/backtest.
- -G/--gan: enables GAN training/validation/test pathways; requires -s/--samples > 0 in forecast mode (enforced elsewhere).
- --plot_every_n_steps (optional, default 10 for GAN train) and --plot_max_batches (optional, default 1 in CI) reserved for future extension.
- WXB_PLOTS_ROOT (optional env): overrides plot root. If not set, prefer logger.log_dir/plots, else cwd/plots.

9. Backtest (wxb backtest / wxb eval) Specifics

- When producing NetCDF (nc) outputs, also write output/{date}/var_day_rmse.json keyed by calendar dates:
  {
    "t2m": {
      "2025-01-01": 2.31,
      "2025-01-02": 2.45
    }
  }
- Day-by-day RMSE uses compute_rmse_by_time and maps offsets to dates based on initialization date and pred_span.
- Only rank 0 writes artifacts to output/{date}.

10. Acceptance Criteria

- Deterministic:
  - val_rmse/test_rmse scalars logged as the mean over vars_out; per-var per-day RMSE persisted in JSON.
  - ACC per-var per-day values persisted; per-var epoch ACC logged; anomaly plots emitted when -p=true.
- GAN:
  - crps and absb scalars logged (per day if applicable); optional per-pixel PNG maps when -p=true.
  - GAN training: image artifacts produced every N steps when -p=true.
- Plotting:
  - All images go under logger.log_dir/plots (fallback to cwd/plots).
  - No file clobbering under DDP; rank 0 writes only.
- CLI:
  - -p toggles all plotting consistently across commands.
- CI:
  - CI path executes quickly: minimal or no plotting; climatology fast path.

11. Implementation Plan (Sequenced)

- Metrics module (new): wxbtool/nn/metrics.py
  - rmse_weighted(), rmse_by_time(), acc_anomaly_by_time(), crps_ensemble()
  - Typed functions with unit tests.
- Lightning integration:
  - Refactor nn/lightning.py to call metrics.* and enforce rank-0 for file writes.
  - Normalize shapes ([B, 1, P, H, W]) inside metrics functions.
  - Denormalize within metrics rather than callsites to avoid duplication.
- Plotting utilities:
  - Centralize plotting helpers that return matplotlib Figure or write to a file-like object.
  - Switch path root to logger.log_dir/plots with fallback.
- Callback:
  - Implement UniversalLoggingCallback with Wandb/TensorBoard/CSV support and rank-0 guard.
- CLI:
  - Standardize -p/--plot across train, test, forecast, backtest; bool with default false.
  - (Optional) Add --plot_every_n_steps and --plot_max_batches; wire into opt.
- Docs:
  - Update user/evaluation and user/inference to mention metrics/plots and -p toggle.
  - Cross-link from README Quick Start.

12. Testing Strategy

- Unit tests for metrics functions (deterministic RMSE/ACC with toy grids and known weights; GAN CRPS with small S).
- Serialization:
  - Verify JSON schemas for val/test artifacts and backtest var_day_rmse.json.
- Plot tests:
  - Smoke tests that -p writes expected files on rank 0 and not on other ranks (simulate with environment).
- Performance:
  - Ensure CI mode reduces work (skip/limit plotting and climatology).
- DDP:
  - Multi-process test (torchrun --nproc_per_node=2) verifying single-writer behavior.

13. Risks and Mitigations

- Shape mismatches: Normalize shapes inside metrics and include clear error messages.
- Weight misuse: Validate model.weight presence/shape; raise with actionable message.
- Climatology correctness: Document date mapping and add unit tests for index→date logic.
- Logging backend divergence: Keep callback narrow; add new backends via adapter pattern.
- Disk churn in large runs: -p defaults to false; add knobs to throttle plotting frequency.

14. Open Questions

- Should we emit an overall val_acc scalar across vars? If yes, define clearly (mean over vars and days).
- For GAN CRPS with multi-day outputs, do we require S samples per day or reuse day-0 samples? (Recommended: per-day sampling if runtime permits; otherwise compute day-0 only and document.)
- Should we move anomaly PNGs into logger.log_dir/plots/anomaly/<var>/ for clearer structure? (Recommended.)
- Do we deprecate writing PNGs to cwd/plots entirely in a future release?

15. Deliverables Summary

- Deterministic: RMSE and ACC defined, computed per-day and aggregated; JSON + scalars; anomaly plots gated by -p.
- GAN: CRPS (+absorb) defined and logged; optional maps under -p.
- Plots: Unified location, naming, and rank-0 behavior; -p toggles.
- Callback: Universal logger adapter to keep LightningModule and training scripts backend-agnostic.
