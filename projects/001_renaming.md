# Proposal: wxb Subcommand Naming and Documentation Alignment

This document summarizes the current state, identifies discrepancies, and proposes fixes to align CLI behavior and documentation. It also proposes improved command naming while preserving backward compatibility.

- Objectives:
  - Complete and unify documentation for all `wxb` subcommands (including the new `wxb eval`).
  - Fix inconsistencies between docs and implementation (especially time formats and output descriptions).
  - Propose improved subcommand names with a compatibility plan.

- Sources:
  - CLI: `wxbtool/wxb.py`
  - Implementations: `wxbtool/data/dsserver.py`, `wxbtool/nn/train.py`, `wxbtool/nn/test.py`, `wxbtool/nn/infer.py`, `wxbtool/nn/eval.py`, `wxbtool/data/download.py`
  - Current docs: `README.md` and `docs/user/*`

---

## 1. Current Subcommands and Options (as implemented)

### 1.1 wxb dserve (Dataset Server)

- Options:
  - `-b/--bind`: Binding address, supports `ip:port` or `unix:/path/to/xxx.sock`
  - `-p/--port`: Port (currently not used by the implementation)
  - `-w/--workers`: Number of Gunicorn workers
  - `-m/--module`: Spec module, default `wxbtool.specs.res5_625.t850weyn`
  - `-s/--setting`: Setting class name within the module, default `Setting`
  - `-t/--test`: `true/false`; if `false` actually starts Gunicorn
- Behavior: Loads train/eval/test splits and exposes them via a Flask server.

### 1.2 wxb train (Training)

- Options:
  - Hardware/Runtime: `-g/--gpu`, `-c/--n_cpu`, `-O/--optimize` (CI fast mode), `-t/--test` (limit to 1 epoch)
  - Data/Model: `-m/--module`, `-l/--load` (Lightning checkpoint or saved model), `-k/--check`, `-d/--data`
  - Hyperparameters: `-b/--batch_size`, `-e/--epoch`, `-n/--n_epochs`, `-r/--rate`, `-w/--weightdecay`
  - GAN: `-G/--gan`, `-R/--ratio` (G/D LR ratio), `-A/--alpha`, `-B/--balance`, `-T/--tolerance`
  - Other: `-p/--plot`
- Behavior: Uses PyTorch Lightning. Non-GAN branch includes EarlyStopping and ModelCheckpoint; GAN branch uses DDP strategy without callbacks.

### 1.3 wxb test (Testing)

- Options:
  - `-g/--gpu`, `-c/--n_cpu`, `-b/--batch_size`, `-m/--module`, `-l/--load`, `-d/--data`, `-G/--gan`, `-t/--test`, `-O/--optimize`
- Behavior: Lightning test flow; `--optimize` reduces validation/test batch counts for faster CI runs.

### 1.4 wxb infer (Deterministic Inference)

- Options:
  - `-g/--gpu`, `-c/--n_cpu`, `-b/--batch_size`, `-m/--module`, `-l/--load`, `-d/--data`
  - `-t/--datetime`: Must be `%Y-%m-%d` (date only)
  - `-o/--output`: Output filename; suffix must be `png` or `nc`
- Output: Writes `output/{date}/{module_last}.{png|nc}`
- Behavior: Builds WxDataset based on model setting and slices tensors; generates PNG only for `t2m`.

### 1.5 wxb inferg (GAN Probabilistic Inference)

- Options:
  - Same hardware/loading options as `infer`, plus:
  - `-t/--datetime`: Must be `YYYY-MM-DDTHH:MM:SS` (with time)
  - `-s/--samples`: Sample size (required)
  - `-o/--output`: `png` or `nc`
- Behavior: Iteratively samples noise for ensemble generation; `nc` output contains multi-member variables.

### 1.6 wxb eval (Backtesting / Day-by-day Scoring)

- Options:
  - Same as `infer`, and:
  - `-t/--datetime`: Must be `%Y-%m-%d` (date only)
  - `-o/--output`: `png` or `nc`
- Output:
  - Same PNG/NC as `infer`
  - Additionally, when outputting `nc`, writes `output/{date}/var_day_rmse.json` with day-by-day RMSE (keys are calendar dates)
- Behavior: For each variable in `vars_out`, calls `compute_drmse` and maps day offsets to real dates.

### 1.7 wxb download (Download Latest Hourly ERA5)

- Options:
  - `-m/--module`, `-G/--gan` (affects how variables are read from the setting)
  - `--coverage`: `daily | weekly | monthly | integer-days`
- Behavior: Infers variables/levels/resolution from the model setting; saves files into `era5/{variable}/YYYY/MM/DD/*.nc`; on first run, creates `~/.cdsapirc` after prompting for key.

---

## 2. Documentation Inconsistencies and Gaps

- Time format inconsistency
  - Docs show `infer -t` as `YYYY-MM-DDTHH:MM:SS`, but the implementation requires date-only `%Y-%m-%d`.
  - `inferg -t` requires a date-time (`YYYY-MM-DDTHH:MM:SS`) — the docs should explicitly state “time required”.
- New `wxb eval` is undocumented
  - Missing purpose (backtest/day-by-day scoring), usage, outputs (including `var_day_rmse.json`), and examples.
- dserve option caveat
  - `-p/--port` is not used by the current implementation; recommend using `-b/--bind`.
- Fast/CI modes for train/test
  - `-O/--optimize` and `-t/--test` behaviors are not clearly documented.
- Missing user docs for `download`
  - Add a section/page under Data Handling or a dedicated ERA5 download guide.
- Output path and naming
  - `infer/eval` outputs under `output/{date}/`; only `eval` (when using `nc`) writes the RMSE JSON. Docs should state this explicitly.

---

## 3. Documentation Update Plan (by file)

### 3.1 README.md

- Unify examples and explain time format differences:
  - infer (deterministic):
    ```bash
    wxb infer -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2025-01-01 -o output.png
    ```
  - inferg (GAN/probabilistic):
    ```bash
    wxb inferg -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2025-01-01T00:00:00 -s 10 -o output.nc
    ```
- Add `wxb eval` example:
  ```bash
  wxb eval -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2025-01-01 -o output.nc
  # Produces output/2025-01-01/*.nc and var_day_rmse.json (day-by-day RMSE)
  ```
- Add `wxb download` example:
  ```bash
  wxb download -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn --coverage weekly
  ```
- Note under dserve: current implementation uses `--bind`; `--port` is not effective.

### 3.2 docs/user/quickstart.md

- Add step 5: “Backtesting (wxb eval)”.
- Fix time formats:
  - infer: `YYYY-MM-DD`
  - inferg: `YYYY-MM-DDTHH:MM:SS`
- Add backtest example and outputs (including `var_day_rmse.json`):
  ```bash
  wxb eval -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2025-01-01 -o forecast.nc
  ```
- In “Common Options”, document `--optimize`, `--gan`, and GAN hyperparameters `--ratio/--alpha/--balance/--tolerance`.
- dserve note: use `--bind`; `--port` currently not honored.

### 3.3 docs/user/inference/overview.md

- Enforce strict time formats:
  - `wxb infer -t YYYY-MM-DD`
  - `wxb inferg -t YYYY-MM-DDTHH:MM:SS`
- Document output directory structure and naming: `output/{date}/{module_last}.{png|nc}`
- Mention `nc` variable dimensions for single/multi-variable outputs.

### 3.4 docs/user/evaluation/overview.md

- Add “Backtesting with wxb eval”:
  - Usage:
    ```bash
    wxb eval -m ... -t 2025-01-01 -o forecast.nc
    ```
  - Outputs:
    - `png/nc` files
    - If `nc`, creates `var_day_rmse.json` with calendar-date keys and RMSE values
  - Difference vs `wxb test`:
    - `test`: bulk evaluation on the test split
    - `eval` (backtest): rolling/day-by-day scoring for a specific initialization date (operational use-case)
  - Optional JSON example:
    ```json
    {
      "t2m": {
        "2025-01-01": 2.31,
        "2025-01-02": 2.45,
        "2025-01-03": 2.62
      }
    }
    ```

### 3.5 docs/user/data_handling/overview.md (or a new download.md)

- Add ERA5 download section:
  - Command:
    ```bash
    wxb download -m ... --coverage {daily|weekly|monthly|integer}
    ```
  - Reads variables/levels/resolution from the model setting
  - Directory structure: `era5/{variable}/YYYY/MM/DD/*.nc`
  - First run creates `~/.cdsapirc` (prompts for key)

### 3.6 Training/Testing docs enhancements

- Document:
  - `--optimize` (fast CI mode) and `--test true` (short training/testing)
  - Default checkpoint/log directory structure (e.g., `trains/{model_name}/`), if desired

---

## 4. Naming Discussion and Recommendations

We evaluate the suitability of the current names: `wxb dserve`, `wxb train`, `wxb test`, `wxb infer`, `wxb eval`, and propose improvements.

- dserve (dataset serve)
  - Issue: non-obvious abbreviation; not clearly grouped with data-related commands like `download`
  - Recommendation:
    - Prefer: `wxb data serve` (clear domain grouping)
    - Alternative: `wxb dataset serve`
    - Compatibility: keep `dserve` as an alias for a period; show both names in help

- train/test
  - Standard and widely understood; keep as-is

- infer (deterministic forecast)
  - Domain-preferred term: “forecast”; ML-preferred: “predict”
  - Recommendation:
    - Prefer: `wxb forecast` (domain-aligned)
    - Compatibility: keep `infer` as alias
  - Merge `inferg`:
    - Single command with options:
      ```bash
      wxb forecast -m ... -t 2025-01-01 --gan --samples 20 -o ensemble.nc
      ```
    - Reduces command surface and learning burden

- eval (backtesting/day-by-day scoring)
  - Issue: easily confused with `test` (dataset-level evaluation)
  - Recommendation:
    - Prefer: `wxb backtest` (precise in time-series/operational context)
    - Compatibility: keep `eval` as alias
  - In docs, explicitly contrast `test` vs `backtest`

- download
  - Recommendation: group under data as `wxb data download`
  - Compatibility: keep `download` as alias

### Recommended hierarchy (with backward compatibility)

- `wxb data serve` (alias: `dserve`)
- `wxb data download` (alias: `download`)
- `wxb train`
- `wxb test`
- `wxb forecast [--gan --samples N]` (aliases: `infer`, `inferg`)
- `wxb backtest` (alias: `eval`)

---

## 5. Rollout Plan

- Phase A (Docs alignment, no CLI changes)
  1. Update `README.md`, `docs/user/quickstart.md`, `docs/user/inference/overview.md`, `docs/user/evaluation/overview.md` to cover `wxb eval` and unify time formats
  2. Add ERA5 download section under Data Handling (or a new page)
  3. Clarify `dserve` `--bind` vs non-functional `--port`; document `--optimize`/`--test` semantics
  4. Document output directory structure and naming

- Phase B (Naming improvements with backward compatibility)
  1. Add aliases to the CLI (e.g., `data serve`, `forecast`, `backtest`, etc.)
  2. Switch help text and docs to the recommended names while noting deprecation of legacy names later

- Phase C (Major-version migration)
  1. Publish migration notice and CHANGELOG
  2. Remove legacy aliases (major releases only)

---

## 6. Appendix

### 6.1 Time Format Matrix

- `wxb infer -t`: `%Y-%m-%d` (date only)
- `wxb inferg -t`: `YYYY-MM-DDTHH:MM:SS` (date + time required)
- `wxb eval -t`: `%Y-%m-%d` (date only)

### 6.2 Output Paths and Files

- `infer/eval` output directory: `output/{date}/`
- Output filename: `{module_last}.{png|nc}`
- `eval` additional artifact: when outputting `nc`, writes `var_day_rmse.json` (day-by-day RMSE)

---

## 7. Open Questions

- Do we adopt the new naming (`data serve/download`, `forecast`, `backtest`) now, and do we merge `inferg` into `forecast` via options?
- If we only update docs for now, we will still implement all alignment described above (recommended baseline).
