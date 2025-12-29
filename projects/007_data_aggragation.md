# Project 007: Data Aggregation and Windowing

Status: Proposed
Created: 2025-12-28
Author: AI Assistant
Type: Architecture Decision Record & Technical Specification

1. Executive Summary

While `data-download` provides raw high-frequency (e.g., hourly) data, climate modeling (S2S) often requires data at coarser granularities (daily, weekly) or processed via sliding windows.

This project introduces a new command `wxb data-agg`. It functions as a **Target-Driven Windowing Operator**. It iterates through the target timestamps defined by a destination `Setting`, fetches a configurable input window (forward, backward, or centered) from source data, applies a reduction operator (default: mean), and saves the result according to the destination `Setting`'s layout.

2. Problem Statement

- **Granularity Mismatch**: Raw data is typically hourly/daily, but models may need weekly/monthly inputs.
- **Sliding Window Requirement**: Users need to generate files where each timestep represents an aggregation over a window (e.g., 7-day running mean).
- **Alignment Flexibility**: Different tasks require different window alignments:
  - *Input Features*: Often use **Backward** windows (past 7 days).
  - *Training Targets*: Often use **Forward** windows (future 7 days average).
  - *Smoothing*: Often uses **Centered** windows.

3. Design Goals

- **Target-Driven**: Output follows the `granularity` and `years` of the destination `Setting`.
- **Configurable Alignment**: Support `forward` (future), `backward` (past), and `centered` windows.
- **Format Compliance**: Output files adhere to `Setting.data_path_format` (Project 003).
- **Parallelism**: Multi-process execution for efficiency.

4. Proposed Design

4.1 CLI Interface

```bash
wxb data-agg \
  -m wxbtool.zoo.my_model \
  -s MyDailySetting \         # Defines the TARGET layout (Daily)
  --src ./era5_raw \          # Source directory (Raw Hourly/Daily)
  --window 168 \              # Window size in hours (e.g., 7 days)
  --align backward \          # Alignment: backward (default), forward, center
  --step 24 \                 # (Optional) Explicit step size if not inferable from Setting
  --workers 8

```

4.2 Alignment Logic ( is the timestamp of the file being generated)

* **Backward (Default)**: Window looks at the past.
* Range:  (Time interval is right-closed usually).
* Use case: Generating input features from history.
* *Note*: Matches pandas `rolling(align='right')`.


* **Forward**: Window looks at the future.
* Range: 
* Use case: Generating ground-truth labels (Targets) for S2S forecasting.
* *Note*: Matches pandas `rolling(align='left')`.


* **Centered**: Window is centered on .
* Range: 
* Use case: Smoothing / filtering analysis.
* *Note*: Matches pandas `rolling(align='center')`.



4.3 Workflow

1. **Context Loading**: Load destination `Setting` to get target `vars`, `years`, and `granularity`.
2. **Target Generation**: Create sequence of target timestamps () (e.g., every day at 00:00).
3. **Task Dispatch**: For each  and each variable:
* Calculate  based on `--window` and `--align`.
* Pass to worker.


4. **Worker Execution**:
* Identify source files overlapping .
* Load data subset.
* **Operator**: Calculate Mean (V1 implementation).
* Write to destination path defined by `Setting`.


5. Implementation Details

5.1 `Aggregator` Class
Core logic for time calculation.

```python
class Aggregator:
    def get_window_range(self, t_out, window_hours, alignment):
        delta = timedelta(hours=window_hours)
        if alignment == 'backward':
            return t_out - delta, t_out
        elif alignment == 'forward':
            return t_out, t_out + delta
        elif alignment == 'center':
            half = timedelta(hours=window_hours / 2)
            return t_out - half, t_out + half

```

5.2 Parallel Strategy

* Dispatch by **(Year, Variable)** chunks to minimize process overhead.
* Inside each chunk, process day-by-day.

6. Backward Compatibility

* No impact on existing datasets.
* Fully compatible with Project 003 path formats.

7. Risks and Mitigations

* **Boundary Handling**: "Center" alignment on hourly data uses standard `datetime` arithmetic. V1 strict mode requires full window coverage.
* **Missing Data**:
* V1 Strict Mode: If any required source file in the window is missing, skip and log warning.
* **Performance**: Repeated reads for sliding windows rely on OS file cache for V1.
