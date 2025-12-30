# Data Aggregation

The `wxb data-agg` command allows you to process high-frequency raw data (e.g., hourly ERA5 output) into lower-frequency datasets (e.g., daily or weekly averages) suitable for climate modeling (sub-seasonal to seasonal forecasting, S2S).

## Usage

```bash
wxb data-agg \
  -m <module_path> \
  -s <setting_class> \
  --src <source_data_root> \
  --window <window_hours> \
  [--align <alignment>] \
  [--workers <num_workers>]
```

### Arguments

- `-m, --module`: Python module path containing your Setting class (e.g., `wxbtool.specs.my_spec`).
- `-s, --setting`: Name of the Setting class that defines the **destination** layout (variables, years, and granularity).
- `--src`: Path to the root directory of your source observations (usually raw hourly data).
- `--window`: Aggregation window size in hours (e.g., `24` for daily average, `168` for weekly).
- `--align`: Alignment of the window relative to the target timestamp.
  - `backward` (default): Statistics for the PAST window `(t - w, t]`. Useful for creating input features.
  - `forward`: Statistics for the FUTURE window `[t, t + w)`. Useful for creating training targets (ground truth).
  - `center`: Statistics for the CENTERED window `(t - w/2, t + w/2]`. Useful for smoothing or climatology.
- `--workers`: Number of parallel worker processes (default: 4).

## Creating a Setting for Aggregation

> [!IMPORTANT]
> **Do not reuse your training Setting class blindly.**
> Deep learning model settings usually default to `granularity="yearly"` (one file per year). If you want to generate daily files, you MUST create a dedicated Setting class.

The `data-agg` command is **Target-Driven**: it generates one output file for each timestamp defined by your Setting's `years` and `granularity`.

### Example: Generating Daily Averages

If your goal is to create daily average files (e.g., one file per day), create a specific setting class:

```python
from wxbtool.core.setting import Setting

class SettingDailyAgg(Setting):
    def __init__(self):
        super().__init__()
        # 1. Set Granularity to 'daily' to generate one output per day
        self.granularity = "daily" 
        
        # 2. Define a Path Format that includes Day/Month to avoid overwriting files
        # Format placeholders: {var}, {year}, {month}, {day}, {resolution}
        self.data_path_format = "{var}/{year}/{month:02d}/{var}_{year}-{month:02d}-{day:02d}_{resolution}.nc"
        
        # 3. Define the years you want to process
        self.years_train = range(1980, 2015)
        
        # 4. Define variables
        self.vars = ["2m_temperature", "total_precipitation"]
```

Then run the command:

```bash
wxb data-agg -m my_module -s SettingDailyAgg --src ./raw_data --window 24 --align backward
```

### Why a separate Setting?

If you use a standard training configuration `SettingClimate` (which defaults to `granularity="yearly"` and path `{var}_{year}.nc`), the command will:
1. Only generate **one file per year** (usually for Jan 1st).
2. Ignore the rest of the year's data.

By defining a lightweight `SettingDailyAgg`, you explicitly tell the tool: "I want output for every single day, organized in this specific directory structure."

## Metadata recorded in output NetCDF

When writing each aggregated file, the command infers and saves key metadata into the NetCDF:

- Aggregation and window
  - aggregation: mean
  - alignment: backward/forward/center
  - window_hours: integer hours
  - window_start, window_end: ISO timestamps of the aggregated window
  - time_coverage_start, time_coverage_end: same as window range
  - target_time: the timestamp of this output file
- Time-step inference
  - source_time_step_hours: inferred from source ds.time differences (mode in hours)
- Grid and regridding
  - src_grid_* and dst_grid_*: nlat/nlon, min/max, and approximate resolution for lat/lon
  - lon_convention: 0-360 or -180-180 applied to output
  - regrid_method_lat: interp-linear if 33->32 latitude interpolation was applied, otherwise unchanged
  - regrid_method_lon: convention-convert-only (no longitudinal resampling in v1)
- Variable and vertical levels
  - variable_code: canonical code (e.g., t2m)
  - is_3d: true/false
  - level_dim, level_count, level_units, levels (for 3D variables; empty for 2D)
- Provenance and conventions
  - Conventions: CF-1.8
  - source_files: semicolon-separated list of hourly source files
  - history: processing history line with window and alignment

Note: In v1, longitude is not resampled—only the convention is normalized. Latitude interpolation is applied only for the common 33->32 case at the target resolution.

## Workflow Example: Preparing S2S Data

To prepare data for a generic S2S model (e.g., input = past 7 days, target = future 7 days):

1. **Input Features (History)**:
   ```bash
   wxb data-agg -s SettingDailyAgg ... --window 168 --align backward
   ```
   *Generates daily files where each day represents the average of the PREVIOUS 7 days.*

2. **Ground Truth (Targets)**:
   ```bash
   wxb data-agg -s SettingDailyAgg ... --window 168 --align forward
   ```
   *Generates daily files where each day represents the average of the NEXT 7 days.*

(Note: You typically output these to different folders or variable names to differentiate them).
