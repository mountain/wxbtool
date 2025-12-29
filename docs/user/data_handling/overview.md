# Data Handling Overview

This guide explains how wxbtool manages and processes meteorological data for weather prediction models.

## Data Requirements

wxbtool works with meteorological data in NetCDF format, organized by variable directories. By default, datasets are yearly files, but the organization is now configurable per Setting.

Default expectations (backward-compatible):
- Each variable has its own directory (e.g., `temperature/`, `geopotential/`)
- Files are named according to the pattern: `{var}_{year}_{resolution}.nc`
- The default resolution is `5.625deg` (approximately 5.625° × 5.625° grid)

Configurable dataset organization:
- You can change the temporal granularity (yearly, monthly, daily, hourly, weekly, quarterly) and file path format per Setting using:
  - `Setting.granularity`
  - `Setting.data_path_format`
- See “Configuring Dataset Organization” below for details and examples.

## Supported Variables

wxbtool supports both 2D and 3D meteorological variables:

### 3D Variables (with vertical levels)
- `geopotential`: Geopotential height at various pressure levels
- `temperature`: Air temperature at various pressure levels
- `u_component_of_wind`: Eastward wind component
- `v_component_of_wind`: Northward wind component
- `specific_humidity`: Moisture content

### 2D Variables (surface or single-level)
- `2m_temperature`: Temperature at 2 meters above surface
- `10m_u_component_of_wind`: Eastward wind component at 10m
- `10m_v_component_of_wind`: Northward wind component at 10m
- `toa_incident_solar_radiation`: Incoming solar radiation at top of atmosphere
- `total_precipitation`: Accumulated precipitation
- `total_cloud_cover`: Cloud cover fraction

### Extending Supported Variables and Normalization

You can extend the set of variables and their normalization without modifying library code:

- Register variables (2D/3D), codes, and aliases via the official Variable Registry APIs.
- Register normalization/denormalization functions via the Normalization Registry APIs.
- Prefer registries over monkeypatching; registries are idempotent, override-aware, and logged.

See:
- Technical guide: Variable Registry — docs/technical/extension/new_variables.md
- Technical guide: Normalization Systems — docs/technical/specifications/normalization.md

## Data Organization

### Directory Structure

Default yearly layout (backward-compatible):
```
WXBHOME/
├── geopotential/
│   ├── geopotential_1980_5.625deg.nc
│   ├── geopotential_1981_5.625deg.nc
│   └── ...
├── temperature/
│   ├── temperature_1980_5.625deg.nc
│   ├── temperature_1981_5.625deg.nc
│   └── ...
└── ... (other variables)
```

Example monthly layout:
```
WXBHOME/
└── 2m_temperature/
    ├── 1980/
    │   ├── 2m_temperature_1980-01_5.625deg.nc
    │   ├── 2m_temperature_1980-02_5.625deg.nc
    │   └── ...
    └── 1981/
        └── 2m_temperature_1981-01_5.625deg.nc
```

Example daily layout:
```
WXBHOME/
└── temperature/
    ├── 1981/
    │   └── 12/
    │       ├── temperature_1981-12-30_5.625deg.nc
    │       └── temperature_1981-12-31_5.625deg.nc
    └── 1982/
        └── 01/
            ├── temperature_1982-01-01_5.625deg.nc
            └── temperature_1982-01-02_5.625deg.nc
```

### Temporal Coverage

Models typically use data from specific year ranges:
- Training: 1980-2014 (default)
- Validation: 2015-2016 (default)
- Testing: 2017-2018 (default)

These ranges can be configured in your model's `Setting` class.

## Configuring Dataset Organization (granularity and data_path_format)

wxbtool’s dataset loader can discover files at different temporal grains by combining a pandas date range with a user-specified file path format.

- `Setting.granularity` controls the date_range frequency used for discovery:
  - Supported values: `yearly`, `quarterly`, `monthly`, `weekly`, `daily`, `hourly`
  - Mapping: `yearly -> YS`, `quarterly -> QS`, `monthly -> MS`, `weekly -> W-MON`, `daily -> D`, `hourly -> H`
  - Unknown values default to daily with a warning

- `Setting.data_path_format` is a Python format string relative to the variable directory:
  - Full path is resolved as: `<root>/<var>/<data_path_format.format(**fields)>`
  - Supported placeholders:
    - `{var}`: variable directory/name (e.g., `2m_temperature`)
    - `{resolution}`: e.g., `5.625deg`
    - `{year}`: 4-digit year
    - `{month}`: 1–12 (use `{month:02d}` for zero-padded month)
    - `{day}`: 1–31 (use `{day:02d}`)
    - `{hour}`: 0–23 (use `{hour:02d}`)
    - `{week}`: ISO week number 1–53 (use `{week:02d}`)
    - `{quarter}`: 1–4

Default (yearly, backward-compatible):
```python
class MySetting(Setting):
    def __init__(self):
        super().__init__()
        self.granularity = "yearly"
        self.data_path_format = "{var}_{year}_{resolution}.nc"
```

Monthly example:
```python
class MySetting(Setting):
    def __init__(self):
        super().__init__()
        self.granularity = "monthly"
        self.data_path_format = "{year}/{var}_{year}-{month:02d}_{resolution}.nc"
```

Daily example:
```python
class MySetting(Setting):
    def __init__(self):
        super().__init__()
        self.granularity = "daily"
        self.data_path_format = "{year}/{month:02d}/{var}_{year}-{month:02d}-{day:02d}_{resolution}.nc"
```

Hourly example:
```python
class MySetting(Setting):
    def __init__(self):
        super().__init__()
        self.granularity = "hourly"
        self.data_path_format = "{year}/{month:02d}/{day:02d}/{var}_{year}-{month:02d}-{day:02d}T{hour:02d}_{resolution}.nc"
```

Notes:
- The loader skips non-existent files and continues. Use debug logs to identify missing files.
- Cache keys now include `granularity` and `data_path_format` to prevent collisions across different layouts.

## Dataset Server

The dataset server is a crucial component that:
1. Loads and preprocesses meteorological data
2. Caches processed data for faster access
3. Delivers data samples for training and inference
4. Handles normalization and augmentation

### Running the Dataset Server

```bash
wxb data-serve -m wxbtool.specs.res5_625.t850weyn -s Setting3d
```

Options:
- `-m/--module`: Specifies the model specification module
- `-s/--setting`: Specifies the setting class
- `-b/--bind`: Optional binding address (e.g., `127.0.0.1:8088`)
- `-w/--workers`: Number of worker processes
- Note: `--port` is currently not used by the implementation; prefer `--bind` to set the address/port or a Unix socket (e.g., `unix:/tmp/wxb.sock`).

### Client-Server Architecture

The dataset server can be run:
1. Locally with direct data access (default)
2. Remotely via HTTP server
3. On the same machine via Unix socket

This architecture allows:
- Separation of data processing from model training
- Efficient data caching
- Distributed computing capabilities

## Data Processing Pipeline

### 1. Loading

Raw NetCDF files are loaded using xarray:
- 3D variables include multiple vertical levels
- Time dimension is indexed to create sequences
- Data is converted to 32-bit floating point for efficient processing

### 2. Caching

Processed data is cached for efficiency:
- First run generates cache files in `.cache/` under `WXBHOME`
- Subsequent runs use cached data
- Cache invalidation occurs when settings change (now includes granularity and path format)

### 3. Windowing

wxbtool uses a windowing approach for sequence data:
- `input_span`: Number of timesteps in input sequence
- `pred_shift`: Number of hours between input end and prediction start
- `pred_span`: Number of timesteps in prediction sequence

For example, for 3-day forecasts with 8-hour steps:
- `input_span = 3` (3 input timesteps)
- `pred_shift = 72` (72 hours = 3 days)
- `pred_span = 1` (1 output timestep)

### 4. Normalization

All variables are normalized before use:
- Normalization methods are defined in the `norms` module
- Each variable has specific normalization functions
- Inverse transforms are applied during inference

### 5. Augmentation

Data augmentation is optionally applied:
- Configured in the model specification
- Can include spatial shifts, rotations, etc.
- Helps improve model generalization

## Dataset Classes

Two primary dataset classes are used:

### `WxDataset`

Directly loads data from disk:
- Used for local training and testing
- Handles caching and preprocessing
- Returns (inputs, targets) tuples
- Now supports flexible dataset organization via `granularity` and `data_path_format`

### `WxDatasetClient`

Fetches data from a dataset server:
- Used when the dataset server is running separately
- Communicates via HTTP or Unix socket
- Uses efficient msgpack serialization

## Example: Accessing a Dataset in Code

For custom processing, you can access the dataset directly:

```python
from wxbtool.data.dataset import WxDataset
from wxbtool.specs.res5_625.t850weyn import Setting3d

# Create a dataset with custom settings
dataset = WxDataset(
    resolution="5.625deg",
    years=[2015, 2016],
    vars=["geopotential", "temperature"],
    levels=["500", "850"],
    step=8,
    input_span=3,
    pred_shift=72,
    pred_span=1,
    # You could override layout via a custom Setting subclass if desired
)

# Access a specific data sample
inputs, targets, index = dataset[0]
```

## Downloading Data

wxbtool includes a data download command to fetch the latest ERA5 data:

```bash
wxb data-download -m wxbtool.zoo.unet.t850d3sm_weyn --coverage weekly
```

## Data Aggregation

For climate modeling (S2S), you often need to aggregate high-frequency data (e.g., hourly) into lower-frequency averages (e.g., daily or weekly). Use the `data-agg` command for this:

- **[Data Aggregation Guide](data_agg.md)**: Detailed instructions on creating sliding window datasets.

```bash
# Example: Create daily averages from hourly data
wxb data-agg -s SettingDailyAgg --src ./era5 --window 24 --align backward
```

Notes:
- Coverage options: `daily`, `weekly`, `monthly`, or an integer number of days (e.g., `--coverage 10`).
- Variables, levels, and resolution are inferred from the model's `Setting`.
  ```
  era5/{variable}/{YYYY}/{MM}/{DD}/{YYYYMMDD}_{HH}.nc
  ```
- First run: the tool will prompt for your ECMWF CDS API key and create `~/.cdsapirc` automatically.
- Ensure you have valid ECMWF CDS credentials for data access.

## Common Issues and Solutions

### Missing Data

If files are missing:
- Check the directory structure matches your configured `data_path_format`
- Ensure filenames follow the correct convention for your chosen granularity
- Verify all required variable directories exist

### Performance Issues

For performance optimization:
- Use Unix sockets for faster local communication
- Increase the number of worker processes
- Consider a dedicated machine for the dataset server

### Memory Issues

If experiencing memory problems:
- Reduce batch size
- Limit the number of variables and years loaded
- Run the dataset server as a separate process

## Next Steps

- Learn how to [train models](../training/overview.md) using this data
- Understand [model specifications](../../technical/specifications/overview.md)
- Explore [custom data processing](../../technical/dataset/data_classes.md)
