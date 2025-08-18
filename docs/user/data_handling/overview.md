# Data Handling Overview

This guide explains how wxbtool manages and processes meteorological data for weather prediction models.

## Data Requirements

wxbtool works with meteorological data in NetCDF format, organized by variable and year:

- Each variable has its own directory (e.g., `temperature/`, `geopotential/`)
- Files are named according to the pattern: `{variable}_{year}_{resolution}.nc`
- The default resolution is `5.625deg` (approximately 5.625° × 5.625° grid)

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

## Data Organization

### Directory Structure

The expected data structure in your `WXBHOME` directory is:

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

### Temporal Coverage

Models typically use data from specific year ranges:
- Training: 1980-2014 (default)
- Validation: 2015-2016 (default)
- Testing: 2017-2018 (default)

These ranges can be configured in your model's `Setting` class.

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
1. **Locally** with direct data access (default)
2. **Remotely** via HTTP server
3. **On the same machine** via Unix socket

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
- Cache invalidation occurs when settings change

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
)

# Access a specific data sample
inputs, targets, index = dataset[0]
```

## Downloading Data


wxbtool includes a data download command to fetch the latest ERA5 data:

```bash
wxb data-download -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn --coverage weekly
```

Notes:
- Coverage options: `daily`, `weekly`, `monthly`, or an integer number of days (e.g., `--coverage 10`).
- Variables, levels, and resolution are inferred from the model's `Setting`.
- Output directory structure:
  ```
  era5/{variable}/{YYYY}/{MM}/{DD}/{YYYYMMDD}_{HH}.nc
  ```
- First run: the tool will prompt for your ECMWF CDS API key and create `~/.cdsapirc` automatically.
- Ensure you have valid ECMWF CDS credentials for data access.

## Common Issues and Solutions

### Missing Data

If files are missing:
- Check the directory structure matches the expected pattern
- Ensure filenames follow the correct convention
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
