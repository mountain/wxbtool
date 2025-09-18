# Model Inference Guide

This guide explains how to use trained wxbtool models to generate weather predictions.

## Prerequisites

Before running inference, you need:
- A trained model or a pre-trained model file
- wxbtool properly installed and configured
- The `WXBHOME` environment variable set with appropriate data
- A dataset server running (for real-time data access)

## Inference Types

wxbtool supports two types of inference:

1. **Deterministic Inference**: Produces a single forecast
2. **Probabilistic Inference (GAN)**: Generates multiple ensemble members for uncertainty estimation

## Forecast Command

You can also use a unified forecast entrypoint that covers both deterministic and GAN ensemble forecasts:

```bash
# Deterministic forecast (date only)
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o forecast.png

# GAN ensemble forecast (date and time required)
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01T00:00:00 -G true -s 10 -o ensemble.nc
```

Time format:
- Deterministic forecast: use YYYY-MM-DD (date only)
- GAN forecast: use YYYY-MM-DDTHH:MM:SS (date and time)

## Basic Deterministic Forecast

The basic command to generate a deterministic forecast is:

```bash
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o forecast.png
```
Note: For deterministic forecast, -t must be in YYYY-MM-DD (date only).

This command:
- Loads the model implementation from the specified module
- Makes a prediction for January 1, 2023
- Saves the output as a PNG image

## Inference Options

You can customize the inference process with various options:

### Model and Hardware Options

```bash
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -g 0 -c 8
```

- `-m/--module`: Specifies the model module to load
- `-g/--gpu`: Specifies which GPU to use for inference
- `-c/--n_cpu`: Number of CPU threads for data processing

### Time Selection and Output Options

```bash
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o output.nc
```

- `-t/--datetime`: Initialization time for prediction
  - For deterministic forecast: use `YYYY-MM-DD` (date only)
  - For GAN forecast: use `YYYY-MM-DDTHH:MM:SS` (date and time)
- `-o/--output`: The output file path and format (supports .png or .nc)

### Data Source Configuration

```bash
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -d http://localhost:8088
```

- `-d/--data`: URL of the dataset server or Unix socket path

### Loading Custom Models

```bash
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -l custom_model.ckpt
```

- `-l/--load`: Path to a specific model checkpoint file

## GAN Inference

For probabilistic forecasting using a GAN model:

```bash
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01T00:00:00 -G true -s 10 -o ensemble.nc
```
Note: For GAN forecast, -t must be in YYYY-MM-DDTHH:MM:SS (date and time).

This command:
- Uses the GAN-based inference mode
- Generates 10 different ensemble members
- Saves the output as a NetCDF file for further analysis

### GAN-Specific Options

- `-s/--samples`: Number of ensemble members to generate (required)
- Other options are the same as standard inference

## Output Formats

wxbtool supports two output formats. By default, outputs are written under output/{YYYY-MM-DD}/ as {module_last}.{png|nc}:

### PNG Format

```bash
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o forecast.png
```

- Produces a visual representation of the forecast
- Automatically applies appropriate color maps
- Includes geographic boundaries and labels
- Useful for quick visualization and presentations

### NetCDF Format

```bash
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o forecast.nc
```

- Stores the raw numerical forecast data
- Preserves metadata and coordinates
- Can be used for further analysis or visualization in other tools
- Required for ensemble output from GAN inference

## Inference Process Under the Hood

The inference process involves these steps:

1. **Model Loading**: The model is imported from the specified module
2. **Data Retrieval**: Historical data is fetched for the input window
3. **Data Preprocessing**: Normalization and formatting of input data
4. **Forward Pass**: The model generates predictions
5. **Post-processing**: Denormalization and formatting of output data
6. **Output Generation**: Creation of PNG visualization or NetCDF file

## Example Inference Workflows

### Basic Inference Workflow

```bash
# Start the dataset server
wxb data-serve -m wxbtool.specs.res5_625.t850weyn -s Setting3d &

# Run inference for a specific date
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o forecast.png
```

### GAN Ensemble Workflow

```bash
# Start the dataset server
wxb data-serve -m wxbtool.specs.res5_625.t850weyn -s Setting3d &

# Generate an ensemble forecast
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01T00:00:00 -G true -s 20 -o ensemble.nc

# Analyze ensemble results with your preferred tool (e.g., Python with xarray)
```

### Batch Inference Workflow

For generating forecasts for multiple dates, you can create a simple script:

```bash
#!/bin/bash
DATES=("2023-01-01" "2023-01-02" "2023-01-03")
MODEL="wxbtool.zoo.res5_625.unet.t850d3sm_weyn"

for date in "${DATES[@]}"; do
    outfile="forecast_${date}.png"
    wxb forecast -m $MODEL -t "$date" -o "$outfile"
    echo "Generated forecast for $date"
done
```

## Visualizing Inference Results

### PNG Visualizations

PNG outputs include:
- Color-coded representation of the predicted variable
- Geographic boundaries
- Coordinate grid
- Prediction time label
- Color scale

### Working with NetCDF Files

For NetCDF output, you can use tools like:
- **xarray and matplotlib**: For custom Python visualizations
- **NCView**: For quick visual inspection
- **Panoply**: For advanced visualization and analysis

Example Python code to analyze an ensemble NetCDF file:

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load ensemble data
ds = xr.open_dataset('ensemble.nc')

# Calculate ensemble mean
ensemble_mean = ds.mean(dim='ensemble')

# Calculate ensemble spread (standard deviation)
ensemble_spread = ds.std(dim='ensemble')

# Plot mean and spread
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ensemble_mean.t850.plot(ax=ax1, cmap='viridis')
ax1.set_title('Ensemble Mean')
ensemble_spread.t850.plot(ax=ax2, cmap='Reds')
ax2.set_title('Ensemble Spread')
plt.tight_layout()
plt.savefig('ensemble_analysis.png')
```

## Using a Custom Date Range for Inference

The model needs historical data leading up to the forecast start time. The specific input window depends on the model's configuration, which is defined in its settings class.

For a model with `input_span = 3` and 8-hour steps, you need data from:
- t-16h
- t-8h
- t (forecast start time)

Make sure your dataset includes the necessary historical data.

## Artifacts and Plotting

- The -p/--plot toggle applies to training and testing flows (wxb train, wxb test). When enabled, image artifacts (inputs, forecast vs target, ACC anomalies) are produced and saved via a universal logging callback under logger.log_dir/plots (fallback to ./plots).
- In distributed runs (torchrun), only global rank 0 writes artifacts to avoid file clobbering.
- Forecast (wxb forecast) produces PNG/NetCDF outputs as requested; backtesting (wxb backtest) additionally writes output/{YYYY-MM-DD}/var_day_rmse.json with day-by-day RMSE keyed by calendar date when using .nc output.

## Distributed Inference (torchrun)

You can accelerate inference/backtesting across multiple GPUs/nodes using torchrun. Under torchrun, -g/--gpu is ignored (device placement is controlled by LOCAL_RANK), and only rank 0 writes outputs to avoid file clobbering.

Example (single node, 4 GPUs):
```bash
torchrun --nproc_per_node=4 -m wxbtool.wxb forecast \
  -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn \
  -t 2023-01-01 -o forecast.nc
```

Notes:
- Do not pass -g/--gpu under torchrun; one process is launched per GPU.
- Outputs are written only by global rank 0; logs/metrics are aggregated by Lightning.

## Troubleshooting Inference Issues

### Common Problems and Solutions

1. **Missing Data Errors**
   - Ensure your data directory contains the required historical data
   - Check the datetime format:
     - For deterministic forecast: use `YYYY-MM-DD` (date only)
     - For GAN forecast: use `YYYY-MM-DDTHH:MM:SS` (date and time)
   - Verify the dataset server is running properly

2. **Model Loading Issues**
   - Check that the model module path is correct
   - Verify the checkpoint file exists (if specified)
   - Ensure compatibility between the model specification and available data

3. **Output Problems**
   - For PNG issues, check file permissions and directory existence
   - For NetCDF issues, ensure you have write access to the output directory
   - For visualization problems, verify that the variable is being correctly denormalized

## Next Steps

- Learn how to [create custom models](../../technical/extension/custom_models.md)
- Understand how to [evaluate model performance](../evaluation/overview.md)
- Explore [combining neural and physical models](../../examples/hybrid_models.md)
