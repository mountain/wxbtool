# Installation Guide

This guide provides detailed instructions for installing and configuring wxbtool on your system.

## Prerequisites

Before installing wxbtool, ensure that you have the following prerequisites:

- Python 3.10 or higher
- pip (Python package installer)
- PyTorch 2.5 or higher
- CUDA-compatible GPU (recommended for training)

## Installation Methods

### Method 1: Install from PyPI (Recommended)

The simplest way to install wxbtool is through PyPI:

```bash
pip install wxbtool
```

### Method 2: Install from Source

For the latest development version or if you want to contribute to the project:

```bash
git clone https://github.com/caiyunapp/wxbtool.git
cd wxbtool
pip install -e .
```

## Environment Configuration

### Setting up WXBHOME

wxbtool requires the `WXBHOME` environment variable to be set, pointing to the directory where your meteorological data is stored.

#### For Linux/macOS:

Add the following line to your `.bashrc`, `.zshrc`, or equivalent shell configuration file:

```bash
export WXBHOME="/path/to/your/weather/data"
```

Then, reload your shell configuration:

```bash
source ~/.bashrc  # or ~/.zshrc
```

#### For Windows:

Set the environment variable through the System Properties:

1. Right-click on "Computer" and select "Properties"
2. Click on "Advanced system settings"
3. Click on "Environment Variables"
4. Under "System variables", click "New"
5. Set the Variable name as `WXBHOME` and the Variable value as the path to your weather data (e.g., `C:\WeatherData`)

### Data Directory Structure

Within your `WXBHOME` directory, wxbtool expects a specific structure:

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
├── 2m_temperature/
│   ├── 2m_temperature_1980_5.625deg.nc
│   ├── 2m_temperature_1981_5.625deg.nc
│   └── ...
└── toa_incident_solar_radiation/
    ├── toa_incident_solar_radiation_1980_5.625deg.nc
    ├── toa_incident_solar_radiation_1981_5.625deg.nc
    └── ...
```

Each variable directory contains NetCDF files with annual data at the specified resolution.

## Verifying the Installation

To verify that wxbtool has been installed correctly, run:

```bash
wxb help
```

This should display the help menu listing all available commands.

### Testing Data Access

To test if wxbtool can access your data correctly:

```bash
wxb data-serve -m wxbtool.specs.res5_625.t850weyn -s Setting3d --test true
```

If the command runs without errors, wxbtool is correctly set up to access your data.

## Dependencies

wxbtool has the following primary dependencies:

- PyTorch: Neural network framework
- xarray: Working with NetCDF files and labeled multi-dimensional arrays
- numpy: Numerical computing
- lightning.pytorch: PyTorch Lightning for simplified training
- msgpack and msgpack-numpy: Efficient serialization
- decouple: Configuration management

These dependencies should be automatically installed when you install wxbtool via pip.

## Troubleshooting

### Common Installation Issues

1. **ImportError: No module named wxbtool**
   - Ensure that you've installed wxbtool correctly
   - Check if your Python environment is activated (if using a virtual environment)

2. **Environment variable WXBHOME not found**
   - Verify that you've set the WXBHOME variable as described above
   - Check that the variable is accessible in your current shell session

3. **CUDA/GPU-related errors**
   - Ensure your PyTorch installation matches your CUDA version
   - If you don't have a compatible GPU, set `--gpu ""` when running commands

4. **Missing data errors**
   - Verify your data directory structure matches the expected format
   - Check file permissions for the data directory

For more troubleshooting help, see the [Troubleshooting Guide](troubleshooting.md).
