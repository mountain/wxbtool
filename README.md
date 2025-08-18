# wxbtool

[![DOI](https://zenodo.org/badge/269931312.svg)](https://zenodo.org/badge/latestdoi/269931312)

A toolkit for WeatherBench based on PyTorch

## Installation

```bash
pip install wxbtool
```

For detailed installation instructions, see the [Installation Guide](docs/user/installation.md).

## Quick Start

Preferred naming aliases are available:
- wxb forecast (aliases: infer, inferg) — unified deterministic or GAN forecasting
- wxb backtest (alias: eval) — operational-style backtesting
- wxb data-serve (alias: dserve) — dataset server
- wxb data-download (alias: download) — ERA5 downloader

### Start a data set server for 3-days prediction of t850 by Weyn's solution
```bash
wxb dserve -m wxbtool.specs.res5_625.t850weyn -s Setting3d
```

### Start a training process for a UNet model following Weyn's solution
```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

### Start a testing process for a UNet model following Weyn's solution
```bash
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

### Start an inference process for a UNet model following Weyn's solution
```bash
wxb infer -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o output.png
```
Note: For wxb infer, -t must be in YYYY-MM-DD (date only).

### Start a GAN inference process for a UNet model following Weyn's solution
```bash
wxb inferg -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01T00:00:00 -s 10 -o output.nc
```
Note: For wxb inferg, -t must be in YYYY-MM-DDTHH:MM:SS (date and time).

Equivalent using unified forecast command:
```bash
# Deterministic
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o output.png
# GAN ensemble
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01T00:00:00 -G true -s 10 -o output.nc
```

### Start a data set server with http binding (alias: wxb data-serve)
```bash
wxb dserve -m wxbtool.specs.res5_625.t850weyn -s Setting3d -b 0.0.0.0:8088
```
Note: Use --bind to specify the address. The --port option is currently not used by the implementation.

### Start a training process with unix socket binding
```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -d unix:/tmp/test.sock
```

### Start a backtesting (evaluation) process for a UNet model (alias: wxb backtest)
```bash
wxb eval -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o output.nc
```
This will write outputs under output/2023-01-01/ and, when using .nc, also create var_day_rmse.json containing day-by-day RMSE.

### Download recent ERA5 data based on the model setting (alias: wxb data-download)
```bash
wxb download -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn --coverage weekly
```

For more detailed examples and explanations, see the [Quick Start Guide](docs/user/quickstart.md).

## Documentation

### User Documentation
- [Installation Guide](docs/user/installation.md)
- [Quick Start Guide](docs/user/quickstart.md)
- [Data Handling Guide](docs/user/data_handling/overview.md)
- [Training Guide](docs/user/training/overview.md)
- [Evaluation Guide](docs/user/evaluation/overview.md)
- [Inference Guide](docs/user/inference/overview.md)
- [Troubleshooting Guide](docs/user/troubleshooting.md)

### Technical Documentation
- [Architecture Overview](docs/technical/architecture/overview.md)
- [Model Specifications](docs/technical/specifications/overview.md)
- [Creating Custom Models](docs/technical/extension/custom_models.md)

## How to use

See the comprehensive documentation in the [docs](docs) directory.

## How to release

```bash
uv build
uv publish

git tag va.b.c master
git push origin va.b.c
```

## Contributors

- Mingli Yuan ([Mountain](https://github.com/mountain))
- Ren Lu
