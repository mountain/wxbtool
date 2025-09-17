# wxbtool

[![DOI](https://zenodo.org/badge/269931312.svg)](https://zenodo.org/badge/latestdoi/269931312)

A toolkit for WeatherBench based on PyTorch

## Installation

```bash
pip install wxbtool
```

For detailed installation instructions, see the [Installation Guide](docs/user/installation.md).

## Quick Start

### Start a data set server for 3-days prediction of t850 by Weyn's solution
```bash
wxb data-serve -m wxbtool.specs.res5_625.t850weyn -s Setting3d
```

### Start a training process for a UNet model following Weyn's solution
```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

### Start a testing process for a UNet model following Weyn's solution
```bash
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

### Start a forecast (deterministic) for a UNet model following Weyn's solution
```bash
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o output.png
```
Note: For deterministic forecast, -t must be in YYYY-MM-DD (date only).

### Start a GAN ensemble forecast for a UNet model following Weyn's solution
```bash
wxb forecast -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01T00:00:00 -G true -s 10 -o output.nc
```
Note: For GAN forecast, -t must be in YYYY-MM-DDTHH:MM:SS (date and time).

### Start a data set server with http binding
```bash
wxb data-serve -m wxbtool.specs.res5_625.t850weyn -s Setting3d -b 0.0.0.0:8088
```
Note: Use --bind to specify the address. The --port option is currently not used by the implementation.

### Start a training process with unix socket binding
```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -d unix:/tmp/test.sock
```

### Start a backtesting (evaluation) process for a UNet model
```bash
wxb backtest -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01 -o output.nc
```
This will write outputs under output/2023-01-01/ and, when using .nc, also create var_day_rmse.json containing day-by-day RMSE.

### Download recent ERA5 data based on the model setting
```bash
wxb data-download -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn --coverage weekly
```

For more detailed examples and explanations, see the [Quick Start Guide](docs/user/quickstart.md).

## Distributed Training (torchrun)

For multi-node or multi-process execution, use PyTorch's torchrun to launch one process per GPU. Under torchrun, -g/--gpu is ignored (device placement is controlled by LOCAL_RANK). Only rank 0 writes outputs in forecast/backtest to avoid file clobbering.

Example (single node, 4 GPUs):
```bash
torchrun --nproc_per_node=4 -m wxbtool.wxb train \
  -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn \
  --batch_size 64 --n_epochs 200 --rate 0.001
```

See detailed guidance in the Training Guide: docs/user/training/overview.md.

## Flexible Dataset Organization

wxbtool supports flexible dataset layouts beyond the default yearly files. You can configure how data files are discovered by setting two fields in your Setting:

- granularity: one of yearly, quarterly, monthly, weekly, daily, hourly
- data_path_format: a Python format string relative to the variable directory, supporting placeholders {var}, {resolution}, {year}, {month}, {day}, {hour}, {week}, {quarter}

Example for monthly files:
```python
from wxbtool.nn.setting import Setting

class MySetting(Setting):
    def __init__(self):
        super().__init__()
        self.granularity = "monthly"
        self.data_path_format = "{year}/{var}_{year}-{month:02d}_{resolution}.nc"
```

See details and more examples in the [Data Handling Guide](docs/user/data_handling/overview.md).

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
