# Quick Start Guide

This guide will help you quickly get started with wxbtool for weather prediction tasks.

## Prerequisites

Before you begin, make sure you have:
- wxbtool properly installed (see the [Installation Guide](installation.md))
- The `WXBHOME` environment variable set to your weather data directory
- Access to appropriate meteorological data in the required format

## Basic Workflow

The typical workflow with wxbtool consists of these steps:

1. Starting a dataset server
2. Training a model
3. Testing the model
4. Using the model for inference

## Command Cheat Sheet

Here are the essential commands to get started:

### Starting a Dataset Server

The dataset server prepares and serves data for training and testing. This is typically the first step:

```bash
wxb dserve -m wxbtool.specs.res5_625.t850weyn -s Setting3d
```

This command:
- Starts a data server for 3-day temperature at 850hPa prediction
- Uses Weyn's specification
- Uses the 3-day setting configuration

### Training a Model

To train a UNet model for temperature prediction:

```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

This command:
- Starts training the small UNet model
- Uses default hyperparameters
- Uses data from the local dataset server

### Testing a Model

To evaluate a trained model:

```bash
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

This command:
- Tests the model on the test dataset
- Calculates performance metrics
- Outputs evaluation results

### Running Inference

To generate a prediction for a specific date:

```bash
wxb infer -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01T00:00:00 -o output.png
```

This command:
- Loads the specified model
- Makes a prediction for January 1, 2023
- Saves the output as a PNG image

### GAN Inference

For probabilistic forecasts using a GAN:

```bash
wxb inferg -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01T00:00:00 -s 10 -o output.nc
```

This command:
- Uses the GAN version of the model
- Generates 10 ensemble members
- Saves the output as a NetCDF file

## Common Command Options

Here are some common options you can use with wxbtool commands:

### Dataset Server Options

- `-b` or `--bind`: Specify binding address (e.g., `-b 127.0.0.1:8088`)
- `-w` or `--workers`: Set number of worker processes (e.g., `-w 8`)
- `-m` or `--module`: Specify the model specification module
- `-s` or `--setting`: Specify the setting class within the module

### Training Options

- `-g` or `--gpu`: Specify GPU indices to use (e.g., `-g 0,1` for multi-GPU)
- `-b` or `--batch_size`: Set the batch size (e.g., `-b 32`)
- `-n` or `--n_epochs`: Set the number of training epochs (e.g., `-n 100`)
- `-l` or `--load`: Load a previously saved model (e.g., `-l model.ckpt`)
- `-r` or `--rate`: Set the learning rate (e.g., `-r 0.001`)
- `-d` or `--data`: Specify dataset server URL (e.g., `-d http://localhost:8088`)

### Testing Options

Similar to training options, with focus on model evaluation.

### Inference Options

- `-t` or `--datetime`: Specify the date and time for prediction
- `-o` or `--output`: Specify the output file (PNG or NC format)
- `-s` or `--samples`: For GAN inference, specify the number of samples to generate

## Example: Complete Workflow

Here's a complete example workflow:

```bash
# Start the dataset server in the background
wxb dserve -m wxbtool.specs.res5_625.t850weyn -s Setting3d &

# Train the model (this may take some time)
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -n 50 -b 32 -r 0.001

# Test the model
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn

# Run inference for a specific date
wxb infer -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2023-01-01T00:00:00 -o forecast.png
```

## Advanced Server Configurations

### HTTP Server with Custom Binding

```bash
wxb dserve -m wxbtool.specs.res5_625.t850weyn -s Setting3d -b 0.0.0.0:8088
```

This makes the server accessible from other machines on the network.

### Unix Socket Binding

```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -d unix:/tmp/wxb.sock
```

This uses a Unix socket instead of HTTP for communication.

## Next Steps

After becoming familiar with the basic workflow, you might want to:

1. Explore different model architectures in the `wxbtool.zoo` namespace
2. Experiment with different training hyperparameters
3. Create your own model specifications
4. Combine neural and physical models

For more detailed information on these topics, please refer to:

- [Data Handling Guide](data_handling/overview.md)
- [Training Guide](training/overview.md)
- [Model Evaluation Guide](evaluation/overview.md)
- [Technical Documentation](../technical/index.md)
