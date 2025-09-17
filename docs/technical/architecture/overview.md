# Architecture Overview

This document provides a high-level overview of the wxbtool architecture, explaining how the different components interact to facilitate weather prediction using deep learning.

## System Architecture

wxbtool is designed with a modular architecture to support different stages of the weather prediction workflow, from data preparation to model inference. The system is built on PyTorch and follows a client-server architecture for data handling.

### Core Components

```
wxbtool/
├── data/        # Data handling components
├── nn/          # Neural network components
├── specs/       # Model specifications
├── zoo/         # Pre-built model implementations
├── norms/       # Normalization utilities
├── phys/        # Physical models and constraints
└── util/        # Utility functions
```

## Component Relationships

The following diagram illustrates the relationships between the major components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Module   │────▶│  Model Specs    │────▶│  Neural Models  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                        │                       │
        │                        │                       │
        ▼                        ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Dataset Server  │────▶│ Training Module │────▶│ Inference Module│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Data Flow Architecture

1. **Data Preparation**:
   - Raw meteorological data (NetCDF files) are processed by the `data` module
   - Normalization is applied using components from the `norms` module
   - A dataset server prepares and serves data for training and inference

2. **Model Definition**:
   - Model specifications (`specs`) define the inputs, outputs, and transformation logic
   - Actual model architectures are implemented in the `zoo` module
   - Physical constraints can be incorporated using the `phys` module

3. **Training and Inference**:
   - The `nn` module handles model training, testing, and inference
   - PyTorch Lightning is used for standardized training procedures
   - Results can be visualized using utilities in the `util` module

## Client-Server Architecture for Data

wxbtool employs a client-server architecture for data handling:

```
┌───────────────┐          ┌───────────────┐
│  Data Files   │──────────▶               │
│   (NetCDF)    │          │    Dataset    │
└───────────────┘          │    Server     │
                           │               │◀────────────────┐
┌───────────────┐          └───────────────┘                 │
│ Configuration │──────────▶      │                          │
└───────────────┘                 │                          │
                                  ▼                          │
                           ┌───────────────┐          ┌──────────────┐
                           │  HTTP/Unix    │          │    Model     │
                           │    Server     │◀─────────▶   Training   │
                           └───────────────┘          └──────────────┘
```

This architecture provides several advantages:
- Decoupling of data preparation from model training
- Efficient data caching and preprocessing
- Ability to run data server and training on separate machines
- Support for both HTTP and Unix socket communication

## Command Line Interface

The command-line interface (`wxb.py`) ties all components together, providing commands for:
- Dataset server management (`data-serve`)
- Data download (`data-download`)
- Model training (`train`)
- Model testing (`test`)
- Forecasting (`forecast`) – deterministic and GAN ensemble (via `-G/--gan` and `-s/--samples`)
- Backtesting (`backtest`) – operational day-by-day evaluation

Device & distributed configuration is centralized in `wxbtool/nn/config.py`:
- `add_device_arguments(parser)` – injects shared CLI flags (e.g., `--num_nodes`)
- `configure_trainer(opt, ...)` – constructs a Lightning `Trainer` with proper accelerator/devices/strategy
- `get_runtime_device()`, `detect_torchrun()`, `is_rank_zero()` – helpers for imperative inference/backtesting

torchrun usage (multi-node/process):
- `-g/--gpu` is ignored under torchrun; placement is controlled by `LOCAL_RANK`
- one process per GPU; only global rank 0 writes forecast/backtest outputs to avoid clobbering

## Model Specification System

The model specification system is a key architectural feature:

1. **Base Setting Classes**:
   - Define the resolution, variables, levels, and temporal scope
   - Configure input/output spans and prediction shifts

2. **Model Specifications**:
   - Inherit from base classes (e.g., `Base2d`)
   - Define data transformations and normalization
   - Implement loss functions and evaluation metrics

3. **Model Implementations**:
   - Implement the actual PyTorch models
   - Utilize the specification for standardized input/output handling

## Normalization Framework

The normalization framework:
- Provides consistent input/output transformations
- Implements domain-specific normalization techniques
- Ensures models train on normalized data but output in physical units

## Extension Architecture

wxbtool is designed to be extensible:
- New models can be added to the `zoo` module
- New variables and physical processes can be added to the `phys` module
- Custom specifications can be created by inheriting from base classes

## Dependencies and External Libraries

wxbtool leverages several external libraries:
- PyTorch/PyTorch Lightning: Neural network framework
- xarray: NetCDF file handling
- numpy: Numerical computing
- msgpack: Efficient serialization for client-server communication

## Summary

The modular architecture of wxbtool facilitates:
1. **Separation of concerns**: Data handling, model specification, and training are decoupled
2. **Extensibility**: New models and specifications can be easily added
3. **Reproducibility**: Standardized training and evaluation procedures
4. **Efficiency**: Client-server architecture for optimized data handling

This architecture allows for rapid experimentation with different model architectures while maintaining a consistent interface for training, testing, and deployment.
