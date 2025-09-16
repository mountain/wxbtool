# wxbtool Documentation

Welcome to the official documentation for wxbtool, a PyTorch-based toolkit for WeatherBench.

[![DOI](https://zenodo.org/badge/269931312.svg)](https://zenodo.org/badge/latestdoi/269931312)

## About wxbtool

wxbtool is a powerful toolkit designed for meteorological modeling, specifically built on the WeatherBench framework. It provides a streamlined interface for training, testing, and deploying deep learning models for weather prediction tasks. Built on PyTorch, it offers both pre-configured models and the flexibility to create custom solutions.

## Documentation Sections

### For Users

- [Installation Guide](user/installation.md) - Instructions for installing and configuring wxbtool
- [Quick Start Guide](user/quickstart.md) - Get up and running quickly with basic commands
- [Data Handling](user/data_handling/overview.md) - Working with meteorological datasets
- [Training Models](user/training/overview.md) - Guide to training neural network models
- [Evaluation & Testing](user/evaluation/overview.md) - Evaluating model performance
- [Inference](user/inference/overview.md) - Using trained models for prediction
- [Troubleshooting](user/troubleshooting.md) - Solutions to common issues

### For Developers

- [Architecture Overview](technical/architecture/overview.md) - System design and component relationships
- [Core Modules](technical/modules/overview.md) - Detailed explanations of key modules
- [Model Specifications](technical/specifications/overview.md) - How model specs are structured
- [Dataset Implementation](technical/dataset/overview.md) - Technical details of dataset handling
- [Neural Network Models](technical/models/overview.md) - Implementation details of provided models
- [Extension Guide](technical/extension/overview.md) - How to extend wxbtool with custom components
- [Variable Registry](technical/extension/new_variables.md) - Official APIs to register variables (2D/3D), codes, and aliases
- [Normalization Systems](technical/specifications/normalization.md) - Normalization/denormalization registry APIs and usage
- [API Reference](technical/api/overview.md) - Comprehensive API documentation

### Examples & Resources

- [Example Workflows](examples/basic_workflow.md) - Complete workflow examples
- [Custom Model Example](examples/custom_model.md) - Creating your own models
- [Operational Forecasting](examples/operational_forecast.md) - Using wxbtool in operational settings

## Citation

If you use wxbtool in your research, please cite:

```bibtex
@software{wxbtool2023,
  author = {Yuan, Mingli and Lu, Ren},
  title = {wxbtool: A toolkit for WeatherBench based on PyTorch},
  url = {https://github.com/caiyunapp/wxbtool},
  version = {1.0.0},
  year = {2023},
}
```

## Contributors

- Mingli Yuan ([Mountain](https://github.com/mountain))
- Ren Lu
