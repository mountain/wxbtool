# Technical Documentation

This section provides detailed technical information about wxbtool's architecture, components, and extension points. It's aimed at developers who want to understand the internal workings of wxbtool or extend its functionality.

## Core Architecture

- [Architecture Overview](architecture/overview.md) - High-level system design and component relationships
- [Data Flow](architecture/data_flow.md) - How data moves through the system
- [Client-Server Architecture](architecture/client_server.md) - Details on the dataset server implementation

## Key Components

- [Modules Overview](modules/overview.md) - Detailed explanation of core modules
- [Dataset Implementation](dataset/overview.md) - Technical details of dataset handling
- [Neural Network Models](models/overview.md) - Implementation details of models

## Model Specifications

- [Specifications Overview](specifications/overview.md) - How model specifications work
- [Normalization Systems](specifications/normalization.md) - Details on data normalization
- [Custom Specifications](specifications/custom_specs.md) - Creating your own specifications

## Extension Points

- [Creating Custom Models](extension/custom_models.md) - How to implement your own models
- [Adding New Variables](extension/new_variables.md) - Extending wxbtool with new meteorological variables
- [Contribution Guide](extension/contribution_guide.md) - Guidelines for contributing to wxbtool

## API Reference

- [Command Line Interface](api/wxb_command.md) - Complete CLI reference
- [Data API](api/data_api.md) - Dataset and data handling API
- [Model API](api/model_api.md) - Model and specification API
- [Utility API](api/utils_api.md) - Utility functions and helpers

## Internals

- [Caching System](internals/caching.md) - How data caching works
- [PyTorch Lightning Integration](internals/lightning_integration.md) - Details on training infrastructure
- [Testing Framework](internals/testing_framework.md) - How models are evaluated
