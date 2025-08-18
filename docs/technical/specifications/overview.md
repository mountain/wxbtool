# Model Specifications Overview

This document explains the model specification system in wxbtool, which defines how models interact with meteorological data.

## Introduction to Model Specifications

In wxbtool, model specifications (or "specs") are a crucial component that defines:

1. How input data is selected, processed, and normalized
2. How prediction targets are processed and evaluated
3. The spatiotemporal configuration of the prediction task
4. Loss functions and evaluation metrics

Specifications separate the data handling logic from the model architecture, allowing you to:
- Use the same model architecture for different prediction tasks
- Ensure consistent data processing across training and inference
- Compare different models on the same task with standardized evaluation

## Specification Class Hierarchy

The specification system follows this class hierarchy:

```
Base2d (abstract)
  └── Spec (implementation for specific variables)
       └── ModelImplementation (concrete model)
```

### Setting Classes

Each specification has an associated Setting class that defines:

```
Setting (base)
  └── SettingWeyn (base with Weyn's configuration)
       ├── Setting3d (3-day forecast)
       ├── Setting5d (5-day forecast)
       └── SettingTest (test configuration with limited data)
```

## Key Components of a Specification

A complete specification includes:

### 1. Setting Class

Defines the basic configuration parameters:

```python
class Setting3d(SettingWeyn):
    def __init__(self):
        super().__init__()
        self.step = 8           # Time step in hours
        self.input_span = 3     # Number of time steps for input
        self.pred_span = 1      # Number of time steps for prediction
        self.pred_shift = 72    # Hours between input end and prediction start
```

Important setting attributes include:
- `resolution`: Spatial resolution (e.g., "5.625deg")
- `levels`: Vertical pressure levels to include
- `vars`: Meteorological variables to use
- `vars_in`: Input variable codes
- `vars_out`: Output variable codes
- `years_train`, `years_eval`, `years_test`: Temporal ranges for data

### 2. Spec Class

Implements the data processing logic:

```python
class Spec(Base2d):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "t850_weyn"  # Specification name

    def get_inputs(self, **kwargs):
        # Process and normalize input data
        z500 = norm_z500(kwargs["geopotential"][:, :, self.setting.levels.index("500")])
        # ... process other variables
        return {
            "z500": z500,
            # ... other variables
        }, concatenated_input

    def get_targets(self, **kwargs):
        # Process target data
        t850 = kwargs["temperature"][:, :, self.setting.levels.index("850")]
        return {"t850": t850}, t850

    def get_results(self, **kwargs):
        # Denormalize model output
        t850 = denorm_t850(kwargs["t850"])
        return {"t850": t850}, t850

    def lossfun(self, inputs, result, target):
        # Define the loss function
        losst = mse(rst[:, 0], tgt[:, 0])
        return losst
```

### 3. Model Implementation

Concrete model that uses the specification:

```python
class ResUNetModel(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "t850d3sm-weyn"
        self.resunet = resunet(
            # Neural network architecture parameters
        )

    def forward(self, **kwargs):
        # Model forward pass implementation
        _, input = self.get_inputs(**kwargs)
        constant = self.get_augmented_constant(input)
        input = th.cat((input, constant), dim=1)
        output = self.resunet(input)
        return {"t850": output}
```

## Available Specifications

wxbtool includes several pre-defined specifications:

### Temperature at 850hPa (t850)

- **t850weyn.py**: Follows Weyn's approach for temperature prediction
- **t850rasp.py**: Implementation based on Rasp's approach
- **t850recur.py**: Recurrent neural network approach

### Geopotential at 500hPa (z500)

- **z500weyn.py**: Follows Weyn's approach for geopotential prediction

## Data Flow Through Specifications

The specification defines the full data flow from raw NetCDF files to model predictions:

```
Raw NetCDF Files → Dataset Loading → Variable Selection → 
Normalization → Model Input → Model Forward Pass → 
Denormalization → Prediction Output
```

During this process:
1. Raw meteorological data is loaded from NetCDF files
2. Variables and levels are selected according to the setting
3. Selected variables are normalized using functions from the `norms` module
4. Data is formatted into tensors for model input
5. Model produces raw output
6. Output is denormalized to physical units
7. Results are returned for evaluation or visualization

## Normalization Systems

Each specification uses normalization functions to prepare data:

```python
from wxbtool.norms.meanstd import (
    norm_z500,        # Normalize Z500 using mean/std statistics
    norm_t850,        # Normalize T850 using mean/std statistics
    denorm_t850,      # Denormalize T850 back to physical units
)
```

Normalization is critical for:
- Ensuring numerical stability during training
- Making different variables comparable in magnitude
- Improving model convergence

## Creating a Custom Specification

To create a custom specification:

1. Create a new setting class inheriting from an appropriate base:

```python
class SettingCustom(Setting):
    def __init__(self):
        super().__init__()
        # Configure your settings
        self.resolution = "5.625deg"
        self.vars = ["temperature", "geopotential"]
        self.vars_in = ["t850", "z500"]
        self.vars_out = ["t850"]
        # Set temporal parameters
        self.step = 6
        self.input_span = 4
        self.pred_span = 1
        self.pred_shift = 48
```

2. Create a specification class:

```python
class SpecCustom(Base2d):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "custom_spec"
        
    def get_inputs(self, **kwargs):
        # Implement your input processing
        
    def get_targets(self, **kwargs):
        # Implement your target processing
        
    def get_results(self, **kwargs):
        # Implement your result processing
        
    def lossfun(self, inputs, result, target):
        # Implement your loss function
```

3. Implement your model using the specification:

```python
class CustomModel(SpecCustom):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "custom-model"
        # Define your neural network here
        
    def forward(self, **kwargs):
        # Implement your forward pass
```

4. Save these in appropriate module files following the wxbtool structure

## Using a Model Specification

To use a specification in the command line:

```bash
# For dataset server
wxb data-serve -m wxbtool.specs.res5_625.my_custom_spec -s SettingCustom

# For training
wxb train -m wxbtool.zoo.res5_625.my_custom_model
```

## Advanced Specification Features

### Data Augmentation

Specifications can include data augmentation:

```python
def augment_data(self, x):
    # Add random noise, rotations, or other transformations
    return augmented_x
```

### Multi-Variable Prediction

To predict multiple variables:

```python
def get_targets(self, **kwargs):
    t850 = kwargs["temperature"][:, :, self.setting.levels.index("850")]
    z500 = kwargs["geopotential"][:, :, self.setting.levels.index("500")]
    return {"t850": t850, "z500": z500}, th.cat((t850, z500), dim=1)
```

### Custom Loss Functions

To implement specialized loss functions:

```python
def lossfun(self, inputs, result, target):
    # Physical consistency loss
    physical_loss = self.calculate_physical_consistency(result)
    # Standard MSE loss
    data_loss = mse(result["t850"], target["t850"])
    # Combined loss
    return data_loss + 0.1 * physical_loss
```

## Best Practices for Specifications

1. **Consistent Normalization**: Ensure normalization and denormalization are exact inverses
2. **Clear Documentation**: Document the physical meaning of each variable
3. **Modularity**: Keep specifications modular so they can be reused
4. **Reproducibility**: Use fixed random seeds for any stochastic processes
5. **Validation**: Include validation logic to ensure data is properly formatted

## Summary

The specification system in wxbtool provides a structured way to define how models interact with meteorological data. By separating the data handling logic from model architecture, wxbtool enables consistent, reproducible experimentation with different models and prediction tasks.
