# Creating Custom Models

This guide explains how to extend wxbtool with your own custom models for weather prediction tasks.

## Overview

wxbtool is designed to be extensible, allowing you to create custom models while leveraging the existing data handling, training, and evaluation infrastructure. Creating a custom model typically involves:

1. Creating or selecting an appropriate specification
2. Implementing your model architecture
3. Integrating your model with the wxbtool architecture
4. Testing and using your model

## Prerequisites

Before creating a custom model, you should be familiar with:

- Basic wxbtool concepts and architecture
- The [model specification system](../specifications/overview.md)
- PyTorch neural network programming
- Meteorological variables and their relationships

## Step 1: Selecting a Specification

Your model needs a specification that defines:
- Input and output variables
- Data normalization
- Loss functions
- Evaluation metrics

You can either:
- Use an existing specification (e.g., `wxbtool.specs.res5_625.t850weyn`)
- Create a custom specification for your specific needs

For example, to use an existing specification:

```python
from wxbtool.specs.res5_625.t850weyn import Spec, Setting3d
```

## Step 2: Implementing Your Model

Create a new Python file in an appropriate location, such as `wxbtool/zoo/res5_625/custom/my_model.py`. Your model should inherit from the chosen specification and implement the required methods:

```python
# wxbtool/zoo/res5_625/custom/my_model.py
import torch as th
import torch.nn as nn

from wxbtool.specs.res5_625.t850weyn import Spec, Setting3d

class MyCustomModel(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "my-custom-model"
        
        # Define your neural network architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(setting.input_span * (len(setting.vars) + 2) + self.constant_size + 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, **kwargs):
        batch_size = kwargs["temperature"].size()[0]
        self.update_da_status(batch_size)

        # Get inputs using the specification's methods
        _, input = self.get_inputs(**kwargs)
        
        # Get constants (like latitude, longitude info)
        constant = self.get_augmented_constant(input)
        
        # Combine inputs and constants
        input = th.cat((input, constant), dim=1)
        
        # Add padding (following the pattern in other models)
        input = th.cat((input[:, :, :, 63:64], input, input[:, :, :, 0:1]), dim=3)
        
        # Forward pass through your network
        x = self.encoder(input)
        output = self.decoder(x)
        
        # Remove padding
        output = output[:, :, :, 1:65]
        
        # Return results in the expected format
        return {"t850": output}

# Create setting and model instances
setting = Setting3d()
model = MyCustomModel(setting)
```

### Key Components to Implement

1. **`__init__` method**: Define your model architecture and parameters
2. **`forward` method**: Implement the forward pass
3. **Network architecture**: Design your neural network layers
4. **Model name**: Set a unique name for your model

## Step 3: Advanced Model Architectures

For more complex models, you can leverage PyTorch's capabilities:

### Using Pre-trained Models

```python
def __init__(self, setting):
    super().__init__(setting)
    self.name = "pretrained-model"
    
    # Load a pre-trained ResNet backbone
    resnet = models.resnet18(pretrained=True)
    self.backbone = nn.Sequential(*list(resnet.children())[:-2])
    
    # Custom decoder
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
    )
```

### Implementing GAN Models

For GAN models, you need to create both a generator and discriminator:

```python
# Generator model (in one file)
class Generator(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "my-gan-generator"
        # Implement generator architecture
        
    def forward(self, **kwargs):
        # Implement generator forward pass
        return {"t850": output}

# Discriminator model (in the same file)
class Discriminator(nn.Module):
    def __init__(self, setting):
        super().__init__()
        # Implement discriminator architecture
        
    def forward(self, x):
        # Implement discriminator forward pass
        return validity

# Create instances
setting = Setting3d()
generator = Generator(setting)
discriminator = Discriminator(setting)
```

## Step 4: Testing Your Model

Test your model implementation before full training:

```python
import torch as th
from wxbtool.data.dataset import WxDataset

# Create a small dataset
dataset = WxDataset(
    resolution="5.625deg",
    years=[2015],
    vars=["geopotential", "temperature", "2m_temperature", "toa_incident_solar_radiation"],
    levels=["300", "500", "700", "850", "1000"],
    step=8,
    input_span=3,
    pred_shift=72,
    pred_span=1,
)

# Get a sample
inputs, targets, _ = dataset[0]

# Create model
from wxbtool.zoo.res5_625.custom.my_model import model

# Test forward pass
with th.no_grad():
    output = model(**inputs)
    print(f"Output shape: {output['t850'].shape}")
```

## Step 5: Training Your Custom Model

Train your model using the wxbtool CLI:

```bash
wxb train -m wxbtool.zoo.res5_625.custom.my_model
```

This leverages wxbtool's training infrastructure, including PyTorch Lightning integration, dataset handling, and checkpointing.

## Step 6: Inference with Your Model

Use your trained model for inference:

```bash
wxb forecast -m wxbtool.zoo.res5_625.custom.my_model -t 2023-01-01 -o forecast.png
```

## Advanced: Physics-Informed Neural Networks

You can incorporate physical constraints into your models:

```python
def lossfun(self, inputs, result, target):
    # Standard data loss
    data_loss = mse(result["t850"], target["t850"])
    
    # Physics-based constraint (e.g., thermal wind balance)
    physics_loss = self.compute_thermal_wind_violation(result, inputs)
    
    # Combined loss
    return data_loss + 0.1 * physics_loss
    
def compute_thermal_wind_violation(self, result, inputs):
    # Implement a physical constraint based on meteorological principles
    # This is just a placeholder - implement actual physical equations
    return th.mean(th.abs(result["t850"][:, :, 1:, :] - result["t850"][:, :, :-1, :]))
```

## Best Practices for Custom Models

1. **Start Simple**: Begin with a simpler architecture and gradually add complexity
2. **Test Incrementally**: Validate each component separately
3. **Document Thoroughly**: Add clear documentation about model architecture and expected inputs/outputs
4. **Compare with Baselines**: Benchmark against existing models
5. **Version Control**: Track model changes using git

## Troubleshooting Common Issues

### Input/Output Dimension Mismatch

If you encounter tensor dimension errors:
- Check the input and output shapes expected by your specification
- Ensure convolutional layers maintain appropriate dimensions
- Verify padding and kernel sizes

### Memory Issues

If you run into memory problems:
- Reduce batch size
- Simplify model architecture
- Use gradient checkpointing

### Convergence Problems

If your model doesn't learn effectively:
- Check normalization in the specification
- Verify loss function implementation
- Adjust learning rate and optimizer parameters
- Inspect input data for anomalies

## Examples of Custom Model Architectures

### 1. Convolutional LSTM for Sequence Prediction

```python
import torch as th
import torch.nn as nn
from wxbtool.specs.res5_625.t850weyn import Spec, Setting3d

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = th.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = th.split(combined_conv, self.hidden_dim, dim=1)
        i = th.sigmoid(cc_i)
        f = th.sigmoid(cc_f)
        o = th.sigmoid(cc_o)
        g = th.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * th.tanh(c_next)
        return h_next, c_next

class ConvLSTMModel(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "convlstm-weather"
        input_channels = setting.input_span * (len(setting.vars) + 2) + self.constant_size
        self.convlstm_cell = ConvLSTMCell(
            input_dim=input_channels,
            hidden_dim=64,
            kernel_size=3,
            bias=True
        )
        self.conv_output = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, **kwargs):
        batch_size = kwargs["temperature"].size()[0]
        self.update_da_status(batch_size)
        
        _, input = self.get_inputs(**kwargs)
        constant = self.get_augmented_constant(input)
        input = th.cat((input, constant), dim=1)
        
        # Initialize hidden state
        h, c = (
            th.zeros(batch_size, 64, 32, 64).to(input.device),
            th.zeros(batch_size, 64, 32, 64).to(input.device)
        )
        
        # Process sequence
        for t in range(3):  # Assuming 3 timesteps in the input
            h, c = self.convlstm_cell(input, (h, c))
            
        # Final output projection
        output = self.conv_output(h)
        
        return {"t850": output}

setting = Setting3d()
model = ConvLSTMModel(setting)
```

### 2. UNet with Attention

```python
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map

class AttentionUNet(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = "attention-unet-weather"
        # Implementation details...
```

## Conclusion

Creating custom models in wxbtool allows you to leverage the framework's data handling and training infrastructure while implementing your own model architectures. By following this guide, you can create, train, and deploy custom models for weather prediction tasks.

Remember that successful weather prediction models often require careful consideration of the physical processes involved, appropriate data preprocessing, and thorough validation.
