# Training Models with wxbtool

This guide covers the process of training weather prediction models using wxbtool's training framework.

## Prerequisites

Before training a model, you need:

- wxbtool [properly installed](../installation.md)
- The `WXBHOME` environment variable set with appropriate meteorological data
- A running dataset server (unless you're using direct data access)
- A CUDA-compatible GPU for efficient training (optional but recommended)

## Training Process Overview

The training process in wxbtool follows these steps:

1. **Data Preparation**: Set up a dataset server to provide training data
2. **Model Selection**: Choose a model architecture from the `wxbtool.zoo` namespace
3. **Training Configuration**: Set hyperparameters and training options
4. **Model Training**: Run the training process with the selected configuration
5. **Model Evaluation**: Evaluate the trained model on validation data

## Basic Training Command

The basic command to train a model is:

```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

This command:
- Loads the UNet model implementation from the specified module
- Uses default hyperparameters
- Connects to a dataset server running on the local machine
- Trains the model for the default number of epochs

## Training Options

You can customize the training process with various options:

### Hardware Options

```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -g 0,1 -c 16
```

- `-g/--gpu`: Specify which GPUs to use (comma-separated indices)
- `-c/--n_cpu`: Number of CPU threads for data loading

### Training Hyperparameters

```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -b 32 -n 100 -r 0.0005
```

- `-b/--batch_size`: Number of samples per batch
- `-n/--n_epochs`: Total number of training epochs
- `-r/--rate`: Learning rate
- `-w/--weightdecay`: Weight decay for regularization

### Data Source Configuration

```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -d http://localhost:8088
```

- `-d/--data`: URL of the dataset server or Unix socket path

### Resuming Training

```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -e 50 -l model.ckpt
```

- `-e/--epoch`: Current epoch to start from
- `-l/--load`: Path to a saved model checkpoint
- `-k/--check`: Path to a PyTorch Lightning checkpoint

## Training GANs

wxbtool supports training Generative Adversarial Networks (GANs) for probabilistic forecasting:

```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -G true -R 10 -A 0.1
```

- `-G/--gan`: Set to `true` to enable GAN training
- `-R/--ratio`: Ratio between generator and discriminator learning rates
- `-A/--alpha`: Weight for loss calculation in GAN training
- `-B/--balance`: Exit balance for GAN training
- `-T/--tolerance`: Exit balance tolerance for GAN training

## PyTorch Lightning Integration

wxbtool uses PyTorch Lightning for standardized training:

- **LightningModel**: Wrapper for standard models
- **GANModel**: Wrapper for GAN models
- **EarlyStopping**: Automatically stops training when validation loss stops improving
- **DDP Strategy**: Distributed Data Parallel for multi-GPU training

## Training Process Under the Hood

The training process involves these steps:

1. **Model Loading**: The model is imported from the specified module
2. **Dataset Connection**: Connects to the dataset server to fetch training/validation data
3. **Trainer Setup**: Configures the PyTorch Lightning trainer
4. **Training Loop**: Runs the training loop with automatic validation
5. **Checkpointing**: Periodically saves model checkpoints
6. **Model Saving**: Saves the final model after training

## Model Checkpointing

During training, models are automatically checkpointed:

- Early stopping monitors validation loss and saves the best model
- The best model is saved as `{model_name}.ckpt`
- Use the `-l/--load` option to resume training from a checkpoint

## Multi-GPU Training

For faster training on systems with multiple GPUs:

```bash
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -g 0,1,2,3
```

This distributes the training across all specified GPUs using PyTorch's Distributed Data Parallel (DDP).

## Optimizing Training Performance

To get the best training performance:

1. **Batch Size**: Use the largest batch size that fits in GPU memory
2. **Number of Workers**: Set `-c` to approximately 4-8 times the number of GPUs
3. **Learning Rate**: Start with the default learning rate and adjust as needed
4. **Data Connection**: Use Unix sockets for faster data transfer when possible
5. **GPU Selection**: Choose the GPUs with the most memory and fastest interconnect

## Example Training Workflows

### Basic Training Workflow

```bash
# Start the dataset server
wxb data-serve -m wxbtool.specs.res5_625.t850weyn -s Setting3d &

# Train the model
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -b 32 -n 100

# Test the model after training
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

### GAN Training Workflow

```bash
# Start the dataset server
wxb data-serve -m wxbtool.specs.res5_625.t850weyn -s Setting3d &

# Train the GAN model
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -G true -n 100 -b 16 -r 0.0002 -R 5

# Test the GAN model
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -G true
```

### Distributed Training Workflow

```bash
# Start the dataset server
wxb data-serve -m wxbtool.specs.res5_625.t850weyn -s Setting3d -b 0.0.0.0:8088 &

# Train on multiple machines (machine 1)
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -d http://server-ip:8088 -g 0,1,2,3

# Train on multiple machines (machine 2)
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -d http://server-ip:8088 -g 0,1,2,3
```

## Monitoring Training Progress

During training, wxbtool outputs:

- Loss values for each epoch
- Validation metrics
- Training speed (samples per second)
- Checkpoint information
- Early stopping status

## Troubleshooting Training Issues

### Common Problems and Solutions

1. **Out of Memory Errors**
   - Reduce batch size
   - Use fewer input variables or a smaller model
   - Train on multiple GPUs

2. **Slow Training**
   - Check dataset server performance
   - Increase number of workers
   - Use a faster connection method (Unix sockets)

3. **Poor Convergence**
   - Adjust learning rate
   - Check normalization of inputs
   - Inspect data for quality issues
   - Try a different model architecture

## Next Steps

- Learn how to [test and evaluate](../evaluation/overview.md) your trained models
- Understand [inference procedures](../inference/overview.md) for making predictions
- Explore [creating custom models](../../technical/extension/custom_models.md)
