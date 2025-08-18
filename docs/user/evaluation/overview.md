# Model Evaluation Guide

This guide explains how to test and evaluate trained wxbtool models to assess their performance.

## Prerequisites

Before evaluating a model, you need:
- A trained model or a pre-trained model checkpoint
- wxbtool properly installed and configured
- The `WXBHOME` environment variable set with appropriate test data
- A running dataset server (unless using direct data access)

## Basic Test Command

The basic command to evaluate a model is:

```bash
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

This command:
- Loads the model implementation from the specified module
- Evaluates it on the test dataset defined in the model's settings
- Reports performance metrics

## Test Options

You can customize the testing process with various options:

### Hardware Options

```bash
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -g 0 -c 8
```

- `-g/--gpu`: Specify which GPU to use (single value)
- `-c/--n_cpu`: Number of CPU threads for data loading

### Data and Batch Configuration

```bash
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -b 32 -d http://localhost:8088
```

- `-b/--batch_size`: Number of samples per batch
- `-d/--data`: URL of the dataset server or Unix socket path

### Model Loading Options

```bash
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -l custom_model.ckpt
```

- `-l/--load`: Path to a specific model checkpoint file

### GAN Testing

```bash
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -G true
```

- `-G/--gan`: Set to `true` to test a GAN model

### Additional Options

In addition to the options above, testing supports:

- `-O/--optimize`: Fast/CI mode. Limits validation/test batches and reduces patience to speed up continuous integration runs.
- `-t/--test`: Set to `true` to shorten runs (e.g., forces 1 epoch in some flows). Useful for smoke testing.

## Performance Metrics

wxbtool evaluates models using several performance metrics:

### Root Mean Square Error (RMSE)

The primary metric for deterministic forecasts, measuring the square root of the average squared differences between predicted and actual values.

```
RMSE = sqrt(mean((predicted - actual)²))
```

Lower RMSE values indicate better performance.

### Mean Absolute Error (MAE)

Measures the average magnitude of errors without considering their direction.

```
MAE = mean(|predicted - actual|)
```

### Anomaly Correlation Coefficient (ACC)

Measures the correlation between predicted and observed anomalies (deviations from climatology).

```
ACC = correlation(predicted - climatology, actual - climatology)
```

ACC ranges from -1 to 1, with values closer to 1 indicating better performance.

### Continuous Ranked Probability Score (CRPS)

For GAN models, CRPS evaluates probabilistic forecasts by comparing the predicted probability distribution to the observations.

Lower CRPS values indicate better ensemble performance.

## Spatial Analysis

wxbtool can evaluate performance across different spatial regions:

- **Global**: Overall performance across the entire domain
- **Northern Hemisphere**: Performance in the northern half of the domain
- **Southern Hemisphere**: Performance in the southern half of the domain
- **Tropics**: Performance in the tropical region

## Temporal Analysis

Performance can also be evaluated across different time scales:

- **Lead time analysis**: How performance changes with forecast lead time
- **Seasonal analysis**: Performance variations across seasons
- **Diurnal analysis**: Performance variations throughout the day

## Test Process Under the Hood

The testing process involves these steps:

1. **Model Loading**: The model is imported and initialized
2. **Dataset Connection**: Test dataset is accessed
3. **Batch Evaluation**: Model predictions are generated for test batches
4. **Metric Calculation**: Performance metrics are calculated
5. **Report Generation**: Results are displayed and optionally saved

## Example Testing Workflows

### Basic Testing Workflow

```bash
# Start the dataset server
wxb dserve -m wxbtool.specs.res5_625.t850weyn -s Setting3d &

# Test the model
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

### Testing Multiple Models

To compare different models, you can create a simple script:

```bash
#!/bin/bash
MODELS=(
  "wxbtool.zoo.res5_625.unet.t850d3sm_weyn"
  "wxbtool.zoo.res5_625.unet.t850d3bg_weyn"
  "wxbtool.zoo.res5_625.unet.t850d3hg_weyn"
)

echo "Model,RMSE,MAE,ACC" > model_comparison.csv

for model in "${MODELS[@]}"; do
  echo "Testing $model..."
  output=$(wxb test -m $model --optimize)
  rmse=$(echo "$output" | grep "RMSE" | awk '{print $2}')
  mae=$(echo "$output" | grep "MAE" | awk '{print $2}')
  acc=$(echo "$output" | grep "ACC" | awk '{print $2}')
  echo "$model,$rmse,$mae,$acc" >> model_comparison.csv
done

echo "Comparison complete. Results saved to model_comparison.csv"
```

### GAN Ensemble Evaluation

```bash
# Test a GAN model to evaluate ensemble performance
wxb test -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -G true
```

## Backtesting with wxb eval

Backtesting evaluates rolling, day-by-day performance starting from a specific initialization date. This is complementary to `wxb test` (which evaluates the full test split).

- Usage:
  ```bash
  wxb eval -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn -t 2025-01-01 -o forecast.nc
  ```
  - `-t/--datetime`: Initialization date in `YYYY-MM-DD` (date only)
  - `-o/--output`: `png` or `nc`

- Outputs:
  - Same PNG/NC artifacts as `wxb infer`
  - Additionally, when using `.nc`, writes `output/{YYYY-MM-DD}/var_day_rmse.json` containing day-by-day RMSE keyed by calendar date, for each variable in `vars_out`. Example:
    ```json
    {
      "t2m": {
        "2025-01-01": 2.31,
        "2025-01-02": 2.45,
        "2025-01-03": 2.62
      }
    }
    ```

- Positioning vs `wxb test`:
  - `wxb test`: Bulk evaluation on the test split
  - `wxb eval`: Operational-style backtest for one init date, computing forward-looking day-by-day scores

## Understanding Test Results

Test results include:

- **Metrics Overview**: Summary of overall performance
- **Detailed Breakdowns**: Performance by region and time
- **Confidence Intervals**: Statistical significance of results
- **Comparison to Baselines**: How the model compares to reference methods

## Visualizing Evaluation Results

You can visualize test results using standard Python libraries:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example test results (you would load these from your test output)
models = ["t850d3sm", "t850d3bg", "t850d3hg"]
rmse_values = [2.41, 2.38, 2.32]
mae_values = [1.85, 1.81, 1.79]

# Create a bar chart comparison
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, rmse_values, width, label='RMSE (K)')
rects2 = ax.bar(x + width/2, mae_values, width, label='MAE (K)')

ax.set_ylabel('Error (K)')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.savefig('model_comparison.png')
```

## Best Practices for Model Evaluation

### Comparison to Baselines

Always compare your model against:
- Climatology (long-term average)
- Persistence (using the previous state as the forecast)
- Operational numerical weather prediction (NWP) models

### Test Set Independence

Ensure your test dataset:
- Is completely separate from training data
- Covers a different time period
- Represents a realistic operational scenario

### Comprehensive Evaluation

Evaluate across multiple dimensions:
- Different variables (temperature, pressure, etc.)
- Various spatial regions
- Different time periods
- Multiple metrics

## Troubleshooting Evaluation Issues

### Common Problems and Solutions

1. **Poor Performance**
   - Check for overfitting during training
   - Verify normalization consistency
   - Ensure test data quality is good

2. **Testing Errors**
   - Check model compatibility with test data
   - Verify the dataset server is running correctly
   - Ensure GPU memory is sufficient for batch size

3. **Inconsistent Results**
   - Set a fixed random seed for reproducibility
   - Use the same test dataset for fair comparisons
   - Average results across multiple runs for stability

## Next Steps

- Learn how to [use your model for inference](../inference/overview.md)
- Understand how to [improve model performance](../../technical/models/optimization.md)
- Explore [model interpretation techniques](../../technical/models/interpretation.md)
