# Troubleshooting Guide

This guide provides solutions to common issues you might encounter when working with wxbtool.

## Installation Issues

### Issue: "ImportError: No module named wxbtool"

**Possible causes and solutions:**
- **Cause**: wxbtool is not installed or not in your Python path
- **Solution**: Verify installation with `pip list | grep wxbtool`
- **Solution**: Reinstall using `pip install wxbtool`
- **Solution**: If installed in development mode, ensure you're in the correct virtual environment

### Issue: "Environment variable WXBHOME not found"

**Possible causes and solutions:**
- **Cause**: The required WXBHOME environment variable is not set
- **Solution**: Set the variable as described in the [Installation Guide](installation.md)
- **Solution**: Verify the variable is set with `echo $WXBHOME` (Linux/macOS) or `echo %WXBHOME%` (Windows)

### Issue: CUDA-related errors

**Possible causes and solutions:**
- **Cause**: Incompatible CUDA version with PyTorch
- **Cause**: GPU drivers not installed properly
- **Solution**: Install the correct PyTorch version for your CUDA version using instructions from the PyTorch website
- **Solution**: Run with CPU only by setting `--gpu ""`

## Data Handling Issues

### Issue: "FileNotFoundError: No such file or directory"

**Possible causes and solutions:**
- **Cause**: Data files not present in the expected location
- **Cause**: Wrong directory structure
- **Solution**: Verify your data directory structure matches the expected pattern
- **Solution**: Check file permissions
- **Solution**: Use absolute paths rather than relative paths

### Issue: "No module named 'xarray'"

**Possible causes and solutions:**
- **Cause**: Missing dependencies
- **Solution**: Install required dependencies with `pip install xarray netCDF4`

### Issue: "Dataset server not responding"

**Possible causes and solutions:**
- **Cause**: Dataset server not running
- **Cause**: Server address incorrect
- **Solution**: Check if the dataset server is running with `ps aux | grep dserve`
- **Solution**: Verify the server address is correct
- **Solution**: Try using Unix socket instead of HTTP for local communication

## Training Issues

### Issue: "CUDA out of memory"

**Possible causes and solutions:**
- **Cause**: Batch size too large
- **Cause**: Model too complex for available GPU memory
- **Solution**: Reduce batch size with `-b` option
- **Solution**: Use a smaller model variant
- **Solution**: Distribute training across multiple GPUs

### Issue: "Loss is NaN"

**Possible causes and solutions:**
- **Cause**: Learning rate too high
- **Cause**: Input data not properly normalized
- **Cause**: Numerical instability in the model
- **Solution**: Reduce learning rate with `-r` option
- **Solution**: Check normalization functions in the specification
- **Solution**: Add gradient clipping to the model

### Issue: "Model not learning (loss not decreasing)"

**Possible causes and solutions:**
- **Cause**: Learning rate too low
- **Cause**: Problem with data preprocessing
- **Cause**: Issue with model architecture
- **Solution**: Increase learning rate
- **Solution**: Verify data normalization and preprocessing
- **Solution**: Check model architecture for issues

## Inference Issues

### Issue: "ValueError: incorrect datetime format"

**Possible causes and solutions:**
- **Cause**: Datetime format incorrect for the `-t` parameter
- **Solution**: Use the format `YYYY-MM-DDTHH:MM:SS` (e.g., `2023-01-01T00:00:00`)

### Issue: "No historical data available for specified date"

**Possible causes and solutions:**
- **Cause**: Missing data files for the required historical period
- **Solution**: Ensure your data directory contains data for the date range needed
- **Solution**: Download additional historical data if needed

### Issue: "Error creating output file"

**Possible causes and solutions:**
- **Cause**: No permission to write to output location
- **Cause**: Invalid file format specified
- **Solution**: Check permissions for the output directory
- **Solution**: Ensure output filename ends with supported extension (`.png` or `.nc`)

## Model and Performance Issues

### Issue: "Poor model performance (high RMSE)"

**Possible causes and solutions:**
- **Cause**: Insufficient training
- **Cause**: Model complexity not suitable for the task
- **Cause**: Data quality issues
- **Solution**: Train for more epochs
- **Solution**: Try a different model architecture
- **Solution**: Check data quality and preprocessing

### Issue: "Performance differs from paper results"

**Possible causes and solutions:**
- **Cause**: Different evaluation metrics
- **Cause**: Different test dataset
- **Cause**: Implementation differences
- **Solution**: Verify that evaluation metrics match the paper exactly
- **Solution**: Ensure test dataset matches the paper's description
- **Solution**: Check hyperparameters against the paper's specifications

## Configuration Issues

### Issue: "ModuleNotFoundError: No module named 'wxbtool.specs...'"

**Possible causes and solutions:**
- **Cause**: Incorrect module path
- **Cause**: Custom module not in Python path
- **Solution**: Check module path spelling
- **Solution**: Ensure custom modules are in the correct location
- **Solution**: Use absolute imports in custom modules

### Issue: "AttributeError: 'module' object has no attribute 'model'"

**Possible causes and solutions:**
- **Cause**: Module structure differs from what wxbtool expects
- **Solution**: Ensure your module defines `model` at the top level
- **Solution**: Check for syntax errors in your module

## System-Specific Issues

### Linux/macOS Issues

**Issue**: "Permission denied when running wxb command"
- **Solution**: Make sure wxb is executable: `chmod +x $(which wxb)`

### Windows Issues

**Issue**: "wxb is not recognized as an internal or external command"
- **Solution**: Ensure Python Scripts directory is in your PATH
- **Solution**: Use `python -m wxbtool.wxb` instead

## Reporting Bugs

If you encounter an issue not covered in this guide:

1. Check the GitHub repository issues page
2. Collect relevant information:
   - wxbtool version (`pip show wxbtool`)
   - Python version (`python --version`)
   - OS information
   - Full error traceback
   - Steps to reproduce
3. Create a new issue with this information

## Getting Help

For additional help:
- Post on the GitHub Discussions page
- Check the [technical documentation](../technical/index.md) for in-depth explanations
- Refer to the [examples directory](../examples/) for working examples
