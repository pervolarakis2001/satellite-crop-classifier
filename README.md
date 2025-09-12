# BDEO Exercise 2: Multispectral Satellite Time Series Classification

A machine learning project for classifying multispectral satellite time series data using PyTorch.

## Overview

This project implements a deep learning pipeline for crop classification using satellite time series data from Denmark. It combines a Pixel Set Encoder (PSE) with a Transformer model to achieve high classification accuracy.

## Architecture

- **Pixel Set Encoder**: Processes multispectral pixel data with MLP layers and pooling
- **Transformer with CLS token**: Handles temporal sequence modeling for time series classification
- **Cross-entropy loss**: Training with 5-fold cross-validation

## Performance

The model achieves excellent performance on the Danish crop classification dataset:
- **Accuracy**: 94.03% (±0.65%)
- **F1-score**: 94.04% (±0.66%)
- **Precision**: 94.03% (±0.65%)
- **Recall**: 94.03% (±0.65%)

## Setup

```bash
conda activate conda_env
pip install -r requirements.txt
```

## Usage

Run the complete training pipeline:
```bash
jupyter notebook pipeline.ipynb
```

## Project Structure

- `models/`: Neural network architectures (PSE, Transformer)
- `utils/`: Data loading and preprocessing utilities
- `pipeline.ipynb`: Main training and evaluation notebook
- `output_cross_entropy_model/`: Saved model checkpoints
