# California Real Estate Price Predictor - Model

This directory contains the machine learning model for predicting real estate prices in California.

## Directory Structure

```
model/
├── data/                      # Data directory
│   └── realtor-data.zip.csv  # Raw data file
├── experiments/
│   └── pricePredFinal.ipynb  # Jupyter notebook for experimentation
├── src/
│   ├── __init__.py
│   ├── data_processor.py     # Data preprocessing module
│   └── model_trainer.py      # Model training module
├── artifacts/                # Model artifacts
│   ├── California_RealEstate_model.pickle  # Trained model
│   └── columns.json         # Feature configuration
├── train.py                 # Main training script
└── README.md               # This file
```

## Components

### 1. Data Processor (`src/data_processor.py`)
Handles all data preprocessing tasks:
- Loading and cleaning data
- Feature engineering
- Categorical encoding
- Feature selection

### 2. Model Trainer (`src/model_trainer.py`)
Manages model training and evaluation:
- Data splitting
- Model training
- Performance evaluation
- Model persistence

### 3. Training Script (`train.py`)
Main script that orchestrates the training process:
- Loads and preprocesses data
- Trains the model
- Evaluates performance
- Saves model artifacts

### 4. Experiment Notebook (`experiments/pricePredFinal.ipynb`)
Jupyter notebook used for initial experimentation and model development.

## Usage

1. Ensure your data file is in the correct location:
```
model/data/realtor-data.zip.csv
```

2. Train the model:
```bash
cd model
python train.py
```

3. The script will:
   - Load and preprocess the data
   - Train the model
   - Print evaluation metrics
   - Save the model and preprocessing configuration

## Model Details

- Algorithm: Random Forest Regressor
- Features:
  - Number of bedrooms
  - Number of bathrooms
  - House size (square feet)
  - City (encoded)

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- jupyter (for experiments) 