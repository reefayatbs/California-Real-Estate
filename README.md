# California Real Estate Price Predictor API

A machine learning API that predicts real estate prices in California based on property features.

## Features

- Get a list of all available California cities
- Predict house prices based on:
  - Location (city)
  - House size (square feet)
  - Number of bedrooms
  - Number of bathrooms

## Project Structure

```
├── model/                  # Machine learning model and training
│   ├── experiments/       # Model experimentation notebooks
│   ├── src/              # Model source code
│   └── artifacts/        # Trained model and configurations
├── server/               # API server
│   ├── server.py        # Flask server implementation
│   └── util.py          # Utility functions
└── requirements.txt      # Python dependencies
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API Server

Start the prediction server:

```bash
cd server
python server.py
```

The server will start on `http://localhost:5001`.

## API Endpoints

### 1. Get Available Cities

```
GET /api/v1/cities
```

Returns a list of all available California cities.

Example response:
```json
{
  "status": "success",
  "cities": ["san francisco", "los angeles", "palo alto", ...]
}
```

### 2. Predict House Price

```
POST /api/v1/predict
```

Request body:
```json
{
  "city": "san francisco",
  "house_size": 2000,
  "bed": 3,
  "bath": 2
}
```

Example response:
```json
{
  "status": "success",
  "estimated_price": 1985804.60
}
```

## Model Information

The machine learning model uses XGBoost (eXtreme Gradient Boosting), a powerful gradient boosting framework optimized for efficiency and performance. The model is configured with the following parameters:

- Number of trees (n_estimators): 200
- Learning rate: 0.01
- Maximum tree depth: 5
- Minimum loss reduction (gamma): 0

XGBoost was chosen over linear regression, random forest, and neural network through cross validation.

