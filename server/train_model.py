import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pickle
import json
import os

# Create artifacts directory if it doesn't exist
os.makedirs('artifacts', exist_ok=True)

# Load and preprocess data
df = pd.read_csv('../data/realtor-data.zip.csv')

# Clean and prepare data
df = df[df['price'].notna()]
df = df[df['house_size'].notna()]
df = df[df['bed'].notna()]
df = df[df['bath'].notna()]
df = df[df['city'].notna()]

# Convert city to lowercase
df['city'] = df['city'].str.lower()

# Get top 10 cities by frequency
top_cities = df['city'].value_counts().nlargest(10).index.tolist()
df = df[df['city'].isin(top_cities)]

# Create dummy variables for cities
city_dummies = pd.get_dummies(df['city'], prefix='', prefix_sep='')

# Prepare feature columns
X = pd.concat([
    df[['bed', 'bath', 'house_size']],
    city_dummies
], axis=1)

y = df['price']

# Train XGBoost model with optimal parameters
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=5,
    gamma=0,
    random_state=42
)

model.fit(X, y)

# Save the model
with open('artifacts/California_RealEstate_model.pickle', 'wb') as f:
    pickle.dump(model, f)

# Save column information
columns = {
    'data_columns': ['bed', 'bath', 'house_size'] + top_cities
}

with open('artifacts/columns.json', 'w') as f:
    json.dump(columns, f)

print("Model trained and saved successfully!")
print("R² Score:", model.score(X, y)) 