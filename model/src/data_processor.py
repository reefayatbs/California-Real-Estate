import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict
import json
import os

class DataProcessor:
    """
    Handles data preprocessing for the California Real Estate Price Predictor.
    This class implements the same preprocessing steps as in the experiment notebook
    but in a modular, reusable way.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.columns = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and perform initial data cleaning."""
        df = pd.read_csv(file_path)
        
        # Drop irrelevant features (same as in notebook)
        columns_to_drop = [
            'brokered_by', 'status', 'acre_lot', 
            'street', 'zip_code', 'prev_sold_date'
        ]
        df = df.drop(columns_to_drop, axis=1)
        
        # Filter for California properties
        df = df[df['state'] == 'California']
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for model training.
        Returns features (X) and target (y).
        """
        # Handle missing values
        df = df.dropna()
        
        # Extract features and target
        y = df['price'].values
        
        # Encode categorical variables
        df_processed = df.copy()
        categorical_columns = ['city']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Select features for model
        feature_columns = ['bed', 'bath', 'house_size', 'city']
        X = df_processed[feature_columns].values
        
        # Store columns configuration
        self.columns = {
            'feature_columns': feature_columns,
            'categorical_columns': categorical_columns,
            'cities': self.label_encoders['city'].classes_.tolist()
        }
        
        return X, y
    
    def save_preprocessing_config(self, output_path: str):
        """Save preprocessing configuration for inference."""
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
            
        with open(output_path, 'w') as f:
            json.dump(self.columns, f)
    
    def load_preprocessing_config(self, config_path: str):
        """Load preprocessing configuration for inference."""
        with open(config_path, 'r') as f:
            self.columns = json.load(f)
    
    def transform_inference_data(self, data: Dict) -> np.ndarray:
        """Transform input data for model inference."""
        # Create a single-row dataframe
        df = pd.DataFrame([data])
        
        # Encode city
        city_encoder = self.label_encoders.get('city')
        if city_encoder is None:
            raise ValueError("Label encoder not initialized. Run preprocess_data first.")
            
        df['city'] = city_encoder.transform([data['city']])
        
        # Return features in correct order
        return df[self.columns['feature_columns']].values 