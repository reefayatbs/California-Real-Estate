import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from typing import Tuple, Dict

class ModelTrainer:
    """
    Handles model training and evaluation for the California Real Estate Price Predictor.
    This class implements the same model training steps as in the experiment notebook
    but in a modular, reusable way.
    """
    
    def __init__(self, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state
        )
        self.random_state = random_state
    
    def train_test_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model."""
        self.model.fit(X_train, y_train)
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model and return performance metrics.
        """
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def save_model(self, output_path: str):
        """Save the trained model to disk."""
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
            
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f) 