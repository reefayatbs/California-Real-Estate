"""
Simple Linear Regression Test for California Real Estate Price Prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import logging
from src.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load and preprocess data
        logger.info("Loading data...")
        data_processor = DataProcessor()
        df = data_processor.load_data('data/realtor-data.zip.csv')
        X, y = data_processor.preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train linear regression model
        logger.info("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        rmse_scores = np.sqrt(-cross_val_score(
            model, X, y, cv=5, 
            scoring='neg_mean_squared_error'
        ))
        
        # Print results
        logger.info("\nLinear Regression Results:")
        logger.info(f"Test Set R² Score: {r2:.3f}")
        logger.info(f"Test Set RMSE: ${rmse:,.2f}")
        logger.info("\nCross-validation Results:")
        logger.info(f"R² Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        logger.info(f"RMSE: ${rmse_scores.mean():,.2f} (+/- ${rmse_scores.std() * 2:,.2f})")
        
        # Print feature coefficients
        feature_names = data_processor.columns['feature_columns']
        logger.info("\nFeature Coefficients:")
        for name, coef in zip(feature_names, model.coef_):
            logger.info(f"{name}: {coef:.4f}")
            
    except Exception as e:
        logger.error(f"Error during linear regression test: {str(e)}")
        raise

if __name__ == "__main__":
    main() 