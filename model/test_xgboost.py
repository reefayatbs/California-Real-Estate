"""
XGBoost Implementation with Hyperparameter Tuning for California Real Estate Price Prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import logging
from src.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def tune_hyperparameters(X_train, y_train):
    """Perform grid search for hyperparameter tuning"""
    
    logger.info("Starting hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Initialize XGBoost regressor
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=2,
        n_jobs=-1
    )
    
    # Perform grid search
    grid_search.fit(X_train, y_train)
    
    # Log results
    logger.info("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        logger.info(f"{param}: {value}")
    
    logger.info(f"\nBest RMSE: ${-grid_search.best_score_:,.2f}")
    
    return grid_search.best_estimator_

def analyze_feature_importance(model, feature_names):
    """Analyze and display feature importance"""
    importance_dict = {name: score for name, score in zip(
        feature_names, 
        model.feature_importances_
    )}
    importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\nFeature Importance:")
    for feature, importance in importances:
        logger.info(f"{feature}: {importance:.4f}")

def main():
    try:
        # Load and preprocess data
        logger.info("Loading data...")
        data_processor = DataProcessor()
        df = data_processor.load_data('data/realtor-data.zip.csv')
        X, y = data_processor.preprocess_data(df)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Tune hyperparameters and get best model
        best_model = tune_hyperparameters(X_train, y_train)
        
        # Make predictions with best model
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Print final results
        logger.info("\nFinal XGBoost Results:")
        logger.info(f"Test Set R² Score: {r2:.3f}")
        logger.info(f"Test Set RMSE: ${rmse:,.2f}")
        
        # Analyze feature importance
        feature_names = data_processor.columns['feature_columns']
        analyze_feature_importance(best_model, feature_names)
        
        # Save best model
        best_model.save_model('artifacts/xgboost_model_tuned.json')
        logger.info("\nTuned model saved to artifacts/xgboost_model_tuned.json")
            
    except Exception as e:
        logger.error(f"Error during XGBoost training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 