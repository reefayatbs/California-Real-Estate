"""
Model Optimization Script for California Real Estate Price Predictor.
Compares different algorithms and performs hyperparameter tuning.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from src.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(model, X, y, cv=5):
    """Evaluate a model using cross-validation."""
    # Calculate cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    rmse_scores = np.sqrt(-cross_val_score(
        model, X, y, cv=cv, 
        scoring='neg_mean_squared_error'
    ))
    
    logger.info(f"\n{model.__class__.__name__}:")
    logger.info(f"R² Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    logger.info(f"RMSE: ${rmse_scores.mean():,.2f} (+/- ${rmse_scores.std() * 2:,.2f})")
    
    return {
        'model': model.__class__.__name__,
        'r2_mean': cv_scores.mean(),
        'r2_std': cv_scores.std(),
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std()
    }

def compare_models(X, y):
    """Compare different algorithms."""
    logger.info("Comparing different algorithms...")
    
    models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        ExtraTreesRegressor(n_estimators=100, random_state=42),
        LassoCV(cv=5, random_state=42),
        RidgeCV(cv=5),
        ElasticNetCV(cv=5, random_state=42)
    ]
    
    results = []
    for model in models:
        results.append(evaluate_model(model, X, y))
    
    return results

def tune_hyperparameters(X, y):
    """Perform grid search for hyperparameter tuning."""
    logger.info("\nPerforming hyperparameter tuning...")
    
    param_grids = {
        'RandomForestRegressor': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'GradientBoostingRegressor': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        },
        'ExtraTreesRegressor': {
            'model': ExtraTreesRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }
    
    best_models = {}
    best_score = 0
    best_overall_model = None
    
    for name, config in param_grids.items():
        logger.info(f"\nTuning {name}...")
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best R² score: {grid_search.best_score_:.3f}")
        
        best_models[name] = grid_search.best_estimator_
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_overall_model = grid_search.best_estimator_
    
    return best_overall_model, best_models

def analyze_feature_importance(model, feature_names):
    """Analyze and log feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        logger.info("\nFeature Importance:")
        for f in range(len(feature_names)):
            logger.info(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

def main():
    try:
        # Load and preprocess data
        logger.info("Loading data...")
        data_processor = DataProcessor()
        df = data_processor.load_data('data/realtor-data.zip.csv')
        X, y = data_processor.preprocess_data(df)
        
        # Compare different models
        results = compare_models(X, y)
        
        # Perform hyperparameter tuning
        best_model, all_best_models = tune_hyperparameters(X, y)
        
        # Analyze feature importance
        feature_names = data_processor.columns['feature_columns']
        analyze_feature_importance(best_model, feature_names)
        
        # Save the best model
        logger.info("\nSaving the best model...")
        model_path = 'artifacts/California_RealEstate_model_optimized.pickle'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        logger.info("Model optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 