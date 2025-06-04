"""
Main training script for the California Real Estate Price Predictor.
This script uses the modular components to train and save the model.
"""

import os
import logging
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = os.path.join('data', 'realtor-data.zip.csv')
ARTIFACTS_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'California_RealEstate_model.pickle')
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, 'columns.json')

def main():
    # Initialize components
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    
    try:
        # Create artifacts directory if it doesn't exist
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        
        # Load and preprocess data
        logger.info("Loading data...")
        df = data_processor.load_data(DATA_PATH)
        
        logger.info("Preprocessing data...")
        X, y = data_processor.preprocess_data(df)
        
        # Split data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = model_trainer.train_test_split(X, y)
        
        # Train model
        logger.info("Training model...")
        model_trainer.train(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = model_trainer.evaluate(X_test, y_test)
        logger.info("Model performance:")
        logger.info(f"MSE: {metrics['mse']:.2f}")
        logger.info(f"RMSE: {metrics['rmse']:.2f}")
        logger.info(f"R2 Score: {metrics['r2']:.2f}")
        
        # Save artifacts
        logger.info("Saving model artifacts...")
        model_trainer.save_model(MODEL_PATH)
        data_processor.save_preprocessing_config(CONFIG_PATH)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 