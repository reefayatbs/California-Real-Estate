"""
Simple Neural Network Test for California Real Estate Price Prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from src.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealEstateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100):
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch.view(-1, 1)).item()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'artifacts/best_model.pth')

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - actuals.mean()) ** 2)
    
    return rmse, r2

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
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
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = RealEstateDataset(X_train, y_train)
        val_dataset = RealEstateDataset(X_val, y_val)
        test_dataset = RealEstateDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Initialize model
        input_dim = X.shape[1]
        model = SimpleNN(input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        logger.info("Training Neural Network...")
        train_model(model, train_loader, val_loader, criterion, optimizer, device)
        
        # Load best model and evaluate
        model.load_state_dict(torch.load('artifacts/best_model.pth'))
        rmse, r2 = evaluate_model(model, test_loader, device)
        
        # Print results
        logger.info("\nNeural Network Results:")
        logger.info(f"Test Set R² Score: {r2:.3f}")
        logger.info(f"Test Set RMSE: ${rmse:,.2f}")
            
    except Exception as e:
        logger.error(f"Error during neural network test: {str(e)}")
        raise

if __name__ == "__main__":
    main() 