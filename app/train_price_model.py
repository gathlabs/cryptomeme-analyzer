import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(PricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Progress bar untuk epochs
    pbar = tqdm(range(num_epochs), desc='Training Progress')
    
    for epoch in pbar:
        model.train()
        train_loss = 0
        # Progress bar untuk batch training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        for X_batch, y_batch in train_pbar:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        
        model.eval()
        val_loss = 0
        # Progress bar untuk validation
        val_pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for X_batch, y_batch in val_pbar:
                y_pred = model(X_batch)
                batch_loss = criterion(y_pred, y_batch.reshape(-1, 1)).item()
                val_loss += batch_loss
                val_pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})
        
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update main progress bar dengan losses
        pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}'
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
    
    return train_losses, val_losses, best_model_state

def main():
    # Parameters
    seq_length = 30
    train_split = 0.8
    batch_size = 32
    num_epochs = 100
    
    print("\n=== Data Preparation ===")
    # Get data from 2016
    print("Fetching BTC-USD data from 2016...")
    crypto = yf.Ticker('BTC-USD')
    df = crypto.history(start='2016-01-01')
    prices = df['Close'].values.reshape(-1, 1)
    dates = df.index
    
    print(f"Total data points: {len(prices)}")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # Scale data
    print("Scaling features...")
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(prices_scaled, seq_length)
    dates_seq = dates[seq_length:]
    
    # Split data
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_train, dates_test = dates_seq[:train_size], dates_seq[train_size:]
    
    print(f"\nDataset Info:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Sequence length: {seq_length} days")
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n=== Model Training ===")
    # Initialize model
    model = PricePredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_losses, val_losses, best_model_state = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        num_epochs=num_epochs, patience=10
    )
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    print("\n=== Saving and Evaluation ===")
    # Save model and scaler
    print("Saving best model...")
    checkpoint = {
        'model_state_dict': best_model_state,
        'scaler': scaler,
        'seq_length': seq_length,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, 'crypto_price_model.pth', _use_new_zipfile_serialization=False)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    test_predictions = []
    test_dates = []
    
    eval_pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Evaluating')
    with torch.no_grad():
        for i, (X_batch, y_batch) in eval_pbar:
            pred = model(X_batch)
            test_predictions.extend(pred.numpy().flatten())
            start_idx = i * batch_size
            end_idx = start_idx + len(y_batch)
            test_dates.extend(dates_test[start_idx:end_idx])
    
    # Convert predictions back to original scale
    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mse = np.mean((test_predictions - actual_prices) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_prices - test_predictions) / actual_prices)) * 100
    print(f'\nTest Metrics:')
    print(f'RMSE: ${rmse:.2f}')
    print(f'MAPE: {mape:.2f}%')
    
    # Plot predictions vs actuals with dates
    print("\nGenerating plots...")
    plt.figure(figsize=(15, 7))
    plt.plot(test_dates, actual_prices, label='Actual Prices', marker='o', markersize=3)
    plt.plot(test_dates, test_predictions, label='Predicted Prices', marker='o', markersize=3)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('BTC-USD: Actual vs Predicted Prices (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print some sample predictions
    print("\nSample Predictions (Last 5 days):")
    print("Date | Actual Price | Predicted Price | Difference (%)")
    print("-" * 60)
    for i in range(-5, 0):
        date = test_dates[i]
        actual = actual_prices[i][0]
        pred = test_predictions[i][0]
        diff_percent = ((pred - actual) / actual) * 100
        print(f"{date.strftime('%Y-%m-%d')} | ${actual:.2f} | ${pred:.2f} | {diff_percent:.2f}%")

if __name__ == "__main__":
    main()
