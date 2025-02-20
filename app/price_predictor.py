import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

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

def load_model(model_path='crypto_price_model.pth'):
    """Load the trained model and scaler"""
    try:
        checkpoint = torch.load(model_path)
        model = PricePredictor()
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']
        model.eval()
        return model, scaler, checkpoint.get('seq_length', 30)
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def calculate_confidence(volatility, price_change):
    """Calculate prediction confidence based on volatility"""
    # Higher volatility = lower confidence
    # Lower price change relative to volatility = higher confidence
    volatility_factor = 1 / (1 + np.abs(volatility))
    change_factor = 1 / (1 + np.abs(price_change / (volatility + 1e-6)))
    confidence = (volatility_factor + change_factor) / 2 * 100
    return min(max(confidence, 0), 100)  # Clip between 0 and 100

def get_price_prediction(symbol='BTC-USD', days=1):
    """
    Get price prediction for a cryptocurrency
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC-USD')
        days (int): Number of days to predict into the future
        
    Returns:
        dict: Prediction results including current price, predicted price,
              price change, trend, and confidence
    """
    try:
        # Load model
        model, scaler, seq_length = load_model()
        
        # Get historical data
        crypto = yf.Ticker(symbol)
        hist = crypto.history(period=f'{seq_length+10}d')
        prices = hist['Close'].values[-seq_length:]
        
        # Calculate volatility for confidence
        volatility = np.std(np.diff(prices) / prices[:-1])
        
        # Scale data
        prices_scaled = scaler.transform(prices.reshape(-1, 1))
        
        # Prepare input sequence
        input_seq = torch.FloatTensor(prices_scaled).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            pred_scaled = model(input_seq)
            prediction = scaler.inverse_transform(pred_scaled.numpy())[0][0]
        
        # Calculate metrics
        current_price = prices[-1]
        price_change = prediction - current_price
        percent_change = (price_change / current_price) * 100
        confidence = calculate_confidence(volatility, percent_change)
        
        return {
            "current_price": current_price,
            "predicted_price": prediction,
            "price_change": price_change,
            "percent_change": percent_change,
            "trend": "bullish" if price_change > 0 else "bearish",
            "confidence": confidence,
            "prediction_date": (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d"),
            "volatility": volatility
        }
        
    except Exception as e:
        raise Exception(f"Failed to get prediction: {str(e)}")

if __name__ == "__main__":
    # Test prediction
    try:
        result = get_price_prediction('BTC-USD')
        print("\nPrice Prediction Results:")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Predicted Price: ${result['predicted_price']:.2f}")
        print(f"Change: {result['percent_change']:+.2f}%")
        print(f"Trend: {result['trend']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Prediction Date: {result['prediction_date']}")
    except Exception as e:
        print(f"Error: {str(e)}")
