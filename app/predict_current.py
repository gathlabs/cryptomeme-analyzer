import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from price_predictor import get_price_prediction, load_model
import yfinance as yf
import numpy as np
import torch

def get_historical_predictions(prices, dates, model, scaler):
    """
    Get historical predictions for comparison
    """
    predictions = []
    seq_length = 30
    
    # For each day in the historical data (except the last day)
    for i in range(len(prices) - seq_length):
        # Get the sequence of 30 days before this point
        sequence = prices[i:i+seq_length]
        sequence_scaled = scaler.transform(sequence.reshape(-1, 1))
        
        # Prepare input
        input_seq = torch.FloatTensor(sequence_scaled).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            pred_scaled = model(input_seq)
            prediction = scaler.inverse_transform(pred_scaled.numpy())[0][0]
            predictions.append(prediction)
    
    # The predictions start from day 31 (after the first sequence)
    prediction_dates = dates[seq_length:]
    
    return predictions, prediction_dates

def plot_predictions(symbol='BTC-USD', days_to_predict=1):
    """
    Plot historical prices, historical predictions, and future prediction
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC-USD')
        days_to_predict (int): Number of days to predict (typically 1)
    """
    # Load model and scaler
    model, scaler, seq_length = load_model()
    
    # Get historical data
    crypto = yf.Ticker(symbol)
    hist = crypto.history(period='60d')  # Get more data for better visualization
    prices = hist['Close'].values
    dates = hist.index
    
    # Get historical predictions
    historical_preds, historical_pred_dates = get_historical_predictions(
        prices, dates, model, scaler
    )
    
    # Get future prediction
    future_pred = get_price_prediction(symbol, days=1)
    future_date = datetime.strptime(future_pred['prediction_date'], '%Y-%m-%d')
    
    # Create visualization
    plt.figure(figsize=(15, 7))
    
    # Plot actual prices
    plt.plot(dates, prices, label='Actual Prices', color='blue', marker='o', markersize=3)
    
    # Plot historical predictions
    plt.plot(historical_pred_dates, historical_preds, 
            label='Historical Predictions', color='green', 
            marker='o', markersize=3, alpha=0.6)
    
    # Plot future prediction
    plt.plot([dates[-1], future_date], 
            [prices[-1], future_pred['predicted_price']], 
            label='Future Prediction', color='red', 
            marker='o', markersize=5, linestyle='--')
    
    # Add vertical line for current date
    plt.axvline(x=dates[-1], color='gray', linestyle='--', alpha=0.5)
    plt.text(dates[-1], plt.ylim()[0], 'Current', rotation=0, ha='right')
    
    # Calculate and display metrics
    historical_rmse = np.sqrt(np.mean((np.array(historical_preds) - prices[seq_length:])**2))
    historical_mape = np.mean(np.abs((prices[seq_length:] - np.array(historical_preds)) / prices[seq_length:])) * 100
    
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    title = f'{symbol}: Price Prediction\n'
    title += f'Historical RMSE: ${historical_rmse:.2f}, MAPE: {historical_mape:.2f}%\n'
    title += f'Next Day Prediction: ${future_pred["predicted_price"]:.2f} ({future_pred["percent_change"]:+.2f}%)'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'prediction_{symbol}_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename, historical_rmse, historical_mape

def main():
    parser = argparse.ArgumentParser(description='Predict cryptocurrency prices')
    parser.add_argument('--symbol', type=str, default='BTC-USD',
                      help='Trading symbol (e.g., BTC-USD, ETH-USD)')
    parser.add_argument('--days', type=int, default=1,
                      help='Number of days to predict')
    parser.add_argument('--no-plot', action='store_true',
                      help='Disable plot generation')
    
    args = parser.parse_args()
    
    try:
        # Get next day prediction
        pred = get_price_prediction(args.symbol)
        
        print("\nPrice Prediction Results:")
        print(f"Current Price: ${pred['current_price']:.2f}")
        print(f"Predicted Price: ${pred['predicted_price']:.2f}")
        print(f"Change: {pred['percent_change']:+.2f}%")
        print(f"Trend: {pred['trend']}")
        print(f"Confidence: {pred['confidence']:.1f}%")
        print(f"Prediction Date: {pred['prediction_date']}")
        print(f"Volatility: {pred['volatility']:.4f}")
        
        if not args.no_plot:
            plot_file, rmse, mape = plot_predictions(args.symbol, args.days)
            print(f"\nHistorical Prediction Metrics:")
            print(f"RMSE: ${rmse:.2f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"\nPlot saved as: {plot_file}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
