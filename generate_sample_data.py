"""
Generate sample historical price data for backtesting.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(symbol='BTCUSDT', days=180, interval='1h', volatility=0.02, trend=0.0001):
    """
    Generate sample OHLCV data for backtesting.
    
    Args:
        symbol (str): Trading pair symbol
        days (int): Number of days of data to generate
        interval (str): Time interval ('1h' for 1 hour)
        volatility (float): Price volatility factor
        trend (float): Price trend factor (positive for uptrend, negative for downtrend)
        
    Returns:
        pandas.DataFrame: DataFrame with OHLCV data
    """
    if interval == '1h':
        periods = days * 24
    elif interval == '4h':
        periods = days * 6
    elif interval == '1d':
        periods = days
    else:
        periods = days * 24  # Default to 1h
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    if interval == '1h':
        timestamps = [start_date + timedelta(hours=i) for i in range(periods)]
    elif interval == '4h':
        timestamps = [start_date + timedelta(hours=i*4) for i in range(periods)]
    elif interval == '1d':
        timestamps = [start_date + timedelta(days=i) for i in range(periods)]
    else:
        timestamps = [start_date + timedelta(hours=i) for i in range(periods)]
    
    np.random.seed(42)  # For reproducibility
    
    initial_price = 50000.0
    
    log_returns = np.random.normal(trend, volatility, periods)
    
    price_series = initial_price * np.exp(np.cumsum(log_returns))
    
    data = []
    for i in range(periods):
        price = price_series[i]
        price_range = price * volatility
        
        open_price = price
        close_price = price_series[i]
        high_price = max(open_price, close_price) + abs(np.random.normal(0, price_range/2))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, price_range/2))
        
        volume = np.random.normal(1000, 500) * price / 10000
        
        if np.random.random() < 0.05:  # 5% chance of a price spike
            high_price *= 1.05
        if np.random.random() < 0.05:  # 5% chance of a price drop
            low_price *= 0.95
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': max(volume, 0)  # Ensure volume is positive
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def save_sample_data(df, filename='sample_data.csv'):
    """
    Save sample data to CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        filename (str): Output filename
    """
    os.makedirs('data', exist_ok=True)
    
    filepath = os.path.join('data', filename)
    df.to_csv(filepath)
    print(f"Sample data saved to {filepath}")
    
    return filepath

if __name__ == "__main__":
    print("Generating sample data...")
    df = generate_sample_data(symbol='BTCUSDT', days=180, interval='1h')
    
    filepath = save_sample_data(df, 'btcusdt_1h_sample.csv')
    
    print("\nSample data:")
    print(df.head())
    
    print(f"\nGenerated {len(df)} rows of sample data.")
