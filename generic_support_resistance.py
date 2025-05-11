"""
Generic Support and Resistance Detection for Multiple Cryptocurrencies

This script provides a unified approach to detect support and resistance levels
for any cryptocurrency, ensuring resistance levels are only shown above current price
and support levels are only shown below current price.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

from coindesk_client import CoinDeskClient
from indicators import add_indicators
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generic_sr_detection.log')
    ]
)

logger = logging.getLogger(__name__)

def get_data(symbol, timeframe='hours', interval='1h', limit=2000):
    """
    Get historical data for the specified symbol.
    
    Args:
        symbol (str): Symbol to get data for (e.g., 'BTC-USDT-VANILLA-PERPETUAL')
        timeframe (str): Data timeframe ('hours' or 'days')
        interval (str): Interval for the data (e.g., '1h', '4h', '1d')
        limit (int): Maximum number of records to return
        
    Returns:
        pandas.DataFrame: DataFrame with OHLCV data and indicators
    """
    logger.info(f"Collecting data from CoinDesk for {symbol} with {interval} interval")
    
    client = CoinDeskClient(api_key=config.COINDESK_API_KEY)
    
    api_interval = '1h' if interval == '4h' and timeframe == 'hours' else interval
    api_timeframe = timeframe
    
    try:
        df = client.get_historical_klines(
            symbol=symbol,
            market="binance",
            limit=limit,
            interval=api_interval,
            timeframe=api_timeframe
        )
        
        if df is None or len(df) == 0:
            logger.error(f"Failed to retrieve data for {symbol} with {interval} interval")
            return None
            
        logger.info(f"Collected {len(df)} candles from CoinDesk")
        
        if interval == '4h' and timeframe == 'hours' and df is not None and len(df) > 0:
            logger.info("Resampling 1h data to 4h")
            resampled = df.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            df = resampled.dropna()
            logger.info(f"Resampled to {len(df)} 4h candles")
        
        logger.info("Processing data and adding technical indicators")
        processed_df = add_indicators(df)
        logger.info(f"Processed {len(processed_df)} rows of data")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error retrieving or processing data: {str(e)}")
        return None

def detect_pivot_points(df, lookback=10):
    """
    Detect pivot points (highs and lows) in the price data.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        lookback (int): Number of periods to look back and forward
        
    Returns:
        tuple: (pivot_highs, pivot_lows) - DataFrames with pivot points
    """
    logger.info(f"Detecting pivot points with lookback={lookback}")
    
    pivot_highs = []
    pivot_lows = []
    
    for i in range(lookback, len(df) - lookback):
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, lookback+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, lookback+1)):
            pivot_highs.append({
                'timestamp': df.index[i],
                'pivot_high': df['high'].iloc[i],
                'volume': df['volume'].iloc[i]
            })
        
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, lookback+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, lookback+1)):
            pivot_lows.append({
                'timestamp': df.index[i],
                'pivot_low': df['low'].iloc[i],
                'volume': df['volume'].iloc[i]
            })
    
    pivot_highs_df = pd.DataFrame(pivot_highs) if pivot_highs else pd.DataFrame(columns=['timestamp', 'pivot_high', 'volume'])
    pivot_lows_df = pd.DataFrame(pivot_lows) if pivot_lows else pd.DataFrame(columns=['timestamp', 'pivot_low', 'volume'])
    
    logger.info(f"Detected {len(pivot_highs_df)} pivot highs and {len(pivot_lows_df)} pivot lows")
    
    return pivot_highs_df, pivot_lows_df

def cluster_price_levels(price_points, tolerance_pct=1.0):
    """
    Cluster price points into zones based on proximity.
    
    Args:
        price_points (numpy.ndarray): Array of price points
        tolerance_pct (float): Percentage tolerance for clustering
        
    Returns:
        list: List of tuples (price_level, count)
    """
    if len(price_points) == 0:
        return []
    
    sorted_points = np.sort(price_points)
    
    clusters = []
    current_cluster = [sorted_points[0]]
    
    for i in range(1, len(sorted_points)):
        current_point = sorted_points[i]
        cluster_avg = np.mean(current_cluster)
        
        if abs(current_point - cluster_avg) / cluster_avg * 100 <= tolerance_pct:
            current_cluster.append(current_point)
        else:
            clusters.append((np.mean(current_cluster), len(current_cluster)))
            current_cluster = [current_point]
    
    if current_cluster:
        clusters.append((np.mean(current_cluster), len(current_cluster)))
    
    price_levels = sorted(clusters, key=lambda x: x[1], reverse=True)
    
    return price_levels

def detect_resistance_levels(df, symbol, num_levels=5):
    """
    Detect resistance levels for any cryptocurrency.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        symbol (str): Symbol being analyzed
        num_levels (int): Number of levels to detect
        
    Returns:
        list: List of resistance levels above current price
    """
    logger.info(f"Detecting resistance levels for {symbol}")
    
    current_price = df['close'].iloc[-1]
    logger.info(f"Current price: {current_price}")
    
    all_resistance_levels = []
    
    lookback_periods = [5, 7, 10, 15, 20]
    
    for lookback in lookback_periods:
        logger.info(f"Using lookback period: {lookback}")
        
        pivot_highs, _ = detect_pivot_points(df, lookback)
        
        if len(pivot_highs) > 0:
            pivot_high_values = pivot_highs['pivot_high'].values
            
            resistance_zones = cluster_price_levels(pivot_high_values, tolerance_pct=1.0)
            
            for level, count in resistance_zones:
                all_resistance_levels.append((level, count))
    
    unique_levels = {}
    for level, count in all_resistance_levels:
        if level in unique_levels:
            unique_levels[level] += count
        else:
            unique_levels[level] = count
    
    resistance_levels = [(level, count) for level, count in unique_levels.items()]
    resistance_levels.sort(key=lambda x: x[1], reverse=True)
    
    filtered_levels = [level for level, _ in resistance_levels if level > current_price]
    
    filtered_levels.sort()
    
    top_levels = filtered_levels[:num_levels]
    
    logger.info(f"Detected {len(top_levels)} resistance levels above current price")
    
    return top_levels

def detect_support_levels(df, symbol, num_levels=5):
    """
    Detect support levels for any cryptocurrency.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        symbol (str): Symbol being analyzed
        num_levels (int): Number of levels to detect
        
    Returns:
        list: List of support levels below current price
    """
    logger.info(f"Detecting support levels for {symbol}")
    
    current_price = df['close'].iloc[-1]
    logger.info(f"Current price: {current_price}")
    
    all_support_levels = []
    
    lookback_periods = [5, 7, 10, 15, 20]
    
    for lookback in lookback_periods:
        logger.info(f"Using lookback period: {lookback}")
        
        _, pivot_lows = detect_pivot_points(df, lookback)
        
        if len(pivot_lows) > 0:
            pivot_low_values = pivot_lows['pivot_low'].values
            
            support_zones = cluster_price_levels(pivot_low_values, tolerance_pct=1.0)
            
            for level, count in support_zones:
                all_support_levels.append((level, count))
    
    unique_levels = {}
    for level, count in all_support_levels:
        if level in unique_levels:
            unique_levels[level] += count
        else:
            unique_levels[level] = count
    
    support_levels = [(level, count) for level, count in unique_levels.items()]
    support_levels.sort(key=lambda x: x[1], reverse=True)
    
    filtered_levels = [level for level, _ in support_levels if level < current_price]
    
    filtered_levels.sort(reverse=True)
    
    top_levels = filtered_levels[:num_levels]
    
    logger.info(f"Detected {len(top_levels)} support levels below current price")
    
    return top_levels

def visualize_levels(df, symbol, resistance_levels, support_levels, interval='1h'):
    """
    Visualize price with support and resistance levels.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        symbol (str): Symbol being analyzed
        resistance_levels (list): List of resistance levels
        support_levels (list): List of support levels
        interval (str): Time interval of the data
        
    Returns:
        str: Path to the saved plot
    """
    logger.info(f"Visualizing levels for {symbol} with {interval} timeframe")
    
    plt.figure(figsize=(14, 7))
    
    if interval == '1h':
        display_candles = 100
    elif interval == '4h':
        display_candles = 50
    else:  # 1d
        display_candles = 30
    
    plt.plot(df.index[-display_candles:], df['close'].iloc[-display_candles:], label='Close Price', color='blue', alpha=0.7)
    
    current_price = df['close'].iloc[-1]
    plt.axhline(y=current_price, color='black', linestyle='-', alpha=0.5, label=f'Current Price: {current_price:.2f}')
    
    for i, level in enumerate(resistance_levels):
        distance = ((level - current_price) / current_price) * 100
        plt.axhline(y=level, color='red', linestyle='--', alpha=0.7, 
                    label=f'R{i+1}: {level:.2f} (+{distance:.2f}%)')
    
    for i, level in enumerate(support_levels):
        distance = ((current_price - level) / current_price) * 100
        plt.axhline(y=level, color='green', linestyle='--', alpha=0.7,
                    label=f'S{i+1}: {level:.2f} (-{distance:.2f}%)')
    
    symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
    plt.title(f'{symbol_name} Price with Support and Resistance Levels ({interval})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/generic_levels_{symbol.replace('-', '_')}_{interval}_{timestamp}.png"
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Plot saved to {save_path}")
    return save_path

def analyze_levels(symbol, resistance_levels, support_levels, current_price, interval='1h'):
    """
    Analyze support and resistance levels relative to current price.
    
    Args:
        symbol (str): Symbol being analyzed
        resistance_levels (list): List of resistance levels
        support_levels (list): List of support levels
        current_price (float): Current price
        interval (str): Time interval of the data
        
    Returns:
        str: Analysis text
    """
    symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
    
    analysis = f"\n{'='*50}\n"
    analysis += f"SUPPORT AND RESISTANCE ANALYSIS FOR {symbol_name} ({interval})\n"
    analysis += f"{'='*50}\n\n"
    
    analysis += f"Current Price: {current_price:.2f}\n\n"
    
    analysis += "RESISTANCE LEVELS (above current price):\n"
    analysis += f"{'-'*50}\n"
    
    if resistance_levels:
        for i, level in enumerate(resistance_levels):
            distance = ((level - current_price) / current_price) * 100
            strength = 5 - i if i < 5 else 1  # Simple strength metric based on proximity
            analysis += f"R{i+1}: {level:.2f} (+{distance:.2f}%) - Strength: {strength}/5\n"
    else:
        analysis += "No significant resistance levels detected above current price.\n"
    
    analysis += f"\nSUPPORT LEVELS (below current price):\n"
    analysis += f"{'-'*50}\n"
    
    if support_levels:
        for i, level in enumerate(support_levels):
            distance = ((current_price - level) / current_price) * 100
            strength = 5 - i if i < 5 else 1  # Simple strength metric based on proximity
            analysis += f"S{i+1}: {level:.2f} (-{distance:.2f}%) - Strength: {strength}/5\n"
    else:
        analysis += "No significant support levels detected below current price.\n"
    
    analysis += f"\n{'='*50}\n"
    
    return analysis

def analyze_cryptocurrency(symbol, timeframes=[('1h', 'hours'), ('4h', 'hours')], num_levels=5):
    """
    Analyze support and resistance levels for a cryptocurrency across multiple timeframes.
    
    Args:
        symbol (str): Symbol to analyze (e.g., 'BTC-USDT-VANILLA-PERPETUAL')
        timeframes (list): List of tuples (interval, timeframe) to analyze
        num_levels (int): Number of levels to detect
        
    Returns:
        dict: Dictionary with analysis results
    """
    logger.info(f"Analyzing {symbol} across multiple timeframes")
    
    results = {}
    
    for interval, timeframe in timeframes:
        logger.info(f"Processing {symbol} with {interval} interval")
        
        df = get_data(symbol, timeframe, interval)
        
        if df is None or len(df) == 0:
            logger.error(f"No data available for {symbol} with {interval} interval. Skipping.")
            continue
        
        resistance_levels = detect_resistance_levels(df, symbol, num_levels)
        
        support_levels = detect_support_levels(df, symbol, num_levels)
        
        current_price = df['close'].iloc[-1]
        
        plot_path = visualize_levels(df, symbol, resistance_levels, support_levels, interval)
        
        analysis = analyze_levels(symbol, resistance_levels, support_levels, current_price, interval)
        
        print(analysis)
        
        results[interval] = {
            'df': df,
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'current_price': current_price,
            'plot_path': plot_path,
            'analysis': analysis
        }
    
    return results

def main():
    """
    Main function to analyze multiple cryptocurrencies.
    """
    logger.info("Starting generic support and resistance detection")
    
    os.makedirs('results', exist_ok=True)
    
    cryptocurrencies = [
        'BTC-USDT-VANILLA-PERPETUAL',
        'SUI-USDT-VANILLA-PERPETUAL',
    ]
    
    timeframes = [
        ('1h', 'hours'),
        ('4h', 'hours'),
        ('1d', 'days')
    ]
    
    all_results = {}
    
    for symbol in cryptocurrencies:
        logger.info(f"Analyzing {symbol}")
        
        symbol_results = analyze_cryptocurrency(symbol, timeframes)
        
        all_results[symbol] = symbol_results
    
    logger.info("Generic support and resistance detection completed")
    
    return all_results

if __name__ == "__main__":
    main()
