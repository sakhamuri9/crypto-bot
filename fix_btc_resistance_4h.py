"""
Script to fix BTC resistance detection to show multiple levels like SUI.
This script uses a modified approach to detect resistance levels for BTC.
Handles 4h timeframe by resampling 1h data.
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
        logging.FileHandler('btc_resistance_fix_4h.log')
    ]
)

logger = logging.getLogger(__name__)

def get_data(symbol, timeframe='hours', interval='1h', limit=2000):
    """
    Get historical data for the specified symbol.
    
    Args:
        symbol (str): Symbol to get data for (e.g., 'BTC-USDT-VANILLA-PERPETUAL')
        timeframe (str): Data timeframe ('hours' or 'days')
        interval (str): Interval for the data (e.g., '1h', '4h')
        limit (int): Maximum number of records to return
        
    Returns:
        pandas.DataFrame: DataFrame with OHLCV data and indicators
    """
    logger.info(f"Collecting data from CoinDesk for {symbol} with {interval} interval")
    
    client = CoinDeskClient(api_key=config.COINDESK_API_KEY)
    
    df = client.get_historical_klines(
        symbol=symbol,
        market="binance",
        limit=limit,
        interval="1h",
        timeframe=timeframe
    )
    
    logger.info(f"Collected {len(df)} candles from CoinDesk")
    
    if interval == '4h':
        logger.info("Resampling 1h data to 4h")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        resampled = df.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        resampled = resampled.dropna()
        
        logger.info(f"Resampled to {len(resampled)} 4h candles")
        df = resampled
    
    logger.info("Processing data and adding technical indicators")
    processed_df = add_indicators(df)
    logger.info(f"Processed {len(processed_df)} rows of data")
    
    return processed_df

def detect_pivot_points(df, lookback=10):
    """
    Detect pivot high and low points in the price series.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        lookback (int): Number of periods to look back and forward
        
    Returns:
        tuple: (pivot_highs, pivot_lows) - DataFrames with pivot points
    """
    logger.info(f"Detecting pivot points with lookback={lookback}")
    
    highs = df['high'].copy()
    lows = df['low'].copy()
    
    pivot_highs = np.zeros(len(df))
    pivot_lows = np.zeros(len(df))
    
    for i in range(lookback, len(df) - lookback):
        if highs.iloc[i] > max(highs.iloc[i-lookback:i]) and highs.iloc[i] > max(highs.iloc[i+1:i+lookback+1]):
            pivot_highs[i] = highs.iloc[i]
        
        if lows.iloc[i] < min(lows.iloc[i-lookback:i]) and lows.iloc[i] < min(lows.iloc[i+1:i+lookback+1]):
            pivot_lows[i] = lows.iloc[i]
    
    pivot_high_df = pd.DataFrame(index=df.index)
    pivot_high_df['pivot_high'] = pivot_highs
    pivot_high_df = pivot_high_df[pivot_high_df['pivot_high'] > 0]
    
    pivot_low_df = pd.DataFrame(index=df.index)
    pivot_low_df['pivot_low'] = pivot_lows
    pivot_low_df = pivot_low_df[pivot_low_df['pivot_low'] > 0]
    
    logger.info(f"Detected {len(pivot_high_df)} pivot highs and {len(pivot_low_df)} pivot lows")
    
    return pivot_high_df, pivot_low_df

def cluster_price_levels(price_points, tolerance_pct=1.0):
    """
    Cluster price points into zones based on proximity.
    
    Args:
        price_points (numpy.ndarray): Array of price points
        tolerance_pct (float): Percentage tolerance for clustering
        
    Returns:
        list: List of (price_level, count) tuples
    """
    if len(price_points) == 0:
        return []
    
    sorted_points = np.sort(price_points)
    
    clusters = []
    current_cluster = [sorted_points[0]]
    
    for i in range(1, len(sorted_points)):
        pct_diff = (sorted_points[i] - current_cluster[0]) / current_cluster[0] * 100
        
        if pct_diff <= tolerance_pct:
            current_cluster.append(sorted_points[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_points[i]]
    
    clusters.append(current_cluster)
    
    price_levels = [(sum(cluster) / len(cluster), len(cluster)) for cluster in clusters]
    
    price_levels.sort(key=lambda x: x[1], reverse=True)
    
    return price_levels

def detect_btc_resistance_levels(df, num_levels=5, include_historical=True):
    """
    Detect BTC resistance levels using a modified approach.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        num_levels (int): Number of levels to detect
        include_historical (bool): Whether to include historical resistance levels
        
    Returns:
        list: List of resistance levels
    """
    logger.info("Detecting BTC resistance levels with modified approach")
    
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
    
    if include_historical:
        above_current = [level for level, _ in resistance_levels if level > current_price]
        below_current = [level for level, _ in resistance_levels if level <= current_price]
        
        below_current.sort(key=lambda x: current_price - x)
        
        filtered_levels = above_current + below_current
    else:
        filtered_levels = [level for level, _ in resistance_levels if level > current_price]
    
    top_levels = filtered_levels[:num_levels]
    
    logger.info(f"Detected {len(top_levels)} resistance levels")
    
    return top_levels

def detect_support_levels(df, num_levels=5):
    """
    Detect support levels below current price.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        num_levels (int): Number of levels to detect
        
    Returns:
        list: List of support levels
    """
    logger.info("Detecting support levels")
    
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
    
    filtered_levels.sort(key=lambda x: current_price - x)
    
    top_levels = filtered_levels[:num_levels]
    
    logger.info(f"Detected {len(top_levels)} support levels")
    
    return top_levels

def visualize_levels(df, symbol, resistance_levels, support_levels, timeframe='1h'):
    """
    Visualize price with resistance and support levels.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        symbol (str): Symbol being analyzed
        resistance_levels (list): List of resistance levels
        support_levels (list): List of support levels
        timeframe (str): Timeframe of the data
        
    Returns:
        str: Path to the saved plot
    """
    logger.info(f"Visualizing levels for {symbol} with {timeframe} timeframe")
    
    plt.figure(figsize=(14, 7))
    
    display_bars = min(200, len(df))
    plt.plot(df.index[-display_bars:], df['close'].iloc[-display_bars:], label='Close Price', color='blue', alpha=0.7)
    
    current_price = df['close'].iloc[-1]
    plt.axhline(y=current_price, color='black', linestyle='-', alpha=0.5, label=f'Current Price: {current_price:.2f}')
    
    for i, level in enumerate(resistance_levels):
        distance = ((level - current_price) / current_price) * 100
        direction = "+" if level > current_price else "-"
        plt.axhline(y=level, color='red', linestyle='--', alpha=0.7, 
                    label=f'R{i+1}: {level:.2f} ({direction}{abs(distance):.2f}%)')
    
    for i, level in enumerate(support_levels):
        distance = ((current_price - level) / current_price) * 100
        plt.axhline(y=level, color='green', linestyle='--', alpha=0.7,
                    label=f'S{i+1}: {level:.2f} (-{distance:.2f}%)')
    
    symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
    plt.title(f'{symbol_name} Price with Fixed Resistance and Support Levels ({timeframe})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/fixed_levels_{symbol.replace('-', '_')}_{timeframe}_{timestamp}.png"
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Plot saved to {save_path}")
    return save_path

def analyze_levels(symbol, resistance_levels, support_levels, current_price, timeframe='1h'):
    """
    Analyze resistance and support levels relative to current price.
    
    Args:
        symbol (str): Symbol being analyzed
        resistance_levels (list): List of resistance levels
        support_levels (list): List of support levels
        current_price (float): Current price
        timeframe (str): Timeframe of the data
        
    Returns:
        str: Analysis text
    """
    symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
    
    analysis = f"\n{'='*50}\n"
    analysis += f"FIXED RESISTANCE AND SUPPORT ANALYSIS FOR {symbol_name} ({timeframe})\n"
    analysis += f"{'='*50}\n\n"
    
    analysis += f"Current Price: {current_price:.2f}\n\n"
    
    analysis += "RESISTANCE LEVELS:\n"
    analysis += f"{'-'*50}\n"
    
    if resistance_levels:
        for i, level in enumerate(resistance_levels):
            distance = ((level - current_price) / current_price) * 100
            direction = "+" if level > current_price else "-"
            strength = 5 - i  # Simple strength metric based on proximity
            analysis += f"R{i+1}: {level:.2f} ({direction}{abs(distance):.2f}%) - Strength: {strength}/5\n"
    else:
        analysis += "No significant resistance levels detected.\n"
    
    analysis += f"\nSUPPORT LEVELS (below current price):\n"
    analysis += f"{'-'*50}\n"
    
    if support_levels:
        for i, level in enumerate(support_levels):
            distance = ((current_price - level) / current_price) * 100
            strength = 5 - i  # Simple strength metric based on proximity
            analysis += f"S{i+1}: {level:.2f} (-{distance:.2f}%) - Strength: {strength}/5\n"
    else:
        analysis += "No significant support levels detected below current price.\n"
    
    analysis += f"\n{'='*50}\n"
    
    return analysis

def main():
    """
    Main function to fix BTC resistance detection.
    """
    logger.info("Starting BTC resistance fix with 4h timeframe support")
    
    os.makedirs('results', exist_ok=True)
    
    timeframes = [
        ('1h', 'hours'),
        ('4h', 'hours')
    ]
    
    for interval, timeframe in timeframes:
        logger.info(f"Processing BTC-USDT with {interval} interval")
        
        df = get_data('BTC-USDT-VANILLA-PERPETUAL', timeframe, interval)
        
        resistance_levels = detect_btc_resistance_levels(df, num_levels=5, include_historical=True)
        
        support_levels = detect_support_levels(df, num_levels=5)
        
        current_price = df['close'].iloc[-1]
        
        plot_path = visualize_levels(df, 'BTC-USDT-VANILLA-PERPETUAL', resistance_levels, support_levels, interval)
        
        analysis = analyze_levels('BTC-USDT-VANILLA-PERPETUAL', resistance_levels, support_levels, current_price, interval)
        
        print(analysis)
        
        logger.info(f"Processing SUI-USDT with {interval} interval")
        
        sui_df = get_data('SUI-USDT-VANILLA-PERPETUAL', timeframe, interval)
        
        sui_resistance_levels = detect_btc_resistance_levels(sui_df, num_levels=5, include_historical=True)
        
        sui_support_levels = detect_support_levels(sui_df, num_levels=5)
        
        sui_current_price = sui_df['close'].iloc[-1]
        
        sui_plot_path = visualize_levels(sui_df, 'SUI-USDT-VANILLA-PERPETUAL', sui_resistance_levels, sui_support_levels, interval)
        
        sui_analysis = analyze_levels('SUI-USDT-VANILLA-PERPETUAL', sui_resistance_levels, sui_support_levels, sui_current_price, interval)
        
        print(sui_analysis)
    
    logger.info("BTC resistance fix with 4h timeframe support completed")

if __name__ == "__main__":
    main()
