"""
Script to visualize support and resistance levels for BTC-USDT and SUI-USDT using 4-hour timeframe.
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
from enhanced_resistance_detection import calculate_enhanced_support_resistance
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sr_4h_visualization.log')
    ]
)

logger = logging.getLogger(__name__)

def get_data(symbol, timeframe='hours', interval='4h', limit=2000):
    """
    Get historical data for the specified symbol with 4-hour timeframe.
    
    Args:
        symbol (str): Symbol to get data for (e.g., 'BTC-USDT-VANILLA-PERPETUAL')
        timeframe (str): Data timeframe ('hours' or 'days')
        interval (str): Interval for the data (e.g., '4h')
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
        interval=interval,
        timeframe=timeframe
    )
    
    logger.info(f"Collected {len(df)} candles from CoinDesk")
    
    logger.info("Processing data and adding technical indicators")
    processed_df = add_indicators(df)
    logger.info(f"Processed {len(processed_df)} rows of data")
    
    return processed_df

def detect_multiple_sr_levels(df, symbol, num_levels=5):
    """
    Detect multiple support and resistance levels using different parameter sets.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        symbol (str): Symbol being analyzed
        num_levels (int): Number of levels to detect
        
    Returns:
        tuple: (resistance_levels, support_levels) - Lists of detected levels
    """
    logger.info(f"Detecting multiple S/R levels for {symbol} with 4h timeframe")
    
    parameter_sets = [
        (5, 40, 5, num_levels, 1, 0.8),  # High resistance bias
        (7, 30, 7, num_levels, 2, 0.7),  # Medium resistance bias
        (10, 20, 10, num_levels, 3, 0.6),  # Low resistance bias
        (3, 50, 3, num_levels, 1, 0.9),   # Very high resistance bias
        (8, 25, 8, num_levels, 2, 0.65)   # Custom balance
    ]
    
    all_resistance_levels = []
    all_support_levels = []
    
    current_price = df['close'].iloc[-1]
    logger.info(f"Current price: {current_price}")
    
    for i, params in enumerate(parameter_sets):
        pivot_period, max_pivot_count, channel_width_pct, max_sr_count, min_strength, resistance_bias = params
        
        logger.info(f"Parameter set {i+1}: pivot_period={pivot_period}, resistance_bias={resistance_bias}")
        
        sr_df = calculate_enhanced_support_resistance(
            df.copy(), 
            pivot_period=pivot_period,
            max_pivot_count=max_pivot_count,
            channel_width_pct=channel_width_pct,
            max_sr_count=max_sr_count,
            min_strength=min_strength,
            resistance_bias=resistance_bias,
            support_bias=1.0 - resistance_bias
        )
        
        if 'resistance' in sr_df.columns:
            resistance_values = sr_df['resistance'].dropna().unique()
            resistance_above = [r for r in resistance_values if r > current_price]
            all_resistance_levels.extend(resistance_above)
            
            logger.info(f"Found {len(resistance_above)} resistance levels above current price")
        
        if 'support' in sr_df.columns:
            support_values = sr_df['support'].dropna().unique()
            support_below = [s for s in support_values if s < current_price]
            all_support_levels.extend(support_below)
            
            logger.info(f"Found {len(support_below)} support levels below current price")
    
    resistance_levels = sorted(list(set(all_resistance_levels)))
    support_levels = sorted(list(set(all_support_levels)), reverse=True)
    
    resistance_levels = [r for r in resistance_levels if r > current_price]
    resistance_levels.sort()
    resistance_levels = resistance_levels[:num_levels]
    
    support_levels = [s for s in support_levels if s < current_price]
    support_levels.sort(reverse=True)
    support_levels = support_levels[:num_levels]
    
    return resistance_levels, support_levels

def visualize_sr_levels(df, symbol, resistance_levels, support_levels):
    """
    Visualize price with multiple support and resistance levels.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        symbol (str): Symbol being analyzed
        resistance_levels (list): List of resistance levels
        support_levels (list): List of support levels
        
    Returns:
        str: Path to the saved plot
    """
    logger.info(f"Visualizing 4h S/R levels for {symbol}")
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(df.index[-100:], df['close'].iloc[-100:], label='Close Price', color='blue', alpha=0.7)
    
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
    plt.title(f'{symbol_name} Price with 4h Support and Resistance Levels')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/4h_sr_levels_{symbol.replace('-', '_')}_{timestamp}.png"
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Plot saved to {save_path}")
    return save_path

def analyze_sr_levels(symbol, resistance_levels, support_levels, current_price):
    """
    Analyze support and resistance levels relative to current price.
    
    Args:
        symbol (str): Symbol being analyzed
        resistance_levels (list): List of resistance levels
        support_levels (list): List of support levels
        current_price (float): Current price
        
    Returns:
        str: Analysis text
    """
    symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
    
    analysis = f"\n{'='*50}\n"
    analysis += f"4H SUPPORT AND RESISTANCE ANALYSIS FOR {symbol_name}\n"
    analysis += f"{'='*50}\n\n"
    
    analysis += f"Current Price: {current_price:.2f}\n\n"
    
    analysis += "RESISTANCE LEVELS (above current price):\n"
    analysis += f"{'-'*50}\n"
    
    if resistance_levels:
        for i, level in enumerate(resistance_levels):
            distance = ((level - current_price) / current_price) * 100
            strength = 5 - i  # Simple strength metric based on proximity
            analysis += f"R{i+1}: {level:.2f} (+{distance:.2f}%) - Strength: {strength}/5\n"
    else:
        analysis += "No significant resistance levels detected above current price.\n"
    
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
    Main function to visualize 4-hour support and resistance levels.
    """
    logger.info("Starting 4h S/R level visualization")
    
    os.makedirs('results', exist_ok=True)
    
    symbols = [
        ('BTC-USDT-VANILLA-PERPETUAL', 'hours', '4h'),
        ('SUI-USDT-VANILLA-PERPETUAL', 'hours', '4h')  # Note: Using hours timeframe for SUI with 4h interval
    ]
    
    plot_paths = []
    analysis_texts = []
    
    for symbol, timeframe, interval in symbols:
        logger.info(f"Processing {symbol} with {interval} interval")
        
        df = get_data(symbol, timeframe, interval)
        
        resistance_levels, support_levels = detect_multiple_sr_levels(df, symbol)
        
        current_price = df['close'].iloc[-1]
        
        plot_path = visualize_sr_levels(df, symbol, resistance_levels, support_levels)
        plot_paths.append(plot_path)
        
        analysis = analyze_sr_levels(symbol, resistance_levels, support_levels, current_price)
        analysis_texts.append(analysis)
        
        print(analysis)
    
    logger.info("4h S/R level visualization completed")
    
    return plot_paths, analysis_texts

if __name__ == "__main__":
    main()
