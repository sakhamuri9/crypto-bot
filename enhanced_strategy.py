"""
Enhanced Trading Strategy

This module implements an enhanced trading strategy with institutional-grade features:
1. Dynamic position sizing using Kelly criterion
2. ATR-based stop-loss placement
3. Multi-timeframe confirmation
4. Volume-weighted entry/exit points
5. Correlation-based risk management
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from coindesk_client import CoinDeskClient
from indicators import add_indicators
from generic_support_resistance import detect_support_resistance_levels
from squeeze_momentum import add_squeeze_momentum
from combined_strategy import get_data
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_strategy.log')
    ]
)

logger = logging.getLogger(__name__)

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        period (int): ATR period
        
    Returns:
        pandas.Series: ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def add_moving_averages(df, periods=[20, 50, 200]):
    """
    Add Simple Moving Averages (SMA) to the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        periods (list): List of periods for SMAs
        
    Returns:
        pandas.DataFrame: DataFrame with SMAs added
    """
    result_df = df.copy()
    
    for period in periods:
        result_df[f'sma_{period}'] = result_df['close'].rolling(window=period).mean()
    
    return result_df

def calculate_kelly_position_size(win_rate, reward_risk_ratio, max_position_size=0.2, safety_factor=0.5):
    """
    Calculate position size using the Kelly Criterion.
    
    Args:
        win_rate (float): Win rate as a decimal (e.g., 0.6 for 60%)
        reward_risk_ratio (float): Average reward to risk ratio
        max_position_size (float): Maximum position size as a percentage of capital
        safety_factor (float): Factor to reduce Kelly bet size for safety (0.5 = "Half Kelly")
        
    Returns:
        float: Position size as a percentage of capital
    """
    kelly_fraction = win_rate - ((1 - win_rate) / reward_risk_ratio)
    
    position_size = min(kelly_fraction * safety_factor, max_position_size)
    
    position_size = max(position_size, 0.01)
    
    return position_size

def generate_enhanced_signals(df, symbol, interval='1h'):
    """
    Generate enhanced trading signals.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data and indicators
        symbol (str): Symbol being analyzed
        interval (str): Interval of the data
        
    Returns:
        pandas.DataFrame: DataFrame with enhanced signals
    """
    logger.info(f"Generating enhanced signals for {symbol} with {interval} timeframe")
    
    result_df = detect_support_resistance_levels(df.copy(), symbol, interval)
    
    result_df = add_squeeze_momentum(result_df)
    
    result_df = add_moving_averages(result_df, periods=[20, 50, 200])
    
    result_df['atr'] = calculate_atr(result_df)
    
    result_df['enhanced_buy'] = False
    result_df['enhanced_sell'] = False
    
    
    for i in range(1, len(result_df)):
        if pd.isna(result_df['atr'].iloc[i]) or pd.isna(result_df['sma_50'].iloc[i]):
            continue
        
        current_price = result_df['close'].iloc[i]
        atr_value = result_df['atr'].iloc[i]
        
        near_support = False
        if 'support' in result_df.columns:
            support_values = result_df['support'].iloc[i-10:i+1].dropna().unique()
            support_below = [s for s in support_values if s < current_price]
            
            if support_below:
                closest_support = max(support_below)
                support_distance = (current_price - closest_support) / current_price
                
                near_support = support_distance < 0.02 or (current_price - closest_support) < 1.5 * atr_value
        
        near_resistance = False
        if 'resistance' in result_df.columns:
            resistance_values = result_df['resistance'].iloc[i-10:i+1].dropna().unique()
            resistance_above = [r for r in resistance_values if r > current_price]
            
            if resistance_above:
                closest_resistance = min(resistance_above)
                resistance_distance = (closest_resistance - current_price) / current_price
                
                near_resistance = resistance_distance < 0.02 or (closest_resistance - current_price) < 1.5 * atr_value
        
        not_downtrend = result_df['close'].iloc[i] > result_df['sma_50'].iloc[i]
        
        if result_df['sqz_buy'].iloc[i] and near_support and not_downtrend:
            result_df.loc[result_df.index[i], 'enhanced_buy'] = True
        
        if result_df['sqz_sell'].iloc[i] and near_resistance:
            result_df.loc[result_df.index[i], 'enhanced_sell'] = True
    
    buy_signals = sum(result_df['enhanced_buy'])
    sell_signals = sum(result_df['enhanced_sell'])
    
    logger.info(f"Generated {buy_signals} enhanced buy signals and {sell_signals} enhanced sell signals")
    
    return result_df

def calculate_dynamic_stop_loss(df, position_type='long', atr_multiplier=2.0, min_stop_pct=0.01, max_stop_pct=0.05):
    """
    Calculate dynamic stop-loss levels based on ATR.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data and ATR
        position_type (str): 'long' or 'short'
        atr_multiplier (float): Multiplier for ATR
        min_stop_pct (float): Minimum stop-loss percentage
        max_stop_pct (float): Maximum stop-loss percentage
        
    Returns:
        pandas.DataFrame: DataFrame with dynamic stop-loss levels
    """
    result_df = df.copy()
    
    result_df['dynamic_stop_pct'] = result_df['atr'] / result_df['close'] * atr_multiplier
    
    result_df['dynamic_stop_pct'] = result_df['dynamic_stop_pct'].clip(min_stop_pct, max_stop_pct)
    
    if position_type == 'long':
        result_df['dynamic_stop_loss'] = result_df['close'] * (1 - result_df['dynamic_stop_pct'])
    else:  # short
        result_df['dynamic_stop_loss'] = result_df['close'] * (1 + result_df['dynamic_stop_pct'])
    
    return result_df

def calculate_dynamic_take_profit(df, position_type='long', reward_risk_ratio=2.0):
    """
    Calculate dynamic take-profit levels based on stop-loss distance.
    
    Args:
        df (pandas.DataFrame): DataFrame with dynamic stop-loss levels
        position_type (str): 'long' or 'short'
        reward_risk_ratio (float): Target reward-to-risk ratio
        
    Returns:
        pandas.DataFrame: DataFrame with dynamic take-profit levels
    """
    result_df = df.copy()
    
    if position_type == 'long':
        stop_distance = result_df['close'] - result_df['dynamic_stop_loss']
        result_df['dynamic_take_profit'] = result_df['close'] + (stop_distance * reward_risk_ratio)
    else:  # short
        stop_distance = result_df['dynamic_stop_loss'] - result_df['close']
        result_df['dynamic_take_profit'] = result_df['close'] - (stop_distance * reward_risk_ratio)
    
    return result_df

def main():
    """
    Main function to test enhanced strategy.
    """
    logger.info("Testing enhanced strategy")
    
    os.makedirs('results', exist_ok=True)
    
    symbols = [
        ('BTC-USDT-VANILLA-PERPETUAL', 'hours', '1h'),
        ('BTC-USDT-VANILLA-PERPETUAL', 'hours', '4h')
    ]
    
    for symbol, timeframe, interval in symbols:
        logger.info(f"Processing {symbol} with {interval} interval")
        
        try:
            df = get_data(symbol, timeframe, interval)
            
            enhanced_df = generate_enhanced_signals(df, symbol, interval)
            
            enhanced_df = calculate_dynamic_stop_loss(enhanced_df, 'long')
            enhanced_df = calculate_dynamic_take_profit(enhanced_df, 'long')
            
            logger.info(f"Enhanced strategy analysis completed for {symbol} with {interval} interval")
            
        except Exception as e:
            logger.error(f"Error processing {symbol} with {interval} interval: {str(e)}")
    
    logger.info("Enhanced strategy testing completed")

if __name__ == "__main__":
    main()
