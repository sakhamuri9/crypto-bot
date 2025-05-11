"""
Hedge Fund Strategy

This module implements an institutional-grade trading strategy with advanced
risk management and signal generation techniques used by hedge funds.
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
from ultimate_macd import add_multi_timeframe_macd
from enhanced_strategy import (
    generate_enhanced_signals,
    calculate_dynamic_stop_loss,
    calculate_dynamic_take_profit,
    calculate_kelly_position_size,
    add_moving_averages
)
from combined_strategy import get_data
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hedge_fund_strategy.log')
    ]
)

logger = logging.getLogger(__name__)

def calculate_volume_profile(df, num_bins=20):
    """
    Calculate volume profile to identify high-volume price levels.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        num_bins (int): Number of price bins for volume profile
        
    Returns:
        pandas.DataFrame: DataFrame with volume profile
    """
    result_df = df.copy()
    
    price_min = result_df['low'].min()
    price_max = result_df['high'].max()
    
    price_bins = np.linspace(price_min, price_max, num_bins + 1)
    
    volume_profile = np.zeros(num_bins)
    
    for i in range(len(result_df)):
        candle_min = result_df['low'].iloc[i]
        candle_max = result_df['high'].iloc[i]
        candle_volume = result_df['volume'].iloc[i]
        
        for j in range(num_bins):
            bin_min = price_bins[j]
            bin_max = price_bins[j + 1]
            
            if candle_max >= bin_min and candle_min <= bin_max:
                overlap = min(candle_max, bin_max) - max(candle_min, bin_min)
                candle_range = candle_max - candle_min
                
                if candle_range > 0:
                    ratio = overlap / candle_range
                    volume_profile[j] += candle_volume * ratio
    
    result_df['volume_profile_bin'] = pd.cut(result_df['close'], bins=price_bins, labels=False)
    
    result_df['volume_profile'] = result_df['volume_profile_bin'].apply(lambda x: volume_profile[int(x)] if not pd.isna(x) else np.nan)
    
    return result_df, price_bins, volume_profile

def calculate_correlation_matrix(symbols, timeframe='hours', interval='1h', limit=500):
    """
    Calculate correlation matrix between multiple symbols.
    
    Args:
        symbols (list): List of symbols to analyze
        timeframe (str): Data timeframe ('hours' or 'days')
        interval (str): Interval for the data (e.g., '1h', '4h', '1d')
        limit (int): Maximum number of records to return
        
    Returns:
        pandas.DataFrame: Correlation matrix
    """
    logger.info(f"Calculating correlation matrix for {len(symbols)} symbols")
    
    symbol_data = {}
    for symbol in symbols:
        df = get_data(symbol, timeframe, interval, limit)
        symbol_data[symbol] = df['close']
    
    prices_df = pd.DataFrame(symbol_data)
    
    returns_df = prices_df.pct_change().dropna()
    
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix

def generate_hedge_fund_signals(df, symbol, interval='1h', correlation_matrix=None):
    """
    Generate hedge fund grade trading signals.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data and indicators
        symbol (str): Symbol being analyzed
        interval (str): Interval of the data
        correlation_matrix (pandas.DataFrame, optional): Correlation matrix for risk management
        
    Returns:
        pandas.DataFrame: DataFrame with hedge fund signals
    """
    logger.info(f"Generating hedge fund signals for {symbol} with {interval} timeframe")
    
    result_df = detect_support_resistance_levels(df.copy(), symbol, interval)
    
    result_df = add_squeeze_momentum(result_df)
    
    result_df = add_multi_timeframe_macd(result_df, symbol, interval)
    
    result_df = add_moving_averages(result_df, periods=[20, 50, 200])
    
    result_df['atr'] = calculate_atr(result_df)
    
    result_df, price_bins, volume_profile = calculate_volume_profile(result_df)
    
    result_df['hedge_fund_buy'] = False
    result_df['hedge_fund_sell'] = False
    result_df['signal_strength'] = 0
    
    for i in range(1, len(result_df)):
        if pd.isna(result_df['atr'].iloc[i]) or pd.isna(result_df['sma_50'].iloc[i]):
            continue
        
        current_price = result_df['close'].iloc[i]
        atr_value = result_df['atr'].iloc[i]
        
        near_support = False
        support_strength = 0
        if 'support' in result_df.columns:
            support_values = result_df['support'].iloc[i-10:i+1].dropna().unique()
            support_below = [s for s in support_values if s < current_price]
            
            if support_below:
                closest_support = max(support_below)
                support_distance = (current_price - closest_support) / current_price
                
                near_support = support_distance < 0.02 or (current_price - closest_support) < 1.5 * atr_value
                
                if near_support:
                    support_strength = 5 - min(5, int(support_distance * 100))
        
        near_resistance = False
        resistance_strength = 0
        if 'resistance' in result_df.columns:
            resistance_values = result_df['resistance'].iloc[i-10:i+1].dropna().unique()
            resistance_above = [r for r in resistance_values if r > current_price]
            
            if resistance_above:
                closest_resistance = min(resistance_above)
                resistance_distance = (closest_resistance - current_price) / current_price
                
                near_resistance = resistance_distance < 0.02 or (closest_resistance - current_price) < 1.5 * atr_value
                
                if near_resistance:
                    resistance_strength = 5 - min(5, int(resistance_distance * 100))
        
        uptrend = result_df['close'].iloc[i] > result_df['sma_50'].iloc[i] > result_df['sma_200'].iloc[i]
        downtrend = result_df['close'].iloc[i] < result_df['sma_50'].iloc[i] < result_df['sma_200'].iloc[i]
        
        high_volume = result_df['volume'].iloc[i] > result_df['volume'].rolling(20).mean().iloc[i] * 1.5
        
        momentum_strength = 0
        if result_df['sqz_buy'].iloc[i]:
            momentum_strength = 2
        if result_df['macd_buy'].iloc[i]:
            momentum_strength += 1
        if result_df['mtf_macd_buy'].iloc[i]:
            momentum_strength += 2
        
        buy_strength = 0
        if near_support and uptrend:
            buy_strength = support_strength + momentum_strength
            if high_volume:
                buy_strength += 1
            
            if buy_strength >= 7:
                result_df.loc[result_df.index[i], 'hedge_fund_buy'] = True
                result_df.loc[result_df.index[i], 'signal_strength'] = buy_strength
        
        sell_strength = 0
        if near_resistance and downtrend:
            sell_strength = resistance_strength + momentum_strength
            if high_volume:
                sell_strength += 1
            
            if sell_strength >= 7:
                result_df.loc[result_df.index[i], 'hedge_fund_sell'] = True
                result_df.loc[result_df.index[i], 'signal_strength'] = sell_strength
    
    buy_signals = sum(result_df['hedge_fund_buy'])
    sell_signals = sum(result_df['hedge_fund_sell'])
    
    logger.info(f"Generated {buy_signals} hedge fund buy signals and {sell_signals} hedge fund sell signals")
    
    return result_df

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

def calculate_optimal_position_size(df, trade_type='long', risk_per_trade=0.01, win_rate=None, reward_risk_ratio=None):
    """
    Calculate optimal position size using Kelly criterion and ATR-based risk.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data and ATR
        trade_type (str): 'long' or 'short'
        risk_per_trade (float): Maximum risk per trade as percentage of capital
        win_rate (float, optional): Win rate as a decimal (e.g., 0.6 for 60%)
        reward_risk_ratio (float, optional): Average reward to risk ratio
        
    Returns:
        tuple: (position_size, stop_loss_price, take_profit_price)
    """
    current_price = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    
    stop_loss_distance = atr * 2
    
    if trade_type == 'long':
        stop_loss_price = current_price - stop_loss_distance
    else:  # short
        stop_loss_price = current_price + stop_loss_distance
    
    risk_per_share = abs(current_price - stop_loss_price)
    
    position_size_risk = risk_per_trade / (risk_per_share / current_price)
    
    if win_rate is not None and reward_risk_ratio is not None:
        kelly_fraction = win_rate - ((1 - win_rate) / reward_risk_ratio)
        
        kelly_position_size = max(0.01, kelly_fraction * 0.5)
        
        position_size = min(position_size_risk, kelly_position_size)
    else:
        position_size = position_size_risk
    
    position_size = min(position_size, 0.2)
    
    if reward_risk_ratio is not None:
        if trade_type == 'long':
            take_profit_price = current_price + (stop_loss_distance * reward_risk_ratio)
        else:  # short
            take_profit_price = current_price - (stop_loss_distance * reward_risk_ratio)
    else:
        if trade_type == 'long':
            take_profit_price = current_price + (stop_loss_distance * 2)
        else:  # short
            take_profit_price = current_price - (stop_loss_distance * 2)
    
    return position_size, stop_loss_price, take_profit_price

def main():
    """
    Main function to test hedge fund strategy.
    """
    logger.info("Testing hedge fund strategy")
    
    os.makedirs('results', exist_ok=True)
    
    symbols = [
        ('BTC-USDT-VANILLA-PERPETUAL', 'hours', '1h'),
        ('BTC-USDT-VANILLA-PERPETUAL', 'hours', '4h'),
        ('SUI-USDT-VANILLA-PERPETUAL', 'hours', '1h'),
        ('SUI-USDT-VANILLA-PERPETUAL', 'hours', '4h')
    ]
    
    correlation_symbols = ['BTC-USDT-VANILLA-PERPETUAL', 'SUI-USDT-VANILLA-PERPETUAL']
    correlation_matrix = calculate_correlation_matrix(correlation_symbols)
    
    for symbol, timeframe, interval in symbols:
        logger.info(f"Processing {symbol} with {interval} interval")
        
        try:
            df = get_data(symbol, timeframe, interval)
            
            hedge_fund_df = generate_hedge_fund_signals(df, symbol, interval, correlation_matrix)
            
            win_rate = 0.6  # Example win rate
            reward_risk_ratio = 2.0  # Example reward/risk ratio
            
            position_size, stop_loss_price, take_profit_price = calculate_optimal_position_size(
                hedge_fund_df,
                'long',
                0.01,
                win_rate,
                reward_risk_ratio
            )
            
            logger.info(f"Optimal position size: {position_size:.2%}")
            logger.info(f"Stop loss price: {stop_loss_price:.2f}")
            logger.info(f"Take profit price: {take_profit_price:.2f}")
            
            logger.info(f"Hedge fund strategy analysis completed for {symbol} with {interval} interval")
            
        except Exception as e:
            logger.error(f"Error processing {symbol} with {interval} interval: {str(e)}")
    
    logger.info("Hedge fund strategy testing completed")

if __name__ == "__main__":
    main()
