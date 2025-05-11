"""
Combined Strategy Module

This module implements a trading strategy that combines support/resistance levels
with the Squeeze Momentum indicator to generate more reliable buy and sell signals.
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
from squeeze_momentum import add_squeeze_momentum, plot_squeeze_momentum
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('combined_strategy.log')
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
    df = None
    
    if interval == '1h' or interval == '1d':
        try:
            df = client.get_historical_klines(
                symbol=symbol,
                market="binance",
                limit=limit,
                interval=interval,
                timeframe=timeframe
            )
            
            logger.info(f"Collected {len(df)} candles from CoinDesk")
            
        except Exception as e:
            logger.warning(f"Error fetching {interval} data: {str(e)}")
            df = None
    
    if df is None or interval == '4h':
        logger.info(f"Using 1h data and resampling to {interval}")
        
        try:
            # Get 1h data with more candles for resampling, but respect API limits
            resample_limit = min(2000, limit * 4)
            df = client.get_historical_klines(
                symbol=symbol,
                market="binance",
                limit=resample_limit,
                interval="1h",
                timeframe="hours"
            )
            
            if len(df) == 0:
                raise ValueError(f"No data returned for {symbol} with 1h interval")
                
            logger.info(f"Collected {len(df)} 1h candles from CoinDesk for resampling")
            
            if interval == '4h':
                df = df.resample('4H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                logger.info(f"Resampled to {len(df)} 4h candles")
            elif interval == '1d':
                df = df.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                logger.info(f"Resampled to {len(df)} 1d candles")
                
        except Exception as e:
            logger.error(f"Failed to get data for {symbol}: {str(e)}")
            raise ValueError(f"Could not retrieve data for {symbol} with {interval} interval")
    
    if len(df) == 0:
        raise ValueError(f"No data available for {symbol} with {interval} interval")
    
    logger.info("Processing data and adding technical indicators")
    processed_df = add_indicators(df)
    logger.info(f"Processed {len(processed_df)} rows of data")
    
    return processed_df

def generate_combined_signals(df, symbol, timeframe='1h'):
    """
    Generate trading signals using a combination of support/resistance levels
    and the Squeeze Momentum indicator.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        symbol (str): Symbol being analyzed
        timeframe (str): Timeframe of the data (e.g., '1h', '4h', '1d')
        
    Returns:
        pandas.DataFrame: DataFrame with combined signals
    """
    logger.info(f"Generating combined signals for {symbol} with {timeframe} timeframe")
    
    sr_df = detect_support_resistance_levels(df.copy(), symbol, timeframe)
    
    sqz_df = add_squeeze_momentum(sr_df.copy())
    
    result_df = sqz_df.copy()
    
    current_price = result_df['close'].iloc[-1]
    
    resistance_levels = []
    support_levels = []
    
    if 'resistance' in result_df.columns:
        resistance_values = result_df['resistance'].dropna().unique()
        resistance_above = [r for r in resistance_values if r > current_price]
        resistance_levels = sorted(resistance_above)
    
    if 'support' in result_df.columns:
        support_values = result_df['support'].dropna().unique()
        support_below = [s for s in support_values if s < current_price]
        support_levels = sorted(support_below, reverse=True)
    
    nearest_resistance = resistance_levels[0] if resistance_levels else None
    nearest_support = support_levels[0] if support_levels else None
    
    if nearest_resistance:
        result_df['resistance_distance'] = (nearest_resistance - result_df['close']) / result_df['close'] * 100
    else:
        result_df['resistance_distance'] = np.nan
        
    if nearest_support:
        result_df['support_distance'] = (result_df['close'] - nearest_support) / result_df['close'] * 100
    else:
        result_df['support_distance'] = np.nan
    
    result_df['combined_buy'] = False
    result_df['combined_sell'] = False
    
    if nearest_support is not None:
        result_df['combined_buy'] = (
            result_df['sqz_buy'] & 
            (result_df['support_distance'] < 2.0)
        )
    
    result_df['combined_buy_strong'] = False
    if nearest_support is not None:
        result_df['combined_buy_strong'] = (
            result_df['sqz_buy_strong'] & 
            (result_df['support_distance'] < 1.0)
        )
    
    if nearest_resistance is not None:
        result_df['combined_sell'] = (
            result_df['sqz_sell'] & 
            (result_df['resistance_distance'] < 2.0)
        )
    
    result_df['combined_sell_strong'] = False
    if nearest_resistance is not None:
        result_df['combined_sell_strong'] = (
            result_df['sqz_sell_strong'] & 
            (result_df['resistance_distance'] < 1.0)
        )
    
    logger.info(f"Generated {sum(result_df['combined_buy'])} combined buy signals and {sum(result_df['combined_sell'])} combined sell signals")
    
    return result_df

def plot_combined_strategy(df, symbol, timeframe='1h'):
    """
    Plot the combined strategy with support/resistance levels and Squeeze Momentum.
    
    Args:
        df (pandas.DataFrame): DataFrame with combined signals
        symbol (str): Symbol being analyzed
        timeframe (str): Timeframe of the data (e.g., '1h', '4h', '1d')
        
    Returns:
        str: Path to the saved plot
    """
    logger.info(f"Plotting combined strategy for {symbol} with {timeframe} timeframe")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(df.index[-200:], df['close'].iloc[-200:], label='Close Price', color='blue', alpha=0.7)
    
    current_price = df['close'].iloc[-1]
    ax1.axhline(y=current_price, color='black', linestyle='-', alpha=0.5, label=f'Current Price: {current_price:.2f}')
    
    if 'resistance' in df.columns:
        resistance_values = df['resistance'].dropna().unique()
        resistance_above = [r for r in resistance_values if r > current_price]
        for i, level in enumerate(sorted(resistance_above)[:5]):
            distance = ((level - current_price) / current_price) * 100
            ax1.axhline(y=level, color='red', linestyle='--', alpha=0.7, 
                        label=f'R{i+1}: {level:.2f} (+{distance:.2f}%)')
    
    if 'support' in df.columns:
        support_values = df['support'].dropna().unique()
        support_below = [s for s in support_values if s < current_price]
        for i, level in enumerate(sorted(support_below, reverse=True)[:5]):
            distance = ((current_price - level) / current_price) * 100
            ax1.axhline(y=level, color='green', linestyle='--', alpha=0.7,
                        label=f'S{i+1}: {level:.2f} (-{distance:.2f}%)')
    
    for i in range(max(0, len(df)-200), len(df)):
        if df['combined_buy_strong'].iloc[i]:
            ax1.scatter(df.index[i], df['low'].iloc[i] * 0.99, color='lime', marker='^', s=100)
        elif df['combined_buy'].iloc[i]:
            ax1.scatter(df.index[i], df['low'].iloc[i] * 0.99, color='green', marker='^', s=60)
        
        if df['combined_sell_strong'].iloc[i]:
            ax1.scatter(df.index[i], df['high'].iloc[i] * 1.01, color='red', marker='v', s=100)
        elif df['combined_sell'].iloc[i]:
            ax1.scatter(df.index[i], df['high'].iloc[i] * 1.01, color='maroon', marker='v', s=60)
    
    symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
    ax1.set_title(f'{symbol_name} ({timeframe}) - Combined Strategy')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    colors = []
    for i in range(len(df)):
        if pd.isna(df['squeeze_momentum'].iloc[i]):
            colors.append('gray')
        elif df['squeeze_momentum'].iloc[i] >= 0:
            if i > 0 and df['squeeze_momentum'].iloc[i] > df['squeeze_momentum'].iloc[i-1]:
                colors.append('lime')  # Increasing positive momentum
            else:
                colors.append('green')  # Decreasing positive momentum
        else:
            if i > 0 and df['squeeze_momentum'].iloc[i] < df['squeeze_momentum'].iloc[i-1]:
                colors.append('red')  # Increasing negative momentum
            else:
                colors.append('maroon')  # Decreasing negative momentum
    
    ax2.bar(df.index[-200:], df['squeeze_momentum'].iloc[-200:], color=colors[-200:], width=0.8)
    
    for i in range(max(0, len(df)-200), len(df)):
        if pd.isna(df['squeeze_momentum'].iloc[i]):
            continue
            
        if df['sqz_on'].iloc[i]:
            ax2.scatter(df.index[i], 0, color='black', marker='x', s=30)
        elif df['sqz_off'].iloc[i]:
            ax2.scatter(df.index[i], 0, color='gray', marker='x', s=30)
        else:
            ax2.scatter(df.index[i], 0, color='blue', marker='x', s=30)
    
    ax2.set_title("Squeeze Momentum")
    ax2.set_ylabel('Momentum')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='lime', label='Increasing Positive Momentum'),
        Patch(facecolor='green', label='Decreasing Positive Momentum'),
        Patch(facecolor='red', label='Increasing Negative Momentum'),
        Patch(facecolor='maroon', label='Decreasing Negative Momentum'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=8, label='Squeeze On'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markersize=8, label='Squeeze Off'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='blue', markersize=8, label='No Squeeze'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='lime', markersize=10, label='Strong Buy Signal'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Buy Signal'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Strong Sell Signal'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='maroon', markersize=8, label='Sell Signal')
    ]
    
    ax2.legend(handles=legend_elements, loc='upper right', ncol=2)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/combined_strategy_{symbol.replace('-', '_')}_{timeframe}_{timestamp}.png"
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    
    logger.info(f"Plot saved to {save_path}")
    return save_path

def analyze_combined_signals(df, symbol, timeframe='1h'):
    """
    Analyze the combined signals and generate a summary.
    
    Args:
        df (pandas.DataFrame): DataFrame with combined signals
        symbol (str): Symbol being analyzed
        timeframe (str): Timeframe of the data (e.g., '1h', '4h', '1d')
        
    Returns:
        str: Analysis text
    """
    symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
    
    analysis = f"\n{'='*50}\n"
    analysis += f"COMBINED STRATEGY ANALYSIS FOR {symbol_name} ({timeframe})\n"
    analysis += f"{'='*50}\n\n"
    
    current_price = df['close'].iloc[-1]
    analysis += f"Current Price: {current_price:.2f}\n\n"
    
    total_sqz_buy = sum(df['sqz_buy'])
    total_sqz_sell = sum(df['sqz_sell'])
    total_combined_buy = sum(df['combined_buy'])
    total_combined_sell = sum(df['combined_sell'])
    total_combined_buy_strong = sum(df['combined_buy_strong'])
    total_combined_sell_strong = sum(df['combined_sell_strong'])
    
    analysis += f"SIGNAL SUMMARY:\n"
    analysis += f"{'-'*50}\n"
    analysis += f"Squeeze Momentum Buy Signals: {total_sqz_buy}\n"
    analysis += f"Squeeze Momentum Sell Signals: {total_sqz_sell}\n"
    analysis += f"Combined Buy Signals: {total_combined_buy}\n"
    analysis += f"Combined Sell Signals: {total_combined_sell}\n"
    analysis += f"Strong Combined Buy Signals: {total_combined_buy_strong}\n"
    analysis += f"Strong Combined Sell Signals: {total_combined_sell_strong}\n\n"
    
    recent_df = df.iloc[-20:]
    recent_buy = sum(recent_df['combined_buy'])
    recent_sell = sum(recent_df['combined_sell'])
    recent_buy_strong = sum(recent_df['combined_buy_strong'])
    recent_sell_strong = sum(recent_df['combined_sell_strong'])
    
    analysis += f"RECENT SIGNALS (Last 20 periods):\n"
    analysis += f"{'-'*50}\n"
    analysis += f"Recent Buy Signals: {recent_buy}\n"
    analysis += f"Recent Sell Signals: {recent_sell}\n"
    analysis += f"Recent Strong Buy Signals: {recent_buy_strong}\n"
    analysis += f"Recent Strong Sell Signals: {recent_sell_strong}\n\n"
    
    current_sqz_on = df['sqz_on'].iloc[-1]
    current_sqz_off = df['sqz_off'].iloc[-1]
    current_momentum = df['squeeze_momentum'].iloc[-1]
    momentum_direction = "Increasing" if df['momentum_increasing'].iloc[-1] else "Decreasing"
    
    analysis += f"CURRENT MARKET STATE:\n"
    analysis += f"{'-'*50}\n"
    analysis += f"Squeeze State: {'Squeeze ON' if current_sqz_on else 'Squeeze OFF' if current_sqz_off else 'No Squeeze'}\n"
    analysis += f"Momentum: {current_momentum:.4f} ({momentum_direction})\n"
    
    current_signal = "NEUTRAL"
    if df['combined_buy_strong'].iloc[-1]:
        current_signal = "STRONG BUY"
    elif df['combined_buy'].iloc[-1]:
        current_signal = "BUY"
    elif df['combined_sell_strong'].iloc[-1]:
        current_signal = "STRONG SELL"
    elif df['combined_sell'].iloc[-1]:
        current_signal = "SELL"
    
    analysis += f"Current Signal: {current_signal}\n\n"
    
    analysis += f"NEAREST SUPPORT AND RESISTANCE LEVELS:\n"
    analysis += f"{'-'*50}\n"
    
    if 'resistance' in df.columns:
        resistance_values = df['resistance'].dropna().unique()
        resistance_above = [r for r in resistance_values if r > current_price]
        if resistance_above:
            for i, level in enumerate(sorted(resistance_above)[:3]):
                distance = ((level - current_price) / current_price) * 100
                analysis += f"R{i+1}: {level:.2f} (+{distance:.2f}%)\n"
        else:
            analysis += "No resistance levels detected above current price.\n"
    
    if 'support' in df.columns:
        support_values = df['support'].dropna().unique()
        support_below = [s for s in support_values if s < current_price]
        if support_below:
            for i, level in enumerate(sorted(support_below, reverse=True)[:3]):
                distance = ((current_price - level) / current_price) * 100
                analysis += f"S{i+1}: {level:.2f} (-{distance:.2f}%)\n"
        else:
            analysis += "No support levels detected below current price.\n"
    
    analysis += f"\n{'='*50}\n"
    
    return analysis

def main():
    """
    Main function to run the combined strategy.
    """
    logger.info("Starting combined strategy analysis")
    
    os.makedirs('results', exist_ok=True)
    
    symbols = [
        ('BTC-USDT-VANILLA-PERPETUAL', 'hours', '1h'),
        ('BTC-USDT-VANILLA-PERPETUAL', 'hours', '4h'),
        ('SUI-USDT-VANILLA-PERPETUAL', 'hours', '1h'),
        ('SUI-USDT-VANILLA-PERPETUAL', 'hours', '4h')
    ]
    
    plot_paths = []
    analysis_texts = []
    
    for symbol, timeframe, interval in symbols:
        logger.info(f"Processing {symbol} with {interval} interval")
        
        df = get_data(symbol, timeframe, interval)
        
        combined_df = generate_combined_signals(df, symbol, interval)
        
        plot_path = plot_combined_strategy(combined_df, symbol, interval)
        plot_paths.append(plot_path)
        
        analysis = analyze_combined_signals(combined_df, symbol, interval)
        analysis_texts.append(analysis)
        
        print(analysis)
    
    logger.info("Combined strategy analysis completed")
    
    return plot_paths, analysis_texts

if __name__ == "__main__":
    main()
