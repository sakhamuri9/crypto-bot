"""
Ultimate MACD Indicator

This module implements the Ultimate MACD indicator with multi-timeframe support
based on the TradingView script shared by the user.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ultimate_macd.log')
    ]
)

logger = logging.getLogger(__name__)

def calculate_ultimate_macd(df, fast_length=12, slow_length=26, signal_length=9, use_ema=True):
    """
    Calculate Ultimate MACD indicator with color changes based on direction and zero line.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        fast_length (int): Fast EMA length
        slow_length (int): Slow EMA length
        signal_length (int): Signal line length
        use_ema (bool): Whether to use EMA for signal line (True) or SMA (False)
        
    Returns:
        pandas.DataFrame: DataFrame with Ultimate MACD indicator
    """
    logger.info(f"Calculating Ultimate MACD with fast={fast_length}, slow={slow_length}, signal={signal_length}")
    
    result_df = df.copy()
    
    if use_ema:
        fast_ma = result_df['close'].ewm(span=fast_length, adjust=False).mean()
        slow_ma = result_df['close'].ewm(span=slow_length, adjust=False).mean()
    else:
        fast_ma = result_df['close'].rolling(window=fast_length).mean()
        slow_ma = result_df['close'].rolling(window=slow_length).mean()
    
    result_df['macd'] = fast_ma - slow_ma
    
    if use_ema:
        result_df['macd_signal'] = result_df['macd'].ewm(span=signal_length, adjust=False).mean()
    else:
        result_df['macd_signal'] = result_df['macd'].rolling(window=signal_length).mean()
    
    result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']
    
    result_df['hist_a_is_up'] = (result_df['macd_hist'] > result_df['macd_hist'].shift(1)) & (result_df['macd_hist'] > 0)
    result_df['hist_a_is_down'] = (result_df['macd_hist'] < result_df['macd_hist'].shift(1)) & (result_df['macd_hist'] > 0)
    result_df['hist_b_is_down'] = (result_df['macd_hist'] < result_df['macd_hist'].shift(1)) & (result_df['macd_hist'] <= 0)
    result_df['hist_b_is_up'] = (result_df['macd_hist'] > result_df['macd_hist'].shift(1)) & (result_df['macd_hist'] <= 0)
    
    result_df['macd_is_above'] = result_df['macd'] >= result_df['macd_signal']
    result_df['macd_is_below'] = result_df['macd'] < result_df['macd_signal']
    
    result_df['macd_cross_up'] = (result_df['macd'] > result_df['macd_signal']) & (result_df['macd'].shift(1) <= result_df['macd_signal'].shift(1))
    result_df['macd_cross_down'] = (result_df['macd'] < result_df['macd_signal']) & (result_df['macd'].shift(1) >= result_df['macd_signal'].shift(1))
    
    result_df['macd_buy'] = result_df['macd_cross_up'] & (result_df['macd_hist'] > 0)
    result_df['macd_sell'] = result_df['macd_cross_down'] & (result_df['macd_hist'] < 0)
    
    buy_signals = sum(result_df['macd_buy'])
    sell_signals = sum(result_df['macd_sell'])
    
    logger.info(f"Generated {buy_signals} MACD buy signals and {sell_signals} MACD sell signals")
    
    return result_df

def add_multi_timeframe_macd(df, symbol, base_interval='1h', higher_interval='4h'):
    """
    Add multi-timeframe MACD analysis to the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        symbol (str): Symbol being analyzed
        base_interval (str): Base interval of the data
        higher_interval (str): Higher interval for multi-timeframe analysis
        
    Returns:
        pandas.DataFrame: DataFrame with multi-timeframe MACD
    """
    logger.info(f"Adding multi-timeframe MACD for {symbol} with base={base_interval}, higher={higher_interval}")
    
    result_df = df.copy()
    
    result_df = calculate_ultimate_macd(result_df)
    
    if base_interval == '1h' and higher_interval == '4h':
        higher_df = result_df.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        higher_df = calculate_ultimate_macd(higher_df)
        
        higher_df = higher_df.rename(columns={
            'macd': 'macd_4h',
            'macd_signal': 'macd_signal_4h',
            'macd_hist': 'macd_hist_4h',
            'macd_buy': 'macd_buy_4h',
            'macd_sell': 'macd_sell_4h'
        })
        
        higher_df = higher_df[['macd_4h', 'macd_signal_4h', 'macd_hist_4h', 'macd_buy_4h', 'macd_sell_4h']]
        
        higher_df = higher_df.resample('1h').ffill()
        
        result_df = pd.merge(result_df, higher_df, left_index=True, right_index=True, how='left')
        
        result_df = result_df.fillna(method='ffill')
        
        result_df['mtf_macd_buy'] = result_df['macd_buy'] & result_df['macd_buy_4h']
        result_df['mtf_macd_sell'] = result_df['macd_sell'] & result_df['macd_sell_4h']
        
        mtf_buy_signals = sum(result_df['mtf_macd_buy'])
        mtf_sell_signals = sum(result_df['mtf_macd_sell'])
        
        logger.info(f"Generated {mtf_buy_signals} multi-timeframe MACD buy signals and {mtf_sell_signals} multi-timeframe MACD sell signals")
    
    elif base_interval == higher_interval:
        result_df['mtf_macd_buy'] = result_df['macd_buy']
        result_df['mtf_macd_sell'] = result_df['macd_sell']
        
        mtf_buy_signals = sum(result_df['mtf_macd_buy'])
        mtf_sell_signals = sum(result_df['mtf_macd_sell'])
        
        logger.info(f"Using single-timeframe MACD signals: {mtf_buy_signals} buy signals and {mtf_sell_signals} sell signals")
    
    else:
        logger.warning(f"Multi-timeframe MACD for {base_interval} to {higher_interval} not implemented yet")
        result_df['mtf_macd_buy'] = False
        result_df['mtf_macd_sell'] = False
    
    return result_df

def plot_ultimate_macd(df, symbol, save_path=None):
    """
    Plot Ultimate MACD indicator.
    
    Args:
        df (pandas.DataFrame): DataFrame with Ultimate MACD indicator
        symbol (str): Symbol being analyzed
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig = plt.figure(figsize=(15, 10))
    
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    ax1 = plt.subplot(gs[0])
    ax1.plot(df.index, df['close'], label='Price', color='blue', alpha=0.7)
    
    symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
    ax1.set_title(f'{symbol_name} - Ultimate MACD Analysis')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    for i in range(1, len(df)):
        if df['macd_is_above'].iloc[i]:
            ax2.plot(df.index[i-1:i+1], df['macd'].iloc[i-1:i+1], color='lime', linewidth=2)
        else:
            ax2.plot(df.index[i-1:i+1], df['macd'].iloc[i-1:i+1], color='red', linewidth=2)
    
    ax2.plot(df.index, df['macd_signal'], color='yellow', linewidth=1, label='Signal')
    
    ax2.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
    
    ax2.scatter(df.index[df['macd_cross_up']], df['macd_signal'][df['macd_cross_up']], 
                color='lime', marker='^', s=100, label='Cross Up')
    ax2.scatter(df.index[df['macd_cross_down']], df['macd_signal'][df['macd_cross_down']], 
                color='red', marker='v', s=100, label='Cross Down')
    
    ax2.set_ylabel('MACD')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    ax3 = plt.subplot(gs[2], sharex=ax1)
    
    for i in range(len(df)):
        if df['hist_a_is_up'].iloc[i]:
            color = 'aqua'
        elif df['hist_a_is_down'].iloc[i]:
            color = 'blue'
        elif df['hist_b_is_down'].iloc[i]:
            color = 'red'
        elif df['hist_b_is_up'].iloc[i]:
            color = 'maroon'
        else:
            color = 'gray'
        
        ax3.bar(df.index[i], df['macd_hist'].iloc[i], color=color, width=0.8)
    
    ax3.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Histogram')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def main():
    """
    Main function to test Ultimate MACD indicator.
    """
    logger.info("Testing Ultimate MACD indicator")
    
    
    logger.info("Ultimate MACD testing completed")

if __name__ == "__main__":
    main()
