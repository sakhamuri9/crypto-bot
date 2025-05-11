"""
Fix for the strategy comparison script to address the indexing error.
This script modifies the run_backtest function to handle the entry_date indexing properly.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_strategy_comparison.log')
    ]
)

logger = logging.getLogger(__name__)

def fix_run_backtest(df, buy_signal_col, sell_signal_col, initial_capital=10000.0, position_size=1.0):
    """
    Fixed version of run_backtest function that properly handles entry_date indexing.
    
    Args:
        df (pandas.DataFrame): DataFrame with signals
        buy_signal_col (str): Column name for buy signals
        sell_signal_col (str): Column name for sell signals
        initial_capital (float): Initial capital
        position_size (float): Position size as a percentage of capital
        
    Returns:
        tuple: (final_capital, trades, returns, equity_curve)
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_idx = 0  # Store the index position instead of using position
    trades = []
    equity_curve = [capital]
    
    for i in range(1, len(df)):
        if pd.isna(df['close'].iloc[i]):
            equity_curve.append(equity_curve[-1])
            continue
        
        # Close position on sell signal
        if position > 0 and df[sell_signal_col].iloc[i]:
            exit_price = df['close'].iloc[i]
            profit_loss = position * (exit_price - entry_price)
            capital += profit_loss
            trades.append({
                'entry_date': df.index[entry_idx],  # Use stored entry_idx
                'exit_date': df.index[i],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                'type': 'long'
            })
            position = 0
        
        # Open position on buy signal
        if position == 0 and df[buy_signal_col].iloc[i]:
            entry_price = df['close'].iloc[i]
            position_capital = capital * position_size
            position = position_capital / entry_price
            entry_idx = i  # Store the current index
        
        # Update equity curve
        if position > 0:
            current_price = df['close'].iloc[i]
            current_value = capital + position * (current_price - entry_price)
        else:
            current_value = capital
        
        equity_curve.append(current_value)
    
    # Close any open position at the end
    if position > 0:
        exit_price = df['close'].iloc[-1]
        profit_loss = position * (exit_price - entry_price)
        capital += profit_loss
        trades.append({
            'entry_date': df.index[entry_idx],  # Use stored entry_idx
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'profit_loss_pct': (exit_price / entry_price - 1) * 100,
            'type': 'long'
        })
    
    # Calculate returns
    returns = (capital / initial_capital - 1) * 100
    
    return capital, trades, returns, equity_curve

def main():
    """
    Main function to demonstrate the fix.
    """
    logger.info("This script contains the fixed run_backtest function that properly handles entry_date indexing")
    logger.info("To use this fix, replace the run_backtest function in strategy_comparison_fixed.py with this version")
    logger.info("The key change is storing the entry index (entry_idx) instead of using position as an offset")

if __name__ == "__main__":
    main()
