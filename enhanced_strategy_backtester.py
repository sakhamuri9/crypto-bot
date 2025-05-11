"""
Enhanced Strategy Backtester

This module implements a backtesting framework for the enhanced strategy with
institutional-grade features for improved win rates and returns.
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
from enhanced_strategy import (
    generate_enhanced_signals,
    calculate_dynamic_stop_loss,
    calculate_dynamic_take_profit,
    calculate_kelly_position_size,
    add_moving_averages
)
from ultimate_macd import add_multi_timeframe_macd
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_strategy_backtest.log')
    ]
)

logger = logging.getLogger(__name__)

class EnhancedStrategyBacktester:
    """
    Backtester for the enhanced strategy with institutional-grade features.
    """
    
    def __init__(self, initial_capital=10000.0, max_position_size=0.2, safety_factor=0.5):
        """
        Initialize the backtester.
        
        Args:
            initial_capital (float): Initial capital for the backtest
            max_position_size (float): Maximum position size as a percentage of capital
            safety_factor (float): Factor to reduce Kelly bet size for safety (0.5 = "Half Kelly")
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.safety_factor = safety_factor
        
    def run_backtest(self, df, use_enhanced_signals=True, use_dynamic_position_sizing=True):
        """
        Run backtest on the provided data.
        
        Args:
            df (pandas.DataFrame): DataFrame with enhanced signals
            use_enhanced_signals (bool): Whether to use enhanced signals or just Squeeze Momentum signals
            use_dynamic_position_sizing (bool): Whether to use dynamic position sizing
            
        Returns:
            tuple: (DataFrame with backtest results, dict of performance metrics, list of trades)
        """
        logger.info(f"Running backtest with enhanced_signals={use_enhanced_signals}, dynamic_position_sizing={use_dynamic_position_sizing}")
        
        results = df.copy()
        
        results['position'] = 0
        results['position_size'] = 0.0
        results['entry_price'] = np.nan
        results['stop_loss'] = np.nan
        results['take_profit'] = np.nan
        results['exit_price'] = np.nan
        results['pnl'] = 0.0
        results['capital'] = self.initial_capital
        results['equity'] = self.initial_capital
        
        if use_enhanced_signals:
            buy_signal_col = 'enhanced_buy'
            sell_signal_col = 'enhanced_sell'
        else:
            buy_signal_col = 'sqz_buy'
            sell_signal_col = 'sqz_sell'
        
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        capital = self.initial_capital
        trades = []
        
        win_rate = 0.5  # Start with 50% win rate assumption
        reward_risk_ratio = 2.0  # Start with 2:1 reward/risk assumption
        
        for i in range(1, len(results)):
            if position != 0:
                current_price = results['close'].iloc[i]
                
                if position > 0:  # Long position
                    if current_price <= stop_loss:  # Stop-loss hit
                        exit_price = stop_loss
                        position_size = results['position_size'].iloc[i-1]
                        pnl = (exit_price - entry_price) / entry_price * position * capital * position_size
                        capital += pnl
                        
                        trades.append({
                            'entry_date': results.index[i-position],
                            'exit_date': results.index[i],
                            'position': 'LONG',
                            'position_size': position_size,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': (exit_price - entry_price) / entry_price,
                            'exit_reason': 'STOP_LOSS'
                        })
                        
                        results.loc[results.index[i], 'position'] = 0
                        results.loc[results.index[i], 'position_size'] = 0.0
                        results.loc[results.index[i], 'exit_price'] = exit_price
                        results.loc[results.index[i], 'pnl'] = pnl
                        results.loc[results.index[i], 'capital'] = capital
                        
                        position = 0
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
                    
                    elif current_price >= take_profit:  # Take-profit hit
                        exit_price = take_profit
                        position_size = results['position_size'].iloc[i-1]
                        pnl = (exit_price - entry_price) / entry_price * position * capital * position_size
                        capital += pnl
                        
                        trades.append({
                            'entry_date': results.index[i-position],
                            'exit_date': results.index[i],
                            'position': 'LONG',
                            'position_size': position_size,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': (exit_price - entry_price) / entry_price,
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        
                        results.loc[results.index[i], 'position'] = 0
                        results.loc[results.index[i], 'position_size'] = 0.0
                        results.loc[results.index[i], 'exit_price'] = exit_price
                        results.loc[results.index[i], 'pnl'] = pnl
                        results.loc[results.index[i], 'capital'] = capital
                        
                        position = 0
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
                
                elif position < 0:  # Short position
                    if current_price >= stop_loss:  # Stop-loss hit
                        exit_price = stop_loss
                        position_size = results['position_size'].iloc[i-1]
                        pnl = (entry_price - exit_price) / entry_price * abs(position) * capital * position_size
                        capital += pnl
                        
                        trades.append({
                            'entry_date': results.index[i-abs(position)],
                            'exit_date': results.index[i],
                            'position': 'SHORT',
                            'position_size': position_size,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': (entry_price - exit_price) / entry_price,
                            'exit_reason': 'STOP_LOSS'
                        })
                        
                        results.loc[results.index[i], 'position'] = 0
                        results.loc[results.index[i], 'position_size'] = 0.0
                        results.loc[results.index[i], 'exit_price'] = exit_price
                        results.loc[results.index[i], 'pnl'] = pnl
                        results.loc[results.index[i], 'capital'] = capital
                        
                        position = 0
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
                    
                    elif current_price <= take_profit:  # Take-profit hit
                        exit_price = take_profit
                        position_size = results['position_size'].iloc[i-1]
                        pnl = (entry_price - exit_price) / entry_price * abs(position) * capital * position_size
                        capital += pnl
                        
                        trades.append({
                            'entry_date': results.index[i-abs(position)],
                            'exit_date': results.index[i],
                            'position': 'SHORT',
                            'position_size': position_size,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': (entry_price - exit_price) / entry_price,
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        
                        results.loc[results.index[i], 'position'] = 0
                        results.loc[results.index[i], 'position_size'] = 0.0
                        results.loc[results.index[i], 'exit_price'] = exit_price
                        results.loc[results.index[i], 'pnl'] = pnl
                        results.loc[results.index[i], 'capital'] = capital
                        
                        position = 0
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
            
            if position == 0:
                if len(trades) >= 5:
                    win_count = sum(1 for trade in trades if trade['pnl'] > 0)
                    win_rate = win_count / len(trades)
                    
                    wins = [trade['pnl_pct'] for trade in trades if trade['pnl'] > 0]
                    losses = [trade['pnl_pct'] for trade in trades if trade['pnl'] <= 0]
                    
                    avg_win = sum(wins) / len(wins) if wins else 0.04
                    avg_loss = sum(losses) / len(losses) if losses else 0.02
                    
                    reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 2.0
                
                if use_dynamic_position_sizing:
                    position_size = calculate_kelly_position_size(
                        win_rate, 
                        reward_risk_ratio,
                        self.max_position_size,
                        self.safety_factor
                    )
                else:
                    position_size = 0.1  # Fixed 10% position size
                
                if results[buy_signal_col].iloc[i]:
                    position = 1
                    entry_price = results['close'].iloc[i]
                    
                    if 'dynamic_stop_loss' in results.columns and not pd.isna(results['dynamic_stop_loss'].iloc[i]):
                        stop_loss = results['dynamic_stop_loss'].iloc[i]
                    else:
                        stop_loss = entry_price * 0.98  # Default 2% stop-loss
                    
                    if 'dynamic_take_profit' in results.columns and not pd.isna(results['dynamic_take_profit'].iloc[i]):
                        take_profit = results['dynamic_take_profit'].iloc[i]
                    else:
                        take_profit = entry_price * 1.04  # Default 4% take-profit
                    
                    results.loc[results.index[i], 'position'] = position
                    results.loc[results.index[i], 'position_size'] = position_size
                    results.loc[results.index[i], 'entry_price'] = entry_price
                    results.loc[results.index[i], 'stop_loss'] = stop_loss
                    results.loc[results.index[i], 'take_profit'] = take_profit
                
                elif results[sell_signal_col].iloc[i]:
                    position = -1
                    entry_price = results['close'].iloc[i]
                    
                    if 'dynamic_stop_loss' in results.columns and not pd.isna(results['dynamic_stop_loss'].iloc[i]):
                        stop_loss = results['dynamic_stop_loss'].iloc[i]
                    else:
                        stop_loss = entry_price * 1.02  # Default 2% stop-loss
                    
                    if 'dynamic_take_profit' in results.columns and not pd.isna(results['dynamic_take_profit'].iloc[i]):
                        take_profit = results['dynamic_take_profit'].iloc[i]
                    else:
                        take_profit = entry_price * 0.96  # Default 4% take-profit
                    
                    results.loc[results.index[i], 'position'] = position
                    results.loc[results.index[i], 'position_size'] = position_size
                    results.loc[results.index[i], 'entry_price'] = entry_price
                    results.loc[results.index[i], 'stop_loss'] = stop_loss
                    results.loc[results.index[i], 'take_profit'] = take_profit
            
            if position != 0:
                current_price = results['close'].iloc[i]
                position_size = results['position_size'].iloc[i]
                
                if position > 0:
                    unrealized_pnl = (current_price - entry_price) / entry_price * position * capital * position_size
                else:
                    unrealized_pnl = (entry_price - current_price) / entry_price * abs(position) * capital * position_size
                
                results.loc[results.index[i], 'equity'] = capital + unrealized_pnl
            else:
                results.loc[results.index[i], 'equity'] = capital
        
        if position != 0:
            current_price = results['close'].iloc[-1]
            position_size = results['position_size'].iloc[-1]
            
            if position > 0:
                exit_price = current_price
                pnl = (exit_price - entry_price) / entry_price * position * capital * position_size
                capital += pnl
                
                trades.append({
                    'entry_date': results.index[len(results)-position],
                    'exit_date': results.index[-1],
                    'position': 'LONG',
                    'position_size': position_size,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': (exit_price - entry_price) / entry_price,
                    'exit_reason': 'END_OF_BACKTEST'
                })
                
                results.loc[results.index[-1], 'position'] = 0
                results.loc[results.index[-1], 'position_size'] = 0.0
                results.loc[results.index[-1], 'exit_price'] = exit_price
                results.loc[results.index[-1], 'pnl'] = pnl
                results.loc[results.index[-1], 'capital'] = capital
            
            elif position < 0:
                exit_price = current_price
                pnl = (entry_price - exit_price) / entry_price * abs(position) * capital * position_size
                capital += pnl
                
                trades.append({
                    'entry_date': results.index[len(results)-abs(position)],
                    'exit_date': results.index[-1],
                    'position': 'SHORT',
                    'position_size': position_size,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': (entry_price - exit_price) / entry_price,
                    'exit_reason': 'END_OF_BACKTEST'
                })
                
                results.loc[results.index[-1], 'position'] = 0
                results.loc[results.index[-1], 'position_size'] = 0.0
                results.loc[results.index[-1], 'exit_price'] = exit_price
                results.loc[results.index[-1], 'pnl'] = pnl
                results.loc[results.index[-1], 'capital'] = capital
        
        metrics = self.calculate_performance_metrics(results, trades)
        
        logger.info(f"Backtest completed with {metrics['total_trades']} trades and {metrics['total_return']:.2%} return")
        
        return results, metrics, trades
    
    def calculate_performance_metrics(self, results, trades):
        """
        Calculate performance metrics for the backtest.
        
        Args:
            results (pandas.DataFrame): DataFrame with backtest results
            trades (list): List of trade dictionaries
            
        Returns:
            dict: Performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0.0,
                'annualized_return': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0
            }
        
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        losing_trades = sum(1 for trade in trades if trade['pnl'] <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        initial_capital = self.initial_capital
        final_capital = results['capital'].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        
        days = (results.index[-1] - results.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0
        
        profits = [trade['pnl'] for trade in trades if trade['pnl'] > 0]
        losses = [trade['pnl'] for trade in trades if trade['pnl'] <= 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        total_profit = sum(profits)
        total_loss = abs(sum(losses))
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
        
        equity_curve = results['equity']
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        daily_returns = results['equity'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0.0
        
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = daily_returns.mean() * 252 / downside_deviation if downside_deviation != 0 else 0.0
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    def plot_backtest_results(self, results, metrics, trades, symbol, interval, save_path=None):
        """
        Plot backtest results.
        
        Args:
            results (pandas.DataFrame): DataFrame with backtest results
            metrics (dict): Performance metrics
            trades (list): List of trade dictionaries
            symbol (str): Symbol being analyzed
            interval (str): Interval of the data
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig = plt.figure(figsize=(15, 12))
        
        gs = plt.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
        
        ax1 = plt.subplot(gs[0])
        ax1.plot(results.index, results['close'], label='Price', color='blue', alpha=0.7)
        
        if 'resistance' in results.columns:
            resistance_values = results['resistance'].dropna().unique()
            current_price = results['close'].iloc[-1]
            resistance_above = [r for r in resistance_values if r > current_price]
            for i, level in enumerate(sorted(resistance_above)[:3]):
                ax1.axhline(y=level, color='red', linestyle='--', alpha=0.7)
        
        if 'support' in results.columns:
            support_values = results['support'].dropna().unique()
            current_price = results['close'].iloc[-1]
            support_below = [s for s in support_values if s < current_price]
            for i, level in enumerate(sorted(support_below, reverse=True)[:3]):
                ax1.axhline(y=level, color='green', linestyle='--', alpha=0.7)
        
        for trade in trades:
            if trade['position'] == 'LONG':
                ax1.scatter(trade['entry_date'], trade['entry_price'], marker='o', color='green', s=100)
                if trade['exit_reason'] == 'TAKE_PROFIT':
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='green', s=100)
                elif trade['exit_reason'] == 'STOP_LOSS':
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='red', s=100)
                else:
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='blue', s=100)
            else:  # SHORT
                ax1.scatter(trade['entry_date'], trade['entry_price'], marker='o', color='red', s=100)
                if trade['exit_reason'] == 'TAKE_PROFIT':
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='green', s=100)
                elif trade['exit_reason'] == 'STOP_LOSS':
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='red', s=100)
                else:
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='blue', s=100)
        
        symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
        ax1.set_title(f'{symbol_name} ({interval}) - Enhanced Strategy Backtest')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(results.index, results['equity'], label='Equity', color='green')
        ax2.set_ylabel('Equity')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        ax3 = plt.subplot(gs[2], sharex=ax1)
        
        colors = []
        for i in range(len(results)):
            if pd.isna(results['squeeze_momentum'].iloc[i]):
                colors.append('gray')
            elif results['squeeze_momentum'].iloc[i] >= 0:
                if i > 0 and results['squeeze_momentum'].iloc[i] > results['squeeze_momentum'].iloc[i-1]:
                    colors.append('lime')  # Increasing positive momentum
                else:
                    colors.append('green')  # Decreasing positive momentum
            else:
                if i > 0 and results['squeeze_momentum'].iloc[i] < results['squeeze_momentum'].iloc[i-1]:
                    colors.append('red')  # Increasing negative momentum
                else:
                    colors.append('maroon')  # Decreasing negative momentum
        
        ax3.bar(results.index, results['squeeze_momentum'], color=colors, width=0.8)
        ax3.set_ylabel('Momentum')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(gs[3], sharex=ax1)
        
        for i in range(len(results)):
            if pd.isna(results['squeeze_momentum'].iloc[i]):
                continue
                
            if results['sqz_on'].iloc[i]:
                ax4.scatter(results.index[i], 0, color='black', marker='x', s=30)
            elif results['sqz_off'].iloc[i]:
                ax4.scatter(results.index[i], 0, color='gray', marker='x', s=30)
            else:
                ax4.scatter(results.index[i], 0, color='blue', marker='x', s=30)
        
        ax4.set_yticks([])
        ax4.set_ylabel('Squeeze')
        ax4.grid(True, alpha=0.3)
        
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annualized Return: {metrics['annualized_return']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Total Trades: {metrics['total_trades']}"
        )
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig

def run_enhanced_strategy_backtest(symbol, timeframe='hours', interval='1h', limit=2000):
    """
    Run a backtest of the enhanced strategy.
    
    Args:
        symbol (str): Symbol to backtest (e.g., 'BTC-USDT-VANILLA-PERPETUAL')
        timeframe (str): Data timeframe ('hours' or 'days')
        interval (str): Interval for the data (e.g., '1h', '4h', '1d')
        limit (int): Maximum number of records to return
        
    Returns:
        tuple: (DataFrame with backtest results, dict of performance metrics, list of trades)
    """
    logger.info(f"Running enhanced strategy backtest for {symbol} with {interval} interval")
    
    df = get_data(symbol, timeframe, interval, limit)
    
    enhanced_df = generate_enhanced_signals(df, symbol, interval)
    
    enhanced_df = calculate_dynamic_stop_loss(enhanced_df, 'long')
    enhanced_df = calculate_dynamic_take_profit(enhanced_df, 'long')
    
    backtester = EnhancedStrategyBacktester(
        initial_capital=10000.0,
        max_position_size=0.2,
        safety_factor=0.5
    )
    
    enhanced_results, enhanced_metrics, enhanced_trades = backtester.run_backtest(
        enhanced_df,
        use_enhanced_signals=True,
        use_dynamic_position_sizing=True
    )
    
    combined_results, combined_metrics, combined_trades = backtester.run_backtest(
        enhanced_df,
        use_enhanced_signals=False,
        use_dynamic_position_sizing=False
    )
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    enhanced_plot_path = f"results/enhanced_backtest_{symbol.replace('-', '_')}_{interval}_{timestamp}.png"
    backtester.plot_backtest_results(
        enhanced_results,
        enhanced_metrics,
        enhanced_trades,
        symbol,
        interval,
        save_path=enhanced_plot_path
    )
    
    combined_plot_path = f"results/combined_backtest_{symbol.replace('-', '_')}_{interval}_{timestamp}.png"
    backtester.plot_backtest_results(
        combined_results,
        combined_metrics,
        combined_trades,
        symbol,
        interval,
        save_path=combined_plot_path
    )
    
    metrics_df = pd.DataFrame({
        'Metric': list(enhanced_metrics.keys()),
        'Enhanced Strategy': list(enhanced_metrics.values()),
        'Combined Strategy': list(combined_metrics.values())
    })
    
    metrics_path = f"results/backtest_comparison_{symbol.replace('-', '_')}_{interval}_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    logger.info(f"Enhanced Strategy: {enhanced_metrics['total_return']:.2%} return, {enhanced_metrics['win_rate']:.2%} win rate, {enhanced_metrics['total_trades']} trades")
    logger.info(f"Combined Strategy: {combined_metrics['total_return']:.2%} return, {combined_metrics['win_rate']:.2%} win rate, {combined_metrics['total_trades']} trades")
    
    return enhanced_results, enhanced_metrics, enhanced_trades, combined_results, combined_metrics, combined_trades

def main():
    """
    Main function to run enhanced strategy backtests.
    """
    logger.info("Starting enhanced strategy backtests")
    
    os.makedirs('results', exist_ok=True)
    
    symbols = [
        ('BTC-USDT-VANILLA-PERPETUAL', 'hours', '1h'),
        ('BTC-USDT-VANILLA-PERPETUAL', 'hours', '4h'),
        ('SUI-USDT-VANILLA-PERPETUAL', 'hours', '1h'),
        ('SUI-USDT-VANILLA-PERPETUAL', 'hours', '4h')
    ]
    
    all_results = {}
    
    for symbol, timeframe, interval in symbols:
        logger.info(f"Processing {symbol} with {interval} interval")
        
        enhanced_results, enhanced_metrics, enhanced_trades, combined_results, combined_metrics, combined_trades = run_enhanced_strategy_backtest(
            symbol,
            timeframe,
            interval
        )
        
        all_results[f"{symbol}_{interval}"] = {
            'enhanced': {
                'results': enhanced_results,
                'metrics': enhanced_metrics,
                'trades': enhanced_trades
            },
            'combined': {
                'results': combined_results,
                'metrics': combined_metrics,
                'trades': combined_trades
            }
        }
        
        symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
        print(f"\n{'='*50}")
        print(f"BACKTEST SUMMARY FOR {symbol_name} ({interval})")
        print(f"{'='*50}")
        print(f"Enhanced Strategy:")
        print(f"  Total Return: {enhanced_metrics['total_return']:.2%}")
        print(f"  Annualized Return: {enhanced_metrics['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {enhanced_metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {enhanced_metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {enhanced_metrics['win_rate']:.2%}")
        print(f"  Total Trades: {enhanced_metrics['total_trades']}")
        print(f"\nCombined Strategy:")
        print(f"  Total Return: {combined_metrics['total_return']:.2%}")
        print(f"  Annualized Return: {combined_metrics['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {combined_metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {combined_metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {combined_metrics['win_rate']:.2%}")
        print(f"  Total Trades: {combined_metrics['total_trades']}")
        print(f"{'='*50}\n")
    
    logger.info("Enhanced strategy backtests completed")
    
    return all_results

if __name__ == "__main__":
    main()
