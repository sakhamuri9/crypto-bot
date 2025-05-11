"""
Combined Strategy Backtester

This module implements a backtesting framework for the combined strategy that integrates
support/resistance levels with the Squeeze Momentum indicator.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import time

from coindesk_client import CoinDeskClient
from indicators import add_indicators
from generic_support_resistance import detect_support_resistance_levels, detect_resistance_levels, detect_support_levels
from squeeze_momentum import add_squeeze_momentum
from combined_strategy import generate_combined_signals, get_data
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('combined_strategy_backtest.log')
    ]
)

logger = logging.getLogger(__name__)

class CombinedStrategyBacktester:
    """
    Backtester for the combined strategy that integrates support/resistance levels
    with the Squeeze Momentum indicator.
    """
    
    def __init__(self, initial_capital=10000.0, position_size=0.1, stop_loss_pct=0.02, take_profit_pct=0.04):
        """
        Initialize the backtester.
        
        Args:
            initial_capital (float): Initial capital for the backtest
            position_size (float): Position size as a percentage of capital (0.1 = 10%)
            stop_loss_pct (float): Stop loss percentage (0.02 = 2%)
            take_profit_pct (float): Take profit percentage (0.04 = 4%)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
    def run_backtest(self, df, use_combined_signals=True, use_strong_signals_only=False):
        """
        Run backtest on the provided data.
        
        Args:
            df (pandas.DataFrame): DataFrame with combined signals
            use_combined_signals (bool): Whether to use combined signals or just Squeeze Momentum signals
            use_strong_signals_only (bool): Whether to use only strong signals
            
        Returns:
            tuple: (DataFrame with backtest results, dict of performance metrics)
        """
        logger.info(f"Running backtest with combined_signals={use_combined_signals}, strong_signals_only={use_strong_signals_only}")
        
        results = df.copy()
        
        results['position'] = 0
        results['entry_price'] = np.nan
        results['stop_loss'] = np.nan
        results['take_profit'] = np.nan
        results['exit_price'] = np.nan
        results['pnl'] = 0.0
        results['capital'] = self.initial_capital
        results['equity'] = self.initial_capital
        
        if use_combined_signals:
            if use_strong_signals_only:
                buy_signal_col = 'combined_buy_strong'
                sell_signal_col = 'combined_sell_strong'
            else:
                buy_signal_col = 'combined_buy'
                sell_signal_col = 'combined_sell'
        else:
            if use_strong_signals_only:
                buy_signal_col = 'sqz_buy_strong'
                sell_signal_col = 'sqz_sell_strong'
            else:
                buy_signal_col = 'sqz_buy'
                sell_signal_col = 'sqz_sell'
        
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        capital = self.initial_capital
        trades = []
        
        for i in range(1, len(results)):
            if position != 0:
                current_price = results['close'].iloc[i]
                
                if position > 0 and current_price <= stop_loss:
                    exit_price = stop_loss
                    pnl = (exit_price - entry_price) / entry_price * position * capital * self.position_size
                    capital += pnl
                    
                    trades.append({
                        'entry_date': results.index[i-position],
                        'exit_date': results.index[i],
                        'position': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': (exit_price - entry_price) / entry_price,
                        'exit_reason': 'STOP_LOSS'
                    })
                    
                    results.loc[results.index[i], 'position'] = 0
                    results.loc[results.index[i], 'exit_price'] = exit_price
                    results.loc[results.index[i], 'pnl'] = pnl
                    results.loc[results.index[i], 'capital'] = capital
                    
                    position = 0
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
                
                elif position < 0 and current_price >= stop_loss:
                    exit_price = stop_loss
                    pnl = (entry_price - exit_price) / entry_price * abs(position) * capital * self.position_size
                    capital += pnl
                    
                    trades.append({
                        'entry_date': results.index[i-abs(position)],
                        'exit_date': results.index[i],
                        'position': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': (entry_price - exit_price) / entry_price,
                        'exit_reason': 'STOP_LOSS'
                    })
                    
                    results.loc[results.index[i], 'position'] = 0
                    results.loc[results.index[i], 'exit_price'] = exit_price
                    results.loc[results.index[i], 'pnl'] = pnl
                    results.loc[results.index[i], 'capital'] = capital
                    
                    position = 0
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
                
                elif position > 0 and current_price >= take_profit:
                    exit_price = take_profit
                    pnl = (exit_price - entry_price) / entry_price * position * capital * self.position_size
                    capital += pnl
                    
                    trades.append({
                        'entry_date': results.index[i-position],
                        'exit_date': results.index[i],
                        'position': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': (exit_price - entry_price) / entry_price,
                        'exit_reason': 'TAKE_PROFIT'
                    })
                    
                    results.loc[results.index[i], 'position'] = 0
                    results.loc[results.index[i], 'exit_price'] = exit_price
                    results.loc[results.index[i], 'pnl'] = pnl
                    results.loc[results.index[i], 'capital'] = capital
                    
                    position = 0
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
                
                elif position < 0 and current_price <= take_profit:
                    exit_price = take_profit
                    pnl = (entry_price - exit_price) / entry_price * abs(position) * capital * self.position_size
                    capital += pnl
                    
                    trades.append({
                        'entry_date': results.index[i-abs(position)],
                        'exit_date': results.index[i],
                        'position': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': (entry_price - exit_price) / entry_price,
                        'exit_reason': 'TAKE_PROFIT'
                    })
                    
                    results.loc[results.index[i], 'position'] = 0
                    results.loc[results.index[i], 'exit_price'] = exit_price
                    results.loc[results.index[i], 'pnl'] = pnl
                    results.loc[results.index[i], 'capital'] = capital
                    
                    position = 0
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
            
            if position == 0:
                if results[buy_signal_col].iloc[i]:
                    position = 1
                    entry_price = results['close'].iloc[i]
                    stop_loss = entry_price * (1 - self.stop_loss_pct)
                    take_profit = entry_price * (1 + self.take_profit_pct)
                    
                    results.loc[results.index[i], 'position'] = position
                    results.loc[results.index[i], 'entry_price'] = entry_price
                    results.loc[results.index[i], 'stop_loss'] = stop_loss
                    results.loc[results.index[i], 'take_profit'] = take_profit
                
                elif results[sell_signal_col].iloc[i]:
                    position = -1
                    entry_price = results['close'].iloc[i]
                    stop_loss = entry_price * (1 + self.stop_loss_pct)
                    take_profit = entry_price * (1 - self.take_profit_pct)
                    
                    results.loc[results.index[i], 'position'] = position
                    results.loc[results.index[i], 'entry_price'] = entry_price
                    results.loc[results.index[i], 'stop_loss'] = stop_loss
                    results.loc[results.index[i], 'take_profit'] = take_profit
            
            if position != 0:
                current_price = results['close'].iloc[i]
                if position > 0:
                    unrealized_pnl = (current_price - entry_price) / entry_price * position * capital * self.position_size
                else:
                    unrealized_pnl = (entry_price - current_price) / entry_price * abs(position) * capital * self.position_size
                
                results.loc[results.index[i], 'equity'] = capital + unrealized_pnl
            else:
                results.loc[results.index[i], 'equity'] = capital
        
        if position != 0:
            current_price = results['close'].iloc[-1]
            
            if position > 0:
                exit_price = current_price
                pnl = (exit_price - entry_price) / entry_price * position * capital * self.position_size
                capital += pnl
                
                trades.append({
                    'entry_date': results.index[len(results)-position],
                    'exit_date': results.index[-1],
                    'position': 'LONG',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': (exit_price - entry_price) / entry_price,
                    'exit_reason': 'END_OF_BACKTEST'
                })
                
                results.loc[results.index[-1], 'position'] = 0
                results.loc[results.index[-1], 'exit_price'] = exit_price
                results.loc[results.index[-1], 'pnl'] = pnl
                results.loc[results.index[-1], 'capital'] = capital
            
            elif position < 0:
                exit_price = current_price
                pnl = (entry_price - exit_price) / entry_price * abs(position) * capital * self.position_size
                capital += pnl
                
                trades.append({
                    'entry_date': results.index[len(results)-abs(position)],
                    'exit_date': results.index[-1],
                    'position': 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': (entry_price - exit_price) / entry_price,
                    'exit_reason': 'END_OF_BACKTEST'
                })
                
                results.loc[results.index[-1], 'position'] = 0
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
        sortino_ratio = daily_returns.mean() / negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 and negative_returns.std() != 0 else 0.0
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        metrics = {
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
        
        return metrics
    
    def plot_backtest_results(self, results, metrics, trades, symbol, timeframe, save_path=None):
        """
        Plot backtest results.
        
        Args:
            results (pandas.DataFrame): DataFrame with backtest results
            metrics (dict): Performance metrics
            trades (list): List of trade dictionaries
            symbol (str): Symbol being analyzed
            timeframe (str): Timeframe of the data
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig = plt.figure(figsize=(15, 12))
        
        gs = plt.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
        
        ax1 = plt.subplot(gs[0])
        ax1.plot(results.index, results['close'], label='Price', color='blue', alpha=0.7)
        
        current_price = results['close'].iloc[-1]
        
        if 'resistance' in results.columns:
            resistance_values = results['resistance'].dropna().unique()
            resistance_above = [r for r in resistance_values if r > current_price]
            for i, level in enumerate(sorted(resistance_above)[:3]):
                distance = ((level - current_price) / current_price) * 100
                ax1.axhline(y=level, color='red', linestyle='--', alpha=0.7, 
                            label=f'R{i+1}: {level:.2f} (+{distance:.2f}%)')
        
        if 'support' in results.columns:
            support_values = results['support'].dropna().unique()
            support_below = [s for s in support_values if s < current_price]
            for i, level in enumerate(sorted(support_below, reverse=True)[:3]):
                distance = ((current_price - level) / current_price) * 100
                ax1.axhline(y=level, color='green', linestyle='--', alpha=0.7,
                            label=f'S{i+1}: {level:.2f} (-{distance:.2f}%)')
        
        for trade in trades:
            if trade['position'] == 'LONG':
                ax1.scatter(trade['entry_date'], trade['entry_price'], marker='^', color='green', s=100)
                
                if trade['exit_reason'] == 'STOP_LOSS':
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='v', color='red', s=100)
                elif trade['exit_reason'] == 'TAKE_PROFIT':
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='v', color='blue', s=100)
                else:
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='v', color='gray', s=100)
            else:
                ax1.scatter(trade['entry_date'], trade['entry_price'], marker='v', color='red', s=100)
                
                if trade['exit_reason'] == 'STOP_LOSS':
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='red', s=100)
                elif trade['exit_reason'] == 'TAKE_PROFIT':
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='blue', s=100)
                else:
                    ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='gray', s=100)
        
        symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
        ax1.set_title(f'{symbol_name} ({timeframe}) - Backtest Results')
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

def run_combined_strategy_backtest(symbol, timeframe='hours', interval='1h', limit=2000):
    """
    Run a backtest of the combined strategy.
    
    Args:
        symbol (str): Symbol to backtest (e.g., 'BTC-USDT-VANILLA-PERPETUAL')
        timeframe (str): Data timeframe ('hours' or 'days')
        interval (str): Interval for the data (e.g., '1h', '4h', '1d')
        limit (int): Maximum number of records to return
        
    Returns:
        tuple: (DataFrame with backtest results, dict of performance metrics, list of trades)
    """
    logger.info(f"Running combined strategy backtest for {symbol} with {interval} interval")
    
    df = get_data(symbol, timeframe, interval, limit)
    
    combined_df = generate_combined_signals(df, symbol, interval)
    
    backtester = CombinedStrategyBacktester(
        initial_capital=10000.0,
        position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    combined_results, combined_metrics, combined_trades = backtester.run_backtest(
        combined_df,
        use_combined_signals=True,
        use_strong_signals_only=False
    )
    
    squeeze_results, squeeze_metrics, squeeze_trades = backtester.run_backtest(
        combined_df,
        use_combined_signals=False,
        use_strong_signals_only=False
    )
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    combined_plot_path = f"results/combined_backtest_{symbol.replace('-', '_')}_{interval}_{timestamp}.png"
    backtester.plot_backtest_results(
        combined_results,
        combined_metrics,
        combined_trades,
        symbol,
        interval,
        save_path=combined_plot_path
    )
    
    squeeze_plot_path = f"results/squeeze_backtest_{symbol.replace('-', '_')}_{interval}_{timestamp}.png"
    backtester.plot_backtest_results(
        squeeze_results,
        squeeze_metrics,
        squeeze_trades,
        symbol,
        interval,
        save_path=squeeze_plot_path
    )
    
    metrics_df = pd.DataFrame({
        'Metric': list(combined_metrics.keys()),
        'Combined Strategy': list(combined_metrics.values()),
        'Squeeze Momentum': list(squeeze_metrics.values())
    })
    
    metrics_path = f"results/backtest_comparison_{symbol.replace('-', '_')}_{interval}_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    logger.info(f"Combined Strategy: {combined_metrics['total_return']:.2%} return, {combined_metrics['win_rate']:.2%} win rate, {combined_metrics['total_trades']} trades")
    logger.info(f"Squeeze Momentum: {squeeze_metrics['total_return']:.2%} return, {squeeze_metrics['win_rate']:.2%} win rate, {squeeze_metrics['total_trades']} trades")
    
    return combined_results, combined_metrics, combined_trades, squeeze_results, squeeze_metrics, squeeze_trades

def main():
    """
    Main function to run combined strategy backtests.
    """
    logger.info("Starting combined strategy backtests")
    
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
        
        combined_results, combined_metrics, combined_trades, squeeze_results, squeeze_metrics, squeeze_trades = run_combined_strategy_backtest(
            symbol,
            timeframe,
            interval
        )
        
        all_results[f"{symbol}_{interval}"] = {
            'combined': {
                'results': combined_results,
                'metrics': combined_metrics,
                'trades': combined_trades
            },
            'squeeze': {
                'results': squeeze_results,
                'metrics': squeeze_metrics,
                'trades': squeeze_trades
            }
        }
        
        symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
        print(f"\n{'='*50}")
        print(f"BACKTEST SUMMARY FOR {symbol_name} ({interval})")
        print(f"{'='*50}")
        print(f"Combined Strategy:")
        print(f"  Total Return: {combined_metrics['total_return']:.2%}")
        print(f"  Annualized Return: {combined_metrics['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {combined_metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {combined_metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {combined_metrics['win_rate']:.2%}")
        print(f"  Total Trades: {combined_metrics['total_trades']}")
        print(f"\nSqueeze Momentum:")
        print(f"  Total Return: {squeeze_metrics['total_return']:.2%}")
        print(f"  Annualized Return: {squeeze_metrics['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {squeeze_metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {squeeze_metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {squeeze_metrics['win_rate']:.2%}")
        print(f"  Total Trades: {squeeze_metrics['total_trades']}")
        print(f"{'='*50}\n")
    
    logger.info("Combined strategy backtests completed")
    
    return all_results

if __name__ == "__main__":
    main()
