"""
Backtesting module for SUI-USDT analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtester for SUI-USDT trading strategies.
    """
    
    def __init__(self, initial_capital=10000.0, position_size=0.1, stop_loss_pct=0.02, take_profit_pct=0.04):
        """
        Initialize the backtester.
        
        Args:
            initial_capital (float): Initial capital for backtesting
            position_size (float): Position size as a percentage of capital
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.results = None
        
    def run_backtest(self, df):
        """
        Run backtest on the provided DataFrame with signals.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data and signals
            
        Returns:
            pandas.DataFrame: DataFrame with backtest results
        """
        logger.info(f"Running backtest with {len(df)} candles")
        
        results = df.copy()
        
        results['position'] = 0
        results['entry_price'] = np.nan
        results['exit_price'] = np.nan
        results['stop_loss'] = np.nan
        results['take_profit'] = np.nan
        results['pnl'] = 0.0
        results['capital'] = self.initial_capital
        
        position = 0
        capital = self.initial_capital
        entry_price = 0
        
        for i in range(1, len(results)):
            prev = results.iloc[i-1]
            curr = results.iloc[i]
            
            if position == 0:  # No position
                if curr['signal'] == 1:  # Buy signal
                    position = 1
                    entry_price = curr['close']
                    position_size = capital * self.position_size
                    stop_loss = entry_price * (1 - self.stop_loss_pct)
                    take_profit = entry_price * (1 + self.take_profit_pct)
                    
                    results.iloc[i, results.columns.get_loc('position')] = position
                    results.iloc[i, results.columns.get_loc('entry_price')] = entry_price
                    results.iloc[i, results.columns.get_loc('stop_loss')] = stop_loss
                    results.iloc[i, results.columns.get_loc('take_profit')] = take_profit
                    
                elif curr['signal'] == -1:  # Sell signal (short)
                    position = -1
                    entry_price = curr['close']
                    position_size = capital * self.position_size
                    stop_loss = entry_price * (1 + self.stop_loss_pct)
                    take_profit = entry_price * (1 - self.take_profit_pct)
                    
                    results.iloc[i, results.columns.get_loc('position')] = position
                    results.iloc[i, results.columns.get_loc('entry_price')] = entry_price
                    results.iloc[i, results.columns.get_loc('stop_loss')] = stop_loss
                    results.iloc[i, results.columns.get_loc('take_profit')] = take_profit
            
            elif position == 1:  # Long position
                if curr['low'] <= prev['stop_loss']:
                    position = 0
                    exit_price = prev['stop_loss']
                    pnl = (exit_price - entry_price) / entry_price * position_size
                    capital += pnl
                    
                    results.iloc[i, results.columns.get_loc('position')] = position
                    results.iloc[i, results.columns.get_loc('exit_price')] = exit_price
                    results.iloc[i, results.columns.get_loc('pnl')] = pnl
                    results.iloc[i, results.columns.get_loc('capital')] = capital
                    
                elif curr['high'] >= prev['take_profit']:
                    position = 0
                    exit_price = prev['take_profit']
                    pnl = (exit_price - entry_price) / entry_price * position_size
                    capital += pnl
                    
                    results.iloc[i, results.columns.get_loc('position')] = position
                    results.iloc[i, results.columns.get_loc('exit_price')] = exit_price
                    results.iloc[i, results.columns.get_loc('pnl')] = pnl
                    results.iloc[i, results.columns.get_loc('capital')] = capital
                    
                elif curr['signal'] == -1:
                    position = 0
                    exit_price = curr['close']
                    pnl = (exit_price - entry_price) / entry_price * position_size
                    capital += pnl
                    
                    results.iloc[i, results.columns.get_loc('position')] = position
                    results.iloc[i, results.columns.get_loc('exit_price')] = exit_price
                    results.iloc[i, results.columns.get_loc('pnl')] = pnl
                    results.iloc[i, results.columns.get_loc('capital')] = capital
                
                else:
                    results.iloc[i, results.columns.get_loc('position')] = position
                    results.iloc[i, results.columns.get_loc('entry_price')] = entry_price
                    results.iloc[i, results.columns.get_loc('stop_loss')] = prev['stop_loss']
                    results.iloc[i, results.columns.get_loc('take_profit')] = prev['take_profit']
                    results.iloc[i, results.columns.get_loc('capital')] = capital
            
            elif position == -1:  # Short position
                if curr['high'] >= prev['stop_loss']:
                    position = 0
                    exit_price = prev['stop_loss']
                    pnl = (entry_price - exit_price) / entry_price * position_size
                    capital += pnl
                    
                    results.iloc[i, results.columns.get_loc('position')] = position
                    results.iloc[i, results.columns.get_loc('exit_price')] = exit_price
                    results.iloc[i, results.columns.get_loc('pnl')] = pnl
                    results.iloc[i, results.columns.get_loc('capital')] = capital
                    
                elif curr['low'] <= prev['take_profit']:
                    position = 0
                    exit_price = prev['take_profit']
                    pnl = (entry_price - exit_price) / entry_price * position_size
                    capital += pnl
                    
                    results.iloc[i, results.columns.get_loc('position')] = position
                    results.iloc[i, results.columns.get_loc('exit_price')] = exit_price
                    results.iloc[i, results.columns.get_loc('pnl')] = pnl
                    results.iloc[i, results.columns.get_loc('capital')] = capital
                    
                elif curr['signal'] == 1:
                    position = 0
                    exit_price = curr['close']
                    pnl = (entry_price - exit_price) / entry_price * position_size
                    capital += pnl
                    
                    results.iloc[i, results.columns.get_loc('position')] = position
                    results.iloc[i, results.columns.get_loc('exit_price')] = exit_price
                    results.iloc[i, results.columns.get_loc('pnl')] = pnl
                    results.iloc[i, results.columns.get_loc('capital')] = capital
                
                else:
                    results.iloc[i, results.columns.get_loc('position')] = position
                    results.iloc[i, results.columns.get_loc('entry_price')] = entry_price
                    results.iloc[i, results.columns.get_loc('stop_loss')] = prev['stop_loss']
                    results.iloc[i, results.columns.get_loc('take_profit')] = prev['take_profit']
                    results.iloc[i, results.columns.get_loc('capital')] = capital
        
        self.results = results
        
        logger.info(f"Backtest completed with final capital: ${capital:.2f}")
        
        return results
    
    def calculate_metrics(self):
        """
        Calculate performance metrics from backtest results.
        
        Returns:
            dict: Dictionary of performance metrics
        """
        if self.results is None:
            logger.error("No backtest results available. Run backtest first.")
            return {}
        
        trades = self.results[self.results['pnl'] != 0].copy()
        
        initial_capital = self.initial_capital
        final_capital = self.results['capital'].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        
        days = (self.results.index[-1] - self.results.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0
        
        self.results['peak'] = self.results['capital'].cummax()
        self.results['drawdown'] = (self.results['capital'] - self.results['peak']) / self.results['peak']
        max_drawdown = self.results['drawdown'].min()
        
        if 'daily_return' not in self.results.columns:
            self.results['daily_return'] = self.results['capital'].pct_change()
        
        daily_returns = self.results['daily_return'].dropna()
        if len(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * daily_returns.mean() / negative_returns.std()
        else:
            sortino_ratio = 0
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_profit = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        total_profit = trades[trades['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        total_loss = abs(trades[trades['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        metrics = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
        
        return metrics
    
    def plot_results(self, save_path=None):
        """
        Plot backtest results.
        
        Args:
            save_path (str, optional): Path to save the plot. If None, plot will be displayed.
        """
        if self.results is None:
            logger.error("No backtest results available. Run backtest first.")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        ax1 = axes[0]
        ax1.plot(self.results.index, self.results['close'], label='Price', color='blue', alpha=0.7)
        
        if 'support' in self.results.columns and not self.results['support'].isna().all():
            support_levels = self.results['support'].dropna().unique()
            for level in support_levels:
                ax1.axhline(y=level, color='green', linestyle='--', alpha=0.5, 
                           label=f'Support {level:.2f}' if level == support_levels[0] else "")
        
        if 'resistance' in self.results.columns and not self.results['resistance'].isna().all():
            resistance_levels = self.results['resistance'].dropna().unique()
            for level in resistance_levels:
                ax1.axhline(y=level, color='red', linestyle='--', alpha=0.5, 
                           label=f'Resistance {level:.2f}' if level == resistance_levels[0] else "")
        
        buy_signals = self.results[self.results['signal'] == 1]
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
        
        sell_signals = self.results[self.results['signal'] == -1]
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
        
        ax1.scatter(self.results[~self.results['stop_loss'].isna()].index, 
                   self.results[~self.results['stop_loss'].isna()]['stop_loss'], 
                   marker='_', color='red', s=50, alpha=0.5, label='Stop Loss')
        ax1.scatter(self.results[~self.results['take_profit'].isna()].index, 
                   self.results[~self.results['take_profit'].isna()]['take_profit'], 
                   marker='_', color='green', s=50, alpha=0.5, label='Take Profit')
        
        ax1.set_title('SUI-USDT Price Chart with Signals')
        ax1.set_ylabel('Price (USDT)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        ax2 = axes[1]
        ax2.plot(self.results.index, self.results['capital'], label='Portfolio Value', color='green')
        ax2.set_ylabel('Portfolio Value (USDT)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        ax3 = axes[2]
        ax3.fill_between(self.results.index, self.results['drawdown'] * 100, 0, color='red', alpha=0.3)
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_ylim(self.results['drawdown'].min() * 100 * 1.5, 5)
        ax3.grid(True, alpha=0.3)
        
        metrics = self.calculate_metrics()
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annualized Return: {metrics['annualized_return']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Total Trades: {metrics['total_trades']}"
        )
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
