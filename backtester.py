"""
Backtesting module for the trading bot.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
from binance_client import BinanceClient
from strategy import TradingStrategy
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtester for trading strategies.
    """
    
    def __init__(self, strategy, client=None):
        """
        Initialize the backtester.
        
        Args:
            strategy: Trading strategy to backtest
            client (BinanceClient, optional): Binance client. If None, a new client will be created.
        """
        self.strategy = strategy
        self.client = client or BinanceClient(testnet=True)
        
    def run(self, symbol=None, interval=None, start_date=None, end_date=None, data=None):
        """
        Run backtest on historical data.
        
        Args:
            symbol (str, optional): Trading pair symbol. Defaults to config.SYMBOL.
            interval (str, optional): Kline interval. Defaults to config.TIMEFRAME.
            start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to 6 months ago.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to today.
            data (pandas.DataFrame, optional): Pre-loaded data to use instead of fetching from API.
            
        Returns:
            tuple: (DataFrame with backtest results, dict of performance metrics)
        """
        if data is not None:
            df = data
        else:
            symbol = symbol or config.SYMBOL
            interval = interval or config.TIMEFRAME
            
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
            
            df = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date,
                end_str=end_date
            )
        
        results, metrics = self.strategy.backtest(df)
        
        logger.info(f"Backtest completed with {metrics['total_trades']} trades")
        
        return results, metrics
    
    def plot_results(self, results, metrics, save_path=None):
        """
        Plot backtest results.
        
        Args:
            results (pandas.DataFrame): DataFrame with backtest results
            metrics (dict): Performance metrics
            save_path (str, optional): Path to save the plot. If None, plot will be displayed.
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig = plt.figure(figsize=(15, 12))
        
        gs = plt.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
        
        ax1 = plt.subplot(gs[0])
        ax1.plot(results.index, results['close'], label='Price', color='blue', alpha=0.7)
        
        ax1.plot(results.index, results['ema9'], label='EMA9', color='purple', alpha=0.5)
        ax1.plot(results.index, results['ema21'], label='EMA21', color='orange', alpha=0.5)
        ax1.plot(results.index, results['ema50'], label='EMA50', color='green', alpha=0.5)
        
        ax1.plot(results.index, results['bb_upper'], label='BB Upper', color='gray', linestyle='--', alpha=0.5)
        ax1.plot(results.index, results['bb_middle'], label='BB Middle', color='gray', linestyle='-', alpha=0.5)
        ax1.plot(results.index, results['bb_lower'], label='BB Lower', color='gray', linestyle='--', alpha=0.5)
        
        buy_signals = results[results['signal'] == 1]
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
        
        sell_signals = results[results['signal'] == -1]
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
        
        ax1.scatter(results[~results['stop_loss'].isna()].index, results[~results['stop_loss'].isna()]['stop_loss'], 
                   marker='_', color='red', s=50, alpha=0.5, label='Stop Loss')
        ax1.scatter(results[~results['take_profit'].isna()].index, results[~results['take_profit'].isna()]['take_profit'], 
                   marker='_', color='green', s=50, alpha=0.5, label='Take Profit')
        
        ax1.set_title('Price Chart with Signals')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(results.index, results['total'], label='Portfolio Value', color='green')
        ax2.set_ylabel('Portfolio Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(results.index, results['rsi'], label='RSI', color='purple')
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.plot(results.index, results['macd'], label='MACD', color='blue')
        ax4.plot(results.index, results['macd_signal'], label='Signal', color='red')
        ax4.bar(results.index, results['macd_diff'], label='Histogram', color='green', alpha=0.5)
        ax4.set_ylabel('MACD')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        
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
