"""
Main module for the trading bot.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from binance_client import BinanceClient
from strategy import TradingStrategy
from backtester import Backtester
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    
    parser.add_argument('--mode', type=str, default='backtest',
                        choices=['backtest', 'live'],
                        help='Trading mode: backtest or live')
    
    parser.add_argument('--symbol', type=str, default=config.SYMBOL,
                        help=f'Trading pair symbol (default: {config.SYMBOL})')
    
    parser.add_argument('--timeframe', type=str, default=config.TIMEFRAME,
                        help=f'Kline interval (default: {config.TIMEFRAME})')
    
    parser.add_argument('--start-date', type=str,
                        default=(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
                        help='Start date for backtest in format YYYY-MM-DD (default: 6 months ago)')
    
    parser.add_argument('--end-date', type=str,
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date for backtest in format YYYY-MM-DD (default: today)')
    
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital for backtest (default: 10000.0)')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    
    return parser.parse_args()

def run_backtest(args):
    """
    Run backtest with the specified parameters.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Starting backtest for {args.symbol} from {args.start_date} to {args.end_date}")
    
    client = BinanceClient(testnet=True)
    
    strategy = TradingStrategy()
    
    backtester = Backtester(strategy, client)
    
    results, metrics = backtester.run(
        symbol=args.symbol,
        interval=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_file = os.path.join(args.output_dir, f"backtest_results_{args.symbol}_{args.start_date}_{args.end_date}.csv")
    results.to_csv(results_file)
    logger.info(f"Results saved to {results_file}")
    
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(args.output_dir, f"backtest_metrics_{args.symbol}_{args.start_date}_{args.end_date}.csv")
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Metrics saved to {metrics_file}")
    
    plot_file = os.path.join(args.output_dir, f"backtest_plot_{args.symbol}_{args.start_date}_{args.end_date}.png")
    backtester.plot_results(results, metrics, save_path=plot_file)
    
    print("\n" + "="*50)
    print(f"BACKTEST RESULTS FOR {args.symbol} ({args.start_date} to {args.end_date})")
    print("="*50)
    print(f"Initial Capital: ${args.initial_capital:.2f}")
    print(f"Final Capital: ${metrics['final_capital']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Average Profit: ${metrics['avg_profit']:.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    print("="*50)
    
    buy_signals = results[results['signal'] == 1].copy()
    sell_signals = results[results['signal'] == -1].copy()
    
    print("\nBUY SIGNALS:")
    print("-"*50)
    for idx, row in buy_signals.iterrows():
        print(f"BUY {args.symbol} at {row['close']:.2f} on {idx.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nSELL SIGNALS:")
    print("-"*50)
    for idx, row in sell_signals.iterrows():
        print(f"SELL {args.symbol} at {row['close']:.2f} on {idx.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, metrics

def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == 'backtest':
        run_backtest(args)
    elif args.mode == 'live':
        logger.error("Live trading mode not implemented yet")
        print("Live trading mode not implemented yet")
    else:
        logger.error(f"Invalid mode: {args.mode}")
        print(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()
