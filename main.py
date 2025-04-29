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
from coinbase_client import CoinbaseClient
from mock_coinbase_client import MockCoinbaseClient
from strategy import TradingStrategy
from backtester import Backtester
from live_trader import LiveTrader
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
    
    parser.add_argument('--exchange', type=str, default='binance',
                        choices=['binance', 'coinbase'],
                        help='Exchange to use (default: binance)')
    
    parser.add_argument('--symbol', type=str, default=config.SYMBOL,
                        help=f'Trading pair symbol (default: {config.SYMBOL})')
    
    parser.add_argument('--timeframe', type=str, default=config.TIMEFRAME,
                        help=f'Kline interval (default: {config.TIMEFRAME})')
    
    # Backtest specific arguments
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
    
    # Live trading specific arguments
    parser.add_argument('--interval', type=int, default=60,
                        help='Interval between checks in seconds for live trading (default: 60)')
    
    parser.add_argument('--runtime', type=int, default=None,
                        help='Maximum runtime in seconds for live trading (default: None, run indefinitely)')
    
    parser.add_argument('--risk-per-trade', type=float, default=0.02,
                        help='Percentage of account balance to risk per trade (default: 0.02)')
    
    parser.add_argument('--stop-loss', type=float, default=0.02,
                        help='Stop loss percentage (default: 0.02)')
    
    parser.add_argument('--take-profit', type=float, default=0.04,
                        help='Take profit percentage (default: 0.04)')
    
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode without executing actual trades')
    
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

def run_live_trading(args):
    """
    Run live trading with the specified parameters.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Starting live trading for {args.symbol} with {args.timeframe} timeframe")
    logger.info(f"Exchange: {args.exchange}")
    
    if args.test_mode:
        logger.info("Using mock client for test mode")
        if args.exchange == 'binance':
            client = BinanceClient(testnet=True)
        elif args.exchange == 'coinbase':
            client = MockCoinbaseClient()
        else:
            logger.error(f"Invalid exchange: {args.exchange}")
            return
    else:
        if args.exchange == 'binance':
            client = BinanceClient(testnet=True)
        elif args.exchange == 'coinbase':
            client = CoinbaseClient()
        else:
            logger.error(f"Invalid exchange: {args.exchange}")
            return
    
    try:
        balance = client.get_account_balance()
        logger.info(f"Account balance: ${balance:.2f}")
    except Exception as e:
        logger.error(f"Error getting account balance: {e}")
        return
    
    strategy = TradingStrategy()
    
    trader = LiveTrader(
        client=client,
        strategy=strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        risk_per_trade=args.risk_per_trade,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit
    )
    
    if args.test_mode:
        logger.info("Running in TEST MODE - no actual trades will be executed")
        
        trader.open_position = lambda side, price: logger.info(f"TEST MODE: Would open {side} position at {price}")
        trader.close_position = lambda: logger.info("TEST MODE: Would close position")
    
    try:
        trader.run(interval_seconds=args.interval, max_runtime=args.runtime)
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
    except Exception as e:
        logger.error(f"Error in live trading: {e}")
    
    logger.info("Live trading completed")

def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == 'backtest':
        run_backtest(args)
    elif args.mode == 'live':
        run_live_trading(args)
    else:
        logger.error(f"Invalid mode: {args.mode}")
        print(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()
