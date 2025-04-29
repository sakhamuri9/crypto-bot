"""
Test script for live trading functionality.
"""
import logging
import argparse
from datetime import datetime
from binance_client import BinanceClient
from coinbase_client import CoinbaseClient
from strategy import TradingStrategy
from live_trader import LiveTrader
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Live Trading')
    
    parser.add_argument('--exchange', type=str, default='binance',
                        choices=['binance', 'coinbase'],
                        help='Exchange to use (default: binance)')
    
    parser.add_argument('--symbol', type=str, default=config.SYMBOL,
                        help=f'Trading pair symbol (default: {config.SYMBOL})')
    
    parser.add_argument('--timeframe', type=str, default=config.TIMEFRAME,
                        help=f'Candle interval (default: {config.TIMEFRAME})')
    
    parser.add_argument('--interval', type=int, default=60,
                        help='Interval between checks in seconds (default: 60)')
    
    parser.add_argument('--runtime', type=int, default=300,
                        help='Maximum runtime in seconds (default: 300)')
    
    parser.add_argument('--risk-per-trade', type=float, default=0.02,
                        help='Percentage of account balance to risk per trade (default: 0.02)')
    
    parser.add_argument('--stop-loss', type=float, default=0.02,
                        help='Stop loss percentage (default: 0.02)')
    
    parser.add_argument('--take-profit', type=float, default=0.04,
                        help='Take profit percentage (default: 0.04)')
    
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode without executing actual trades')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"Starting live trading test for {args.symbol} with {args.timeframe} timeframe")
    logger.info(f"Exchange: {args.exchange}")
    logger.info(f"Test will run for {args.runtime} seconds with {args.interval}s interval")
    
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
        logger.info("Live trading test stopped by user")
    except Exception as e:
        logger.error(f"Error in live trading test: {e}")
    
    logger.info("Live trading test completed")

if __name__ == "__main__":
    main()
