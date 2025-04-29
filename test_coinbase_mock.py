"""
Test script to demonstrate the mock Coinbase client functionality.
"""
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from mock_coinbase_client import MockCoinbaseClient
from strategy import TradingStrategy
from live_trader import LiveTrader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mock_data_generation():
    """Test mock data generation and visualization."""
    client = MockCoinbaseClient()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 30 days of data
    symbol = 'BTC-USD'
    interval = '1h'
    
    logger.info(f"Generating mock data for {symbol} from {start_date} to {end_date}")
    df = client.get_historical_candles(symbol, interval, start_date, end_date)
    
    logger.info(f"Generated {len(df)} candles with {len(df.columns)} columns")
    logger.info(f"Columns: {', '.join(df.columns[:10])}...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'])
    plt.title(f"Mock {symbol} Price Data")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.savefig("mock_price_data.png")
    logger.info("Saved price chart to mock_price_data.png")
    
    balance = client.get_account_balance('USD')
    logger.info(f"Mock USD balance: {balance}")
    
    ticker = client.get_ticker(symbol)
    logger.info(f"Mock ticker: {ticker}")
    
    return df

def test_mock_trading():
    """Test mock trading functionality."""
    client = MockCoinbaseClient()
    strategy = TradingStrategy()
    
    symbol = 'BTC-USD'
    timeframe = '1h'
    
    trader = LiveTrader(
        client=client,
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        risk_per_trade=0.02,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    trader.open_position = lambda side, price: logger.info(f"TEST MODE: Would open {side} position at {price}")
    trader.close_position = lambda: logger.info("TEST MODE: Would close position")
    
    logger.info("Running mock trading for 30 seconds...")
    try:
        trader.run(interval_seconds=10, max_runtime=30)
    except KeyboardInterrupt:
        logger.info("Mock trading stopped by user")
    
    logger.info("Mock trading test completed")

if __name__ == "__main__":
    logger.info("Starting mock client tests")
    
    df = test_mock_data_generation()
    
    test_mock_trading()
    
    logger.info("All tests completed successfully")
