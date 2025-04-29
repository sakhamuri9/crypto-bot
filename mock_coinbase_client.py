"""
Mock Coinbase client for testing without API authentication.
"""
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from indicators import add_indicators

logger = logging.getLogger(__name__)

class MockCoinbaseClient:
    """Mock Coinbase client for testing without API authentication."""
    
    def __init__(self, api_key=None, private_key=None):
        """
        Initialize the mock Coinbase client.
        
        Args:
            api_key (str, optional): Not used in mock
            private_key (str, optional): Not used in mock
        """
        self.base_url = "https://api.coinbase.com"
        self.advanced_trade_url = "https://api.coinbase.com/api/v3/brokerage"
        
        self.balances = {
            'USD': 10000.0,
            'BTC': 0.1,
            'ETH': 1.0
        }
        
        self.prices = {
            'BTC-USD': 60000.0,
            'ETH-USD': 3000.0
        }
        
        self.volatility = 0.005  # 0.5% price movement
        
        logger.info("Mock Coinbase client initialized")
    
    def get_historical_candles(self, symbol, interval, start_time, end_time=None):
        """
        Get mock historical candles (OHLCV data).
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC-USD')
            interval (str): Candle interval (e.g., '1h', '1d')
            start_time (datetime): Start time
            end_time (datetime, optional): End time
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        if end_time is None:
            end_time = datetime.now()
            
        # Extend start_time to ensure we have enough data for indicators
        extended_start = start_time - timedelta(days=365)  # Add a year of data
        
        interval_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '6h': 21600,
            '1d': 86400
        }
        
        if interval not in interval_seconds:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {list(interval_seconds.keys())}")
        
        seconds = interval_seconds[interval]
        
        timestamps = []
        current = extended_start  # Use extended start time instead of original start time
        while current <= end_time:
            timestamps.append(current)
            current += timedelta(seconds=seconds)
        
        base_price = self.prices.get(symbol, 50000.0)
        
        prices = [base_price]
        for i in range(1, len(timestamps)):
            change = np.random.normal(0, self.volatility) * prices[-1]
            new_price = max(prices[-1] + change, 1.0)  # Ensure price doesn't go below 1
            prices.append(new_price)
        
        data = []
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            high = price * (1 + random.uniform(0, self.volatility))
            low = price * (1 - random.uniform(0, self.volatility))
            open_price = price * (1 + random.uniform(-self.volatility, self.volatility))
            volume = random.uniform(1, 100) * price / 10000
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Generated {len(df)} mock candles for {symbol} {interval}")
        
        # Add technical indicators
        try:
            logger.info(f"Data shape before adding indicators: {df.shape}")
            df = add_indicators(df)
            logger.info(f"Added indicators to mock data")
            logger.info(f"Data shape after adding indicators: {df.shape}")
            
            if df.empty:
                logger.warning("DataFrame is empty after adding indicators")
            else:
                logger.info(f"DataFrame has {len(df)} rows after adding indicators")
                logger.info(f"First few rows of ema9: {df['ema9'].head() if 'ema9' in df.columns else 'ema9 not found'}")
        except Exception as e:
            logger.error(f"Error adding indicators to mock data: {e}")
        
        return df
    
    def get_account_balance(self, currency='USD'):
        """
        Get mock account balance for a specific currency.
        
        Args:
            currency (str, optional): Currency symbol. Defaults to 'USD'.
            
        Returns:
            float: Balance amount
        """
        balance = self.balances.get(currency, 0.0)
        logger.info(f"Mock account balance for {currency}: {balance}")
        return balance
    
    def get_ticker(self, symbol):
        """
        Get mock current ticker for a symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC-USD')
            
        Returns:
            dict: Ticker data
        """
        base_price = self.prices.get(symbol, 50000.0)
        
        price = base_price * (1 + np.random.normal(0, self.volatility))
        
        self.prices[symbol] = price
        
        ticker = {
            'product_id': symbol,
            'price': str(price),
            'volume_24h': str(random.uniform(1000, 10000)),
            'low_24h': str(price * 0.95),
            'high_24h': str(price * 1.05),
            'last_trade_id': str(random.randint(1000000, 9999999))
        }
        
        logger.info(f"Mock ticker for {symbol}: {price}")
        
        return ticker
    
    def create_market_order(self, symbol, side, size):
        """
        Create a mock market order.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC-USD')
            side (str): Order side ('buy' or 'sell')
            size (float): Order size in base currency
            
        Returns:
            dict: Order data
        """
        base_currency = symbol.split('-')[0]
        quote_currency = symbol.split('-')[1]
        
        price = float(self.get_ticker(symbol)['price'])
        
        if side == 'buy':
            cost = price * size
            if self.balances.get(quote_currency, 0) < cost:
                logger.error(f"Insufficient {quote_currency} balance for buy order")
                raise Exception(f"Insufficient {quote_currency} balance")
            
            self.balances[quote_currency] = self.balances.get(quote_currency, 0) - cost
            self.balances[base_currency] = self.balances.get(base_currency, 0) + size
            
        elif side == 'sell':
            if self.balances.get(base_currency, 0) < size:
                logger.error(f"Insufficient {base_currency} balance for sell order")
                raise Exception(f"Insufficient {base_currency} balance")
            
            proceeds = price * size
            self.balances[base_currency] = self.balances.get(base_currency, 0) - size
            self.balances[quote_currency] = self.balances.get(quote_currency, 0) + proceeds
        
        order_id = f"mock-order-{int(time.time())}-{random.randint(1000, 9999)}"
        
        order = {
            'order_id': order_id,
            'product_id': symbol,
            'side': side,
            'size': str(size),
            'price': str(price),
            'status': 'filled',
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"Created mock {side} order for {size} {base_currency} at {price} {quote_currency}")
        
        return order
