"""
Module for interacting with the Binance API.
"""
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from datetime import datetime
import config

logger = logging.getLogger(__name__)

class BinanceClient:
    """Client for interacting with Binance API."""
    
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        """
        Initialize the Binance client.
        
        Args:
            api_key (str, optional): Binance API key
            api_secret (str, optional): Binance API secret
            testnet (bool, optional): Whether to use testnet/sandbox. Defaults to True.
        """
        self.api_key = api_key or config.API_KEY
        self.api_secret = api_secret or config.API_SECRET
        self.testnet = testnet
        
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet
        )
        logger.info(f"Binance client initialized (testnet={self.testnet})")
    
    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """
        Get historical klines (candlestick data) from Binance.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Kline interval (e.g., '1h', '4h', '1d')
            start_str (str): Start time in format 'YYYY-MM-DD' or timestamp
            end_str (str, optional): End time in format 'YYYY-MM-DD' or timestamp
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} klines for {symbol} {interval}")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Error retrieving klines: {e}")
            raise
    
    def get_account_balance(self, asset='USDT'):
        """
        Get account balance for a specific asset.
        
        Args:
            asset (str, optional): Asset symbol. Defaults to 'USDT'.
            
        Returns:
            float: Balance amount
        """
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except BinanceAPIException as e:
            logger.error(f"Error retrieving account balance: {e}")
            raise
    
    def get_symbol_info(self, symbol):
        """
        Get trading information for a symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Symbol information
        """
        try:
            return self.client.get_symbol_info(symbol)
        except BinanceAPIException as e:
            logger.error(f"Error retrieving symbol info: {e}")
            raise
