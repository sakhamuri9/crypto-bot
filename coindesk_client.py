"""
Module for interacting with the CoinDesk API.
"""
import logging
import requests
import pandas as pd
from datetime import datetime
import sys

logger = logging.getLogger(__name__)

class CoinDeskClient:
    """Client for interacting with CoinDesk API."""
    
    def __init__(self, api_key=None):
        """
        Initialize the CoinDesk client.
        
        Args:
            api_key (str, optional): CoinDesk API key
        """
        self.api_key = api_key or "01d76bad5bada316e6df17b512d7e2c1835923a1b89382db1b4d5cbc26b50d17"
        print("CoinDesk client initialized")
    
    def get_historical_klines(self, symbol='BTC-USDT-VANILLA-PERPETUAL', interval='1h', limit=2000, market='binance', timeframe='hours'):
        """
        Get historical klines (candlestick data) from CoinDesk.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC-USDT-VANILLA-PERPETUAL')
            interval (str): Kline interval (e.g., '1h', '4h', '1d')
            limit (int): Maximum number of records to return
            market (str): Market name (e.g., 'binance')
            timeframe (str): Data timeframe ('hours' or 'days')
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        try:
            print(f"Fetching data for {symbol} from CoinDesk API with {timeframe} timeframe...")
            
            url = f'https://data-api.coindesk.com/futures/v1/historical/{timeframe}'
            
            params = {
                'market': market,
                'instrument': symbol,
                'groups': 'ID,MAPPING,OHLC,TRADE,VOLUME',
                'limit': limit,
                'aggregate': 1,
                'fill': 'true',
                'apply_mapping': 'true',
                'api_key': self.api_key
            }
            
            if interval != '1h':
                if interval == '4h':
                    params['aggregate'] = 4
                elif interval == '1d' and timeframe == 'hours':
                    params['aggregate'] = 24
                elif interval == '1d' and timeframe == 'days':
                    params['aggregate'] = 1  # For days timeframe, use aggregate=1
            
            if timeframe == 'days':
                import time
                params['to_ts'] = int(time.time())
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Data' not in data:
                print("Error: Invalid API response format - missing 'Data'")
                return pd.DataFrame()
            
            candles = data['Data']
            
            if not candles:
                print("Error: No candles found in API response")
                return pd.DataFrame()
            
            print(f"Received {len(candles)} candles from API")
            
            # Create a list to store the processed OHLCV data
            ohlcv_data = []
            
            for candle in candles:
                try:
                    # Extract OHLCV data
                    ohlcv_data.append({
                        'timestamp': candle['TIMESTAMP'],
                        'open': float(candle['OPEN']),
                        'high': float(candle['HIGH']),
                        'low': float(candle['LOW']),
                        'close': float(candle['CLOSE']),
                        'volume': float(candle['VOLUME']) if 'VOLUME' in candle else 0.0
                    })
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Warning: Skipping candle due to error: {e}")
                    continue
            
            if not ohlcv_data:
                print("Error: Failed to extract any valid OHLCV data")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv_data)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            df.set_index('timestamp', inplace=True)
            
            print(f"Successfully processed {len(df)} candles")
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame head:\n{df.head()}")
            
            return df
            
        except Exception as e:
            print(f"Error retrieving klines from CoinDesk: {str(e)}")
            import traceback
            traceback.print_exc(file=sys.stdout)
            return pd.DataFrame()
