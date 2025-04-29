"""
Module for interacting with the Coinbase Advanced API.
"""
import logging
import time
import hmac
import hashlib
import base64
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Dict, Any, List, Union
import config

logger = logging.getLogger(__name__)

class CoinbaseClient:
    """Client for interacting with Coinbase Advanced API."""
    
    def __init__(self, api_key=None, private_key=None):
        """
        Initialize the Coinbase client.
        
        Args:
            api_key (str, optional): Coinbase API key
            private_key (str, optional): Coinbase private key
        """
        self.api_key = api_key or config.COINBASE_API_KEY
        self.private_key = private_key or config.PRIVATE_KEY
        self.base_url = "https://api.coinbase.com"
        self.advanced_trade_url = "https://api.coinbase.com/api/v3/brokerage"
        
        logger.info("Coinbase client initialized")
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """
        Generate signature for Coinbase API request.
        
        Args:
            timestamp (str): Request timestamp
            method (str): HTTP method
            request_path (str): Request path
            body (str): Request body
            
        Returns:
            str: Base64 encoded signature
        """
        message = timestamp + method + request_path + body
        
        try:
            logger.info(f"Signature message: {message}")
            logger.info(f"API Key: {self.api_key[:10]}...")
            
            private_key = self.private_key
            private_key = private_key.replace("-----BEGIN EC PRIVATE KEY-----\\n", "")
            private_key = private_key.replace("\\n-----END EC PRIVATE KEY-----\\n", "")
            private_key = private_key.replace("-----BEGIN EC PRIVATE KEY-----\n", "")
            private_key = private_key.replace("\n-----END EC PRIVATE KEY-----\n", "")
            private_key = private_key.replace("\\n", "")
            
            logger.info(f"Cleaned private key (first 10 chars): {private_key[:10]}...")
            
            try:
                key_bytes = base64.b64decode(private_key)
                logger.info("Successfully decoded private key")
            except Exception as decode_error:
                logger.error(f"Error decoding private key: {decode_error}")
                raise
            
            signature = hmac.new(key_bytes, message.encode('utf-8'), hashlib.sha256).digest()
            encoded_signature = base64.b64encode(signature).decode('utf-8')
            
            logger.info(f"Generated signature (first 10 chars): {encoded_signature[:10]}...")
            
            return encoded_signature
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            raise
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make request to Coinbase API.
        
        Args:
            method (str): HTTP method
            endpoint (str): API endpoint
            params (dict, optional): Query parameters
            data (dict, optional): Request body
            
        Returns:
            dict: Response data
        """
        url = self.advanced_trade_url + endpoint
        
        timestamp = str(int(time.time()))
        
        body = ""
        if data:
            body = json.dumps(data)
        
        signature = self._generate_signature(timestamp, method, endpoint, body)
        
        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=body
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error making request: {e}")
            raise
    
    def get_historical_candles(self, symbol: str, interval: str, start_time: datetime, end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical candles (OHLCV data) from Coinbase.
        
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
        
        granularity_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '6h': 21600,
            '1d': 86400
        }
        
        if interval not in granularity_map:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {list(granularity_map.keys())}")
        
        granularity = granularity_map[interval]
        
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()
        
        endpoint = f"/products/{symbol}/candles"
        params = {
            "start": start_iso,
            "end": end_iso,
            "granularity": granularity
        }
        
        try:
            response = self._make_request("GET", endpoint, params=params)
            
            if 'candles' not in response:
                logger.error(f"Invalid response format: {response}")
                return pd.DataFrame()
            
            candles = response['candles']
            
            if not candles:
                logger.warning(f"No candles returned for {symbol} from {start_iso} to {end_iso}")
                return pd.DataFrame()
            
            df = pd.DataFrame.from_records(candles)
            
            df = df.rename(columns={
                'start': 'timestamp',
                'low': 'low',
                'high': 'high',
                'open': 'open',
                'close': 'close',
                'volume': 'volume'
            })
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df.sort_values('timestamp')
            
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} candles for {symbol} {interval}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving candles: {e}")
            raise
    
    def get_account_balance(self, currency: str = 'USD') -> float:
        """
        Get account balance for a specific currency.
        
        Args:
            currency (str, optional): Currency symbol. Defaults to 'USD'.
            
        Returns:
            float: Balance amount
        """
        endpoint = "/accounts"
        
        try:
            response = self._make_request("GET", endpoint)
            
            if 'accounts' not in response:
                logger.error(f"Invalid response format: {response}")
                return 0.0
            
            accounts = response['accounts']
            
            for account in accounts:
                if account['currency'] == currency:
                    return float(account['available_balance']['value'])
            
            logger.warning(f"No account found for currency: {currency}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error retrieving account balance: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker for a symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC-USD')
            
        Returns:
            dict: Ticker data
        """
        endpoint = f"/products/{symbol}/ticker"
        
        try:
            response = self._make_request("GET", endpoint)
            
            return response
            
        except Exception as e:
            logger.error(f"Error retrieving ticker: {e}")
            raise
    
    def create_market_order(self, symbol: str, side: str, size: float) -> Dict:
        """
        Create a market order.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC-USD')
            side (str): Order side ('buy' or 'sell')
            size (float): Order size in base currency
            
        Returns:
            dict: Order data
        """
        endpoint = "/orders"
        
        data = {
            "client_order_id": f"bot-{int(time.time())}",
            "product_id": symbol,
            "side": side,
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": str(size)
                }
            }
        }
        
        try:
            response = self._make_request("POST", endpoint, data=data)
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating market order: {e}")
            raise
