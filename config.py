"""
Configuration settings for the trading bot with advanced parameters.
"""
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

COINBASE_API_KEY = os.getenv('COINBASE_API_KEY', '')
PRIVATE_KEY = os.getenv('COINBASE_PRIVATE_KEY', '')

SYMBOL = 'BTCUSDT'
TIMEFRAME = '1h'  # 1 hour candles
LOOKBACK_PERIOD = 100  # Number of candles to analyze

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

SIGNAL_THRESHOLD = 0.3  # Minimum signal strength to generate a trade signal

USE_ML = True  # Whether to use machine learning for signal optimization
ML_THRESHOLD = 0.005  # Minimum price change to consider for ML classification
ML_WEIGHT = 0.4  # Weight of ML predictions in signal generation
USE_GRID_SEARCH = True  # Whether to use grid search for hyperparameter tuning

POSITION_SIZE = 0.1  # Base position size (% of available balance)
MAX_POSITION_SIZE = 0.2  # Maximum position size
MAX_OPEN_TRADES = 3  # Maximum number of open trades

STOP_LOSS_PCT = 0.02  # Fixed percentage stop loss (fallback)
TAKE_PROFIT_PCT = 0.04  # Fixed percentage take profit (fallback)
TRAILING_STOP_PCT = 0.015  # Trailing stop percentage
ATR_STOP_LOSS_MULTIPLIER = 2.0  # ATR multiplier for dynamic stop loss
ATR_TAKE_PROFIT_MULTIPLIER = 4.0  # ATR multiplier for dynamic take profit
MAX_DRAWDOWN_PCT = 0.15  # Maximum allowed drawdown before reducing position size
