"""
Configuration settings for the trading bot.
"""
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

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

POSITION_SIZE = 0.1  # 10% of available balance
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.04  # 4% take profit
MAX_OPEN_TRADES = 3
