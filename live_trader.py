"""
Live trading module for the Coinbase trading bot.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
import pandas as pd
from coinbase_client import CoinbaseClient
from strategy import TradingStrategy
import config

logger = logging.getLogger(__name__)

class LiveTrader:
    """Live trading implementation for Coinbase."""
    
    def __init__(self, client, strategy, symbol, timeframe, risk_per_trade=0.02, 
                 stop_loss_pct=0.02, take_profit_pct=0.04):
        """
        Initialize the live trader.
        
        Args:
            client (CoinbaseClient): Coinbase client instance
            strategy (TradingStrategy): Trading strategy instance
            symbol (str): Trading pair symbol (e.g., 'BTC-USD')
            timeframe (str): Candle interval (e.g., '1h', '1d')
            risk_per_trade (float): Percentage of account balance to risk per trade
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
        """
        self.client = client
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.active_position = False
        self.position_side = None
        self.position_entry_price = None
        self.position_size = None
        self.position_entry_time = None
        self.stop_loss_price = None
        self.take_profit_price = None
        
        if '-' not in self.symbol:
            base, quote = self.symbol[:3], self.symbol[3:]
            self.symbol = f"{base}-{quote}"
        
        self.base_currency = self.symbol.split('-')[0]
        self.quote_currency = self.symbol.split('-')[1]
        
        logger.info(f"LiveTrader initialized for {symbol} with {timeframe} timeframe")
        logger.info(f"Risk per trade: {risk_per_trade:.2%}, Stop loss: {stop_loss_pct:.2%}, Take profit: {take_profit_pct:.2%}")
    
    def get_account_balance(self):
        """
        Get account balance for the quote currency.
        
        Returns:
            float: Account balance
        """
        return self.client.get_account_balance(self.quote_currency)
    
    def get_position_size(self):
        """
        Get current position size for the base currency.
        
        Returns:
            float: Position size
        """
        return self.client.get_account_balance(self.base_currency)
    
    def calculate_position_size(self, price, risk_amount):
        """
        Calculate position size based on risk amount and stop loss.
        
        Args:
            price (float): Current price
            risk_amount (float): Amount to risk in quote currency
            
        Returns:
            float: Position size in base currency
        """
        if self.stop_loss_pct <= 0:
            logger.warning("Stop loss percentage must be greater than 0")
            return 0
            
        price_risk = price * self.stop_loss_pct
        if price_risk <= 0:
            logger.warning("Price risk must be greater than 0")
            return 0
            
        position_size = risk_amount / price_risk
        
        position_size = round(position_size, 8)
        
        return position_size
    
    def fetch_latest_data(self, lookback_days=7):
        """
        Fetch latest market data.
        
        Args:
            lookback_days (int): Number of days to look back
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data and signals
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            df = self.client.get_historical_candles(
                symbol=self.symbol,
                interval=self.timeframe,
                start_time=start_date,
                end_time=end_date
            )
            
            if df.empty:
                logger.warning(f"No data retrieved for {self.symbol}")
                return None
                
            logger.info(f"Retrieved {len(df)} candles for {self.symbol}")
            
            df = self.strategy.generate_signals(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            return None
    
    def check_open_position(self):
        """
        Check if there's an open position and update its status.
        
        Returns:
            bool: True if position should be closed, False otherwise
        """
        if not self.active_position:
            return False
            
        try:
            ticker = self.client.get_ticker(self.symbol)
            current_price = float(ticker['price'])
            
            if self.position_side == 'buy' and self.stop_loss_price is not None and self.take_profit_price is not None:
                if current_price <= self.stop_loss_price:
                    logger.info(f"Stop loss hit at {current_price} (entry: {self.position_entry_price})")
                    return True
                elif current_price >= self.take_profit_price:
                    logger.info(f"Take profit hit at {current_price} (entry: {self.position_entry_price})")
                    return True
            elif self.position_side == 'sell' and self.stop_loss_price is not None and self.take_profit_price is not None:
                if current_price >= self.stop_loss_price:
                    logger.info(f"Stop loss hit at {current_price} (entry: {self.position_entry_price})")
                    return True
                elif current_price <= self.take_profit_price:
                    logger.info(f"Take profit hit at {current_price} (entry: {self.position_entry_price})")
                    return True
            
            pnl_pct = 0.0
            if self.position_side == 'buy' and self.position_entry_price is not None:
                pnl_pct = (current_price - self.position_entry_price) / self.position_entry_price
            elif self.position_side == 'sell' and self.position_entry_price is not None:
                pnl_pct = (self.position_entry_price - current_price) / self.position_entry_price
                
            position_side_str = self.position_side.upper() if self.position_side else "UNKNOWN"
            logger.info(f"Current position: {position_side_str} {self.position_size} {self.base_currency} at {self.position_entry_price}")
            logger.info(f"Current price: {current_price}, P&L: {pnl_pct:.2%}")
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking open position: {e}")
            return False
    
    def open_position(self, side, price):
        """
        Open a new position.
        
        Args:
            side (str): Position side ('buy' or 'sell')
            price (float): Entry price
            
        Returns:
            bool: True if position opened successfully, False otherwise
        """
        if self.active_position:
            logger.warning("Cannot open position, already have an active position")
            return False
            
        try:
            balance = self.get_account_balance()
            
            risk_amount = balance * self.risk_per_trade
            
            position_size = self.calculate_position_size(price, risk_amount)
            
            if position_size <= 0:
                logger.warning("Position size must be greater than 0")
                return False
                
            if side == 'buy':
                stop_loss_price = price * (1 - self.stop_loss_pct)
                take_profit_price = price * (1 + self.take_profit_pct)
            else:
                stop_loss_price = price * (1 + self.stop_loss_pct)
                take_profit_price = price * (1 - self.take_profit_pct)
                
            order = self.client.create_market_order(
                symbol=self.symbol,
                side=side,
                size=position_size
            )
            
            self.active_position = True
            self.position_side = side
            self.position_entry_price = price
            self.position_size = position_size
            self.position_entry_time = datetime.now()
            self.stop_loss_price = stop_loss_price
            self.take_profit_price = take_profit_price
            
            logger.info(f"Opened {side.upper()} position of {position_size} {self.base_currency} at {price}")
            logger.info(f"Stop loss: {stop_loss_price}, Take profit: {take_profit_price}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
    
    def close_position(self):
        """
        Close the current position.
        
        Returns:
            bool: True if position closed successfully, False otherwise
        """
        if not self.active_position:
            logger.warning("No active position to close")
            return False
            
        try:
            close_side = 'sell' if self.position_side == 'buy' else 'buy'
            
            position_size = self.get_position_size()
            
            if position_size <= 0:
                logger.warning("Position size must be greater than 0")
                self.active_position = False
                return False
                
            order = self.client.create_market_order(
                symbol=self.symbol,
                side=close_side,
                size=position_size
            )
            
            ticker = self.client.get_ticker(self.symbol)
            current_price = float(ticker['price'])
            
            pnl_pct = 0.0
            if self.position_side == 'buy' and self.position_entry_price is not None:
                pnl_pct = (current_price - self.position_entry_price) / self.position_entry_price
            elif self.position_side == 'sell' and self.position_entry_price is not None:
                pnl_pct = (self.position_entry_price - current_price) / self.position_entry_price
                
            position_side_str = self.position_side.upper() if self.position_side else "UNKNOWN"
            logger.info(f"Closed {position_side_str} position of {position_size} {self.base_currency} at {current_price}")
            logger.info(f"P&L: {pnl_pct:.2%}")
            
            self.active_position = False
            self.position_side = None
            self.position_entry_price = None
            self.position_size = None
            self.position_entry_time = None
            self.stop_loss_price = None
            self.take_profit_price = None
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def run(self, interval_seconds=60, max_runtime=None):
        """
        Run the live trader.
        
        Args:
            interval_seconds (int): Interval between checks in seconds
            max_runtime (int, optional): Maximum runtime in seconds
            
        Returns:
            None
        """
        logger.info(f"Starting live trader for {self.symbol} with {interval_seconds}s interval")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                iteration += 1
                logger.info(f"Iteration {iteration}")
                
                if max_runtime and time.time() - start_time > max_runtime:
                    logger.info(f"Maximum runtime of {max_runtime}s reached")
                    break
                    
                if self.active_position:
                    should_close = self.check_open_position()
                    if should_close:
                        self.close_position()
                
                df = self.fetch_latest_data()
                
                if df is None or df.empty:
                    logger.warning("No data available, skipping iteration")
                    time.sleep(interval_seconds)
                    continue
                    
                latest_signal = df['signal'].iloc[-1]
                latest_price = df['close'].iloc[-1]
                
                logger.info(f"Latest price: {latest_price}, Signal: {latest_signal}")
                
                if not self.active_position:
                    if latest_signal == 1:  # Buy signal
                        logger.info(f"BUY signal for {self.symbol} at {latest_price}")
                        self.open_position('buy', latest_price)
                    elif latest_signal == -1:  # Sell signal
                        logger.info(f"SELL signal for {self.symbol} at {latest_price}")
                        self.open_position('sell', latest_price)
                
                logger.info(f"Waiting {interval_seconds}s for next iteration")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Live trader stopped by user")
            
        except Exception as e:
            logger.error(f"Error in live trader: {e}")
            
        finally:
            if self.active_position:
                logger.info("Closing position before exit")
                self.close_position()
                
            logger.info("Live trader stopped")
