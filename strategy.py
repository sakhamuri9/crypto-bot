"""
Trading strategy implementation.
"""
import pandas as pd
import numpy as np
import logging
from indicators import add_indicators
import config

logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    Advanced trading strategy combining multiple indicators.
    """
    
    def __init__(self):
        """Initialize the trading strategy."""
        self.signals = []
        self.current_position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
    
    def generate_signals(self, df):
        """
        Generate trading signals based on the strategy.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data and indicators
            
        Returns:
            pandas.DataFrame: DataFrame with added signal column
        """
        df = df.copy()
        
        df['signal'] = 0
        
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            signal = 0
            
            
            rsi_buy_signal = prev['rsi'] < config.RSI_OVERSOLD and curr['rsi'] > config.RSI_OVERSOLD
            
            macd_buy_signal = prev['macd'] < prev['macd_signal'] and curr['macd'] > curr['macd_signal']
            
            ema_buy_signal = prev['close'] < prev['ema50'] and curr['close'] > curr['ema50']
            
            bb_buy_signal = prev['close'] <= prev['bb_lower'] and curr['close'] > curr['bb_lower']
            
            stoch_buy_signal = (
                prev['stoch_k'] < prev['stoch_d'] and 
                curr['stoch_k'] > curr['stoch_d'] and 
                curr['stoch_k'] < 30
            )
            
            ema_cross_buy = prev['ema9'] < prev['ema21'] and curr['ema9'] > curr['ema21']
            
            buy_conditions = [
                rsi_buy_signal, 
                macd_buy_signal, 
                ema_buy_signal, 
                bb_buy_signal, 
                stoch_buy_signal,
                ema_cross_buy
            ]
            
            buy_signal = sum(buy_conditions) >= 2
            
            
            rsi_sell_signal = prev['rsi'] > config.RSI_OVERBOUGHT and curr['rsi'] < config.RSI_OVERBOUGHT
            
            macd_sell_signal = prev['macd'] > prev['macd_signal'] and curr['macd'] < curr['macd_signal']
            
            ema_sell_signal = prev['close'] > prev['ema50'] and curr['close'] < curr['ema50']
            
            bb_sell_signal = prev['close'] >= prev['bb_upper'] and curr['close'] < curr['bb_upper']
            
            stoch_sell_signal = (
                prev['stoch_k'] > prev['stoch_d'] and 
                curr['stoch_k'] < curr['stoch_d'] and 
                curr['stoch_k'] > 70
            )
            
            ema_cross_sell = prev['ema9'] > prev['ema21'] and curr['ema9'] < curr['ema21']
            
            sell_conditions = [
                rsi_sell_signal, 
                macd_sell_signal, 
                ema_sell_signal, 
                bb_sell_signal, 
                stoch_sell_signal,
                ema_cross_sell
            ]
            
            sell_signal = sum(sell_conditions) >= 2
            
            if buy_signal:
                signal = 1  # Buy
            elif sell_signal:
                signal = -1  # Sell
                
            df.iloc[i, df.columns.get_loc('signal')] = signal
        
        return df
    
    def apply_risk_management(self, df):
        """
        Apply risk management rules to the signals.
        
        Args:
            df (pandas.DataFrame): DataFrame with signals
            
        Returns:
            pandas.DataFrame: DataFrame with risk-adjusted signals
        """
        df = df.copy()
        
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        in_position = False
        entry_price = 0
        
        for i in range(len(df)):
            if not in_position and df.iloc[i]['signal'] == 1:
                in_position = True
                entry_price = df.iloc[i]['close']
                
                stop_loss = entry_price * (1 - config.STOP_LOSS_PCT)
                take_profit = entry_price * (1 + config.TAKE_PROFIT_PCT)
                
                df.iloc[i, df.columns.get_loc('stop_loss')] = stop_loss
                df.iloc[i, df.columns.get_loc('take_profit')] = take_profit
                
            elif in_position:
                current_price = df.iloc[i]['close']
                
                if current_price <= stop_loss or current_price >= take_profit:
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                    in_position = False
                    entry_price = 0
                elif df.iloc[i]['signal'] == -1:
                    in_position = False
                    entry_price = 0
                else:
                    df.iloc[i, df.columns.get_loc('stop_loss')] = stop_loss
                    df.iloc[i, df.columns.get_loc('take_profit')] = take_profit
        
        return df
    
    def backtest(self, df, initial_capital=10000.0):
        """
        Backtest the strategy on historical data.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            initial_capital (float, optional): Initial capital. Defaults to 10000.0.
            
        Returns:
            tuple: (DataFrame with backtest results, dict of performance metrics)
        """
        df = add_indicators(df)
        
        df = self.generate_signals(df)
        
        df = self.apply_risk_management(df)
        
        df['position'] = 0
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['pnl'] = 0.0
        df['capital'] = initial_capital
        df['holdings'] = 0.0
        df['total'] = initial_capital
        
        position = 0
        entry_price = 0
        
        for i in range(len(df)):
            if df.iloc[i]['signal'] == 1 and position == 0:
                position = 1
                entry_price = df.iloc[i]['close']
                
                capital_to_use = df.iloc[i-1]['capital'] * config.POSITION_SIZE if i > 0 else initial_capital * config.POSITION_SIZE
                holdings = capital_to_use
                
                df.iloc[i, df.columns.get_loc('position')] = position
                df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                df.iloc[i, df.columns.get_loc('capital')] = df.iloc[i-1]['capital'] - capital_to_use if i > 0 else initial_capital - capital_to_use
                df.iloc[i, df.columns.get_loc('holdings')] = holdings
                df.iloc[i, df.columns.get_loc('total')] = df.iloc[i]['capital'] + df.iloc[i]['holdings']
                
            elif df.iloc[i]['signal'] == -1 and position == 1:
                position = 0
                exit_price = df.iloc[i]['close']
                
                pnl = df.iloc[i-1]['holdings'] * (exit_price / entry_price - 1)
                
                df.iloc[i, df.columns.get_loc('position')] = position
                df.iloc[i, df.columns.get_loc('exit_price')] = exit_price
                df.iloc[i, df.columns.get_loc('pnl')] = pnl
                df.iloc[i, df.columns.get_loc('capital')] = df.iloc[i-1]['capital'] + df.iloc[i-1]['holdings'] + pnl
                df.iloc[i, df.columns.get_loc('holdings')] = 0
                df.iloc[i, df.columns.get_loc('total')] = df.iloc[i]['capital']
                
                entry_price = 0
                
            else:
                df.iloc[i, df.columns.get_loc('position')] = position
                
                if position == 1:
                    current_price = df.iloc[i]['close']
                    df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                    df.iloc[i, df.columns.get_loc('holdings')] = df.iloc[i-1]['holdings'] * (current_price / df.iloc[i-1]['close'])
                    df.iloc[i, df.columns.get_loc('capital')] = df.iloc[i-1]['capital']
                    df.iloc[i, df.columns.get_loc('total')] = df.iloc[i]['capital'] + df.iloc[i]['holdings']
                else:
                    df.iloc[i, df.columns.get_loc('capital')] = df.iloc[i-1]['capital'] if i > 0 else initial_capital
                    df.iloc[i, df.columns.get_loc('total')] = df.iloc[i]['capital']
        
        metrics = self._calculate_performance_metrics(df, initial_capital)
        
        return df, metrics
    
    def _calculate_performance_metrics(self, df, initial_capital):
        """
        Calculate performance metrics for the backtest.
        
        Args:
            df (pandas.DataFrame): DataFrame with backtest results
            initial_capital (float): Initial capital
            
        Returns:
            dict: Performance metrics
        """
        trades = df[df['pnl'] != 0].copy()
        
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        final_capital = df.iloc[-1]['total']
        total_return = (final_capital - initial_capital) / initial_capital
        
        avg_profit = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades[trades['pnl'] > 0]['pnl'].sum() / trades[trades['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')
        
        df['peak'] = df['total'].cummax()
        df['drawdown'] = (df['total'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min()
        
        returns = df['total'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if len(returns) > 0 and returns.std() > 0 else 0
        
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * (252 ** 0.5) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        days = (df.index[-1] - df.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
