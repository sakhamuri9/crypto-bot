"""
Ensemble Trading Strategy

This module implements an ensemble trading strategy that combines signals from multiple
strategies and timeframes using a weighted voting mechanism.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
import json

from coindesk_client import get_data
from indicators import add_indicators
from squeeze_momentum import add_squeeze_momentum
from ultimate_macd import add_ultimate_macd
from generic_support_resistance import detect_support_resistance_levels
from combined_strategy import generate_combined_signals
from enhanced_strategy import generate_enhanced_signals
from hedge_fund_strategy import generate_hedge_fund_signals
from ml_parameter_optimization import MarketRegimeClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ensemble_strategy.log')
    ]
)

logger = logging.getLogger(__name__)

class EnsembleStrategy:
    """
    Ensemble trading strategy that combines signals from multiple strategies and timeframes.
    """
    
    def __init__(self, symbol, timeframes=['1h', '4h', '1d'], initial_capital=10000.0):
        """
        Initialize the ensemble strategy.
        
        Args:
            symbol (str): Symbol to trade
            timeframes (list): List of timeframes to use
            initial_capital (float): Initial capital
        """
        self.symbol = symbol
        self.timeframes = timeframes
        self.initial_capital = initial_capital
        self.strategy_weights = {
            'combined': 0.2,
            'enhanced': 0.3,
            'hedge_fund': 0.5
        }
        self.timeframe_weights = {
            '1h': 0.3,
            '4h': 0.4,
            '1d': 0.3
        }
        self.regime_classifier = MarketRegimeClassifier(n_regimes=3)
        try:
            self.regime_classifier.load()
        except:
            logger.warning("Market regime classifier not found, will train a new one")
    
    def prepare_data(self, timeframe, lookback=500):
        """
        Prepare data for a specific timeframe.
        
        Args:
            timeframe (str): Timeframe to prepare data for
            lookback (int): Number of candles to fetch
            
        Returns:
            pandas.DataFrame: Prepared data
        """
        df = get_data(self.symbol, 'hours' if timeframe in ['1h', '4h'] else 'days', timeframe, limit=lookback)
        
        df = add_indicators(df)
        df = add_squeeze_momentum(df)
        df = add_ultimate_macd(df)
        
        df = detect_support_resistance_levels(df, self.symbol, timeframe)
        
        return df
    
    def generate_signals(self, df, timeframe):
        """
        Generate signals for all strategies.
        
        Args:
            df (pandas.DataFrame): Prepared data
            timeframe (str): Timeframe
            
        Returns:
            pandas.DataFrame: DataFrame with signals
        """
        df = generate_combined_signals(df, self.symbol, timeframe)
        df = generate_enhanced_signals(df, self.symbol, timeframe)
        df = generate_hedge_fund_signals(df, self.symbol, timeframe)
        
        return df
    
    def identify_market_regime(self, df):
        """
        Identify the current market regime.
        
        Args:
            df (pandas.DataFrame): Prepared data
            
        Returns:
            int: Market regime
        """
        if not hasattr(self.regime_classifier, 'kmeans') or self.regime_classifier.kmeans is None:
            self.regime_classifier.fit(df)
        
        regimes = self.regime_classifier.predict(df)
        
        current_regime = regimes[-1]
        
        return current_regime
    
    def adjust_weights_for_regime(self, regime):
        """
        Adjust strategy and timeframe weights based on market regime.
        
        Args:
            regime (int): Market regime
            
        Returns:
            tuple: Updated strategy_weights, timeframe_weights
        """
        try:
            with open(f'models/ensemble_weights_regime_{regime}.json', 'r') as f:
                weights = json.load(f)
                strategy_weights = weights.get('strategy_weights', self.strategy_weights)
                timeframe_weights = weights.get('timeframe_weights', self.timeframe_weights)
        except FileNotFoundError:
            if regime == 0:  # Low volatility regime
                strategy_weights = {
                    'combined': 0.1,
                    'enhanced': 0.3,
                    'hedge_fund': 0.6
                }
                timeframe_weights = {
                    '1h': 0.2,
                    '4h': 0.3,
                    '1d': 0.5
                }
            elif regime == 1:  # Medium volatility regime
                strategy_weights = {
                    'combined': 0.2,
                    'enhanced': 0.4,
                    'hedge_fund': 0.4
                }
                timeframe_weights = {
                    '1h': 0.3,
                    '4h': 0.4,
                    '1d': 0.3
                }
            else:  # High volatility regime
                strategy_weights = {
                    'combined': 0.3,
                    'enhanced': 0.4,
                    'hedge_fund': 0.3
                }
                timeframe_weights = {
                    '1h': 0.4,
                    '4h': 0.4,
                    '1d': 0.2
                }
        
        return strategy_weights, timeframe_weights
    
    def calculate_ensemble_signals(self, dataframes):
        """
        Calculate ensemble signals by combining signals from all strategies and timeframes.
        
        Args:
            dataframes (dict): Dictionary of DataFrames with signals for each timeframe
            
        Returns:
            pandas.DataFrame: DataFrame with ensemble signals
        """
        highest_timeframe = self.timeframes[-1]
        current_regime = self.identify_market_regime(dataframes[highest_timeframe])
        
        strategy_weights, timeframe_weights = self.adjust_weights_for_regime(current_regime)
        
        logger.info(f"Current market regime: {current_regime}")
        logger.info(f"Strategy weights: {strategy_weights}")
        logger.info(f"Timeframe weights: {timeframe_weights}")
        
        lowest_timeframe = self.timeframes[0]
        resampled_signals = {}
        
        for timeframe, df in dataframes.items():
            if timeframe == lowest_timeframe:
                resampled_signals[timeframe] = df
            else:
                signals = df[['combined_buy', 'combined_sell', 'enhanced_buy', 'enhanced_sell', 
                             'hedge_fund_buy', 'hedge_fund_sell']].copy()
                
                resampled = signals.resample(lowest_timeframe).ffill()
                
                resampled_signals[timeframe] = resampled
        
        ensemble_df = dataframes[lowest_timeframe].copy()
        
        ensemble_df['ensemble_buy_score'] = 0.0
        ensemble_df['ensemble_sell_score'] = 0.0
        
        for timeframe in self.timeframes:
            timeframe_weight = timeframe_weights[timeframe]
            signals_df = resampled_signals[timeframe]
            
            if signals_df.isna().any(axis=1).any():
                continue
            
            for strategy, strategy_weight in strategy_weights.items():
                weight = timeframe_weight * strategy_weight
                
                if f'{strategy}_buy' in signals_df.columns:
                    buy_signal = signals_df[f'{strategy}_buy'].astype(float)
                    ensemble_df['ensemble_buy_score'] += buy_signal * weight
                
                if f'{strategy}_sell' in signals_df.columns:
                    sell_signal = signals_df[f'{strategy}_sell'].astype(float)
                    ensemble_df['ensemble_sell_score'] += sell_signal * weight
        
        ensemble_df['ensemble_buy'] = ensemble_df['ensemble_buy_score'] > 0.5
        ensemble_df['ensemble_sell'] = ensemble_df['ensemble_sell_score'] > 0.5
        
        ensemble_df['market_regime'] = current_regime
        
        return ensemble_df
    
    def run_backtest(self, ensemble_df, position_size=1.0):
        """
        Run backtest for the ensemble strategy.
        
        Args:
            ensemble_df (pandas.DataFrame): DataFrame with ensemble signals
            position_size (float): Position size as a percentage of capital
            
        Returns:
            tuple: (final_capital, trades, returns, equity_curve)
        """
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(ensemble_df)):
            if pd.isna(ensemble_df['close'].iloc[i]):
                equity_curve.append(equity_curve[-1])
                continue
            
            if position > 0 and ensemble_df['ensemble_sell'].iloc[i]:
                exit_price = ensemble_df['close'].iloc[i]
                profit_loss = position * (exit_price - entry_price)
                capital += profit_loss
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': ensemble_df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                    'type': 'long',
                    'market_regime': ensemble_df['market_regime'].iloc[i]
                })
                position = 0
            
            if position == 0 and ensemble_df['ensemble_buy'].iloc[i]:
                entry_price = ensemble_df['close'].iloc[i]
                position_capital = capital * position_size
                position = position_capital / entry_price
                entry_date = ensemble_df.index[i]
            
            if position > 0:
                current_price = ensemble_df['close'].iloc[i]
                current_value = capital + position * (current_price - entry_price)
            else:
                current_value = capital
            
            equity_curve.append(current_value)
        
        if position > 0:
            exit_price = ensemble_df['close'].iloc[-1]
            profit_loss = position * (exit_price - entry_price)
            capital += profit_loss
            trades.append({
                'entry_date': entry_date,
                'exit_date': ensemble_df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                'type': 'long',
                'market_regime': ensemble_df['market_regime'].iloc[-1]
            })
        
        returns = (capital / self.initial_capital - 1) * 100
        
        return capital, trades, returns, equity_curve
    
    def calculate_metrics(self, trades, equity_curve):
        """
        Calculate performance metrics.
        
        Args:
            trades (list): List of trades
            equity_curve (list): Equity curve
            
        Returns:
            dict: Performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_return': 0,
                'annualized_return': 0
            }
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['profit_loss'] > 0]
        losing_trades = [t for t in trades if t['profit_loss'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_profit = np.mean([t['profit_loss_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_loss_pct'] for t in losing_trades]) if losing_trades else 0
        
        total_profit = sum([t['profit_loss'] for t in winning_trades])
        total_loss = abs(sum([t['profit_loss'] for t in losing_trades]))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        peak = self.initial_capital
        drawdowns = []
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns)
        
        total_return = (equity_curve[-1] / self.initial_capital - 1) * 100
        
        days = (trades[-1]['exit_date'] - trades[0]['entry_date']).days
        annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100 if days > 0 else 0
        
        equity_returns = [equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve))]
        sharpe_ratio = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252) if np.std(equity_returns) > 0 else 0
        
        regime_metrics = {}
        for regime in set([t['market_regime'] for t in trades]):
            regime_trades = [t for t in trades if t['market_regime'] == regime]
            regime_winning_trades = [t for t in regime_trades if t['profit_loss'] > 0]
            
            regime_metrics[f'regime_{regime}'] = {
                'total_trades': len(regime_trades),
                'win_rate': len(regime_winning_trades) / len(regime_trades) * 100 if regime_trades else 0,
                'avg_profit_loss': np.mean([t['profit_loss_pct'] for t in regime_trades]) if regime_trades else 0
            }
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'regime_metrics': regime_metrics
        }
    
    def run_strategy(self):
        """
        Run the ensemble strategy.
        
        Returns:
            tuple: (ensemble_df, metrics, trades, equity_curve)
        """
        dataframes = {}
        for timeframe in self.timeframes:
            logger.info(f"Preparing data for {self.symbol} {timeframe}")
            df = self.prepare_data(timeframe)
            df = self.generate_signals(df, timeframe)
            dataframes[timeframe] = df
        
        logger.info("Calculating ensemble signals")
        ensemble_df = self.calculate_ensemble_signals(dataframes)
        
        logger.info("Running backtest")
        final_capital, trades, returns, equity_curve = self.run_backtest(ensemble_df)
        
        metrics = self.calculate_metrics(trades, equity_curve)
        
        logger.info(f"Backtest complete: {len(trades)} trades, {metrics['win_rate']:.2f}% win rate, {metrics['total_return']:.2f}% return")
        
        return ensemble_df, metrics, trades, equity_curve
    
    def plot_results(self, ensemble_df, trades, equity_curve, metrics):
        """
        Plot backtest results.
        
        Args:
            ensemble_df (pandas.DataFrame): DataFrame with ensemble signals
            trades (list): List of trades
            equity_curve (list): Equity curve
            metrics (dict): Performance metrics
        """
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(ensemble_df['close'], label='Price')
        
        for trade in trades:
            if trade['profit_loss'] > 0:
                plt.plot(trade['entry_date'], trade['entry_price'], '^', markersize=8, color='g')
                plt.plot(trade['exit_date'], trade['exit_price'], 'v', markersize=8, color='g')
            else:
                plt.plot(trade['entry_date'], trade['entry_price'], '^', markersize=8, color='r')
                plt.plot(trade['exit_date'], trade['exit_price'], 'v', markersize=8, color='r')
        
        plt.title(f'Ensemble Strategy: {self.symbol}')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(ensemble_df.index[:len(equity_curve)], equity_curve, label='Equity Curve')
        plt.title('Equity Curve')
        plt.ylabel('Capital')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/ensemble_backtest_{self.symbol}_{timestamp}.png')
        
        with open(f'{results_dir}/ensemble_metrics_{self.symbol}_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(f'{results_dir}/ensemble_trades_{self.symbol}_{timestamp}.csv', index=False)
        
        logger.info(f"Results saved to {results_dir}")

def run_ensemble_strategy(symbol='BTC-USDT-VANILLA-PERPETUAL'):
    """
    Run the ensemble strategy for a specific symbol.
    
    Args:
        symbol (str): Symbol to run the strategy for
        
    Returns:
        tuple: (ensemble_df, metrics, trades, equity_curve)
    """
    strategy = EnsembleStrategy(symbol)
    ensemble_df, metrics, trades, equity_curve = strategy.run_strategy()
    strategy.plot_results(ensemble_df, trades, equity_curve, metrics)
    
    return ensemble_df, metrics, trades, equity_curve

if __name__ == "__main__":
    btc_results = run_ensemble_strategy('BTC-USDT-VANILLA-PERPETUAL')
    sui_results = run_ensemble_strategy('SUI-USDT-VANILLA-PERPETUAL')
    
    logger.info("Ensemble strategy complete")
