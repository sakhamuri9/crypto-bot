"""
ML-Enhanced Trading Strategy

This module integrates all machine learning enhancements into a unified strategy:
1. Machine learning-based parameter optimization
2. Ensemble strategy with weighted voting
3. Risk parity allocation
4. Reinforcement learning for entry/exit timing
5. Sentiment analysis integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

from coindesk_client import get_data
from indicators import add_indicators
from squeeze_momentum import add_squeeze_momentum
from ultimate_macd import add_ultimate_macd
from generic_support_resistance import detect_support_resistance_levels
from combined_strategy import generate_combined_signals
from enhanced_strategy import generate_enhanced_signals
from hedge_fund_strategy import generate_hedge_fund_signals, run_hedge_fund_strategy
from ml_parameter_optimization import MarketRegimeClassifier, ParameterOptimizer
from ensemble_strategy import EnsembleStrategy
from risk_parity_allocation import RiskParityAllocator
from reinforcement_learning import DQNAgent, TradingEnvironment
from sentiment_analysis import SentimentAnalyzer, integrate_sentiment_with_strategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_enhanced_strategy.log')
    ]
)

logger = logging.getLogger(__name__)

class MLEnhancedStrategy:
    """
    ML-Enhanced trading strategy that integrates all machine learning enhancements.
    """
    
    def __init__(self, symbols, timeframes=['1h', '4h', '1d'], initial_capital=10000.0):
        """
        Initialize the ML-enhanced strategy.
        
        Args:
            symbols (list): List of symbols to trade
            timeframes (list): List of timeframes to use
            initial_capital (float): Initial capital
        """
        self.symbols = symbols
        self.timeframes = timeframes
        self.initial_capital = initial_capital
        
        self.regime_classifier = MarketRegimeClassifier(n_regimes=3)
        try:
            self.regime_classifier.load()
        except:
            logger.warning("Market regime classifier not found, will train a new one")
        
        self.sentiment_analyzer = SentimentAnalyzer(symbols)
        
        self.rl_agents = {}
        
        for symbol in symbols:
            try:
                agent_path = f'models/rl_agent_{symbol}_1h_final.h5'
                if os.path.exists(agent_path):
                    state_size = 18  # From TradingEnvironment._get_state
                    action_size = 3  # hold, buy, sell
                    agent = DQNAgent(state_size, action_size)
                    agent.load(agent_path)
                    self.rl_agents[symbol] = agent
                    logger.info(f"Loaded RL agent for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load RL agent for {symbol}: {e}")
    
    def prepare_data(self, symbol, timeframe, lookback=500):
        """
        Prepare data for a specific symbol and timeframe.
        
        Args:
            symbol (str): Symbol to prepare data for
            timeframe (str): Timeframe to prepare data for
            lookback (int): Number of candles to fetch
            
        Returns:
            pandas.DataFrame: Prepared data
        """
        df = get_data(symbol, 'hours' if timeframe in ['1h', '4h'] else 'days', timeframe, limit=lookback)
        
        df = add_indicators(df)
        df = add_squeeze_momentum(df)
        df = add_ultimate_macd(df)
        
        df = detect_support_resistance_levels(df, symbol, timeframe)
        
        return df
    
    def optimize_parameters(self, symbol, timeframe):
        """
        Optimize strategy parameters using machine learning.
        
        Args:
            symbol (str): Symbol to optimize for
            timeframe (str): Timeframe to optimize for
            
        Returns:
            dict: Optimized parameters
        """
        logger.info(f"Optimizing parameters for {symbol} {timeframe}")
        
        df = self.prepare_data(symbol, timeframe)
        
        if not hasattr(self.regime_classifier, 'kmeans') or self.regime_classifier.kmeans is None:
            self.regime_classifier.fit(df)
        
        regimes = self.regime_classifier.predict(df)
        current_regime = regimes[-1]
        
        param_grid = {
            'squeeze_length': [20, 30, 40],
            'squeeze_mult': [1.5, 2.0, 2.5],
            'macd_fast': [8, 12, 16],
            'macd_slow': [21, 26, 34],
            'macd_signal': [7, 9, 12],
            'atr_stop_mult': [1.5, 2.0, 2.5, 3.0],
            'atr_take_profit_mult': [3.0, 4.0, 5.0],
            'min_strength': [2, 3, 4],
            'kelly_safety': [0.3, 0.5, 0.7]
        }
        
        optimizer = ParameterOptimizer(f'{symbol}_{timeframe}_strategy', param_grid)
        
        optimized_params = optimizer.get_optimized_parameters(current_regime)
        
        if not optimized_params:
            X_train, y_train = optimizer.create_training_data(df, regimes)
            
            optimized_params = optimizer.optimize_parameters(X_train, y_train, regimes)
            
            optimized_params = optimizer.get_optimized_parameters(current_regime)
        
        logger.info(f"Optimized parameters for {symbol} {timeframe}: {optimized_params}")
        
        return optimized_params
    
    def generate_ensemble_signals(self, symbol):
        """
        Generate ensemble signals for a symbol.
        
        Args:
            symbol (str): Symbol to generate signals for
            
        Returns:
            pandas.DataFrame: DataFrame with ensemble signals
        """
        logger.info(f"Generating ensemble signals for {symbol}")
        
        ensemble = EnsembleStrategy(symbol, self.timeframes)
        
        ensemble_df, metrics, trades, equity_curve = ensemble.run_strategy()
        
        logger.info(f"Ensemble strategy metrics for {symbol}: {metrics}")
        
        return ensemble_df, metrics
    
    def calculate_risk_parity_allocation(self):
        """
        Calculate risk parity allocation for all symbols.
        
        Returns:
            dict: Allocation for each symbol
        """
        logger.info("Calculating risk parity allocation")
        
        allocator = RiskParityAllocator(self.symbols)
        
        allocation = allocator.calculate_allocation()
        
        logger.info(f"Risk parity allocation: {allocation}")
        
        return allocation
    
    def get_rl_signal(self, symbol, df):
        """
        Get reinforcement learning signal for a symbol.
        
        Args:
            symbol (str): Symbol to get signal for
            df (pandas.DataFrame): DataFrame with price data and indicators
            
        Returns:
            int: Signal (0: hold, 1: buy, 2: sell)
        """
        if symbol not in self.rl_agents:
            return None
        
        env = TradingEnvironment(df)
        
        state = env._get_state()
        
        action = self.rl_agents[symbol].act(state, training=False)
        
        return action
    
    def get_sentiment_scores(self, symbol):
        """
        Get sentiment scores for a symbol.
        
        Args:
            symbol (str): Symbol to get sentiment for
            
        Returns:
            dict: Sentiment scores
        """
        scores = self.sentiment_analyzer.calculate_sentiment_scores(symbol)
        
        return scores
    
    def generate_ml_enhanced_signals(self, symbol, timeframe='1h'):
        """
        Generate ML-enhanced signals for a symbol.
        
        Args:
            symbol (str): Symbol to generate signals for
            timeframe (str): Timeframe to generate signals for
            
        Returns:
            pandas.DataFrame: DataFrame with ML-enhanced signals
        """
        logger.info(f"Generating ML-enhanced signals for {symbol} {timeframe}")
        
        df = self.prepare_data(symbol, timeframe)
        
        df = generate_combined_signals(df, symbol, timeframe)
        df = generate_enhanced_signals(df, symbol, timeframe)
        df = generate_hedge_fund_signals(df, symbol, timeframe)
        
        optimized_params = self.optimize_parameters(symbol, timeframe)
        
        ensemble_df, _ = self.generate_ensemble_signals(symbol)
        
        df['ensemble_buy'] = ensemble_df['ensemble_buy']
        df['ensemble_sell'] = ensemble_df['ensemble_sell']
        
        rl_signal = self.get_rl_signal(symbol, df)
        
        if rl_signal is not None:
            df['rl_buy'] = rl_signal == 1
            df['rl_sell'] = rl_signal == 2
        
        sentiment_scores = self.get_sentiment_scores(symbol)
        
        df = integrate_sentiment_with_strategy(df, symbol, sentiment_scores['sentiment_score'])
        
        df['ml_enhanced_buy'] = False
        df['ml_enhanced_sell'] = False
        
        weights = {
            'hedge_fund': 0.3,
            'ensemble': 0.3,
            'rl': 0.2,
            'sentiment': 0.2
        }
        
        buy_score = (
            weights['hedge_fund'] * df['hedge_fund_buy'].astype(float) +
            weights['ensemble'] * df['ensemble_buy'].astype(float) +
            (weights['rl'] * df['rl_buy'].astype(float) if 'rl_buy' in df.columns else 0) +
            weights['sentiment'] * df['sentiment_adjusted_buy'].astype(float)
        )
        
        sell_score = (
            weights['hedge_fund'] * df['hedge_fund_sell'].astype(float) +
            weights['ensemble'] * df['ensemble_sell'].astype(float) +
            (weights['rl'] * df['rl_sell'].astype(float) if 'rl_sell' in df.columns else 0) +
            weights['sentiment'] * df['sentiment_adjusted_sell'].astype(float)
        )
        
        df['ml_enhanced_buy'] = buy_score > 0.5
        df['ml_enhanced_sell'] = sell_score > 0.5
        
        logger.info(f"ML-enhanced signals generated for {symbol} {timeframe}")
        
        return df
    
    def run_backtest(self, df, symbol, timeframe='1h'):
        """
        Run backtest for ML-enhanced signals.
        
        Args:
            df (pandas.DataFrame): DataFrame with ML-enhanced signals
            symbol (str): Symbol to run backtest for
            timeframe (str): Timeframe to run backtest for
            
        Returns:
            tuple: (final_capital, trades, returns, equity_curve)
        """
        logger.info(f"Running ML-enhanced backtest for {symbol} {timeframe}")
        
        allocation = self.calculate_risk_parity_allocation()
        position_size = allocation.get(symbol, 0.2)  # Default to 20% if not found
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(df)):
            if pd.isna(df['close'].iloc[i]):
                equity_curve.append(equity_curve[-1])
                continue
            
            if position > 0 and df['ml_enhanced_sell'].iloc[i]:
                exit_price = df['close'].iloc[i]
                profit_loss = position * (exit_price - entry_price)
                capital += profit_loss
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                    'type': 'long'
                })
                position = 0
            
            if position == 0 and df['ml_enhanced_buy'].iloc[i]:
                entry_price = df['close'].iloc[i]
                position_capital = capital * position_size
                position = position_capital / entry_price
                entry_date = df.index[i]
            
            if position > 0:
                current_price = df['close'].iloc[i]
                current_value = capital + position * (current_price - entry_price)
            else:
                current_value = capital
            
            equity_curve.append(current_value)
        
        if position > 0:
            exit_price = df['close'].iloc[-1]
            profit_loss = position * (exit_price - entry_price)
            capital += profit_loss
            trades.append({
                'entry_date': entry_date,
                'exit_date': df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                'type': 'long'
            })
        
        returns = (capital / self.initial_capital - 1) * 100
        
        logger.info(f"ML-enhanced backtest complete: {len(trades)} trades, {returns:.2f}% return")
        
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
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'annualized_return': annualized_return
        }
    
    def run_strategy(self, symbol, timeframe='1h'):
        """
        Run ML-enhanced strategy for a symbol.
        
        Args:
            symbol (str): Symbol to run strategy for
            timeframe (str): Timeframe to run strategy for
            
        Returns:
            tuple: (ml_metrics, hf_metrics)
        """
        logger.info(f"Running ML-enhanced strategy for {symbol} {timeframe}")
        
        df = self.generate_ml_enhanced_signals(symbol, timeframe)
        
        final_capital, trades, returns, equity_curve = self.run_backtest(df, symbol, timeframe)
        
        ml_metrics = self.calculate_metrics(trades, equity_curve)
        
        hf_metrics = run_hedge_fund_strategy(symbol, timeframe, df)
        
        self.plot_comparison(symbol, timeframe, df, trades, equity_curve, ml_metrics, hf_metrics)
        
        logger.info(f"Strategy comparison for {symbol} {timeframe}:")
        logger.info(f"ML-enhanced: {ml_metrics['total_return']:.2f}% return, {ml_metrics['win_rate']:.2f}% win rate, {ml_metrics['sharpe_ratio']:.2f} Sharpe")
        logger.info(f"Hedge Fund: {hf_metrics['total_return']:.2f}% return, {hf_metrics['win_rate']:.2f}% win rate, {hf_metrics['sharpe_ratio']:.2f} Sharpe")
        
        return ml_metrics, hf_metrics
    
    def plot_comparison(self, symbol, timeframe, df, trades, equity_curve, ml_metrics, hf_metrics):
        """
        Plot comparison between ML-enhanced and hedge fund strategies.
        
        Args:
            symbol (str): Symbol to plot comparison for
            timeframe (str): Timeframe to plot comparison for
            df (pandas.DataFrame): DataFrame with signals
            trades (list): List of ML-enhanced trades
            equity_curve (list): ML-enhanced equity curve
            ml_metrics (dict): ML-enhanced metrics
            hf_metrics (dict): Hedge fund metrics
        """
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(df['close'], label='Price')
        
        for trade in trades:
            if trade['profit_loss'] > 0:
                plt.plot(trade['entry_date'], trade['entry_price'], '^', markersize=8, color='g')
                plt.plot(trade['exit_date'], trade['exit_price'], 'v', markersize=8, color='g')
            else:
                plt.plot(trade['entry_date'], trade['entry_price'], '^', markersize=8, color='r')
                plt.plot(trade['exit_date'], trade['exit_price'], 'v', markersize=8, color='r')
        
        plt.title(f'ML-Enhanced Strategy: {symbol} {timeframe}')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(df.index[:len(equity_curve)], equity_curve, label='ML-Enhanced')
        
        hf_equity = [self.initial_capital * (1 + hf_metrics['total_return'] / 100 * i / len(equity_curve)) for i in range(len(equity_curve))]
        plt.plot(df.index[:len(equity_curve)], hf_equity, label='Hedge Fund')
        
        plt.title('Equity Curve Comparison')
        plt.ylabel('Capital')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 1, 3)
        metrics = ['total_return', 'win_rate', 'sharpe_ratio', 'max_drawdown']
        labels = ['Total Return (%)', 'Win Rate (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
        x = np.arange(len(metrics))
        width = 0.35
        
        ml_values = [ml_metrics[m] for m in metrics]
        hf_values = [hf_metrics[m] for m in metrics]
        
        plt.bar(x - width/2, ml_values, width, label='ML-Enhanced')
        plt.bar(x + width/2, hf_values, width, label='Hedge Fund')
        
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, labels)
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/ml_enhanced_vs_hedge_fund_{symbol}_{timeframe}_{timestamp}.png')
        
        comparison = {
            'ml_enhanced': ml_metrics,
            'hedge_fund': hf_metrics,
            'improvement': {
                'total_return': ml_metrics['total_return'] - hf_metrics['total_return'],
                'win_rate': ml_metrics['win_rate'] - hf_metrics['win_rate'],
                'sharpe_ratio': ml_metrics['sharpe_ratio'] - hf_metrics['sharpe_ratio'],
                'max_drawdown': ml_metrics['max_drawdown'] - hf_metrics['max_drawdown']
            }
        }
        
        with open(f'{results_dir}/ml_enhanced_vs_hedge_fund_{symbol}_{timeframe}_{timestamp}.json', 'w') as f:
            json.dump(comparison, f, indent=4)
        
        logger.info(f"Comparison plot saved to {results_dir}")

def run_ml_enhanced_strategy(symbols=['BTC-USDT-VANILLA-PERPETUAL', 'SUI-USDT-VANILLA-PERPETUAL'], timeframe='1h'):
    """
    Run ML-enhanced strategy for multiple symbols.
    
    Args:
        symbols (list): List of symbols to run strategy for
        timeframe (str): Timeframe to run strategy for
        
    Returns:
        dict: Metrics for all symbols
    """
    logger.info(f"Running ML-enhanced strategy for {len(symbols)} symbols")
    
    strategy = MLEnhancedStrategy(symbols)
    
    all_metrics = {}
    
    for symbol in symbols:
        ml_metrics, hf_metrics = strategy.run_strategy(symbol, timeframe)
        
        all_metrics[symbol] = {
            'ml_enhanced': ml_metrics,
            'hedge_fund': hf_metrics
        }
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'{results_dir}/ml_enhanced_all_metrics_{timestamp}.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    x = np.arange(len(symbols))
    width = 0.35
    
    ml_returns = [all_metrics[s]['ml_enhanced']['total_return'] for s in symbols]
    hf_returns = [all_metrics[s]['hedge_fund']['total_return'] for s in symbols]
    
    plt.bar(x - width/2, ml_returns, width, label='ML-Enhanced')
    plt.bar(x + width/2, hf_returns, width, label='Hedge Fund')
    
    plt.title('Total Return (%)')
    plt.xticks(x, symbols)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    
    ml_win_rates = [all_metrics[s]['ml_enhanced']['win_rate'] for s in symbols]
    hf_win_rates = [all_metrics[s]['hedge_fund']['win_rate'] for s in symbols]
    
    plt.bar(x - width/2, ml_win_rates, width, label='ML-Enhanced')
    plt.bar(x + width/2, hf_win_rates, width, label='Hedge Fund')
    
    plt.title('Win Rate (%)')
    plt.xticks(x, symbols)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    
    ml_sharpe = [all_metrics[s]['ml_enhanced']['sharpe_ratio'] for s in symbols]
    hf_sharpe = [all_metrics[s]['hedge_fund']['sharpe_ratio'] for s in symbols]
    
    plt.bar(x - width/2, ml_sharpe, width, label='ML-Enhanced')
    plt.bar(x + width/2, hf_sharpe, width, label='Hedge Fund')
    
    plt.title('Sharpe Ratio')
    plt.xticks(x, symbols)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    
    ml_drawdown = [all_metrics[s]['ml_enhanced']['max_drawdown'] for s in symbols]
    hf_drawdown = [all_metrics[s]['hedge_fund']['max_drawdown'] for s in symbols]
    
    plt.bar(x - width/2, ml_drawdown, width, label='ML-Enhanced')
    plt.bar(x + width/2, hf_drawdown, width, label='Hedge Fund')
    
    plt.title('Max Drawdown (%)')
    plt.xticks(x, symbols)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/ml_enhanced_summary_{timestamp}.png')
    
    logger.info(f"ML-enhanced strategy complete for {len(symbols)} symbols")
    
    return all_metrics

if __name__ == "__main__":
    all_metrics = run_ml_enhanced_strategy()
    
    logger.info("ML-enhanced strategy complete")
