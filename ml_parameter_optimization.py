"""
Machine Learning-based Parameter Optimization for Trading Strategies

This module implements machine learning techniques to optimize strategy parameters
based on historical performance across different market regimes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
import os
from datetime import datetime
import json

from coindesk_client import get_data
from indicators import add_indicators
from squeeze_momentum import add_squeeze_momentum
from ultimate_macd import add_ultimate_macd
from generic_support_resistance import detect_support_resistance_levels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_parameter_optimization.log')
    ]
)

logger = logging.getLogger(__name__)

class MarketRegimeClassifier:
    """
    Identifies market regimes using unsupervised learning on price and volatility features.
    """
    
    def __init__(self, n_regimes=3):
        """
        Initialize the market regime classifier.
        
        Args:
            n_regimes (int): Number of market regimes to identify
        """
        self.n_regimes = n_regimes
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        self.model_path = 'models/market_regime_classifier.pkl'
        
    def create_features(self, df):
        """
        Create features for market regime classification.
        
        Args:
            df (pandas.DataFrame): Price data with indicators
            
        Returns:
            pandas.DataFrame: Feature DataFrame
        """
        features = pd.DataFrame(index=df.index)
        
        features['atr_pct'] = df['atr'] / df['close']
        features['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
        
        features['rsi'] = df['rsi']
        features['macd_hist'] = df['macd_hist']
        features['sma_ratio'] = df['sma_50'] / df['sma_200']
        
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        features['returns_5d'] = df['close'].pct_change(5)
        features['returns_20d'] = df['close'].pct_change(20)
        
        features = features.dropna()
        
        return features
    
    def fit(self, df):
        """
        Fit the market regime classifier.
        
        Args:
            df (pandas.DataFrame): Price data with indicators
            
        Returns:
            self
        """
        features = self.create_features(df)
        
        scaled_features = self.scaler.fit_transform(features)
        
        self.kmeans.fit(scaled_features)
        
        os.makedirs('models', exist_ok=True)
        joblib.dump((self.kmeans, self.scaler), self.model_path)
        
        logger.info(f"Market regime classifier trained with {self.n_regimes} regimes")
        
        return self
    
    def predict(self, df):
        """
        Predict market regimes.
        
        Args:
            df (pandas.DataFrame): Price data with indicators
            
        Returns:
            numpy.ndarray: Predicted market regimes
        """
        features = self.create_features(df)
        
        scaled_features = self.scaler.transform(features)
        
        regimes = self.kmeans.predict(scaled_features)
        
        return regimes
    
    def load(self):
        """
        Load the market regime classifier.
        
        Returns:
            self
        """
        if os.path.exists(self.model_path):
            self.kmeans, self.scaler = joblib.load(self.model_path)
            logger.info("Market regime classifier loaded")
        else:
            logger.warning("Market regime classifier not found")
        
        return self
    
    def analyze_regimes(self, df, regimes):
        """
        Analyze market regimes.
        
        Args:
            df (pandas.DataFrame): Price data with indicators
            regimes (numpy.ndarray): Predicted market regimes
            
        Returns:
            dict: Regime analysis
        """
        features = self.create_features(df)
        features['regime'] = regimes
        
        regime_analysis = {}
        
        for regime in range(self.n_regimes):
            regime_data = features[features['regime'] == regime]
            
            regime_analysis[f'regime_{regime}'] = {
                'count': len(regime_data),
                'pct': len(regime_data) / len(features) * 100,
                'avg_atr_pct': regime_data['atr_pct'].mean(),
                'avg_rsi': regime_data['rsi'].mean(),
                'avg_bb_width': regime_data['bb_width'].mean(),
                'avg_returns_5d': regime_data['returns_5d'].mean() * 100,
                'avg_returns_20d': regime_data['returns_20d'].mean() * 100,
                'avg_volume_sma_ratio': regime_data['volume_sma_ratio'].mean()
            }
        
        return regime_analysis

class ParameterOptimizer:
    """
    Optimizes strategy parameters using machine learning.
    """
    
    def __init__(self, strategy_name, param_grid):
        """
        Initialize the parameter optimizer.
        
        Args:
            strategy_name (str): Name of the strategy
            param_grid (dict): Parameter grid for optimization
        """
        self.strategy_name = strategy_name
        self.param_grid = param_grid
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_path = f'models/{strategy_name}_parameter_optimizer.pkl'
        
    def create_training_data(self, df, regimes, window_size=20):
        """
        Create training data for parameter optimization.
        
        Args:
            df (pandas.DataFrame): Price data with indicators
            regimes (numpy.ndarray): Market regimes
            window_size (int): Window size for feature creation
            
        Returns:
            tuple: X_train, y_train
        """
        features = pd.DataFrame(index=df.index[window_size:])
        
        features_with_regimes = df.iloc[window_size:].copy()
        features_with_regimes['regime'] = regimes
        
        for regime in range(len(np.unique(regimes))):
            features[f'regime_{regime}'] = (features_with_regimes['regime'] == regime).astype(int)
        
        for i in range(window_size):
            features[f'rsi_{i}'] = df['rsi'].shift(i).iloc[window_size:]
            features[f'macd_hist_{i}'] = df['macd_hist'].shift(i).iloc[window_size:]
            features[f'atr_pct_{i}'] = (df['atr'] / df['close']).shift(i).iloc[window_size:]
            features[f'bb_width_{i}'] = ((df['upper_band'] - df['lower_band']) / df['middle_band']).shift(i).iloc[window_size:]
            features[f'volume_sma_ratio_{i}'] = (df['volume'] / df['volume'].rolling(20).mean()).shift(i).iloc[window_size:]
            features[f'returns_{i}'] = df['close'].pct_change().shift(i).iloc[window_size:]
        
        forward_returns = df['close'].pct_change(5).shift(-5).iloc[window_size:] * 100
        
        valid_idx = ~(features.isna().any(axis=1) | forward_returns.isna())
        X_train = features[valid_idx]
        y_train = forward_returns[valid_idx]
        
        return X_train, y_train
    
    def optimize_parameters(self, X_train, y_train, regimes):
        """
        Optimize strategy parameters.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Target variable
            regimes (numpy.ndarray): Market regimes
            
        Returns:
            dict: Optimized parameters for each regime
        """
        unique_regimes = np.unique(regimes)
        optimized_params = {}
        
        for regime in unique_regimes:
            logger.info(f"Optimizing parameters for regime {regime}")
            
            regime_mask = np.array([r == regime for r in regimes if r is not None])
            if sum(regime_mask) < 10:
                logger.warning(f"Not enough data for regime {regime}, using default parameters")
                optimized_params[f'regime_{regime}'] = {param: values[0] for param, values in self.param_grid.items()}
                continue
                
            X_regime = X_train.iloc[regime_mask]
            y_regime = y_train.iloc[regime_mask]
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_regime, y_regime)
            
            optimized_params[f'regime_{regime}'] = grid_search.best_params_
            
            logger.info(f"Best parameters for regime {regime}: {grid_search.best_params_}")
            logger.info(f"Best score for regime {regime}: {-grid_search.best_score_:.4f} MSE")
        
        os.makedirs('models', exist_ok=True)
        with open(f'models/{self.strategy_name}_optimized_params.json', 'w') as f:
            json.dump(optimized_params, f, indent=4)
        
        return optimized_params
    
    def get_optimized_parameters(self, regime):
        """
        Get optimized parameters for a specific regime.
        
        Args:
            regime (int): Market regime
            
        Returns:
            dict: Optimized parameters
        """
        try:
            with open(f'models/{self.strategy_name}_optimized_params.json', 'r') as f:
                optimized_params = json.load(f)
            
            return optimized_params.get(f'regime_{regime}', {param: values[0] for param, values in self.param_grid.items()})
        except FileNotFoundError:
            logger.warning(f"Optimized parameters not found for {self.strategy_name}")
            return {param: values[0] for param, values in self.param_grid.items()}

def prepare_data(symbol, interval='1h', lookback=500):
    """
    Prepare data for parameter optimization.
    
    Args:
        symbol (str): Symbol to get data for
        interval (str): Timeframe interval
        lookback (int): Number of candles to fetch
        
    Returns:
        pandas.DataFrame: Prepared data
    """
    df = get_data(symbol, 'hours', interval, limit=lookback)
    
    df = add_indicators(df)
    df = add_squeeze_momentum(df)
    df = add_ultimate_macd(df)
    
    df = detect_support_resistance_levels(df, symbol, interval)
    
    return df

def optimize_strategy_parameters(symbol='BTC-USDT-VANILLA-PERPETUAL', interval='1h'):
    """
    Optimize strategy parameters for a specific symbol and interval.
    
    Args:
        symbol (str): Symbol to optimize for
        interval (str): Timeframe interval
        
    Returns:
        dict: Optimized parameters
    """
    logger.info(f"Optimizing strategy parameters for {symbol} {interval}")
    
    df = prepare_data(symbol, interval)
    
    regime_classifier = MarketRegimeClassifier(n_regimes=3)
    regime_classifier.fit(df)
    regimes = regime_classifier.predict(df)
    
    regime_analysis = regime_classifier.analyze_regimes(df, regimes)
    logger.info(f"Market regime analysis: {json.dumps(regime_analysis, indent=4)}")
    
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
    
    optimizer = ParameterOptimizer('hedge_fund_strategy', param_grid)
    X_train, y_train = optimizer.create_training_data(df, regimes)
    optimized_params = optimizer.optimize_parameters(X_train, y_train, regimes)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f'{results_dir}/optimized_params_{symbol}_{interval}_{timestamp}.json', 'w') as f:
        json.dump(optimized_params, f, indent=4)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['close'])
    plt.title(f'{symbol} Price')
    
    plt.subplot(2, 1, 2)
    regime_series = pd.Series(regimes, index=df.index[len(df)-len(regimes):])
    plt.scatter(regime_series.index, regime_series, c=regime_series, cmap='viridis')
    plt.title('Market Regimes')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/market_regimes_{symbol}_{interval}_{timestamp}.png')
    
    return optimized_params

if __name__ == "__main__":
    btc_params = optimize_strategy_parameters('BTC-USDT-VANILLA-PERPETUAL', '1h')
    sui_params = optimize_strategy_parameters('SUI-USDT-VANILLA-PERPETUAL', '1h')
    
    logger.info("Parameter optimization complete")
