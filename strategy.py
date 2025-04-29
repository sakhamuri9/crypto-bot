"""
Trading strategy implementation with machine learning optimization.
"""
import pandas as pd
import numpy as np
import logging
from indicators import add_indicators
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    Advanced trading strategy combining multiple indicators with machine learning optimization.
    """
    
    def __init__(self):
        """Initialize the trading strategy."""
        self.signals = []
        self.current_position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.market_regime = 'neutral'  # Can be 'bullish', 'bearish', or 'neutral'
        
    def detect_market_regime(self, df, lookback=20):
        """
        Detect the current market regime (trend, volatility).
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data and indicators
            lookback (int): Number of periods to look back
            
        Returns:
            str: Market regime ('bullish', 'bearish', or 'neutral')
        """
        if len(df) < lookback:
            return 'neutral'
            
        ema_short_slope = (df['ema21'].iloc[-1] - df['ema21'].iloc[-lookback]) / lookback
        ema_long_slope = (df['ema50'].iloc[-1] - df['ema50'].iloc[-lookback]) / lookback
        
        current_atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].iloc[-lookback:].mean()
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        if ema_short_slope > 0.001 and ema_long_slope > 0.0005:
            regime = 'bullish'
        elif ema_short_slope < -0.001 and ema_long_slope < -0.0005:
            regime = 'bearish'
        else:
            regime = 'neutral'
            
        if volatility_ratio > 1.5:
            logger.info(f"High volatility detected: {volatility_ratio:.2f}x average")
            
        return regime
    
    def train_ml_model(self, df, prediction_window=5):
        """
        Train a machine learning model to predict price direction.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data and indicators
            prediction_window (int): Number of periods to predict ahead
            
        Returns:
            object: Trained model
        """
        logger.info("Training machine learning model for signal optimization")
        
        df['future_return'] = df['close'].pct_change(prediction_window).shift(-prediction_window)
        df['target'] = np.where(df['future_return'] > config.ML_THRESHOLD, 1, 
                               np.where(df['future_return'] < -config.ML_THRESHOLD, -1, 0))
        
        df = df.dropna()
        
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_diff', 
            'bb_width', 'stoch_k', 'stoch_d', 'atr',
            'price_roc', 'trend_strength', 'volatility_ratio',
            'ema9', 'ema21', 'ema50', 'ema_ratio_short', 'ema_ratio_long'
        ]
        
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns].values
        y = df['target'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        model = RandomForestClassifier(random_state=42)
        
        if config.USE_GRID_SEARCH:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        logger.info(f"Model performance: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = dict(zip(feature_columns, model.feature_importances_))
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top 5 important features: {top_features}")
        
        self.model = model
        return model
    
    def generate_signals(self, df):
        """
        Generate trading signals based on the strategy with weighted indicators.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data and indicators
            
        Returns:
            pandas.DataFrame: DataFrame with added signal column
        """
        df = df.copy()
        
        df['ema_ratio_short'] = df['ema9'] / df['ema21']
        df['ema_ratio_long'] = df['ema21'] / df['ema50']
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(window=20).mean()
        
        # Calculate trend strength
        df['price_change'] = df['close'].pct_change(20)
        df['trend_strength'] = df['price_change'].abs() / (df['atr'] * 20)
        
        self.market_regime = self.detect_market_regime(df)
        logger.info(f"Detected market regime: {self.market_regime}")
        
        if config.USE_ML and self.model is None and len(df) > 100:
            self.train_ml_model(df)
        
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            signal = 0
            signal_strength = 0.0
            
            weights = self._get_indicator_weights(self.market_regime)
            
            rsi_buy_signal = prev['rsi'] < config.RSI_OVERSOLD and curr['rsi'] > config.RSI_OVERSOLD
            rsi_buy_weight = weights['rsi'] if rsi_buy_signal else 0
            
            macd_buy_signal = prev['macd'] < prev['macd_signal'] and curr['macd'] > curr['macd_signal']
            macd_buy_weight = weights['macd'] if macd_buy_signal else 0
            
            ema_buy_signal = prev['close'] < prev['ema50'] and curr['close'] > curr['ema50']
            ema_buy_weight = weights['ema'] if ema_buy_signal else 0
            
            bb_buy_signal = prev['close'] <= prev['bb_lower'] and curr['close'] > curr['bb_lower']
            bb_buy_weight = weights['bb'] if bb_buy_signal else 0
            
            stoch_buy_signal = (
                prev['stoch_k'] < prev['stoch_d'] and 
                curr['stoch_k'] > curr['stoch_d'] and 
                curr['stoch_k'] < 30
            )
            stoch_buy_weight = weights['stoch'] if stoch_buy_signal else 0
            
            ema_cross_buy = prev['ema9'] < prev['ema21'] and curr['ema9'] > curr['ema21']
            ema_cross_buy_weight = weights['ema_cross'] if ema_cross_buy else 0
            
            ichimoku_buy = curr['close'] > curr['ichimoku_a'] and curr['close'] > curr['ichimoku_b']
            ichimoku_buy_weight = weights['ichimoku'] if ichimoku_buy else 0
            
            volume_buy = curr['obv'] > prev['obv'] * 1.02  # 2% increase in OBV
            volume_buy_weight = weights['volume'] if volume_buy else 0
            
            # Calculate buy signal strength
            buy_signal_strength = (
                rsi_buy_weight + macd_buy_weight + ema_buy_weight + 
                bb_buy_weight + stoch_buy_weight + ema_cross_buy_weight +
                ichimoku_buy_weight + volume_buy_weight
            )
            
            rsi_sell_signal = prev['rsi'] > config.RSI_OVERBOUGHT and curr['rsi'] < config.RSI_OVERBOUGHT
            rsi_sell_weight = weights['rsi'] if rsi_sell_signal else 0
            
            macd_sell_signal = prev['macd'] > prev['macd_signal'] and curr['macd'] < curr['macd_signal']
            macd_sell_weight = weights['macd'] if macd_sell_signal else 0
            
            ema_sell_signal = prev['close'] > prev['ema50'] and curr['close'] < curr['ema50']
            ema_sell_weight = weights['ema'] if ema_sell_signal else 0
            
            bb_sell_signal = prev['close'] >= prev['bb_upper'] and curr['close'] < prev['bb_upper']
            bb_sell_weight = weights['bb'] if bb_sell_signal else 0
            
            stoch_sell_signal = (
                prev['stoch_k'] > prev['stoch_d'] and 
                curr['stoch_k'] < curr['stoch_d'] and 
                curr['stoch_k'] > 70
            )
            stoch_sell_weight = weights['stoch'] if stoch_sell_signal else 0
            
            ema_cross_sell = prev['ema9'] > prev['ema21'] and curr['ema9'] < curr['ema21']
            ema_cross_sell_weight = weights['ema_cross'] if ema_cross_sell else 0
            
            ichimoku_sell = curr['close'] < curr['ichimoku_a'] and curr['close'] < curr['ichimoku_b']
            ichimoku_sell_weight = weights['ichimoku'] if ichimoku_sell else 0
            
            volume_sell = curr['obv'] < prev['obv'] * 0.98  # 2% decrease in OBV
            volume_sell_weight = weights['volume'] if volume_sell else 0
            
            # Calculate sell signal strength
            sell_signal_strength = (
                rsi_sell_weight + macd_sell_weight + ema_sell_weight + 
                bb_sell_weight + stoch_sell_weight + ema_cross_sell_weight +
                ichimoku_sell_weight + volume_sell_weight
            )
            
            if buy_signal_strength >= config.SIGNAL_THRESHOLD:
                signal = 1  # Buy
                signal_strength = buy_signal_strength
            elif sell_signal_strength >= config.SIGNAL_THRESHOLD:
                signal = -1  # Sell
                signal_strength = sell_signal_strength
            
            if config.USE_ML and self.model is not None:
                try:
                    feature_columns = [
                        'rsi', 'macd', 'macd_signal', 'macd_diff', 
                        'bb_width', 'stoch_k', 'stoch_d', 'atr',
                        'price_roc', 'trend_strength', 'volatility_ratio',
                        'ema9', 'ema21', 'ema50', 'ema_ratio_short', 'ema_ratio_long'
                    ]
                    
                    feature_columns = [col for col in feature_columns if col in df.columns]
                    
                    features = curr[feature_columns].values.reshape(1, -1)
                    
                    scaled_features = self.scaler.transform(features)
                    
                    ml_prediction = self.model.predict(scaled_features)[0]
                    
                    if ml_prediction == 1 and signal == 0 and buy_signal_strength > 0:
                        signal = 1
                        signal_strength = max(buy_signal_strength, config.ML_WEIGHT)
                    elif ml_prediction == -1 and signal == 0 and sell_signal_strength > 0:
                        signal = -1
                        signal_strength = max(sell_signal_strength, config.ML_WEIGHT)
                    elif ml_prediction == 0 and signal != 0 and signal_strength < config.ML_WEIGHT:
                        signal_strength *= 0.5
                        
                except Exception as e:
                    logger.error(f"Error applying ML model: {e}")
            
            df.iloc[i, df.columns.get_loc('signal')] = signal
            df.iloc[i, df.columns.get_loc('signal_strength')] = signal_strength
        
        return df
    
    def _get_indicator_weights(self, market_regime):
        """
        Get indicator weights based on market regime.
        
        Args:
            market_regime (str): Current market regime
            
        Returns:
            dict: Dictionary of indicator weights
        """
        if market_regime == 'bullish':
            return {
                'rsi': 0.15,
                'macd': 0.20,
                'ema': 0.15,
                'bb': 0.10,
                'stoch': 0.10,
                'ema_cross': 0.15,
                'ichimoku': 0.10,
                'volume': 0.05
            }
        elif market_regime == 'bearish':
            return {
                'rsi': 0.10,
                'macd': 0.15,
                'ema': 0.10,
                'bb': 0.20,
                'stoch': 0.15,
                'ema_cross': 0.10,
                'ichimoku': 0.10,
                'volume': 0.10
            }
        else:  # neutral
            return {
                'rsi': 0.15,
                'macd': 0.15,
                'ema': 0.10,
                'bb': 0.15,
                'stoch': 0.15,
                'ema_cross': 0.10,
                'ichimoku': 0.10,
                'volume': 0.10
            }
    
    def apply_risk_management(self, df):
        """
        Apply advanced risk management rules to the signals.
        
        Args:
            df (pandas.DataFrame): DataFrame with signals
            
        Returns:
            pandas.DataFrame: DataFrame with risk-adjusted signals
        """
        df = df.copy()
        
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['trailing_stop'] = np.nan
        df['position_size'] = np.nan
        
        in_position = False
        entry_price = 0
        trailing_stop = 0
        highest_price = 0
        
        for i in range(len(df)):
            curr = df.iloc[i]
            
            # Calculate volatility-based position size and risk parameters
            volatility = curr['atr'] / curr['close'] if curr['close'] > 0 else 0.01
            
            base_position_size = config.POSITION_SIZE
            signal_strength = curr['signal_strength'] if 'signal_strength' in df.columns else 1.0
            
            volatility_factor = 1.0
            if volatility > 0.03:  # High volatility
                volatility_factor = 0.5
            elif volatility < 0.01:  # Low volatility
                volatility_factor = 1.5
                
            # Calculate adaptive position size
            position_size = base_position_size * volatility_factor * (signal_strength / config.SIGNAL_THRESHOLD)
            position_size = min(position_size, config.MAX_POSITION_SIZE)
            
            # Calculate dynamic stop loss and take profit based on ATR
            atr_stop_loss_multiplier = config.ATR_STOP_LOSS_MULTIPLIER
            atr_take_profit_multiplier = config.ATR_TAKE_PROFIT_MULTIPLIER
            
            if not in_position and curr['signal'] == 1:
                in_position = True
                entry_price = curr['close']
                highest_price = entry_price
                
                atr_value = curr['atr']
                stop_loss = entry_price - (atr_value * atr_stop_loss_multiplier)
                take_profit = entry_price + (atr_value * atr_take_profit_multiplier)
                trailing_stop = stop_loss
                
                df.iloc[i, df.columns.get_loc('stop_loss')] = stop_loss
                df.iloc[i, df.columns.get_loc('take_profit')] = take_profit
                df.iloc[i, df.columns.get_loc('trailing_stop')] = trailing_stop
                df.iloc[i, df.columns.get_loc('position_size')] = position_size
                
            elif in_position:
                current_price = curr['close']
                
                if current_price > highest_price:
                    highest_price = current_price
                    new_trailing_stop = highest_price * (1 - config.TRAILING_STOP_PCT)
                    if new_trailing_stop > trailing_stop:
                        trailing_stop = new_trailing_stop
                
                if (current_price <= stop_loss or 
                    current_price <= trailing_stop or 
                    current_price >= take_profit):
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                    in_position = False
                    entry_price = 0
                    highest_price = 0
                elif curr['signal'] == -1:
                    in_position = False
                    entry_price = 0
                    highest_price = 0
                else:
                    df.iloc[i, df.columns.get_loc('stop_loss')] = stop_loss
                    df.iloc[i, df.columns.get_loc('take_profit')] = take_profit
                    df.iloc[i, df.columns.get_loc('trailing_stop')] = trailing_stop
                    df.iloc[i, df.columns.get_loc('position_size')] = position_size
        
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
