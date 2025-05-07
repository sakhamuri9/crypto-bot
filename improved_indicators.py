"""
Advanced technical indicators for trading strategy with adaptive features.
Includes improved support/resistance zone test counting.
"""
import pandas as pd
import numpy as np
import ta
from ta.trend import MACD, SMAIndicator, EMAIndicator, IchimokuIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, KeltnerChannel, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, MFIIndicator
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
from scipy.signal import find_peaks
import config
import logging

logger = logging.getLogger(__name__)

def add_indicators(df):
    """
    Add technical indicators to the dataframe with adaptive features.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pandas.DataFrame: DataFrame with added indicators
    """
    df = df.copy()
    
    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()
    
    volatility = df['close'].pct_change().rolling(window=20).std()
    volatility_clean = volatility.fillna(0).replace([np.inf, -np.inf], 0)
    adaptive_rsi_window = np.maximum(5, np.minimum(21, 14 - (volatility_clean * 100).astype(int)))
    df['adaptive_rsi'] = df.apply(
        lambda x: RSIIndicator(close=df['close'], window=int(adaptive_rsi_window[x.name] if not pd.isna(adaptive_rsi_window[x.name]) else 14)).rsi()[x.name], 
        axis=1
    )
    
    macd = MACD(
        close=df['close'], 
        window_slow=26, 
        window_fast=12, 
        window_sign=9
    )
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    keltner = KeltnerChannel(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=20,
        window_atr=10
    )
    df['kc_upper'] = keltner.keltner_channel_hband()
    df['kc_middle'] = keltner.keltner_channel_mband()
    df['kc_lower'] = keltner.keltner_channel_lband()
    
    df['squeeze'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
    df['squeeze_off'] = (df['bb_lower'] < df['kc_lower']) & (df['bb_upper'] > df['kc_upper'])
    
    df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema100'] = EMAIndicator(close=df['close'], window=100).ema_indicator()
    df['ema200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
    
    
    try:
        price_series = df['close'].values
        peaks, _ = find_peaks(price_series, distance=10, prominence=1)
        troughs, _ = find_peaks(-price_series, distance=10, prominence=1)
        
        df['resistance'] = np.nan
        df['support'] = np.nan
        df['resistance_tests'] = 0
        df['support_tests'] = 0
        
        df['in_resistance_zone'] = False
        df['in_support_zone'] = False
        
        df['resistance_test_events'] = 0
        df['support_test_events'] = 0
        
        for peak in peaks:
            df.iloc[peak, df.columns.get_loc('resistance')] = df.iloc[peak]['close']
        
        for trough in troughs:
            df.iloc[trough, df.columns.get_loc('support')] = df.iloc[trough]['close']
        
        df['resistance'] = df['resistance'].fillna(method='ffill')
        df['support'] = df['support'].fillna(method='ffill')
        
        resistance_zone_threshold = 0.005  # 0.5% threshold
        support_zone_threshold = 0.005     # 0.5% threshold
        
        for i in range(1, len(df)):
            if df['resistance'].iloc[i] > 0:
                price_to_res_ratio = (df['high'].iloc[i] - df['resistance'].iloc[i]) / df['resistance'].iloc[i]
                
                in_zone_now = -resistance_zone_threshold <= price_to_res_ratio <= 0.001
                
                df.iloc[i, df.columns.get_loc('in_resistance_zone')] = in_zone_now
                
                if in_zone_now and not df['in_resistance_zone'].iloc[i-1]:
                    df.iloc[i, df.columns.get_loc('resistance_test_events')] = df['resistance_test_events'].iloc[i-1] + 1
                else:
                    df.iloc[i, df.columns.get_loc('resistance_test_events')] = df['resistance_test_events'].iloc[i-1]
                
                if in_zone_now:
                    df.iloc[i, df.columns.get_loc('resistance_tests')] = df['resistance_tests'].iloc[i-1] + 1
                else:
                    df.iloc[i, df.columns.get_loc('resistance_tests')] = df['resistance_tests'].iloc[i-1]
            
            if df['support'].iloc[i] > 0:
                price_to_sup_ratio = (df['low'].iloc[i] - df['support'].iloc[i]) / df['support'].iloc[i]
                
                in_zone_now = -0.001 <= price_to_sup_ratio <= support_zone_threshold
                
                df.iloc[i, df.columns.get_loc('in_support_zone')] = in_zone_now
                
                if in_zone_now and not df['in_support_zone'].iloc[i-1]:
                    df.iloc[i, df.columns.get_loc('support_test_events')] = df['support_test_events'].iloc[i-1] + 1
                else:
                    df.iloc[i, df.columns.get_loc('support_test_events')] = df['support_test_events'].iloc[i-1]
                
                if in_zone_now:
                    df.iloc[i, df.columns.get_loc('support_tests')] = df['support_tests'].iloc[i-1] + 1
                else:
                    df.iloc[i, df.columns.get_loc('support_tests')] = df['support_tests'].iloc[i-1]
        
        df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        df['dist_to_support'] = (df['close'] - df['support']) / df['close']
        
        logger.info(f"Support and resistance detection completed with {len(peaks)} resistance levels and {len(troughs)} support levels")
        logger.info(f"Maximum resistance test events: {df['resistance_test_events'].max()}")
        logger.info(f"Maximum support test events: {df['support_test_events'].max()}")
        
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        pass
    
    df['daily_return'] = df['close'].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    df['volatility'] = df['daily_return'].rolling(window=20).std()
    df['sharpe'] = df['daily_return'].rolling(window=20).mean() / df['daily_return'].rolling(window=20).std() * np.sqrt(252)
    
    df['ema_ratio_short'] = df['ema9'] / df['ema21']
    df['ema_ratio_medium'] = df['ema21'] / df['ema50']
    df['ema_ratio_long'] = df['ema50'] / df['ema200']
    
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    
    df.dropna(inplace=True)
    
    return df

def analyze_support_resistance(df):
    """
    Analyze support and resistance zones.
    
    Args:
        df: DataFrame with processed data
        
    Returns:
        dict: Dictionary with support/resistance statistics
    """
    if 'resistance' not in df.columns or 'support' not in df.columns:
        return {}
    
    resistance_levels = df[df['resistance'].diff() != 0]['resistance'].dropna().unique()
    support_levels = df[df['support'].diff() != 0]['support'].dropna().unique()
    
    resistance_tests = {}
    for level in resistance_levels:
        level_data = df[df['resistance'] == level]
        if not level_data.empty:
            max_tests = level_data['resistance_test_events'].max()
            if not pd.isna(max_tests):
                resistance_tests[level] = int(max_tests)
    
    support_tests = {}
    for level in support_levels:
        level_data = df[df['support'] == level]
        if not level_data.empty:
            max_tests = level_data['support_test_events'].max()
            if not pd.isna(max_tests):
                support_tests[level] = int(max_tests)
    
    resistance_tests = {k: v for k, v in sorted(resistance_tests.items(), key=lambda item: item[1], reverse=True)}
    support_tests = {k: v for k, v in sorted(support_tests.items(), key=lambda item: item[1], reverse=True)}
    
    stats = {
        'resistance_levels': resistance_tests,
        'support_levels': support_tests
    }
    
    return stats

def calculate_market_features(df):
    """
    Calculate additional market features for machine learning.
    
    Args:
        df (pandas.DataFrame): DataFrame with indicators
        
    Returns:
        pandas.DataFrame: DataFrame with additional features
    """
    df = df.copy()
    
    df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    df['higher_close'] = df['close'] > df['close'].shift(1)
    
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
    
    df['doji'] = df['body_size'] < 0.001
    
    df['hammer'] = (df['body_size'] < 0.01) & (df['lower_shadow'] > 2 * df['body_size']) & (df['upper_shadow'] < 0.005)
    
    df['bullish_engulfing'] = (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'].shift(1) > df['close'].shift(1))
    df['bearish_engulfing'] = (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['open'].shift(1) < df['close'].shift(1))
    
    df['volume_change'] = df['volume'] / df['volume'].shift(1) - 1
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['relative_volume'] = df['volume'] / df['volume_ma']
    df['high_volume'] = df['relative_volume'] > 1.5
    
    df['price_above_ema50'] = df['close'] > df['ema50']
    df['price_above_ema200'] = df['close'] > df['ema200']
    df['ema50_above_ema200'] = df['ema50'] > df['ema200']
    df['golden_cross'] = (df['ema50'] > df['ema200']) & (df['ema50'].shift(1) <= df['ema200'].shift(1))
    df['death_cross'] = (df['ema50'] < df['ema200']) & (df['ema50'].shift(1) >= df['ema200'].shift(1))
    
    df['bollinger_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=50).quantile(0.2)
    df['volatility_expansion'] = df['atr'] > df['atr'].rolling(window=20).mean() * 1.5
    
    df['bull_market'] = df['ema50_above_ema200'] & (df['adx'] > 20)
    df['bear_market'] = ~df['ema50_above_ema200'] & (df['adx'] > 20)
    df['sideways_market'] = df['adx'] < 20
    
    return df
