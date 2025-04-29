"""
Advanced technical indicators for trading strategy with adaptive features.
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

def add_indicators(df):
    """
    Add technical indicators to the dataframe with adaptive features.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pandas.DataFrame: DataFrame with added indicators
    """
    df = df.copy()
    
    # Basic indicators
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
    
    df['wma_half'] = df['close'].rolling(window=int(25/2)).apply(
        lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1))
    )
    df['wma_full'] = df['close'].rolling(window=25).apply(
        lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1))
    )
    df['hma'] = df['wma_half'].rolling(window=int(np.sqrt(25))).apply(
        lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1))
    )
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14,
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    williams_r = WilliamsRIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        lbp=14
    )
    df['williams_r'] = williams_r.williams_r()
    
    df['obv'] = OnBalanceVolumeIndicator(
        close=df['close'],
        volume=df['volume']
    ).on_balance_volume()
    
    mfi = MFIIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        volume=df['volume'],
        window=14
    )
    df['mfi'] = mfi.money_flow_index()
    
    atr = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    )
    df['atr'] = atr.average_true_range()
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR as percentage of price
    
    df['ichimoku_a'] = ta.trend.ichimoku_a(
        high=df['high'],
        low=df['low'],
        window1=9,
        window2=26
    )
    df['ichimoku_b'] = ta.trend.ichimoku_b(
        high=df['high'],
        low=df['low'],
        window2=26,
        window3=52
    )
    df['ichimoku_base'] = ta.trend.ichimoku_base_line(
        high=df['high'],
        low=df['low'],
        window1=9,
        window2=26
    )
    df['ichimoku_conversion'] = ta.trend.ichimoku_conversion_line(
        high=df['high'],
        low=df['low'],
        window1=9
    )
    
    adx = ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    )
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    tsi = TSIIndicator(
        close=df['close'],
        window_slow=25,
        window_fast=13
    )
    df['tsi'] = tsi.tsi()
    
    df['price_roc'] = ta.momentum.ROCIndicator(
        close=df['close'], 
        window=12
    ).roc()
    
    # Volatility indicators
    df['volatility_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(20)
    df['volatility_ratio'] = df['volatility_20'] / df['volatility_20'].rolling(window=100).mean()
    
    try:
        price_series = df['close'].values
        peaks, _ = find_peaks(price_series, distance=10, prominence=1)
        troughs, _ = find_peaks(-price_series, distance=10, prominence=1)
        
        df['resistance'] = np.nan
        df['support'] = np.nan
        
        for peak in peaks:
            df.iloc[peak, df.columns.get_loc('resistance')] = df.iloc[peak]['close']
        
        for trough in troughs:
            df.iloc[trough, df.columns.get_loc('support')] = df.iloc[trough]['close']
        
        df['resistance'] = df['resistance'].fillna(method='ffill')
        df['support'] = df['support'].fillna(method='ffill')
        
        df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        df['dist_to_support'] = (df['close'] - df['support']) / df['close']
    except Exception as e:
        # If peak detection fails, continue without these indicators
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
