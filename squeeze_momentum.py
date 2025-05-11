"""
Squeeze Momentum Indicator Implementation

This module implements the LazyBear Squeeze Momentum indicator from TradingView.
The indicator combines Bollinger Bands and Keltner Channels to detect market "squeezes"
(low volatility periods) and uses a linear regression calculation to determine
momentum direction and strength.

Original TradingView Pine Script by LazyBear:
https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def highest_n(series, n):
    """Return the highest value in the last n values of a series."""
    return series.rolling(window=n).max()

def lowest_n(series, n):
    """Return the lowest value in the last n values of a series."""
    return series.rolling(window=n).min()

def true_range(df):
    """Calculate True Range."""
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def linear_regression(series, length):
    """Calculate linear regression value for the series over the specified length."""
    y = series.values
    size = len(y)
    
    if size < length:
        return pd.Series([np.nan] * size)
    
    result = np.full(size, np.nan)
    
    for i in range(length - 1, size):
        x = np.arange(length)
        y_section = y[i - length + 1:i + 1]
        
        slope, intercept, _, _, _ = stats.linregress(x, y_section)
        result[i] = slope * (length - 1) + intercept
    
    return pd.Series(result, index=series.index)

def add_squeeze_momentum(df, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5, use_true_range=True):
    """
    Add Squeeze Momentum indicator to the dataframe.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        bb_length (int): Bollinger Bands length
        bb_mult (float): Bollinger Bands multiplier
        kc_length (int): Keltner Channel length
        kc_mult (float): Keltner Channel multiplier
        use_true_range (bool): Whether to use True Range for Keltner Channels
        
    Returns:
        pandas.DataFrame: DataFrame with Squeeze Momentum indicator columns added
    """
    logger.info(f"Adding Squeeze Momentum indicator with BB length={bb_length}, KC length={kc_length}")
    
    result_df = df.copy()
    
    source = result_df['close']
    basis = source.rolling(window=bb_length).mean()
    dev = source.rolling(window=bb_length).std() * bb_mult
    upper_bb = basis + dev
    lower_bb = basis - dev
    
    ma = source.rolling(window=kc_length).mean()
    range_val = true_range(result_df) if use_true_range else (result_df['high'] - result_df['low'])
    range_ma = range_val.rolling(window=kc_length).mean()
    upper_kc = ma + range_ma * kc_mult
    lower_kc = ma - range_ma * kc_mult
    
    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    no_sqz = (~sqz_on) & (~sqz_off)
    
    highest_high = highest_n(result_df['high'], kc_length)
    lowest_low = lowest_n(result_df['low'], kc_length)
    avg_hl = (highest_high + lowest_low) / 2
    avg_hlc = (avg_hl + source.rolling(window=kc_length).mean()) / 2
    
    val = source - avg_hlc
    
    linreg_val = linear_regression(val, kc_length)
    
    momentum_increasing = linreg_val > linreg_val.shift(1)
    momentum_decreasing = linreg_val < linreg_val.shift(1)
    
    result_df['sqz_on'] = sqz_on
    result_df['sqz_off'] = sqz_off
    result_df['no_sqz'] = no_sqz
    result_df['squeeze_momentum'] = linreg_val
    result_df['momentum_increasing'] = momentum_increasing
    result_df['momentum_decreasing'] = momentum_decreasing
    
    sqz_off_prev = sqz_off.shift(1).fillna(False).astype(bool)
    result_df['sqz_buy'] = (sqz_off & (~sqz_off_prev) & (linreg_val > 0))
    result_df['sqz_sell'] = (sqz_off & (~sqz_off_prev) & (linreg_val < 0))
    
    result_df['sqz_buy_strong'] = result_df['sqz_buy'] & momentum_increasing
    result_df['sqz_sell_strong'] = result_df['sqz_sell'] & momentum_decreasing
    
    logger.info(f"Added Squeeze Momentum indicator with {sum(result_df['sqz_buy'])} buy signals and {sum(result_df['sqz_sell'])} sell signals")
    
    return result_df

def plot_squeeze_momentum(df, title="Squeeze Momentum Indicator"):
    """
    Plot the Squeeze Momentum indicator.
    
    Args:
        df (pandas.DataFrame): DataFrame with Squeeze Momentum indicator columns
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if 'squeeze_momentum' not in df.columns:
        raise ValueError("DataFrame does not contain Squeeze Momentum indicator columns. Run add_squeeze_momentum first.")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(df.index, df['close'], label='Close Price')
    ax1.set_title(f"{title} - Price Chart")
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    colors = []
    for i in range(len(df)):
        if pd.isna(df['squeeze_momentum'].iloc[i]):
            colors.append('gray')
        elif df['squeeze_momentum'].iloc[i] >= 0:
            if i > 0 and df['squeeze_momentum'].iloc[i] > df['squeeze_momentum'].iloc[i-1]:
                colors.append('lime')  # Increasing positive momentum
            else:
                colors.append('green')  # Decreasing positive momentum
        else:
            if i > 0 and df['squeeze_momentum'].iloc[i] < df['squeeze_momentum'].iloc[i-1]:
                colors.append('red')  # Increasing negative momentum
            else:
                colors.append('maroon')  # Decreasing negative momentum
    
    ax2.bar(df.index, df['squeeze_momentum'], color=colors, width=0.8)
    
    for i in range(len(df)):
        if pd.isna(df['squeeze_momentum'].iloc[i]):
            continue
            
        if df['sqz_on'].iloc[i]:
            ax2.scatter(df.index[i], 0, color='black', marker='x', s=30)
        elif df['sqz_off'].iloc[i]:
            ax2.scatter(df.index[i], 0, color='gray', marker='x', s=30)
        else:
            ax2.scatter(df.index[i], 0, color='blue', marker='x', s=30)
    
    for i in range(len(df)):
        if df['sqz_buy_strong'].iloc[i]:
            ax1.scatter(df.index[i], df['low'].iloc[i] * 0.99, color='lime', marker='^', s=100)
        elif df['sqz_buy'].iloc[i]:
            ax1.scatter(df.index[i], df['low'].iloc[i] * 0.99, color='green', marker='^', s=60)
        
        if df['sqz_sell_strong'].iloc[i]:
            ax1.scatter(df.index[i], df['high'].iloc[i] * 1.01, color='red', marker='v', s=100)
        elif df['sqz_sell'].iloc[i]:
            ax1.scatter(df.index[i], df['high'].iloc[i] * 1.01, color='maroon', marker='v', s=60)
    
    ax2.set_title("Squeeze Momentum")
    ax2.set_ylabel('Momentum')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='lime', label='Increasing Positive Momentum'),
        Patch(facecolor='green', label='Decreasing Positive Momentum'),
        Patch(facecolor='red', label='Increasing Negative Momentum'),
        Patch(facecolor='maroon', label='Decreasing Negative Momentum'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=8, label='Squeeze On'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markersize=8, label='Squeeze Off'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='blue', markersize=8, label='No Squeeze'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='lime', markersize=10, label='Strong Buy Signal'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Buy Signal'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Strong Sell Signal'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='maroon', markersize=8, label='Sell Signal')
    ]
    
    ax2.legend(handles=legend_elements, loc='upper right', ncol=2)
    
    plt.tight_layout()
    return fig

def test_squeeze_momentum():
    """
    Test the Squeeze Momentum indicator on sample data.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    close = np.random.normal(100, 5, 100)
    close = close + np.linspace(0, 20, 100)
    close[30:40] = close[30:40] * 0.2 + close[29]  # Low volatility
    close[60:70] = close[60:70] * 2 + close[59] - close[60]  # High volatility
    
    high = close + np.random.normal(0, 1, 100)
    low = close - np.random.normal(0, 1, 100)
    open_price = close - np.random.normal(0, 0.5, 100)
    volume = np.random.normal(1000, 200, 100)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    result_df = add_squeeze_momentum(df)
    
    fig = plot_squeeze_momentum(result_df, "Squeeze Momentum Test")
    
    plt.savefig("results/squeeze_momentum_test.png")
    plt.close(fig)
    
    return result_df

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_result = test_squeeze_momentum()
    print(f"Generated {sum(test_result['sqz_buy'])} buy signals and {sum(test_result['sqz_sell'])} sell signals")
