"""
Dynamic Support and Resistance implementation based on PineScript logic.
Converted from PineScript to Python for the crypto-bot repository.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def find_pivot_high(df, period=10):
    """
    Find pivot highs in price data.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        period (int): Pivot period
        
    Returns:
        pandas.Series: Series with pivot high values
    """
    pivot_highs = pd.Series(index=df.index, dtype=float)
    
    for i in range(period, len(df) - period):
        left_range = df['high'].iloc[i-period:i]
        right_range = df['high'].iloc[i+1:i+period+1]
        
        if df['high'].iloc[i] > left_range.max() and df['high'].iloc[i] > right_range.max():
            pivot_highs.iloc[i] = df['high'].iloc[i]
    
    return pivot_highs

def find_pivot_low(df, period=10):
    """
    Find pivot lows in price data.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        period (int): Pivot period
        
    Returns:
        pandas.Series: Series with pivot low values
    """
    pivot_lows = pd.Series(index=df.index, dtype=float)
    
    for i in range(period, len(df) - period):
        left_range = df['low'].iloc[i-period:i]
        right_range = df['low'].iloc[i+1:i+period+1]
        
        if df['low'].iloc[i] < left_range.min() and df['low'].iloc[i] < right_range.min():
            pivot_lows.iloc[i] = df['low'].iloc[i]
    
    return pivot_lows

def get_sr_zones(pivot_values, max_channel_width, min_strength=2):
    """
    Calculate support and resistance zones from pivot values.
    
    Args:
        pivot_values (list): List of pivot values
        max_channel_width (float): Maximum channel width as percentage of price range
        min_strength (int): Minimum strength for a valid S/R zone
        
    Returns:
        list: List of tuples (high, low, strength) for each S/R zone
    """
    sr_zones = []
    
    for i in range(len(pivot_values)):
        if np.isnan(pivot_values[i]):
            continue
            
        lo = pivot_values[i]
        hi = lo
        strength = 0
        
        for j in range(len(pivot_values)):
            if np.isnan(pivot_values[j]):
                continue
                
            cpp = pivot_values[j]
            width = hi - lo if cpp <= lo else cpp - lo
            
            if width <= max_channel_width:
                if cpp <= hi:
                    lo = min(lo, cpp)
                else:
                    hi = max(hi, cpp)
                
                strength += 1
        
        if strength >= min_strength:
            sr_zones.append((hi, lo, strength))
    
    return sr_zones

def filter_sr_zones(sr_zones, max_zones=5):
    """
    Filter S/R zones by strength and remove overlapping zones.
    
    Args:
        sr_zones (list): List of tuples (high, low, strength) for each S/R zone
        max_zones (int): Maximum number of S/R zones to return
        
    Returns:
        list: Filtered list of S/R zones
    """
    sr_zones.sort(key=lambda x: x[2], reverse=True)
    
    filtered_zones = []
    
    for hi, lo, strength in sr_zones:
        overlap = False
        
        for f_hi, f_lo, _ in filtered_zones:
            if (lo <= f_hi and lo >= f_lo) or (hi >= f_lo and hi <= f_hi) or (lo <= f_lo and hi >= f_hi):
                overlap = True
                break
        
        if not overlap:
            filtered_zones.append((hi, lo, strength))
            
            if len(filtered_zones) >= max_zones:
                break
    
    return filtered_zones

def calculate_dynamic_support_resistance(df, pivot_period=10, max_pivot_count=20, 
                                        channel_width_pct=10, max_sr_count=5, min_strength=2):
    """
    Calculate dynamic support and resistance levels based on PineScript logic.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        pivot_period (int): Period for pivot point calculation
        max_pivot_count (int): Maximum number of pivot points to consider
        channel_width_pct (int): Maximum channel width as percentage of price range
        max_sr_count (int): Maximum number of S/R levels to return
        min_strength (int): Minimum strength for a valid S/R level
        
    Returns:
        pandas.DataFrame: DataFrame with added support and resistance columns
    """
    result_df = df.copy()
    
    # Ensure we have enough data for pivot calculation
    if len(df) < pivot_period * 2 + 1:
        logger.warning(f"Not enough data for pivot calculation. Need at least {pivot_period * 2 + 1} rows, got {len(df)}")
        result_df['resistance'] = np.nan
        result_df['support'] = np.nan
        result_df['resistance_tests'] = 0
        result_df['support_tests'] = 0
        result_df['dist_to_resistance'] = np.nan
        result_df['dist_to_support'] = np.nan
        return result_df
    
    # Find pivot highs and lows
    pivot_highs = find_pivot_high(df, period=pivot_period)
    pivot_lows = find_pivot_low(df, period=pivot_period)
    
    valid_pivots = []
    for i in range(len(df)):
        if not np.isnan(pivot_highs.iloc[i]):
            valid_pivots.append(pivot_highs.iloc[i])
        elif not np.isnan(pivot_lows.iloc[i]):
            valid_pivots.append(pivot_lows.iloc[i])
    
    valid_pivots = valid_pivots[:max_pivot_count]
    
    if not valid_pivots:
        logger.warning("No valid pivot points found")
        result_df['resistance'] = np.nan
        result_df['support'] = np.nan
        result_df['resistance_tests'] = 0
        result_df['support_tests'] = 0
        result_df['dist_to_resistance'] = np.nan
        result_df['dist_to_support'] = np.nan
        return result_df
    
    # Calculate maximum channel width
    price_highest = df['high'].max()
    price_lowest = df['low'].min()
    max_channel_width = (price_highest - price_lowest) * channel_width_pct / 100
    
    # Get S/R zones
    sr_zones = get_sr_zones(valid_pivots, max_channel_width, min_strength)
    
    # Filter S/R zones
    filtered_zones = filter_sr_zones(sr_zones, max_sr_count)
    
    result_df['resistance'] = np.nan
    result_df['support'] = np.nan
    result_df['resistance_tests'] = 0
    result_df['support_tests'] = 0
    
    last_close = df['close'].iloc[-1]
    
    for hi, lo, strength in filtered_zones:
        mid_price = (hi + lo) / 2
        
        if mid_price > last_close:
            for i in range(len(result_df)):
                result_df.loc[result_df.index[i], 'resistance'] = mid_price
        else:
            for i in range(len(result_df)):
                result_df.loc[result_df.index[i], 'support'] = mid_price
    
    resistance_zone_threshold = 0.005  # 0.5% threshold
    support_zone_threshold = 0.005     # 0.5% threshold
    
    for i in range(1, len(result_df)):
        if not np.isnan(result_df['resistance'].iloc[i-1]) and result_df['resistance'].iloc[i-1] > 0:
            price_to_res_ratio = (df['high'].iloc[i] - result_df['resistance'].iloc[i-1]) / result_df['resistance'].iloc[i-1]
            if -resistance_zone_threshold <= price_to_res_ratio <= 0.001:
                result_df.iloc[i, result_df.columns.get_loc('resistance_tests')] = result_df.iloc[i-1, result_df.columns.get_loc('resistance_tests')] + 1
            else:
                result_df.iloc[i, result_df.columns.get_loc('resistance_tests')] = result_df.iloc[i-1, result_df.columns.get_loc('resistance_tests')]
        
        if not np.isnan(result_df['support'].iloc[i-1]) and result_df['support'].iloc[i-1] > 0:
            price_to_sup_ratio = (df['low'].iloc[i] - result_df['support'].iloc[i-1]) / result_df['support'].iloc[i-1]
            if -0.001 <= price_to_sup_ratio <= support_zone_threshold:
                result_df.iloc[i, result_df.columns.get_loc('support_tests')] = result_df.iloc[i-1, result_df.columns.get_loc('support_tests')] + 1
            else:
                result_df.iloc[i, result_df.columns.get_loc('support_tests')] = result_df.iloc[i-1, result_df.columns.get_loc('support_tests')]
    
    # Calculate distance to resistance and support
    result_df['dist_to_resistance'] = np.nan
    result_df['dist_to_support'] = np.nan
    
    mask_resistance = ~np.isnan(result_df['resistance']) & (result_df['resistance'] > 0) & (result_df['close'] > 0)
    if mask_resistance.any():
        result_df.loc[mask_resistance, 'dist_to_resistance'] = (result_df.loc[mask_resistance, 'resistance'] - result_df.loc[mask_resistance, 'close']) / result_df.loc[mask_resistance, 'close']
    
    mask_support = ~np.isnan(result_df['support']) & (result_df['support'] > 0) & (result_df['close'] > 0)
    if mask_support.any():
        result_df.loc[mask_support, 'dist_to_support'] = (result_df.loc[mask_support, 'close'] - result_df.loc[mask_support, 'support']) / result_df.loc[mask_support, 'close']
    
    return result_df

def detect_sr_breakouts(df):
    """
    Detect breakouts of support and resistance levels.
    
    Args:
        df (pandas.DataFrame): DataFrame with support and resistance data
        
    Returns:
        tuple: (resistance_breakouts, support_breakouts)
    """
    resistance_breakouts = []
    support_breakouts = []
    
    for i in range(1, len(df)):
        if not np.isnan(df['resistance'].iloc[i]) and df['close'].iloc[i-1] <= df['resistance'].iloc[i] and df['close'].iloc[i] > df['resistance'].iloc[i]:
            resistance_breakouts.append(df.index[i])
        
        if not np.isnan(df['support'].iloc[i]) and df['close'].iloc[i-1] >= df['support'].iloc[i] and df['close'].iloc[i] < df['support'].iloc[i]:
            support_breakouts.append(df.index[i])
    
    return resistance_breakouts, support_breakouts
