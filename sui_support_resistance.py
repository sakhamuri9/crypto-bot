"""
Enhanced Support and Resistance implementation optimized for SUI-USDT.
Based on the original support_resistance.py but with parameters tuned for SUI.
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

def calculate_sui_support_resistance(df, pivot_period=7, max_pivot_count=40, 
                                    channel_width_pct=5, max_sr_count=8, min_strength=1):
    """
    Calculate dynamic support and resistance levels optimized for SUI-USDT.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        pivot_period (int): Period for pivot point calculation (smaller for SUI)
        max_pivot_count (int): Maximum number of pivot points to consider (larger for SUI)
        channel_width_pct (int): Maximum channel width as percentage of price range (smaller for SUI)
        max_sr_count (int): Maximum number of S/R levels to return (larger for SUI)
        min_strength (int): Minimum strength for a valid S/R level (smaller for SUI)
        
    Returns:
        pandas.DataFrame: DataFrame with added support and resistance columns
    """
    result_df = df.copy()
    
    if len(df) < pivot_period * 2 + 1:
        logger.warning(f"Not enough data for pivot calculation. Need at least {pivot_period * 2 + 1} rows, got {len(df)}")
        result_df['resistance'] = np.nan
        result_df['support'] = np.nan
        result_df['resistance_tests'] = 0
        result_df['support_tests'] = 0
        result_df['dist_to_resistance'] = np.nan
        result_df['dist_to_support'] = np.nan
        return result_df
    
    src_high = df['high']
    src_low = df['low']
    
    pivot_highs_short = find_pivot_high(df, period=max(3, pivot_period // 2))
    pivot_lows_short = find_pivot_low(df, period=max(3, pivot_period // 2))
    
    pivot_highs_medium = find_pivot_high(df, period=pivot_period)
    pivot_lows_medium = find_pivot_low(df, period=pivot_period)
    
    pivot_highs_long = find_pivot_high(df, period=pivot_period * 2)
    pivot_lows_long = find_pivot_low(df, period=pivot_period * 2)
    
    all_pivots = []
    
    for i in range(len(df)):
        if not np.isnan(pivot_highs_short.iloc[i]):
            volume_factor = min(df['volume'].iloc[i] / df['volume'].rolling(20).mean().iloc[i], 3) if i >= 20 else 1
            recency_factor = 1 + 0.7 * (i / len(df))  # Increased from 0.5 to 0.7
            strength = 1 * volume_factor * recency_factor
            all_pivots.append((pivot_highs_short.iloc[i], 1, strength, i))  # value, type (1=high), strength, index
        
        if not np.isnan(pivot_highs_medium.iloc[i]):
            volume_factor = min(df['volume'].iloc[i] / df['volume'].rolling(20).mean().iloc[i], 3) if i >= 20 else 1
            recency_factor = 1 + 0.7 * (i / len(df))
            strength = 2 * volume_factor * recency_factor  # Medium-term pivots are stronger
            all_pivots.append((pivot_highs_medium.iloc[i], 1, strength, i))
        
        if not np.isnan(pivot_highs_long.iloc[i]):
            volume_factor = min(df['volume'].iloc[i] / df['volume'].rolling(20).mean().iloc[i], 3) if i >= 20 else 1
            recency_factor = 1 + 0.7 * (i / len(df))
            strength = 3 * volume_factor * recency_factor  # Long-term pivots are strongest
            all_pivots.append((pivot_highs_long.iloc[i], 1, strength, i))
    
    for i in range(len(df)):
        if not np.isnan(pivot_lows_short.iloc[i]):
            volume_factor = min(df['volume'].iloc[i] / df['volume'].rolling(20).mean().iloc[i], 3) if i >= 20 else 1
            recency_factor = 1 + 0.7 * (i / len(df))
            strength = 1 * volume_factor * recency_factor
            all_pivots.append((pivot_lows_short.iloc[i], -1, strength, i))  # value, type (-1=low), strength, index
        
        if not np.isnan(pivot_lows_medium.iloc[i]):
            volume_factor = min(df['volume'].iloc[i] / df['volume'].rolling(20).mean().iloc[i], 3) if i >= 20 else 1
            recency_factor = 1 + 0.7 * (i / len(df))
            strength = 2 * volume_factor * recency_factor
            all_pivots.append((pivot_lows_medium.iloc[i], -1, strength, i))
        
        if not np.isnan(pivot_lows_long.iloc[i]):
            volume_factor = min(df['volume'].iloc[i] / df['volume'].rolling(20).mean().iloc[i], 3) if i >= 20 else 1
            recency_factor = 1 + 0.7 * (i / len(df))
            strength = 3 * volume_factor * recency_factor
            all_pivots.append((pivot_lows_long.iloc[i], -1, strength, i))
    
    if not all_pivots:
        logger.warning("No valid pivot points found")
        result_df['resistance'] = np.nan
        result_df['support'] = np.nan
        result_df['resistance_tests'] = 0
        result_df['support_tests'] = 0
        result_df['dist_to_resistance'] = np.nan
        result_df['dist_to_support'] = np.nan
        return result_df
    
    all_pivots.sort(key=lambda x: x[2], reverse=True)
    
    top_pivots = all_pivots[:max_pivot_count]
    
    highest = df['high'].max()
    lowest = df['low'].min()
    price_range = highest - lowest
    
    if 'atr' in df.columns:
        recent_volatility = df['atr'].iloc[-20:].mean() if len(df) >= 20 else df['atr'].mean()
    else:
        recent_high = df['high'].iloc[-20:].max() if len(df) >= 20 else df['high'].max()
        recent_low = df['low'].iloc[-20:].min() if len(df) >= 20 else df['low'].min()
        recent_volatility = (recent_high - recent_low) / df['close'].mean()
    
    volatility_factor = max(0.4, min(2.5, 1.0 / (recent_volatility * 80) if recent_volatility > 0 else 1.0))
    adaptive_channel_width = price_range * (channel_width_pct / 100) * volatility_factor
    
    sr_zones = []
    
    for pivot_value, pivot_type, pivot_strength, pivot_idx in top_pivots:
        zone_high = pivot_value
        zone_low = pivot_value
        zone_strength = pivot_strength
        zone_type = pivot_type  # 1 for resistance, -1 for support
        
        for other_value, other_type, other_strength, other_idx in all_pivots:
            if pivot_idx == other_idx:
                continue
            
            price_diff_pct = abs(other_value - pivot_value) / pivot_value
            if price_diff_pct <= adaptive_channel_width / price_range:
                zone_high = max(zone_high, other_value)
                zone_low = min(zone_low, other_value)
                
                zone_strength += other_strength * 0.6  # Increased from 0.5
                
                if other_type != zone_type:
                    if other_strength > pivot_strength:
                        zone_type = other_type
        
        if zone_strength >= min_strength:
            sr_zones.append((zone_high, zone_low, zone_strength, zone_type))
    
    filtered_zones = []
    
    sr_zones.sort(key=lambda x: x[2], reverse=True)
    
    for zone_high, zone_low, zone_strength, zone_type in sr_zones:
        overlaps = False
        
        for f_high, f_low, _, _ in filtered_zones:
            overlap_threshold = 0.7  # 70% overlap allowed
            zone_range = zone_high - zone_low
            f_range = f_high - f_low
            
            overlap_high = min(zone_high, f_high)
            overlap_low = max(zone_low, f_low)
            
            if overlap_high > overlap_low:  # There is overlap
                overlap_amount = (overlap_high - overlap_low) / min(zone_range, f_range) if min(zone_range, f_range) > 0 else 0
                if overlap_amount > overlap_threshold:
                    overlaps = True
                    break
        
        if not overlaps:
            filtered_zones.append((zone_high, zone_low, zone_strength, zone_type))
            
            if len(filtered_zones) >= max_sr_count:
                break
    
    result_df['resistance'] = np.nan
    result_df['support'] = np.nan
    result_df['resistance_tests'] = 0
    result_df['support_tests'] = 0
    
    for zone_high, zone_low, zone_strength, zone_type in filtered_zones:
        zone_mid = (zone_high + zone_low) / 2
        
        crosses_above = 0
        crosses_below = 0
        touches = 0
        
        for i in range(1, len(df)):
            if df['close'].iloc[i-1] < zone_mid and df['close'].iloc[i] > zone_mid:
                crosses_above += 1
            
            if df['close'].iloc[i-1] > zone_mid and df['close'].iloc[i] < zone_mid:
                crosses_below += 1
            
            if abs(df['close'].iloc[i] - zone_mid) / zone_mid < 0.003:  # Increased from 0.002
                touches += 1
        
        recent_start = max(0, int(len(df) * 0.7))  # Changed from 0.8 to 0.7
        recent_df = df.iloc[recent_start:]
        
        time_above = sum(1 for i in range(len(recent_df)) if recent_df['close'].iloc[i] > zone_mid)
        time_below = len(recent_df) - time_above
        
        current_price = df['close'].iloc[-1]
        price_above = current_price > zone_mid
        
        is_resistance = False
        is_support = False
        
        if zone_type == 1:
            is_resistance = True
        elif zone_type == -1:
            is_support = True
        
        if time_above > time_below * 1.5:  # Reduced from 2 to 1.5
            is_support = True
            is_resistance = False
        elif time_below > time_above * 1.5:  # Reduced from 2 to 1.5
            is_resistance = True
            is_support = False
        
        if price_above:
            if crosses_below > crosses_above * 1.2:  # Added factor for SUI
                is_resistance = True
                is_support = False
            else:
                is_support = True
                is_resistance = False
        else:
            if crosses_above > crosses_below * 1.2:  # Added factor for SUI
                is_support = True
                is_resistance = False
            else:
                is_resistance = True
                is_support = False
        
        if touches > 3:
            if is_resistance:
                is_resistance = True
                is_support = False
            elif is_support:
                is_support = True
                is_resistance = False
        
        if is_resistance:
            result_df['resistance'] = zone_mid
        elif is_support:
            result_df['support'] = zone_mid
    
    resistance_test_threshold = 0.007  # Increased from 0.005
    support_test_threshold = 0.007  # Increased from 0.005
    
    for i in range(1, len(result_df)):
        if not np.isnan(result_df['resistance'].iloc[i-1]) and result_df['resistance'].iloc[i-1] > 0:
            price_to_res_ratio = (df['high'].iloc[i] - result_df['resistance'].iloc[i-1]) / result_df['resistance'].iloc[i-1]
            
            if -resistance_test_threshold <= price_to_res_ratio <= 0.002:  # Increased upper bound
                result_df.iloc[i, result_df.columns.get_loc('resistance_tests')] = result_df.iloc[i-1, result_df.columns.get_loc('resistance_tests')] + 1
            else:
                result_df.iloc[i, result_df.columns.get_loc('resistance_tests')] = result_df.iloc[i-1, result_df.columns.get_loc('resistance_tests')]
        
        if not np.isnan(result_df['support'].iloc[i-1]) and result_df['support'].iloc[i-1] > 0:
            price_to_sup_ratio = (df['low'].iloc[i] - result_df['support'].iloc[i-1]) / result_df['support'].iloc[i-1]
            
            if -0.002 <= price_to_sup_ratio <= support_test_threshold:  # Increased lower bound
                result_df.iloc[i, result_df.columns.get_loc('support_tests')] = result_df.iloc[i-1, result_df.columns.get_loc('support_tests')] + 1
            else:
                result_df.iloc[i, result_df.columns.get_loc('support_tests')] = result_df.iloc[i-1, result_df.columns.get_loc('support_tests')]
    
    result_df['dist_to_resistance'] = np.nan
    result_df['dist_to_support'] = np.nan
    
    mask_resistance = ~np.isnan(result_df['resistance']) & (result_df['resistance'] > 0) & (df['close'] > 0)
    if mask_resistance.any():
        result_df.loc[mask_resistance, 'dist_to_resistance'] = (result_df.loc[mask_resistance, 'resistance'] - df.loc[mask_resistance.index, 'close']) / df.loc[mask_resistance.index, 'close']
    
    mask_support = ~np.isnan(result_df['support']) & (result_df['support'] > 0) & (df['close'] > 0)
    if mask_support.any():
        result_df.loc[mask_support, 'dist_to_support'] = (df.loc[mask_support.index, 'close'] - result_df.loc[mask_support, 'support']) / df.loc[mask_support.index, 'close']
    
    return result_df

def detect_sui_sr_breakouts(df):
    """
    Detect breakouts of support and resistance levels for SUI-USDT.
    
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
