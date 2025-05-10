"""
Test script for the enhanced resistance detection implementation.
"""
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from coindesk_client import CoinDeskClient
from indicators import add_indicators
from enhanced_resistance_detection import calculate_enhanced_support_resistance
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_resistance_test.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function to test the enhanced resistance detection.
    """
    logger.info("Starting enhanced resistance detection test for SUI-USDT")
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    logger.info("Collecting data from CoinDesk for SUI-USDT-VANILLA-PERPETUAL")
    
    client = CoinDeskClient(api_key=config.COINDESK_API_KEY)
    
    df = client.get_historical_klines(
        symbol="SUI-USDT-VANILLA-PERPETUAL",
        market="binance",
        limit=3000,
        interval="1d",
        timeframe="days"
    )
    
    logger.info(f"Collected {len(df)} candles from CoinDesk")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("Processing data and adding technical indicators")
    processed_df = add_indicators(df)
    logger.info(f"Processed {len(processed_df)} rows of data")
    
    logger.info("Analyzing support and resistance zones with enhanced resistance detection")
    
    for resistance_bias in [0.6, 0.8, 1.0]:
        support_bias = 1.0 - resistance_bias
        
        logger.info(f"Testing with resistance_bias={resistance_bias}, support_bias={support_bias}")
        
        enhanced_df = calculate_enhanced_support_resistance(
            processed_df, 
            pivot_period=7,
            max_pivot_count=40,
            channel_width_pct=5,
            max_sr_count=8,
            min_strength=1,
            resistance_bias=resistance_bias,
            support_bias=support_bias
        )
        
        plot_path = f"results/enhanced_resistance_SUI_USDT_rb{int(resistance_bias*10)}_{timestamp}.png"
        plot_support_resistance(enhanced_df, plot_path)
        logger.info(f"Plot saved to {plot_path}")
        
        print_detected_levels(enhanced_df, resistance_bias, support_bias)

def plot_support_resistance(df, save_path):
    """
    Plot price with support and resistance zones.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and S/R data
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(14, 7))
    
    plt.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.7)
    
    if 'resistance' in df.columns and not df['resistance'].isna().all():
        resistance_levels = df['resistance'].dropna().unique()
        for level in resistance_levels:
            plt.axhline(y=level, color='red', linestyle='--', alpha=0.5, 
                        label=f'Resistance {level:.2f}' if level == resistance_levels[0] else "")
    
    if 'support' in df.columns and not df['support'].isna().all():
        support_levels = df['support'].dropna().unique()
        for level in support_levels:
            plt.axhline(y=level, color='green', linestyle='--', alpha=0.5,
                        label=f'Support {level:.2f}' if level == support_levels[0] else "")
    
    plt.title('SUI-USDT Price with Enhanced Support and Resistance Levels')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_detected_levels(df, resistance_bias, support_bias):
    """
    Print detected support and resistance levels.
    
    Args:
        df (pandas.DataFrame): DataFrame with support and resistance data
        resistance_bias (float): Resistance bias used
        support_bias (float): Support bias used
    """
    print("\n" + "=" * 50)
    print(f"DETECTED LEVELS WITH BIAS: Resistance={resistance_bias}, Support={support_bias}")
    print("=" * 50)
    
    print("\nResistance Levels:")
    if 'resistance' in df.columns and not df['resistance'].isna().all():
        resistance_df = df[['resistance', 'resistance_tests']].drop_duplicates().dropna()
        if not resistance_df.empty:
            resistance_df = resistance_df.sort_values('resistance_tests', ascending=False).head(5)
            for _, row in resistance_df.iterrows():
                print(f"  {row['resistance']:.2f}: tested {int(row['resistance_tests'])} times")
        else:
            print("  No significant resistance levels detected")
    else:
        print("  Resistance data not available")
    
    print("\nSupport Levels:")
    if 'support' in df.columns and not df['support'].isna().all():
        support_df = df[['support', 'support_tests']].drop_duplicates().dropna()
        if not support_df.empty:
            support_df = support_df.sort_values('support_tests', ascending=False).head(5)
            for _, row in support_df.iterrows():
                print(f"  {row['support']:.2f}: tested {int(row['support_tests'])} times")
        else:
            print("  No significant support levels detected")
    else:
        print("  Support data not available")
    
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
