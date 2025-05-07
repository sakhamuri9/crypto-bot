"""
Script to compare different support and resistance detection methods.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from coindesk_client import CoinDeskClient
from indicators import add_indicators
from improved_indicators import add_indicators as add_improved_indicators
from improved_indicators import analyze_support_resistance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("support_resistance_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Support/Resistance Comparison')
    
    parser.add_argument('--symbol', type=str, default='BTC-USDT-VANILLA-PERPETUAL',
                        help='Trading pair symbol (default: BTC-USDT-VANILLA-PERPETUAL)')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Kline interval (default: 1h)')
    
    parser.add_argument('--limit', type=int, default=2000,
                        help='Maximum number of records to fetch (default: 2000)')
    
    parser.add_argument('--market', type=str, default='binance',
                        help='Market name (default: binance)')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    
    return parser.parse_args()

def collect_data(args):
    """
    Collect data from CoinDesk API.
    
    Args:
        args: Command line arguments
        
    Returns:
        pandas.DataFrame: DataFrame with OHLCV data
    """
    logger.info(f"Collecting data from CoinDesk for {args.symbol} with {args.timeframe} timeframe")
    
    client = CoinDeskClient()
    
    df = client.get_historical_klines(
        symbol=args.symbol,
        interval=args.timeframe,
        limit=args.limit,
        market=args.market
    )
    
    if df.empty:
        logger.error("Failed to collect data from CoinDesk")
        return None
    
    logger.info(f"Collected {len(df)} candles from CoinDesk")
    return df

def plot_support_resistance_comparison(original_df, improved_df, original_stats, improved_stats, save_path=None):
    """
    Plot price chart with support and resistance zones from both methods.
    
    Args:
        original_df: DataFrame with original indicators
        improved_df: DataFrame with improved indicators
        original_stats: Dict with original support/resistance statistics
        improved_stats: Dict with improved support/resistance statistics
        save_path: Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info("Plotting support and resistance comparison")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    ax1.plot(original_df.index, original_df['close'], label='Price', color='blue', alpha=0.7)
    ax1.set_title('Original Support/Resistance Method (Counts Each Candle)')
    
    resistance_levels = list(original_stats['resistance_levels'].items())
    for i, (level, tests) in enumerate(resistance_levels[:5]):
        color = f'C{i+1}'
        ax1.axhline(y=level, color=color, linestyle='--', alpha=0.7, 
                   label=f'Resistance {level:.2f} (tested {tests} times)')
    
    support_levels = list(original_stats['support_levels'].items())
    for i, (level, tests) in enumerate(support_levels[:5]):
        color = f'C{i+6}'
        ax1.axhline(y=level, color=color, linestyle='-.', alpha=0.7,
                   label=f'Support {level:.2f} (tested {tests} times)')
    
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(improved_df.index, improved_df['close'], label='Price', color='blue', alpha=0.7)
    ax2.set_title('Improved Support/Resistance Method (Counts Test Events)')
    
    resistance_levels = list(improved_stats['resistance_levels'].items())
    for i, (level, tests) in enumerate(resistance_levels[:5]):
        color = f'C{i+1}'
        ax2.axhline(y=level, color=color, linestyle='--', alpha=0.7, 
                   label=f'Resistance {level:.2f} (tested {tests} times)')
    
    support_levels = list(improved_stats['support_levels'].items())
    for i, (level, tests) in enumerate(support_levels[:5]):
        color = f'C{i+6}'
        ax2.axhline(y=level, color=color, linestyle='-.', alpha=0.7,
                   label=f'Support {level:.2f} (tested {tests} times)')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def run_comparison(args):
    """
    Run the comparison between original and improved support/resistance methods.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting support/resistance comparison")
    
    raw_df = collect_data(args)
    if raw_df is None:
        return
    
    logger.info("Processing data with original indicators")
    original_df = add_indicators(raw_df)
    
    logger.info("Processing data with improved indicators")
    improved_df = add_improved_indicators(raw_df)
    
    logger.info("Analyzing support and resistance zones")
    
    original_resistance_levels = original_df[original_df['resistance'].diff() != 0]['resistance'].dropna().unique()
    original_support_levels = original_df[original_df['support'].diff() != 0]['support'].dropna().unique()
    
    original_resistance_tests = {}
    for level in original_resistance_levels:
        max_tests = original_df[original_df['resistance'] == level]['resistance_tests'].max()
        if not pd.isna(max_tests):
            original_resistance_tests[level] = int(max_tests)
    
    original_support_tests = {}
    for level in original_support_levels:
        max_tests = original_df[original_df['support'] == level]['support_tests'].max()
        if not pd.isna(max_tests):
            original_support_tests[level] = int(max_tests)
    
    original_resistance_tests = {k: v for k, v in sorted(original_resistance_tests.items(), key=lambda item: item[1], reverse=True)}
    original_support_tests = {k: v for k, v in sorted(original_support_tests.items(), key=lambda item: item[1], reverse=True)}
    
    original_stats = {
        'resistance_levels': original_resistance_tests,
        'support_levels': original_support_tests
    }
    
    improved_stats = analyze_support_resistance(improved_df)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_file = os.path.join(args.output_dir, f"support_resistance_comparison_{args.symbol.replace('-', '_')}_{timestamp}.png")
    plot_support_resistance_comparison(original_df, improved_df, original_stats, improved_stats, save_path=plot_file)
    
    print("\n" + "="*50)
    print(f"SUPPORT AND RESISTANCE COMPARISON FOR {args.symbol}")
    print("="*50)
    
    print("\nORIGINAL METHOD (Counts Each Candle):")
    print("-"*50)
    print("\nTop Resistance Levels (price: test count):")
    for level, tests in list(original_stats['resistance_levels'].items())[:5]:
        print(f"  {level:.2f}: tested {tests} times")
    
    print("\nTop Support Levels (price: test count):")
    for level, tests in list(original_stats['support_levels'].items())[:5]:
        print(f"  {level:.2f}: tested {tests} times")
    
    print("\nIMPROVED METHOD (Counts Test Events):")
    print("-"*50)
    print("\nTop Resistance Levels (price: test count):")
    for level, tests in list(improved_stats['resistance_levels'].items())[:5]:
        print(f"  {level:.2f}: tested {tests} times")
    
    print("\nTop Support Levels (price: test count):")
    for level, tests in list(improved_stats['support_levels'].items())[:5]:
        print(f"  {level:.2f}: tested {tests} times")
    
    print("\n" + "="*50)
    print(f"Comparison plot saved to: {plot_file}")
    print("="*50)

def main():
    """Main function."""
    args = parse_args()
    run_comparison(args)

if __name__ == "__main__":
    main()
