"""
Script to collect data from CoinDesk, train a model, run backtest, and analyze support/resistance zones.
"""
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
from coindesk_client import CoinDeskClient
from strategy import TradingStrategy
from backtester import Backtester
from indicators import add_indicators
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("coindesk_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CoinDesk Data Analysis')
    
    parser.add_argument('--symbol', type=str, default='BTC-USDT-VANILLA-PERPETUAL',
                        help='Trading pair symbol (default: BTC-USDT-VANILLA-PERPETUAL)')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Kline interval (default: 1h)')
    
    parser.add_argument('--limit', type=int, default=2000,
                        help='Maximum number of records to fetch (default: 2000)')
    
    parser.add_argument('--market', type=str, default='binance',
                        help='Market name (default: binance)')
    
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital for backtest (default: 10000.0)')
    
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
    
    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = os.path.join('data', f"coindesk_raw_{args.symbol.replace('-', '_')}_{timestamp}.csv")
    df.to_csv(raw_file)
    logger.info(f"Raw data saved to {raw_file}")
    
    return df

def process_data(df):
    """
    Process raw data by adding technical indicators.
    
    Args:
        df: DataFrame with raw OHLCV data
        
    Returns:
        pandas.DataFrame: DataFrame with added indicators
    """
    logger.info("Processing data and adding technical indicators")
    
    processed_df = add_indicators(df)
    
    logger.info(f"Processed {len(processed_df)} rows of data")
    
    return processed_df

def analyze_support_resistance(df):
    """
    Analyze support and resistance zones.
    
    Args:
        df: DataFrame with processed data
        
    Returns:
        tuple: (DataFrame, dict of support/resistance stats)
    """
    logger.info("Analyzing support and resistance zones")
    
    if 'resistance' not in df.columns or 'support' not in df.columns:
        logger.warning("Support/resistance data not available")
        return df, {}
    
    resistance_levels = df[df['resistance'].diff() != 0]['resistance'].dropna().unique()
    support_levels = df[df['support'].diff() != 0]['support'].dropna().unique()
    
    resistance_tests = {}
    for level in resistance_levels:
        max_tests = df[df['resistance'] == level]['resistance_tests'].max()
        if not pd.isna(max_tests):
            resistance_tests[level] = int(max_tests)
    
    support_tests = {}
    for level in support_levels:
        max_tests = df[df['support'] == level]['support_tests'].max()
        if not pd.isna(max_tests):
            support_tests[level] = int(max_tests)
    
    resistance_tests = {k: v for k, v in sorted(resistance_tests.items(), key=lambda item: item[1], reverse=True)}
    support_tests = {k: v for k, v in sorted(support_tests.items(), key=lambda item: item[1], reverse=True)}
    
    stats = {
        'resistance_levels': resistance_tests,
        'support_levels': support_tests
    }
    
    return df, stats

def plot_support_resistance(df, stats, save_path=None):
    """
    Plot price chart with support and resistance zones.
    
    Args:
        df: DataFrame with processed data
        stats: Dict with support/resistance statistics
        save_path: Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    logger.info("Plotting support and resistance zones")
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(df.index, df['close'], label='Price', color='blue', alpha=0.7)
    
    resistance_levels = list(stats['resistance_levels'].items())
    for i, (level, tests) in enumerate(resistance_levels[:5]):
        color = f'C{i+1}'
        ax.axhline(y=level, color=color, linestyle='--', alpha=0.7, 
                   label=f'Resistance {level:.2f} (tested {tests} times)')
    
    support_levels = list(stats['support_levels'].items())
    for i, (level, tests) in enumerate(support_levels[:5]):
        color = f'C{i+6}'
        ax.axhline(y=level, color=color, linestyle='-.', alpha=0.7,
                   label=f'Support {level:.2f} (tested {tests} times)')
    
    ax.set_title('Price with Support and Resistance Zones')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def run_analysis(args):
    """
    Run the full analysis pipeline.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting analysis pipeline")
    
    raw_df = collect_data(args)
    if raw_df is None:
        return
    
    processed_df = process_data(raw_df)
    
    analyzed_df, sr_stats = analyze_support_resistance(processed_df)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    processed_file = os.path.join(args.output_dir, f"processed_{args.symbol.replace('-', '_')}_{timestamp}.csv")
    processed_df.to_csv(processed_file)
    logger.info(f"Processed data saved to {processed_file}")
    
    plot_file = os.path.join(args.output_dir, f"support_resistance_{args.symbol.replace('-', '_')}_{timestamp}.png")
    plot_support_resistance(analyzed_df, sr_stats, save_path=plot_file)
    
    strategy = TradingStrategy()
    backtester = Backtester(strategy)
    
    results, metrics = backtester.run(
        symbol=args.symbol.split('-')[0] + args.symbol.split('-')[1],  # Convert format to BTCUSDT
        data=processed_df
    )
    
    results_file = os.path.join(args.output_dir, f"backtest_results_{args.symbol.replace('-', '_')}_{timestamp}.csv")
    results.to_csv(results_file)
    logger.info(f"Backtest results saved to {results_file}")
    
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(args.output_dir, f"backtest_metrics_{args.symbol.replace('-', '_')}_{timestamp}.csv")
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Backtest metrics saved to {metrics_file}")
    
    backtest_plot_file = os.path.join(args.output_dir, f"backtest_plot_{args.symbol.replace('-', '_')}_{timestamp}.png")
    backtester.plot_results(results, metrics, save_path=backtest_plot_file)
    
    print("\n" + "="*50)
    print(f"ANALYSIS RESULTS FOR {args.symbol}")
    print("="*50)
    print("\nSUPPORT AND RESISTANCE ZONES:")
    print("-"*50)
    print("\nTop Resistance Levels (price: test count):")
    for level, tests in list(sr_stats['resistance_levels'].items())[:5]:
        print(f"  {level:.2f}: tested {tests} times")
    
    print("\nTop Support Levels (price: test count):")
    for level, tests in list(sr_stats['support_levels'].items())[:5]:
        print(f"  {level:.2f}: tested {tests} times")
    
    print("\n" + "="*50)
    print(f"BACKTEST RESULTS:")
    print("="*50)
    print(f"Initial Capital: ${args.initial_capital:.2f}")
    print(f"Final Capital: ${metrics['final_capital']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print("="*50)
    
    return results, metrics, sr_stats

def main():
    """Main function."""
    args = parse_args()
    run_analysis(args)

if __name__ == "__main__":
    main()
