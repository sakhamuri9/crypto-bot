"""
Comprehensive multi-cryptocurrency analysis script with optimized support/resistance detection.
Supports both BTC-USDT and SUI-USDT with cryptocurrency-specific parameter optimization.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

from coindesk_client import CoinDeskClient
from indicators import add_indicators
from strategy import TradingStrategy
from backtester import Backtester
from support_resistance import calculate_dynamic_support_resistance
from sui_support_resistance import calculate_sui_support_resistance
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multi_crypto_analysis.log')
    ]
)

logger = logging.getLogger(__name__)

def analyze_cryptocurrency(symbol, timeframe='hours', interval='1h', limit=2000, 
                          use_sui_optimized=False, plot_results=True):
    """
    Analyze a cryptocurrency with optimized parameters.
    
    Args:
        symbol (str): Symbol to analyze (e.g., 'BTC-USDT-VANILLA-PERPETUAL')
        timeframe (str): Data timeframe ('hours' or 'days')
        interval (str): Interval for data (e.g., '1h', '1d')
        limit (int): Maximum number of candles to retrieve
        use_sui_optimized (bool): Whether to use SUI-optimized parameters
        plot_results (bool): Whether to plot results
        
    Returns:
        dict: Analysis results
    """
    logger.info(f"Starting analysis for {symbol} with {timeframe} timeframe")
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    base_symbol = symbol.split('-')[0]
    
    logger.info(f"Collecting data from CoinDesk for {symbol}")
    
    client = CoinDeskClient(api_key=config.COINDESK_API_KEY)
    
    df = client.get_historical_klines(
        symbol=symbol,
        market="binance",
        limit=limit,
        interval=interval,
        timeframe=timeframe
    )
    
    logger.info(f"Collected {len(df)} candles from CoinDesk")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_data_path = f"data/coindesk_raw_{symbol}_{timestamp}.csv"
    df.to_csv(raw_data_path)
    logger.info(f"Raw data saved to {raw_data_path}")
    
    logger.info("Processing data and adding technical indicators")
    processed_df = add_indicators(df)
    logger.info(f"Processed {len(processed_df)} rows of data")
    
    if use_sui_optimized:
        logger.info("Analyzing support and resistance zones with enhanced resistance detection")
        from enhanced_resistance_detection import calculate_enhanced_support_resistance
        processed_df = calculate_enhanced_support_resistance(
            processed_df, 
            pivot_period=7,
            max_pivot_count=40,
            channel_width_pct=5,
            max_sr_count=8,
            min_strength=1,
            resistance_bias=0.8,
            support_bias=0.2
        )
    else:
        logger.info("Analyzing support and resistance zones with enhanced detection for BTC")
        from enhanced_resistance_detection import calculate_enhanced_support_resistance
        processed_df = calculate_enhanced_support_resistance(
            processed_df,
            pivot_period=10,
            max_pivot_count=30,
            channel_width_pct=8,
            max_sr_count=6,
            min_strength=2,
            resistance_bias=0.7,
            support_bias=0.3
        )
    
    processed_data_path = f"results/processed_{symbol}_{timestamp}.csv"
    processed_df.to_csv(processed_data_path)
    logger.info(f"Processed data saved to {processed_data_path}")
    
    if plot_results:
        logger.info("Plotting support and resistance zones")
        plot_support_resistance(processed_df, symbol, f"results/support_resistance_{symbol}_{timestamp}.png")
    
    strategy = TradingStrategy()
    market_regime = strategy.detect_market_regime(processed_df)
    logger.info(f"Detected market regime: {market_regime}")
    
    strategy.train_ml_model(processed_df, prediction_window=10)
    
    signals_df = strategy.generate_signals(processed_df)
    
    if 'SUI' in symbol:
        from sui_backtester import Backtester as SuiBacktester
        backtester = SuiBacktester(
            initial_capital=config.INITIAL_CAPITAL,
            position_size=config.POSITION_SIZE,
            stop_loss_pct=config.STOP_LOSS_PCT,
            take_profit_pct=config.TAKE_PROFIT_PCT
        )
    else:
        strategy = TradingStrategy()
        backtester = Backtester(strategy=strategy)
    
    if 'SUI' in symbol:
        backtest_results = backtester.run_backtest(signals_df)
        
        backtest_results.to_csv(f"results/backtest_results_{symbol}_{timestamp}.csv")
        logger.info(f"Backtest results saved to results/backtest_results_{symbol}_{timestamp}.csv")
        
        metrics = backtester.calculate_metrics()
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"results/backtest_metrics_{symbol}_{timestamp}.csv")
        logger.info(f"Backtest metrics saved to results/backtest_metrics_{symbol}_{timestamp}.csv")
        
        if plot_results:
            backtester.plot_results(f"results/backtest_plot_{symbol}_{timestamp}.png")
    else:
        backtest_results, metrics = backtester.run(data=signals_df)
        
        backtest_results.to_csv(f"results/backtest_results_{symbol}_{timestamp}.csv")
        logger.info(f"Backtest results saved to results/backtest_results_{symbol}_{timestamp}.csv")
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"results/backtest_metrics_{symbol}_{timestamp}.csv")
        logger.info(f"Backtest metrics saved to results/backtest_metrics_{symbol}_{timestamp}.csv")
        
        if plot_results:
            backtester.plot_results(backtest_results, metrics, f"results/backtest_plot_{symbol}_{timestamp}.png")
    
    support_levels = {}
    resistance_levels = {}
    
    if 'support' in processed_df.columns and 'support_tests' in processed_df.columns:
        support_df = processed_df[['support', 'support_tests']].drop_duplicates().dropna()
        if not support_df.empty:
            support_df = support_df.sort_values('support_tests', ascending=False).head(5)
            for _, row in support_df.iterrows():
                support_levels[str(row['support'])] = int(row['support_tests'])
    
    if 'resistance' in processed_df.columns and 'resistance_tests' in processed_df.columns:
        resistance_df = processed_df[['resistance', 'resistance_tests']].drop_duplicates().dropna()
        if not resistance_df.empty:
            resistance_df = resistance_df.sort_values('resistance_tests', ascending=False).head(5)
            for _, row in resistance_df.iterrows():
                resistance_levels[str(row['resistance'])] = int(row['resistance_tests'])
    
    results = {
        "symbol": symbol,
        "timestamp": timestamp,
        "market_regime": market_regime,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "metrics": metrics,
        "data_points": len(processed_df),
        "timeframe": timeframe,
        "interval": interval,
        "sr_method": "SUI-optimized" if use_sui_optimized else "Standard"
    }
    
    print_analysis_summary(results)
    
    return results

def plot_support_resistance(df, symbol, save_path):
    """
    Plot price with support and resistance zones.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and S/R data
        symbol (str): Symbol being analyzed
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
    
    plt.title(f'{symbol} Price with Support and Resistance Levels')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_analysis_summary(results):
    """
    Print a summary of the analysis results.
    
    Args:
        results (dict): Analysis results
    """
    symbol = results["symbol"]
    
    print("\n" + "=" * 50)
    print(f"ANALYSIS RESULTS FOR {symbol}")
    print("=" * 50 + "\n")
    
    print(f"Analysis Method: {results['sr_method']}")
    print(f"Market Regime: {results['market_regime']}")
    print(f"Data Points: {results['data_points']}")
    print(f"Timeframe: {results['timeframe']}, Interval: {results['interval']}")
    print("\n")
    
    print("SUPPORT AND RESISTANCE ZONES:")
    print("-" * 50 + "\n")
    
    print("Top Resistance Levels (price: test count):")
    if results["resistance_levels"]:
        for level, tests in results["resistance_levels"].items():
            print(f"  {float(level):.2f}: tested {tests} times")
    else:
        print("  No significant resistance levels detected")
    
    print("\nTop Support Levels (price: test count):")
    if results["support_levels"]:
        for level, tests in results["support_levels"].items():
            print(f"  {float(level):.2f}: tested {tests} times")
    else:
        print("  No significant support levels detected")
    
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS:")
    print("=" * 50)
    metrics = results["metrics"]
    print(f"Initial Capital: ${metrics['initial_capital']:.2f}")
    print(f"Final Capital: ${metrics['final_capital']:.2f}")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print("=" * 50 + "\n")

def main():
    """
    Main function to run the multi-cryptocurrency analysis.
    """
    parser = argparse.ArgumentParser(description='Multi-cryptocurrency analysis with optimized support/resistance detection')
    parser.add_argument('--symbols', nargs='+', default=['BTC-USDT-VANILLA-PERPETUAL', 'SUI-USDT-VANILLA-PERPETUAL'],
                        help='Symbols to analyze')
    parser.add_argument('--timeframes', nargs='+', default=['hours', 'days'],
                        help='Timeframes to use for each symbol (must match symbols length)')
    parser.add_argument('--intervals', nargs='+', default=['1h', '1d'],
                        help='Intervals to use for each symbol (must match symbols length)')
    parser.add_argument('--limits', nargs='+', type=int, default=[2000, 3000],
                        help='Limits to use for each symbol (must match symbols length)')
    parser.add_argument('--output', type=str, default='results/multi_crypto_analysis_results.json',
                        help='Path to save the combined results')
    
    args = parser.parse_args()
    
    if len(args.timeframes) != len(args.symbols) or len(args.intervals) != len(args.symbols) or len(args.limits) != len(args.symbols):
        logger.error("The number of timeframes, intervals, and limits must match the number of symbols")
        return
    
    all_results = {}
    
    for i, symbol in enumerate(args.symbols):
        timeframe = args.timeframes[i]
        interval = args.intervals[i]
        limit = args.limits[i]
        
        use_sui_optimized = 'SUI' in symbol
        
        results = analyze_cryptocurrency(
            symbol=symbol,
            timeframe=timeframe,
            interval=interval,
            limit=limit,
            use_sui_optimized=use_sui_optimized
        )
        
        all_results[symbol] = results
    
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"Combined results saved to {args.output}")
    
    print("\n" + "=" * 50)
    print("MULTI-CRYPTOCURRENCY ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Analyzed {len(args.symbols)} cryptocurrencies: {', '.join(args.symbols)}")
    print(f"Combined results saved to {args.output}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
