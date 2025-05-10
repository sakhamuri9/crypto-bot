"""
Script to analyze SUI-USDT data from CoinDesk API using the enhanced support/resistance detection.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

from coindesk_client import CoinDeskClient
from indicators import add_indicators
from strategy import TradingStrategy
from backtester import Backtester
from support_resistance import calculate_dynamic_support_resistance, detect_sr_breakouts
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('coindesk_sui_analysis.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the analysis pipeline.
    """
    logger.info("Starting SUI-USDT analysis pipeline")
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    logger.info("Collecting data from CoinDesk for SUI-USDT-VANILLA-PERPETUAL with 1d timeframe")
    
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
    raw_data_path = f"data/coindesk_raw_SUI_USDT_VANILLA_PERPETUAL_{timestamp}.csv"
    df.to_csv(raw_data_path)
    logger.info(f"Raw data saved to {raw_data_path}")
    
    logger.info("Processing data and adding technical indicators")
    processed_df = add_indicators(df)
    logger.info(f"Processed {len(processed_df)} rows of data")
    
    logger.info("Analyzing support and resistance zones")
    
    processed_data_path = f"results/processed_SUI_USDT_VANILLA_PERPETUAL_{timestamp}.csv"
    processed_df.to_csv(processed_data_path)
    logger.info(f"Processed data saved to {processed_data_path}")
    
    logger.info("Plotting support and resistance zones")
    plot_support_resistance(processed_df, f"results/support_resistance_SUI_USDT_VANILLA_PERPETUAL_{timestamp}.png")
    
    strategy = TradingStrategy()
    market_regime = strategy.detect_market_regime(processed_df)
    logger.info(f"Detected market regime: {market_regime}")
    
    strategy.train_ml_model(processed_df, prediction_window=10)
    
    signals_df = strategy.generate_signals(processed_df)
    
    backtester = Backtester(
        initial_capital=config.INITIAL_CAPITAL,
        position_size=config.POSITION_SIZE,
        stop_loss_pct=config.STOP_LOSS_PCT,
        take_profit_pct=config.TAKE_PROFIT_PCT
    )
    
    backtest_results = backtester.run_backtest(signals_df)
    
    backtest_results.to_csv(f"results/backtest_results_SUI_USDT_VANILLA_PERPETUAL_{timestamp}.csv")
    logger.info(f"Backtest results saved to results/backtest_results_SUI_USDT_VANILLA_PERPETUAL_{timestamp}.csv")
    
    metrics = backtester.calculate_metrics()
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"results/backtest_metrics_SUI_USDT_VANILLA_PERPETUAL_{timestamp}.csv")
    logger.info(f"Backtest metrics saved to results/backtest_metrics_SUI_USDT_VANILLA_PERPETUAL_{timestamp}.csv")
    
    backtester.plot_results(f"results/backtest_plot_SUI_USDT_VANILLA_PERPETUAL_{timestamp}.png")
    
    print("\n" + "=" * 50)
    print(f"ANALYSIS RESULTS FOR SUI-USDT-VANILLA-PERPETUAL")
    print("=" * 50 + "\n")
    
    print("SUPPORT AND RESISTANCE ZONES:")
    print("-" * 50 + "\n")
    
    print("Top Resistance Levels (price: test count):")
    if 'resistance' in processed_df.columns and 'resistance_tests' in processed_df.columns:
        resistance_df = processed_df[['resistance', 'resistance_tests']].drop_duplicates().dropna()
        if not resistance_df.empty:
            resistance_df = resistance_df.sort_values('resistance_tests', ascending=False).head(5)
            for _, row in resistance_df.iterrows():
                print(f"  {row['resistance']:.2f}: tested {int(row['resistance_tests'])} times")
        else:
            print("  No significant resistance levels detected")
    else:
        print("  Resistance data not available")
    
    print("\nTop Support Levels (price: test count):")
    if 'support' in processed_df.columns and 'support_tests' in processed_df.columns:
        support_df = processed_df[['support', 'support_tests']].drop_duplicates().dropna()
        if not support_df.empty:
            support_df = support_df.sort_values('support_tests', ascending=False).head(5)
            for _, row in support_df.iterrows():
                print(f"  {row['support']:.2f}: tested {int(row['support_tests'])} times")
        else:
            print("  No significant support levels detected")
    else:
        print("  Support data not available")
    
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS:")
    print("=" * 50)
    print(f"Initial Capital: ${metrics['initial_capital']:.2f}")
    print(f"Final Capital: ${metrics['final_capital']:.2f}")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print("=" * 50 + "\n")

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
    
    plt.title('SUI-USDT Price with Support and Resistance Levels')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()
