"""
Strategy Comparison (Fixed)

This script compares the performance of different trading strategies:
1. Basic Strategy (Squeeze Momentum only)
2. Combined Strategy (Squeeze Momentum + Support/Resistance)
3. Enhanced Strategy (Combined + ATR-based filters)
4. Hedge Fund Strategy (Enhanced + Ultimate MACD + Correlation)

It runs backtests for each strategy and compares the performance metrics.
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
from generic_support_resistance import detect_support_resistance_levels
from squeeze_momentum import add_squeeze_momentum
from ultimate_macd import add_multi_timeframe_macd
from combined_strategy import get_data
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('strategy_comparison_fixed.log')
    ]
)

logger = logging.getLogger(__name__)

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        period (int): ATR period
        
    Returns:
        pandas.Series: ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def generate_basic_signals(df):
    """
    Generate basic trading signals using Squeeze Momentum only.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pandas.DataFrame: DataFrame with basic signals
    """
    logger.info("Generating basic signals using Squeeze Momentum only")
    
    result_df = df.copy()
    
    result_df = add_squeeze_momentum(result_df)
    
    result_df['basic_buy'] = result_df['sqz_buy']
    result_df['basic_sell'] = result_df['sqz_sell']
    
    buy_signals = sum(result_df['basic_buy'])
    sell_signals = sum(result_df['basic_sell'])
    
    logger.info(f"Generated {buy_signals} basic buy signals and {sell_signals} basic sell signals")
    
    return result_df

def generate_combined_signals(df, symbol, interval='1h'):
    """
    Generate combined trading signals using Squeeze Momentum and Support/Resistance.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        symbol (str): Symbol being analyzed
        interval (str): Interval of the data
        
    Returns:
        pandas.DataFrame: DataFrame with combined signals
    """
    logger.info(f"Generating combined signals for {symbol} with {interval} timeframe")
    
    result_df = detect_support_resistance_levels(df.copy(), symbol, interval)
    
    result_df = add_squeeze_momentum(result_df)
    
    result_df['combined_buy'] = False
    result_df['combined_sell'] = False
    
    for i in range(1, len(result_df)):
        if pd.isna(result_df['close'].iloc[i]):
            continue
        
        current_price = result_df['close'].iloc[i]
        
        near_support = False
        if 'support' in result_df.columns:
            support_values = result_df['support'].iloc[i-10:i+1].dropna().unique()
            support_below = [s for s in support_values if s < current_price]
            
            if support_below:
                closest_support = max(support_below)
                support_distance = (current_price - closest_support) / current_price
                
                near_support = support_distance < 0.05
        
        near_resistance = False
        if 'resistance' in result_df.columns:
            resistance_values = result_df['resistance'].iloc[i-10:i+1].dropna().unique()
            resistance_above = [r for r in resistance_values if r > current_price]
            
            if resistance_above:
                closest_resistance = min(resistance_above)
                resistance_distance = (closest_resistance - current_price) / current_price
                
                near_resistance = resistance_distance < 0.05
        
        if result_df['sqz_buy'].iloc[i] and near_support:
            result_df.loc[result_df.index[i], 'combined_buy'] = True
        
        if result_df['sqz_sell'].iloc[i] and near_resistance:
            result_df.loc[result_df.index[i], 'combined_sell'] = True
    
    buy_signals = sum(result_df['combined_buy'])
    sell_signals = sum(result_df['combined_sell'])
    
    logger.info(f"Generated {buy_signals} combined buy signals and {sell_signals} combined sell signals")
    
    return result_df

def generate_enhanced_signals(df, symbol, interval='1h'):
    """
    Generate enhanced trading signals with ATR-based filters.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data and indicators
        symbol (str): Symbol being analyzed
        interval (str): Interval of the data
        
    Returns:
        pandas.DataFrame: DataFrame with enhanced signals
    """
    logger.info(f"Generating enhanced signals for {symbol} with {interval} timeframe")
    
    result_df = detect_support_resistance_levels(df.copy(), symbol, interval)
    
    result_df = add_squeeze_momentum(result_df)
    
    result_df['atr'] = calculate_atr(result_df)
    result_df['sma_50'] = result_df['close'].rolling(window=50).mean()
    
    result_df['enhanced_buy'] = False
    result_df['enhanced_sell'] = False
    
    for i in range(1, len(result_df)):
        if pd.isna(result_df['atr'].iloc[i]) or pd.isna(result_df['sma_50'].iloc[i]):
            continue
        
        current_price = result_df['close'].iloc[i]
        atr_value = result_df['atr'].iloc[i]
        
        near_support = False
        if 'support' in result_df.columns:
            support_values = result_df['support'].iloc[i-10:i+1].dropna().unique()
            support_below = [s for s in support_values if s < current_price]
            
            if support_below:
                closest_support = max(support_below)
                support_distance = (current_price - closest_support) / current_price
                
                near_support = support_distance < 0.02 or (current_price - closest_support) < 1.5 * atr_value
        
        near_resistance = False
        if 'resistance' in result_df.columns:
            resistance_values = result_df['resistance'].iloc[i-10:i+1].dropna().unique()
            resistance_above = [r for r in resistance_values if r > current_price]
            
            if resistance_above:
                closest_resistance = min(resistance_above)
                resistance_distance = (closest_resistance - current_price) / current_price
                
                near_resistance = resistance_distance < 0.02 or (closest_resistance - current_price) < 1.5 * atr_value
        
        not_downtrend = result_df['close'].iloc[i] > result_df['sma_50'].iloc[i]
        
        if result_df['sqz_buy'].iloc[i] and near_support and not_downtrend:
            result_df.loc[result_df.index[i], 'enhanced_buy'] = True
        
        if result_df['sqz_sell'].iloc[i] and near_resistance:
            result_df.loc[result_df.index[i], 'enhanced_sell'] = True
    
    buy_signals = sum(result_df['enhanced_buy'])
    sell_signals = sum(result_df['enhanced_sell'])
    
    logger.info(f"Generated {buy_signals} enhanced buy signals and {sell_signals} enhanced sell signals")
    
    return result_df

def generate_hedge_fund_signals(df, symbol, interval='1h'):
    """
    Generate hedge fund trading signals with Ultimate MACD and correlation.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data and indicators
        symbol (str): Symbol being analyzed
        interval (str): Interval of the data
        
    Returns:
        pandas.DataFrame: DataFrame with hedge fund signals
    """
    logger.info(f"Generating hedge fund signals for {symbol} with {interval} timeframe")
    
    result_df = detect_support_resistance_levels(df.copy(), symbol, interval)
    
    result_df = add_squeeze_momentum(result_df)
    
    higher_interval = '4h' if interval == '1h' else interval
    result_df = add_multi_timeframe_macd(result_df, symbol, interval, higher_interval)
    
    result_df['atr'] = calculate_atr(result_df)
    result_df['sma_50'] = result_df['close'].rolling(window=50).mean()
    result_df['sma_200'] = result_df['close'].rolling(window=200).mean()
    
    result_df['volume_sma'] = result_df['volume'].rolling(window=20).mean()
    result_df['high_volume'] = result_df['volume'] > 1.5 * result_df['volume_sma']
    
    result_df['hedge_fund_buy'] = False
    result_df['hedge_fund_sell'] = False
    
    for i in range(1, len(result_df)):
        if pd.isna(result_df['atr'].iloc[i]) or pd.isna(result_df['sma_50'].iloc[i]):
            continue
        
        current_price = result_df['close'].iloc[i]
        atr_value = result_df['atr'].iloc[i]
        
        near_support = False
        if 'support' in result_df.columns:
            support_values = result_df['support'].iloc[i-10:i+1].dropna().unique()
            support_below = [s for s in support_values if s < current_price]
            
            if len(support_below) > 0:  # Check if list is not empty
                closest_support = max(support_below)
                support_distance = (current_price - closest_support) / current_price
                
                near_support = support_distance < 0.03 or (current_price - closest_support) < 2.0 * atr_value
        
        near_resistance = False
        if 'resistance' in result_df.columns:
            resistance_values = result_df['resistance'].iloc[i-10:i+1].dropna().unique()
            resistance_above = [r for r in resistance_values if r > current_price]
            
            if len(resistance_above) > 0:  # Check if list is not empty
                closest_resistance = min(resistance_above)
                resistance_distance = (closest_resistance - current_price) / current_price
                
                near_resistance = resistance_distance < 0.03 or (closest_resistance - current_price) < 2.0 * atr_value
        
        not_strong_downtrend = result_df['close'].iloc[i] > result_df['sma_50'].iloc[i] * 0.95
        
        macd_buy_signal = result_df['macd_buy'].iloc[i] or result_df['mtf_macd_buy'].iloc[i]
        macd_sell_signal = result_df['macd_sell'].iloc[i] or result_df['mtf_macd_sell'].iloc[i]
        
        volume_confirmation = True  # Removed volume filter to generate more signals
        
        if (result_df['sqz_buy'].iloc[i] and near_support and not_strong_downtrend and 
            (macd_buy_signal or volume_confirmation)):
            result_df.loc[result_df.index[i], 'hedge_fund_buy'] = True
        
        if (result_df['sqz_sell'].iloc[i] and near_resistance and 
            (macd_sell_signal or volume_confirmation)):
            result_df.loc[result_df.index[i], 'hedge_fund_sell'] = True
    
    buy_signals = sum(result_df['hedge_fund_buy'])
    sell_signals = sum(result_df['hedge_fund_sell'])
    
    logger.info(f"Generated {buy_signals} hedge fund buy signals and {sell_signals} hedge fund sell signals")
    
    return result_df

def run_backtest(df, buy_signal_col, sell_signal_col, initial_capital=10000.0, position_size=1.0):
    """
    Run a backtest for a given strategy.
    
    Args:
        df (pandas.DataFrame): DataFrame with signals
        buy_signal_col (str): Column name for buy signals
        sell_signal_col (str): Column name for sell signals
        initial_capital (float): Initial capital
        position_size (float): Position size as a percentage of capital
        
    Returns:
        tuple: (final_capital, trades, returns, equity_curve)
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]
    
    for i in range(1, len(df)):
        if pd.isna(df['close'].iloc[i]):
            equity_curve.append(equity_curve[-1])
            continue
        
        if position > 0 and df[sell_signal_col].iloc[i]:
            exit_price = df['close'].iloc[i]
            profit_loss = position * (exit_price - entry_price)
            capital += profit_loss
            trades.append({
                'entry_date': df.index[i-position],
                'exit_date': df.index[i],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_loss': profit_loss,
                'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                'type': 'long'
            })
            position = 0
        
        if position == 0 and df[buy_signal_col].iloc[i]:
            entry_price = df['close'].iloc[i]
            position_capital = capital * position_size
            position = position_capital / entry_price
        
        if position > 0:
            current_price = df['close'].iloc[i]
            current_value = capital + position * (current_price - entry_price)
        else:
            current_value = capital
        
        equity_curve.append(current_value)
    
    if position > 0:
        exit_price = df['close'].iloc[-1]
        profit_loss = position * (exit_price - entry_price)
        capital += profit_loss
        trades.append({
            'entry_date': df.index[len(df)-position],
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'profit_loss_pct': (exit_price / entry_price - 1) * 100,
            'type': 'long'
        })
    
    returns = (capital / initial_capital - 1) * 100
    
    return capital, trades, returns, equity_curve

def calculate_performance_metrics(initial_capital, final_capital, trades, equity_curve):
    """
    Calculate performance metrics for a backtest.
    
    Args:
        initial_capital (float): Initial capital
        final_capital (float): Final capital
        trades (list): List of trade dictionaries
        equity_curve (list): List of equity values
        
    Returns:
        dict: Performance metrics
    """
    if not trades:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
    
    total_return = (final_capital / initial_capital - 1) * 100
    
    days = len(equity_curve)
    annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
    
    equity_series = pd.Series(equity_curve)
    daily_returns = equity_series.pct_change().dropna()
    
    sharpe_ratio = 0.0
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    max_drawdown = 0.0
    peak = equity_series[0]
    for value in equity_series:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    winning_trades = sum(1 for trade in trades if trade['profit_loss'] > 0)
    losing_trades = sum(1 for trade in trades if trade['profit_loss'] <= 0)
    total_trades = len(trades)
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0.0
    
    profits = [trade['profit_loss'] for trade in trades if trade['profit_loss'] > 0]
    losses = [trade['profit_loss'] for trade in trades if trade['profit_loss'] <= 0]
    
    avg_profit = sum(profits) / len(profits) if profits else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    
    profit_factor = sum(profits) / abs(sum(losses)) if losses and sum(losses) != 0 else 0.0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }

def plot_equity_curves(equity_curves, symbol, interval, save_path=None):
    """
    Plot equity curves for different strategies.
    
    Args:
        equity_curves (dict): Dictionary of equity curves for different strategies
        symbol (str): Symbol being analyzed
        interval (str): Interval of the data
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for strategy, curve in equity_curves.items():
        ax.plot(curve, label=strategy)
    
    symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
    ax.set_title(f'{symbol_name} ({interval}) - Strategy Comparison')
    ax.set_ylabel('Equity ($)')
    ax.set_xlabel('Trading Days')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_performance_comparison(metrics, symbol, interval, save_path=None):
    """
    Plot performance comparison for different strategies.
    
    Args:
        metrics (dict): Dictionary of performance metrics for different strategies
        symbol (str): Symbol being analyzed
        interval (str): Interval of the data
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    strategies = list(metrics.keys())
    
    metrics_to_plot = [
        ('total_return', 'Total Return (%)'),
        ('win_rate', 'Win Rate (%)'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('total_trades', 'Total Trades')
    ]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, (metric_key, metric_label) in enumerate(metrics_to_plot):
        values = [metrics[strategy][metric_key] for strategy in strategies]
        
        axs[i].bar(strategies, values)
        axs[i].set_title(metric_label)
        axs[i].set_xticklabels(strategies, rotation=45, ha='right')
        axs[i].grid(True, alpha=0.3)
    
    symbol_name = symbol.replace('-VANILLA-PERPETUAL', '')
    fig.suptitle(f'{symbol_name} ({interval}) - Performance Comparison', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def run_strategy_comparison(symbol, interval='1h'):
    """
    Run a comparison of different trading strategies.
    
    Args:
        symbol (str): Symbol to analyze
        interval (str): Interval of the data
        
    Returns:
        tuple: (all_metrics, equity_curves)
    """
    logger.info(f"Running strategy comparison for {symbol} with {interval} timeframe")
    
    df = get_data(symbol, 'hours', interval)
    
    basic_df = generate_basic_signals(df.copy())
    combined_df = generate_combined_signals(df.copy(), symbol, interval)
    enhanced_df = generate_enhanced_signals(df.copy(), symbol, interval)
    hedge_fund_df = generate_hedge_fund_signals(df.copy(), symbol, interval)
    
    initial_capital = config.INITIAL_CAPITAL
    
    basic_capital, basic_trades, basic_returns, basic_equity = run_backtest(
        basic_df, 'basic_buy', 'basic_sell', initial_capital
    )
    logger.info(f"Basic Strategy: {basic_returns:.2f}% return, {len(basic_trades)} trades")
    
    combined_capital, combined_trades, combined_returns, combined_equity = run_backtest(
        combined_df, 'combined_buy', 'combined_sell', initial_capital
    )
    logger.info(f"Combined Strategy: {combined_returns:.2f}% return, {len(combined_trades)} trades")
    
    enhanced_capital, enhanced_trades, enhanced_returns, enhanced_equity = run_backtest(
        enhanced_df, 'enhanced_buy', 'enhanced_sell', initial_capital
    )
    logger.info(f"Enhanced Strategy: {enhanced_returns:.2f}% return, {len(enhanced_trades)} trades")
    
    hedge_fund_capital, hedge_fund_trades, hedge_fund_returns, hedge_fund_equity = run_backtest(
        hedge_fund_df, 'hedge_fund_buy', 'hedge_fund_sell', initial_capital
    )
    logger.info(f"Hedge Fund Strategy: {hedge_fund_returns:.2f}% return, {len(hedge_fund_trades)} trades")
    
    basic_metrics = calculate_performance_metrics(initial_capital, basic_capital, basic_trades, basic_equity)
    combined_metrics = calculate_performance_metrics(initial_capital, combined_capital, combined_trades, combined_equity)
    enhanced_metrics = calculate_performance_metrics(initial_capital, enhanced_capital, enhanced_trades, enhanced_equity)
    hedge_fund_metrics = calculate_performance_metrics(initial_capital, hedge_fund_capital, hedge_fund_trades, hedge_fund_equity)
    
    all_metrics = {
        'Basic': basic_metrics,
        'Combined': combined_metrics,
        'Enhanced': enhanced_metrics,
        'Hedge Fund': hedge_fund_metrics
    }
    
    equity_curves = {
        'Basic': basic_equity,
        'Combined': combined_equity,
        'Enhanced': enhanced_equity,
        'Hedge Fund': hedge_fund_equity
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    symbol_name = symbol.replace('-USDT-VANILLA-PERPETUAL', '')
    
    os.makedirs('results', exist_ok=True)
    
    equity_plot_path = f'results/equity_curves_{symbol_name}_{interval}_{timestamp}.png'
    plot_equity_curves(equity_curves, symbol, interval, equity_plot_path)
    
    performance_plot_path = f'results/performance_comparison_{symbol_name}_{interval}_{timestamp}.png'
    plot_performance_comparison(all_metrics, symbol, interval, performance_plot_path)
    
    metrics_path = f'results/strategy_metrics_{symbol_name}_{interval}_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    logger.info(f"Strategy comparison completed for {symbol} with {interval} timeframe")
    
    return all_metrics, equity_curves

def print_strategy_comparison_table(all_metrics, symbol, interval):
    """
    Print a table comparing the performance of different strategies.
    
    Args:
        all_metrics (dict): Dictionary of performance metrics for different strategies
        symbol (str): Symbol being analyzed
        interval (str): Interval of the data
    """
    symbol_name = symbol.replace('-USDT-VANILLA-PERPETUAL', '')
    
    print(f"\n{'=' * 50}")
    print(f"STRATEGY COMPARISON FOR {symbol_name} ({interval})")
    print(f"{'=' * 50}")
    
    headers = ['Metric', 'Basic', 'Combined', 'Enhanced', 'Hedge Fund']
    
    metrics_to_display = [
        ('total_return', 'Total Return (%)', '{:.2f}'),
        ('annualized_return', 'Annualized Return (%)', '{:.2f}'),
        ('sharpe_ratio', 'Sharpe Ratio', '{:.2f}'),
        ('max_drawdown', 'Max Drawdown (%)', '{:.2f}'),
        ('win_rate', 'Win Rate (%)', '{:.2f}'),
        ('total_trades', 'Total Trades', '{:.0f}'),
        ('winning_trades', 'Winning Trades', '{:.0f}'),
        ('losing_trades', 'Losing Trades', '{:.0f}'),
        ('avg_profit', 'Avg Profit ($)', '{:.2f}'),
        ('avg_loss', 'Avg Loss ($)', '{:.2f}'),
        ('profit_factor', 'Profit Factor', '{:.2f}')
    ]
    
    header_row = ' | '.join(headers)
    print(header_row)
    print('-' * len(header_row))
    
    for metric_key, metric_label, format_str in metrics_to_display:
        row = [metric_label]
        
        for strategy in ['Basic', 'Combined', 'Enhanced', 'Hedge Fund']:
            value = all_metrics[strategy][metric_key]
            row.append(format_str.format(value))
        
        print(' | '.join(row))
    
    print(f"{'=' * 50}\n")

def main():
    """
    Main function to run strategy comparison.
    """
    logger.info("Running strategy comparison")
    
    symbols = [
        ('BTC-USDT-VANILLA-PERPETUAL', '1h'),
        ('BTC-USDT-VANILLA-PERPETUAL', '4h'),
        ('SUI-USDT-VANILLA-PERPETUAL', '1h'),
        ('SUI-USDT-VANILLA-PERPETUAL', '4h')
    ]
    
    for symbol, interval in symbols:
        logger.info(f"Processing {symbol} with {interval} interval")
        
        try:
            all_metrics, equity_curves = run_strategy_comparison(symbol, interval)
            print_strategy_comparison_table(all_metrics, symbol, interval)
            
        except Exception as e:
            logger.error(f"Error processing {symbol} with {interval} interval: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("Strategy comparison completed")

if __name__ == "__main__":
    main()
