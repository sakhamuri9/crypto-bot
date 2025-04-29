# Advanced Crypto Trading Bot

This is a Python-based trading bot that generates buy and sell signals for cryptocurrency trading using data from Binance Sandbox or Coinbase Advanced API. The bot implements a complex trading strategy combining multiple technical indicators and includes risk management features.

## Features

- Connects to Binance Sandbox API or Coinbase Advanced API to fetch historical price data
- Implements an advanced trading strategy using multiple technical indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - EMA (Exponential Moving Average) crossovers
  - Stochastic Oscillator
  - On-Balance Volume
  - Ichimoku Cloud
- Generates clear buy and sell signals with timestamps
- Includes risk management features:
  - Position sizing
  - Stop-loss and take-profit levels
  - Maximum open trades limit
- Backtesting capabilities with performance metrics:
  - Total and annualized returns
  - Sharpe, Sortino, and Calmar ratios
  - Maximum drawdown
  - Win rate and profit factor
  - Trade statistics
- Visualization of backtest results with interactive charts

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file with your API credentials:
```
cp .env.example .env
```
Then edit the `.env` file and add your API credentials:
- For Binance: Add your Binance Sandbox API key and secret
- For Coinbase: Add your Coinbase Advanced API key and private key

## Usage

### Generating Sample Data

Before running a backtest, you can generate sample historical data:

```
python generate_sample_data.py
```

This will create a CSV file with 180 days of simulated BTC/USDT price data in the `data` directory.

### Backtesting

#### Using Sample Data

To run a backtest with the generated sample data:

```
python run_backtest.py
```

This will run a backtest on the sample BTC/USDT data with an initial capital of $10,000.

You can customize the backtest parameters:

```
python run_backtest.py --data-file data/btcusdt_1h_sample.csv --symbol BTCUSDT --initial-capital 5000
```

#### Using Binance API Data

To run a backtest with data from Binance API:

```
python main.py --mode backtest
```

This will run a backtest on BTC/USDT for the last 6 months with 1-hour candles and an initial capital of $10,000.

You can customize the backtest parameters:

```
python main.py --mode backtest --symbol ETHUSDT --timeframe 4h --start-date 2023-01-01 --end-date 2023-12-31 --initial-capital 5000
```

### Live Trading

To run the bot in live trading mode:

```
python main.py --mode live --exchange coinbase --symbol BTC-USD
```

This will connect to the specified exchange API, fetch the latest data, generate trading signals, and execute trades based on those signals.

You can customize the live trading parameters:

```
python main.py --mode live --exchange coinbase --symbol ETH-USD --timeframe 15m --interval 30 --risk-per-trade 0.01 --stop-loss 0.02 --take-profit 0.04
```

Parameters:
- `--exchange`: Exchange to use (binance or coinbase)
- `--symbol`: Trading pair symbol (default: BTCUSDT for Binance, BTC-USD for Coinbase)
- `--timeframe`: Candle interval (default: 1h)
- `--interval`: Seconds between checks (default: 60)
- `--runtime`: Maximum runtime in seconds (default: None, run indefinitely)
- `--risk-per-trade`: Percentage of account balance to risk per trade (default: 0.02)
- `--stop-loss`: Stop loss percentage (default: 0.02)
- `--take-profit`: Take profit percentage (default: 0.04)
- `--test-mode`: Run in test mode without executing actual trades

### Testing Live Trading

For testing the live trading functionality without executing actual trades:

```
python test_live_trading.py --exchange coinbase --test-mode
```

This will run the live trader in test mode, which simulates trade execution without actually placing orders. It's useful for verifying that your strategy generates the expected signals and would execute trades correctly.

You can customize the test parameters:

```
python test_live_trading.py --exchange coinbase --symbol ETH-USD --timeframe 15m --interval 30 --runtime 300 --test-mode
```

The `--runtime` parameter specifies how long the test should run in seconds (default: 300 seconds).

### Mock Client for Testing

The bot includes a mock client implementation for testing without API authentication:

```
python main.py --mode live --exchange coinbase --symbol BTC-USD --test-mode
```

When running in test mode, the bot uses a mock client that:
- Generates realistic price data with built-in volatility
- Simulates account balances and order execution
- Calculates all technical indicators
- Provides a complete testing environment without requiring API credentials

This is useful for:
- Developing and testing trading strategies without API access
- Debugging the trading logic without making real API calls
- Testing the bot's behavior in different market conditions
- Verifying signal generation and trade execution logic

The mock client generates a full year of historical data to ensure all indicators can be properly calculated.

### Output

The backtest will generate:
- CSV file with detailed backtest results
- CSV file with performance metrics
- PNG image with backtest visualization
- Console output with performance summary and trade signals

All output files are saved in the `results` directory by default.

## Configuration

You can modify the trading parameters in the `config.py` file:

- Trading pair symbol and timeframe
- Technical indicator parameters
- Risk management settings (position size, stop-loss, take-profit)

## Project Structure

- `main.py`: Entry point for the application
- `binance_client.py`: Client for interacting with Binance API
- `coinbase_client.py`: Client for interacting with Coinbase Advanced API
- `indicators.py`: Technical indicators implementation
- `strategy.py`: Trading strategy implementation
- `backtester.py`: Backtesting module
- `live_trader.py`: Live trading implementation
- `test_live_trading.py`: Script to test live trading functionality
- `config.py`: Configuration settings
- `generate_sample_data.py`: Script to generate sample historical data
- `run_backtest.py`: Script to run backtest with sample data
- `data/`: Directory containing sample historical data
- `results/`: Directory containing backtest results

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational purposes only. Use it at your own risk. Cryptocurrency trading involves substantial risk and is not suitable for everyone. The author is not responsible for any financial losses incurred from using this software.
