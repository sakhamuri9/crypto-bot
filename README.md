# Advanced Crypto Trading Bot

This is a Python-based trading bot that generates buy and sell signals for cryptocurrency trading using data from Binance Sandbox. The bot implements a complex trading strategy combining multiple technical indicators and includes risk management features.

## Features

- Connects to Binance Sandbox API to fetch historical price data
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

3. Create a `.env` file with your Binance API credentials:
```
cp .env.example .env
```
Then edit the `.env` file and add your Binance Sandbox API key and secret.

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
- `indicators.py`: Technical indicators implementation
- `strategy.py`: Trading strategy implementation
- `backtester.py`: Backtesting module
- `config.py`: Configuration settings
- `generate_sample_data.py`: Script to generate sample historical data
- `run_backtest.py`: Script to run backtest with sample data
- `data/`: Directory containing sample historical data
- `results/`: Directory containing backtest results

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational purposes only. Use it at your own risk. Cryptocurrency trading involves substantial risk and is not suitable for everyone. The author is not responsible for any financial losses incurred from using this software.
