"""
Reinforcement Learning for Optimal Entry/Exit Timing

This module implements a reinforcement learning agent for optimizing trade entry and exit timing
in cryptocurrency trading strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
import json
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

from coindesk_client import get_data
from indicators import add_indicators
from squeeze_momentum import add_squeeze_momentum
from ultimate_macd import add_ultimate_macd
from generic_support_resistance import detect_support_resistance_levels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reinforcement_learning.log')
    ]
)

logger = logging.getLogger(__name__)

class TradingEnvironment:
    """
    Trading environment for reinforcement learning.
    """
    
    def __init__(self, df, initial_balance=10000.0, transaction_fee=0.001):
        """
        Initialize the trading environment.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data and indicators
            initial_balance (float): Initial balance
            transaction_fee (float): Transaction fee as a percentage
        """
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reset()
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            numpy.ndarray: Initial state
        """
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.done = False
        self.trades = []
        self.equity_curve = [self.balance]
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get the current state.
        
        Returns:
            numpy.ndarray: Current state
        """
        current_row = self.df.iloc[self.current_step]
        
        state = [
            current_row['close'] / current_row['open'] - 1,  # Current candle return
            current_row['high'] / current_row['close'] - 1,  # Distance to high
            current_row['close'] / current_row['low'] - 1,  # Distance to low
            
            current_row['rsi'] / 100,  # RSI normalized
            current_row['macd'] / current_row['close'],  # MACD normalized
            current_row['macd_signal'] / current_row['close'],  # MACD signal normalized
            current_row['macd_hist'] / current_row['close'],  # MACD histogram normalized
            
            (current_row['close'] - current_row['lower_band']) / (current_row['upper_band'] - current_row['lower_band']),  # BB position
            (current_row['upper_band'] - current_row['lower_band']) / current_row['middle_band'],  # BB width
            
            current_row['sqz_on'] if 'sqz_on' in current_row else 0,  # Squeeze on
            current_row['sqz_off'] if 'sqz_off' in current_row else 0,  # Squeeze off
            current_row['momentum'] / current_row['close'] if 'momentum' in current_row else 0,  # Momentum normalized
            
            current_row['sma_50'] / current_row['close'] - 1,  # SMA 50 distance
            current_row['sma_200'] / current_row['close'] - 1,  # SMA 200 distance
            
            current_row['atr'] / current_row['close'],  # ATR normalized
            
            current_row['volume'] / current_row['volume'].rolling(20).mean().iloc[self.current_step],  # Volume ratio
            
            1 if self.position > 0 else 0,  # Long position flag
            self.entry_price / current_row['close'] - 1 if self.position > 0 else 0,  # Unrealized return
        ]
        
        return np.array(state)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take (0: hold, 1: buy, 2: sell)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        current_price = self.df['close'].iloc[self.current_step]
        
        reward = 0
        
        if action == 1 and self.position == 0:  # Buy
            position_size = self.balance * 0.95  # Use 95% of balance
            self.position = position_size / current_price
            self.entry_price = current_price
            
            fee = position_size * self.transaction_fee
            self.balance -= fee
            
            self.trades.append({
                'type': 'buy',
                'step': self.current_step,
                'price': current_price,
                'balance': self.balance,
                'position': self.position,
                'fee': fee
            })
            
            reward = -fee / self.initial_balance
            
        elif action == 2 and self.position > 0:  # Sell
            profit_loss = self.position * (current_price - self.entry_price)
            
            fee = self.position * current_price * self.transaction_fee
            profit_loss -= fee
            
            self.balance += self.position * current_price - fee
            
            self.trades.append({
                'type': 'sell',
                'step': self.current_step,
                'price': current_price,
                'balance': self.balance,
                'profit_loss': profit_loss,
                'profit_loss_pct': (current_price / self.entry_price - 1) * 100,
                'fee': fee
            })
            
            self.position = 0
            self.entry_price = 0
            
            reward = profit_loss / self.initial_balance
        
        if self.position > 0:
            equity = self.balance + self.position * current_price
        else:
            equity = self.balance
        
        self.equity_curve.append(equity)
        
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            self.done = True
            
            if self.position > 0:
                current_price = self.df['close'].iloc[self.current_step]
                profit_loss = self.position * (current_price - self.entry_price)
                
                fee = self.position * current_price * self.transaction_fee
                profit_loss -= fee
                
                self.balance += self.position * current_price - fee
                
                self.trades.append({
                    'type': 'sell',
                    'step': self.current_step,
                    'price': current_price,
                    'balance': self.balance,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': (current_price / self.entry_price - 1) * 100,
                    'fee': fee
                })
                
                self.position = 0
                self.entry_price = 0
                
                reward += profit_loss / self.initial_balance
        
        next_state = self._get_state()
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'equity': equity
        }
        
        return next_state, reward, self.done, info
    
    def render(self):
        """
        Render the environment.
        """
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Step')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.savefig(f'{results_dir}/rl_equity_curve_{timestamp}.png')
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.df['close'])
        plt.title('Price and Trades')
        plt.ylabel('Price')
        
        buy_steps = [trade['step'] for trade in self.trades if trade['type'] == 'buy']
        buy_prices = [trade['price'] for trade in self.trades if trade['type'] == 'buy']
        
        sell_steps = [trade['step'] for trade in self.trades if trade['type'] == 'sell']
        sell_prices = [trade['price'] for trade in self.trades if trade['type'] == 'sell']
        
        plt.scatter(buy_steps, buy_prices, marker='^', color='g', s=100)
        plt.scatter(sell_steps, sell_prices, marker='v', color='r', s=100)
        
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Step')
        plt.ylabel('Equity')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/rl_trades_{timestamp}.png')
        
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(f'{results_dir}/rl_trades_{timestamp}.csv', index=False)
        
        metrics = self.calculate_metrics()
        
        with open(f'{results_dir}/rl_metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Rendering complete: {metrics}")
    
    def calculate_metrics(self):
        """
        Calculate performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        final_balance = self.balance
        total_return = (final_balance / self.initial_balance - 1) * 100
        
        sell_trades = [trade for trade in self.trades if trade['type'] == 'sell']
        
        if not sell_trades:
            return {
                'final_balance': final_balance,
                'total_return': total_return,
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }
        
        total_trades = len(sell_trades)
        winning_trades = [trade for trade in sell_trades if trade['profit_loss'] > 0]
        losing_trades = [trade for trade in sell_trades if trade['profit_loss'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_profit = np.mean([trade['profit_loss_pct'] for trade in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([trade['profit_loss_pct'] for trade in losing_trades]) if losing_trades else 0
        
        total_profit = sum([trade['profit_loss'] for trade in winning_trades])
        total_loss = abs(sum([trade['profit_loss'] for trade in losing_trades]))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        peak = self.initial_balance
        drawdowns = []
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns)
        
        return {
            'final_balance': final_balance,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown
        }

class DQNAgent:
    """
    Deep Q-Network agent for reinforcement learning.
    """
    
    def __init__(self, state_size, action_size):
        """
        Initialize the DQN agent.
        
        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """
        Build the neural network model.
        
        Returns:
            tensorflow.keras.models.Sequential: Neural network model
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """
        Update the target model with weights from the main model.
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory.
        
        Args:
            state (numpy.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.ndarray): Next state
            done (bool): Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose an action based on the current state.
        
        Args:
            state (numpy.ndarray): Current state
            training (bool): Whether the agent is training
            
        Returns:
            int: Action to take
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """
        Train the model with experiences from memory.
        
        Args:
            batch_size (int): Batch size for training
        """
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0])
            
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """
        Load model weights.
        
        Args:
            name (str): Model name
        """
        self.model.load_weights(name)
    
    def save(self, name):
        """
        Save model weights.
        
        Args:
            name (str): Model name
        """
        self.model.save_weights(name)

def prepare_data(symbol, interval='1h', lookback=500):
    """
    Prepare data for reinforcement learning.
    
    Args:
        symbol (str): Symbol to get data for
        interval (str): Timeframe interval
        lookback (int): Number of candles to fetch
        
    Returns:
        pandas.DataFrame: Prepared data
    """
    df = get_data(symbol, 'hours', interval, limit=lookback)
    
    df = add_indicators(df)
    df = add_squeeze_momentum(df)
    df = add_ultimate_macd(df)
    
    df = detect_support_resistance_levels(df, symbol, interval)
    
    return df

def train_rl_agent(symbol='BTC-USDT-VANILLA-PERPETUAL', interval='1h', episodes=50, batch_size=32):
    """
    Train a reinforcement learning agent.
    
    Args:
        symbol (str): Symbol to train on
        interval (str): Timeframe interval
        episodes (int): Number of episodes to train for
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (agent, metrics)
    """
    logger.info(f"Training RL agent for {symbol} {interval}")
    
    df = prepare_data(symbol, interval)
    
    env = TradingEnvironment(df)
    
    state = env.reset()
    state_size = len(state)
    
    agent = DQNAgent(state_size, 3)  # 3 actions: hold, buy, sell
    
    results_dir = 'models'
    os.makedirs(results_dir, exist_ok=True)
    
    metrics = {
        'episode_rewards': [],
        'episode_returns': [],
        'episode_trades': []
    }
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while not env.done:
            action = agent.act(state)
            
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            
            total_reward += reward
            
            agent.replay(batch_size)
        
        if episode % 10 == 0:
            agent.update_target_model()
        
        episode_metrics = env.calculate_metrics()
        
        metrics['episode_rewards'].append(total_reward)
        metrics['episode_returns'].append(episode_metrics['total_return'])
        metrics['episode_trades'].append(episode_metrics['total_trades'])
        
        logger.info(f"Episode {episode + 1}/{episodes}: Reward = {total_reward:.4f}, Return = {episode_metrics['total_return']:.2f}%, Trades = {episode_metrics['total_trades']}")
        
        if (episode + 1) % 10 == 0:
            agent.save(f'{results_dir}/rl_agent_{symbol}_{interval}_episode_{episode + 1}.h5')
    
    agent.save(f'{results_dir}/rl_agent_{symbol}_{interval}_final.h5')
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(metrics['episode_returns'])
    plt.title('Episode Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(metrics['episode_trades'])
    plt.title('Episode Trades')
    plt.xlabel('Episode')
    plt.ylabel('Trades')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/rl_training_metrics_{symbol}_{interval}.png')
    
    env.render()
    
    return agent, metrics

def test_rl_agent(agent, symbol='BTC-USDT-VANILLA-PERPETUAL', interval='1h', test_data=None):
    """
    Test a trained reinforcement learning agent.
    
    Args:
        agent (DQNAgent): Trained agent
        symbol (str): Symbol to test on
        interval (str): Timeframe interval
        test_data (pandas.DataFrame): Test data (if None, new data will be fetched)
        
    Returns:
        tuple: (env, metrics)
    """
    logger.info(f"Testing RL agent for {symbol} {interval}")
    
    if test_data is None:
        test_data = prepare_data(symbol, interval)
    
    env = TradingEnvironment(test_data)
    
    state = env.reset()
    
    while not env.done:
        action = agent.act(state, training=False)
        
        next_state, reward, done, info = env.step(action)
        
        state = next_state
    
    env.render()
    
    metrics = env.calculate_metrics()
    
    logger.info(f"Test complete: Return = {metrics['total_return']:.2f}%, Trades = {metrics['total_trades']}, Win Rate = {metrics['win_rate']:.2f}%")
    
    return env, metrics

def compare_rl_with_hedge_fund(symbol='BTC-USDT-VANILLA-PERPETUAL', interval='1h'):
    """
    Compare reinforcement learning agent with hedge fund strategy.
    
    Args:
        symbol (str): Symbol to compare on
        interval (str): Timeframe interval
        
    Returns:
        dict: Comparison metrics
    """
    logger.info(f"Comparing RL agent with hedge fund strategy for {symbol} {interval}")
    
    df = prepare_data(symbol, interval)
    
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    agent, train_metrics = train_rl_agent(symbol, interval, episodes=50, batch_size=32)
    
    rl_env, rl_metrics = test_rl_agent(agent, symbol, interval, test_data)
    
    from hedge_fund_strategy import run_hedge_fund_strategy
    
    hf_metrics = run_hedge_fund_strategy(symbol, interval, test_data)
    
    comparison = {
        'rl_metrics': rl_metrics,
        'hf_metrics': hf_metrics,
        'comparison': {
            'total_return_diff': rl_metrics['total_return'] - hf_metrics['total_return'],
            'win_rate_diff': rl_metrics['win_rate'] - hf_metrics['win_rate'],
            'max_drawdown_diff': rl_metrics['max_drawdown'] - hf_metrics['max_drawdown'],
            'profit_factor_diff': rl_metrics['profit_factor'] - hf_metrics['profit_factor']
        }
    }
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.bar(['RL Agent', 'Hedge Fund'], [rl_metrics['total_return'], hf_metrics['total_return']])
    plt.title('Total Return (%)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.bar(['RL Agent', 'Hedge Fund'], [rl_metrics['win_rate'], hf_metrics['win_rate']])
    plt.title('Win Rate (%)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.bar(['RL Agent', 'Hedge Fund'], [rl_metrics['max_drawdown'], hf_metrics['max_drawdown']])
    plt.title('Max Drawdown (%)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.bar(['RL Agent', 'Hedge Fund'], [rl_metrics['profit_factor'], hf_metrics['profit_factor']])
    plt.title('Profit Factor')
    plt.grid(True)
    
    plt.tight_layout()
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plt.savefig(f'{results_dir}/rl_vs_hf_comparison_{symbol}_{interval}_{timestamp}.png')
    
    with open(f'{results_dir}/rl_vs_hf_comparison_{symbol}_{interval}_{timestamp}.json', 'w') as f:
        json.dump(comparison, f, indent=4)
    
    logger.info(f"Comparison complete: RL Return = {rl_metrics['total_return']:.2f}%, HF Return = {hf_metrics['total_return']:.2f}%")
    
    return comparison

if __name__ == "__main__":
    btc_comparison = compare_rl_with_hedge_fund('BTC-USDT-VANILLA-PERPETUAL', '1h')
    sui_comparison = compare_rl_with_hedge_fund('SUI-USDT-VANILLA-PERPETUAL', '1h')
    
    logger.info("Reinforcement learning comparison complete")
