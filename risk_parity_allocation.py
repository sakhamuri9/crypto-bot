"""
Risk Parity Allocation

This module implements risk parity allocation for a portfolio of cryptocurrencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
import json
from scipy.optimize import minimize

from coindesk_client import get_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('risk_parity_allocation.log')
    ]
)

logger = logging.getLogger(__name__)

class RiskParityAllocator:
    """
    Risk parity allocator for a portfolio of cryptocurrencies.
    """
    
    def __init__(self, symbols, lookback=90, rebalance_frequency=7):
        """
        Initialize the risk parity allocator.
        
        Args:
            symbols (list): List of symbols to include in the portfolio
            lookback (int): Lookback period for calculating covariance matrix
            rebalance_frequency (int): Rebalance frequency in days
        """
        self.symbols = symbols
        self.lookback = lookback
        self.rebalance_frequency = rebalance_frequency
    
    def get_data(self, interval='1d'):
        """
        Get price data for all symbols.
        
        Args:
            interval (str): Timeframe interval
            
        Returns:
            pandas.DataFrame: DataFrame with price data
        """
        dfs = {}
        for symbol in self.symbols:
            df = get_data(symbol, 'days', interval, limit=self.lookback)
            dfs[symbol] = df['close']
        
        prices_df = pd.DataFrame(dfs)
        
        return prices_df
    
    def calculate_returns(self, prices_df):
        """
        Calculate returns from price data.
        
        Args:
            prices_df (pandas.DataFrame): DataFrame with price data
            
        Returns:
            pandas.DataFrame: DataFrame with returns
        """
        returns_df = prices_df.pct_change().dropna()
        
        return returns_df
    
    def calculate_covariance_matrix(self, returns_df):
        """
        Calculate covariance matrix from returns.
        
        Args:
            returns_df (pandas.DataFrame): DataFrame with returns
            
        Returns:
            pandas.DataFrame: Covariance matrix
        """
        cov_matrix = returns_df.cov() * 252  # Annualize
        
        return cov_matrix
    
    def risk_contribution(self, weights, cov_matrix):
        """
        Calculate risk contribution for each asset.
        
        Args:
            weights (numpy.ndarray): Portfolio weights
            cov_matrix (pandas.DataFrame): Covariance matrix
            
        Returns:
            numpy.ndarray: Risk contribution
        """
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = np.multiply(marginal_contrib, weights) / portfolio_vol
        
        return risk_contrib
    
    def risk_parity_objective(self, weights, cov_matrix):
        """
        Objective function for risk parity optimization.
        
        Args:
            weights (numpy.ndarray): Portfolio weights
            cov_matrix (pandas.DataFrame): Covariance matrix
            
        Returns:
            float: Objective value
        """
        risk_contrib = self.risk_contribution(weights, cov_matrix)
        target_risk = 1.0 / len(weights)
        return np.sum((risk_contrib - target_risk)**2)
    
    def optimize_risk_parity(self, cov_matrix):
        """
        Optimize portfolio weights using risk parity.
        
        Args:
            cov_matrix (pandas.DataFrame): Covariance matrix
            
        Returns:
            numpy.ndarray: Optimized weights
        """
        n = len(self.symbols)
        initial_weights = np.ones(n) / n
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]  # Weights sum to 1
        bounds = tuple((0.01, 0.5) for _ in range(n))  # Min 1%, max 50% per asset
        
        result = minimize(
            self.risk_parity_objective,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return initial_weights
        
        return result.x
    
    def calculate_allocation(self):
        """
        Calculate risk parity allocation.
        
        Returns:
            dict: Allocation for each symbol
        """
        prices_df = self.get_data()
        
        returns_df = self.calculate_returns(prices_df)
        
        cov_matrix = self.calculate_covariance_matrix(returns_df)
        
        weights = self.optimize_risk_parity(cov_matrix)
        
        allocation = {symbol: weight for symbol, weight in zip(self.symbols, weights)}
        
        return allocation
    
    def calculate_portfolio_metrics(self, allocation, returns_df):
        """
        Calculate portfolio metrics.
        
        Args:
            allocation (dict): Allocation for each symbol
            returns_df (pandas.DataFrame): DataFrame with returns
            
        Returns:
            dict: Portfolio metrics
        """
        weights = np.array([allocation[symbol] for symbol in returns_df.columns])
        
        portfolio_returns = returns_df.dot(weights)
        
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak - 1) * 100
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return * 100,
            'annual_volatility': annual_vol * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def plot_allocation(self, allocation):
        """
        Plot allocation.
        
        Args:
            allocation (dict): Allocation for each symbol
        """
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        plt.figure(figsize=(10, 6))
        plt.bar(allocation.keys(), allocation.values())
        plt.title('Risk Parity Allocation')
        plt.ylabel('Weight')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/risk_parity_allocation_{timestamp}.png')
        
        with open(f'{results_dir}/risk_parity_allocation_{timestamp}.json', 'w') as f:
            json.dump(allocation, f, indent=4)
        
        logger.info(f"Allocation saved to {results_dir}")
    
    def run_backtest(self, initial_capital=10000.0):
        """
        Run backtest for risk parity allocation.
        
        Args:
            initial_capital (float): Initial capital
            
        Returns:
            tuple: (final_capital, equity_curve, metrics)
        """
        prices_df = self.get_data()
        
        returns_df = self.calculate_returns(prices_df)
        
        capital = initial_capital
        equity_curve = [capital]
        allocations = []
        rebalance_dates = []
        
        current_allocation = None
        days_since_rebalance = 0
        
        for i in range(1, len(returns_df)):
            if current_allocation is None or days_since_rebalance >= self.rebalance_frequency:
                cov_matrix = returns_df.iloc[:i].cov() * 252
                
                weights = self.optimize_risk_parity(cov_matrix)
                
                current_allocation = {symbol: weight for symbol, weight in zip(self.symbols, weights)}
                
                days_since_rebalance = 0
                
                allocations.append(current_allocation)
                rebalance_dates.append(returns_df.index[i])
            
            daily_returns = returns_df.iloc[i]
            
            portfolio_return = sum([current_allocation[symbol] * daily_returns[symbol] for symbol in self.symbols])
            capital *= (1 + portfolio_return)
            
            equity_curve.append(capital)
            
            days_since_rebalance += 1
        
        metrics = self.calculate_portfolio_metrics(current_allocation, returns_df)
        metrics['final_capital'] = capital
        metrics['total_return'] = (capital / initial_capital - 1) * 100
        
        return capital, equity_curve, metrics, allocations, rebalance_dates
    
    def plot_backtest_results(self, equity_curve, metrics, allocations, rebalance_dates):
        """
        Plot backtest results.
        
        Args:
            equity_curve (list): Equity curve
            metrics (dict): Portfolio metrics
            allocations (list): List of allocations
            rebalance_dates (list): List of rebalance dates
        """
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(equity_curve)
        plt.title('Risk Parity Portfolio Equity Curve')
        plt.ylabel('Capital')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        
        allocation_df = pd.DataFrame(allocations, index=rebalance_dates)
        
        allocation_df.plot.area(stacked=True, ax=plt.gca())
        plt.title('Asset Allocation Over Time')
        plt.ylabel('Weight')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/risk_parity_backtest_{timestamp}.png')
        
        with open(f'{results_dir}/risk_parity_metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Backtest results saved to {results_dir}")

def run_risk_parity_allocation(symbols=['BTC-USDT-VANILLA-PERPETUAL', 'SUI-USDT-VANILLA-PERPETUAL', 'ETH-USDT-VANILLA-PERPETUAL']):
    """
    Run risk parity allocation for a list of symbols.
    
    Args:
        symbols (list): List of symbols to include in the portfolio
        
    Returns:
        tuple: (allocation, metrics)
    """
    allocator = RiskParityAllocator(symbols)
    
    allocation = allocator.calculate_allocation()
    allocator.plot_allocation(allocation)
    
    capital, equity_curve, metrics, allocations, rebalance_dates = allocator.run_backtest()
    allocator.plot_backtest_results(equity_curve, metrics, allocations, rebalance_dates)
    
    logger.info(f"Risk parity allocation complete: {allocation}")
    logger.info(f"Backtest metrics: {metrics}")
    
    return allocation, metrics

if __name__ == "__main__":
    allocation, metrics = run_risk_parity_allocation()
    
    logger.info("Risk parity allocation complete")
