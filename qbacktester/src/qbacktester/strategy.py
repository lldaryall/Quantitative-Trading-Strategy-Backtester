"""
Strategy implementation with signal generation.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from .indicators import sma, crossover


@dataclass
class StrategyParams:
    """Parameters for trading strategy configuration."""
    symbol: str
    start: str
    end: str
    fast_window: int
    slow_window: int
    initial_cash: float = 100_000
    fee_bps: float = 1.0  # round-trip cost in basis points
    slippage_bps: float = 0.5
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.fast_window <= 0:
            raise ValueError("fast_window must be positive")
        if self.slow_window <= 0:
            raise ValueError("slow_window must be positive")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be less than slow_window")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")


def generate_signals(
    price_df: pd.DataFrame, 
    fast_window: int, 
    slow_window: int
) -> pd.DataFrame:
    """
    Generate trading signals using SMA crossover strategy.
    
    Computes fast and slow moving averages, detects crossovers, and generates
    signals with proper timing to avoid look-ahead bias.
    
    Args:
        price_df: DataFrame with price data (must have 'close' column)
        fast_window: Period for fast moving average
        slow_window: Period for slow moving average
        
    Returns:
        DataFrame with additional columns:
        - 'sma_fast': Fast moving average
        - 'sma_slow': Slow moving average  
        - 'crossover': Crossover signals (-1, 0, 1)
        - 'signal': Trading signals (0 flat, 1 long)
        
    Raises:
        ValueError: If required columns are missing or parameters are invalid
    """
    if 'close' not in price_df.columns:
        raise ValueError("price_df must contain 'close' column")
    
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("Windows must be positive")
    
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window")
    
    # Create a copy to avoid modifying original data
    signals_df = price_df.copy()
    
    # Calculate moving averages
    signals_df['sma_fast'] = sma(price_df, fast_window, 'close')
    signals_df['sma_slow'] = sma(price_df, slow_window, 'close')
    
    # Detect crossovers between fast and slow SMA
    signals_df['crossover'] = crossover(signals_df['sma_fast'], signals_df['sma_slow'])
    
    # Generate trading signals
    # 1: Enter long on crossover up (fast crosses above slow)
    # 0: Flat position (no signal or crossover down)
    signals_df['signal'] = 0
    signals_df.loc[signals_df['crossover'] == 1, 'signal'] = 1
    
    # Shift signals by 1 day to avoid look-ahead bias
    # This ensures we only trade on the day AFTER the crossover is detected
    signals_df['signal'] = signals_df['signal'].shift(1)
    
    # First signal will be NaN due to shift, set to 0 (flat)
    signals_df['signal'] = signals_df['signal'].fillna(0)
    
    # Ensure signal is integer type
    signals_df['signal'] = signals_df['signal'].astype(int)
    
    return signals_df


def calculate_position_sizes(
    signals_df: pd.DataFrame,
    initial_cash: float,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5
) -> pd.DataFrame:
    """
    Calculate position sizes and costs based on signals.
    
    Args:
        signals_df: DataFrame with signals (must have 'signal' and 'close' columns)
        initial_cash: Starting cash amount
        fee_bps: Round-trip fee in basis points
        slippage_bps: Slippage cost in basis points
        
    Returns:
        DataFrame with additional columns:
        - 'position': Number of shares held
        - 'cash': Cash remaining
        - 'portfolio_value': Total portfolio value
        - 'trade_cost': Cost of each trade
    """
    if 'signal' not in signals_df.columns or 'close' not in signals_df.columns:
        raise ValueError("signals_df must contain 'signal' and 'close' columns")
    
    # Create a copy to avoid modifying original data
    portfolio_df = signals_df.copy()
    
    # Initialize portfolio tracking
    portfolio_df['position'] = 0
    portfolio_df['cash'] = initial_cash
    portfolio_df['trade_cost'] = 0.0
    
    # Calculate position changes (when signal changes from 0 to 1 or 1 to 0)
    position_changes = portfolio_df['signal'].diff()
    
    # Calculate trade costs (fees + slippage) for actual trades
    total_cost_bps = fee_bps + slippage_bps
    portfolio_df['trade_cost'] = (
        abs(position_changes) * portfolio_df['close'] * total_cost_bps / 10000
    )
    
    # Fill NaN values in trade_cost (first row will be NaN due to diff)
    portfolio_df['trade_cost'] = portfolio_df['trade_cost'].fillna(0)
    
    # Calculate position (just use the signal directly - 0 or 1)
    portfolio_df['position'] = portfolio_df['signal']
    
    # Calculate cash remaining (simplified - just track cash spent on trades)
    # This is a simplified model - in reality you'd track each trade separately
    total_trade_costs = portfolio_df['trade_cost'].cumsum()
    portfolio_df['cash'] = initial_cash - total_trade_costs
    
    # Calculate portfolio value
    portfolio_df['portfolio_value'] = (
        portfolio_df['cash'] + 
        portfolio_df['position'] * portfolio_df['close']
    )
    
    return portfolio_df


def backtest_strategy(
    price_df: pd.DataFrame,
    params: StrategyParams
) -> pd.DataFrame:
    """
    Run a complete backtest for the SMA crossover strategy.
    
    Args:
        price_df: DataFrame with price data
        params: Strategy parameters
        
    Returns:
        DataFrame with signals, positions, and portfolio metrics
    """
    # Generate signals
    signals_df = generate_signals(price_df, params.fast_window, params.slow_window)
    
    # Calculate position sizes and portfolio metrics
    portfolio_df = calculate_position_sizes(
        signals_df, 
        params.initial_cash, 
        params.fee_bps, 
        params.slippage_bps
    )
    
    return portfolio_df


def get_strategy_metrics(portfolio_df: pd.DataFrame) -> dict:
    """
    Calculate key performance metrics for the strategy.
    
    Args:
        portfolio_df: DataFrame with portfolio values
        
    Returns:
        Dictionary with performance metrics
    """
    if 'portfolio_value' not in portfolio_df.columns:
        raise ValueError("portfolio_df must contain 'portfolio_value' column")
    
    # Calculate returns
    returns = portfolio_df['portfolio_value'].pct_change().dropna()
    
    # Basic metrics
    total_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown
    running_max = portfolio_df['portfolio_value'].expanding().max()
    drawdown = (portfolio_df['portfolio_value'] - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Trade statistics (only if signal column exists)
    if 'signal' in portfolio_df.columns:
        signals = portfolio_df['signal']
        trades = signals.diff().abs().sum() / 2  # Each round trip = 2 trades
    else:
        trades = 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': int(trades),
        'final_value': portfolio_df['portfolio_value'].iloc[-1]
    }
