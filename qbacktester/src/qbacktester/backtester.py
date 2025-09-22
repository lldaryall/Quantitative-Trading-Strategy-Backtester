"""
Vectorized backtesting engine for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Optional
from .strategy import StrategyParams


class Backtester:
    """
    Vectorized backtesting engine for trading strategies.
    
    Implements a long-only, fully invested strategy with transaction costs.
    All calculations are vectorized for performance.
    """
    
    def __init__(
        self, 
        price_df: pd.DataFrame, 
        signals: pd.Series, 
        params: StrategyParams
    ):
        """
        Initialize the backtester.
        
        Args:
            price_df: DataFrame with price data (must have 'close' and optionally 'open')
            signals: Series with trading signals (0 or 1)
            params: Strategy parameters including costs and initial cash
        """
        if 'close' not in price_df.columns:
            raise ValueError("price_df must contain 'close' column")
        
        if not signals.index.equals(price_df.index):
            raise ValueError("signals and price_df must have the same index")
        
        self.price_df = price_df.copy()
        self.signals = signals.copy()
        self.params = params
        
        # Ensure signals are 0 or 1
        if not signals.isin([0, 1]).all():
            raise ValueError("signals must contain only 0 or 1 values")
        
        # Use open prices if available, otherwise use close prices
        self.trade_prices = price_df.get('open', price_df['close'])
        
        # Calculate total transaction cost in basis points
        self.total_cost_bps = params.fee_bps + params.slippage_bps
    
    def run(self) -> pd.DataFrame:
        """
        Run the backtest and return results.
        
        Returns:
            DataFrame with columns:
            - Date index
            - Close: Close prices
            - signal: Trading signals
            - position: Position size (0 or 1)
            - holdings_value: Value of holdings
            - cash: Cash remaining
            - total_equity: Total portfolio value
            - trade_flag: True when a trade occurs
        """
        # Initialize result DataFrame
        result = pd.DataFrame(index=self.price_df.index)
        result['close'] = self.price_df['close']
        result['signal'] = self.signals
        
        # Calculate position changes (when signal changes from 0 to 1 or 1 to 0)
        position_changes = self.signals.diff()
        
        # Mark trade flags (True when position changes, including first day if signal is 1)
        trade_flags = (position_changes != 0) & (position_changes.notna())
        
        # Also mark first day as trade if signal is 1 (entering position immediately)
        if len(result) > 0 and result['signal'].iloc[0] == 1:
            trade_flags.iloc[0] = True
        
        result['trade_flag'] = trade_flags
        
        # Calculate position (same as signal for long-only strategy)
        result['position'] = self.signals
        
        # Calculate trade costs for each trade
        # Cost is applied to the notional value of the trade
        trade_notional = abs(position_changes) * self.trade_prices
        result['trade_cost'] = trade_notional * self.total_cost_bps / 10000
        
        # Fill NaN values (first row will be NaN due to diff)
        result['trade_cost'] = result['trade_cost'].fillna(0)
        
        # Also calculate cost for first day if signal is 1 (entering position immediately)
        if len(result) > 0 and result['signal'].iloc[0] == 1:
            first_day_cost = self.trade_prices.iloc[0] * self.total_cost_bps / 10000
            result.loc[result.index[0], 'trade_cost'] = first_day_cost
        
        # Calculate holdings value (position * close price)
        result['holdings_value'] = result['position'] * result['close']
        
        # Calculate cash flow for each trade
        # In this simplified model, we only track trade costs
        # The position value is tracked separately in holdings_value
        cash_flow = pd.Series(0.0, index=result.index)
        
        # All trades: cash decreases by trade cost
        cash_flow[result['trade_flag']] = -result['trade_cost'][result['trade_flag']]
        
        # Calculate cumulative cash flow
        cumulative_cash_flow = cash_flow.cumsum()
        
        # Calculate cash remaining
        result['cash'] = self.params.initial_cash + cumulative_cash_flow
        
        # Calculate total equity (cash + holdings value)
        result['total_equity'] = result['cash'] + result['holdings_value']
        
        return result
    
    def get_performance_metrics(self, result_df: Optional[pd.DataFrame] = None) -> dict:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            result_df: Backtest results DataFrame (if None, runs backtest)
            
        Returns:
            Dictionary with performance metrics
        """
        if result_df is None:
            result_df = self.run()
        
        # Calculate returns
        equity = result_df['total_equity']
        returns = equity.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(equity)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        total_trades = result_df['trade_flag'].sum()
        total_costs = result_df['trade_cost'].sum()
        
        # Win/Loss analysis - Vectorized approach
        # Find trade entry and exit points
        trade_flags = result_df['trade_flag']
        positions = result_df['position']
        prices = result_df['close']
        
        # Get indices where trades occur
        trade_indices = result_df[trade_flags].index
        
        if len(trade_indices) > 0:
            # Find entry and exit pairs
            entry_indices = []
            exit_indices = []
            
            # Vectorized approach to find entry/exit pairs
            position_changes = positions.diff().fillna(0)
            entry_mask = (position_changes == 1) & trade_flags
            exit_mask = (position_changes == -1) & trade_flags
            
            entry_indices = result_df[entry_mask].index.tolist()
            exit_indices = result_df[exit_mask].index.tolist()
            
            # Calculate trade returns vectorized
            if len(entry_indices) > 0 and len(exit_indices) > 0:
                # Ensure we have matching pairs
                min_pairs = min(len(entry_indices), len(exit_indices))
                entry_prices = result_df.loc[entry_indices[:min_pairs], 'close'].values
                exit_prices = result_df.loc[exit_indices[:min_pairs], 'close'].values
                trade_returns = (exit_prices - entry_prices) / entry_prices
                trade_returns = trade_returns.tolist()
            else:
                trade_returns = []
            
            # Handle case where we have an open position at the end
            if len(entry_indices) > len(exit_indices):
                last_entry_price = result_df.loc[entry_indices[-1], 'close']
                final_price = prices.iloc[-1]
                final_return = (final_price - last_entry_price) / last_entry_price
                trade_returns.append(final_return)
        else:
            trade_returns = []
        
        # Calculate win/loss statistics vectorized
        if trade_returns:
            trade_returns = np.array(trade_returns)
            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns < 0]
            
            win_rate = len(wins) / len(trade_returns)
            avg_win = np.mean(wins) if len(wins) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': int(total_trades),
            'total_costs': total_costs,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': equity.iloc[-1]
        }
    
    def get_trade_log(self, result_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get detailed trade log with entry/exit information.
        
        Args:
            result_df: Backtest results DataFrame (if None, runs backtest)
            
        Returns:
            DataFrame with trade details
        """
        if result_df is None:
            result_df = self.run()
        
        # Find all trade points
        trade_points = result_df[result_df['trade_flag']].copy()
        
        if len(trade_points) == 0:
            return pd.DataFrame(columns=['date', 'action', 'price', 'cost', 'equity'])
        
        # Determine action based on position change
        position_changes = result_df['signal'].diff()
        trade_changes = position_changes[result_df['trade_flag']]
        
        trade_points['action'] = 'entry'
        trade_points.loc[trade_changes == -1, 'action'] = 'exit'
        
        # Select relevant columns
        trade_log = trade_points[['close', 'action', 'trade_cost', 'total_equity']].copy()
        trade_log.columns = ['price', 'action', 'cost', 'equity']
        trade_log.index.name = 'date'
        
        return trade_log.reset_index()
