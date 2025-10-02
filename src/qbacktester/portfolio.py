"""
Portfolio management and simulation.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class Portfolio:
    """Portfolio management class."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []

    def get_value(self, prices: pd.Series) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            quantity * prices.get(symbol, 0)
            for symbol, quantity in self.positions.items()
        )
        return self.cash + position_value

    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: pd.Timestamp,
    ) -> None:
        """Execute a trade."""
        # Apply slippage
        if quantity > 0:  # Buy
            price *= 1 + self.slippage
        else:  # Sell
            price *= 1 - self.slippage

        # Calculate cost including commission
        cost = abs(quantity) * price * (1 + self.commission)

        if quantity > 0:  # Buy
            if cost <= self.cash:
                self.cash -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            else:
                return  # Insufficient funds
        else:  # Sell
            if abs(quantity) <= self.positions.get(symbol, 0):
                self.cash += cost
                self.positions[symbol] += quantity
            else:
                return  # Insufficient position

        # Record trade
        self.trades.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "cost": cost,
            }
        )
