"""Tests for core functionality."""

import pytest
import pandas as pd
import numpy as np
from qbacktester.core import Backtester, Strategy


class DummyStrategy(Strategy):
    """Dummy strategy for testing."""
    
    def __init__(self) -> None:
        super().__init__("DummyStrategy")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate random signals for testing."""
        signals = pd.DataFrame(
            np.random.choice([-1, 0, 1], size=len(data)),
            index=data.index,
            columns=["signal"]
        )
        return signals


def test_backtester_initialization():
    """Test backtester initialization."""
    backtester = Backtester(initial_capital=100000, commission=0.001)
    assert backtester.initial_capital == 100000
    assert backtester.commission == 0.001
    assert backtester.slippage == 0.0


def test_strategy_initialization():
    """Test strategy initialization."""
    strategy = DummyStrategy()
    assert strategy.name == "DummyStrategy"


def test_generate_signals():
    """Test signal generation."""
    strategy = DummyStrategy()
    data = pd.DataFrame({
        "open": [100, 101, 102],
        "high": [105, 106, 107],
        "low": [95, 96, 97],
        "close": [102, 103, 104],
        "volume": [1000, 1100, 1200]
    })
    
    signals = strategy.generate_signals(data)
    assert len(signals) == len(data)
    assert "signal" in signals.columns
    assert all(signal in [-1, 0, 1] for signal in signals["signal"])


def test_run_backtest():
    """Test backtest execution."""
    backtester = Backtester()
    strategy = DummyStrategy()
    
    data = pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "volume": [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range("2023-01-01", periods=5))
    
    results = backtester.run_backtest(strategy, data)
    
    assert "strategy_name" in results
    assert results["strategy_name"] == "DummyStrategy"
    assert "total_return" in results
    assert "sharpe_ratio" in results
