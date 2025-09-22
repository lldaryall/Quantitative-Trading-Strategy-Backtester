#!/usr/bin/env python3
"""
Vectorization Performance Profiling Script

This script validates the vectorization performance of the qbacktester by:
1. Generating synthetic price series (~2500 trading days)
2. Running 100 backtests with random fast/slow parameters
3. Timing overall runtime and asserting completion under reasonable threshold
4. Scanning for explicit Python for-loops in backtester core using AST
"""

import os
import sys
import time
import random
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Set

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qbacktester.backtester import Backtester
from qbacktester.strategy import StrategyParams


def generate_synthetic_price_series(n_days: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic price series for profiling.
    
    Args:
        n_days: Number of trading days to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with 'close' and 'open' columns
    """
    np.random.seed(seed)
    
    # Generate realistic price series with trend and volatility
    returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily return, 2% volatility
    prices = 100 * np.cumprod(1 + returns)
    
    # Add some intraday volatility for open prices
    open_returns = np.random.normal(0, 0.005, n_days)  # 0.5% intraday volatility
    open_prices = prices * (1 + open_returns)
    
    # Create date index
    dates = pd.date_range('2015-01-01', periods=n_days, freq='D')
    
    return pd.DataFrame({
        'close': prices,
        'open': open_prices
    }, index=dates)


def generate_random_signals(n_days: int, signal_frequency: float = 0.1) -> pd.Series:
    """
    Generate random trading signals for profiling.
    
    Args:
        n_days: Number of days
        signal_frequency: Probability of signal on any given day
        
    Returns:
        Series of 0/1 signals
    """
    np.random.seed(42)
    signals = np.random.random(n_days) < signal_frequency
    return pd.Series(signals.astype(int), index=pd.date_range('2015-01-01', periods=n_days, freq='D'))


def run_backtest_profile(
    price_df: pd.DataFrame,
    signals: pd.Series,
    fast_window: int,
    slow_window: int,
    initial_cash: float = 100000,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5
) -> dict:
    """
    Run a single backtest and return timing information.
    
    Args:
        price_df: Price data
        signals: Trading signals
        fast_window: Fast SMA window
        slow_window: Slow SMA window
        initial_cash: Initial capital
        fee_bps: Trading fees in basis points
        slippage_bps: Slippage costs in basis points
        
    Returns:
        Dictionary with timing and performance metrics
    """
    start_time = time.time()
    
    params = StrategyParams(
        symbol="PROFILE",
        start="2015-01-01",
        end="2025-01-01",
        fast_window=fast_window,
        slow_window=slow_window,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps
    )
    
    backtester = Backtester(price_df, signals, params)
    result = backtester.run()
    metrics = backtester.get_performance_metrics()
    
    end_time = time.time()
    
    return {
        'runtime': end_time - start_time,
        'final_equity': result['total_equity'].iloc[-1],
        'total_trades': metrics['total_trades'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'total_costs': metrics['total_costs']
    }


def scan_for_loops(file_path: Path) -> List[Tuple[int, str]]:
    """
    Scan a Python file for explicit for-loops using AST.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        List of (line_number, loop_type) tuples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    loops = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            loops.append((node.lineno, 'for'))
        elif isinstance(node, ast.While):
            loops.append((node.lineno, 'while'))
    
    return loops


def check_backtester_vectorization() -> bool:
    """
    Check that backtester core has no explicit for-loops.
    
    Returns:
        True if no loops found, False otherwise
    """
    backtester_path = Path(__file__).parent.parent / "src" / "qbacktester" / "backtester.py"
    
    if not backtester_path.exists():
        print(f"❌ Backtester file not found: {backtester_path}")
        return False
    
    loops = scan_for_loops(backtester_path)
    
    if loops:
        print(f"❌ Found explicit loops in backtester.py:")
        for line_num, loop_type in loops:
            print(f"   Line {line_num}: {loop_type} loop")
        return False
    
    print("✅ No explicit for-loops found in backtester.py")
    return True


def run_performance_profile(
    n_backtests: int = 100,
    n_days: int = 2500,
    max_runtime_seconds: float = 30.0
) -> dict:
    """
    Run performance profiling with multiple backtests.
    
    Args:
        n_backtests: Number of backtests to run
        n_days: Number of trading days
        max_runtime_seconds: Maximum allowed runtime
        
    Returns:
        Dictionary with profiling results
    """
    print(f"🚀 Starting vectorization performance profiling...")
    print(f"   📊 Generating {n_days} days of synthetic data")
    print(f"   🔄 Running {n_backtests} backtests with random parameters")
    print(f"   ⏱️  Target runtime: < {max_runtime_seconds} seconds")
    
    # Generate synthetic data
    price_df = generate_synthetic_price_series(n_days)
    signals = generate_random_signals(n_days)
    
    print(f"   📈 Price range: ${price_df['close'].min():.2f} - ${price_df['close'].max():.2f}")
    print(f"   📊 Signal frequency: {signals.sum() / len(signals):.1%}")
    
    # Run backtests with random parameters
    start_time = time.time()
    results = []
    
    for i in range(n_backtests):
        # Generate random parameters
        fast_window = random.randint(5, 50)
        slow_window = random.randint(fast_window + 1, 100)
        
        result = run_backtest_profile(
            price_df, signals, fast_window, slow_window
        )
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"   ✅ Completed {i + 1}/{n_backtests} backtests")
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    # Calculate statistics
    runtimes = [r['runtime'] for r in results]
    final_equities = [r['final_equity'] for r in results]
    total_trades = [r['total_trades'] for r in results]
    sharpe_ratios = [r['sharpe_ratio'] for r in results if r['sharpe_ratio'] is not None]
    
    stats = {
        'total_runtime': total_runtime,
        'avg_backtest_runtime': np.mean(runtimes),
        'min_backtest_runtime': np.min(runtimes),
        'max_backtest_runtime': np.max(runtimes),
        'avg_final_equity': np.mean(final_equities),
        'avg_total_trades': np.mean(total_trades),
        'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
        'success_rate': len([r for r in results if r['final_equity'] > 0]) / len(results)
    }
    
    # Print results
    print(f"\n📊 Performance Results:")
    print(f"   ⏱️  Total runtime: {total_runtime:.2f} seconds")
    print(f"   📈 Average backtest: {stats['avg_backtest_runtime']*1000:.1f} ms")
    print(f"   🚀 Fastest backtest: {stats['min_backtest_runtime']*1000:.1f} ms")
    print(f"   🐌 Slowest backtest: {stats['max_backtest_runtime']*1000:.1f} ms")
    print(f"   💰 Average final equity: ${stats['avg_final_equity']:,.2f}")
    print(f"   📊 Average trades: {stats['avg_total_trades']:.1f}")
    print(f"   📈 Average Sharpe: {stats['avg_sharpe_ratio']:.3f}")
    print(f"   ✅ Success rate: {stats['success_rate']:.1%}")
    
    # Validate performance
    if total_runtime > max_runtime_seconds:
        print(f"❌ Performance test FAILED: {total_runtime:.2f}s > {max_runtime_seconds}s")
        return False
    else:
        print(f"✅ Performance test PASSED: {total_runtime:.2f}s < {max_runtime_seconds}s")
        return True


def main():
    """Main profiling function."""
    print("🔍 QBacktester Vectorization Performance Profiling")
    print("=" * 60)
    
    # Check for explicit loops in backtester core
    print("\n1. Checking for explicit loops in backtester core...")
    if not check_backtester_vectorization():
        print("❌ Vectorization check FAILED")
        sys.exit(1)
    
    # Run performance profiling
    print("\n2. Running performance profiling...")
    success = run_performance_profile(
        n_backtests=100,
        n_days=2500,
        max_runtime_seconds=30.0
    )
    
    if success:
        print("\n🎉 All vectorization checks PASSED!")
        print("   ✅ No explicit for-loops in backtester core")
        print("   ✅ Performance meets requirements")
        sys.exit(0)
    else:
        print("\n❌ Vectorization checks FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
