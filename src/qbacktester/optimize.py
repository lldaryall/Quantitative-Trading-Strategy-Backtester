"""
Parameter optimization module for backtesting strategies.

This module provides grid search optimization to find the best parameters
for trading strategies based on various performance metrics.
"""

import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import get_all_metrics
from .run import run_crossover_backtest
from .strategy import StrategyParams


def _run_single_backtest(
    args: Tuple[str, str, str, int, int, float, float, float, str],
) -> Dict[str, Any]:
    """
    Run a single backtest for optimization.

    This function is designed to be used with multiprocessing.

    Args:
        args: Tuple containing (symbol, start, end, fast, slow, initial_cash, fee_bps, slippage_bps, metric)

    Returns:
        Dictionary with optimization results
    """
    symbol, start, end, fast, slow, initial_cash, fee_bps, slippage_bps, metric = args

    try:
        # Create strategy parameters
        params = StrategyParams(
            symbol=symbol,
            start=start,
            end=end,
            fast_window=fast,
            slow_window=slow,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        # Run backtest
        results = run_crossover_backtest(params)

        # Extract metrics
        metrics = results["metrics"]
        equity_curve = results["equity_curve"]

        return {
            "fast": fast,
            "slow": slow,
            "sharpe": metrics.get("sharpe", np.nan),
            "max_dd": metrics.get("max_drawdown", np.nan),
            "cagr": metrics.get("cagr", np.nan),
            "equity_final": (
                equity_curve["total_equity"].iloc[-1]
                if not equity_curve.empty
                else np.nan
            ),
            "volatility": metrics.get("volatility", np.nan),
            "calmar": metrics.get("calmar", np.nan),
            "hit_rate": metrics.get("hit_rate", np.nan),
            "total_trades": len(results.get("trades", [])),
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "fast": fast,
            "slow": slow,
            "sharpe": np.nan,
            "max_dd": np.nan,
            "cagr": np.nan,
            "equity_final": np.nan,
            "volatility": np.nan,
            "calmar": np.nan,
            "hit_rate": np.nan,
            "total_trades": 0,
            "success": False,
            "error": str(e),
        }


def grid_search(
    symbol: str,
    start: str,
    end: str,
    fast_grid: List[int],
    slow_grid: List[int],
    metric: str = "sharpe",
    initial_cash: float = 100000,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Perform grid search optimization for SMA crossover strategy.

    Args:
        symbol: Stock symbol to optimize
        start: Start date for backtesting
        end: End date for backtesting
        fast_grid: List of fast window values to test
        slow_grid: List of slow window values to test
        metric: Primary metric to optimize ('sharpe', 'cagr', 'calmar', 'max_dd')
        initial_cash: Initial capital for backtesting
        fee_bps: Trading fees in basis points
        slippage_bps: Slippage costs in basis points
        n_jobs: Number of parallel jobs (None for auto, 1 for sequential)
        verbose: Whether to print progress updates

    Returns:
        DataFrame with optimization results sorted by the chosen metric

    Raises:
        ValueError: If metric is not supported or grids are invalid
    """
    # Validate inputs
    if metric not in ["sharpe", "cagr", "calmar", "max_dd"]:
        raise ValueError(
            f"Unsupported metric: {metric}. Must be one of: sharpe, cagr, calmar, max_dd"
        )

    if not fast_grid or not slow_grid:
        raise ValueError("fast_grid and slow_grid cannot be empty")

    if any(fast >= slow for fast in fast_grid for slow in slow_grid):
        raise ValueError("All fast values must be less than all slow values")

    if any(x <= 0 for x in fast_grid + slow_grid):
        raise ValueError("All window values must be positive")

    # Generate parameter combinations
    param_combinations = list(product(fast_grid, slow_grid))
    total_combinations = len(param_combinations)

    if verbose:
        print(f"🔍 Starting grid search optimization for {symbol}")
        print(f"📊 Testing {total_combinations} parameter combinations")
        print(f"📈 Fast windows: {fast_grid}")
        print(f"📉 Slow windows: {slow_grid}")
        print(f"🎯 Optimizing for: {metric}")

    # Prepare arguments for parallel processing
    args_list = [
        (symbol, start, end, fast, slow, initial_cash, fee_bps, slippage_bps, metric)
        for fast, slow in param_combinations
    ]

    results = []

    # Determine number of jobs
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), total_combinations)
    elif n_jobs <= 0:
        n_jobs = 1

    if n_jobs == 1 or total_combinations == 1:
        # Sequential processing
        if verbose:
            print("🔄 Running optimization sequentially...")

        for i, args in enumerate(args_list):
            if verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                print(
                    f"   Progress: {i + 1}/{total_combinations} ({100 * (i + 1) / total_combinations:.1f}%)"
                )

            result = _run_single_backtest(args)
            results.append(result)

    else:
        # Parallel processing
        if verbose:
            print(f"🚀 Running optimization in parallel with {n_jobs} workers...")

        # Suppress warnings for multiprocessing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all jobs
                future_to_args = {
                    executor.submit(_run_single_backtest, args): args
                    for args in args_list
                }

                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_args):
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if verbose and completed % max(1, total_combinations // 10) == 0:
                        print(
                            f"   Progress: {completed}/{total_combinations} ({100 * completed / total_combinations:.1f}%)"
                        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by the chosen metric (descending for most metrics, ascending for max_dd)
    if metric == "max_dd":
        results_df = results_df.sort_values(metric, ascending=True)
    else:
        results_df = results_df.sort_values(metric, ascending=False)

    # Reset index
    results_df = results_df.reset_index(drop=True)

    # Add some summary statistics
    successful_runs = results_df["success"].sum()
    failed_runs = total_combinations - successful_runs

    if verbose:
        print(f"✅ Optimization completed!")
        print(f"📊 Successful runs: {successful_runs}/{total_combinations}")
        if failed_runs > 0:
            print(f"❌ Failed runs: {failed_runs}")

        # Show top 3 results
        print(f"\n🏆 Top 3 parameter sets by {metric}:")
        top_results = results_df.head(3)
        for i, (_, row) in enumerate(top_results.iterrows(), 1):
            print(
                f"   {i}. Fast={row['fast']}, Slow={row['slow']}: "
                f"{metric}={row[metric]:.4f}, Sharpe={row['sharpe']:.4f}, "
                f"CAGR={row['cagr']:.4f}, MaxDD={row['max_dd']:.4f}"
            )

    return results_df


def optimize_strategy(
    symbol: str,
    start: str,
    end: str,
    fast_range: Tuple[int, int, int] = (5, 50, 5),
    slow_range: Tuple[int, int, int] = (50, 200, 10),
    metric: str = "sharpe",
    initial_cash: float = 100000,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convenience function for strategy optimization with range-based grids.

    Args:
        symbol: Stock symbol to optimize
        start: Start date for backtesting
        end: End date for backtesting
        fast_range: (start, stop, step) for fast window grid
        slow_range: (start, stop, step) for slow window grid
        metric: Primary metric to optimize
        initial_cash: Initial capital for backtesting
        fee_bps: Trading fees in basis points
        slippage_bps: Slippage costs in basis points
        n_jobs: Number of parallel jobs
        verbose: Whether to print progress updates

    Returns:
        DataFrame with optimization results
    """
    fast_start, fast_stop, fast_step = fast_range
    slow_start, slow_stop, slow_step = slow_range

    fast_grid = list(range(fast_start, fast_stop + 1, fast_step))
    slow_grid = list(range(slow_start, slow_stop + 1, slow_step))

    return grid_search(
        symbol=symbol,
        start=start,
        end=end,
        fast_grid=fast_grid,
        slow_grid=slow_grid,
        metric=metric,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        n_jobs=n_jobs,
        verbose=verbose,
    )


def save_optimization_results(
    results_df: pd.DataFrame, symbol: str, output_dir: str = "reports"
) -> str:
    """
    Save optimization results to CSV file.

    Args:
        results_df: DataFrame with optimization results
        symbol: Stock symbol for filename
        output_dir: Directory to save the file

    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate filename
    filename = f"opt_grid_{symbol.lower()}.csv"
    file_path = output_path / filename

    # Save results
    results_df.to_csv(file_path, index=False)

    return str(file_path)


def print_optimization_summary(
    results_df: pd.DataFrame, metric: str, top_n: int = 5
) -> None:
    """
    Print a summary of optimization results.

    Args:
        results_df: DataFrame with optimization results
        metric: Primary metric that was optimized
        top_n: Number of top results to display
    """
    print(f"\n📊 Optimization Results Summary")
    print(f"{'='*60}")

    # Show top results
    top_results = results_df.head(top_n)

    print(f"\n🏆 Top {top_n} parameter sets by {metric}:")
    print(
        f"{'Rank':<4} {'Fast':<6} {'Slow':<6} {'Sharpe':<8} {'CAGR':<8} {'MaxDD':<8} {'Calmar':<8} {'HitRate':<8}"
    )
    print(f"{'-'*60}")

    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(
            f"{i:<4} {row['fast']:<6} {row['slow']:<6} "
            f"{row['sharpe']:<8.4f} {row['cagr']:<8.4f} {row['max_dd']:<8.4f} "
            f"{row['calmar']:<8.4f} {row['hit_rate']:<8.4f}"
        )

    # Show statistics
    successful_results = results_df[results_df["success"]]
    if len(successful_results) > 0:
        print(f"\n📈 Performance Statistics:")
        print(f"   Best {metric}: {successful_results[metric].max():.4f}")
        print(f"   Worst {metric}: {successful_results[metric].min():.4f}")
        print(f"   Average {metric}: {successful_results[metric].mean():.4f}")
        print(f"   Std {metric}: {successful_results[metric].std():.4f}")

        print(f"\n🎯 Best Parameters:")
        best_row = successful_results.iloc[0]
        print(f"   Fast Window: {best_row['fast']}")
        print(f"   Slow Window: {best_row['slow']}")
        print(f"   Sharpe Ratio: {best_row['sharpe']:.4f}")
        print(f"   CAGR: {best_row['cagr']:.4f}")
        print(f"   Max Drawdown: {best_row['max_dd']:.4f}")
        print(f"   Calmar Ratio: {best_row['calmar']:.4f}")
        print(f"   Hit Rate: {best_row['hit_rate']:.4f}")
        print(f"   Total Trades: {best_row['total_trades']}")


# Safe multiprocessing guard
if __name__ == "__main__":
    # This allows the module to be run as a script for testing
    import sys

    if len(sys.argv) > 1:
        # Example usage
        results = grid_search(
            symbol="AAPL",
            start="2020-01-01",
            end="2023-12-31",
            fast_grid=[5, 10, 20],
            slow_grid=[50, 100, 150],
            metric="sharpe",
            verbose=True,
        )

        print_optimization_summary(results, "sharpe")
        save_optimization_results(results, "AAPL")
