"""
Walk-forward analysis module for backtesting strategies.

This module provides walk-forward analysis functionality to test strategy robustness
by optimizing parameters on in-sample data and evaluating on out-of-sample data
across rolling time windows.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings

from .optimize import grid_search
from .run import run_crossover_backtest
from .metrics import get_all_metrics
from .strategy import StrategyParams


def walkforward_crossover(
    symbol: str,
    start: str,
    end: str,
    in_sample_years: int = 3,
    out_sample_years: int = 1,
    fast_grid: List[int] = [10, 20, 50],
    slow_grid: List[int] = [50, 100, 200],
    initial_cash: float = 100000,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    optimization_metric: str = "sharpe",
    n_jobs: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform walk-forward analysis for SMA crossover strategy.
    
    This function implements a rolling window approach where:
    1. Parameters are optimized on in-sample data
    2. Best parameters are selected based on optimization metric
    3. Strategy is evaluated on out-of-sample data
    4. Process repeats with rolling windows
    
    Args:
        symbol: Stock symbol to analyze
        start: Start date for analysis (YYYY-MM-DD)
        end: End date for analysis (YYYY-MM-DD)
        in_sample_years: Years of data for parameter optimization
        out_sample_years: Years of data for out-of-sample evaluation
        fast_grid: List of fast window values to test
        slow_grid: List of slow window values to test
        initial_cash: Initial capital for backtesting
        fee_bps: Trading fees in basis points
        slippage_bps: Slippage costs in basis points
        optimization_metric: Metric to optimize ('sharpe', 'cagr', 'calmar', 'max_dd')
        n_jobs: Number of parallel jobs for optimization
        verbose: Whether to print progress updates
        
    Returns:
        Dictionary containing:
        - 'windows': List of window results with best parameters and OOS metrics
        - 'equity_curve': Concatenated out-of-sample equity curve
        - 'summary': Overall walk-forward performance metrics
        - 'parameters': Best parameters for each window
        - 'dates': Window start/end dates
        
    Raises:
        ValueError: If date ranges are invalid or insufficient data
    """
    # Validate inputs
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
    
    if in_sample_years <= 0 or out_sample_years <= 0:
        raise ValueError("In-sample and out-of-sample years must be positive")
    
    if in_sample_years + out_sample_years > (end_date - start_date).days / 365.25:
        raise ValueError("Total analysis period must be longer than in-sample + out-of-sample years")
    
    # Generate rolling windows
    windows = _generate_walkforward_windows(
        start_date, end_date, in_sample_years, out_sample_years
    )
    
    if verbose:
        print(f"🔄 Starting walk-forward analysis for {symbol}")
        print(f"📊 Total windows: {len(windows)}")
        print(f"📈 In-sample period: {in_sample_years} years")
        print(f"📉 Out-of-sample period: {out_sample_years} years")
        print(f"🎯 Optimization metric: {optimization_metric}")
    
    # Process each window
    window_results = []
    oos_equity_curves = []
    
    for i, window in enumerate(windows):
        if verbose:
            print(f"\n🪟 Processing window {i+1}/{len(windows)}")
            print(f"   IS: {window['is_start']} to {window['is_end']}")
            print(f"   OOS: {window['oos_start']} to {window['oos_end']}")
        
        # Optimize parameters on in-sample data
        try:
            is_results = grid_search(
                symbol=symbol,
                start=window['is_start'].strftime('%Y-%m-%d'),
                end=window['is_end'].strftime('%Y-%m-%d'),
                fast_grid=fast_grid,
                slow_grid=slow_grid,
                metric=optimization_metric,
                initial_cash=initial_cash,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                n_jobs=n_jobs,
                verbose=False
            )
            
            # Get best parameters
            best_params = is_results.iloc[0]
            best_fast = int(best_params['fast'])
            best_slow = int(best_params['slow'])
            best_metric = best_params[optimization_metric]
            
            if verbose:
                print(f"   ✅ Best params: Fast={best_fast}, Slow={best_slow} ({optimization_metric}={best_metric:.4f})")
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Optimization failed: {e}")
            
            # Store failed window
            window_result = {
                'window': i + 1,
                'is_start': window['is_start'],
                'is_end': window['is_end'],
                'oos_start': window['oos_start'],
                'oos_end': window['oos_end'],
                'best_fast': None,
                'best_slow': None,
                'best_is_metric': None,
                'oos_equity_curve': None,
                'oos_metrics': None,
                'success': False,
                'error': str(e)
            }
            window_results.append(window_result)
            continue
        
        # Evaluate on out-of-sample data
        try:
            oos_params = StrategyParams(
                symbol=symbol,
                start=window['oos_start'].strftime('%Y-%m-%d'),
                end=window['oos_end'].strftime('%Y-%m-%d'),
                fast_window=best_fast,
                slow_window=best_slow,
                initial_cash=initial_cash,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps
            )
            
            oos_results = run_crossover_backtest(oos_params)
            oos_equity_curve = oos_results['equity_curve']['total_equity']
            oos_metrics = oos_results['metrics']
            
            # Store window results
            window_result = {
                'window': i + 1,
                'is_start': window['is_start'],
                'is_end': window['is_end'],
                'oos_start': window['oos_start'],
                'oos_end': window['oos_end'],
                'best_fast': best_fast,
                'best_slow': best_slow,
                'best_is_metric': best_metric,
                'oos_equity_curve': oos_equity_curve,
                'oos_metrics': oos_metrics,
                'success': True,
                'error': None
            }
            
            window_results.append(window_result)
            oos_equity_curves.append(oos_equity_curve)
            
            if verbose:
                print(f"   📊 OOS Performance: CAGR={oos_metrics['cagr']:.4f}, "
                      f"Sharpe={oos_metrics['sharpe']:.4f}, MaxDD={oos_metrics['max_drawdown']:.4f}")
            
        except Exception as e:
            if verbose:
                print(f"   ❌ OOS evaluation failed: {e}")
            
            # Store failed window
            window_result = {
                'window': i + 1,
                'is_start': window['is_start'],
                'is_end': window['is_end'],
                'oos_start': window['oos_start'],
                'oos_end': window['oos_end'],
                'best_fast': best_fast,
                'best_slow': best_slow,
                'best_is_metric': best_metric,
                'oos_equity_curve': None,
                'oos_metrics': None,
                'success': False,
                'error': str(e)
            }
            window_results.append(window_result)
    
    # Concatenate out-of-sample equity curves
    if oos_equity_curves:
        # Normalize each curve to start at the same value
        normalized_curves = []
        for i, curve in enumerate(oos_equity_curves):
            if i == 0:
                # First curve starts at initial cash
                normalized_curve = curve.copy()
            else:
                # Subsequent curves start where previous ended
                prev_end = normalized_curves[-1].iloc[-1]
                normalized_curve = curve * (prev_end / curve.iloc[0])
            
            normalized_curves.append(normalized_curve)
        
        # Concatenate all curves and remove duplicate indices
        full_equity_curve = pd.concat(normalized_curves, ignore_index=False)
        full_equity_curve = full_equity_curve.groupby(full_equity_curve.index).last()  # Remove duplicates
        full_equity_curve = full_equity_curve.sort_index()
    else:
        full_equity_curve = pd.Series(dtype=float)
    
    # Calculate overall performance metrics
    if not full_equity_curve.empty:
        overall_metrics = get_all_metrics(full_equity_curve)
    else:
        overall_metrics = {}
    
    # Create summary
    successful_windows = [w for w in window_results if w['success']]
    failed_windows = [w for w in window_results if not w['success']]
    
    summary = {
        'total_windows': len(window_results),
        'successful_windows': len(successful_windows),
        'failed_windows': len(failed_windows),
        'success_rate': len(successful_windows) / len(window_results) if window_results else 0,
        'overall_metrics': overall_metrics,
        'parameter_stability': _calculate_parameter_stability(successful_windows),
        'performance_consistency': _calculate_performance_consistency(successful_windows)
    }
    
    if verbose:
        print(f"\n✅ Walk-forward analysis completed!")
        print(f"📊 Successful windows: {len(successful_windows)}/{len(window_results)}")
        if successful_windows:
            print(f"📈 Overall CAGR: {overall_metrics.get('cagr', 0):.4f}")
            print(f"📊 Overall Sharpe: {overall_metrics.get('sharpe', 0):.4f}")
            print(f"📉 Overall Max DD: {overall_metrics.get('max_drawdown', 0):.4f}")
    
    return {
        'windows': window_results,
        'equity_curve': full_equity_curve,
        'summary': summary,
        'parameters': [(w['best_fast'], w['best_slow']) for w in successful_windows],
        'dates': [(w['oos_start'], w['oos_end']) for w in successful_windows]
    }


def _generate_walkforward_windows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    in_sample_years: int,
    out_sample_years: int
) -> List[Dict[str, pd.Timestamp]]:
    """
    Generate rolling windows for walk-forward analysis.
    
    Args:
        start_date: Analysis start date
        end_date: Analysis end date
        in_sample_years: Years of in-sample data
        out_sample_years: Years of out-of-sample data
        
    Returns:
        List of window dictionaries with IS and OOS start/end dates
    """
    windows = []
    current_date = start_date
    
    while current_date < end_date:
        # Calculate window dates
        is_start = current_date
        is_end = is_start + pd.DateOffset(years=in_sample_years)
        oos_start = is_end
        oos_end = oos_start + pd.DateOffset(years=out_sample_years)
        
        # Check if we have enough data for this window
        if oos_end > end_date:
            break
        
        windows.append({
            'is_start': is_start,
            'is_end': is_end,
            'oos_start': oos_start,
            'oos_end': oos_end
        })
        
        # Move to next window (overlapping by out_sample_years)
        current_date = oos_start
    
    return windows


def _calculate_parameter_stability(successful_windows: List[Dict]) -> Dict[str, float]:
    """
    Calculate parameter stability metrics.
    
    Args:
        successful_windows: List of successful window results
        
    Returns:
        Dictionary with stability metrics
    """
    if not successful_windows:
        return {}
    
    fast_values = [w['best_fast'] for w in successful_windows]
    slow_values = [w['best_slow'] for w in successful_windows]
    
    return {
        'fast_mean': np.mean(fast_values),
        'fast_std': np.std(fast_values),
        'fast_cv': np.std(fast_values) / np.mean(fast_values) if np.mean(fast_values) > 0 else 0,
        'slow_mean': np.mean(slow_values),
        'slow_std': np.std(slow_values),
        'slow_cv': np.std(slow_values) / np.mean(slow_values) if np.mean(slow_values) > 0 else 0,
        'parameter_changes': sum(1 for i in range(1, len(fast_values)) 
                                if fast_values[i] != fast_values[i-1] or slow_values[i] != slow_values[i-1])
    }


def _calculate_performance_consistency(successful_windows: List[Dict]) -> Dict[str, float]:
    """
    Calculate performance consistency metrics.
    
    Args:
        successful_windows: List of successful window results
        
    Returns:
        Dictionary with consistency metrics
    """
    if not successful_windows:
        return {}
    
    # Extract OOS metrics
    cagr_values = [w['oos_metrics']['cagr'] for w in successful_windows if w['oos_metrics']]
    sharpe_values = [w['oos_metrics']['sharpe'] for w in successful_windows if w['oos_metrics']]
    max_dd_values = [w['oos_metrics']['max_drawdown'] for w in successful_windows if w['oos_metrics']]
    
    if not cagr_values:
        return {}
    
    return {
        'cagr_mean': np.mean(cagr_values),
        'cagr_std': np.std(cagr_values),
        'cagr_cv': np.std(cagr_values) / np.mean(cagr_values) if np.mean(cagr_values) > 0 else 0,
        'sharpe_mean': np.mean(sharpe_values),
        'sharpe_std': np.std(sharpe_values),
        'sharpe_cv': np.std(sharpe_values) / np.mean(sharpe_values) if np.mean(sharpe_values) > 0 else 0,
        'max_dd_mean': np.mean(max_dd_values),
        'max_dd_std': np.std(max_dd_values),
        'positive_periods': sum(1 for cagr in cagr_values if cagr > 0),
        'win_rate': sum(1 for cagr in cagr_values if cagr > 0) / len(cagr_values)
    }


def print_walkforward_summary(results: Dict[str, Any], top_n: int = 5) -> None:
    """
    Print a summary of walk-forward analysis results.
    
    Args:
        results: Results from walkforward_crossover
        top_n: Number of top windows to display
    """
    windows = results['windows']
    summary = results['summary']
    successful_windows = [w for w in windows if w['success']]
    
    print(f"\n📊 Walk-Forward Analysis Summary")
    print(f"{'='*60}")
    
    # Overall statistics
    print(f"\n📈 Overall Performance:")
    print(f"   Total Windows: {summary['total_windows']}")
    print(f"   Successful Windows: {summary['successful_windows']}")
    print(f"   Success Rate: {summary['success_rate']:.1%}")
    
    if summary['overall_metrics']:
        metrics = summary['overall_metrics']
        print(f"   Overall CAGR: {metrics.get('cagr', 0):.4f}")
        print(f"   Overall Sharpe: {metrics.get('sharpe', 0):.4f}")
        print(f"   Overall Max DD: {metrics.get('max_drawdown', 0):.4f}")
        print(f"   Overall Calmar: {metrics.get('calmar', 0):.4f}")
    
    # Parameter stability
    if summary['parameter_stability']:
        stability = summary['parameter_stability']
        print(f"\n🔧 Parameter Stability:")
        print(f"   Fast Window: {stability['fast_mean']:.1f} ± {stability['fast_std']:.1f} (CV: {stability['fast_cv']:.3f})")
        print(f"   Slow Window: {stability['slow_mean']:.1f} ± {stability['slow_std']:.1f} (CV: {stability['slow_cv']:.3f})")
        print(f"   Parameter Changes: {stability['parameter_changes']}")
    
    # Performance consistency
    if summary['performance_consistency']:
        consistency = summary['performance_consistency']
        print(f"\n📊 Performance Consistency:")
        print(f"   CAGR: {consistency['cagr_mean']:.4f} ± {consistency['cagr_std']:.4f}")
        print(f"   Sharpe: {consistency['sharpe_mean']:.4f} ± {consistency['sharpe_std']:.4f}")
        print(f"   Win Rate: {consistency['win_rate']:.1%}")
    
    # Top performing windows
    if successful_windows:
        # Sort by OOS Sharpe ratio
        sorted_windows = sorted(successful_windows, 
                              key=lambda x: x['oos_metrics']['sharpe'] if x['oos_metrics'] else -np.inf, 
                              reverse=True)
        
        print(f"\n🏆 Top {min(top_n, len(sorted_windows))} Windows by OOS Sharpe:")
        print(f"{'Window':<6} {'Period':<20} {'Fast':<6} {'Slow':<6} {'Sharpe':<8} {'CAGR':<8} {'MaxDD':<8}")
        print(f"{'-'*60}")
        
        for i, window in enumerate(sorted_windows[:top_n]):
            if window['oos_metrics']:
                period = f"{window['oos_start'].strftime('%Y-%m')} to {window['oos_end'].strftime('%Y-%m')}"
                print(f"{window['window']:<6} {period:<20} {window['best_fast']:<6} {window['best_slow']:<6} "
                      f"{window['oos_metrics']['sharpe']:<8.4f} {window['oos_metrics']['cagr']:<8.4f} "
                      f"{window['oos_metrics']['max_drawdown']:<8.4f}")


def save_walkforward_results(
    results: Dict[str, Any],
    symbol: str,
    output_dir: str = "reports"
) -> str:
    """
    Save walk-forward analysis results to CSV files.
    
    Args:
        results: Results from walkforward_crossover
        symbol: Stock symbol for filename
        output_dir: Directory to save files
        
    Returns:
        Path to the main results file
    """
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save window results
    windows_df = pd.DataFrame(results['windows'])
    windows_file = output_path / f"walkforward_windows_{symbol.lower()}.csv"
    windows_df.to_csv(windows_file, index=False)
    
    # Save equity curve
    if not results['equity_curve'].empty:
        equity_df = pd.DataFrame({
            'date': results['equity_curve'].index,
            'equity': results['equity_curve'].values
        })
        equity_file = output_path / f"walkforward_equity_{symbol.lower()}.csv"
        equity_df.to_csv(equity_file, index=False)
    
    # Save summary
    summary_file = output_path / f"walkforward_summary_{symbol.lower()}.csv"
    summary_data = []
    
    # Add overall metrics
    if results['summary']['overall_metrics']:
        for metric, value in results['summary']['overall_metrics'].items():
            summary_data.append({'metric': f'overall_{metric}', 'value': value})
    
    # Add parameter stability
    if results['summary']['parameter_stability']:
        for metric, value in results['summary']['parameter_stability'].items():
            summary_data.append({'metric': f'stability_{metric}', 'value': value})
    
    # Add performance consistency
    if results['summary']['performance_consistency']:
        for metric, value in results['summary']['performance_consistency'].items():
            summary_data.append({'metric': f'consistency_{metric}', 'value': value})
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    
    return str(windows_file)


# Safe multiprocessing guard
if __name__ == "__main__":
    # Example usage
    results = walkforward_crossover(
        symbol="SPY",
        start="2010-01-01",
        end="2020-12-31",
        in_sample_years=3,
        out_sample_years=1,
        fast_grid=[10, 20, 50],
        slow_grid=[50, 100, 200],
        verbose=True
    )
    
    print_walkforward_summary(results)
    save_walkforward_results(results, "SPY")
