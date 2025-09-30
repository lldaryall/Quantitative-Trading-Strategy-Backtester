"""
Command-line interface for qbacktester.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from .strategy import StrategyParams
from .run import run_crossover_backtest, print_backtest_report, print_quick_summary
from .plotting import create_equity_plot, create_drawdown_plot, create_price_signals_plot
from .optimize import grid_search, save_optimization_results, print_optimization_summary
from .walkforward import walkforward_crossover, print_walkforward_summary, save_walkforward_results


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog='qbt',
        description='Quantitative Backtester - Run SMA crossover backtests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qbt run --symbol SPY --start 2020-01-01 --end 2023-12-31 --fast 20 --slow 50
  qbt run --symbol AAPL --start 2020-01-01 --end 2023-12-31 --fast 10 --slow 30 --plot
  qbt run --symbol MSFT --start 2020-01-01 --end 2023-12-31 --fast 5 --slow 20 --cash 50000 --fee-bps 2.0
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a backtest')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize strategy parameters')
    
    # Walkforward command
    walkforward_parser = subparsers.add_parser('walkforward', help='Perform walk-forward analysis')
    
    # Required arguments
    run_parser.add_argument(
        '--symbol', '-s',
        type=str,
        required=True,
        help='Stock symbol to backtest (e.g., SPY, AAPL)'
    )
    
    run_parser.add_argument(
        '--start', '-st',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    
    run_parser.add_argument(
        '--end', '-e',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    
    run_parser.add_argument(
        '--fast', '-f',
        type=int,
        required=True,
        help='Fast SMA window period'
    )
    
    run_parser.add_argument(
        '--slow', '-sl',
        type=int,
        required=True,
        help='Slow SMA window period'
    )
    
    # Optional arguments
    run_parser.add_argument(
        '--cash', '-c',
        type=float,
        default=100000,
        help='Initial cash amount (default: 100000)'
    )
    
    run_parser.add_argument(
        '--fee-bps',
        type=float,
        default=1.0,
        help='Trading fee in basis points (default: 1.0)'
    )
    
    run_parser.add_argument(
        '--slippage-bps',
        type=float,
        default=0.5,
        help='Slippage cost in basis points (default: 0.5)'
    )
    
    run_parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Generate equity curve and drawdown plots'
    )
    
    run_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='reports',
        help='Output directory for plots (default: reports)'
    )
    
    run_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    run_parser.add_argument(
        '--quick',
        action='store_true',
        help='Show only quick summary instead of full report'
    )
    
    # Optimize command arguments
    optimize_parser.add_argument(
        '--symbol', '-s',
        type=str,
        required=True,
        help='Stock symbol to optimize (e.g., SPY, AAPL)'
    )
    
    optimize_parser.add_argument(
        '--start', '-st',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    
    optimize_parser.add_argument(
        '--end', '-e',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    
    optimize_parser.add_argument(
        '--fast',
        type=str,
        required=True,
        help='Fast window values as comma-separated list (e.g., 5,10,20,50)'
    )
    
    optimize_parser.add_argument(
        '--slow',
        type=str,
        required=True,
        help='Slow window values as comma-separated list (e.g., 50,100,150,200)'
    )
    
    optimize_parser.add_argument(
        '--metric',
        type=str,
        default='sharpe',
        choices=['sharpe', 'cagr', 'calmar', 'max_dd'],
        help='Metric to optimize (default: sharpe)'
    )
    
    optimize_parser.add_argument(
        '--cash', '-c',
        type=float,
        default=100000,
        help='Initial cash amount (default: 100000)'
    )
    
    optimize_parser.add_argument(
        '--fee-bps',
        type=float,
        default=1.0,
        help='Trading fee in basis points (default: 1.0)'
    )
    
    optimize_parser.add_argument(
        '--slippage-bps',
        type=float,
        default=0.5,
        help='Slippage cost in basis points (default: 0.5)'
    )
    
    optimize_parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=None,
        help='Number of parallel jobs (default: auto)'
    )
    
    optimize_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='reports',
        help='Output directory for results (default: reports)'
    )
    
    optimize_parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top results to display (default: 5)'
    )
    
    optimize_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    # Walkforward command arguments
    walkforward_parser.add_argument(
        '--symbol', '-s',
        type=str,
        required=True,
        help='Stock symbol to analyze (e.g., SPY, AAPL)'
    )
    
    walkforward_parser.add_argument(
        '--start', '-st',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    
    walkforward_parser.add_argument(
        '--end', '-e',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    
    walkforward_parser.add_argument(
        '--is',
        type=int,
        default=3,
        help='In-sample years for parameter optimization (default: 3)'
    )
    
    walkforward_parser.add_argument(
        '--oos',
        type=int,
        default=1,
        help='Out-of-sample years for evaluation (default: 1)'
    )
    
    walkforward_parser.add_argument(
        '--fast',
        type=str,
        default='10,20,50',
        help='Fast window values as comma-separated list (default: 10,20,50)'
    )
    
    walkforward_parser.add_argument(
        '--slow',
        type=str,
        default='50,100,200',
        help='Slow window values as comma-separated list (default: 50,100,200)'
    )
    
    walkforward_parser.add_argument(
        '--metric',
        type=str,
        default='sharpe',
        choices=['sharpe', 'cagr', 'calmar', 'max_dd'],
        help='Metric to optimize (default: sharpe)'
    )
    
    walkforward_parser.add_argument(
        '--cash', '-c',
        type=float,
        default=100000,
        help='Initial cash amount (default: 100000)'
    )
    
    walkforward_parser.add_argument(
        '--fee-bps',
        type=float,
        default=1.0,
        help='Trading fee in basis points (default: 1.0)'
    )
    
    walkforward_parser.add_argument(
        '--slippage-bps',
        type=float,
        default=0.5,
        help='Slippage cost in basis points (default: 0.5)'
    )
    
    walkforward_parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=None,
        help='Number of parallel jobs for optimization (default: auto)'
    )
    
    walkforward_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='reports',
        help='Output directory for results (default: reports)'
    )
    
    walkforward_parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top windows to display (default: 5)'
    )
    
    walkforward_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    return parser


def validate_arguments(args) -> None:
    """Validate command line arguments."""
    if args.command not in ['run', 'optimize', 'walkforward']:
        return
    
    # Validate dates
    try:
        start_date = pd.to_datetime(args.start)
        end_date = pd.to_datetime(args.end)
    except Exception as e:
        print(f"Error: Invalid date format. {e}")
        sys.exit(1)
    
    if start_date >= end_date:
        print("Error: Start date must be before end date")
        sys.exit(1)
    
    if args.command == 'run':
        # Validate run command specific arguments
        if args.fast >= args.slow:
            print("Error: Fast window must be less than slow window")
            sys.exit(1)
        
        if args.fast <= 0 or args.slow <= 0:
            print("Error: Window periods must be positive")
            sys.exit(1)
    
    elif args.command == 'optimize':
        # Validate optimize command specific arguments
        try:
            fast_values = [int(x.strip()) for x in args.fast.split(',')]
            slow_values = [int(x.strip()) for x in args.slow.split(',')]
        except ValueError as e:
            print(f"Error: Invalid fast/slow values format. {e}")
            sys.exit(1)
        
        if not fast_values or not slow_values:
            print("Error: Fast and slow values cannot be empty")
            sys.exit(1)
        
        if any(x <= 0 for x in fast_values + slow_values):
            print("Error: All window values must be positive")
            sys.exit(1)
        
        if max(fast_values) >= min(slow_values):
            print("Error: All fast values must be less than all slow values")
            sys.exit(1)
        
        if args.top_n <= 0:
            print("Error: top-n must be positive")
            sys.exit(1)
    
    elif args.command == 'walkforward':
        # Validate walkforward command specific arguments
        try:
            fast_values = [int(x.strip()) for x in args.fast.split(',')]
            slow_values = [int(x.strip()) for x in args.slow.split(',')]
        except ValueError as e:
            print(f"Error: Invalid fast/slow values format. {e}")
            sys.exit(1)
        
        if not fast_values or not slow_values:
            print("Error: Fast and slow values cannot be empty")
            sys.exit(1)
        
        if any(x <= 0 for x in fast_values + slow_values):
            print("Error: All window values must be positive")
            sys.exit(1)
        
        if max(fast_values) >= min(slow_values):
            print("Error: All fast values must be less than all slow values")
            sys.exit(1)
        
        if getattr(args, 'is', 0) <= 0 or getattr(args, 'oos', 0) <= 0:
            print("Error: In-sample and out-of-sample years must be positive")
            sys.exit(1)
        
        if args.top_n <= 0:
            print("Error: top-n must be positive")
            sys.exit(1)
    
    # Validate common arguments
    if args.cash <= 0:
        print("Error: Initial cash must be positive")
        sys.exit(1)
    
    if args.fee_bps < 0 or args.slippage_bps < 0:
        print("Error: Fee and slippage must be non-negative")
        sys.exit(1)


def create_comprehensive_plot(results: dict, output_dir: str, symbol: str, fast: int, slow: int) -> None:
    """
    Create comprehensive plots using the plotting module.
    
    Args:
        results: Backtest results dictionary
        output_dir: Directory to save plots
        symbol: Stock symbol for title
        fast: Fast window period
        slow: Slow window period
    """
    equity_curve_df = results["equity_curve"]
    equity_curve = equity_curve_df['total_equity']
    trades = results["trades"]
    
    # Create title with performance metrics
    metrics = results["metrics"]
    cagr = metrics['cagr'] * 100
    sharpe = metrics['sharpe']
    max_dd_pct = metrics['max_drawdown'] * 100
    hit_rate = metrics['hit_rate'] * 100
    
    title_base = f"{symbol} SMA Crossover Strategy ({fast}/{slow})"
    metrics_text = f"CAGR: {cagr:.1f}% | Sharpe: {sharpe:.2f} | Max DD: {max_dd_pct:.1f}% | Hit Rate: {hit_rate:.1f}%"
    
    # Create equity curve plot
    equity_title = f"{title_base} - {metrics_text}"
    equity_path = create_equity_plot(equity_curve, trades=trades, 
                                   title=equity_title, output_dir=output_dir)
    print(f"📊 Equity curve saved to: {equity_path}")
    
    # Create drawdown plot
    drawdown_title = f"{title_base} - Drawdown Analysis"
    drawdown_path = create_drawdown_plot(equity_curve, 
                                       title=drawdown_title, output_dir=output_dir)
    print(f"📉 Drawdown analysis saved to: {drawdown_path}")
    
    # Create price signals plot if we have the required data
    if 'price_data' in results and 'signals' in results:
        price_df = results['price_data']
        signals = results['signals']
        
        # Calculate moving averages for the plot
        from .indicators import sma
        fast_ma = sma(price_df, window=fast, col='close')
        slow_ma = sma(price_df, window=slow, col='close')
        
        price_title = f"{title_base} - Price Chart with Signals"
        price_path = create_price_signals_plot(price_df, signals, fast_ma, slow_ma,
                                             title=price_title, output_dir=output_dir)
        print(f"📈 Price chart saved to: {price_path}")


def run_backtest_command(args) -> int:
    """Run the backtest command."""
    try:
        # Create strategy parameters
        params = StrategyParams(
            symbol=args.symbol.upper(),
            start=args.start,
            end=args.end,
            fast_window=args.fast,
            slow_window=args.slow,
            initial_cash=args.cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps
        )
        
        # Suppress progress messages if quiet
        if args.quiet:
            import sys
            from contextlib import redirect_stdout
            with redirect_stdout(open(os.devnull, 'w')):
                results = run_crossover_backtest(params)
        else:
            results = run_crossover_backtest(params)
        
        # Print results
        if args.quick:
            print_quick_summary(results)
        else:
            print_backtest_report(results, f"{args.symbol.upper()} Backtest Results")
        
        # Generate plots if requested
        if args.plot:
            # Create output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Create comprehensive plots
            create_comprehensive_plot(results, str(output_dir), args.symbol.upper(), args.fast, args.slow)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Backtest interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


def run_optimize_command(args) -> int:
    """Run the optimize command."""
    try:
        # Parse fast and slow values
        fast_values = [int(x.strip()) for x in args.fast.split(',')]
        slow_values = [int(x.strip()) for x in args.slow.split(',')]
        
        # Run grid search optimization
        results_df = grid_search(
            symbol=args.symbol.upper(),
            start=args.start,
            end=args.end,
            fast_grid=fast_values,
            slow_grid=slow_values,
            metric=args.metric,
            initial_cash=args.cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            n_jobs=args.jobs,
            verbose=not args.quiet
        )
        
        # Print summary
        print_optimization_summary(results_df, args.metric, args.top_n)
        
        # Save results to CSV
        csv_path = save_optimization_results(results_df, args.symbol.upper(), args.output_dir)
        print(f"\n💾 Results saved to: {csv_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Optimization interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running optimization: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


def run_walkforward_command(args) -> int:
    """Run the walkforward command."""
    try:
        # Parse fast and slow values
        fast_values = [int(x.strip()) for x in args.fast.split(',')]
        slow_values = [int(x.strip()) for x in args.slow.split(',')]
        
        # Run walk-forward analysis
        results = walkforward_crossover(
            symbol=args.symbol.upper(),
            start=args.start,
            end=args.end,
            in_sample_years=getattr(args, 'is', 3),
            out_sample_years=getattr(args, 'oos', 1),
            fast_grid=fast_values,
            slow_grid=slow_values,
            initial_cash=args.cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            optimization_metric=args.metric,
            n_jobs=args.jobs,
            verbose=not args.quiet
        )
        
        # Print summary
        print_walkforward_summary(results, args.top_n)
        
        # Save results
        csv_path = save_walkforward_results(results, args.symbol.upper(), args.output_dir)
        print(f"\n💾 Results saved to: {csv_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Walk-forward analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running walk-forward analysis: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        return 1
    
    # Validate arguments
    validate_arguments(args)
    
    # Route to appropriate command
    if args.command == 'run':
        return run_backtest_command(args)
    elif args.command == 'optimize':
        return run_optimize_command(args)
    elif args.command == 'walkforward':
        return run_walkforward_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())