"""
Main backtesting pipeline and reporting functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .strategy import StrategyParams, generate_signals
from .backtester import Backtester
from .data import get_price_data
from .metrics import get_all_metrics, daily_returns


def run_crossover_backtest(params: StrategyParams) -> Dict[str, Any]:
    """
    Run a complete crossover backtest pipeline.
    
    Args:
        params: Strategy parameters including symbol, dates, windows, and costs
        
    Returns:
        Dictionary containing:
        - "params": Strategy parameters
        - "equity_curve": DataFrame with equity curve data
        - "metrics": Dictionary of performance metrics
        - "trades": DataFrame of trade details
    """
    # Pull data
    print(f"📊 Fetching data for {params.symbol} from {params.start} to {params.end}...")
    price_df = get_price_data(params.symbol, params.start, params.end)
    
    # Build signals
    print(f"🔍 Generating signals with {params.fast_window}/{params.slow_window} SMA crossover...")
    signals_df = generate_signals(price_df, params.fast_window, params.slow_window)
    signals = signals_df['signal']
    
    # Run Backtester
    print(f"⚡ Running backtest with {params.initial_cash:,.0f} initial capital...")
    backtester = Backtester(price_df, signals, params)
    result_df = backtester.run()
    
    # Compute metrics
    print("📈 Calculating performance metrics...")
    equity_curve = result_df['total_equity']
    returns = daily_returns(equity_curve)
    
    # Handle case where there are no valid returns (e.g., single day data)
    if returns.dropna().empty:
        # Create minimal metrics for single day or no valid returns
        metrics = {
            'cagr': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'volatility': 0.0,
            'calmar': 0.0,
            'hit_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'var_5pct': 0.0,
            'cvar_5pct': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_peak_date': equity_curve.index[0],
            'max_drawdown_trough_date': equity_curve.index[0],
        }
    else:
        metrics = get_all_metrics(equity_curve, returns)
    
    # Extract trade details
    trades = _extract_trades(result_df, price_df)
    
    print("✅ Backtest completed successfully!")
    
    return {
        "params": params,
        "equity_curve": result_df,
        "metrics": metrics,
        "trades": trades
    }


def _extract_trades(result_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract trade details from backtest results.
    
    Args:
        result_df: Backtest results DataFrame
        price_df: Original price DataFrame
        
    Returns:
        DataFrame with trade details (timestamp, side, price)
    """
    # Find all trade points
    trade_points = result_df[result_df['trade_flag']].copy()
    
    if len(trade_points) == 0:
        return pd.DataFrame(columns=['timestamp', 'side', 'price', 'equity'])
    
    # Determine trade side based on position change
    position_changes = result_df['signal'].diff()
    trade_changes = position_changes[result_df['trade_flag']]
    
    trades = pd.DataFrame({
        'timestamp': trade_points.index,
        'side': ['entry' if change == 1 else 'exit' for change in trade_changes],
        'price': trade_points['close'],
        'equity': trade_points['total_equity']
    })
    
    return trades


def print_backtest_report(results: Dict[str, Any], title: str = "Backtest Results") -> None:
    """
    Print a beautiful backtest report using Rich.
    
    Args:
        results: Results dictionary from run_crossover_backtest
        title: Title for the report
    """
    console = Console()
    
    # Extract data
    params = results["params"]
    metrics = results["metrics"]
    trades = results["trades"]
    equity_curve_df = results["equity_curve"]
    
    # Extract equity curve as Series
    if isinstance(equity_curve_df, pd.DataFrame):
        equity_curve = equity_curve_df['total_equity']
    else:
        equity_curve = equity_curve_df
    
    # Print header
    console.print(f"\n[bold blue]{title}[/bold blue]")
    console.print("=" * len(title))
    
    # Strategy parameters panel
    params_table = Table(title="Strategy Parameters", box=box.ROUNDED)
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="white")
    
    params_table.add_row("Symbol", params.symbol)
    params_table.add_row("Start Date", params.start)
    params_table.add_row("End Date", params.end)
    params_table.add_row("Fast Window", str(params.fast_window))
    params_table.add_row("Slow Window", str(params.slow_window))
    params_table.add_row("Initial Cash", f"${params.initial_cash:,.0f}")
    params_table.add_row("Fee (bps)", f"{params.fee_bps:.1f}")
    params_table.add_row("Slippage (bps)", f"{params.slippage_bps:.1f}")
    
    console.print(params_table)
    
    # Performance metrics panel
    metrics_table = Table(title="Performance Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="white")
    metrics_table.add_column("Format", style="dim")
    
    # Format metrics with appropriate styling
    def format_percentage(value: float) -> str:
        return f"{value:.2%}"
    
    def format_number(value: float) -> str:
        return f"{value:,.2f}"
    
    def format_currency(value: float) -> str:
        return f"${value:,.0f}"
    
    # Add key metrics
    metrics_table.add_row("CAGR", format_percentage(metrics['cagr']), "Annual return")
    metrics_table.add_row("Sharpe Ratio", format_number(metrics['sharpe']), "Risk-adjusted return")
    metrics_table.add_row("Sortino Ratio", format_number(metrics['sortino']), "Downside risk-adjusted")
    metrics_table.add_row("Volatility", format_percentage(metrics['volatility']), "Annual volatility")
    metrics_table.add_row("Max Drawdown", format_percentage(metrics['max_drawdown']), "Largest loss")
    metrics_table.add_row("Calmar Ratio", format_number(metrics['calmar']), "CAGR / |Max DD|")
    metrics_table.add_row("Hit Rate", format_percentage(metrics['hit_rate']), "Win percentage")
    metrics_table.add_row("Avg Win", format_percentage(metrics['avg_win']), "Average winning return")
    metrics_table.add_row("Avg Loss", format_percentage(metrics['avg_loss']), "Average losing return")
    metrics_table.add_row("VaR (5%)", format_percentage(metrics['var_5pct']), "5% Value at Risk")
    metrics_table.add_row("CVaR (5%)", format_percentage(metrics['cvar_5pct']), "5% Conditional VaR")
    
    console.print(metrics_table)
    
    # Trade summary
    if len(trades) > 0:
        trade_summary = Table(title="Trade Summary", box=box.ROUNDED)
        trade_summary.add_column("Metric", style="cyan")
        trade_summary.add_column("Value", style="white")
        
        total_trades = len(trades)
        entry_trades = len(trades[trades['side'] == 'entry'])
        exit_trades = len(trades[trades['side'] == 'exit'])
        
        trade_summary.add_row("Total Trades", str(total_trades))
        trade_summary.add_row("Entry Trades", str(entry_trades))
        trade_summary.add_row("Exit Trades", str(exit_trades))
        trade_summary.add_row("Round Trips", str(min(entry_trades, exit_trades)))
        
        console.print(trade_summary)
        
        # Recent trades
        if len(trades) <= 10:
            recent_trades = trades
        else:
            recent_trades = trades.tail(10)
        
        trades_table = Table(title="Recent Trades", box=box.ROUNDED)
        trades_table.add_column("Timestamp", style="cyan")
        trades_table.add_column("Side", style="yellow")
        trades_table.add_column("Price", style="white")
        trades_table.add_column("Equity", style="green")
        
        for _, trade in recent_trades.iterrows():
            side_color = "green" if trade['side'] == 'entry' else "red"
            trades_table.add_row(
                trade['timestamp'].strftime("%Y-%m-%d %H:%M"),
                f"[{side_color}]{trade['side'].upper()}[/{side_color}]",
                f"${trade['price']:.2f}",
                f"${trade['equity']:,.0f}"
            )
        
        console.print(trades_table)
    else:
        console.print("[yellow]No trades executed during this period.[/yellow]")
    
    # Equity curve summary
    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity / params.initial_cash) - 1
    peak_equity = equity_curve.max()
    current_drawdown = (equity_curve.iloc[-1] / peak_equity) - 1
    
    equity_summary = Table(title="Equity Summary", box=box.ROUNDED)
    equity_summary.add_column("Metric", style="cyan")
    equity_summary.add_column("Value", style="white")
    
    equity_summary.add_row("Initial Equity", format_currency(params.initial_cash))
    equity_summary.add_row("Final Equity", format_currency(final_equity))
    equity_summary.add_row("Total Return", format_percentage(total_return))
    equity_summary.add_row("Peak Equity", format_currency(peak_equity))
    equity_summary.add_row("Current Drawdown", format_percentage(current_drawdown))
    
    console.print(equity_summary)
    
    # Risk warnings
    if metrics['max_drawdown'] < -0.20:
        console.print("\n[red]⚠️  WARNING: High maximum drawdown detected![/red]")
    
    if metrics['sharpe'] < 0:
        console.print("\n[yellow]⚠️  WARNING: Negative Sharpe ratio - strategy underperformed risk-free rate[/yellow]")
    
    if metrics['hit_rate'] < 0.4:
        console.print("\n[yellow]⚠️  WARNING: Low hit rate - strategy has more losing trades than winning ones[/yellow]")
    
    console.print(f"\n[dim]Report generated for {params.symbol} ({params.start} to {params.end})[/dim]")


def print_quick_summary(results: Dict[str, Any]) -> None:
    """
    Print a quick summary of backtest results.
    
    Args:
        results: Results dictionary from run_crossover_backtest
    """
    console = Console()
    
    params = results["params"]
    metrics = results["metrics"]
    trades = results["trades"]
    
    # Quick summary in a single line
    total_trades = len(trades)
    cagr = metrics['cagr']
    sharpe = metrics['sharpe']
    max_dd = metrics['max_drawdown']
    hit_rate = metrics['hit_rate']
    
    summary_text = (
        f"[bold]{params.symbol}[/bold] | "
        f"CAGR: [green]{cagr:.1%}[/green] | "
        f"Sharpe: [blue]{sharpe:.2f}[/blue] | "
        f"Max DD: [red]{max_dd:.1%}[/red] | "
        f"Hit Rate: [yellow]{hit_rate:.1%}[/yellow] | "
        f"Trades: [cyan]{total_trades}[/cyan]"
    )
    
    console.print(summary_text)


def run_multiple_backtests(param_list: list[StrategyParams], 
                          title: str = "Multiple Backtest Results") -> Dict[str, Any]:
    """
    Run multiple backtests and compare results.
    
    Args:
        param_list: List of StrategyParams to test
        title: Title for the comparison report
        
    Returns:
        Dictionary with results for each parameter set
    """
    console = Console()
    console.print(f"\n[bold blue]{title}[/bold blue]")
    console.print("=" * len(title))
    
    all_results = {}
    
    for i, params in enumerate(param_list, 1):
        console.print(f"\n[cyan]Running backtest {i}/{len(param_list)}: {params.symbol}[/cyan]")
        
        try:
            results = run_crossover_backtest(params)
            all_results[f"{params.symbol}_{i}"] = results
            print_quick_summary(results)
        except Exception as e:
            console.print(f"[red]Error running backtest for {params.symbol}: {e}[/red]")
            all_results[f"{params.symbol}_{i}"] = {"error": str(e)}
    
    # Print comparison table
    if len(all_results) > 1:
        _print_comparison_table(all_results)
    
    return all_results


def _print_comparison_table(all_results: Dict[str, Any]) -> None:
    """
    Print a comparison table for multiple backtest results.
    
    Args:
        all_results: Dictionary of backtest results
    """
    console = Console()
    
    comparison_table = Table(title="Backtest Comparison", box=box.ROUNDED)
    comparison_table.add_column("Strategy", style="cyan")
    comparison_table.add_column("CAGR", style="green")
    comparison_table.add_column("Sharpe", style="blue")
    comparison_table.add_column("Max DD", style="red")
    comparison_table.add_column("Hit Rate", style="yellow")
    comparison_table.add_column("Trades", style="white")
    
    for name, results in all_results.items():
        if "error" in results:
            comparison_table.add_row(name, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR")
            continue
        
        metrics = results["metrics"]
        trades = results["trades"]
        
        comparison_table.add_row(
            name,
            f"{metrics['cagr']:.1%}",
            f"{metrics['sharpe']:.2f}",
            f"{metrics['max_drawdown']:.1%}",
            f"{metrics['hit_rate']:.1%}",
            str(len(trades))
        )
    
    console.print(comparison_table)
