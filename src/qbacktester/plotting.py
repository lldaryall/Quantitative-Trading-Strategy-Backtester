"""
Professional plotting utilities for backtesting results.

This module provides clean, publication-ready visualizations for:
- Equity curves with trade markers
- Drawdown analysis
- Price charts with technical indicators and signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional
import os


def plot_equity(equity_curve: pd.Series, title: str = "Portfolio Equity Curve") -> Figure:
    """
    Create a professional equity curve plot.
    
    Args:
        equity_curve: Series of portfolio values over time
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Handle empty data
    if equity_curve.empty:
        raise ValueError("equity_curve cannot be empty")
    
    # Create figure with clean aesthetics
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve.values, 
            linewidth=2, color='#2E86AB', label='Portfolio Value')
    
    # Add initial value reference line
    initial_value = equity_curve.iloc[0]
    ax.axhline(y=initial_value, color='gray', linestyle='--', 
               alpha=0.7, label=f'Initial Value (${initial_value:,.0f})')
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_drawdown(equity_curve: pd.Series, title: str = "Portfolio Drawdown") -> Figure:
    """
    Create a professional drawdown analysis plot.
    
    Args:
        equity_curve: Series of portfolio values over time
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Handle empty data
    if equity_curve.empty:
        raise ValueError("equity_curve cannot be empty")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate drawdown
    running_max = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - running_max) / running_max * 100
    
    # Fill area under drawdown curve
    ax.fill_between(equity_curve.index, drawdown.values, 0, 
                    color='red', alpha=0.3, label='Drawdown')
    
    # Plot drawdown line
    ax.plot(equity_curve.index, drawdown.values, 
            linewidth=1, color='darkred', alpha=0.8)
    
    # Add max drawdown line
    max_dd = drawdown.min()
    ax.axhline(y=max_dd, color='darkred', linestyle='--', 
               alpha=0.8, linewidth=2, 
               label=f'Max Drawdown: {max_dd:.1f}%')
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
    
    # Set y-axis to show negative values properly
    ax.set_ylim(bottom=min(drawdown.min() * 1.1, -0.5))
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_price_signals(price_df: pd.DataFrame, signals: pd.Series, 
                      fast: pd.Series, slow: pd.Series, 
                      title: str = "Price Chart with Signals") -> Figure:
    """
    Create a professional price chart with technical indicators and signals.
    
    Args:
        price_df: DataFrame with OHLCV data (must have 'close' column)
        signals: Series of trading signals (1 for long, 0 for flat)
        fast: Series of fast moving average values
        slow: Series of slow moving average values
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Handle empty data
    if price_df.empty or signals.empty or fast.empty or slow.empty:
        raise ValueError("Input data cannot be empty")
    
    if 'close' not in price_df.columns:
        raise ValueError("price_df must contain 'close' column")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Price and Moving Averages
    close_prices = price_df['close']
    
    # Plot price
    ax1.plot(close_prices.index, close_prices.values, 
             linewidth=1.5, color='black', label='Close Price', alpha=0.8)
    
    # Plot moving averages
    ax1.plot(fast.index, fast.values, 
             linewidth=2, color='#FF6B35', label='Fast MA', alpha=0.8)
    ax1.plot(slow.index, slow.values, 
             linewidth=2, color='#2E86AB', label='Slow MA', alpha=0.8)
    
    # Add buy/sell signals
    signal_changes = signals.diff()
    buy_signals = signal_changes[signal_changes == 1]
    sell_signals = signal_changes[signal_changes == -1]
    
    if len(buy_signals) > 0:
        buy_prices = close_prices.loc[buy_signals.index]
        ax1.scatter(buy_signals.index, buy_prices.values, 
                   color='green', marker='^', s=100, 
                   label='Buy Signal', zorder=5, alpha=0.8)
    
    if len(sell_signals) > 0:
        sell_prices = close_prices.loc[sell_signals.index]
        ax1.scatter(sell_signals.index, sell_prices.values, 
                   color='red', marker='v', s=100, 
                   label='Sell Signal', zorder=5, alpha=0.8)
    
    # Formatting for price chart
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Plot 2: Signals
    # Create signal visualization
    signal_colors = ['red' if s == 0 else 'green' for s in signals]
    ax2.fill_between(signals.index, 0, signals.values, 
                     color=signal_colors, alpha=0.3, step='pre')
    ax2.plot(signals.index, signals.values, 
             linewidth=1, color='black', drawstyle='steps-pre')
    
    # Formatting for signals chart
    ax2.set_ylabel('Position', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Flat', 'Long'])
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Format x-axis dates for both subplots
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def save_figure(fig: Figure, filename: str, output_dir: str = "reports") -> str:
    """
    Save a matplotlib figure to a file.
    
    Args:
        fig: matplotlib Figure object
        filename: Name of the file (without extension)
        output_dir: Directory to save the file
        
    Returns:
        Full path to the saved file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate full file path
    file_path = output_path / f"{filename}.png"
    
    # Save figure
    fig.savefig(file_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    return str(file_path)


def create_equity_plot(equity_curve: pd.Series, trades: Optional[pd.DataFrame] = None,
                      title: str = "Portfolio Performance", 
                      output_dir: str = "reports") -> str:
    """
    Create and save a comprehensive equity curve plot with optional trade markers.
    
    Args:
        equity_curve: Series of portfolio values over time
        trades: Optional DataFrame with trade information
        title: Plot title
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve.values, 
            linewidth=2, color='#2E86AB', label='Portfolio Value')
    
    # Add initial value reference line
    initial_value = equity_curve.iloc[0]
    ax.axhline(y=initial_value, color='gray', linestyle='--', 
               alpha=0.7, label=f'Initial Value (${initial_value:,.0f})')
    
    # Add trade markers if provided
    if trades is not None and len(trades) > 0:
        entry_trades = trades[trades['side'] == 'entry']
        exit_trades = trades[trades['side'] == 'exit']
        
        if len(entry_trades) > 0:
            ax.scatter(entry_trades['timestamp'], entry_trades['equity'], 
                      color='green', marker='^', s=100, 
                      label='Buy', zorder=5, alpha=0.8)
        
        if len(exit_trades) > 0:
            ax.scatter(exit_trades['timestamp'], exit_trades['equity'], 
                      color='red', marker='v', s=100, 
                      label='Sell', zorder=5, alpha=0.8)
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure with safe filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title.replace(' ', '_').lower()
    filename = f"equity_{safe_title}"
    file_path = save_figure(fig, filename, output_dir)
    
    # Close figure to free memory
    plt.close(fig)
    
    return file_path


def create_drawdown_plot(equity_curve: pd.Series, 
                        title: str = "Portfolio Drawdown",
                        output_dir: str = "reports") -> str:
    """
    Create and save a drawdown analysis plot.
    
    Args:
        equity_curve: Series of portfolio values over time
        title: Plot title
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate drawdown
    running_max = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - running_max) / running_max * 100
    
    # Fill area under drawdown curve
    ax.fill_between(equity_curve.index, drawdown.values, 0, 
                    color='red', alpha=0.3, label='Drawdown')
    
    # Plot drawdown line
    ax.plot(equity_curve.index, drawdown.values, 
            linewidth=1, color='darkred', alpha=0.8)
    
    # Add max drawdown line
    max_dd = drawdown.min()
    ax.axhline(y=max_dd, color='darkred', linestyle='--', 
               alpha=0.8, linewidth=2, 
               label=f'Max Drawdown: {max_dd:.1f}%')
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
    
    # Set y-axis to show negative values properly
    ax.set_ylim(bottom=min(drawdown.min() * 1.1, -0.5))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure with safe filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title.replace(' ', '_').lower()
    filename = f"drawdown_{safe_title}"
    file_path = save_figure(fig, filename, output_dir)
    
    # Close figure to free memory
    plt.close(fig)
    
    return file_path


def create_price_signals_plot(price_df: pd.DataFrame, signals: pd.Series,
                             fast: pd.Series, slow: pd.Series,
                             title: str = "Price Chart with Signals",
                             output_dir: str = "reports") -> str:
    """
    Create and save a price chart with technical indicators and signals.
    
    Args:
        price_df: DataFrame with OHLCV data
        signals: Series of trading signals
        fast: Series of fast moving average values
        slow: Series of slow moving average values
        title: Plot title
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Price and Moving Averages
    close_prices = price_df['close']
    
    # Plot price
    ax1.plot(close_prices.index, close_prices.values, 
             linewidth=1.5, color='black', label='Close Price', alpha=0.8)
    
    # Plot moving averages
    ax1.plot(fast.index, fast.values, 
             linewidth=2, color='#FF6B35', label='Fast MA', alpha=0.8)
    ax1.plot(slow.index, slow.values, 
             linewidth=2, color='#2E86AB', label='Slow MA', alpha=0.8)
    
    # Add buy/sell signals
    signal_changes = signals.diff()
    buy_signals = signal_changes[signal_changes == 1]
    sell_signals = signal_changes[signal_changes == -1]
    
    if len(buy_signals) > 0:
        buy_prices = close_prices.loc[buy_signals.index]
        ax1.scatter(buy_signals.index, buy_prices.values, 
                   color='green', marker='^', s=100, 
                   label='Buy Signal', zorder=5, alpha=0.8)
    
    if len(sell_signals) > 0:
        sell_prices = close_prices.loc[sell_signals.index]
        ax1.scatter(sell_signals.index, sell_prices.values, 
                   color='red', marker='v', s=100, 
                   label='Sell Signal', zorder=5, alpha=0.8)
    
    # Formatting for price chart
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Plot 2: Signals
    # Create signal visualization
    signal_colors = ['red' if s == 0 else 'green' for s in signals]
    ax2.fill_between(signals.index, 0, signals.values, 
                     color=signal_colors, alpha=0.3, step='pre')
    ax2.plot(signals.index, signals.values, 
             linewidth=1, color='black', drawstyle='steps-pre')
    
    # Formatting for signals chart
    ax2.set_ylabel('Position', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Flat', 'Long'])
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Format x-axis dates for both subplots
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure with safe filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title.replace(' ', '_').lower()
    filename = f"price_signals_{safe_title}"
    file_path = save_figure(fig, filename, output_dir)
    
    # Close figure to free memory
    plt.close(fig)
    
    return file_path
