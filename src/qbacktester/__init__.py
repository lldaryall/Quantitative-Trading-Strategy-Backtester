"""
qbacktester: A quantitative backtesting library for financial strategies.

This library provides tools for backtesting trading strategies, analyzing performance,
and generating reports for quantitative finance research.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import Backtester as CoreBacktester, Strategy
from .data import DataProvider, get_price_data
from .portfolio import Portfolio
from .indicators import sma, ema, crossover, rsi, bollinger_bands, macd
from .strategy import StrategyParams, generate_signals, backtest_strategy, get_strategy_metrics
from .backtester import Backtester
from .metrics import (
    daily_returns, cagr, sharpe, max_drawdown, calmar, hit_rate, avg_win_loss,
    volatility, sortino, var, cvar, get_all_metrics
)
from .run import (
    run_crossover_backtest, print_backtest_report, print_quick_summary,
    run_multiple_backtests
)
from .plotting import (
    plot_equity, plot_drawdown, plot_price_signals,
    create_equity_plot, create_drawdown_plot, create_price_signals_plot,
    save_figure
)
from .optimize import (
    grid_search, optimize_strategy, save_optimization_results,
    print_optimization_summary
)
from .walkforward import (
    walkforward_crossover, print_walkforward_summary, save_walkforward_results
)

__all__ = [
    "Backtester",
    "CoreBacktester",
    "Strategy", 
    "DataProvider",
    "get_price_data",
    "Portfolio",
    "sma",
    "ema", 
    "crossover",
    "rsi",
    "bollinger_bands",
    "macd",
    "StrategyParams",
    "generate_signals",
    "backtest_strategy",
    "get_strategy_metrics",
    "daily_returns",
    "cagr",
    "sharpe",
    "max_drawdown",
    "calmar",
    "hit_rate",
    "avg_win_loss",
    "volatility",
    "sortino",
    "var",
    "cvar",
    "get_all_metrics",
    "run_crossover_backtest",
    "print_backtest_report",
    "print_quick_summary",
    "run_multiple_backtests",
    "plot_equity",
    "plot_drawdown",
    "plot_price_signals",
    "create_equity_plot",
    "create_drawdown_plot",
    "create_price_signals_plot",
    "save_figure",
    "grid_search",
    "optimize_strategy",
    "save_optimization_results",
    "print_optimization_summary",
    "walkforward_crossover",
    "print_walkforward_summary",
    "save_walkforward_results",
]
