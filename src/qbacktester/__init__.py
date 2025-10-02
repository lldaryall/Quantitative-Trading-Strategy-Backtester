"""
qbacktester: A quantitative backtesting library for financial strategies.

This library provides tools for backtesting trading strategies, analyzing performance,
and generating reports for quantitative finance research.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .backtester import Backtester
from .core import Backtester as CoreBacktester
from .core import Strategy
from .data import DataProvider, get_price_data
from .indicators import bollinger_bands, crossover, ema, macd, rsi, sma
from .metrics import (
    avg_win_loss,
    cagr,
    calmar,
    cvar,
    daily_returns,
    get_all_metrics,
    hit_rate,
    max_drawdown,
    sharpe,
    sortino,
    var,
    volatility,
)
from .optimize import (
    grid_search,
    optimize_strategy,
    print_optimization_summary,
    save_optimization_results,
)
from .plotting import (
    create_drawdown_plot,
    create_equity_plot,
    create_price_signals_plot,
    plot_drawdown,
    plot_equity,
    plot_price_signals,
    save_figure,
)
from .portfolio import Portfolio
from .run import (
    print_backtest_report,
    print_quick_summary,
    run_crossover_backtest,
    run_multiple_backtests,
)
from .strategy import (
    StrategyParams,
    backtest_strategy,
    generate_signals,
    get_strategy_metrics,
)
from .walkforward import (
    print_walkforward_summary,
    save_walkforward_results,
    walkforward_crossover,
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
