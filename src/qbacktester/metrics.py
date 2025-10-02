"""
Performance metrics for backtesting analysis.
All functions are vectorized and numerically stable.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd


def daily_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate daily returns from equity curve.

    Args:
        equity_curve: Series of equity values (e.g., portfolio values)

    Returns:
        Series of daily returns (percentage changes)

    Raises:
        ValueError: If equity_curve is empty or contains non-positive values
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve cannot be empty")

    if (equity_curve <= 0).any():
        raise ValueError("equity_curve must contain only positive values")

    # Calculate percentage change, handling potential division by zero
    returns = equity_curve.pct_change()

    # First value will be NaN due to pct_change, which is expected
    return returns


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).

    Args:
        equity_curve: Series of equity values
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        CAGR as a decimal (e.g., 0.1 for 10%)

    Raises:
        ValueError: If equity_curve is empty or contains non-positive values
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve cannot be empty")

    if (equity_curve <= 0).any():
        raise ValueError("equity_curve must contain only positive values")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    # Get start and end values
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]

    # Calculate number of periods
    n_periods = len(equity_curve) - 1

    if n_periods == 0:
        return 0.0

    # Calculate CAGR: (end_value / start_value)^(periods_per_year / n_periods) - 1
    # Use log for numerical stability
    years = n_periods / periods_per_year
    cagr_value = (end_value / start_value) ** (1 / years) - 1

    return cagr_value


def sharpe(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free: Risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        Sharpe ratio

    Raises:
        ValueError: If returns is empty or all values are NaN
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        raise ValueError("returns must contain at least one non-NaN value")

    # Calculate excess returns
    excess_returns = clean_returns - risk_free / periods_per_year

    # Calculate mean and standard deviation
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()

    # Handle case where standard deviation is zero
    if std_return == 0:
        return 0.0 if mean_return == 0 else np.inf if mean_return > 0 else -np.inf

    # Annualize the Sharpe ratio
    sharpe_ratio = mean_return / std_return * np.sqrt(periods_per_year)

    return sharpe_ratio


def max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and its dates.

    Args:
        equity_curve: Series of equity values

    Returns:
        Tuple of (max_drawdown_magnitude, peak_date, trough_date)
        Max drawdown is negative (e.g., -0.1 for 10% drawdown)

    Raises:
        ValueError: If equity_curve is empty
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve cannot be empty")

    # Calculate running maximum (peak)
    running_max = equity_curve.expanding().max()

    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max

    # Find maximum drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_value = drawdown.min()

    # Find the peak date (last occurrence of the peak value before max drawdown)
    peak_value = running_max.loc[max_dd_idx]
    peak_candidates = equity_curve[equity_curve == peak_value]
    peak_candidates_before = peak_candidates[peak_candidates.index <= max_dd_idx]

    if len(peak_candidates_before) > 0:
        peak_date = peak_candidates_before.index[-1]
    else:
        peak_date = equity_curve.index[0]

    return max_dd_value, peak_date, max_dd_idx


def calmar(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (CAGR / |Max Drawdown|).

    Args:
        equity_curve: Series of equity values
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        Calmar ratio

    Raises:
        ValueError: If equity_curve is empty or contains non-positive values
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve cannot be empty")

    if (equity_curve <= 0).any():
        raise ValueError("equity_curve must contain only positive values")

    # Calculate CAGR
    cagr_value = cagr(equity_curve, periods_per_year)

    # Calculate max drawdown
    max_dd_value, _, _ = max_drawdown(equity_curve)

    # Handle case where max drawdown is zero
    if max_dd_value == 0:
        return np.inf if cagr_value > 0 else 0.0

    # Calmar ratio = CAGR / |Max Drawdown|
    calmar_ratio = cagr_value / abs(max_dd_value)

    return calmar_ratio


def hit_rate(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate hit rate (percentage of positive returns above threshold).

    Args:
        returns: Series of returns
        threshold: Minimum return threshold (default: 0.0)

    Returns:
        Hit rate as a decimal (e.g., 0.6 for 60%)

    Raises:
        ValueError: If returns is empty or all values are NaN
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        raise ValueError("returns must contain at least one non-NaN value")

    # Calculate hit rate
    hits = (clean_returns > threshold).sum()
    total = len(clean_returns)

    return hits / total


def avg_win_loss(returns: pd.Series) -> Tuple[float, float]:
    """
    Calculate average win and average loss.

    Args:
        returns: Series of returns

    Returns:
        Tuple of (avg_win, avg_loss)
        Returns 0.0 for avg_win if no wins, 0.0 for avg_loss if no losses

    Raises:
        ValueError: If returns is empty or all values are NaN
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        raise ValueError("returns must contain at least one non-NaN value")

    # Separate wins and losses
    wins = clean_returns[clean_returns > 0]
    losses = clean_returns[clean_returns < 0]

    # Calculate averages
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    return avg_win, avg_loss


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        Annualized volatility as a decimal

    Raises:
        ValueError: If returns is empty or all values are NaN
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        raise ValueError("returns must contain at least one non-NaN value")

    # Calculate standard deviation and annualize
    vol = clean_returns.std() * np.sqrt(periods_per_year)

    return vol


def sortino(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (downside deviation version of Sharpe).

    Args:
        returns: Series of returns
        risk_free: Risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        Sortino ratio

    Raises:
        ValueError: If returns is empty or all values are NaN
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        raise ValueError("returns must contain at least one non-NaN value")

    # Calculate excess returns
    excess_returns = clean_returns - risk_free / periods_per_year

    # Calculate mean return
    mean_return = excess_returns.mean()

    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf if mean_return > 0 else 0.0

    downside_dev = downside_returns.std()

    if downside_dev == 0:
        return 0.0 if mean_return == 0 else np.inf if mean_return > 0 else -np.inf

    # Annualize the Sortino ratio
    sortino_ratio = mean_return / downside_dev * np.sqrt(periods_per_year)

    return sortino_ratio


def var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Series of returns
        confidence_level: Confidence level (default: 0.05 for 5% VaR)

    Returns:
        VaR as a decimal (negative value)

    Raises:
        ValueError: If returns is empty or all values are NaN
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        raise ValueError("returns must contain at least one non-NaN value")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Calculate VaR as the percentile
    var_value = np.percentile(clean_returns, confidence_level * 100)

    return var_value


def cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) or Expected Shortfall.

    Args:
        returns: Series of returns
        confidence_level: Confidence level (default: 0.05 for 5% CVaR)

    Returns:
        CVaR as a decimal (negative value)

    Raises:
        ValueError: If returns is empty or all values are NaN
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        raise ValueError("returns must contain at least one non-NaN value")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Calculate VaR threshold
    var_threshold = var(clean_returns, confidence_level)

    # Calculate CVaR as the mean of returns below VaR threshold
    tail_returns = clean_returns[clean_returns <= var_threshold]

    if len(tail_returns) == 0:
        return var_threshold

    cvar_value = tail_returns.mean()

    return cvar_value


def get_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Calculate all performance metrics in one call.

    Args:
        equity_curve: Series of equity values
        returns: Series of returns (if None, calculated from equity_curve)
        risk_free: Risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Dictionary with all performance metrics
    """
    if returns is None:
        returns = daily_returns(equity_curve)

    metrics = {
        "cagr": cagr(equity_curve, periods_per_year),
        "sharpe": sharpe(returns, risk_free, periods_per_year),
        "sortino": sortino(returns, risk_free, periods_per_year),
        "volatility": volatility(returns, periods_per_year),
        "calmar": calmar(equity_curve, periods_per_year),
        "hit_rate": hit_rate(returns),
        "avg_win": avg_win_loss(returns)[0],
        "avg_loss": avg_win_loss(returns)[1],
        "var_5pct": var(returns, 0.05),
        "cvar_5pct": cvar(returns, 0.05),
    }

    # Add max drawdown info
    max_dd, peak_date, trough_date = max_drawdown(equity_curve)
    metrics.update(
        {
            "max_drawdown": max_dd,
            "max_drawdown_peak_date": peak_date,
            "max_drawdown_trough_date": trough_date,
        }
    )

    return metrics
