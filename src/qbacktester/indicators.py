"""
Technical indicators with vectorized implementations.
"""

from typing import Union

import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, window: int, col: str = "close") -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        df: DataFrame with price data
        window: Number of periods for moving average
        col: Column name to calculate SMA for (default: "close")

    Returns:
        Series with SMA values

    Raises:
        ValueError: If window is not positive or column doesn't exist
    """
    if window <= 0:
        raise ValueError("Window must be positive")

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    return df[col].rolling(window=window, min_periods=1).mean()


def ema(df: pd.DataFrame, span: int, col: str = "close") -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).

    Uses pandas ewm with adjust=False for traditional EMA calculation.

    Args:
        df: DataFrame with price data
        span: Span for EMA calculation (equivalent to 2/(span+1) alpha)
        col: Column name to calculate EMA for (default: "close")

    Returns:
        Series with EMA values

    Raises:
        ValueError: If span is not positive or column doesn't exist
    """
    if span <= 0:
        raise ValueError("Span must be positive")

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    return df[col].ewm(span=span, adjust=False).mean()


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    Detect crossovers between two series.

    Returns 1 when fast crosses above slow, -1 when fast crosses below slow,
    and 0 otherwise. Handles NaN values by propagating them.

    Args:
        fast: Fast moving series
        slow: Slow moving series

    Returns:
        Series with crossover signals: {1, 0, -1, NaN}

    Raises:
        ValueError: If series have different lengths or indices
    """
    if len(fast) != len(slow):
        raise ValueError("Fast and slow series must have the same length")

    if not fast.index.equals(slow.index):
        raise ValueError("Fast and slow series must have the same index")

    # Handle empty series
    if len(fast) == 0:
        return pd.Series([], dtype="float64")

    # Calculate the difference
    diff = fast - slow

    # Shift to get previous period difference
    diff_prev = diff.shift(1)

    # Create crossover signals
    # 1: fast crosses above slow (current diff > 0, previous diff <= 0)
    # -1: fast crosses below slow (current diff < 0, previous diff >= 0)
    # 0: no crossover
    crossover_up = (diff > 0) & (diff_prev <= 0)
    crossover_down = (diff < 0) & (diff_prev >= 0)

    # Combine signals
    signals = pd.Series(0, index=fast.index, dtype="float64")
    signals[crossover_up] = 1
    signals[crossover_down] = -1

    # First value should always be NaN (no previous value to compare)
    if len(signals) > 0:
        signals.iloc[0] = np.nan

    # Handle NaN values - if either series has NaN, result should be NaN
    nan_mask = fast.isna() | slow.isna()
    signals[nan_mask] = np.nan

    return signals


def rsi(df: pd.DataFrame, window: int = 14, col: str = "close") -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        df: DataFrame with price data
        window: Number of periods for RSI calculation (default: 14)
        col: Column name to calculate RSI for (default: "close")

    Returns:
        Series with RSI values (0-100)

    Raises:
        ValueError: If window is not positive or column doesn't exist
    """
    if window <= 0:
        raise ValueError("Window must be positive")

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    # Calculate price changes
    delta = df[col].diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses using EMA
    avg_gains = gains.ewm(span=window, adjust=False).mean()
    avg_losses = losses.ewm(span=window, adjust=False).mean()

    # Calculate RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


def bollinger_bands(
    df: pd.DataFrame, window: int = 20, std_dev: float = 2, col: str = "close"
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        df: DataFrame with price data
        window: Number of periods for moving average (default: 20)
        std_dev: Standard deviation multiplier (default: 2)
        col: Column name to calculate bands for (default: "close")

    Returns:
        DataFrame with 'upper', 'middle', 'lower' columns

    Raises:
        ValueError: If window is not positive or column doesn't exist
    """
    if window <= 0:
        raise ValueError("Window must be positive")

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    # Calculate middle band (SMA)
    middle = sma(df, window, col)

    # Calculate standard deviation
    rolling_std = df[col].rolling(window=window, min_periods=1).std()

    # Calculate upper and lower bands
    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)

    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower}, index=df.index
    )


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        df: DataFrame with price data
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
        col: Column name to calculate MACD for (default: "close")

    Returns:
        DataFrame with 'macd', 'signal', 'histogram' columns

    Raises:
        ValueError: If periods are not positive or column doesn't exist
    """
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("All periods must be positive")

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    # Calculate EMAs
    ema_fast = ema(df, fast, col)
    ema_slow = ema(df, slow, col)

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line (EMA of MACD)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=df.index,
    )
