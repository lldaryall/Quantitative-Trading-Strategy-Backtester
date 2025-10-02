"""
Data handling and providers with caching support.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataError(Exception):
    """Custom exception for data-related errors."""

    pass


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch market data for a given symbol and date range."""
        pass


def get_price_data(
    symbol: str, start: str, end: str, interval: str = "1d"
) -> pd.DataFrame:
    """
    Get price data with local caching support.

    Downloads OHLCV data from Yahoo Finance with automatic caching to parquet files.
    If cached data exists, loads from disk. Otherwise downloads, caches, and returns data.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        interval: Data interval ('1d', '1h', '5m', etc.)

    Returns:
        DataFrame with OHLCV data, sorted by date

    Raises:
        DataError: If data is empty, dates are invalid, or download fails
        ValueError: If symbol is invalid or parameters are malformed
    """
    # Validate inputs
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")

    if not start or not end:
        raise ValueError("Start and end dates must be provided")

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate cache filename
    cache_file = data_dir / f"{symbol}_{interval}_{start}_{end}.parquet"

    # Check if cached data exists
    if cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        try:
            data = pd.read_parquet(cache_file)
            logger.info(f"Loaded {len(data)} records from cache")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}. Re-downloading...")

    # Download data with retry logic
    data = _download_with_retry(symbol, start, end, interval)

    # Validate and process data
    data = _validate_and_process_data(data, symbol, start, end)

    # Cache the data
    try:
        data.to_parquet(cache_file)
        logger.info(f"Cached data to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to cache data: {e}")

    return data


def _download_with_retry(
    symbol: str, start: str, end: str, interval: str, max_retries: int = 3
) -> pd.DataFrame:
    """
    Download data with exponential backoff retry logic.

    Args:
        symbol: Stock symbol
        start: Start date
        end: End date
        interval: Data interval
        max_retries: Maximum number of retry attempts

    Returns:
        Downloaded DataFrame

    Raises:
        DataError: If all retry attempts fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Downloading {symbol} data (attempt {attempt + 1}/{max_retries})"
            )

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start, end=end, interval=interval)

            if data.empty:
                raise DataError(
                    f"No data returned for symbol '{symbol}' in date range {start} to {end}"
                )

            return data

        except Exception as e:
            last_exception = e
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2**attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    # All retries failed
    raise DataError(
        f"Failed to download data for '{symbol}' after {max_retries} attempts. "
        f"Last error: {last_exception}"
    )


def _validate_and_process_data(
    data: pd.DataFrame, symbol: str, start: str, end: str
) -> pd.DataFrame:
    """
    Validate and process downloaded data.

    Args:
        data: Raw downloaded data
        symbol: Stock symbol for error messages
        start: Start date for validation
        end: End date for validation

    Returns:
        Processed and validated DataFrame

    Raises:
        DataError: If data validation fails
    """
    if data.empty:
        raise DataError(
            f"No data available for symbol '{symbol}' in date range {start} to {end}"
        )

    # Check for required columns
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise DataError(
            f"Missing required columns for '{symbol}': {missing_columns}. "
            f"Available columns: {list(data.columns)}"
        )

    # Convert column names to lowercase for consistency
    data.columns = data.columns.str.lower()

    # Sort by date
    data = data.sort_index()

    # Drop rows with all NaN values
    data = data.dropna(how="all")

    if data.empty:
        raise DataError(f"All data for '{symbol}' is NaN after processing")

    # Check for reasonable data ranges
    if (data["close"] <= 0).any():
        logger.warning(f"Found non-positive close prices for '{symbol}'")

    if (data["volume"] < 0).any():
        logger.warning(f"Found negative volume for '{symbol}'")

    logger.info(f"Successfully processed {len(data)} records for '{symbol}'")
    return data


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider with caching support."""

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance with caching.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        return get_price_data(symbol, start_date, end_date, interval)
