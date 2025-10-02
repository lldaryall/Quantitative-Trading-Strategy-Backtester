"""Tests for data module."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from qbacktester.data import (
    DataError,
    YahooFinanceProvider,
    _download_with_retry,
    _validate_and_process_data,
    get_price_data,
)


class TestGetPriceData:
    """Test the main get_price_data function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("qbacktester.data._download_with_retry")
    @patch("qbacktester.data.Path")
    def test_get_price_data_downloads_when_no_cache(self, mock_path, mock_download):
        """Test that data is downloaded when no cache exists."""
        # Mock Path to return our temp directory
        mock_data_dir = Mock()
        mock_data_dir.mkdir.return_value = None
        mock_cache_file = Mock()
        mock_cache_file.exists.return_value = False
        mock_data_dir.__truediv__ = Mock(return_value=mock_cache_file)
        mock_path.return_value = mock_data_dir

        # Mock download to return sample data
        sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [102, 103, 104],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )
        mock_download.return_value = sample_data

        result = get_price_data("AAPL", "2023-01-01", "2023-01-03")

        # Verify download was called
        mock_download.assert_called_once_with("AAPL", "2023-01-01", "2023-01-03", "1d")

        # Verify result is processed correctly
        assert len(result) == 3
        assert "close" in result.columns
        assert result.index.is_monotonic_increasing

    @patch("qbacktester.data.pd.read_parquet")
    @patch("qbacktester.data.Path")
    def test_get_price_data_loads_from_cache(self, mock_path, mock_read_parquet):
        """Test that data is loaded from cache when available."""
        # Mock Path to return our temp directory
        mock_data_dir = Mock()
        mock_data_dir.mkdir.return_value = None
        mock_cache_file = Mock()
        mock_cache_file.exists.return_value = True
        mock_data_dir.__truediv__ = Mock(return_value=mock_cache_file)
        mock_path.return_value = mock_data_dir

        # Mock cached data
        cached_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )
        mock_read_parquet.return_value = cached_data

        result = get_price_data("AAPL", "2023-01-01", "2023-01-03")

        # Verify cache was read
        mock_read_parquet.assert_called_once_with(mock_cache_file)

        # Verify result matches cached data
        pd.testing.assert_frame_equal(result, cached_data)

    def test_get_price_data_invalid_symbol(self):
        """Test that invalid symbol raises ValueError."""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            get_price_data("", "2023-01-01", "2023-01-03")

        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            get_price_data(None, "2023-01-01", "2023-01-03")

    def test_get_price_data_missing_dates(self):
        """Test that missing dates raise ValueError."""
        with pytest.raises(ValueError, match="Start and end dates must be provided"):
            get_price_data("AAPL", "", "2023-01-03")

        with pytest.raises(ValueError, match="Start and end dates must be provided"):
            get_price_data("AAPL", "2023-01-01", "")


class TestDownloadWithRetry:
    """Test the retry logic for downloads."""

    @patch("qbacktester.data.yf.Ticker")
    def test_download_success_first_attempt(self, mock_ticker_class):
        """Test successful download on first attempt."""
        # Mock successful download
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame(
            {
                "Open": [100],
                "High": [105],
                "Low": [95],
                "Close": [102],
                "Volume": [1000],
            }
        )
        mock_ticker_class.return_value = mock_ticker

        result = _download_with_retry("AAPL", "2023-01-01", "2023-01-02", "1d")

        assert len(result) == 1
        mock_ticker.history.assert_called_once_with(
            start="2023-01-01", end="2023-01-02", interval="1d"
        )

    @patch("qbacktester.data.yf.Ticker")
    @patch("qbacktester.data.time.sleep")
    def test_download_retry_on_failure(self, mock_sleep, mock_ticker_class):
        """Test retry logic when first attempts fail."""
        # Mock ticker that fails twice then succeeds
        mock_ticker = Mock()
        mock_ticker.history.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            pd.DataFrame(
                {
                    "Open": [100],
                    "High": [105],
                    "Low": [95],
                    "Close": [102],
                    "Volume": [1000],
                }
            ),
        ]
        mock_ticker_class.return_value = mock_ticker

        result = _download_with_retry(
            "AAPL", "2023-01-01", "2023-01-02", "1d", max_retries=3
        )

        assert len(result) == 1
        assert mock_ticker.history.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries

    @patch("qbacktester.data.yf.Ticker")
    def test_download_empty_data_raises_error(self, mock_ticker_class):
        """Test that empty data raises DataError."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()  # Empty data
        mock_ticker_class.return_value = mock_ticker

        with pytest.raises(DataError, match="No data returned for symbol"):
            _download_with_retry("INVALID", "2023-01-01", "2023-01-02", "1d")

    @patch("qbacktester.data.yf.Ticker")
    def test_download_all_retries_fail(self, mock_ticker_class):
        """Test that all retry attempts failing raises DataError."""
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("Persistent error")
        mock_ticker_class.return_value = mock_ticker

        with pytest.raises(
            DataError, match="Failed to download data.*after 3 attempts"
        ):
            _download_with_retry(
                "INVALID", "2023-01-01", "2023-01-02", "1d", max_retries=3
            )


class TestValidateAndProcessData:
    """Test data validation and processing."""

    def test_validate_empty_data_raises_error(self):
        """Test that empty data raises DataError."""
        empty_data = pd.DataFrame()

        with pytest.raises(DataError, match="No data available"):
            _validate_and_process_data(empty_data, "AAPL", "2023-01-01", "2023-01-02")

    def test_validate_missing_columns_raises_error(self):
        """Test that missing required columns raises DataError."""
        incomplete_data = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [105, 106],
                # Missing Low, Close, Volume
            }
        )

        with pytest.raises(DataError, match="Missing required columns"):
            _validate_and_process_data(
                incomplete_data, "AAPL", "2023-01-01", "2023-01-02"
            )

    def test_validate_processes_data_correctly(self):
        """Test that data is processed correctly."""
        raw_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [102, 103, 104],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2023-01-03", periods=3),
        )  # Unsorted dates

        result = _validate_and_process_data(
            raw_data, "AAPL", "2023-01-01", "2023-01-05"
        )

        # Check column names are lowercase
        expected_columns = ["open", "high", "low", "close", "volume"]
        assert all(col in result.columns for col in expected_columns)

        # Check data is sorted by date
        assert result.index.is_monotonic_increasing

        # Check data integrity
        assert len(result) == 3
        assert not result.empty

    def test_validate_drops_all_nan_rows(self):
        """Test that rows with all NaN values are dropped."""
        data_with_nans = pd.DataFrame(
            {
                "Open": [100, np.nan, 102],
                "High": [105, np.nan, 107],
                "Low": [95, np.nan, 97],
                "Close": [102, np.nan, 104],
                "Volume": [1000, np.nan, 1200],
            }
        )

        result = _validate_and_process_data(
            data_with_nans, "AAPL", "2023-01-01", "2023-01-03"
        )

        # Should drop the all-NaN row
        assert len(result) == 2

    def test_validate_all_nan_raises_error(self):
        """Test that all-NaN data raises DataError."""
        all_nan_data = pd.DataFrame(
            {
                "Open": [np.nan, np.nan],
                "High": [np.nan, np.nan],
                "Low": [np.nan, np.nan],
                "Close": [np.nan, np.nan],
                "Volume": [np.nan, np.nan],
            }
        )

        with pytest.raises(DataError, match="All data.*is NaN after processing"):
            _validate_and_process_data(all_nan_data, "AAPL", "2023-01-01", "2023-01-02")


class TestYahooFinanceProvider:
    """Test the YahooFinanceProvider class."""

    @patch("qbacktester.data.get_price_data")
    def test_yahoo_finance_provider_calls_get_price_data(self, mock_get_price_data):
        """Test that YahooFinanceProvider calls get_price_data."""
        mock_data = pd.DataFrame({"close": [100, 101, 102]})
        mock_get_price_data.return_value = mock_data

        provider = YahooFinanceProvider()
        result = provider.get_data("AAPL", "2023-01-01", "2023-01-03", "1d")

        mock_get_price_data.assert_called_once_with(
            "AAPL", "2023-01-01", "2023-01-03", "1d"
        )
        pd.testing.assert_frame_equal(result, mock_data)


class TestDataError:
    """Test the custom DataError exception."""

    def test_data_error_inheritance(self):
        """Test that DataError inherits from Exception."""
        error = DataError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
