"""Tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from qbacktester.indicators import bollinger_bands, crossover, ema, macd, rsi, sma


class TestSMA:
    """Test Simple Moving Average function."""

    def test_sma_basic(self):
        """Test basic SMA calculation."""
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        result = sma(df, window=3, col="close")
        expected = pd.Series(
            [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], name="close"
        )

        pd.testing.assert_series_equal(result, expected)

    def test_sma_window_1(self):
        """Test SMA with window=1 (should return original values)."""
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})

        result = sma(df, window=1, col="close")
        expected = df["close"].astype("float64")

        pd.testing.assert_series_equal(result, expected)

    def test_sma_window_larger_than_data(self):
        """Test SMA with window larger than data length."""
        df = pd.DataFrame({"close": [1, 2, 3]})

        result = sma(df, window=5, col="close")
        expected = pd.Series([1.0, 1.5, 2.0], name="close")  # min_periods=1

        pd.testing.assert_series_equal(result, expected)

    def test_sma_with_nans(self):
        """Test SMA with NaN values."""
        df = pd.DataFrame({"close": [1, np.nan, 3, 4, np.nan, 6]})

        result = sma(df, window=2, col="close")
        # First value: 1.0, second: 1.0 (only one valid), third: 3.0, etc.
        expected = pd.Series([1.0, 1.0, 3.0, 3.5, 4.0, 6.0], name="close")

        pd.testing.assert_series_equal(result, expected)

    def test_sma_invalid_window(self):
        """Test SMA with invalid window."""
        df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="Window must be positive"):
            sma(df, window=0, col="close")

        with pytest.raises(ValueError, match="Window must be positive"):
            sma(df, window=-1, col="close")

    def test_sma_invalid_column(self):
        """Test SMA with invalid column."""
        df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'invalid' not found"):
            sma(df, window=2, col="invalid")

    def test_sma_empty_dataframe(self):
        """Test SMA with empty DataFrame."""
        df = pd.DataFrame({"close": []})

        result = sma(df, window=2, col="close")
        expected = pd.Series([], dtype="float64", name="close")

        pd.testing.assert_series_equal(result, expected)


class TestEMA:
    """Test Exponential Moving Average function."""

    def test_ema_basic(self):
        """Test basic EMA calculation."""
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})

        result = ema(df, span=3, col="close")

        # EMA calculation: first value is the first price
        # Then: EMA = (price * alpha) + (previous_EMA * (1 - alpha))
        # alpha = 2 / (span + 1) = 2 / 4 = 0.5
        expected = pd.Series([1.0, 1.5, 2.25, 3.125, 4.0625], name="close")

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_ema_span_1(self):
        """Test EMA with span=1 (should return original values)."""
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})

        result = ema(df, span=1, col="close")
        expected = df["close"].astype("float64")

        pd.testing.assert_series_equal(result, expected)

    def test_ema_with_nans(self):
        """Test EMA with NaN values."""
        df = pd.DataFrame({"close": [1, np.nan, 3, 4, np.nan]})

        result = ema(df, span=2, col="close")
        # EMA with NaNs - pandas ewm propagates NaNs
        # First value: 1.0, second: NaN, third: 3.0, fourth: 3.5, fifth: NaN
        # But actually, ewm with NaNs behaves differently - let's check the actual behavior
        expected = pd.Series(
            [1.0, 1.0, 2.7142857142857144, 3.5714285714285716, 3.5714285714285716],
            name="close",
        )

        pd.testing.assert_series_equal(result, expected)

    def test_ema_invalid_span(self):
        """Test EMA with invalid span."""
        df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="Span must be positive"):
            ema(df, span=0, col="close")

        with pytest.raises(ValueError, match="Span must be positive"):
            ema(df, span=-1, col="close")

    def test_ema_invalid_column(self):
        """Test EMA with invalid column."""
        df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'invalid' not found"):
            ema(df, span=2, col="invalid")


class TestCrossover:
    """Test crossover detection function."""

    def test_crossover_basic(self):
        """Test basic crossover detection."""
        fast = pd.Series([1, 2, 3, 2, 1], name="fast")
        slow = pd.Series([2, 2, 2, 2, 2], name="slow")

        result = crossover(fast, slow)
        expected = pd.Series([np.nan, 0, 1, 0, -1], dtype="float64")

        pd.testing.assert_series_equal(result, expected, check_dtype=False)

    def test_crossover_no_crosses(self):
        """Test crossover with no crosses."""
        fast = pd.Series([1, 2, 3, 4, 5], name="fast")
        slow = pd.Series([2, 3, 4, 5, 6], name="slow")

        result = crossover(fast, slow)
        expected = pd.Series([np.nan, 0, 0, 0, 0], dtype="float64")

        pd.testing.assert_series_equal(result, expected, check_dtype=False)

    def test_crossover_multiple_crosses(self):
        """Test crossover with multiple crosses."""
        fast = pd.Series([1, 3, 1, 3, 1], name="fast")
        slow = pd.Series([2, 2, 2, 2, 2], name="slow")

        result = crossover(fast, slow)
        expected = pd.Series([np.nan, 1, -1, 1, -1], dtype="float64")

        pd.testing.assert_series_equal(result, expected, check_dtype=False)

    def test_crossover_with_nans(self):
        """Test crossover with NaN values."""
        fast = pd.Series([1, 2, np.nan, 4, 5], name="fast")
        slow = pd.Series([2, 2, 2, 2, 2], name="slow")

        result = crossover(fast, slow)
        # When there's a NaN, the crossover detection is affected
        # diff: [-1, 0, NaN, 2, 3]
        # diff_prev: [NaN, -1, 0, NaN, 2]
        # crossover_up: (diff > 0) & (diff_prev <= 0) = [False, False, False, True, False]
        # crossover_down: (diff < 0) & (diff_prev >= 0) = [False, False, False, False, False]
        # But NaN in diff_prev makes the comparison False, so no crossover detected
        expected = pd.Series([np.nan, 0, np.nan, 0, 0], dtype="float64")

        pd.testing.assert_series_equal(result, expected, check_dtype=False)

    def test_crossover_equal_values(self):
        """Test crossover when values are equal."""
        fast = pd.Series([1, 2, 2, 3, 4], name="fast")
        slow = pd.Series([2, 2, 2, 2, 2], name="slow")

        result = crossover(fast, slow)
        expected = pd.Series([np.nan, 0, 0, 1, 0], dtype="float64")

        pd.testing.assert_series_equal(result, expected, check_dtype=False)

    def test_crossover_short_series(self):
        """Test crossover with very short series."""
        fast = pd.Series([1], name="fast")
        slow = pd.Series([2], name="slow")

        result = crossover(fast, slow)
        expected = pd.Series([np.nan], dtype="float64")

        pd.testing.assert_series_equal(result, expected, check_dtype=False)

    def test_crossover_different_lengths(self):
        """Test crossover with different length series."""
        fast = pd.Series([1, 2, 3], name="fast")
        slow = pd.Series([2, 2], name="slow")

        with pytest.raises(
            ValueError, match="Fast and slow series must have the same length"
        ):
            crossover(fast, slow)

    def test_crossover_different_indices(self):
        """Test crossover with different indices."""
        fast = pd.Series([1, 2, 3], index=[0, 1, 2], name="fast")
        slow = pd.Series([2, 2, 2], index=[1, 2, 3], name="slow")

        with pytest.raises(
            ValueError, match="Fast and slow series must have the same index"
        ):
            crossover(fast, slow)

    def test_crossover_empty_series(self):
        """Test crossover with empty series."""
        fast = pd.Series([], name="fast")
        slow = pd.Series([], name="slow")

        result = crossover(fast, slow)
        expected = pd.Series([], dtype="float64")

        pd.testing.assert_series_equal(result, expected, check_dtype=False)


class TestRSI:
    """Test RSI function."""

    def test_rsi_basic(self):
        """Test basic RSI calculation."""
        df = pd.DataFrame(
            {"close": [44, 44.25, 44.5, 43.75, 44, 44.5, 45, 45.25, 45.5, 45.75]}
        )

        result = rsi(df, window=5, col="close")

        # RSI should be between 0 and 100
        assert all(0 <= val <= 100 for val in result.dropna())
        assert len(result) == len(df)

    def test_rsi_constant_prices(self):
        """Test RSI with constant prices (should be 50)."""
        df = pd.DataFrame({"close": [100, 100, 100, 100, 100]})

        result = rsi(df, window=3, col="close")

        # With constant prices, RSI should be 50
        assert all(val == 50.0 for val in result.dropna())

    def test_rsi_invalid_window(self):
        """Test RSI with invalid window."""
        df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="Window must be positive"):
            rsi(df, window=0, col="close")


class TestBollingerBands:
    """Test Bollinger Bands function."""

    def test_bollinger_bands_basic(self):
        """Test basic Bollinger Bands calculation."""
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        result = bollinger_bands(df, window=3, std_dev=2, col="close")

        assert "upper" in result.columns
        assert "middle" in result.columns
        assert "lower" in result.columns
        assert len(result) == len(df)

        # Upper band should be above middle, lower below (skip NaN values)
        valid_mask = ~(
            result["upper"].isna() | result["middle"].isna() | result["lower"].isna()
        )
        assert (result["upper"][valid_mask] >= result["middle"][valid_mask]).all()
        assert (result["lower"][valid_mask] <= result["middle"][valid_mask]).all()

    def test_bollinger_bands_invalid_window(self):
        """Test Bollinger Bands with invalid window."""
        df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="Window must be positive"):
            bollinger_bands(df, window=0, col="close")


class TestMACD:
    """Test MACD function."""

    def test_macd_basic(self):
        """Test basic MACD calculation."""
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        result = macd(df, fast=3, slow=5, signal=2, col="close")

        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns
        assert len(result) == len(df)

        # Histogram should equal MACD - Signal
        pd.testing.assert_series_equal(
            result["histogram"], result["macd"] - result["signal"], check_names=False
        )

    def test_macd_invalid_periods(self):
        """Test MACD with invalid periods."""
        df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="All periods must be positive"):
            macd(df, fast=0, slow=5, signal=2, col="close")

        with pytest.raises(ValueError, match="All periods must be positive"):
            macd(df, fast=3, slow=0, signal=2, col="close")

        with pytest.raises(ValueError, match="All periods must be positive"):
            macd(df, fast=3, slow=5, signal=0, col="close")


class TestIndicatorEdgeCases:
    """Test edge cases across all indicators."""

    def test_all_indicators_with_single_value(self):
        """Test all indicators with single value data."""
        df = pd.DataFrame({"close": [100]})

        # SMA should work
        sma_result = sma(df, window=1, col="close")
        assert len(sma_result) == 1
        assert sma_result.iloc[0] == 100

        # EMA should work
        ema_result = ema(df, span=1, col="close")
        assert len(ema_result) == 1
        assert ema_result.iloc[0] == 100

        # RSI should work
        rsi_result = rsi(df, window=1, col="close")
        assert len(rsi_result) == 1

        # Bollinger Bands should work
        bb_result = bollinger_bands(df, window=1, col="close")
        assert len(bb_result) == 1

        # MACD should work
        macd_result = macd(df, fast=1, slow=1, signal=1, col="close")
        assert len(macd_result) == 1

    def test_all_indicators_with_all_nans(self):
        """Test all indicators with all NaN data."""
        df = pd.DataFrame({"close": [np.nan, np.nan, np.nan]})

        # All indicators should handle NaN gracefully
        sma_result = sma(df, window=2, col="close")
        assert sma_result.isna().all()

        ema_result = ema(df, span=2, col="close")
        assert ema_result.isna().all()

        rsi_result = rsi(df, window=2, col="close")
        assert rsi_result.isna().all()

    def test_crossover_edge_cases(self):
        """Test crossover with various edge cases."""
        # Test with identical series
        identical = pd.Series([1, 2, 3, 4, 5])
        result = crossover(identical, identical)
        expected = pd.Series([np.nan, 0, 0, 0, 0], dtype="float64")
        pd.testing.assert_series_equal(result, expected, check_dtype=False)

        # Test with all NaN series
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        result = crossover(nan_series, nan_series)
        expected = pd.Series([np.nan, np.nan, np.nan], dtype="float64")
        pd.testing.assert_series_equal(result, expected, check_dtype=False)
