"""Tests for metrics module."""

import numpy as np
import pandas as pd
import pytest

from qbacktester.metrics import (
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


class TestDailyReturns:
    """Test daily_returns function."""

    def test_basic_daily_returns(self):
        """Test basic daily returns calculation."""
        equity_curve = pd.Series(
            [100, 101, 102, 103, 104],
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )

        returns = daily_returns(equity_curve)

        # First value should be NaN
        assert pd.isna(returns.iloc[0])

        # Check calculated returns
        expected = [np.nan, 0.01, 0.009901, 0.009804, 0.009709]
        np.testing.assert_array_almost_equal(returns.values, expected, decimal=6)

    def test_empty_series(self):
        """Test error with empty series."""
        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            daily_returns(pd.Series([], dtype=float))

    def test_negative_values(self):
        """Test error with negative values."""
        equity_curve = pd.Series([100, 101, -102, 103])

        with pytest.raises(
            ValueError, match="equity_curve must contain only positive values"
        ):
            daily_returns(equity_curve)

    def test_zero_values(self):
        """Test error with zero values."""
        equity_curve = pd.Series([100, 101, 0, 103])

        with pytest.raises(
            ValueError, match="equity_curve must contain only positive values"
        ):
            daily_returns(equity_curve)

    def test_single_value(self):
        """Test with single value."""
        equity_curve = pd.Series(
            [100], index=pd.date_range("2023-01-01", periods=1, freq="D")
        )

        returns = daily_returns(equity_curve)

        # Should have one NaN value
        assert len(returns) == 1
        assert pd.isna(returns.iloc[0])


class TestCAGR:
    """Test cagr function."""

    def test_basic_cagr(self):
        """Test basic CAGR calculation."""
        # 10% growth over 1 year (252 days)
        equity_curve = pd.Series(
            [100, 110], index=pd.date_range("2023-01-01", periods=2, freq="D")
        )

        cagr_value = cagr(
            equity_curve, periods_per_year=1
        )  # Use 1 period per year for 1-day growth

        # Should be approximately 10%
        assert abs(cagr_value - 0.10) < 0.001

    def test_cagr_with_more_periods(self):
        """Test CAGR with more periods."""
        # Create a series that doubles over 252 days
        days = 252
        equity_curve = pd.Series(
            [100] + [100 * (1.01**i) for i in range(1, days + 1)],
            index=pd.date_range("2023-01-01", periods=days + 1, freq="D"),
        )

        cagr_value = cagr(equity_curve, periods_per_year=252)

        # Should be approximately 1% daily compounded over 252 days
        # (1.01)^252 - 1 ≈ 11.27
        assert abs(cagr_value - 11.27) < 0.1

    def test_zero_cagr(self):
        """Test CAGR with no growth."""
        equity_curve = pd.Series(
            [100, 100, 100, 100], index=pd.date_range("2023-01-01", periods=4, freq="D")
        )

        cagr_value = cagr(equity_curve)

        assert cagr_value == 0.0

    def test_single_value_cagr(self):
        """Test CAGR with single value."""
        equity_curve = pd.Series(
            [100], index=pd.date_range("2023-01-01", periods=1, freq="D")
        )

        cagr_value = cagr(equity_curve)

        assert cagr_value == 0.0

    def test_empty_series(self):
        """Test error with empty series."""
        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            cagr(pd.Series([], dtype=float))

    def test_negative_values(self):
        """Test error with negative values."""
        equity_curve = pd.Series([100, 101, -102])

        with pytest.raises(
            ValueError, match="equity_curve must contain only positive values"
        ):
            cagr(equity_curve)

    def test_invalid_periods_per_year(self):
        """Test error with invalid periods_per_year."""
        equity_curve = pd.Series([100, 110])

        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            cagr(equity_curve, periods_per_year=0)


class TestSharpe:
    """Test sharpe function."""

    def test_basic_sharpe(self):
        """Test basic Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])

        sharpe_ratio = sharpe(returns)

        # Should be positive for this series
        assert sharpe_ratio > 0

    def test_sharpe_with_risk_free_rate(self):
        """Test Sharpe ratio with risk-free rate."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])

        sharpe_ratio = sharpe(returns, risk_free=0.02, periods_per_year=252)

        # Should be lower than without risk-free rate
        sharpe_no_rf = sharpe(returns)
        assert sharpe_ratio < sharpe_no_rf

    def test_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])

        sharpe_ratio = sharpe(returns)

        # Should be infinity for positive returns, -infinity for negative
        assert sharpe_ratio == np.inf

    def test_zero_mean_return(self):
        """Test Sharpe ratio with zero mean return."""
        returns = pd.Series([0.01, -0.01, 0.01, -0.01])

        sharpe_ratio = sharpe(returns)

        assert sharpe_ratio == 0.0

    def test_empty_series(self):
        """Test error with empty series."""
        with pytest.raises(ValueError, match="returns cannot be empty"):
            sharpe(pd.Series([], dtype=float))

    def test_all_nan_series(self):
        """Test error with all NaN series."""
        with pytest.raises(
            ValueError, match="returns must contain at least one non-NaN value"
        ):
            sharpe(pd.Series([np.nan, np.nan, np.nan]))

    def test_series_with_nans(self):
        """Test Sharpe ratio with some NaN values."""
        returns = pd.Series([0.01, np.nan, 0.02, np.nan, 0.01])

        sharpe_ratio = sharpe(returns)

        # Should work and ignore NaN values
        assert not np.isnan(sharpe_ratio)


class TestMaxDrawdown:
    """Test max_drawdown function."""

    def test_basic_max_drawdown(self):
        """Test basic max drawdown calculation."""
        equity_curve = pd.Series(
            [100, 110, 105, 120, 115, 130],
            index=pd.date_range("2023-01-01", periods=6, freq="D"),
        )

        max_dd, peak_date, trough_date = max_drawdown(equity_curve)

        # Max drawdown should be negative
        assert max_dd < 0

        # Should be approximately -4.55% (from 110 to 105)
        assert abs(max_dd - (-0.0455)) < 0.001

        # Check that peak and trough dates are valid
        assert peak_date in equity_curve.index
        assert trough_date in equity_curve.index

    def test_no_drawdown(self):
        """Test max drawdown with no drawdowns."""
        equity_curve = pd.Series(
            [100, 101, 102, 103, 104],
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )

        max_dd, peak_date, trough_date = max_drawdown(equity_curve)

        # Should be 0 (no drawdowns)
        assert max_dd == 0.0
        assert peak_date == equity_curve.index[0]
        assert trough_date == equity_curve.index[0]

    def test_constant_series(self):
        """Test max drawdown with constant series."""
        equity_curve = pd.Series(
            [100, 100, 100, 100], index=pd.date_range("2023-01-01", periods=4, freq="D")
        )

        max_dd, peak_date, trough_date = max_drawdown(equity_curve)

        assert max_dd == 0.0
        assert peak_date == equity_curve.index[0]
        assert trough_date == equity_curve.index[0]

    def test_single_value(self):
        """Test max drawdown with single value."""
        equity_curve = pd.Series(
            [100], index=pd.date_range("2023-01-01", periods=1, freq="D")
        )

        max_dd, peak_date, trough_date = max_drawdown(equity_curve)

        assert max_dd == 0.0
        assert peak_date == equity_curve.index[0]
        assert trough_date == equity_curve.index[0]

    def test_empty_series(self):
        """Test error with empty series."""
        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            max_drawdown(pd.Series([], dtype=float))


class TestCalmar:
    """Test calmar function."""

    def test_basic_calmar(self):
        """Test basic Calmar ratio calculation."""
        equity_curve = pd.Series(
            [100, 110, 105, 120, 115, 130],
            index=pd.date_range("2023-01-01", periods=6, freq="D"),
        )

        calmar_ratio = calmar(equity_curve)

        # Should be positive
        assert calmar_ratio > 0

    def test_calmar_with_no_drawdown(self):
        """Test Calmar ratio with no drawdowns."""
        equity_curve = pd.Series(
            [100, 101, 102, 103, 104],
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )

        calmar_ratio = calmar(equity_curve)

        # Should be infinity (no drawdowns)
        assert calmar_ratio == np.inf

    def test_empty_series(self):
        """Test error with empty series."""
        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            calmar(pd.Series([], dtype=float))


class TestHitRate:
    """Test hit_rate function."""

    def test_basic_hit_rate(self):
        """Test basic hit rate calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])

        hit_rate_value = hit_rate(returns)

        # 4 out of 5 positive returns
        assert hit_rate_value == 0.8

    def test_hit_rate_with_threshold(self):
        """Test hit rate with threshold."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])

        hit_rate_value = hit_rate(returns, threshold=0.015)

        # Only 2 out of 5 above 1.5%
        assert hit_rate_value == 0.4

    def test_all_positive_returns(self):
        """Test hit rate with all positive returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.01])

        hit_rate_value = hit_rate(returns)

        assert hit_rate_value == 1.0

    def test_all_negative_returns(self):
        """Test hit rate with all negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.01])

        hit_rate_value = hit_rate(returns)

        assert hit_rate_value == 0.0

    def test_empty_series(self):
        """Test error with empty series."""
        with pytest.raises(ValueError, match="returns cannot be empty"):
            hit_rate(pd.Series([], dtype=float))


class TestAvgWinLoss:
    """Test avg_win_loss function."""

    def test_basic_avg_win_loss(self):
        """Test basic average win/loss calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])

        avg_win, avg_loss = avg_win_loss(returns)

        # Average win: (0.01 + 0.02 + 0.03) / 3 = 0.02
        # Average loss: (-0.01 + -0.02) / 2 = -0.015
        assert abs(avg_win - 0.02) < 0.001
        assert abs(avg_loss - (-0.015)) < 0.001

    def test_only_wins(self):
        """Test with only winning returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.01])

        avg_win, avg_loss = avg_win_loss(returns)

        # Average win: (0.01 + 0.02 + 0.01 + 0.03 + 0.01) / 5 = 0.016
        # Average loss: 0.0 (no losses)
        assert abs(avg_win - 0.016) < 0.001
        assert avg_loss == 0.0

    def test_only_losses(self):
        """Test with only losing returns."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.01])

        avg_win, avg_loss = avg_win_loss(returns)

        # Average win: 0.0 (no wins)
        # Average loss: (-0.01 + -0.02 + -0.01 + -0.03 + -0.01) / 5 = -0.016
        assert avg_win == 0.0
        assert abs(avg_loss - (-0.016)) < 0.001

    def test_empty_series(self):
        """Test error with empty series."""
        with pytest.raises(ValueError, match="returns cannot be empty"):
            avg_win_loss(pd.Series([], dtype=float))


class TestRealisticData:
    """Test with realistic noisy data."""

    def test_realistic_equity_curve(self):
        """Test with realistic equity curve data."""
        # Create a realistic equity curve with trend and noise
        np.random.seed(42)
        n_days = 252  # One year
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

        # Create trend with noise
        trend = np.linspace(100, 120, n_days)  # 20% annual growth
        noise = np.random.normal(0, 0.02, n_days)  # 2% daily volatility
        equity_curve = pd.Series(trend + noise, index=dates)

        # Calculate metrics
        returns = daily_returns(equity_curve)
        cagr_value = cagr(equity_curve)
        sharpe_ratio = sharpe(returns)
        max_dd, peak_date, trough_date = max_drawdown(equity_curve)

        # Check that metrics are reasonable
        assert cagr_value > 0  # Should be positive growth
        assert sharpe_ratio > 0  # Should be positive Sharpe
        assert max_dd < 0  # Should have some drawdown
        assert peak_date in equity_curve.index
        assert trough_date in equity_curve.index

    def test_constant_series_metrics(self):
        """Test metrics with constant series."""
        equity_curve = pd.Series(
            [100] * 100, index=pd.date_range("2023-01-01", periods=100, freq="D")
        )

        returns = daily_returns(equity_curve)
        cagr_value = cagr(equity_curve)
        sharpe_ratio = sharpe(returns)
        max_dd, _, _ = max_drawdown(equity_curve)

        # All returns should be NaN except first
        assert pd.isna(returns.iloc[0])
        assert (returns.iloc[1:] == 0).all()

        # CAGR should be 0
        assert cagr_value == 0.0

        # Sharpe should be 0 (no volatility)
        assert sharpe_ratio == 0.0

        # Max drawdown should be 0
        assert max_dd == 0.0

    def test_negative_only_series(self):
        """Test metrics with negative-only returns."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.01])

        sharpe_ratio = sharpe(returns)
        hit_rate_value = hit_rate(returns)
        avg_win, avg_loss = avg_win_loss(returns)

        # Sharpe should be negative
        assert sharpe_ratio < 0

        # Hit rate should be 0
        assert hit_rate_value == 0.0

        # Average win should be 0, average loss should be negative
        assert avg_win == 0.0
        assert avg_loss < 0

    def test_mixed_positive_negative_series(self):
        """Test metrics with mixed positive and negative returns."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.01])

        hit_rate_value = hit_rate(returns)
        avg_win, avg_loss = avg_win_loss(returns)
        var_5pct = var(returns, 0.05)
        cvar_5pct = cvar(returns, 0.05)

        # Hit rate should be between 0 and 1
        assert 0 <= hit_rate_value <= 1

        # Average win should be positive, average loss should be negative
        assert avg_win > 0
        assert avg_loss < 0

        # VaR and CVaR should be negative
        assert var_5pct < 0
        assert cvar_5pct < 0
        assert cvar_5pct <= var_5pct  # CVaR should be more extreme than VaR


class TestGetAllMetrics:
    """Test get_all_metrics function."""

    def test_basic_get_all_metrics(self):
        """Test basic get_all_metrics calculation."""
        equity_curve = pd.Series(
            [100, 101, 102, 101, 103, 104, 102, 105],
            index=pd.date_range("2023-01-01", periods=8, freq="D"),
        )

        metrics = get_all_metrics(equity_curve)

        # Check that all expected metrics are present
        expected_metrics = [
            "cagr",
            "sharpe",
            "sortino",
            "volatility",
            "calmar",
            "hit_rate",
            "avg_win",
            "avg_loss",
            "var_5pct",
            "cvar_5pct",
            "max_drawdown",
            "max_drawdown_peak_date",
            "max_drawdown_trough_date",
        ]

        for metric in expected_metrics:
            assert metric in metrics

        # Check that values are reasonable
        assert isinstance(metrics["cagr"], float)
        assert isinstance(metrics["sharpe"], float)
        assert isinstance(metrics["max_drawdown"], float)
        assert metrics["max_drawdown"] <= 0  # Should be negative or zero
