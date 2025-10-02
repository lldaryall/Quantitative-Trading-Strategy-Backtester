"""Tests for strategy module."""

import numpy as np
import pandas as pd
import pytest

from qbacktester.strategy import (
    StrategyParams,
    backtest_strategy,
    calculate_position_sizes,
    generate_signals,
    get_strategy_metrics,
)


class TestStrategyParams:
    """Test StrategyParams dataclass."""

    def test_valid_params(self):
        """Test valid parameter initialization."""
        params = StrategyParams(
            symbol="AAPL",
            start="2023-01-01",
            end="2023-12-31",
            fast_window=10,
            slow_window=20,
        )

        assert params.symbol == "AAPL"
        assert params.fast_window == 10
        assert params.slow_window == 20
        assert params.initial_cash == 100_000
        assert params.fee_bps == 1.0
        assert params.slippage_bps == 0.5

    def test_default_values(self):
        """Test default parameter values."""
        params = StrategyParams(
            symbol="AAPL",
            start="2023-01-01",
            end="2023-12-31",
            fast_window=5,
            slow_window=15,
        )

        assert params.initial_cash == 100_000
        assert params.fee_bps == 1.0
        assert params.slippage_bps == 0.5

    def test_invalid_fast_window(self):
        """Test invalid fast_window values."""
        with pytest.raises(ValueError, match="fast_window must be positive"):
            StrategyParams("AAPL", "2023-01-01", "2023-12-31", 0, 20)

        with pytest.raises(ValueError, match="fast_window must be positive"):
            StrategyParams("AAPL", "2023-01-01", "2023-12-31", -1, 20)

    def test_invalid_slow_window(self):
        """Test invalid slow_window values."""
        with pytest.raises(ValueError, match="slow_window must be positive"):
            StrategyParams("AAPL", "2023-01-01", "2023-12-31", 10, 0)

        with pytest.raises(ValueError, match="slow_window must be positive"):
            StrategyParams("AAPL", "2023-01-01", "2023-12-31", 10, -1)

    def test_fast_window_greater_than_slow(self):
        """Test fast_window >= slow_window validation."""
        with pytest.raises(
            ValueError, match="fast_window must be less than slow_window"
        ):
            StrategyParams("AAPL", "2023-01-01", "2023-12-31", 20, 10)

        with pytest.raises(
            ValueError, match="fast_window must be less than slow_window"
        ):
            StrategyParams("AAPL", "2023-01-01", "2023-12-31", 10, 10)

    def test_invalid_cash(self):
        """Test invalid initial_cash values."""
        with pytest.raises(ValueError, match="initial_cash must be positive"):
            StrategyParams("AAPL", "2023-01-01", "2023-12-31", 10, 20, initial_cash=0)

        with pytest.raises(ValueError, match="initial_cash must be positive"):
            StrategyParams(
                "AAPL", "2023-01-01", "2023-12-31", 10, 20, initial_cash=-1000
            )

    def test_invalid_fee_bps(self):
        """Test invalid fee_bps values."""
        with pytest.raises(ValueError, match="fee_bps must be non-negative"):
            StrategyParams("AAPL", "2023-01-01", "2023-12-31", 10, 20, fee_bps=-1)

    def test_invalid_slippage_bps(self):
        """Test invalid slippage_bps values."""
        with pytest.raises(ValueError, match="slippage_bps must be non-negative"):
            StrategyParams("AAPL", "2023-01-01", "2023-12-31", 10, 20, slippage_bps=-1)


class TestGenerateSignals:
    """Test signal generation function."""

    def test_basic_signal_generation(self):
        """Test basic signal generation."""
        # Create synthetic data with clear trend
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        prices = [
            100,
            101,
            102,
            103,
            104,
            105,
            104,
            103,
            102,
            101,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
        ]

        df = pd.DataFrame({"close": prices}, index=dates)

        result = generate_signals(df, fast_window=3, slow_window=5)

        # Check required columns are present
        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns
        assert "crossover" in result.columns
        assert "signal" in result.columns

        # Check signal values are 0 or 1
        assert result["signal"].isin([0, 1]).all()

        # Check no NaN values in signal column (after fillna)
        assert not result["signal"].isna().any()

    def test_no_look_ahead_bias(self):
        """Test that signals are properly shifted to avoid look-ahead bias."""
        # Create data where crossover happens on day 5
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        prices = [
            100,
            100,
            100,
            100,
            100,
            101,
            102,
            103,
            104,
            105,
        ]  # Clear uptrend starts day 6

        df = pd.DataFrame({"close": prices}, index=dates)

        result = generate_signals(df, fast_window=2, slow_window=4)

        # The crossover should be detected on day 5 (when fast SMA crosses above slow)
        # But the signal should be 0 on day 5 and 1 on day 6 (shifted by 1)

        # Check that signal on crossover day is 0 (no look-ahead)
        crossover_day = result["crossover"] == 1
        if crossover_day.any():
            crossover_indices = result[crossover_day].index
            for idx in crossover_indices:
                # Signal should be 0 on the day of crossover
                assert result.loc[idx, "signal"] == 0

        # Check that signal is 1 on the day after crossover
        signal_days = result["signal"] == 1
        if signal_days.any():
            signal_indices = result[signal_days].index
            for idx in signal_indices:
                # Check that there was a crossover on the previous day
                prev_idx = result.index[result.index.get_loc(idx) - 1]
                assert result.loc[prev_idx, "crossover"] == 1

    def test_synthetic_crossover_scenario(self):
        """Test specific synthetic crossover scenario."""
        # Create data with known crossover point
        dates = pd.date_range("2023-01-01", periods=15, freq="D")
        prices = [
            100,
            100,
            100,
            100,
            100,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
        ]

        df = pd.DataFrame({"close": prices}, index=dates)

        result = generate_signals(df, fast_window=3, slow_window=6)

        # With this data:
        # Day 0-5: prices = 100 (slow SMA will be higher initially)
        # Day 6+: prices increase (fast SMA will eventually cross above slow)

        # Find the first crossover
        crossovers = result[result["crossover"] == 1]
        if len(crossovers) > 0:
            first_crossover_idx = crossovers.index[0]
            first_crossover_pos = result.index.get_loc(first_crossover_idx)

            # Signal should be 0 on crossover day
            assert result.loc[first_crossover_idx, "signal"] == 0

            # Signal should be 1 on the next day (if it exists)
            if first_crossover_pos + 1 < len(result):
                next_day_idx = result.index[first_crossover_pos + 1]
                assert result.loc[next_day_idx, "signal"] == 1

    def test_first_tradable_signal_timing(self):
        """Test that first tradable signal occurs at correct time."""
        # Create data with very clear crossover pattern
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        # Start with slow trend, then fast acceleration
        prices = [
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
        ]

        df = pd.DataFrame({"close": prices}, index=dates)

        result = generate_signals(df, fast_window=3, slow_window=8)

        # Find first signal = 1
        first_signal = result[result["signal"] == 1]
        if len(first_signal) > 0:
            first_signal_idx = first_signal.index[0]
            first_signal_pos = result.index.get_loc(first_signal_idx)

            # Check that there was a crossover on the previous day
            prev_idx = result.index[first_signal_pos - 1]
            assert result.loc[prev_idx, "crossover"] == 1

            # Check that signal was 0 on the crossover day
            assert result.loc[prev_idx, "signal"] == 0

    def test_missing_close_column(self):
        """Test error when close column is missing."""
        df = pd.DataFrame(
            {"open": [100, 101, 102], "high": [105, 106, 107], "low": [95, 96, 97]}
        )

        with pytest.raises(ValueError, match="price_df must contain 'close' column"):
            generate_signals(df, 2, 4)

    def test_invalid_windows(self):
        """Test error handling for invalid windows."""
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        with pytest.raises(ValueError, match="Windows must be positive"):
            generate_signals(df, 0, 4)

        with pytest.raises(
            ValueError, match="fast_window must be less than slow_window"
        ):
            generate_signals(df, 4, 2)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"close": []})

        result = generate_signals(df, 2, 4)

        assert len(result) == 0
        assert "signal" in result.columns
        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns
        assert "crossover" in result.columns


class TestCalculatePositionSizes:
    """Test position size calculation."""

    def test_basic_position_calculation(self):
        """Test basic position size calculation."""
        # Create signals DataFrame
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        signals_df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
                "signal": [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            },
            index=dates,
        )

        result = calculate_position_sizes(signals_df, initial_cash=10000)

        # Check required columns
        assert "position" in result.columns
        assert "cash" in result.columns
        assert "portfolio_value" in result.columns
        assert "trade_cost" in result.columns

        # Check position calculation
        assert result["position"].iloc[0] == 0  # First position should be 0
        # Position should equal signals directly
        pd.testing.assert_series_equal(
            result["position"], signals_df["signal"], check_names=False
        )

        # Check that position changes follow signal changes
        signal_changes = signals_df["signal"].diff()
        position_changes = result["position"].diff()
        pd.testing.assert_series_equal(
            signal_changes, position_changes, check_names=False
        )

    def test_missing_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        with pytest.raises(
            ValueError, match="signals_df must contain 'signal' and 'close' columns"
        ):
            calculate_position_sizes(df, 10000)

        df2 = pd.DataFrame({"signal": [0, 1, 0]})
        with pytest.raises(
            ValueError, match="signals_df must contain 'signal' and 'close' columns"
        ):
            calculate_position_sizes(df2, 10000)


class TestBacktestStrategy:
    """Test complete backtest function."""

    def test_basic_backtest(self):
        """Test basic backtest execution."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        prices = [
            100,
            101,
            102,
            103,
            104,
            105,
            104,
            103,
            102,
            101,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
        ]

        price_df = pd.DataFrame({"close": prices}, index=dates)

        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-20",
            fast_window=3,
            slow_window=6,
        )

        result = backtest_strategy(price_df, params)

        # Check that all expected columns are present
        expected_columns = [
            "close",
            "sma_fast",
            "sma_slow",
            "crossover",
            "signal",
            "position",
            "cash",
            "portfolio_value",
            "trade_cost",
        ]
        for col in expected_columns:
            assert col in result.columns

        # Check that result has same length as input
        assert len(result) == len(price_df)

    def test_backtest_with_custom_params(self):
        """Test backtest with custom parameters."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        price_df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101]}, index=dates
        )

        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-10",
            fast_window=2,
            slow_window=4,
            initial_cash=50000,
            fee_bps=2.0,
            slippage_bps=1.0,
        )

        result = backtest_strategy(price_df, params)

        # Check that custom parameters are reflected
        assert result["cash"].iloc[0] == 50000  # Initial cash
        # Trade costs should be higher due to higher fees
        assert result["trade_cost"].sum() > 0


class TestGetStrategyMetrics:
    """Test strategy metrics calculation."""

    def test_basic_metrics(self):
        """Test basic metrics calculation."""
        # Create sample portfolio data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        portfolio_df = pd.DataFrame(
            {
                "portfolio_value": [
                    10000,
                    10100,
                    10200,
                    10150,
                    10100,
                    10200,
                    10300,
                    10250,
                    10200,
                    10300,
                ]
            },
            index=dates,
        )

        metrics = get_strategy_metrics(portfolio_df)

        # Check that all expected metrics are present
        expected_metrics = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
            "total_trades",
            "final_value",
        ]
        for metric in expected_metrics:
            assert metric in metrics

        # Check that metrics are reasonable
        assert metrics["total_return"] > 0  # Portfolio grew
        assert metrics["final_value"] == 10300
        assert metrics["total_trades"] >= 0

    def test_missing_portfolio_value_column(self):
        """Test error when portfolio_value column is missing."""
        df = pd.DataFrame({"other_column": [1, 2, 3]})

        with pytest.raises(
            ValueError, match="portfolio_df must contain 'portfolio_value' column"
        ):
            get_strategy_metrics(df)


class TestStrategyIntegration:
    """Test integration between strategy components."""

    def test_end_to_end_strategy(self):
        """Test complete end-to-end strategy execution."""
        # Create realistic price data with clear trends
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        np.random.seed(42)  # For reproducible results

        # Create trending price data
        trend = np.linspace(100, 120, 50)
        noise = np.random.normal(0, 1, 50)
        prices = trend + noise

        price_df = pd.DataFrame({"close": prices}, index=dates)

        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-02-19",
            fast_window=5,
            slow_window=15,
            initial_cash=100000,
            fee_bps=1.0,
            slippage_bps=0.5,
        )

        # Run backtest
        result = backtest_strategy(price_df, params)

        # Calculate metrics
        metrics = get_strategy_metrics(result)

        # Verify results are reasonable
        assert len(result) == 50
        assert "signal" in result.columns
        assert result["signal"].isin([0, 1]).all()
        assert metrics["final_value"] > 0
        assert metrics["total_trades"] >= 0

        # Verify no look-ahead bias by checking signal timing
        signals = result["signal"]
        crossovers = result["crossover"]

        # Find days with signals
        signal_days = result[signals == 1]
        if len(signal_days) > 0:
            for idx in signal_days.index:
                pos = result.index.get_loc(idx)
                if pos > 0:
                    prev_idx = result.index[pos - 1]
                    # Should have had a crossover on previous day
                    assert result.loc[prev_idx, "crossover"] == 1
