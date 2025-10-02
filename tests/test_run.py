"""Tests for run module."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from qbacktester.run import (
    _extract_trades,
    print_backtest_report,
    print_quick_summary,
    run_crossover_backtest,
    run_multiple_backtests,
)
from qbacktester.strategy import StrategyParams


class TestRunCrossoverBacktest:
    """Test run_crossover_backtest function."""

    @patch("qbacktester.run.get_price_data")
    @patch("qbacktester.run.generate_signals")
    @patch("qbacktester.run.Backtester")
    def test_run_crossover_backtest_success(
        self, mock_backtester_class, mock_generate_signals, mock_get_price_data
    ):
        """Test successful backtest execution."""
        # Mock data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        price_df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
                "open": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            },
            index=dates,
        )

        signals_df = pd.DataFrame(
            {"signal": [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]}, index=dates
        )

        result_df = pd.DataFrame(
            {
                "total_equity": [
                    100000,
                    100000,
                    100100,
                    100200,
                    100300,
                    100200,
                    100100,
                    100000,
                    99900,
                    99800,
                ],
                "trade_flag": [
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
                "signal": [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                "close": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            },
            index=dates,
        )

        # Mock the backtester instance
        mock_backtester = Mock()
        mock_backtester.run.return_value = result_df
        mock_backtester_class.return_value = mock_backtester

        # Mock other functions
        mock_get_price_data.return_value = price_df
        mock_generate_signals.return_value = signals_df

        # Create test parameters
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-10",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=1.0,
            slippage_bps=0.5,
        )

        # Run the function
        result = run_crossover_backtest(params)

        # Verify calls
        mock_get_price_data.assert_called_once_with("TEST", "2023-01-01", "2023-01-10")
        mock_generate_signals.assert_called_once_with(price_df, 3, 6)
        mock_backtester_class.assert_called_once_with(
            price_df, signals_df["signal"], params
        )
        mock_backtester.run.assert_called_once()

        # Verify result structure
        assert "params" in result
        assert "equity_curve" in result
        assert "metrics" in result
        assert "trades" in result

        # Verify params
        assert result["params"] == params

        # Verify equity curve
        pd.testing.assert_frame_equal(result["equity_curve"], result_df)

        # Verify metrics structure
        metrics = result["metrics"]
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

        # Verify trades structure
        trades = result["trades"]
        assert isinstance(trades, pd.DataFrame)
        expected_columns = ["timestamp", "side", "price", "equity"]
        for col in expected_columns:
            assert col in trades.columns

    @patch("qbacktester.run.get_price_data")
    def test_run_crossover_backtest_data_error(self, mock_get_price_data):
        """Test backtest with data fetching error."""
        # Mock data error
        mock_get_price_data.side_effect = Exception("Data fetch failed")

        params = StrategyParams(
            symbol="INVALID",
            start="2023-01-01",
            end="2023-01-10",
            fast_window=3,
            slow_window=6,
        )

        with pytest.raises(Exception, match="Data fetch failed"):
            run_crossover_backtest(params)

    def test_extract_trades_no_trades(self):
        """Test _extract_trades with no trades."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        result_df = pd.DataFrame(
            {
                "trade_flag": [False, False, False, False, False],
                "signal": [0, 0, 0, 0, 0],
                "close": [100, 101, 102, 103, 104],
            },
            index=dates,
        )

        price_df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)

        trades = _extract_trades(result_df, price_df)

        assert len(trades) == 0
        expected_columns = ["timestamp", "side", "price", "equity"]
        for col in expected_columns:
            assert col in trades.columns

    def test_extract_trades_with_trades(self):
        """Test _extract_trades with actual trades."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        result_df = pd.DataFrame(
            {
                "trade_flag": [False, True, False, True, False],
                "signal": [0, 1, 1, 0, 0],
                "close": [100, 101, 102, 103, 104],
                "total_equity": [100000, 100100, 100200, 100150, 100100],
            },
            index=dates,
        )

        price_df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)

        trades = _extract_trades(result_df, price_df)

        assert len(trades) == 2
        assert trades.iloc[0]["side"] == "entry"  # signal changed from 0 to 1
        assert trades.iloc[1]["side"] == "exit"  # signal changed from 1 to 0
        assert trades.iloc[0]["price"] == 101
        assert trades.iloc[1]["price"] == 103


class TestReporting:
    """Test reporting functions."""

    def test_print_backtest_report_structure(self):
        """Test that print_backtest_report doesn't crash and produces output."""
        # Create mock results
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-10",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
        )

        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        equity_curve = pd.DataFrame(
            {"total_equity": [100000, 100100, 100200, 100150, 100100]}, index=dates
        )

        metrics = {
            "cagr": 0.05,
            "sharpe": 1.2,
            "sortino": 1.5,
            "volatility": 0.15,
            "calmar": 2.0,
            "hit_rate": 0.6,
            "avg_win": 0.02,
            "avg_loss": -0.015,
            "var_5pct": -0.03,
            "cvar_5pct": -0.04,
            "max_drawdown": -0.05,
            "max_drawdown_peak_date": dates[2],
            "max_drawdown_trough_date": dates[4],
        }

        trades = pd.DataFrame(
            {
                "timestamp": [dates[1], dates[3]],
                "side": ["entry", "exit"],
                "price": [101, 103],
                "equity": [100100, 100150],
            }
        )

        results = {
            "params": params,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "trades": trades,
        }

        # This should not raise an exception
        print_backtest_report(results)

    def test_print_quick_summary(self):
        """Test print_quick_summary function."""
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-10",
            fast_window=3,
            slow_window=6,
        )

        metrics = {"cagr": 0.05, "sharpe": 1.2, "max_drawdown": -0.05, "hit_rate": 0.6}

        trades = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
                "side": ["entry", "exit", "entry"],
                "price": [100, 101, 102],
                "equity": [100000, 100100, 100200],
            }
        )

        results = {"params": params, "metrics": metrics, "trades": trades}

        # This should not raise an exception
        print_quick_summary(results)


class TestMultipleBacktests:
    """Test multiple backtest functionality."""

    @patch("qbacktester.run.run_crossover_backtest")
    def test_run_multiple_backtests_success(self, mock_run_backtest):
        """Test running multiple backtests successfully."""

        # Mock individual backtest results
        def mock_backtest_result(symbol):
            params = StrategyParams(
                symbol=symbol,
                start="2023-01-01",
                end="2023-01-10",
                fast_window=3,
                slow_window=6,
            )

            metrics = {
                "cagr": 0.05 if symbol == "AAPL" else 0.03,
                "sharpe": 1.2 if symbol == "AAPL" else 0.8,
                "max_drawdown": -0.05 if symbol == "AAPL" else -0.08,
                "hit_rate": 0.6 if symbol == "AAPL" else 0.5,
                "sortino": 1.5,
                "volatility": 0.15,
                "calmar": 2.0,
                "avg_win": 0.02,
                "avg_loss": -0.015,
                "var_5pct": -0.03,
                "cvar_5pct": -0.04,
                "max_drawdown_peak_date": pd.Timestamp("2023-01-05"),
                "max_drawdown_trough_date": pd.Timestamp("2023-01-08"),
            }

            trades = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=2, freq="D"),
                    "side": ["entry", "exit"],
                    "price": [100, 101],
                    "equity": [100000, 100100],
                }
            )

            return {
                "params": params,
                "equity_curve": pd.DataFrame({"total_equity": [100000, 100100]}),
                "metrics": metrics,
                "trades": trades,
            }

        mock_run_backtest.side_effect = lambda params: mock_backtest_result(
            params.symbol
        )

        # Create parameter list
        param_list = [
            StrategyParams("AAPL", "2023-01-01", "2023-01-10", 3, 6),
            StrategyParams("GOOGL", "2023-01-01", "2023-01-10", 3, 6),
        ]

        # Run multiple backtests
        results = run_multiple_backtests(param_list)

        # Verify results
        assert len(results) == 2
        assert "AAPL_1" in results
        assert "GOOGL_2" in results

        # Verify individual results
        aapl_result = results["AAPL_1"]
        assert aapl_result["metrics"]["cagr"] == 0.05
        assert aapl_result["metrics"]["sharpe"] == 1.2

    @patch("qbacktester.run.run_crossover_backtest")
    def test_run_multiple_backtests_with_errors(self, mock_run_backtest):
        """Test running multiple backtests with some errors."""

        def mock_backtest_with_error(params):
            if params.symbol == "ERROR":
                raise Exception("Test error")
            else:
                return {
                    "params": params,
                    "equity_curve": pd.DataFrame({"total_equity": [100000]}),
                    "metrics": {
                        "cagr": 0.05,
                        "sharpe": 1.2,
                        "max_drawdown": -0.05,
                        "hit_rate": 0.6,
                    },
                    "trades": pd.DataFrame(
                        columns=["timestamp", "side", "price", "equity"]
                    ),
                }

        mock_run_backtest.side_effect = mock_backtest_with_error

        param_list = [
            StrategyParams("AAPL", "2023-01-01", "2023-01-10", 3, 6),
            StrategyParams("ERROR", "2023-01-01", "2023-01-10", 3, 6),
        ]

        results = run_multiple_backtests(param_list)

        # Verify results
        assert len(results) == 2
        assert "error" in results["ERROR_2"]
        assert "error" not in results["AAPL_1"]


class TestSmokeTest:
    """Smoke test for the complete pipeline."""

    @patch("qbacktester.run.get_price_data")
    def test_smoke_test_complete_pipeline(self, mock_get_price_data):
        """Test that the complete pipeline works with mock data."""
        # Create realistic mock data
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        np.random.seed(42)

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 20)  # 0.1% daily return, 2% volatility
        prices = 100 * (1 + returns).cumprod()

        price_df = pd.DataFrame(
            {
                "close": prices,
                "open": prices
                * (1 + np.random.normal(0, 0.001, 20)),  # Small open/close differences
            },
            index=dates,
        )

        mock_get_price_data.return_value = price_df

        # Create test parameters
        params = StrategyParams(
            symbol="MOCK",
            start="2023-01-01",
            end="2023-01-20",
            fast_window=5,
            slow_window=10,
            initial_cash=100000,
            fee_bps=1.0,
            slippage_bps=0.5,
        )

        # Run the complete pipeline
        result = run_crossover_backtest(params)

        # Verify the pipeline completed successfully
        assert "params" in result
        assert "equity_curve" in result
        assert "metrics" in result
        assert "trades" in result

        # Verify data was fetched
        mock_get_price_data.assert_called_once_with("MOCK", "2023-01-01", "2023-01-20")

        # Verify equity curve has expected structure
        equity_curve = result["equity_curve"]
        assert "total_equity" in equity_curve.columns
        assert "trade_flag" in equity_curve.columns
        assert "signal" in equity_curve.columns

        # Verify metrics are calculated
        metrics = result["metrics"]
        assert isinstance(metrics["cagr"], float)
        assert isinstance(metrics["sharpe"], float)
        assert isinstance(metrics["max_drawdown"], float)
        assert metrics["max_drawdown"] <= 0  # Should be negative or zero

        # Verify trades structure
        trades = result["trades"]
        assert isinstance(trades, pd.DataFrame)
        expected_columns = ["timestamp", "side", "price", "equity"]
        for col in expected_columns:
            assert col in trades.columns

        # Verify that the pipeline produces reasonable results
        assert len(equity_curve) == 20  # Should have 20 days of data
        assert equity_curve["total_equity"].iloc[0] == 100000  # Initial equity

        print("✅ Smoke test passed - complete pipeline works with mock data!")


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("qbacktester.run.get_price_data")
    def test_no_trades_scenario(self, mock_get_price_data):
        """Test scenario with no trades."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        price_df = pd.DataFrame({"close": [100] * 10, "open": [100] * 10}, index=dates)

        mock_get_price_data.return_value = price_df

        params = StrategyParams(
            symbol="CONSTANT",
            start="2023-01-01",
            end="2023-01-10",
            fast_window=3,
            slow_window=6,
        )

        result = run_crossover_backtest(params)

        # Should complete without error
        assert "trades" in result
        assert len(result["trades"]) == 0  # No trades due to constant price

    @patch("qbacktester.run.get_price_data")
    def test_single_day_data(self, mock_get_price_data):
        """Test with single day of data."""
        dates = pd.date_range("2023-01-01", periods=1, freq="D")
        price_df = pd.DataFrame({"close": [100], "open": [100]}, index=dates)

        mock_get_price_data.return_value = price_df

        params = StrategyParams(
            symbol="SINGLE",
            start="2023-01-01",
            end="2023-01-01",
            fast_window=3,
            slow_window=6,
        )

        result = run_crossover_backtest(params)

        # Should complete without error
        assert "equity_curve" in result
        assert len(result["equity_curve"]) == 1
