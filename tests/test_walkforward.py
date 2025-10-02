"""Tests for walkforward module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from qbacktester.walkforward import (
    _calculate_parameter_stability,
    _calculate_performance_consistency,
    _generate_walkforward_windows,
    print_walkforward_summary,
    save_walkforward_results,
    walkforward_crossover,
)


class TestWalkforwardCrossover:
    """Test walkforward_crossover function."""

    def setup_method(self):
        """Set up test data."""
        self.symbol = "TEST"
        self.start = "2020-01-01"
        self.end = "2023-12-31"
        self.fast_grid = [10, 20]
        self.slow_grid = [50, 100]

    @patch("qbacktester.walkforward.grid_search")
    @patch("qbacktester.walkforward.run_crossover_backtest")
    def test_walkforward_crossover_basic(self, mock_run_backtest, mock_grid_search):
        """Test basic walk-forward analysis functionality."""
        # Mock optimization results
        mock_opt_results = pd.DataFrame(
            {
                "fast": [10, 20],
                "slow": [50, 100],
                "sharpe": [1.2, 1.1],
                "max_dd": [-0.05, -0.04],
                "cagr": [0.15, 0.14],
                "equity_final": [115000, 114000],
                "volatility": [0.12, 0.11],
                "calmar": [3.0, 3.5],
                "hit_rate": [0.6, 0.65],
                "total_trades": [10, 12],
                "success": [True, True],
                "error": [None, None],
            }
        )
        mock_grid_search.return_value = mock_opt_results

        # Mock backtest results
        mock_backtest_results = {
            "metrics": {
                "sharpe": 1.2,
                "max_drawdown": -0.05,
                "cagr": 0.15,
                "volatility": 0.12,
                "calmar": 3.0,
                "hit_rate": 0.6,
            },
            "equity_curve": pd.DataFrame(
                {"total_equity": [100000, 101000, 102000, 103000, 104000]},
                index=pd.date_range("2020-01-01", periods=5, freq="D"),
            ),
            "trades": pd.DataFrame(columns=["timestamp", "side", "price", "equity"]),
        }
        mock_run_backtest.return_value = mock_backtest_results

        # Run walk-forward analysis
        results = walkforward_crossover(
            symbol=self.symbol,
            start=self.start,
            end=self.end,
            in_sample_years=1,
            out_sample_years=1,
            fast_grid=self.fast_grid,
            slow_grid=self.slow_grid,
            verbose=False,
        )

        # Check results structure
        assert "windows" in results
        assert "equity_curve" in results
        assert "summary" in results
        assert "parameters" in results
        assert "dates" in results

        # Check windows
        assert len(results["windows"]) > 0
        for window in results["windows"]:
            assert "window" in window
            assert "is_start" in window
            assert "is_end" in window
            assert "oos_start" in window
            assert "oos_end" in window
            assert "best_fast" in window
            assert "best_slow" in window
            assert "success" in window

    def test_walkforward_crossover_validation(self):
        """Test walk-forward analysis input validation."""
        # Test invalid date range
        with pytest.raises(ValueError, match="Start date must be before end date"):
            walkforward_crossover(
                symbol=self.symbol,
                start="2023-01-01",
                end="2020-01-01",
                in_sample_years=1,
                out_sample_years=1,
                fast_grid=self.fast_grid,
                slow_grid=self.slow_grid,
                verbose=False,
            )

        # Test invalid years
        with pytest.raises(
            ValueError, match="In-sample and out-of-sample years must be positive"
        ):
            walkforward_crossover(
                symbol=self.symbol,
                start=self.start,
                end=self.end,
                in_sample_years=0,
                out_sample_years=1,
                fast_grid=self.fast_grid,
                slow_grid=self.slow_grid,
                verbose=False,
            )

        # Test insufficient data
        with pytest.raises(ValueError, match="Total analysis period must be longer"):
            walkforward_crossover(
                symbol=self.symbol,
                start="2020-01-01",
                end="2020-12-31",
                in_sample_years=2,
                out_sample_years=2,
                fast_grid=self.fast_grid,
                slow_grid=self.slow_grid,
                verbose=False,
            )

    @patch("qbacktester.walkforward.grid_search")
    @patch("qbacktester.walkforward.run_crossover_backtest")
    def test_walkforward_crossover_error_handling(
        self, mock_run_backtest, mock_grid_search
    ):
        """Test walk-forward analysis error handling."""
        # Mock optimization failure
        mock_grid_search.side_effect = Exception("Optimization failed")

        # Mock the window generation to return at least one window
        with patch(
            "qbacktester.walkforward._generate_walkforward_windows"
        ) as mock_windows:
            mock_windows.return_value = [
                {
                    "is_start": pd.Timestamp("2020-01-01"),
                    "is_end": pd.Timestamp("2021-01-01"),
                    "oos_start": pd.Timestamp("2021-01-01"),
                    "oos_end": pd.Timestamp("2022-01-01"),
                }
            ]

            results = walkforward_crossover(
                symbol=self.symbol,
                start=self.start,
                end=self.end,
                in_sample_years=1,
                out_sample_years=1,
                fast_grid=self.fast_grid,
                slow_grid=self.slow_grid,
                verbose=False,
            )

        # Should handle errors gracefully
        assert len(results["windows"]) > 0
        # All windows should fail
        for window in results["windows"]:
            assert not window["success"]
            assert window["error"] is not None


class TestGenerateWalkforwardWindows:
    """Test _generate_walkforward_windows function."""

    def test_generate_windows_basic(self):
        """Test basic window generation."""
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2023-12-31")

        windows = _generate_walkforward_windows(
            start_date, end_date, in_sample_years=1, out_sample_years=1
        )

        assert len(windows) > 0

        for window in windows:
            assert "is_start" in window
            assert "is_end" in window
            assert "oos_start" in window
            assert "oos_end" in window

            # Check that IS comes before OOS
            assert window["is_start"] < window["is_end"]
            assert window["is_end"] <= window["oos_start"]
            assert window["oos_start"] < window["oos_end"]

            # Check that IS period is 1 year
            assert (window["is_end"] - window["is_start"]).days >= 365
            assert (window["is_end"] - window["is_start"]).days <= 366

            # Check that OOS period is 1 year
            assert (window["oos_end"] - window["oos_start"]).days >= 365
            assert (window["oos_end"] - window["oos_start"]).days <= 366

    def test_generate_windows_insufficient_data(self):
        """Test window generation with insufficient data."""
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2020-12-31")

        windows = _generate_walkforward_windows(
            start_date, end_date, in_sample_years=1, out_sample_years=1
        )

        # Should return empty list for insufficient data
        assert len(windows) == 0

    def test_generate_windows_multiple_years(self):
        """Test window generation with multiple years."""
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2025-12-31")

        windows = _generate_walkforward_windows(
            start_date, end_date, in_sample_years=2, out_sample_years=1
        )

        assert len(windows) > 0

        for window in windows:
            # Check that IS period is 2 years
            assert (window["is_end"] - window["is_start"]).days >= 730
            assert (window["is_end"] - window["is_start"]).days <= 732

            # Check that OOS period is 1 year
            assert (window["oos_end"] - window["oos_start"]).days >= 365
            assert (window["oos_end"] - window["oos_start"]).days <= 366


class TestParameterStability:
    """Test parameter stability calculation."""

    def test_calculate_parameter_stability_basic(self):
        """Test basic parameter stability calculation."""
        successful_windows = [
            {"best_fast": 10, "best_slow": 50},
            {"best_fast": 10, "best_slow": 50},
            {"best_fast": 20, "best_slow": 100},
            {"best_fast": 10, "best_slow": 50},
        ]

        stability = _calculate_parameter_stability(successful_windows)

        assert "fast_mean" in stability
        assert "fast_std" in stability
        assert "fast_cv" in stability
        assert "slow_mean" in stability
        assert "slow_std" in stability
        assert "slow_cv" in stability
        assert "parameter_changes" in stability

        # Check that means are calculated correctly
        assert abs(stability["fast_mean"] - 12.5) < 0.1  # (10+10+20+10)/4
        assert abs(stability["slow_mean"] - 62.5) < 0.1  # (50+50+100+50)/4

        # Check parameter changes (2 changes: 10->20, 20->10)
        assert stability["parameter_changes"] == 2

    def test_calculate_parameter_stability_empty(self):
        """Test parameter stability calculation with empty input."""
        stability = _calculate_parameter_stability([])
        assert stability == {}

    def test_calculate_parameter_stability_single_window(self):
        """Test parameter stability calculation with single window."""
        successful_windows = [{"best_fast": 10, "best_slow": 50}]

        stability = _calculate_parameter_stability(successful_windows)

        assert stability["fast_mean"] == 10
        assert stability["fast_std"] == 0
        assert stability["fast_cv"] == 0
        assert stability["parameter_changes"] == 0


class TestPerformanceConsistency:
    """Test performance consistency calculation."""

    def test_calculate_performance_consistency_basic(self):
        """Test basic performance consistency calculation."""
        successful_windows = [
            {"oos_metrics": {"cagr": 0.15, "sharpe": 1.2, "max_drawdown": -0.05}},
            {"oos_metrics": {"cagr": 0.10, "sharpe": 1.0, "max_drawdown": -0.03}},
            {"oos_metrics": {"cagr": 0.20, "sharpe": 1.5, "max_drawdown": -0.08}},
        ]

        consistency = _calculate_performance_consistency(successful_windows)

        assert "cagr_mean" in consistency
        assert "cagr_std" in consistency
        assert "cagr_cv" in consistency
        assert "sharpe_mean" in consistency
        assert "sharpe_std" in consistency
        assert "sharpe_cv" in consistency
        assert "max_dd_mean" in consistency
        assert "max_dd_std" in consistency
        assert "positive_periods" in consistency
        assert "win_rate" in consistency

        # Check that means are calculated correctly
        assert abs(consistency["cagr_mean"] - 0.15) < 0.01  # (0.15+0.10+0.20)/3
        assert abs(consistency["sharpe_mean"] - 1.233) < 0.01  # (1.2+1.0+1.5)/3

        # All periods are positive
        assert consistency["positive_periods"] == 3
        assert consistency["win_rate"] == 1.0

    def test_calculate_performance_consistency_empty(self):
        """Test performance consistency calculation with empty input."""
        consistency = _calculate_performance_consistency([])
        assert consistency == {}

    def test_calculate_performance_consistency_mixed_results(self):
        """Test performance consistency calculation with mixed results."""
        successful_windows = [
            {"oos_metrics": {"cagr": 0.15, "sharpe": 1.2, "max_drawdown": -0.05}},
            {
                "oos_metrics": {
                    "cagr": -0.05,  # Negative return
                    "sharpe": 0.5,
                    "max_drawdown": -0.10,
                }
            },
        ]

        consistency = _calculate_performance_consistency(successful_windows)

        assert consistency["positive_periods"] == 1
        assert consistency["win_rate"] == 0.5


class TestPrintWalkforwardSummary:
    """Test print_walkforward_summary function."""

    def test_print_walkforward_summary(self, capsys):
        """Test printing walk-forward summary."""
        # Create mock results
        results = {
            "windows": [
                {
                    "window": 1,
                    "is_start": pd.Timestamp("2020-01-01"),
                    "is_end": pd.Timestamp("2021-01-01"),
                    "oos_start": pd.Timestamp("2021-01-01"),
                    "oos_end": pd.Timestamp("2022-01-01"),
                    "best_fast": 10,
                    "best_slow": 50,
                    "success": True,
                    "oos_metrics": {"cagr": 0.15, "sharpe": 1.2, "max_drawdown": -0.05},
                }
            ],
            "summary": {
                "total_windows": 1,
                "successful_windows": 1,
                "success_rate": 1.0,
                "overall_metrics": {
                    "cagr": 0.15,
                    "sharpe": 1.2,
                    "max_drawdown": -0.05,
                    "calmar": 3.0,
                },
                "parameter_stability": {
                    "fast_mean": 10,
                    "fast_std": 0,
                    "fast_cv": 0,
                    "slow_mean": 50,
                    "slow_std": 0,
                    "slow_cv": 0,
                    "parameter_changes": 0,
                },
                "performance_consistency": {
                    "cagr_mean": 0.15,
                    "cagr_std": 0,
                    "sharpe_mean": 1.2,
                    "sharpe_std": 0,
                    "win_rate": 1.0,
                },
            },
        }

        print_walkforward_summary(results, top_n=1)

        captured = capsys.readouterr()
        output = captured.out

        # Check that summary was printed
        assert "Walk-Forward Analysis Summary" in output
        assert "Overall Performance" in output
        assert "Parameter Stability" in output
        assert "Performance Consistency" in output
        assert "Top 1 Windows by OOS Sharpe" in output


class TestSaveWalkforwardResults:
    """Test save_walkforward_results function."""

    def test_save_walkforward_results(self):
        """Test saving walk-forward results to CSV files."""
        # Create mock results
        results = {
            "windows": [
                {
                    "window": 1,
                    "is_start": pd.Timestamp("2020-01-01"),
                    "is_end": pd.Timestamp("2021-01-01"),
                    "oos_start": pd.Timestamp("2021-01-01"),
                    "oos_end": pd.Timestamp("2022-01-01"),
                    "best_fast": 10,
                    "best_slow": 50,
                    "success": True,
                    "oos_metrics": {"cagr": 0.15, "sharpe": 1.2, "max_drawdown": -0.05},
                }
            ],
            "equity_curve": pd.Series(
                [100000, 101000, 102000],
                index=pd.date_range("2021-01-01", periods=3, freq="D"),
            ),
            "summary": {
                "overall_metrics": {"cagr": 0.15, "sharpe": 1.2, "max_drawdown": -0.05},
                "parameter_stability": {"fast_mean": 10, "fast_std": 0},
                "performance_consistency": {"cagr_mean": 0.15, "win_rate": 1.0},
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = save_walkforward_results(results, "TEST", temp_dir)

            # Check that files were created
            assert os.path.exists(file_path)
            assert file_path.endswith(".csv")
            assert "walkforward_windows_test.csv" in file_path

            # Check that other files were created
            base_path = Path(temp_dir)
            assert (base_path / "walkforward_equity_test.csv").exists()
            assert (base_path / "walkforward_summary_test.csv").exists()

            # Check file contents
            windows_df = pd.read_csv(file_path)
            assert len(windows_df) == 1
            assert "window" in windows_df.columns
            assert "best_fast" in windows_df.columns
            assert "best_slow" in windows_df.columns

    def test_save_walkforward_results_creates_directory(self):
        """Test that save_walkforward_results creates output directory."""
        results = {
            "windows": [],
            "equity_curve": pd.Series(dtype=float),
            "summary": {
                "overall_metrics": {},
                "parameter_stability": {},
                "performance_consistency": {},
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "new_dir")
            file_path = save_walkforward_results(results, "TEST", output_dir)

            # Check that directory was created
            assert os.path.exists(output_dir)
            assert os.path.exists(file_path)


class TestWalkforwardIntegration:
    """Test walk-forward analysis integration."""

    @patch("qbacktester.walkforward.grid_search")
    @patch("qbacktester.walkforward.run_crossover_backtest")
    def test_walkforward_integration(self, mock_run_backtest, mock_grid_search):
        """Test complete walk-forward analysis integration."""
        # Mock optimization results
        mock_opt_results = pd.DataFrame(
            {
                "fast": [10, 20],
                "slow": [50, 100],
                "sharpe": [1.2, 1.1],
                "max_dd": [-0.05, -0.04],
                "cagr": [0.15, 0.14],
                "equity_final": [115000, 114000],
                "volatility": [0.12, 0.11],
                "calmar": [3.0, 3.5],
                "hit_rate": [0.6, 0.65],
                "total_trades": [10, 12],
                "success": [True, True],
                "error": [None, None],
            }
        )
        mock_grid_search.return_value = mock_opt_results

        # Mock backtest results
        mock_backtest_results = {
            "metrics": {
                "sharpe": 1.2,
                "max_drawdown": -0.05,
                "cagr": 0.15,
                "volatility": 0.12,
                "calmar": 3.0,
                "hit_rate": 0.6,
            },
            "equity_curve": pd.DataFrame(
                {"total_equity": [100000, 101000, 102000, 103000, 104000]},
                index=pd.date_range("2020-01-01", periods=5, freq="D"),
            ),
            "trades": pd.DataFrame(columns=["timestamp", "side", "price", "equity"]),
        }
        mock_run_backtest.return_value = mock_backtest_results

        # Run walk-forward analysis
        results = walkforward_crossover(
            symbol="TEST",
            start="2020-01-01",
            end="2022-12-31",
            in_sample_years=1,
            out_sample_years=1,
            fast_grid=[10, 20],
            slow_grid=[50, 100],
            verbose=False,
        )

        # Check that grid_search was called for each window
        assert mock_grid_search.call_count > 0

        # Check that run_crossover_backtest was called for each successful window
        assert mock_run_backtest.call_count > 0

        # Check results structure
        assert "windows" in results
        assert "equity_curve" in results
        assert "summary" in results

        # Check that summary contains expected metrics
        summary = results["summary"]
        assert "total_windows" in summary
        assert "successful_windows" in summary
        assert "success_rate" in summary
        assert "overall_metrics" in summary
        assert "parameter_stability" in summary
        assert "performance_consistency" in summary


class TestWalkforwardEdgeCases:
    """Test walk-forward analysis edge cases."""

    def test_walkforward_single_window(self):
        """Test walk-forward analysis with single window."""
        with (
            patch("qbacktester.walkforward.grid_search") as mock_grid_search,
            patch(
                "qbacktester.walkforward.run_crossover_backtest"
            ) as mock_run_backtest,
            patch(
                "qbacktester.walkforward._generate_walkforward_windows"
            ) as mock_windows,
        ):

            # Mock results
            mock_opt_results = pd.DataFrame(
                {
                    "fast": [10],
                    "slow": [50],
                    "sharpe": [1.2],
                    "max_dd": [-0.05],
                    "cagr": [0.15],
                    "equity_final": [115000],
                    "volatility": [0.12],
                    "calmar": [3.0],
                    "hit_rate": [0.6],
                    "total_trades": [10],
                    "success": [True],
                    "error": [None],
                }
            )
            mock_grid_search.return_value = mock_opt_results

            mock_backtest_results = {
                "metrics": {
                    "sharpe": 1.2,
                    "max_drawdown": -0.05,
                    "cagr": 0.15,
                    "volatility": 0.12,
                    "calmar": 3.0,
                    "hit_rate": 0.6,
                },
                "equity_curve": pd.DataFrame(
                    {"total_equity": [100000, 101000, 102000]},
                    index=pd.date_range("2020-01-01", periods=3, freq="D"),
                ),
                "trades": pd.DataFrame(
                    columns=["timestamp", "side", "price", "equity"]
                ),
            }
            mock_run_backtest.return_value = mock_backtest_results

            # Mock window generation to return single window
            mock_windows.return_value = [
                {
                    "is_start": pd.Timestamp("2020-01-01"),
                    "is_end": pd.Timestamp("2021-01-01"),
                    "oos_start": pd.Timestamp("2021-01-01"),
                    "oos_end": pd.Timestamp("2022-01-01"),
                }
            ]

            results = walkforward_crossover(
                symbol="TEST",
                start="2020-01-01",
                end="2023-12-31",
                in_sample_years=1,
                out_sample_years=1,
                fast_grid=[10],
                slow_grid=[50],
                verbose=False,
            )

        assert len(results["windows"]) == 1
        assert results["windows"][0]["success"] == True

    def test_walkforward_no_successful_windows(self):
        """Test walk-forward analysis with no successful windows."""
        with patch("qbacktester.walkforward.grid_search") as mock_grid_search:
            # Mock optimization failure
            mock_grid_search.side_effect = Exception("Optimization failed")

        results = walkforward_crossover(
            symbol="TEST",
            start="2020-01-01",
            end="2023-12-31",
            in_sample_years=1,
            out_sample_years=1,
            fast_grid=[10],
            slow_grid=[50],
            verbose=False,
        )

        assert len(results["windows"]) > 0
        assert all(not w["success"] for w in results["windows"])
        assert results["summary"]["success_rate"] == 0.0
