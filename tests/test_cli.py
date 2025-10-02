"""Tests for CLI functionality."""

import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from qbacktester.cli import (
    create_equity_plot,
    create_parser,
    run_backtest_command,
    validate_arguments,
)


def create_mock_results():
    """Helper function to create mock backtest results."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    equity_curve_df = pd.DataFrame(
        {"total_equity": [100000, 100100, 100200, 100150, 100100]}, index=dates
    )

    # Create a proper mock params object
    mock_params = Mock()
    mock_params.symbol = "AAPL"
    mock_params.start = "2023-01-01"
    mock_params.end = "2023-01-05"
    mock_params.fast_window = 3
    mock_params.slow_window = 6
    mock_params.initial_cash = 100000
    mock_params.fee_bps = 1.0
    mock_params.slippage_bps = 0.5

    return {
        "params": mock_params,
        "equity_curve": equity_curve_df,
        "metrics": {
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
        },
        "trades": pd.DataFrame(columns=["timestamp", "side", "price", "equity"]),
    }


class TestCLIParser:
    """Test CLI argument parsing."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser.prog == "qbt"
        assert "SMA crossover" in parser.description

    def test_run_command_help(self):
        """Test run command help."""
        parser = create_parser()
        # Help command will exit, so we need to catch SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--help"])

    def test_run_command_required_args(self):
        """Test run command with required arguments."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2020-01-01",
                "--end",
                "2020-12-31",
                "--fast",
                "10",
                "--slow",
                "20",
            ]
        )

        assert args.symbol == "AAPL"
        assert args.start == "2020-01-01"
        assert args.end == "2020-12-31"
        assert args.fast == 10
        assert args.slow == 20
        assert args.cash == 100000  # default
        assert args.fee_bps == 1.0  # default
        assert args.slippage_bps == 0.5  # default
        assert args.plot is False  # default
        assert args.quiet is False  # default
        assert args.quick is False  # default

    def test_run_command_all_args(self):
        """Test run command with all arguments."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "SPY",
                "--start",
                "2020-01-01",
                "--end",
                "2020-12-31",
                "--fast",
                "5",
                "--slow",
                "15",
                "--cash",
                "50000",
                "--fee-bps",
                "2.0",
                "--slippage-bps",
                "1.0",
                "--plot",
                "--output-dir",
                "custom_reports",
                "--quiet",
                "--quick",
            ]
        )

        assert args.symbol == "SPY"
        assert args.cash == 50000
        assert args.fee_bps == 2.0
        assert args.slippage_bps == 1.0
        assert args.plot is True
        assert args.output_dir == "custom_reports"
        assert args.quiet is True
        assert args.quick is True


class TestArgumentValidation:
    """Test argument validation."""

    def test_validate_arguments_valid(self):
        """Test validation with valid arguments."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2020-01-01",
                "--end",
                "2020-12-31",
                "--fast",
                "10",
                "--slow",
                "20",
            ]
        )

        # Should not raise exception
        validate_arguments(args)

    def test_validate_arguments_invalid_dates(self):
        """Test validation with invalid dates."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "invalid-date",
                "--end",
                "2020-12-31",
                "--fast",
                "10",
                "--slow",
                "20",
            ]
        )

        with pytest.raises(SystemExit):
            validate_arguments(args)

    def test_validate_arguments_start_after_end(self):
        """Test validation with start date after end date."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2020-12-31",
                "--end",
                "2020-01-01",
                "--fast",
                "10",
                "--slow",
                "20",
            ]
        )

        with pytest.raises(SystemExit):
            validate_arguments(args)

    def test_validate_arguments_fast_greater_than_slow(self):
        """Test validation with fast window >= slow window."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2020-01-01",
                "--end",
                "2020-12-31",
                "--fast",
                "20",
                "--slow",
                "10",
            ]
        )

        with pytest.raises(SystemExit):
            validate_arguments(args)

    def test_validate_arguments_negative_windows(self):
        """Test validation with negative window periods."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2020-01-01",
                "--end",
                "2020-12-31",
                "--fast",
                "-10",
                "--slow",
                "20",
            ]
        )

        with pytest.raises(SystemExit):
            validate_arguments(args)

    def test_validate_arguments_negative_cash(self):
        """Test validation with negative cash."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2020-01-01",
                "--end",
                "2020-12-31",
                "--fast",
                "10",
                "--slow",
                "20",
                "--cash",
                "-1000",
            ]
        )

        with pytest.raises(SystemExit):
            validate_arguments(args)

    def test_validate_arguments_negative_bps(self):
        """Test validation with negative basis points."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2020-01-01",
                "--end",
                "2020-12-31",
                "--fast",
                "10",
                "--slow",
                "20",
                "--fee-bps",
                "-1.0",
            ]
        )

        with pytest.raises(SystemExit):
            validate_arguments(args)


class TestEquityPlot:
    """Test equity curve plotting functionality."""

    def test_create_equity_plot(self):
        """Test equity plot creation."""
        # Create mock results
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        equity_curve = pd.Series(
            [
                100000,
                100100,
                100200,
                100150,
                100100,
                100300,
                100250,
                100400,
                100350,
                100500,
            ],
            index=dates,
        )

        trades = pd.DataFrame(
            {
                "timestamp": [dates[2], dates[5]],
                "side": ["entry", "exit"],
                "price": [100, 101],
                "equity": [100200, 100300],
            }
        )

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Should not raise exception
            from qbacktester.plotting import create_equity_plot

            create_equity_plot(
                equity_curve, trades=trades, title="Test Plot", output_dir=tmp_dir
            )

            # Check that files were created in the directory
            files = os.listdir(tmp_dir)
            assert len(files) > 0
            for file in files:
                file_path = os.path.join(tmp_dir, file)
                assert os.path.getsize(file_path) > 0


class TestRunBacktestCommand:
    """Test run_backtest_command function."""

    @patch("qbacktester.cli.run_crossover_backtest")
    def test_run_backtest_command_success(self, mock_run_backtest):
        """Test successful backtest command execution."""
        mock_results = create_mock_results()
        mock_run_backtest.return_value = mock_results

        # Create mock args
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2023-01-01",
                "--end",
                "2023-01-05",
                "--fast",
                "3",
                "--slow",
                "6",
            ]
        )

        # Run command
        result = run_backtest_command(args)

        # Should succeed
        assert result == 0
        mock_run_backtest.assert_called_once()

    @patch("qbacktester.cli.run_crossover_backtest")
    def test_run_backtest_command_with_plot(self, mock_run_backtest):
        """Test backtest command with plotting."""
        mock_results = create_mock_results()
        mock_run_backtest.return_value = mock_results

        # Create mock args with plot
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2023-01-01",
                "--end",
                "2023-01-05",
                "--fast",
                "3",
                "--slow",
                "6",
                "--plot",
            ]
        )

        # Run command
        result = run_backtest_command(args)

        # Should succeed
        assert result == 0

    @patch("qbacktester.cli.run_crossover_backtest")
    def test_run_backtest_command_error(self, mock_run_backtest):
        """Test backtest command with error."""
        # Mock error
        mock_run_backtest.side_effect = Exception("Test error")

        # Create mock args
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2023-01-01",
                "--end",
                "2023-01-05",
                "--fast",
                "3",
                "--slow",
                "6",
            ]
        )

        # Run command
        result = run_backtest_command(args)

        # Should fail
        assert result == 1

    @patch("qbacktester.cli.run_crossover_backtest")
    def test_run_backtest_command_keyboard_interrupt(self, mock_run_backtest):
        """Test backtest command with keyboard interrupt."""
        # Mock keyboard interrupt
        mock_run_backtest.side_effect = KeyboardInterrupt()

        # Create mock args
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2023-01-01",
                "--end",
                "2023-01-05",
                "--fast",
                "3",
                "--slow",
                "6",
            ]
        )

        # Run command
        result = run_backtest_command(args)

        # Should fail with keyboard interrupt code
        assert result == 1


class TestCLIIntegration:
    """Test CLI integration."""

    @patch("qbacktester.cli.run_crossover_backtest")
    def test_cli_main_success(self, mock_run_backtest):
        """Test CLI main function success."""
        from qbacktester.cli import main

        mock_results = create_mock_results()
        mock_run_backtest.return_value = mock_results

        # Test with valid arguments
        with patch(
            "sys.argv",
            [
                "qbt",
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2023-01-01",
                "--end",
                "2023-01-05",
                "--fast",
                "3",
                "--slow",
                "6",
            ],
        ):
            result = main()
            assert result == 0

    def test_cli_main_no_command(self):
        """Test CLI main function with no command."""
        from qbacktester.cli import main

        # Test with no command
        with patch("sys.argv", ["qbt"]):
            result = main()
            assert result == 1

    def test_cli_main_unknown_command(self):
        """Test CLI main function with unknown command."""
        from qbacktester.cli import main

        # Test with unknown command
        with patch("sys.argv", ["qbt", "unknown"]):
            with pytest.raises(SystemExit):
                main()


class TestCLIOutput:
    """Test CLI output functionality."""

    @patch("qbacktester.cli.run_crossover_backtest")
    def test_quiet_mode(self, mock_run_backtest):
        """Test quiet mode suppresses output."""
        mock_results = create_mock_results()
        mock_run_backtest.return_value = mock_results

        # Create mock args with quiet mode
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2023-01-01",
                "--end",
                "2023-01-05",
                "--fast",
                "3",
                "--slow",
                "6",
                "--quiet",
            ]
        )

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = run_backtest_command(args)

        # Should succeed
        assert result == 0
        # In quiet mode, progress messages should be suppressed
        # but the final report should still be printed
        output = mock_stdout.getvalue()
        assert "📊 Fetching data" not in output  # Progress message suppressed
        assert "Backtest Results" in output or "AAPL" in output  # Report still shown

    @patch("qbacktester.cli.run_crossover_backtest")
    def test_quick_mode(self, mock_run_backtest):
        """Test quick mode shows only summary."""
        mock_results = create_mock_results()
        mock_run_backtest.return_value = mock_results

        # Create mock args with quick mode
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--symbol",
                "AAPL",
                "--start",
                "2023-01-01",
                "--end",
                "2023-01-05",
                "--fast",
                "3",
                "--slow",
                "6",
                "--quick",
            ]
        )

        # Capture stdout
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = run_backtest_command(args)

        # Should succeed
        assert result == 0
        # Quick mode should show summary instead of full report
        output = mock_stdout.getvalue()
        assert "AAPL" in output  # Quick summary format
        assert "Strategy Parameters" not in output  # Full report not shown
