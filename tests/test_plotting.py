"""Tests for plotting module."""

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from qbacktester.plotting import (
    create_drawdown_plot,
    create_equity_plot,
    create_price_signals_plot,
    plot_drawdown,
    plot_equity,
    plot_price_signals,
    save_figure,
)


class TestPlottingFunctions:
    """Test basic plotting functions that return Figure objects."""

    def setup_method(self):
        """Set up test data."""
        # Create test equity curve
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 20)
        self.equity_curve = pd.Series(100000 * (1 + returns).cumprod(), index=dates)

        # Create test price data
        self.price_df = pd.DataFrame(
            {
                "close": 100 * (1 + returns).cumprod(),
                "open": 100 * (1 + returns).cumprod() * 1.001,
                "high": 100 * (1 + returns).cumprod() * 1.005,
                "low": 100 * (1 + returns).cumprod() * 0.995,
                "volume": np.random.randint(1000, 10000, 20),
            },
            index=dates,
        )

        # Create test signals and moving averages
        self.signals = pd.Series(
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0], index=dates
        )
        self.fast_ma = self.price_df["close"].rolling(3).mean()
        self.slow_ma = self.price_df["close"].rolling(5).mean()

    def test_plot_equity_returns_figure(self):
        """Test that plot_equity returns a Figure object."""
        fig = plot_equity(self.equity_curve, "Test Equity Curve")

        assert isinstance(fig, Figure)
        assert fig.get_axes() is not None
        assert len(fig.get_axes()) == 1

        # Clean up
        plt.close(fig)

    def test_plot_equity_with_minimal_data(self):
        """Test plot_equity with minimal data."""
        minimal_equity = pd.Series(
            [100000, 101000], index=pd.date_range("2023-01-01", periods=2, freq="D")
        )

        fig = plot_equity(minimal_equity, "Minimal Test")

        assert isinstance(fig, Figure)
        assert fig.get_axes() is not None

        # Clean up
        plt.close(fig)

    def test_plot_drawdown_returns_figure(self):
        """Test that plot_drawdown returns a Figure object."""
        fig = plot_drawdown(self.equity_curve, "Test Drawdown")

        assert isinstance(fig, Figure)
        assert fig.get_axes() is not None
        assert len(fig.get_axes()) == 1

        # Clean up
        plt.close(fig)

    def test_plot_drawdown_with_minimal_data(self):
        """Test plot_drawdown with minimal data."""
        minimal_equity = pd.Series(
            [100000, 101000], index=pd.date_range("2023-01-01", periods=2, freq="D")
        )

        fig = plot_drawdown(minimal_equity, "Minimal Drawdown")

        assert isinstance(fig, Figure)
        assert fig.get_axes() is not None

        # Clean up
        plt.close(fig)

    def test_plot_price_signals_returns_figure(self):
        """Test that plot_price_signals returns a Figure object."""
        fig = plot_price_signals(
            self.price_df,
            self.signals,
            self.fast_ma,
            self.slow_ma,
            "Test Price Signals",
        )

        assert isinstance(fig, Figure)
        assert fig.get_axes() is not None
        assert len(fig.get_axes()) == 2  # Two subplots

        # Clean up
        plt.close(fig)

    def test_plot_price_signals_with_minimal_data(self):
        """Test plot_price_signals with minimal data."""
        minimal_dates = pd.date_range("2023-01-01", periods=3, freq="D")
        minimal_price_df = pd.DataFrame({"close": [100, 101, 102]}, index=minimal_dates)
        minimal_signals = pd.Series([0, 1, 0], index=minimal_dates)
        minimal_fast = pd.Series([100, 100.5, 101], index=minimal_dates)
        minimal_slow = pd.Series([100, 100.3, 100.7], index=minimal_dates)

        fig = plot_price_signals(
            minimal_price_df,
            minimal_signals,
            minimal_fast,
            minimal_slow,
            "Minimal Price Signals",
        )

        assert isinstance(fig, Figure)
        assert fig.get_axes() is not None
        assert len(fig.get_axes()) == 2

        # Clean up
        plt.close(fig)

    def test_plot_functions_handle_empty_data(self):
        """Test that plotting functions handle empty data gracefully."""
        empty_equity = pd.Series([], dtype=float)
        empty_price_df = pd.DataFrame(columns=["close"])
        empty_signals = pd.Series([], dtype=int)
        empty_ma = pd.Series([], dtype=float)

        # These should raise meaningful exceptions for empty data
        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            plot_equity(empty_equity, "Empty Equity")

        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            plot_drawdown(empty_equity, "Empty Drawdown")

        with pytest.raises(ValueError, match="Input data cannot be empty"):
            plot_price_signals(
                empty_price_df, empty_signals, empty_ma, empty_ma, "Empty Price Signals"
            )


class TestSaveFigure:
    """Test save_figure function."""

    def test_save_figure_creates_file(self):
        """Test that save_figure creates a file."""
        # Create a simple figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Figure")

        # Use temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = save_figure(fig, "test_plot", temp_dir)

            # Check that file was created
            assert os.path.exists(file_path)
            assert file_path.endswith(".png")
            assert os.path.getsize(file_path) > 0

        # Clean up
        plt.close(fig)

    def test_save_figure_creates_directory(self):
        """Test that save_figure creates output directory if it doesn't exist."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 2])

        # Use temporary directory with subdirectory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "plots")
            file_path = save_figure(fig, "test_plot", output_dir)

            # Check that directory was created
            assert os.path.exists(output_dir)
            assert os.path.exists(file_path)

        # Clean up
        plt.close(fig)


class TestCreatePlotFunctions:
    """Test create_*_plot functions that save files."""

    def setup_method(self):
        """Set up test data."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 10)
        self.equity_curve = pd.Series(100000 * (1 + returns).cumprod(), index=dates)

        self.price_df = pd.DataFrame(
            {"close": 100 * (1 + returns).cumprod()}, index=dates
        )

        self.signals = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], index=dates)
        self.fast_ma = self.price_df["close"].rolling(3).mean()
        self.slow_ma = self.price_df["close"].rolling(5).mean()

    def test_create_equity_plot_saves_file(self):
        """Test that create_equity_plot saves a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = create_equity_plot(
                self.equity_curve, title="Test Equity", output_dir=temp_dir
            )

            assert os.path.exists(file_path)
            assert file_path.endswith(".png")
            assert os.path.getsize(file_path) > 0

    def test_create_equity_plot_with_trades(self):
        """Test create_equity_plot with trade data."""
        # Create mock trade data
        trades = pd.DataFrame(
            {
                "timestamp": [self.equity_curve.index[2], self.equity_curve.index[5]],
                "side": ["entry", "exit"],
                "equity": [self.equity_curve.iloc[2], self.equity_curve.iloc[5]],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = create_equity_plot(
                self.equity_curve,
                trades=trades,
                title="Test Equity with Trades",
                output_dir=temp_dir,
            )

            assert os.path.exists(file_path)
            assert file_path.endswith(".png")
            assert os.path.getsize(file_path) > 0

    def test_create_drawdown_plot_saves_file(self):
        """Test that create_drawdown_plot saves a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = create_drawdown_plot(
                self.equity_curve, title="Test Drawdown", output_dir=temp_dir
            )

            assert os.path.exists(file_path)
            assert file_path.endswith(".png")
            assert os.path.getsize(file_path) > 0

    def test_create_price_signals_plot_saves_file(self):
        """Test that create_price_signals_plot saves a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = create_price_signals_plot(
                self.price_df,
                self.signals,
                self.fast_ma,
                self.slow_ma,
                title="Test Price Signals",
                output_dir=temp_dir,
            )

            assert os.path.exists(file_path)
            assert file_path.endswith(".png")
            assert os.path.getsize(file_path) > 0

    def test_create_plots_handle_minimal_data(self):
        """Test create_*_plot functions with minimal data."""
        minimal_dates = pd.date_range("2023-01-01", periods=2, freq="D")
        minimal_equity = pd.Series([100000, 101000], index=minimal_dates)
        minimal_price_df = pd.DataFrame({"close": [100, 101]}, index=minimal_dates)
        minimal_signals = pd.Series([0, 1], index=minimal_dates)
        minimal_fast = pd.Series([100, 100.5], index=minimal_dates)
        minimal_slow = pd.Series([100, 100.3], index=minimal_dates)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test all create functions with minimal data
            equity_path = create_equity_plot(minimal_equity, output_dir=temp_dir)
            drawdown_path = create_drawdown_plot(minimal_equity, output_dir=temp_dir)
            price_path = create_price_signals_plot(
                minimal_price_df,
                minimal_signals,
                minimal_fast,
                minimal_slow,
                output_dir=temp_dir,
            )

            # All should create files
            assert os.path.exists(equity_path)
            assert os.path.exists(drawdown_path)
            assert os.path.exists(price_path)


class TestPlottingEdgeCases:
    """Test edge cases and error handling."""

    def test_plotting_with_nan_values(self):
        """Test plotting functions with NaN values."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        equity_with_nan = pd.Series(
            [100000, np.nan, 101000, 102000, np.nan], index=dates
        )

        # Should handle NaN values gracefully
        fig = plot_equity(equity_with_nan, "Equity with NaN")
        assert isinstance(fig, Figure)
        plt.close(fig)

        fig = plot_drawdown(equity_with_nan, "Drawdown with NaN")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plotting_with_constant_values(self):
        """Test plotting functions with constant values."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        constant_equity = pd.Series([100000] * 5, index=dates)

        # Should handle constant values gracefully
        fig = plot_equity(constant_equity, "Constant Equity")
        assert isinstance(fig, Figure)
        plt.close(fig)

        fig = plot_drawdown(constant_equity, "Constant Drawdown")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plotting_with_negative_values(self):
        """Test plotting functions with negative values."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        negative_equity = pd.Series([100000, 95000, 90000, 85000, 80000], index=dates)

        # Should handle negative values gracefully
        fig = plot_equity(negative_equity, "Declining Equity")
        assert isinstance(fig, Figure)
        plt.close(fig)

        fig = plot_drawdown(negative_equity, "Declining Drawdown")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_save_figure_with_invalid_directory(self):
        """Test save_figure with invalid directory path."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 2])

        # Should handle invalid directory gracefully
        with pytest.raises((OSError, PermissionError)):
            save_figure(fig, "test", "/invalid/path/that/does/not/exist")

        plt.close(fig)


class TestPlottingIntegration:
    """Test integration between plotting functions."""

    def test_all_plotting_functions_work_together(self):
        """Test that all plotting functions can be used together."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 20)
        equity_curve = pd.Series(100000 * (1 + returns).cumprod(), index=dates)

        price_df = pd.DataFrame({"close": 100 * (1 + returns).cumprod()}, index=dates)

        signals = pd.Series(
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0], index=dates
        )
        fast_ma = price_df["close"].rolling(3).mean()
        slow_ma = price_df["close"].rolling(5).mean()

        # Create all types of plots
        fig1 = plot_equity(equity_curve, "Integration Test Equity")
        fig2 = plot_drawdown(equity_curve, "Integration Test Drawdown")
        fig3 = plot_price_signals(
            price_df, signals, fast_ma, slow_ma, "Integration Test Price"
        )

        # All should be valid Figure objects
        assert isinstance(fig1, Figure)
        assert isinstance(fig2, Figure)
        assert isinstance(fig3, Figure)

        # Clean up
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    def test_create_plots_in_sequence(self):
        """Test creating multiple plots in sequence."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        equity_curve = pd.Series(
            100000 * (1 + np.random.normal(0.001, 0.02, 10)).cumprod(), index=dates
        )

        price_df = pd.DataFrame({"close": equity_curve.values}, index=dates)
        signals = pd.Series([0, 1, 1, 0, 0, 1, 1, 0, 0, 1], index=dates)
        fast_ma = price_df["close"].rolling(3).mean()
        slow_ma = price_df["close"].rolling(5).mean()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create all types of plots in sequence
            equity_path = create_equity_plot(equity_curve, output_dir=temp_dir)
            drawdown_path = create_drawdown_plot(equity_curve, output_dir=temp_dir)
            price_path = create_price_signals_plot(
                price_df, signals, fast_ma, slow_ma, output_dir=temp_dir
            )

            # All should create files successfully
            assert os.path.exists(equity_path)
            assert os.path.exists(drawdown_path)
            assert os.path.exists(price_path)

            # Files should have different names
            assert equity_path != drawdown_path
            assert drawdown_path != price_path
            assert equity_path != price_path
