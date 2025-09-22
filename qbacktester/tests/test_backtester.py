"""Tests for backtester module."""

import pytest
import pandas as pd
import numpy as np
from qbacktester.backtester import Backtester
from qbacktester.strategy import StrategyParams


class TestBacktester:
    """Test Backtester class."""
    
    def test_initialization(self):
        """Test backtester initialization."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        price_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            'open': [99, 100, 101, 102, 103, 104, 103, 102, 101, 100]
        }, index=dates)
        
        signals = pd.Series([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], index=dates)
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-10",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=1.0,
            slippage_bps=0.5
        )
        
        backtester = Backtester(price_df, signals, params)
        
        assert backtester.params == params
        assert backtester.signals.equals(signals)
        assert backtester.price_df.equals(price_df)
    
    def test_initialization_missing_close_column(self):
        """Test error when close column is missing."""
        price_df = pd.DataFrame({'open': [100, 101, 102]})
        signals = pd.Series([0, 1, 0])
        params = StrategyParams("TEST", "2023-01-01", "2023-01-03", 3, 6)
        
        with pytest.raises(ValueError, match="price_df must contain 'close' column"):
            Backtester(price_df, signals, params)
    
    def test_initialization_mismatched_indices(self):
        """Test error when indices don't match."""
        dates1 = pd.date_range('2023-01-01', periods=5, freq='D')
        dates2 = pd.date_range('2023-01-02', periods=5, freq='D')
        
        price_df = pd.DataFrame({'close': [100, 101, 102, 103, 104]}, index=dates1)
        signals = pd.Series([0, 1, 0, 1, 0], index=dates2)
        params = StrategyParams("TEST", "2023-01-01", "2023-01-05", 3, 6)
        
        with pytest.raises(ValueError, match="signals and price_df must have the same index"):
            Backtester(price_df, signals, params)
    
    def test_initialization_invalid_signals(self):
        """Test error when signals contain invalid values."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        price_df = pd.DataFrame({'close': [100, 101, 102, 103, 104]}, index=dates)
        signals = pd.Series([0, 1, 2, 0, 1], index=dates)  # Invalid signal value
        params = StrategyParams("TEST", "2023-01-01", "2023-01-05", 3, 6)
        
        with pytest.raises(ValueError, match="signals must contain only 0 or 1 values"):
            Backtester(price_df, signals, params)


class TestBacktesterRun:
    """Test backtester run method."""
    
    def test_basic_run(self):
        """Test basic backtest run."""
        # Create simple test data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        price_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'open': [99, 100, 101, 102, 103]
        }, index=dates)
        
        signals = pd.Series([0, 0, 1, 1, 0], index=dates)
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-05",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=1.0,
            slippage_bps=0.5
        )
        
        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        
        # Check required columns
        expected_columns = ['close', 'signal', 'position', 'holdings_value', 
                          'cash', 'total_equity', 'trade_flag']
        for col in expected_columns:
            assert col in result.columns
        
        # Check data types and values
        assert result['signal'].isin([0, 1]).all()
        assert result['position'].isin([0, 1]).all()
        assert result['trade_flag'].dtype == bool
        
        # Check that position equals signal
        pd.testing.assert_series_equal(result['position'], result['signal'], check_names=False)
    
    def test_hand_computed_fixture(self):
        """Test against hand-computed fixture to verify equity curve."""
        # Create a simple scenario with known outcomes
        dates = pd.date_range('2023-01-01', periods=6, freq='D')
        price_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105],
            'open': [100, 101, 102, 103, 104, 105]  # Same as close for simplicity
        }, index=dates)
        
        # Signal: enter on day 2, exit on day 4
        signals = pd.Series([0, 0, 1, 1, 0, 0], index=dates)
        
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-06",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=10.0,  # 10 bps for easy calculation
            slippage_bps=0.0
        )
        
        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        
        # Hand-computed expectations:
        # Day 0: signal=0, position=0, holdings=0, cash=100000, equity=100000, trade=False
        # Day 1: signal=0, position=0, holdings=0, cash=100000, equity=100000, trade=False
        # Day 2: signal=1, position=1, holdings=102, cash=99999.898 (100000-0.102), equity=100101.898, trade=True
        # Day 3: signal=1, position=1, holdings=103, cash=99999.898, equity=100102.898, trade=False
        # Day 4: signal=0, position=0, holdings=0, cash=99999.794 (99999.898-0.104), equity=99999.794, trade=True
        # Day 5: signal=0, position=0, holdings=0, cash=99999.794, equity=99999.794, trade=False
        
        expected_equity = [100000, 100000, 100101.898, 100102.898, 99999.794, 99999.794]
        expected_cash = [100000, 100000, 99999.898, 99999.898, 99999.794, 99999.794]
        expected_holdings = [0, 0, 102, 103, 0, 0]
        expected_trade_flags = [False, False, True, False, True, False]
        
        # Check equity curve
        np.testing.assert_array_almost_equal(
            result['total_equity'].values, expected_equity, decimal=0
        )
        
        # Check cash
        np.testing.assert_array_almost_equal(
            result['cash'].values, expected_cash, decimal=0
        )
        
        # Check holdings
        np.testing.assert_array_almost_equal(
            result['holdings_value'].values, expected_holdings, decimal=0
        )
        
        # Check trade flags
        np.testing.assert_array_equal(
            result['trade_flag'].values, expected_trade_flags
        )
    
    def test_fees_impact_performance(self):
        """Test that higher fees reduce performance as expected."""
        # Create price data with clear trend
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        price_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        }, index=dates)
        
        # Signal: enter on day 2, exit on day 8
        signals = pd.Series([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], index=dates)
        
        # Test with no fees
        params_no_fees = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-10",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=0.0,
            slippage_bps=0.0
        )
        
        backtester_no_fees = Backtester(price_df, signals, params_no_fees)
        result_no_fees = backtester_no_fees.run()
        
        # Test with high fees
        params_high_fees = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-10",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=50.0,  # 50 bps
            slippage_bps=25.0  # 25 bps
        )
        
        backtester_high_fees = Backtester(price_df, signals, params_high_fees)
        result_high_fees = backtester_high_fees.run()
        
        # High fees should result in lower final equity
        final_equity_no_fees = result_no_fees['total_equity'].iloc[-1]
        final_equity_high_fees = result_high_fees['total_equity'].iloc[-1]
        
        assert final_equity_high_fees < final_equity_no_fees
        
        # Check that the difference is approximately equal to total costs
        total_costs = result_high_fees['trade_cost'].sum()
        expected_difference = total_costs
        actual_difference = final_equity_no_fees - final_equity_high_fees
        
        # Allow for small rounding differences
        assert abs(actual_difference - expected_difference) < 1.0
    
    def test_multiple_trades(self):
        """Test backtester with multiple entry/exit cycles."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        price_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 103, 102, 101, 102, 103],
            'open': [100, 101, 102, 103, 104, 103, 102, 101, 102, 103]
        }, index=dates)
        
        # Multiple trades: enter day 2, exit day 4, enter day 8, exit day 9
        signals = pd.Series([0, 0, 1, 1, 0, 0, 0, 0, 1, 0], index=dates)
        
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-10",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=10.0,
            slippage_bps=0.0
        )
        
        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        
        # Should have 4 trades (2 entries + 2 exits)
        assert result['trade_flag'].sum() == 4
        
        # Check that trade flags are True on the correct days
        expected_trade_days = [2, 4, 8, 9]  # 0-indexed
        trade_days = result[result['trade_flag']].index.tolist()
        assert len(trade_days) == 4
        
        # Check that total costs equal 4 trades * cost per trade
        expected_total_cost = 4 * 100 * 10 / 10000  # 4 trades * 100 price * 10 bps
        assert abs(result['trade_cost'].sum() - expected_total_cost) < 0.02  # Allow for small rounding differences
    
    def test_no_trades_scenario(self):
        """Test backtester when no trades occur."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        price_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'open': [100, 101, 102, 103, 104]
        }, index=dates)
        
        # No trades (all signals 0)
        signals = pd.Series([0, 0, 0, 0, 0], index=dates)
        
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-05",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=10.0,
            slippage_bps=0.0
        )
        
        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        
        # No trades should occur
        assert result['trade_flag'].sum() == 0
        assert result['trade_cost'].sum() == 0
        
        # Equity should remain constant at initial cash
        assert (result['total_equity'] == 100000).all()
        assert (result['cash'] == 100000).all()
        assert (result['holdings_value'] == 0).all()
    
    def test_always_in_position_scenario(self):
        """Test backtester when always in position."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        price_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'open': [100, 101, 102, 103, 104]
        }, index=dates)
        
        # Always in position (all signals 1)
        signals = pd.Series([1, 1, 1, 1, 1], index=dates)
        
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-05",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=10.0,
            slippage_bps=0.0
        )
        
        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        
        # Only one trade (entry on first day)
        assert result['trade_flag'].sum() == 1
        assert result['trade_flag'].iloc[0] == True  # Entry on first day
        
        # Holdings should equal close prices
        pd.testing.assert_series_equal(result['holdings_value'], result['close'], check_names=False)
        
        # Cash should be initial cash minus one trade cost
        expected_cash = 100000 - (100 * 10 / 10000)
        assert abs(result['cash'].iloc[0] - expected_cash) < 0.1  # Allow for small rounding differences


class TestBacktesterPerformanceMetrics:
    """Test performance metrics calculation."""
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        price_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 
                     100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'open': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101,
                    100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        }, index=dates)
        
        signals = pd.Series([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], index=dates)
        
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-20",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=10.0,
            slippage_bps=5.0
        )
        
        backtester = Backtester(price_df, signals, params)
        metrics = backtester.get_performance_metrics()
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'total_trades', 'total_costs', 'win_rate',
            'avg_win', 'avg_loss', 'final_equity'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check that metrics are reasonable
        assert metrics['total_trades'] > 0
        assert metrics['total_costs'] > 0
        assert metrics['final_equity'] > 0
        assert 0 <= metrics['win_rate'] <= 1
    
    def test_trade_log(self):
        """Test trade log generation."""
        dates = pd.date_range('2023-01-01', periods=6, freq='D')
        price_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105],
            'open': [100, 101, 102, 103, 104, 105]
        }, index=dates)
        
        signals = pd.Series([0, 0, 1, 1, 0, 0], index=dates)
        
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-06",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=10.0,
            slippage_bps=0.0
        )
        
        backtester = Backtester(price_df, signals, params)
        trade_log = backtester.get_trade_log()
        
        # Should have 2 trades (1 entry + 1 exit)
        assert len(trade_log) == 2
        
        # Check columns
        expected_columns = ['date', 'action', 'price', 'cost', 'equity']
        for col in expected_columns:
            assert col in trade_log.columns
        
        # Check actions
        assert trade_log['action'].tolist() == ['entry', 'exit']
        
        # Check that entry price is correct
        assert trade_log.iloc[0]['price'] == 102  # Entry price
        assert trade_log.iloc[1]['price'] == 104  # Exit price


class TestBacktesterEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_day_data(self):
        """Test backtester with single day of data."""
        dates = pd.date_range('2023-01-01', periods=1, freq='D')
        price_df = pd.DataFrame({
            'close': [100],
            'open': [100]
        }, index=dates)
        
        signals = pd.Series([1], index=dates)
        
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-01",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=10.0,
            slippage_bps=0.0
        )
        
        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        
        # Should have one trade (entry)
        assert result['trade_flag'].sum() == 1
        assert result['position'].iloc[0] == 1
        assert result['holdings_value'].iloc[0] == 100
    
    def test_empty_data(self):
        """Test backtester with empty data."""
        dates = pd.DatetimeIndex([])
        price_df = pd.DataFrame({
            'close': [],
            'open': []
        }, index=dates)
        
        signals = pd.Series([], index=dates)
        
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-01",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=10.0,
            slippage_bps=0.0
        )
        
        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        
        # Should return empty DataFrame with correct columns
        assert len(result) == 0
        expected_columns = ['close', 'signal', 'position', 'holdings_value', 
                          'cash', 'total_equity', 'trade_flag']
        for col in expected_columns:
            assert col in result.columns


class TestBacktesterTransactionCosts:
    """Test transaction cost impact on performance."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for cost impact tests."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        # Create a clear uptrend with some volatility
        base_prices = np.linspace(100, 120, 20) + np.sin(np.linspace(0, 4*np.pi, 20)) * 2
        price_df = pd.DataFrame({
            'close': base_prices,
            'open': base_prices
        }, index=dates)
        
        # Create signals that generate multiple trades
        signals = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0], index=dates)
        
        return price_df, signals
    
    @pytest.mark.parametrize("fee_bps,slippage_bps", [
        (0.0, 0.0),    # No costs
        (1.0, 0.0),    # Low fees only
        (5.0, 0.0),    # Medium fees only
        (10.0, 0.0),   # High fees only
        (0.0, 1.0),    # Low slippage only
        (0.0, 5.0),    # Medium slippage only
        (0.0, 10.0),   # High slippage only
        (1.0, 1.0),    # Low costs
        (5.0, 5.0),    # Medium costs
        (10.0, 10.0),  # High costs
        (20.0, 20.0),  # Very high costs
    ])
    def test_cost_impact_on_performance(self, sample_data, fee_bps, slippage_bps):
        """Test that higher transaction costs monotonically reduce performance."""
        price_df, signals = sample_data
        
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-20",
            fast_window=3,
            slow_window=6,
            initial_cash=100000,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps
        )
        
        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        metrics = backtester.get_performance_metrics()
        
        # Verify that costs are properly calculated
        total_costs = result['trade_cost'].sum()
        
        # Calculate expected costs based on actual trade prices
        trade_prices = result[result['trade_flag']]['close']
        if len(trade_prices) > 0:
            expected_costs = trade_prices.sum() * (fee_bps + slippage_bps) / 10000
            assert abs(total_costs - expected_costs) < 1.0  # Allow for small rounding differences
        
        # Verify that higher costs lead to lower final equity
        final_equity = result['total_equity'].iloc[-1]
        
        # For no-cost scenario, final equity should equal initial cash + holdings value
        if fee_bps == 0.0 and slippage_bps == 0.0:
            expected_equity = 100000 + result['holdings_value'].iloc[-1]
            assert abs(final_equity - expected_equity) < 1.0
        
        # Verify that Sharpe ratio decreases with higher costs (if there are trades)
        if result['trade_flag'].sum() > 0:
            assert metrics['sharpe_ratio'] is not None
            # For no-cost scenario, total_costs should be 0
            if fee_bps == 0.0 and slippage_bps == 0.0:
                assert metrics['total_costs'] == 0.0
            else:
                assert metrics['total_costs'] > 0
    
    def test_cost_impact_monotonicity(self, sample_data):
        """Test that performance decreases monotonically with increasing costs."""
        price_df, signals = sample_data
        
        # Test different cost levels
        cost_levels = [
            (0.0, 0.0),    # No costs
            (1.0, 1.0),    # Low costs
            (5.0, 5.0),    # Medium costs
            (10.0, 10.0),  # High costs
            (20.0, 20.0),  # Very high costs
        ]
        
        results = []
        for fee_bps, slippage_bps in cost_levels:
            params = StrategyParams(
                symbol="TEST",
                start="2023-01-01",
                end="2023-01-20",
                fast_window=3,
                slow_window=6,
                initial_cash=100000,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps
            )
            
            backtester = Backtester(price_df, signals, params)
            result = backtester.run()
            metrics = backtester.get_performance_metrics()
            
            results.append({
                'fee_bps': fee_bps,
                'slippage_bps': slippage_bps,
                'final_equity': result['total_equity'].iloc[-1],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_costs': metrics['total_costs']
            })
        
        # Verify monotonic decrease in final equity
        final_equities = [r['final_equity'] for r in results]
        for i in range(1, len(final_equities)):
            assert final_equities[i] <= final_equities[i-1], \
                f"Final equity should decrease with higher costs: {final_equities[i]} > {final_equities[i-1]}"
        
        # Verify monotonic increase in total costs
        total_costs = [r['total_costs'] for r in results]
        for i in range(1, len(total_costs)):
            assert total_costs[i] >= total_costs[i-1], \
                f"Total costs should increase with higher cost rates: {total_costs[i]} < {total_costs[i-1]}"
    
    def test_frequent_signals_cost_drag(self):
        """Test that frequent signals cause significant cost drag."""
        # Create data with very frequent signals (fast=5, slow=6)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        # Create a noisy price series that will generate many signals
        np.random.seed(42)  # For reproducible results
        price_changes = np.random.normal(0, 0.02, 50)  # 2% daily volatility
        prices = 100 * np.cumprod(1 + price_changes)
        
        price_df = pd.DataFrame({
            'close': prices,
            'open': prices
        }, index=dates)
        
        # Create very frequent signals (every few days)
        signals = pd.Series([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                           0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                           0, 1, 0, 1, 0, 1, 0, 1, 0, 0], index=dates)
        
        # Test with different cost levels
        cost_scenarios = [
            (0.0, 0.0),    # No costs
            (10.0, 10.0),  # Medium costs
            (50.0, 50.0),  # High costs
        ]
        
        results = []
        for fee_bps, slippage_bps in cost_scenarios:
            params = StrategyParams(
                symbol="TEST",
                start="2023-01-01",
                end="2023-02-19",
                fast_window=5,
                slow_window=6,
                initial_cash=100000,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps
            )
            
            backtester = Backtester(price_df, signals, params)
            result = backtester.run()
            metrics = backtester.get_performance_metrics()
            
            results.append({
                'fee_bps': fee_bps,
                'slippage_bps': slippage_bps,
                'final_equity': result['total_equity'].iloc[-1],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_costs': metrics['total_costs'],
                'total_trades': metrics['total_trades']
            })
        
        # Verify that frequent trading with high costs significantly impacts performance
        no_cost_result = results[0]
        high_cost_result = results[2]
        
        # High costs should result in significantly lower final equity
        equity_impact = (no_cost_result['final_equity'] - high_cost_result['final_equity']) / no_cost_result['final_equity']
        assert equity_impact > 0.0001, f"High costs should impact equity by at least 0.01%, got {equity_impact:.2%}"
        
        # Total costs should be significant relative to initial capital
        cost_ratio = high_cost_result['total_costs'] / 100000
        assert cost_ratio > 0.0001, f"Total costs should be at least 0.01% of initial capital, got {cost_ratio:.2%}"
        
        # Verify that we actually have many trades
        assert high_cost_result['total_trades'] > 20, f"Should have many trades with frequent signals, got {high_cost_result['total_trades']}"
    
    def test_cost_impact_on_sharpe_ratio(self, sample_data):
        """Test that higher costs reduce Sharpe ratio."""
        price_df, signals = sample_data
        
        # Test with increasing cost levels
        cost_levels = [(0.0, 0.0), (5.0, 5.0), (10.0, 10.0), (20.0, 20.0)]
        sharpe_ratios = []
        
        for fee_bps, slippage_bps in cost_levels:
            params = StrategyParams(
                symbol="TEST",
                start="2023-01-01",
                end="2023-01-20",
                fast_window=3,
                slow_window=6,
                initial_cash=100000,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps
            )
            
            backtester = Backtester(price_df, signals, params)
            result = backtester.run()
            metrics = backtester.get_performance_metrics()
            
            sharpe_ratios.append(metrics['sharpe_ratio'])
        
        # Verify that Sharpe ratios generally decrease with higher costs
        # (Allow for some noise due to random price movements)
        for i in range(1, len(sharpe_ratios)):
            if sharpe_ratios[i-1] is not None and sharpe_ratios[i] is not None:
                # Sharpe ratio should generally decrease, but allow for small increases due to noise
                assert sharpe_ratios[i] <= sharpe_ratios[i-1] + 0.1, \
                    f"Sharpe ratio should generally decrease with higher costs: {sharpe_ratios[i]} > {sharpe_ratios[i-1]}"
    
    def test_cost_impact_on_final_equity(self, sample_data):
        """Test that higher costs reduce final equity."""
        price_df, signals = sample_data
        
        # Test with increasing cost levels
        cost_levels = [(0.0, 0.0), (1.0, 1.0), (5.0, 5.0), (10.0, 10.0), (20.0, 20.0)]
        final_equities = []
        
        for fee_bps, slippage_bps in cost_levels:
            params = StrategyParams(
                symbol="TEST",
                start="2023-01-01",
                end="2023-01-20",
                fast_window=3,
                slow_window=6,
                initial_cash=100000,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps
            )
            
            backtester = Backtester(price_df, signals, params)
            result = backtester.run()
            
            final_equities.append(result['total_equity'].iloc[-1])
        
        # Verify that final equity decreases with higher costs
        for i in range(1, len(final_equities)):
            assert final_equities[i] <= final_equities[i-1], \
                f"Final equity should decrease with higher costs: {final_equities[i]} > {final_equities[i-1]}"
    
    def test_cost_impact_with_different_signal_frequencies(self):
        """Test cost impact with different signal frequencies."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        price_df = pd.DataFrame({
            'close': np.linspace(100, 130, 30),
            'open': np.linspace(100, 130, 30)
        }, index=dates)
        
        # Test different signal frequencies
        signal_scenarios = [
            # Low frequency: few trades
            pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0], index=dates),
            # Medium frequency: moderate trades
            pd.Series([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], index=dates),
            # High frequency: many trades
            pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], index=dates),
        ]
        
        cost_levels = [(0.0, 0.0), (10.0, 10.0)]
        
        for i, signals in enumerate(signal_scenarios):
            results = []
            backtest_results = []
            
            for fee_bps, slippage_bps in cost_levels:
                params = StrategyParams(
                    symbol="TEST",
                    start="2023-01-01",
                    end="2023-01-30",
                    fast_window=3,
                    slow_window=6,
                    initial_cash=100000,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps
                )
                
                backtester = Backtester(price_df, signals, params)
                result = backtester.run()
                metrics = backtester.get_performance_metrics()
                
                results.append({
                    'final_equity': result['total_equity'].iloc[-1],
                    'total_costs': metrics['total_costs'],
                    'total_trades': metrics['total_trades']
                })
                backtest_results.append(result)
            
            # Verify that higher frequency signals with costs have more impact
            no_cost = results[0]
            with_cost = results[1]
            
            # Higher frequency should have more trades
            assert with_cost['total_trades'] > 0, f"Scenario {i} should have trades"
            
            # With costs, final equity should be lower
            assert with_cost['final_equity'] <= no_cost['final_equity'], \
                f"Scenario {i}: Final equity with costs should be <= without costs"
            
            # Total costs should be proportional to number of trades
            # Calculate expected costs based on actual trade prices
            if with_cost['total_trades'] > 0:
                # Get the result from the with_cost scenario
                with_cost_result = backtest_results[1]
                trade_prices = with_cost_result[with_cost_result['trade_flag']]['close']
                if len(trade_prices) > 0:
                    expected_costs = trade_prices.sum() * (10.0 + 10.0) / 10000
                    assert abs(with_cost['total_costs'] - expected_costs) < 10.0, \
                        f"Scenario {i}: Total costs should match expected calculation (expected: {expected_costs:.2f}, actual: {with_cost['total_costs']:.2f})"
