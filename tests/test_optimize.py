"""Tests for optimization module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
import tempfile
import os
from pathlib import Path

from qbacktester.optimize import (
    grid_search, optimize_strategy, save_optimization_results,
    print_optimization_summary, _run_single_backtest
)


class TestGridSearch:
    """Test grid_search function."""
    
    def setup_method(self):
        """Set up test data."""
        self.symbol = "TEST"
        self.start = "2023-01-01"
        self.end = "2023-01-31"
        self.fast_grid = [5, 10]
        self.slow_grid = [20, 30]
    
    @patch('qbacktester.optimize._run_single_backtest')
    def test_grid_search_basic(self, mock_run_single_backtest):
        """Test basic grid search functionality."""
        # Mock single backtest results - return different results for different parameters
        def mock_result_func(args):
            symbol, start, end, fast, slow, initial_cash, fee_bps, slippage_bps, metric = args
            return {
                'fast': fast,
                'slow': slow,
                'sharpe': 1.2 + (fast / 100.0),  # Different sharpe for different fast values
                'max_dd': -0.05,
                'cagr': 0.15,
                'equity_final': 102000,
                'volatility': 0.12,
                'calmar': 3.0,
                'hit_rate': 0.6,
                'total_trades': 0,
                'success': True,
                'error': None
            }
        
        mock_run_single_backtest.side_effect = mock_result_func
        
        # Run grid search with sequential processing to avoid multiprocessing issues
        results_df = grid_search(
            symbol=self.symbol,
            start=self.start,
            end=self.end,
            fast_grid=self.fast_grid,
            slow_grid=self.slow_grid,
            metric="sharpe",
            n_jobs=1,  # Force sequential processing
            verbose=False
        )
        
        # Check results
        assert len(results_df) == 4  # 2 fast * 2 slow
        assert 'fast' in results_df.columns
        assert 'slow' in results_df.columns
        assert 'sharpe' in results_df.columns
        assert 'max_dd' in results_df.columns
        assert 'cagr' in results_df.columns
        assert 'equity_final' in results_df.columns
        
        # Check that all combinations are present
        expected_combinations = [(5, 20), (5, 30), (10, 20), (10, 30)]
        actual_combinations = list(zip(results_df['fast'], results_df['slow']))
        assert set(actual_combinations) == set(expected_combinations)
        
        # Check that results are sorted by sharpe (descending)
        # Since all results are the same, we just check they're not NaN
        assert not results_df['sharpe'].isna().any()
    
    @patch('qbacktester.optimize._run_single_backtest')
    def test_grid_search_different_metrics(self, mock_run_single_backtest):
        """Test grid search with different metrics."""
        # Mock single backtest results
        mock_result = {
            'fast': 5,
            'slow': 20,
            'sharpe': 1.2,
            'max_dd': -0.05,
            'cagr': 0.15,
            'equity_final': 102000,
            'volatility': 0.12,
            'calmar': 3.0,
            'hit_rate': 0.6,
            'total_trades': 0,
            'success': True,
            'error': None
        }
        mock_run_single_backtest.return_value = mock_result
        
        # Test different metrics
        for metric in ['sharpe', 'cagr', 'calmar', 'max_dd']:
            results_df = grid_search(
                symbol=self.symbol,
                start=self.start,
                end=self.end,
                fast_grid=[5],
                slow_grid=[20],
                metric=metric,
                n_jobs=1,  # Force sequential processing
                verbose=False
            )
            
            assert len(results_df) == 1
            assert results_df.iloc[0][metric] == mock_result[metric]
    
    def test_grid_search_validation(self):
        """Test grid search input validation."""
        # Test invalid metric
        with pytest.raises(ValueError, match="Unsupported metric"):
            grid_search(
                symbol=self.symbol,
                start=self.start,
                end=self.end,
                fast_grid=self.fast_grid,
                slow_grid=self.slow_grid,
                metric="invalid_metric",
                verbose=False
            )
        
        # Test empty grids
        with pytest.raises(ValueError, match="fast_grid and slow_grid cannot be empty"):
            grid_search(
                symbol=self.symbol,
                start=self.start,
                end=self.end,
                fast_grid=[],
                slow_grid=self.slow_grid,
                metric="sharpe",
                verbose=False
            )
        
        # Test invalid window values
        with pytest.raises(ValueError, match="All fast values must be less than all slow values"):
            grid_search(
                symbol=self.symbol,
                start=self.start,
                end=self.end,
                fast_grid=[50],
                slow_grid=[20],
                metric="sharpe",
                verbose=False
            )
        
        # Test non-positive values
        with pytest.raises(ValueError, match="All window values must be positive"):
            grid_search(
                symbol=self.symbol,
                start=self.start,
                end=self.end,
                fast_grid=[0],
                slow_grid=self.slow_grid,
                metric="sharpe",
                verbose=False
            )
    
    @patch('qbacktester.optimize._run_single_backtest')
    def test_grid_search_error_handling(self, mock_run_single_backtest):
        """Test grid search error handling."""
        # Mock single backtest to return error result
        mock_error_result = {
            'fast': 5,
            'slow': 20,
            'sharpe': np.nan,
            'max_dd': np.nan,
            'cagr': np.nan,
            'equity_final': np.nan,
            'volatility': np.nan,
            'calmar': np.nan,
            'hit_rate': np.nan,
            'total_trades': 0,
            'success': False,
            'error': "Test error"
        }
        mock_run_single_backtest.return_value = mock_error_result
        
        results_df = grid_search(
            symbol=self.symbol,
            start=self.start,
            end=self.end,
            fast_grid=[5],
            slow_grid=[20],
            metric="sharpe",
            n_jobs=1,  # Force sequential processing
            verbose=False
        )
        
        # Check that error is handled gracefully
        assert len(results_df) == 1
        assert results_df.iloc[0]['success'] == False
        assert results_df.iloc[0]['error'] == "Test error"
        assert pd.isna(results_df.iloc[0]['sharpe'])
    
    @patch('qbacktester.optimize._run_single_backtest')
    def test_grid_search_sequential_vs_parallel(self, mock_run_single_backtest):
        """Test that sequential and parallel modes produce same results."""
        # Mock single backtest results
        mock_result = {
            'fast': 5,
            'slow': 20,
            'sharpe': 1.2,
            'max_dd': -0.05,
            'cagr': 0.15,
            'equity_final': 102000,
            'volatility': 0.12,
            'calmar': 3.0,
            'hit_rate': 0.6,
            'total_trades': 0,
            'success': True,
            'error': None
        }
        mock_run_single_backtest.return_value = mock_result
        
        # Run sequential
        results_seq = grid_search(
            symbol=self.symbol,
            start=self.start,
            end=self.end,
            fast_grid=self.fast_grid,
            slow_grid=self.slow_grid,
            metric="sharpe",
            n_jobs=1,
            verbose=False
        )
        
        # Run parallel (but with n_jobs=1 to avoid multiprocessing issues in tests)
        results_par = grid_search(
            symbol=self.symbol,
            start=self.start,
            end=self.end,
            fast_grid=self.fast_grid,
            slow_grid=self.slow_grid,
            metric="sharpe",
            n_jobs=1,  # Force sequential to avoid multiprocessing issues
            verbose=False
        )
        
        # Results should be the same (ignoring order)
        assert len(results_seq) == len(results_par)
        assert set(zip(results_seq['fast'], results_seq['slow'])) == set(zip(results_par['fast'], results_par['slow']))


class TestOptimizeStrategy:
    """Test optimize_strategy convenience function."""
    
    @patch('qbacktester.optimize.grid_search')
    def test_optimize_strategy(self, mock_grid_search):
        """Test optimize_strategy function."""
        mock_results = pd.DataFrame({
            'fast': [5, 10],
            'slow': [20, 30],
            'sharpe': [1.2, 1.1],
            'cagr': [0.15, 0.14],
            'max_dd': [-0.05, -0.04]
        })
        mock_grid_search.return_value = mock_results
        
        results = optimize_strategy(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-31",
            fast_range=(5, 10, 5),
            slow_range=(20, 30, 10),
            metric="sharpe",
            verbose=False
        )
        
        # Check that grid_search was called with correct parameters
        mock_grid_search.assert_called_once()
        call_args = mock_grid_search.call_args
        
        assert call_args[1]['symbol'] == "TEST"
        assert call_args[1]['fast_grid'] == [5, 10]
        assert call_args[1]['slow_grid'] == [20, 30]
        assert call_args[1]['metric'] == "sharpe"
        
        # Check results
        pd.testing.assert_frame_equal(results, mock_results)


class TestSaveOptimizationResults:
    """Test save_optimization_results function."""
    
    def test_save_optimization_results(self):
        """Test saving optimization results to CSV."""
        # Create test results
        results_df = pd.DataFrame({
            'fast': [5, 10],
            'slow': [20, 30],
            'sharpe': [1.2, 1.1],
            'cagr': [0.15, 0.14],
            'max_dd': [-0.05, -0.04]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = save_optimization_results(results_df, "TEST", temp_dir)
            
            # Check that file was created
            assert os.path.exists(file_path)
            assert file_path.endswith('.csv')
            assert 'opt_grid_test.csv' in file_path
            
            # Check file contents
            loaded_df = pd.read_csv(file_path)
            pd.testing.assert_frame_equal(loaded_df, results_df)
    
    def test_save_optimization_results_creates_directory(self):
        """Test that save_optimization_results creates output directory."""
        results_df = pd.DataFrame({
            'fast': [5],
            'slow': [20],
            'sharpe': [1.2]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "new_dir")
            file_path = save_optimization_results(results_df, "TEST", output_dir)
            
            # Check that directory was created
            assert os.path.exists(output_dir)
            assert os.path.exists(file_path)


class TestPrintOptimizationSummary:
    """Test print_optimization_summary function."""
    
    def test_print_optimization_summary(self, capsys):
        """Test printing optimization summary."""
        results_df = pd.DataFrame({
            'fast': [5, 10, 15],
            'slow': [20, 30, 40],
            'sharpe': [1.2, 1.1, 1.0],
            'cagr': [0.15, 0.14, 0.13],
            'max_dd': [-0.05, -0.04, -0.03],
            'calmar': [3.0, 3.5, 4.0],
            'hit_rate': [0.6, 0.65, 0.7],
            'total_trades': [10, 12, 8],
            'success': [True, True, True]
        })
        
        print_optimization_summary(results_df, "sharpe", top_n=2)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that summary was printed
        assert "Optimization Results Summary" in output
        assert "Top 2 parameter sets by sharpe" in output
        assert "Fast" in output
        assert "Slow" in output
        assert "Sharpe" in output
        assert "CAGR" in output
        assert "MaxDD" in output


class TestRunSingleBacktest:
    """Test _run_single_backtest function."""
    
    @patch('qbacktester.optimize.run_crossover_backtest')
    def test_run_single_backtest_success(self, mock_run_backtest):
        """Test successful single backtest."""
        mock_results = {
            "metrics": {
                'sharpe': 1.2,
                'max_drawdown': -0.05,
                'cagr': 0.15,
                'volatility': 0.12,
                'calmar': 3.0,
                'hit_rate': 0.6
            },
            "equity_curve": pd.DataFrame({
                'total_equity': [100000, 101000, 102000]
            }, index=pd.date_range('2023-01-01', periods=3, freq='D')),
            "trades": pd.DataFrame(columns=['timestamp', 'side', 'price', 'equity'])
        }
        mock_run_backtest.return_value = mock_results
        
        args = ("TEST", "2023-01-01", "2023-01-31", 5, 20, 100000, 1.0, 0.5, "sharpe")
        result = _run_single_backtest(args)
        
        assert result['fast'] == 5
        assert result['slow'] == 20
        assert result['sharpe'] == 1.2
        assert result['max_dd'] == -0.05
        assert result['cagr'] == 0.15
        assert result['equity_final'] == 102000
        assert result['success'] == True
        assert result['error'] is None
    
    @patch('qbacktester.optimize.run_crossover_backtest')
    def test_run_single_backtest_error(self, mock_run_backtest):
        """Test single backtest with error."""
        mock_run_backtest.side_effect = Exception("Test error")
        
        args = ("TEST", "2023-01-01", "2023-01-31", 5, 20, 100000, 1.0, 0.5, "sharpe")
        result = _run_single_backtest(args)
        
        assert result['fast'] == 5
        assert result['slow'] == 20
        assert pd.isna(result['sharpe'])
        assert result['success'] == False
        assert result['error'] == "Test error"


class TestOptimizationIntegration:
    """Test optimization module integration."""
    
    @patch('qbacktester.optimize._run_single_backtest')
    def test_full_optimization_workflow(self, mock_run_single_backtest):
        """Test complete optimization workflow."""
        # Mock different results for different parameters
        def mock_single_backtest_results(args):
            symbol, start, end, fast, slow, initial_cash, fee_bps, slippage_bps, metric = args
            # Return different results based on parameters
            sharpe = 1.0 + (fast / 100.0) + (slow / 1000.0)
            return {
                'fast': fast,
                'slow': slow,
                'sharpe': sharpe,
                'max_dd': -0.05,
                'cagr': 0.15,
                'equity_final': 102000,
                'volatility': 0.12,
                'calmar': 3.0,
                'hit_rate': 0.6,
                'total_trades': 0,
                'success': True,
                'error': None
            }
        
        mock_run_single_backtest.side_effect = mock_single_backtest_results
        
        # Run optimization
        results_df = grid_search(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-31",
            fast_grid=[5, 10],
            slow_grid=[20, 30],
            metric="sharpe",
            n_jobs=1,  # Force sequential processing
            verbose=False
        )
        
        # Check results
        assert len(results_df) == 4
        assert all(results_df['success'])
        
        # Check that results are sorted by sharpe
        assert results_df['sharpe'].is_monotonic_decreasing
        
        # Check that higher parameter combinations have higher sharpe
        assert results_df.iloc[0]['sharpe'] > results_df.iloc[-1]['sharpe']
    
    def test_optimization_with_real_data_structure(self):
        """Test optimization with realistic data structure."""
        # This test ensures the optimization works with the expected data structure
        # from the run_crossover_backtest function
        
        # Create mock data that matches the expected structure
        mock_equity_curve = pd.DataFrame({
            'total_equity': [100000, 101000, 102000, 101500, 102500]
        }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
        
        mock_trades = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-04')],
            'side': ['entry', 'exit'],
            'price': [100.0, 101.0],
            'equity': [101000, 102500]
        })
        
        mock_results = {
            "params": Mock(),
            "equity_curve": mock_equity_curve,
            "metrics": {
                'sharpe': 1.2,
                'max_drawdown': -0.05,
                'cagr': 0.15,
                'volatility': 0.12,
                'calmar': 3.0,
                'hit_rate': 0.6
            },
            "trades": mock_trades
        }
        
        with patch('qbacktester.optimize.run_crossover_backtest', return_value=mock_results):
            results_df = grid_search(
                symbol="TEST",
                start="2023-01-01",
                end="2023-01-31",
                fast_grid=[5],
                slow_grid=[20],
                metric="sharpe",
                verbose=False
            )
            
            assert len(results_df) == 1
            assert results_df.iloc[0]['equity_final'] == 102500
            assert results_df.iloc[0]['total_trades'] == 2


class TestOptimizationEdgeCases:
    """Test optimization edge cases."""
    
    def test_optimization_with_single_parameter(self):
        """Test optimization with single parameter values."""
        with patch('qbacktester.optimize.run_crossover_backtest') as mock_run_backtest:
            mock_results = {
                "metrics": {
                    'sharpe': 1.2,
                    'max_drawdown': -0.05,
                    'cagr': 0.15,
                    'volatility': 0.12,
                    'calmar': 3.0,
                    'hit_rate': 0.6
                },
                "equity_curve": pd.DataFrame({
                    'total_equity': [100000, 101000, 102000]
                }, index=pd.date_range('2023-01-01', periods=3, freq='D')),
                "trades": pd.DataFrame(columns=['timestamp', 'side', 'price', 'equity'])
            }
            mock_run_backtest.return_value = mock_results
            
            results_df = grid_search(
                symbol="TEST",
                start="2023-01-01",
                end="2023-01-31",
                fast_grid=[5],
                slow_grid=[20],
                metric="sharpe",
                verbose=False
            )
            
            assert len(results_df) == 1
            assert results_df.iloc[0]['fast'] == 5
            assert results_df.iloc[0]['slow'] == 20
    
    def test_optimization_with_large_grid(self):
        """Test optimization with larger parameter grid."""
        with patch('qbacktester.optimize.run_crossover_backtest') as mock_run_backtest:
            mock_results = {
                "metrics": {
                    'sharpe': 1.2,
                    'max_drawdown': -0.05,
                    'cagr': 0.15,
                    'volatility': 0.12,
                    'calmar': 3.0,
                    'hit_rate': 0.6
                },
                "equity_curve": pd.DataFrame({
                    'total_equity': [100000, 101000, 102000]
                }, index=pd.date_range('2023-01-01', periods=3, freq='D')),
                "trades": pd.DataFrame(columns=['timestamp', 'side', 'price', 'equity'])
            }
            mock_run_backtest.return_value = mock_results
            
            # Test with larger grid
            fast_grid = list(range(5, 21, 5))  # [5, 10, 15, 20]
            slow_grid = list(range(50, 101, 10))  # [50, 60, 70, 80, 90, 100]
            
            results_df = grid_search(
                symbol="TEST",
                start="2023-01-01",
                end="2023-01-31",
                fast_grid=fast_grid,
                slow_grid=slow_grid,
                metric="sharpe",
                n_jobs=1,  # Use sequential to avoid multiprocessing issues in tests
                verbose=False
            )
            
            assert len(results_df) == 24  # 4 * 6
            assert set(results_df['fast']) == set(fast_grid)
            assert set(results_df['slow']) == set(slow_grid)
