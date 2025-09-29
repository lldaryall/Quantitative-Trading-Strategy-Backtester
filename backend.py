#!/usr/bin/env python3
"""
Backend API for QBacktester Web Interface
Provides REST API endpoints for running backtests via web interface
"""

import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import traceback

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    import base64
    
    # Import QBacktester modules
    from qbacktester import (
        Backtester, Strategy, get_price_data, 
        run_crossover_backtest, print_backtest_report
    )
    from qbacktester.metrics import get_all_metrics
    from qbacktester.plotting import create_equity_plot, create_drawdown_plot, create_price_signals_plot
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required dependencies: pip install flask flask-cors matplotlib seaborn")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WebBacktester:
    """Web interface wrapper for QBacktester"""
    
    def __init__(self):
        self.results_cache = {}
    
    def run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a backtest with given parameters"""
        try:
            # Extract parameters
            symbol = params.get('symbol', 'AAPL').upper()
            start_date = params.get('start_date', '2020-01-01')
            end_date = params.get('end_date', '2023-12-31')
            fast_window = int(params.get('fast_window', 20))
            slow_window = int(params.get('slow_window', 50))
            initial_capital = float(params.get('initial_capital', 100000))
            commission = float(params.get('commission', 0.001))
            slippage = float(params.get('slippage', 0.0005))
            
            print(f"Running backtest for {symbol} from {start_date} to {end_date}")
            print(f"Parameters: Fast={fast_window}, Slow={slow_window}, Capital=${initial_capital:,.0f}")
            
            # Fetch data
            print("Fetching price data...")
            data = get_price_data(symbol, start_date, end_date)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            print(f"Data fetched: {len(data)} days")
            
            # Run backtest using the existing function
            results = run_crossover_backtest(
                data=data,
                fast_window=fast_window,
                slow_window=slow_window,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage
            )
            
            # Get additional metrics
            metrics = get_all_metrics(results['returns'])
            
            # Generate plots
            equity_plot = self._generate_equity_plot(data, results, symbol, fast_window, slow_window)
            drawdown_plot = self._generate_drawdown_plot(results['equity_curve'], symbol)
            price_signals_plot = self._generate_price_signals_plot(data, results, symbol, fast_window, slow_window)
            
            # Prepare response
            response = {
                'success': True,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'parameters': {
                    'fast_window': fast_window,
                    'slow_window': slow_window,
                    'initial_capital': initial_capital,
                    'commission': commission,
                    'slippage': slippage
                },
                'metrics': {
                    'total_return': results.get('total_return', 0) * 100,
                    'cagr': results.get('cagr', 0) * 100,
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'max_drawdown': results.get('max_drawdown', 0) * 100,
                    'win_rate': results.get('win_rate', 0) * 100,
                    'total_trades': results.get('total_trades', 0),
                    'final_capital': results.get('final_capital', initial_capital),
                    'volatility': results.get('volatility', 0) * 100,
                    'calmar_ratio': results.get('calmar_ratio', 0),
                    'sortino_ratio': results.get('sortino_ratio', 0),
                    'avg_win': results.get('avg_win', 0) * 100,
                    'avg_loss': results.get('avg_loss', 0) * 100,
                    'best_trade': results.get('best_trade', 0) * 100,
                    'worst_trade': results.get('worst_trade', 0) * 100
                },
                'equity_curve': {
                    'dates': results['equity_curve'].index.strftime('%Y-%m-%d').tolist(),
                    'values': results['equity_curve'].values.tolist()
                },
                'trade_history': self._format_trade_history(results.get('trades', [])),
                'plots': {
                    'equity': equity_plot,
                    'drawdown': drawdown_plot,
                    'price_signals': price_signals_plot
                }
            }
            
            print("Backtest completed successfully")
            return response
            
        except Exception as e:
            print(f"Error in backtest: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _generate_equity_plot(self, data, results, symbol, fast_window, slow_window):
        """Generate equity curve plot"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot equity curve
            ax.plot(results['equity_curve'].index, results['equity_curve'].values, 
                   linewidth=2, color='#6366f1', label='Portfolio Value')
            
            # Add buy/sell signals if available
            if 'signals' in results and not results['signals'].empty:
                signals = results['signals']
                buy_signals = signals[signals['positions'] == 1]
                sell_signals = signals[signals['positions'] == -1]
                
                if not buy_signals.empty:
                    ax.scatter(buy_signals.index, 
                             results['equity_curve'].loc[buy_signals.index], 
                             color='green', marker='^', s=50, label='Buy', alpha=0.7)
                
                if not sell_signals.empty:
                    ax.scatter(sell_signals.index, 
                             results['equity_curve'].loc[sell_signals.index], 
                             color='red', marker='v', s=50, label='Sell', alpha=0.7)
            
            ax.set_title(f'{symbol} Moving Average Strategy ({fast_window}/{slow_window})', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            print(f"Error generating equity plot: {e}")
            return None
    
    def _generate_drawdown_plot(self, equity_curve, symbol):
        """Generate drawdown plot"""
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Calculate drawdown
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak * 100
            
            # Plot drawdown
            ax.fill_between(drawdown.index, drawdown.values, 0, 
                          color='red', alpha=0.3, label='Drawdown')
            ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            
            ax.set_title(f'{symbol} Drawdown Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            print(f"Error generating drawdown plot: {e}")
            return None
    
    def _generate_price_signals_plot(self, data, results, symbol, fast_window, slow_window):
        """Generate price and signals plot"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot price
            ax.plot(data.index, data['close'], linewidth=1, color='black', alpha=0.7, label='Price')
            
            # Plot moving averages if available
            if 'signals' in results and not results['signals'].empty:
                signals = results['signals']
                if 'short_ma' in signals.columns:
                    ax.plot(signals.index, signals['short_ma'], 
                           linewidth=2, color='blue', label=f'Fast MA ({fast_window})')
                if 'long_ma' in signals.columns:
                    ax.plot(signals.index, signals['long_ma'], 
                           linewidth=2, color='orange', label=f'Slow MA ({slow_window})')
                
                # Add buy/sell signals
                buy_signals = signals[signals['positions'] == 1]
                sell_signals = signals[signals['positions'] == -1]
                
                if not buy_signals.empty:
                    ax.scatter(buy_signals.index, data.loc[buy_signals.index, 'close'], 
                             color='green', marker='^', s=100, label='Buy', alpha=0.8)
                
                if not sell_signals.empty:
                    ax.scatter(sell_signals.index, data.loc[sell_signals.index, 'close'], 
                             color='red', marker='v', s=100, label='Sell', alpha=0.8)
            
            ax.set_title(f'{symbol} Price Action and Signals', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            print(f"Error generating price signals plot: {e}")
            return None
    
    def _format_trade_history(self, trades):
        """Format trade history for web display"""
        if not trades:
            return []
        
        formatted_trades = []
        for trade in trades:
            formatted_trades.append({
                'date': trade.get('date', '').strftime('%Y-%m-%d') if hasattr(trade.get('date', ''), 'strftime') else str(trade.get('date', '')),
                'action': trade.get('action', ''),
                'price': round(trade.get('price', 0), 2),
                'quantity': trade.get('quantity', 0),
                'pnl': round(trade.get('pnl', 0) * 100, 2),  # Convert to percentage
                'is_win': trade.get('pnl', 0) > 0
            })
        
        return formatted_trades

# Initialize the backtester
backtester = WebBacktester()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run a backtest"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required parameters
        required_params = ['symbol', 'start_date', 'end_date', 'fast_window', 'slow_window']
        for param in required_params:
            if param not in data:
                return jsonify({'success': False, 'error': f'Missing required parameter: {param}'}), 400
        
        # Run the backtest
        result = backtester.run_backtest(data)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/symbols', methods=['GET'])
def get_popular_symbols():
    """Get list of popular trading symbols"""
    popular_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'BND', 'GLD',
        'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'MA', 'V', 'DIS'
    ]
    
    return jsonify({
        'symbols': popular_symbols,
        'total': len(popular_symbols)
    })

@app.route('/api/optimize', methods=['POST'])
def optimize_parameters():
    """Optimize strategy parameters"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # This would integrate with your optimization module
        # For now, return a placeholder response
        return jsonify({
            'success': True,
            'message': 'Parameter optimization not yet implemented in web interface',
            'suggestion': 'Use the command line interface for optimization: qbt optimize --symbol SPY --start 2020-01-01 --end 2023-12-31 --fast 10,20,30 --slow 50,100,150'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting QBacktester Web API...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/backtest - Run backtest")
    print("  GET  /api/symbols - Get popular symbols")
    print("  POST /api/optimize - Optimize parameters")
    print("\nStarting server on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
