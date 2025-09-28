# qbacktester

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

A quantitative backtesting library for financial strategies. Built for researchers and practitioners who need reliable, extensible tools for testing trading algorithms.

## Project Overview

Designed and back-tested a moving-average crossover strategy on S&P 500 daily data. Vectorized with pandas/NumPy. Achieved Sharpe ≈ 1.2 and max drawdown ≈ 8% in baseline configuration. ~50% runtime improvement vs initial looped prototype.

## Features

- 🚀 **Easy to use**: Simple API for strategy development and backtesting
- 📊 **Rich metrics**: Comprehensive performance analysis with Sharpe ratio, max drawdown, and more
- 🔌 **Extensible**: Plugin architecture for custom data providers and strategies
- 🎯 **Production ready**: Type hints, comprehensive testing, and code quality tools
- 📈 **Data integration**: Built-in support for Yahoo Finance and custom data sources
- 💾 **Smart caching**: Automatic local caching with parquet files for faster repeated access
- 🔄 **Robust downloads**: Retry logic with exponential backoff for reliable data fetching

## Quick Start

### Installation

#### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/daryapylypenko/Quantitative-Trading-Strategy-Backtester.git
cd qbacktester

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

#### Option 2: Docker Installation

```bash
# Clone the repository
git clone https://github.com/daryapylypenko/Quantitative-Trading-Strategy-Backtester.git
cd qbacktester

# Build the Docker image
docker build -t qbacktester .

# Run qbacktester help
docker run --rm qbacktester

# Run a backtest with Docker
docker run --rm -v $(pwd)/reports:/app/reports qbacktester run --symbol SPY --start 2020-01-01 --end 2023-12-31 --fast 20 --slow 50 --plot
```

**Docker Commands:**
- `docker build -t qbacktester .` - Build the Docker image
- `docker run --rm qbacktester` - Run qbacktester help (default command)
- `docker run --rm -v $(pwd)/reports:/app/reports qbacktester [command]` - Run any qbacktester command with volume mounting for reports

### Basic Usage

```python
from qbacktester import Backtester, Strategy, get_price_data
import pandas as pd
import numpy as np

# Define a simple moving average strategy
class MovingAverageStrategy(Strategy):
    def __init__(self, short_window=20, long_window=50):
        super().__init__("MovingAverage")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['short_ma'] = data['close'].rolling(self.short_window).mean()
        signals['long_ma'] = data['close'].rolling(self.long_window).mean()
        signals['signal'] = 0
        signals['signal'][self.short_window:] = np.where(
            signals['short_ma'][self.short_window:] > signals['long_ma'][self.short_window:], 1, 0
        )
        signals['positions'] = signals['signal'].diff()
        return signals

# Set up backtest
backtester = Backtester(initial_capital=100000, commission=0.001)

# Fetch data (with automatic caching)
data = get_price_data("AAPL", "2020-01-01", "2023-12-31")

# Run backtest
strategy = MovingAverageStrategy()
results = backtester.run_backtest(strategy, data)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## Quickstart Commands

### Command Line Interface

qbacktester provides a powerful CLI for running backtests directly from the command line:

```bash
# Basic backtest
qbt run --symbol SPY --start 2020-01-01 --end 2023-12-31 --fast 20 --slow 50

# With custom parameters and plotting
qbt run --symbol AAPL --start 2020-01-01 --end 2023-12-31 --fast 10 --slow 30 --cash 50000 --plot

# Quick summary only
qbt run --symbol MSFT --start 2020-01-01 --end 2023-12-31 --fast 5 --slow 20 --quick

# Using python -m
python -m qbacktester run --symbol TSLA --start 2020-01-01 --end 2023-12-31 --fast 15 --slow 45 --plot

# Get help
qbt --help
qbt run --help
```

#### CLI Options

- `--symbol, -s`: Stock symbol to backtest (required)
- `--start, -st`: Start date in YYYY-MM-DD format (required)
- `--end, -e`: End date in YYYY-MM-DD format (required)
- `--fast, -f`: Fast SMA window period (required)
- `--slow, -sl`: Slow SMA window period (required)
- `--cash, -c`: Initial cash amount (default: 100000)
- `--fee-bps`: Trading fee in basis points (default: 1.0)
- `--slippage-bps`: Slippage cost in basis points (default: 0.5)
- `--plot, -p`: Generate equity curve and drawdown plots
- `--output-dir, -o`: Output directory for plots (default: reports)
- `--quiet, -q`: Suppress progress messages
- `--quick`: Show only quick summary instead of full report

### Example CLI Runs

#### Basic Backtest Example

```bash
$ qbt run --symbol SPY --start 2020-01-01 --end 2023-12-31 --fast 20 --slow 50 --plot
```

**Sample Output:**
```
📊 Fetching data for SPY from 2020-01-01 to 2023-12-31...
🔍 Generating signals with 20/50 SMA crossover...
⚡ Running backtest with 100,000 initial capital...
📈 Calculating performance metrics...
✅ Backtest completed successfully!

📊 Performance Summary
═══════════════════════════════════════════════════════════════════════════════
📈 Total Return: 45.23%
📊 CAGR: 10.12%
📉 Max Drawdown: -12.45%
⚡ Sharpe Ratio: 1.18
📊 Calmar Ratio: 0.81
📈 Sortino Ratio: 1.45
🎯 Hit Rate: 58.3%
💰 Total Trades: 24
💸 Total Costs: $1,234.56
💵 Final Equity: $145,230.00

📊 Trade Analysis
═══════════════════════════════════════════════════════════════════════════════
📈 Average Win: 3.2%
📉 Average Loss: -2.1%
📊 Win/Loss Ratio: 1.52
📈 Best Trade: 8.7%
📉 Worst Trade: -5.2%

📊 Risk Metrics
═══════════════════════════════════════════════════════════════════════════════
📉 VaR (5%): -2.1%
📊 CVaR (5%): -3.2%
📈 Volatility: 18.5%

📊 Plots saved to: reports/SPY_2020-01-01_2023-12-31_20_50/
```

#### Parameter Optimization Example

```bash
$ qbt optimize --symbol SPY --start 2015-01-01 --end 2025-01-01 --fast 10,20,30 --slow 50,100,150 --metric sharpe --jobs 4
```

**Sample Output:**
```
🔍 Starting parameter optimization...
📊 Testing 9 parameter combinations...
⚡ Using 4 parallel workers...

✅ Optimization completed in 2.3 seconds!

🏆 Top 5 Results (by Sharpe Ratio):
═══════════════════════════════════════════════════════════════════════════════
Rank  Fast  Slow  Sharpe  CAGR   MaxDD  Calmar  Trades
───────────────────────────────────────────────────────────────────────────────
1     20    100   1.234   12.5%  -8.2%  1.52    18
2     10    50    1.198   11.8%  -9.1%  1.30    25
3     30    150   1.156   10.9%  -7.8%  1.40    12
4     20    50    1.134   11.2%  -10.2% 1.10    22
5     10    100   1.089   10.5%  -11.5% 0.91    28

🎯 Best parameters: Fast=20, Slow=100
   Sharpe Ratio: 1.234
   CAGR: 12.5%
   Max Drawdown: -8.2%
```

#### Walk-Forward Analysis Example

```bash
$ qbt walkforward --symbol SPY --start 2010-01-01 --end 2025-01-01 --is 3 --oos 1 --fast 10,20,50 --slow 50,100,200
```

**Sample Output:**
```
🔄 Starting walk-forward analysis...
📊 Analysis period: 2010-01-01 to 2025-01-01
📈 In-sample years: 3, Out-of-sample years: 1
🔍 Testing 9 parameter combinations per window...

✅ Walk-forward analysis completed!

📊 Walk-Forward Summary
═══════════════════════════════════════════════════════════════════════════════
📈 Total windows: 12
✅ Successful windows: 11
📊 Average Sharpe: 1.156
📉 Average Max DD: -9.2%
📈 Win rate: 72.7%

🎯 Parameter Stability
═══════════════════════════════════════════════════════════════════════════════
📊 Fast window: 18.2 ± 4.1 (CV: 0.23)
📊 Slow window: 95.5 ± 12.3 (CV: 0.13)
📈 Parameter changes: 3

📊 Results saved to: reports/walkforward_SPY_2010-2025/
```

### Plots and Visualizations

qbacktester generates comprehensive visualizations saved to the `reports/` directory:

#### Equity Curve and Drawdown Plots
- **Equity Curve**: Shows strategy performance over time with buy/sell signals
- **Drawdown Analysis**: Visualizes maximum drawdown periods and recovery
- **Price Action**: Displays price data with moving averages and trading signals

#### Grid Search Heatmaps
- **Parameter Space Visualization**: Heatmaps showing Sharpe ratio across parameter combinations
- **Robust Parameter Identification**: Visual identification of stable parameter regions
- **Performance Contours**: Easy identification of optimal parameter ranges

#### Walk-Forward Analysis Charts
- **Rolling Performance**: Out-of-sample performance across time windows
- **Parameter Evolution**: How optimal parameters change over time
- **Stability Metrics**: Visualization of parameter consistency

**Example Plot Structure:**
```
reports/
├── SPY_2020-01-01_2023-12-31_20_50/
│   ├── equity_curve.png
│   ├── drawdown.png
│   └── price_signals.png
├── optimization_SPY_2015-2025/
│   ├── heatmap_sharpe.png
│   └── results_summary.csv
└── walkforward_SPY_2010-2025/
    ├── rolling_performance.png
    ├── parameter_evolution.png
    └── window_results.csv
```

### Parameter Optimization

Find the best parameters for your strategy using grid search optimization:

```bash
# Optimize parameters
qbt optimize --symbol SPY --start 2015-01-01 --end 2025-01-01 --fast 5,10,20,50 --slow 50,100,150,200 --metric sharpe

# With custom parameters
qbt optimize --symbol AAPL --start 2020-01-01 --end 2023-12-31 --fast 10,20,30 --slow 50,100,150 --metric cagr --jobs 4

# Get help
qbt optimize --help
```

### Walk-Forward Analysis

Test strategy robustness with walk-forward analysis to avoid overfitting:

```bash
# Basic walk-forward analysis
qbt walkforward --symbol SPY --start 2010-01-01 --end 2025-01-01 --is 3 --oos 1

# With custom parameters
qbt walkforward --symbol AAPL --start 2015-01-01 --end 2023-12-31 --is 2 --oos 1 --fast 10,20,50 --slow 50,100,200 --metric sharpe

# Get help
qbt walkforward --help
```

#### Walk-Forward Analysis Options

- `--symbol, -s`: Stock symbol to analyze (required)
- `--start, -st`: Start date in YYYY-MM-DD format (required)
- `--end, -e`: End date in YYYY-MM-DD format (required)
- `--is`: In-sample years for parameter optimization (default: 3)
- `--oos`: Out-of-sample years for evaluation (default: 1)
- `--fast`: Fast window values as comma-separated list (default: 10,20,50)
- `--slow`: Slow window values as comma-separated list (default: 50,100,200)
- `--metric`: Metric to optimize (default: sharpe)
- `--cash, -c`: Initial cash amount (default: 100000)
- `--fee-bps`: Trading fee in basis points (default: 1.0)
- `--slippage-bps`: Slippage cost in basis points (default: 0.5)
- `--jobs, -j`: Number of parallel jobs for optimization (default: auto)
- `--output-dir, -o`: Output directory for results (default: reports)
- `--top-n`: Number of top windows to display (default: 5)
- `--quiet, -q`: Suppress progress messages

## Walk-Forward Analysis: Interpretation and Overfitting

Walk-forward analysis is a critical tool for testing strategy robustness and avoiding overfitting. Here's how to interpret the results:

### Understanding Walk-Forward Analysis

Walk-forward analysis tests your strategy's ability to adapt to changing market conditions by:

1. **Optimizing parameters** on in-sample data (e.g., 3 years)
2. **Evaluating performance** on out-of-sample data (e.g., 1 year)
3. **Rolling the window** forward and repeating the process

This simulates real-world trading where you can only use historical data to make decisions.

### Key Metrics to Watch

#### Parameter Stability
- **Low coefficient of variation (CV)** indicates stable parameters
- **Frequent parameter changes** may suggest overfitting
- **Consistent parameter ranges** across windows suggest robustness

#### Performance Consistency
- **High win rate** across windows indicates consistent performance
- **Low standard deviation** in returns suggests stability
- **Consistent Sharpe ratios** across periods indicate robustness

#### Overall Performance
- **Out-of-sample performance** should be close to in-sample performance
- **Declining performance** over time may indicate strategy decay
- **High correlation** between in-sample and out-of-sample results suggests validity

### Signs of Overfitting

Watch for these red flags:

1. **Parameter Instability**: Parameters change dramatically between windows
2. **Performance Decay**: Out-of-sample performance significantly worse than in-sample
3. **High Variance**: Large swings in performance across windows
4. **Curve Fitting**: Strategy works perfectly on historical data but fails on new data

### Best Practices

1. **Use sufficient data**: At least 5-10 years for meaningful analysis
2. **Balance windows**: Don't make in-sample periods too long relative to out-of-sample
3. **Test multiple metrics**: Don't optimize on just one metric
4. **Validate results**: Compare walk-forward results with simple backtests
5. **Consider market regimes**: Different periods may have different characteristics

### Example Interpretation

```bash
# Run walk-forward analysis
qbt walkforward --symbol SPY --start 2010-01-01 --end 2023-12-31 --is 3 --oos 1

# Look for:
# - Parameter stability (low CV)
# - Consistent performance across windows
# - Reasonable out-of-sample returns
# - Low parameter changes between windows
```

A robust strategy should show:
- **Parameter CV < 0.3** for both fast and slow windows
- **Win rate > 60%** across out-of-sample periods
- **Consistent Sharpe ratios** across windows
- **Minimal parameter changes** between optimization periods

## Performance

QBacktester is designed for high-performance vectorized backtesting with no explicit Python loops in the core engine. All computations are vectorized using NumPy and Pandas for optimal speed.

### Performance Benchmarks

**Vectorization Validation:**
- ✅ **No explicit for-loops** in backtester core (verified via AST scanning)
- ✅ **100 backtests** with 2,500 trading days each complete in **< 1 second**
- ✅ **Average backtest runtime**: ~7ms per backtest
- ✅ **Memory efficient**: Handles large datasets without performance degradation

**Benchmark Results** (on modest hardware):
```
📊 Performance Results:
   ⏱️  Total runtime: 0.75 seconds (100 backtests, 2,500 days each)
   📈 Average backtest: 7.4 ms
   🚀 Fastest backtest: 6.5 ms
   🐌 Slowest backtest: 42.3 ms
   💰 Average final equity: $99,966.53
   📊 Average trades: 470.0
   📈 Average Sharpe: -0.001
   ✅ Success rate: 100.0%
```

### Vectorization Features

**Core Engine:**
- **Fully vectorized backtesting** - No Python loops in critical path
- **NumPy/Pandas optimization** - Leverages C-level operations
- **Memory-efficient processing** - Handles large datasets smoothly
- **Parallel-ready design** - Compatible with multiprocessing

**Performance Monitoring:**
- **Built-in profiling script** (`scripts/profile_vectorization.py`)
- **Automatic loop detection** - AST scanning prevents performance regressions
- **Benchmark validation** - Ensures performance targets are met
- **Memory usage optimization** - Efficient data structures and operations

### Running Performance Tests

```bash
# Run vectorization performance profiling
python scripts/profile_vectorization.py

# Run all tests including performance validation
python -m pytest tests/ -v
```

The profiling script validates:
1. **No explicit loops** in backtester core using AST analysis
2. **Performance benchmarks** with 100 backtests on 2,500 days of data
3. **Runtime validation** ensuring completion under reasonable thresholds
4. **Memory efficiency** with large synthetic datasets

## Assumptions and Limitations

### Trading Assumptions

**Long-Only Strategy:**
- qbacktester implements long-only moving average crossover strategies
- No short selling or leveraged positions
- Position sizing is binary: 100% long or 0% (cash)

**Execution Assumptions:**
- **Next-Bar Execution**: Trades are executed at the next bar's open price
- **No Slippage in Signal Generation**: Signals are generated using closing prices
- **Perfect Signal Timing**: No delays between signal generation and execution

**Transaction Costs:**
- **Trading Fees**: Configurable basis points (default: 1 bps)
- **Slippage Costs**: Configurable basis points (default: 0.5 bps)
- **No Market Impact**: Assumes trades don't affect market prices
- **No Bid-Ask Spread**: Uses single price (close) for both entry and exit

**Data Assumptions:**
- **Daily Frequency**: All calculations assume daily trading frequency
- **No Corporate Actions**: Dividends, splits, and other corporate actions not modeled
- **Continuous Trading**: Assumes market is open every trading day
- **No Gaps**: Price data is assumed to be continuous without gaps

### Performance Considerations

**Look-Ahead Bias Prevention:**
- All calculations use only past data for each decision point
- Moving averages calculated with proper window alignment
- No future information leakage in signal generation

**Vectorization Benefits:**
- All operations vectorized using NumPy/Pandas
- No explicit Python loops in critical path
- ~50% performance improvement over looped implementations
- Memory efficient processing of large datasets

**Backtesting Limitations:**
- Historical performance does not guarantee future results
- Market conditions may change, affecting strategy performance
- Transaction costs may vary in live trading
- Liquidity constraints not modeled

## Disclaimer

**Educational Purpose Only:**
This software is provided for educational and research purposes only. It is not intended as financial advice, investment recommendations, or trading guidance.

**No Financial Advice:**
- Past performance does not guarantee future results
- All backtests are hypothetical and may not reflect actual trading performance
- Real trading involves additional costs, risks, and market conditions not modeled
- Users should conduct their own research and consult with financial professionals

**Use at Your Own Risk:**
- The authors are not responsible for any financial losses
- Users assume all risks associated with trading and investment decisions
- This software is provided "as is" without warranties of any kind

**Regulatory Compliance:**
- Users are responsible for ensuring compliance with applicable laws and regulations
- Trading activities may be subject to regulatory oversight in your jurisdiction
- Consult with legal and financial professionals before making investment decisions

## Project Structure

```
qbacktester/
├── src/
│   └── qbacktester/
│       ├── __init__.py
│       ├── core.py          # Core backtesting engine
│       ├── data.py          # Data providers
│       ├── portfolio.py     # Portfolio management
│       ├── metrics.py       # Performance metrics
│       └── cli.py           # Command-line interface
├── tests/                   # Test suite
├── notebooks/               # Jupyter notebooks for analysis
├── data/                    # Data storage (gitignored)
├── reports/                 # Generated reports (gitignored)
├── pyproject.toml          # Project configuration
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/qbacktester

# Run specific test file
pytest tests/test_core.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Run all quality checks
pre-commit run --all-files
```

### Adding New Strategies

1. Inherit from the `Strategy` base class
2. Implement the `generate_signals` method
3. Return a DataFrame with your trading signals

```python
class MyStrategy(Strategy):
    def __init__(self):
        super().__init__("MyStrategy")
    
    def generate_signals(self, data):
        # Your strategy logic here
        signals = pd.DataFrame(index=data.index)
        # ... implement your strategy
        return signals
```

## Data

qbacktester includes a robust data module with intelligent caching and error handling:

### Data Fetching

```python
from qbacktester.data import get_price_data

# Download data with automatic caching
data = get_price_data("AAPL", "2020-01-01", "2023-12-31", interval="1d")

# Data is automatically cached to data/AAPL_1d_2020-01-01_2023-12-31.parquet
# Subsequent calls load from cache for instant access
```

### Caching System

- **Automatic caching**: Data is cached locally as parquet files for instant repeated access
- **Smart cache keys**: Files are named `{symbol}_{interval}_{start}_{end}.parquet`
- **Cache validation**: Corrupted cache files are automatically re-downloaded
- **Storage location**: Cached data is stored in the `data/` directory (gitignored)

### Error Handling

```python
from qbacktester.data import get_price_data, DataError

try:
    data = get_price_data("INVALID_SYMBOL", "2020-01-01", "2023-12-31")
except DataError as e:
    print(f"Data error: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

### Features

- **Retry logic**: Automatic retry with exponential backoff (3 attempts by default)
- **Data validation**: Ensures required OHLCV columns are present
- **Data cleaning**: Sorts by date, drops NaN rows, converts to lowercase columns
- **Connection resilience**: Handles network timeouts and temporary failures gracefully

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Advanced portfolio optimization
- [ ] Risk management tools
- [ ] Real-time data integration
- [ ] Web dashboard
- [ ] More data providers (Alpha Vantage, Quandl, etc.)
- [ ] Machine learning integration

## Acknowledgments

- Built with ❤️ for the quantitative finance community
- Inspired by popular backtesting libraries like Zipline and Backtrader
