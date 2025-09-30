# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-27

### Added
- Initial release of qbacktester
- **Core Backtesting Engine**
  - Vectorized backtesting with pandas/NumPy
  - Moving average crossover strategy implementation
  - Transaction cost modeling (fees and slippage)
  - Look-ahead bias prevention
- **Performance Metrics**
  - Sharpe ratio, CAGR, max drawdown, Calmar ratio
  - Sortino ratio, hit rate, win/loss analysis
  - VaR and CVaR risk metrics
  - Volatility and drawdown analysis
- **Data Management**
  - Yahoo Finance integration with automatic caching
  - Retry logic with exponential backoff
  - Parquet file caching for performance
  - Data validation and error handling
- **Command Line Interface**
  - `qbt run` - Run single backtests
  - `qbt optimize` - Grid search parameter optimization
  - `qbt walkforward` - Walk-forward analysis
  - Rich console output with progress indicators
- **Visualization**
  - Equity curve and drawdown plots
  - Price action with trading signals
  - Grid search heatmaps
  - Walk-forward analysis charts
- **Parameter Optimization**
  - Grid search with parallel processing
  - Multiple optimization metrics (Sharpe, CAGR, Calmar, Max DD)
  - Results ranking and visualization
- **Walk-Forward Analysis**
  - Rolling window optimization
  - Out-of-sample evaluation
  - Parameter stability analysis
  - Performance consistency metrics
- **Jupyter Notebooks**
  - Comprehensive quickstart guide
  - Real data analysis examples
  - Educational content on vectorization and bias prevention
- **Testing and Quality**
  - Comprehensive unit test suite
  - Performance profiling and validation
  - AST-based loop detection for vectorization
  - Transaction cost impact testing
- **Documentation**
  - Detailed README with examples
  - API documentation
  - Performance benchmarks
  - Assumptions and limitations

### Performance
- **Vectorization**: ~50% runtime improvement vs looped implementations
- **Speed**: 100 backtests with 2,500 days each complete in < 1 second
- **Memory**: Efficient processing of large datasets
- **Scalability**: Parallel processing support for optimization

### Technical Details
- **Python**: 3.11+ support
- **Dependencies**: pandas, numpy, yfinance, matplotlib, seaborn, rich
- **Architecture**: Modular design with clear separation of concerns
- **Code Quality**: Type hints, comprehensive testing, linting

---

## [Unreleased]

### Planned
- Additional strategy implementations (RSI, MACD, Bollinger Bands)
- Portfolio-level backtesting
- Risk management features
- Live trading integration
- Additional data providers
- Advanced visualization features
