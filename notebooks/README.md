# QBacktester Notebooks

This directory contains Jupyter notebooks demonstrating qbacktester functionality.

## Available Notebooks

### `quickstart.ipynb`
A comprehensive quickstart guide that demonstrates:

- **Real Data Analysis**: Loads SPY data from 2015-2025
- **Strategy Implementation**: 20/50 SMA crossover strategy
- **Performance Metrics**: Detailed analysis and visualization
- **Grid Search Optimization**: Parameter optimization with heatmap visualization
- **Vectorization Benefits**: Performance demonstration and look-ahead bias prevention

## Running the Notebooks

1. **Install Dependencies**:
   ```bash
   pip install jupyter matplotlib seaborn pandas numpy yfinance
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Open the Notebook**: Navigate to `notebooks/quickstart.ipynb`

## Key Features Demonstrated

### Data Loading
- Real market data fetching with error handling
- Synthetic data fallback for demonstration
- Data validation and preprocessing

### Strategy Analysis
- Vectorized backtesting implementation
- Transaction cost modeling
- Performance metrics calculation
- Risk analysis and drawdown visualization

### Optimization
- Grid search parameter optimization
- Heatmap visualization of parameter space
- Robust parameter identification

### Visualization
- Price action and moving averages
- Trading signals overlay
- Equity curve and drawdown plots
- Performance metrics tables

## Educational Content

The notebooks include detailed explanations of:
- **Look-ahead bias prevention** techniques
- **Vectorization benefits** for performance
- **Transaction cost modeling** best practices
- **Parameter optimization** strategies
- **Risk management** considerations

## Requirements

- Python 3.11+
- Jupyter Notebook
- qbacktester package
- matplotlib, seaborn, pandas, numpy, yfinance

## Notes

- The notebooks use `sys.path` adjustments to import from the `src/` directory
- Real data fetching requires internet connection
- Synthetic data is used as fallback for demonstration purposes
- All visualizations are optimized for notebook display
