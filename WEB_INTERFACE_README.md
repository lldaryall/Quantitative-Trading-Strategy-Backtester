# QBacktester Web Interface

A beautiful, interactive web interface for running quantitative trading strategy backtests without using the terminal.

## 🌟 Features

### 🎨 Beautiful & Interactive UI
- **Modern Design**: Clean, professional interface with smooth animations
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Real-time Feedback**: Loading states, progress indicators, and instant results
- **Interactive Charts**: Zoom, hover, and explore your backtest results

### 📊 Real Backtesting Engine
- **Actual QBacktester Integration**: Uses your real Python backtesting library
- **Live Data**: Fetches real market data from Yahoo Finance
- **Comprehensive Metrics**: All the same metrics as the command-line version
- **Professional Plots**: Equity curves, drawdown analysis, and price action charts

### 🚀 Easy to Use
- **No Terminal Required**: Everything runs in your web browser
- **Form-based Input**: Simple forms for all parameters
- **Demo Data**: One-click demo data for quick testing
- **Export Results**: Download results as CSV files

## 🚀 Quick Start

### Option 1: One-Command Startup (Recommended)
```bash
# Install web dependencies
pip install -r requirements-web.txt

# Start everything with one command
python start_web_interface.py
```

This will:
- Start the backend API on http://localhost:5000
- Start the frontend on http://localhost:8000
- Open your browser automatically
- Show you the interactive backtester

### Option 2: Manual Startup

#### Start Backend API
```bash
# In one terminal
python backend.py
```

#### Start Frontend
```bash
# In another terminal
cd docs
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## 📖 How to Use

### 1. **Input Parameters**
- **Stock Symbol**: Enter any valid ticker (AAPL, SPY, MSFT, etc.)
- **Date Range**: Select start and end dates
- **Strategy Parameters**: Set fast and slow moving average windows
- **Capital & Costs**: Set initial capital, commission, and slippage

### 2. **Run Backtest**
- Click "Run Backtest" to execute with real data
- Watch the loading animation while it processes
- Results appear instantly with professional visualizations

### 3. **Analyze Results**
- **Metrics Dashboard**: 8 key performance indicators
- **Interactive Chart**: Equity curve with zoom and hover
- **Trade History**: Detailed trade-by-trade results
- **Analysis Plots**: Professional equity, drawdown, and price charts

### 4. **Export & Share**
- Download results as CSV
- Share charts and metrics
- Save configurations for later

## 🎯 Example Workflows

### Quick Demo
1. Click "Load Demo Data" for instant parameters
2. Click "Run Backtest" to see results
3. Explore the interactive charts and metrics

### Custom Strategy
1. Enter your stock symbol (e.g., "AAPL")
2. Set date range (e.g., 2020-2023)
3. Adjust moving average windows (e.g., 20/50)
4. Run backtest and analyze results

### Parameter Testing
1. Run backtest with current parameters
2. Adjust fast/slow windows
3. Run again to compare results
4. Export both results for analysis

## 🔧 Technical Details

### Backend API (Flask)
- **Port**: 5000
- **Endpoints**:
  - `GET /api/health` - Health check
  - `POST /api/backtest` - Run backtest
  - `GET /api/symbols` - Get popular symbols
  - `POST /api/optimize` - Parameter optimization

### Frontend (Static HTML/CSS/JS)
- **Port**: 8000
- **Features**: Chart.js, responsive design, real-time updates
- **Fallback**: Works offline with simulated data if backend unavailable

### Data Flow
1. User fills form in browser
2. JavaScript sends request to Flask API
3. Flask calls QBacktester library
4. Real data fetched from Yahoo Finance
5. Backtest executed with actual algorithms
6. Results formatted and sent back
7. Frontend displays charts and metrics

## 🛠️ Development

### Project Structure
```
├── backend.py              # Flask API server
├── start_web_interface.py  # Startup script
├── requirements-web.txt    # Web dependencies
├── docs/                   # Frontend files
│   ├── index.html         # Main webpage
│   ├── styles.css         # Styling
│   └── script.js          # Interactive functionality
└── src/qbacktester/       # Your Python library
```

### Adding New Features

#### Backend (Flask API)
- Add new endpoints in `backend.py`
- Integrate with QBacktester modules
- Return JSON responses

#### Frontend (JavaScript)
- Add UI elements in `index.html`
- Style with CSS in `styles.css`
- Add functionality in `script.js`

### Customization

#### Styling
- Modify CSS variables in `styles.css`
- Change colors, fonts, spacing
- Add new animations and effects

#### Functionality
- Add new form fields
- Create additional charts
- Implement new analysis tools

## 🐛 Troubleshooting

### Backend Issues
```bash
# Check if backend is running
curl http://localhost:5000/api/health

# Check backend logs
python backend.py
```

### Frontend Issues
```bash
# Check if frontend is accessible
curl http://localhost:8000

# Check browser console for JavaScript errors
# Press F12 in browser and look at Console tab
```

### Common Problems

1. **"Connection refused" error**
   - Make sure backend is running on port 5000
   - Check if port is already in use

2. **"No data found" error**
   - Check if symbol is valid
   - Verify date range is correct
   - Ensure internet connection for data fetching

3. **Charts not displaying**
   - Check browser console for errors
   - Ensure Chart.js is loaded
   - Verify data format

## 📊 Performance

### Backend Performance
- **Data Fetching**: ~2-5 seconds for 3 years of data
- **Backtest Execution**: ~1-3 seconds for moving average strategy
- **Plot Generation**: ~1-2 seconds for all charts
- **Total Response Time**: ~5-10 seconds end-to-end

### Frontend Performance
- **Page Load**: < 2 seconds
- **Chart Rendering**: < 1 second
- **Form Interactions**: Instant response
- **Mobile Performance**: Optimized for touch devices

## 🔒 Security

### Current Implementation
- **CORS Enabled**: Allows cross-origin requests
- **Input Validation**: Form validation on both frontend and backend
- **Error Handling**: Graceful error handling and user feedback

### Production Considerations
- Add authentication if needed
- Implement rate limiting
- Use HTTPS in production
- Add input sanitization

## 🚀 Deployment

### Local Development
```bash
python start_web_interface.py
```

### Production Deployment
1. **Backend**: Deploy Flask app to cloud service
2. **Frontend**: Deploy static files to CDN
3. **Database**: Add database for user data if needed
4. **Monitoring**: Add logging and monitoring

### Docker Deployment
```dockerfile
# Backend Dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements-web.txt
EXPOSE 5000
CMD ["python", "backend.py"]
```

## 📈 Future Enhancements

### Planned Features
- **Parameter Optimization**: Grid search through web interface
- **Walk-Forward Analysis**: Out-of-sample testing
- **Portfolio Backtesting**: Multiple assets
- **Real-time Data**: Live market data integration
- **User Accounts**: Save and share strategies
- **Strategy Library**: Pre-built strategies

### Integration Opportunities
- **Database**: Store results and strategies
- **Authentication**: User login and data persistence
- **Notifications**: Email alerts for strategy signals
- **Mobile App**: Native mobile application
- **API**: Public API for third-party integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

Same as the main QBacktester project (MIT License).

## 🆘 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: This README and inline comments

---

**Built with ❤️ for the quantitative finance community**

*Transform your terminal-based backtesting into a beautiful, interactive web experience!*
