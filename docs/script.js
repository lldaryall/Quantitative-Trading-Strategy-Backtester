// Mobile Navigation Toggle
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.getElementById('hamburger');
    const navMenu = document.getElementById('nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            const isExpanded = navMenu.classList.contains('active');
            navMenu.classList.toggle('active');
            hamburger.classList.toggle('active');
            hamburger.setAttribute('aria-expanded', !isExpanded);
        });
    }
    
    // Close mobile menu when clicking on a link
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            hamburger.classList.remove('active');
            hamburger.setAttribute('aria-expanded', 'false');
        });
    });
    
    // Close mobile menu when clicking outside
    document.addEventListener('click', function(event) {
        if (!hamburger.contains(event.target) && !navMenu.contains(event.target)) {
            navMenu.classList.remove('active');
            hamburger.classList.remove('active');
            hamburger.setAttribute('aria-expanded', 'false');
        }
    });
});

// Tab Functionality
document.addEventListener('DOMContentLoaded', function() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Remove active class from all buttons and panels
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));
            
            // Add active class to clicked button and corresponding panel
            this.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });
});

// Smooth scrolling for navigation links
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
});

// Navbar scroll effect
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
});

// Intersection Observer for animations
document.addEventListener('DOMContentLoaded', function() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animatedElements = document.querySelectorAll('.feature-card, .example-card, .performance-card, .doc-card');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Copy code functionality
document.addEventListener('DOMContentLoaded', function() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(block => {
        const pre = block.parentElement;
        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        
        const copyBtn = document.createElement('button');
        copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
        copyBtn.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.5);
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;
        
        wrapper.appendChild(copyBtn);
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);
        
        wrapper.addEventListener('mouseenter', () => {
            copyBtn.style.opacity = '1';
        });
        
        wrapper.addEventListener('mouseleave', () => {
            copyBtn.style.opacity = '0';
        });
        
        copyBtn.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(block.textContent);
                copyBtn.innerHTML = '<i class="fas fa-check"></i>';
                copyBtn.style.background = '#10b981';
                
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
                    copyBtn.style.background = 'rgba(0, 0, 0, 0.5)';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
            }
        });
    });
});

// Performance metrics animation
document.addEventListener('DOMContentLoaded', function() {
    const performanceNumbers = document.querySelectorAll('.performance-number');
    
    const animateNumber = (element, target) => {
        let current = 0;
        const increment = target / 50;
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            
            if (target.toString().includes('%')) {
                element.textContent = current.toFixed(1) + '%';
            } else if (target.toString().includes('s')) {
                element.textContent = current.toFixed(1) + 's';
            } else if (target.toString().includes('ms')) {
                element.textContent = current.toFixed(1) + 'ms';
            } else {
                element.textContent = Math.floor(current).toLocaleString();
            }
        }, 30);
    };
    
    const performanceObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const text = entry.target.textContent;
                let target;
                
                if (text.includes('0.75s')) {
                    target = 0.75;
                    animateNumber(entry.target, target);
                } else if (text.includes('7.4ms')) {
                    target = 7.4;
                    animateNumber(entry.target, target);
                } else if (text.includes('100%')) {
                    target = 100;
                    animateNumber(entry.target, target);
                } else if (text.includes('50%')) {
                    target = 50;
                    animateNumber(entry.target, target);
                }
                
                performanceObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    performanceNumbers.forEach(num => {
        performanceObserver.observe(num);
    });
});

// Typing animation for hero title
document.addEventListener('DOMContentLoaded', function() {
    const heroTitle = document.querySelector('.hero-title');
    if (!heroTitle) return;
    
    const text = heroTitle.innerHTML;
    heroTitle.innerHTML = '';
    
    let i = 0;
    const typeWriter = () => {
        if (i < text.length) {
            heroTitle.innerHTML += text.charAt(i);
            i++;
            setTimeout(typeWriter, 50);
        }
    };
    
    // Start typing animation after a short delay
    setTimeout(typeWriter, 500);
});

// Parallax effect for hero section
window.addEventListener('scroll', function() {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    const heroVisual = document.querySelector('.hero-visual');
    
    if (hero && heroVisual) {
        const rate = scrolled * -0.5;
        heroVisual.style.transform = `translateY(${rate}px)`;
    }
});

// Add loading states for buttons
document.addEventListener('DOMContentLoaded', function() {
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(btn => {
        btn.addEventListener('click', function(e) {
            if (this.href && this.href.startsWith('#')) {
                return; // Don't add loading state for anchor links
            }
            
            const originalText = this.innerHTML;
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            this.disabled = true;
            
            // Reset after 2 seconds (for demo purposes)
            setTimeout(() => {
                this.innerHTML = originalText;
                this.disabled = false;
            }, 2000);
        });
    });
});

// Add tooltips for badges
document.addEventListener('DOMContentLoaded', function() {
    const badges = document.querySelectorAll('.badge');
    
    badges.forEach(badge => {
        badge.addEventListener('mouseenter', function() {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = this.textContent.trim();
            tooltip.style.cssText = `
                position: absolute;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                white-space: nowrap;
                z-index: 1000;
                pointer-events: none;
            `;
            
            document.body.appendChild(tooltip);
            
            const rect = this.getBoundingClientRect();
            tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
            tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
        });
        
        badge.addEventListener('mouseleave', function() {
            const tooltip = document.querySelector('.tooltip');
            if (tooltip) {
                tooltip.remove();
            }
        });
    });
});

// Add keyboard navigation support
document.addEventListener('keydown', function(e) {
    // ESC key closes mobile menu
    if (e.key === 'Escape') {
        const navMenu = document.getElementById('nav-menu');
        const hamburger = document.getElementById('hamburger');
        
        if (navMenu && navMenu.classList.contains('active')) {
            navMenu.classList.remove('active');
            hamburger.classList.remove('active');
        }
    }
    
    // Tab key navigation for tab buttons
    if (e.key === 'Tab' && e.target.classList.contains('tab-btn')) {
        const tabBtns = Array.from(document.querySelectorAll('.tab-btn'));
        const currentIndex = tabBtns.indexOf(e.target);
        
        if (e.shiftKey) {
            // Shift + Tab: go to previous tab
            if (currentIndex > 0) {
                tabBtns[currentIndex - 1].focus();
            }
        } else {
            // Tab: go to next tab
            if (currentIndex < tabBtns.length - 1) {
                tabBtns[currentIndex + 1].focus();
            }
        }
    }
});

// Add error handling for missing elements
document.addEventListener('DOMContentLoaded', function() {
    // Check if required elements exist before adding event listeners
    const requiredElements = [
        'hamburger',
        'nav-menu',
        'tab-btn',
        'tab-panel'
    ];
    
    requiredElements.forEach(id => {
        const element = document.getElementById(id);
        if (!element) {
            console.warn(`Element with id '${id}' not found`);
        }
    });
});

// Add analytics tracking (placeholder)
function trackEvent(eventName, properties = {}) {
    // Placeholder for analytics tracking
    console.log('Event tracked:', eventName, properties);
    
    // Example: Google Analytics 4
    // gtag('event', eventName, properties);
}

// Track button clicks
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('btn')) {
        const buttonText = e.target.textContent.trim();
        trackEvent('button_click', {
            button_text: buttonText,
            button_type: e.target.classList.contains('btn-primary') ? 'primary' : 'secondary'
        });
    }
});

// Track tab switches
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('tab-btn')) {
        const tabName = e.target.getAttribute('data-tab');
        trackEvent('tab_switch', {
            tab_name: tabName
        });
    }
});

// Add performance monitoring
window.addEventListener('load', function() {
    // Track page load time
    const loadTime = performance.now();
    trackEvent('page_load', {
        load_time: Math.round(loadTime)
    });
    
    // Track largest contentful paint
    if ('PerformanceObserver' in window) {
        const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1];
            trackEvent('lcp', {
                lcp_time: Math.round(lastEntry.startTime)
            });
        });
        
        observer.observe({ entryTypes: ['largest-contentful-paint'] });
    }
});

// Interactive Backtester Functionality
document.addEventListener('DOMContentLoaded', function() {
    const backtestForm = document.getElementById('backtest-form');
    const resultsDiv = document.getElementById('backtester-results');
    const metricsGrid = document.getElementById('metrics-grid');
    const tradesTable = document.getElementById('trades-table');
    const runButton = document.getElementById('run-backtest');
    const loadDemoButton = document.getElementById('load-demo');
    const exportButton = document.getElementById('export-results');
    
    let currentChart = null;
    let currentResults = null;
    
    // Demo data for quick testing
    const demoData = {
        'AAPL': {
            symbol: 'AAPL',
            startDate: '2020-01-01',
            endDate: '2023-12-31',
            fastWindow: 20,
            slowWindow: 50,
            initialCapital: 100000,
            commission: 1.0,
            slippage: 0.5
        },
        'SPY': {
            symbol: 'SPY',
            startDate: '2018-01-01',
            endDate: '2023-12-31',
            fastWindow: 15,
            slowWindow: 45,
            initialCapital: 50000,
            commission: 0.5,
            slippage: 0.25
        },
        'MSFT': {
            symbol: 'MSFT',
            startDate: '2019-01-01',
            endDate: '2023-12-31',
            fastWindow: 25,
            slowWindow: 75,
            initialCapital: 75000,
            commission: 1.5,
            slippage: 0.75
        }
    };
    
    // Load demo data
    loadDemoButton.addEventListener('click', function() {
        const symbols = Object.keys(demoData);
        const randomSymbol = symbols[Math.floor(Math.random() * symbols.length)];
        const demo = demoData[randomSymbol];
        
        document.getElementById('symbol').value = demo.symbol;
        document.getElementById('start-date').value = demo.startDate;
        document.getElementById('end-date').value = demo.endDate;
        document.getElementById('fast-window').value = demo.fastWindow;
        document.getElementById('slow-window').value = demo.slowWindow;
        document.getElementById('initial-capital').value = demo.initialCapital;
        document.getElementById('commission').value = demo.commission;
        document.getElementById('slippage').value = demo.slippage;
        
        trackEvent('demo_loaded', { symbol: demo.symbol });
    });
    
    // Form submission
    backtestForm.addEventListener('submit', function(e) {
        e.preventDefault();
        runBacktest();
    });
    
    // Export results
    exportButton.addEventListener('click', function() {
        if (currentResults) {
            exportResults(currentResults);
        }
    });
    
    function runBacktest() {
        const formData = new FormData(backtestForm);
        const params = {
            symbol: formData.get('symbol').toUpperCase(),
            startDate: formData.get('start-date'),
            endDate: formData.get('end-date'),
            fastWindow: parseInt(formData.get('fast-window')),
            slowWindow: parseInt(formData.get('slow-window')),
            initialCapital: parseFloat(formData.get('initial-capital')),
            commission: parseFloat(formData.get('commission')),
            slippage: parseFloat(formData.get('slippage'))
        };
        
        // Validate parameters
        if (params.fastWindow >= params.slowWindow) {
            alert('Fast window must be less than slow window');
            return;
        }
        
        if (params.startDate >= params.endDate) {
            alert('Start date must be before end date');
            return;
        }
        
        // Show loading state
        runButton.disabled = true;
        runButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';
        
        // Simulate backtest (in real implementation, this would call your Python backend)
        setTimeout(() => {
            const results = simulateBacktest(params);
            displayResults(results);
            runButton.disabled = false;
            runButton.innerHTML = '<i class="fas fa-play"></i> Run Backtest';
            
            trackEvent('backtest_completed', {
                symbol: params.symbol,
                fast_window: params.fastWindow,
                slow_window: params.slowWindow
            });
        }, 2000);
    }
    
    function simulateBacktest(params) {
        // This is a simulation - in real implementation, call your Python backend
        const days = Math.floor((new Date(params.endDate) - new Date(params.startDate)) / (1000 * 60 * 60 * 24));
        const trades = Math.floor(days / 30) + Math.floor(Math.random() * 10);
        
        // Generate realistic results
        const totalReturn = (Math.random() - 0.3) * 100; // -30% to +70%
        const sharpeRatio = Math.random() * 2 - 0.5; // -0.5 to 1.5
        const maxDrawdown = -Math.random() * 25; // 0 to -25%
        const winRate = Math.random() * 40 + 40; // 40% to 80%
        
        const finalCapital = params.initialCapital * (1 + totalReturn / 100);
        const totalTrades = trades;
        const avgWin = Math.random() * 5 + 2; // 2% to 7%
        const avgLoss = -Math.random() * 3 - 1; // -1% to -4%
        
        // Generate equity curve data
        const equityData = [];
        const dates = [];
        let currentEquity = params.initialCapital;
        
        for (let i = 0; i < days; i += 5) { // Sample every 5 days
            const date = new Date(params.startDate);
            date.setDate(date.getDate() + i);
            dates.push(date.toISOString().split('T')[0]);
            
            // Add some realistic volatility
            const change = (Math.random() - 0.5) * 0.02;
            currentEquity *= (1 + change);
            equityData.push(currentEquity);
        }
        
        // Generate trade history
        const tradeHistory = [];
        for (let i = 0; i < Math.min(totalTrades, 20); i++) {
            const tradeDate = new Date(params.startDate);
            tradeDate.setDate(tradeDate.getDate() + Math.floor(Math.random() * days));
            
            const isWin = Math.random() < (winRate / 100);
            const pnl = isWin ? 
                (Math.random() * 5 + 1) : 
                -(Math.random() * 3 + 1);
            
            tradeHistory.push({
                date: tradeDate.toISOString().split('T')[0],
                action: Math.random() > 0.5 ? 'BUY' : 'SELL',
                price: 100 + Math.random() * 200,
                quantity: Math.floor(Math.random() * 100) + 10,
                pnl: pnl,
                isWin: isWin
            });
        }
        
        return {
            ...params,
            totalReturn: totalReturn,
            sharpeRatio: sharpeRatio,
            maxDrawdown: maxDrawdown,
            winRate: winRate,
            finalCapital: finalCapital,
            totalTrades: totalTrades,
            avgWin: avgWin,
            avgLoss: avgLoss,
            cagr: (Math.pow(finalCapital / params.initialCapital, 365 / days) - 1) * 100,
            volatility: Math.random() * 20 + 10,
            calmarRatio: Math.abs(totalReturn / maxDrawdown),
            sortinoRatio: Math.random() * 1.5 + 0.5,
            equityData: equityData,
            dates: dates,
            tradeHistory: tradeHistory
        };
    }
    
    function displayResults(results) {
        currentResults = results;
        
        // Show results section
        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth' });
        
        // Display metrics
        displayMetrics(results);
        
        // Create chart
        createChart(results);
        
        // Display trade history
        displayTradeHistory(results.tradeHistory);
    }
    
    function displayMetrics(results) {
        const metrics = [
            { label: 'Total Return', value: `${results.totalReturn.toFixed(2)}%`, class: results.totalReturn >= 0 ? 'positive' : 'negative' },
            { label: 'CAGR', value: `${results.cagr.toFixed(2)}%`, class: results.cagr >= 0 ? 'positive' : 'negative' },
            { label: 'Sharpe Ratio', value: results.sharpeRatio.toFixed(2), class: 'neutral' },
            { label: 'Max Drawdown', value: `${results.maxDrawdown.toFixed(2)}%`, class: 'negative' },
            { label: 'Win Rate', value: `${results.winRate.toFixed(1)}%`, class: 'neutral' },
            { label: 'Total Trades', value: results.totalTrades.toString(), class: 'neutral' },
            { label: 'Final Capital', value: `$${results.finalCapital.toLocaleString()}`, class: results.finalCapital >= results.initialCapital ? 'positive' : 'negative' },
            { label: 'Volatility', value: `${results.volatility.toFixed(1)}%`, class: 'neutral' }
        ];
        
        metricsGrid.innerHTML = metrics.map(metric => `
            <div class="metric-card">
                <div class="metric-label">${metric.label}</div>
                <div class="metric-value ${metric.class}">${metric.value}</div>
            </div>
        `).join('');
    }
    
    function createChart(results) {
        const ctx = document.getElementById('equity-chart').getContext('2d');
        
        if (currentChart) {
            currentChart.destroy();
        }
        
        currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: results.dates,
                datasets: [{
                    label: 'Portfolio Value',
                    data: results.equityData,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `${results.symbol} Moving Average Strategy (${results.fastWindow}/${results.slowWindow})`
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'month'
                        }
                    },
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }
    
    function displayTradeHistory(trades) {
        const tableHTML = `
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Action</th>
                        <th>Price</th>
                        <th>Quantity</th>
                        <th>P&L</th>
                        <th>Result</th>
                    </tr>
                </thead>
                <tbody>
                    ${trades.map(trade => `
                        <tr>
                            <td>${trade.date}</td>
                            <td>${trade.action}</td>
                            <td>$${trade.price.toFixed(2)}</td>
                            <td>${trade.quantity}</td>
                            <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}%</td>
                            <td>
                                <span class="trade-result ${trade.isWin ? 'win' : 'loss'}">
                                    ${trade.isWin ? '✓' : '✗'}
                                </span>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        tradesTable.innerHTML = tableHTML;
    }
    
    function exportResults(results) {
        const csvContent = [
            ['Metric', 'Value'],
            ['Symbol', results.symbol],
            ['Start Date', results.startDate],
            ['End Date', results.endDate],
            ['Fast Window', results.fastWindow],
            ['Slow Window', results.slowWindow],
            ['Initial Capital', results.initialCapital],
            ['Total Return', `${results.totalReturn.toFixed(2)}%`],
            ['CAGR', `${results.cagr.toFixed(2)}%`],
            ['Sharpe Ratio', results.sharpeRatio.toFixed(2)],
            ['Max Drawdown', `${results.maxDrawdown.toFixed(2)}%`],
            ['Win Rate', `${results.winRate.toFixed(1)}%`],
            ['Total Trades', results.totalTrades],
            ['Final Capital', results.finalCapital],
            ['Volatility', `${results.volatility.toFixed(1)}%`]
        ].map(row => row.join(',')).join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `backtest_${results.symbol}_${results.startDate}_${results.endDate}.csv`;
        a.click();
        window.URL.revokeObjectURL(url);
        
        trackEvent('results_exported', { symbol: results.symbol });
    }
});
