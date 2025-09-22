# QBacktester - GitHub Pages Website

This repository contains the GitHub Pages website for QBacktester, a powerful Python library for quantitative trading strategy backtesting.

## 🌐 Live Website

Visit the live website at: [https://yourusername.github.io/Quantitative-Trading-Strategy-Backtester/](https://yourusername.github.io/Quantitative-Trading-Strategy-Backtester/)

## 📁 Website Structure

```
├── index.html          # Main website page
├── styles.css          # CSS styles and responsive design
├── script.js           # JavaScript for interactivity
├── .github/
│   └── workflows/
│       └── deploy.yml  # GitHub Pages deployment workflow
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🚀 Features

### Modern Design
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Modern UI**: Clean, professional design with smooth animations
- **Dark Code Blocks**: Syntax-highlighted code examples with copy functionality
- **Interactive Elements**: Smooth scrolling, tab navigation, and hover effects

### Content Sections
- **Hero Section**: Eye-catching introduction with code example
- **Features**: Comprehensive overview of QBacktester capabilities
- **Quick Start**: Installation and usage examples with interactive tabs
- **Examples**: Real performance data and optimization results
- **Performance**: Benchmark statistics with animated counters
- **Documentation**: Links to guides and API reference

### Technical Features
- **Mobile Navigation**: Hamburger menu for mobile devices
- **Smooth Scrolling**: Seamless navigation between sections
- **Copy Code**: One-click copying of code examples
- **Performance Animations**: Animated counters and scroll effects
- **Accessibility**: Keyboard navigation and focus management
- **SEO Optimized**: Proper meta tags and semantic HTML

## 🛠️ Development

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Quantitative-Trading-Strategy-Backtester.git
   cd Quantitative-Trading-Strategy-Backtester
   ```

2. **Serve locally** (choose one method):
   
   **Option A: Python HTTP Server**
   ```bash
   python -m http.server 8000
   ```
   Then visit: http://localhost:8000
   
   **Option B: Node.js HTTP Server**
   ```bash
   npx http-server -p 8000
   ```
   Then visit: http://localhost:8000
   
   **Option C: Live Server (VS Code)**
   - Install "Live Server" extension in VS Code
   - Right-click on `index.html` and select "Open with Live Server"

### Customization

#### Updating Content
- **Main content**: Edit `index.html`
- **Styling**: Modify `styles.css`
- **Interactivity**: Update `script.js`

#### Key Areas to Customize
1. **Repository URLs**: Update all GitHub links to match your repository
2. **Contact Information**: Add your social media and contact details
3. **Performance Data**: Update example results with real data
4. **Color Scheme**: Modify CSS variables in `:root` selector
5. **Logo/Branding**: Update the logo and brand colors

#### Color Customization
The website uses CSS custom properties for easy theming. Update these in `styles.css`:

```css
:root {
    --primary-color: #6366f1;    /* Main brand color */
    --secondary-color: #8b5cf6;  /* Secondary color */
    --accent-color: #06b6d4;     /* Accent color */
    --success-color: #10b981;    /* Success/positive values */
    --error-color: #ef4444;      /* Error/negative values */
}
```

## 🚀 Deployment

### Automatic Deployment
The website is automatically deployed to GitHub Pages when you push to the `main` branch. The deployment is handled by the GitHub Actions workflow in `.github/workflows/deploy.yml`.

### Manual Deployment
If you need to deploy manually:

1. **Enable GitHub Pages**:
   - Go to repository Settings
   - Navigate to Pages section
   - Select "GitHub Actions" as source

2. **Push changes**:
   ```bash
   git add .
   git commit -m "Update website"
   git push origin main
   ```

3. **Check deployment**:
   - Go to Actions tab in GitHub
   - Monitor the deployment workflow
   - Visit your GitHub Pages URL

### Custom Domain (Optional)
To use a custom domain:

1. **Add CNAME file**:
   ```bash
   echo "yourdomain.com" > CNAME
   git add CNAME
   git commit -m "Add custom domain"
   git push origin main
   ```

2. **Configure DNS**:
   - Add CNAME record pointing to `yourusername.github.io`
   - Or add A records for GitHub Pages IPs

## 📱 Browser Support

- **Modern Browsers**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Mobile**: iOS Safari 14+, Chrome Mobile 90+
- **Features**: CSS Grid, Flexbox, CSS Custom Properties, Intersection Observer

## 🔧 Technical Details

### Dependencies
- **No build process required** - Pure HTML, CSS, and JavaScript
- **External CDNs**: Font Awesome, Google Fonts, Prism.js
- **No Node.js dependencies** - Can be served from any static host

### Performance
- **Optimized Images**: SVG icons and minimal external resources
- **Efficient CSS**: Uses CSS custom properties and modern layout techniques
- **Lazy Loading**: Intersection Observer for scroll animations
- **Minimal JavaScript**: Vanilla JS with no heavy frameworks

### SEO Features
- **Semantic HTML**: Proper heading structure and semantic elements
- **Meta Tags**: Description, viewport, and Open Graph tags
- **Structured Data**: Ready for schema.org markup
- **Fast Loading**: Optimized for Core Web Vitals

## 📊 Analytics (Optional)

The website includes placeholder analytics tracking. To add real analytics:

1. **Google Analytics 4**:
   ```html
   <!-- Add to <head> section -->
   <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
   <script>
     window.dataLayer = window.dataLayer || [];
     function gtag(){dataLayer.push(arguments);}
     gtag('js', new Date());
     gtag('config', 'GA_MEASUREMENT_ID');
   </script>
   ```

2. **Update tracking calls** in `script.js`:
   ```javascript
   function trackEvent(eventName, properties = {}) {
     gtag('event', eventName, properties);
   }
   ```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Update HTML, CSS, or JavaScript
4. **Test locally**: Ensure everything works in your browser
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

## 📝 License

This website is part of the QBacktester project and is licensed under the MIT License. See the main project's [LICENSE](qbacktester/LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Quantitative-Trading-Strategy-Backtester/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Quantitative-Trading-Strategy-Backtester/discussions)
- **Documentation**: [QBacktester Docs](https://github.com/yourusername/Quantitative-Trading-Strategy-Backtester/tree/main/qbacktester)

## 🔄 Updates

The website is automatically updated when the main QBacktester project is updated. Key areas that may need manual updates:

- **Performance benchmarks**: Update with new test results
- **Feature list**: Add new capabilities as they're developed
- **Example data**: Refresh with latest backtest results
- **Documentation links**: Update as new guides are added

---

**Built with ❤️ for the quantitative finance community**
