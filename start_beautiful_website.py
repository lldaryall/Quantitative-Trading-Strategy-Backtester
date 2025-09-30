#!/usr/bin/env python3
"""
Start the Beautiful QBacktester Website
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def main():
    print("🎨 Starting Beautiful QBacktester Website")
    print("=" * 50)
    
    # Change to docs directory
    docs_dir = Path(__file__).parent / "docs"
    os.chdir(docs_dir)
    
    print("📁 Working directory:", os.getcwd())
    print("📄 Website files:")
    for file in ['index.html', 'styles-simple.css', 'script.js']:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
    
    print("\n🚀 Starting web server...")
    
    try:
        # Start HTTP server
        process = subprocess.Popen([
            sys.executable, "-m", "http.server", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(2)
        
        print("✅ Beautiful website is running!")
        print("🌐 URL: http://localhost:8000")
        print("📊 Interactive Backtester: http://localhost:8000#backtester")
        print("\n🎨 Features:")
        print("   • Beautiful gradient hero section")
        print("   • Professional typography (Inter font)")
        print("   • Interactive backtesting form")
        print("   • Real-time charts and metrics")
        print("   • Responsive design")
        print("   • Demo data functionality")
        print("\n💡 The website will open in your browser...")
        print("🛑 Press Ctrl+C to stop the server")
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8000")
        except:
            print("⚠️  Could not open browser automatically")
            print("   Please open http://localhost:8000 in your browser")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            process.terminate()
            print("✅ Server stopped")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
