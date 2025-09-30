#!/usr/bin/env python3
"""
Simple startup script for QBacktester Website
Starts the frontend web server with proper styling
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def main():
    """Start the website"""
    print("🎯 Starting QBacktester Website")
    print("=" * 40)
    
    # Change to docs directory
    docs_dir = Path(__file__).parent / "docs"
    os.chdir(docs_dir)
    
    print("📁 Working directory:", os.getcwd())
    print("📄 Files in directory:")
    for file in os.listdir('.'):
        if file.endswith(('.html', '.css', '.js')):
            print(f"   ✅ {file}")
    
    print("\n🚀 Starting web server...")
    
    # Start HTTP server
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "http.server", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(2)
        
        print("✅ Web server started successfully!")
        print("🌐 Website URL: http://localhost:8000")
        print("📊 Interactive Backtester: http://localhost:8000#backtester")
        print("\n💡 The website will open in your browser...")
        print("🛑 Press Ctrl+C to stop the server")
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8000")
        except:
            print("⚠️  Could not open browser automatically")
            print("   Please open http://localhost:8000 in your browser")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            process.terminate()
            print("✅ Server stopped")
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
