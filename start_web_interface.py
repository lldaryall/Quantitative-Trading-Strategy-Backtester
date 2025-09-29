#!/usr/bin/env python3
"""
Startup script for QBacktester Web Interface
Starts both the backend API and serves the frontend
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import flask_cors
        import matplotlib
        import seaborn
        import pandas
        import numpy
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements-web.txt")
        return False

def start_backend():
    """Start the Flask backend API"""
    print("🚀 Starting QBacktester Backend API...")
    backend_process = subprocess.Popen([
        sys.executable, "backend.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for the backend to start
    time.sleep(3)
    
    # Check if backend is running
    try:
        import requests
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend API is running on http://localhost:5000")
            return backend_process
        else:
            print("❌ Backend API failed to start")
            return None
    except:
        print("⚠️  Backend API may not be ready yet, continuing...")
        return backend_process

def start_frontend():
    """Start the frontend web server"""
    print("🌐 Starting Frontend Web Server...")
    
    # Change to docs directory
    docs_dir = Path(__file__).parent / "docs"
    os.chdir(docs_dir)
    
    # Start HTTP server
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "http.server", "8000"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(2)
    print("✅ Frontend is running on http://localhost:8000")
    return frontend_process

def main():
    """Main function to start the web interface"""
    print("🎯 QBacktester Web Interface Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend")
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend")
        backend_process.terminate()
        sys.exit(1)
    
    print("\n🎉 Web Interface is ready!")
    print("=" * 50)
    print("📊 Backend API: http://localhost:5000")
    print("🌐 Frontend:    http://localhost:8000")
    print("📖 API Docs:    http://localhost:5000/api/health")
    print("\n💡 The web interface will open in your browser...")
    print("🛑 Press Ctrl+C to stop both servers")
    
    # Open browser
    try:
        webbrowser.open("http://localhost:8000")
    except:
        print("⚠️  Could not open browser automatically")
        print("   Please open http://localhost:8000 in your browser")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✅ Servers stopped")

if __name__ == "__main__":
    main()
