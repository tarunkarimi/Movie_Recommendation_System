"""
🎬 Movie Recommendation System - Streamlit Launcher

Easy launcher for the Streamlit web application.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit_installed():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit if not available."""
    print("Installing Streamlit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    print("✅ Streamlit installed successfully!")

def launch_app():
    """Launch the Streamlit application."""
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        print("❌ Error: streamlit_app.py not found!")
        return
    
    print("🚀 Launching Movie Recommendation System...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⚠️  Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.headless", "false",
            "--server.port", "8504",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down the application...")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

def main():
    """Main launcher function."""
    print("🎬 Movie Recommendation System - Streamlit Launcher")
    print("=" * 50)
    
    # Check if Streamlit is installed
    if not check_streamlit_installed():
        install_streamlit()
    
    # Launch the app
    launch_app()

if __name__ == "__main__":
    main()
