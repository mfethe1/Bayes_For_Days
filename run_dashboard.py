#!/usr/bin/env python3
"""
Launch script for Bayes For Days web dashboard.

This script starts the Streamlit web application for the Bayes For Days
platform, providing an interactive interface for optimization campaigns.

Usage:
    python run_dashboard.py
    
    or
    
    streamlit run src/bayes_for_days/dashboard/app.py
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Launch the Bayes For Days dashboard."""
    
    # Get the path to the dashboard app
    dashboard_path = Path(__file__).parent / "src" / "bayes_for_days" / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard app not found at {dashboard_path}")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Launch streamlit
    print("🚀 Launching Bayes For Days Dashboard...")
    print(f"📍 Dashboard will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
