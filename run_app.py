#!/usr/bin/env python3
"""
Quick start script for the Crop Yield Prediction Streamlit app.
This script checks dependencies and launches the Streamlit application.

Usage:
    python run_app.py
"""

import sys
import subprocess
import os

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'scikit-learn',
        'openpyxl',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                __import__('sklearn')  # scikit-learn imports as sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install requirements from requirements.txt."""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements.")
        return False

def check_data_file():
    """Check if the data file exists."""
    data_file = "crop yield data sheet.xlsx"
    if os.path.exists(data_file):
        print(f"✅ Data file found: {data_file}")
        return True
    else:
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the Excel file is in the current directory.")
        return False

def launch_streamlit():
    """Launch the Streamlit application."""
    try:
        print("🚀 Launching Streamlit app...")
        print("📱 The app will open in your default web browser.")
        print("🌐 URL: http://localhost:8501")
        print("\nPress Ctrl+C to stop the app.\n")
        
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'])
        
    except KeyboardInterrupt:
        print("\n👋 App stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        return False
    
    return True

def main():
    """Main function to run the app."""
    print("=" * 60)
    print("🌾 CROP YIELD PREDICTION - STREAMLIT APP")
    print("=" * 60)
    
    # Check if data file exists
    if not check_data_file():
        return 1
    
    # Check requirements
    missing = check_requirements()
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("📦 Attempting to install requirements...")
        
        if not install_requirements():
            print("💡 Try running manually: pip install -r requirements.txt")
            return 1
        
        # Re-check after installation
        missing = check_requirements()
        if missing:
            print(f"❌ Still missing packages: {', '.join(missing)}")
            return 1
    
    print("✅ All requirements satisfied!")
    
    # Launch the app
    if not launch_streamlit():
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
