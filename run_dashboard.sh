#!/bin/bash

echo "Starting Football Data Dashboard..."
echo ""
echo "Make sure your database is configured in .streamlit/secrets.toml"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Installing Streamlit and dependencies..."
    pip3 install -r requirements.txt
fi

# Check if secrets.toml exists
if [ ! -f .streamlit/secrets.toml ]; then
    echo "Warning: .streamlit/secrets.toml not found"
    echo "Please copy .streamlit/secrets.toml.example to .streamlit/secrets.toml"
    echo "and configure your database connection"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

echo "Starting dashboard..."
echo "Dashboard will open in your default browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"

# Make the script executable
chmod +x "$0"

streamlit run dashboard.py