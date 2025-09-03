@echo off
echo Starting Football Data Dashboard...
echo.
echo Make sure your database is configured in .streamlit/secrets.toml
echo.

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Streamlit and dependencies...
    pip install -r requirements.txt
)

:: Check if secrets.toml exists
if not exist .streamlit\secrets.toml (
    echo Warning: .streamlit\secrets.toml not found
    echo Please copy .streamlit\secrets.toml.example to .streamlit\secrets.toml
    echo and configure your database connection
    echo.
    pause
)

echo Starting dashboard...
echo Dashboard will open in your default browser at http://localhost:8501
echo Press Ctrl+C to stop the server
streamlit run dashboard.py

pause