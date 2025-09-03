@echo off
echo FormFinder Notifier System
echo ==========================
echo.
echo Choose an option:
echo 1. Run notifier (test mode)
echo 2. Run tests
echo 3. Setup assistant
echo 4. View logs
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Running notifier in test mode...
    python -m formfinder.notifier --test-mode
) else if "%choice%"=="2" (
    echo Running tests...
    python test_notifier.py
) else if "%choice%"=="3" (
    echo Starting setup assistant...
    python setup_notifier.py
) else if "%choice%"=="4" (
    echo Opening latest log file...
    if exist logs\notifier.log (
        type logs\notifier.log
    ) else (
        echo No log file found. Run the notifier first.
    )
    pause
) else if "%choice%"=="5" (
    echo Exiting...
) else (
    echo Invalid choice. Please run again.
    pause
)