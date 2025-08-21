@REM # run.bat (Windows startup script)
@echo off
echo Starting Cognitive Architecture Orchestrator...
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install/upgrade dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Check if .env file exists
if not exist ".env" (
    echo.
    echo Warning: .env file not found
    echo Please copy .env.example to .env and configure your settings
    echo.
    pause
)

:: Run the startup script
echo.
echo Starting application...
python startup.py

pause