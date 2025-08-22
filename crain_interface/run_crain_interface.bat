@echo off
echo Starting Crain Natural Language Interface...
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
@REM if not exist "venv" (
@REM     echo Creating virtual environment...
@REM     python -m venv venv
@REM )

@REM :: Activate virtual environment
@REM call venv\Scripts\activate.bat

:: Install dependencies
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

:: Start the application
echo.
echo Starting Crain AI Interface...
echo Application will be available at: http://localhost:8081
echo.
python app.py

pause