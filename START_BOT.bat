@echo off
title KuCoin Multiscanner Papertrader
echo ============================================
echo   KuCoin Multiscanner Papertrader
echo ============================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting Paper Trader...
echo Press Ctrl+C to stop
echo.

python main.py

pause
