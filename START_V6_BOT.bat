@echo off
TITLE V6 Final Sniper Bot

echo ========================================
echo   LAUNCHING V6 FINAL SNIPER BOT
echo ========================================
echo.

echo [1/3] Database check...
python init_db.py
timeout /t 2 /nobreak >nul

echo [2/3] Starting V6 Scanner (4H Analysis)...
start "V6 SCANNER" cmd /k python scanner_v6.py

echo [3/3] Starting V6 Executor...
start "V6 EXECUTOR" cmd /k python executor_v6.py

echo.
echo ========================================
echo   V6 BOT IS ACTIVE
echo ========================================
echo.
pause