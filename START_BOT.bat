@echo off
TITLE DERI Trading Bot Control Center

echo ========================================
echo   DERI SMC TRADING BOT v2.0
echo ========================================
echo.

:: Maak logs directory
if not exist "logs" mkdir logs

:: Stap 1: Database initialiseren
echo [1/5] Database initialiseren...
python init_db.py
timeout /t 2 /nobreak >nul

:: Stap 2: Start de HTF Scanner
echo [2/5] HTF Scanner starten (4H Analysis)...
start "HTF SCANNER" cmd /k "python scanner_htf.py"

:: Stap 3: Start de WebSocket Watcher
echo [3/5] WebSocket Watcher starten (Real-time)...
start "WS WATCHER" cmd /k "python watcher_ws.py"

:: Stap 4: Start de LTF Executor
echo [4/5] LTF Executor starten (M15 ChoCH)...
start "LTF EXECUTOR" cmd /k "python executor_ltf.py"

:: Stap 5: Start de Trade Monitor
echo [5/5] Trade Monitor starten (TP/SL)...
start "TRADE MONITOR" cmd /k "python trade_monitor.py"

echo.
echo ========================================
echo   BOT IS NU ACTIEF
echo ========================================
echo.
echo Beschikbare commando's:
echo   python check_status.py  - Bekijk huidige status
echo   python view_stats.py    - Bekijk statistieken
echo   python dashboard.py     - Start web dashboard
echo   python backtester.py    - Run backtest
echo.
echo Web Dashboard: http://127.0.0.1:5000
echo.
echo Logs worden opgeslagen in: .\logs\
echo ========================================
echo.
pause
