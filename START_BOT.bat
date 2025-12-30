@echo off
TITLE SMC Bot Control Center

echo ========================================
echo   LAUNCHING DERI MORGAN SMC TRADING BOT
echo ========================================
echo.

:: Stap 1: Database initialiseren (optioneel, maar veilig)
echo [1/4] Database check...
python init_db.py
timeout /t 2 /nobreak >nul

:: Stap 2: Start de HTF Scanner (Brein)
echo [2/4] Starting HTF Scanner (4H Analysis)...
start "HTF SCANNER" cmd /k python scanner_htf.py

:: Stap 3: Start de WebSocket Watcher (Sensor)
echo [3/4] Starting Real-time Watcher (WebSocket)...
start "WS WATCHER" cmd /k python watcher_ws.py

:: Stap 4: Start de LTF Executor (Actie)
echo [4/4] Starting LTF Executor (M15 ChoCH)...
start "LTF EXECUTOR" cmd /k python executor_ltf.py

echo.
echo ========================================
echo   BOT IS NU ACTIEF - VOLG DE LOGS
echo ========================================
echo.
echo Gebruik 'python view_stats.py' in een nieuwe terminal 
echo om je trade logboek te bekijken.
echo.
pause