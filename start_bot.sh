#!/bin/bash
# start_bot.sh - Start alle trading bot componenten

echo "=========================================="
echo "DERI Trading Bot Starter"
echo "=========================================="

# Controleer of Python beschikbaar is
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 niet gevonden!"
    exit 1
fi

# Maak logs directory
mkdir -p logs

# Initialiseer database
echo "Database initialiseren..."
python3 init_db.py

# Start componenten in achtergrond
echo ""
echo "Starten van componenten..."
echo ""

# Scanner
echo "1. Scanner starten..."
python3 scanner_htf.py > logs/scanner.out 2>&1 &
SCANNER_PID=$!
echo "   Scanner PID: $SCANNER_PID"

# Watcher
echo "2. Watcher starten..."
python3 watcher_ws.py > logs/watcher.out 2>&1 &
WATCHER_PID=$!
echo "   Watcher PID: $WATCHER_PID"

# Executor
echo "3. Executor starten..."
python3 executor_ltf.py > logs/executor.out 2>&1 &
EXECUTOR_PID=$!
echo "   Executor PID: $EXECUTOR_PID"

# Trade Monitor
echo "4. Trade Monitor starten..."
python3 trade_monitor.py > logs/monitor.out 2>&1 &
MONITOR_PID=$!
echo "   Monitor PID: $MONITOR_PID"

# Sla PIDs op voor stop script
echo "$SCANNER_PID" > .scanner.pid
echo "$WATCHER_PID" > .watcher.pid
echo "$EXECUTOR_PID" > .executor.pid
echo "$MONITOR_PID" > .monitor.pid

echo ""
echo "=========================================="
echo "Alle componenten gestart!"
echo ""
echo "Dashboard starten met: python3 dashboard.py"
echo "Status bekijken met:   python3 check_status.py"
echo "Stats bekijken met:    python3 view_stats.py"
echo "Stoppen met:           ./stop_bot.sh"
echo ""
echo "Logs in: ./logs/"
echo "=========================================="
