#!/bin/bash
# stop_bot.sh - Stop alle trading bot componenten

echo "=========================================="
echo "DERI Trading Bot Stoppen"
echo "=========================================="

# Stop scanner
if [ -f .scanner.pid ]; then
    PID=$(cat .scanner.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Scanner gestopt (PID: $PID)"
    fi
    rm .scanner.pid
fi

# Stop watcher
if [ -f .watcher.pid ]; then
    PID=$(cat .watcher.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Watcher gestopt (PID: $PID)"
    fi
    rm .watcher.pid
fi

# Stop executor
if [ -f .executor.pid ]; then
    PID=$(cat .executor.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Executor gestopt (PID: $PID)"
    fi
    rm .executor.pid
fi

# Stop monitor
if [ -f .monitor.pid ]; then
    PID=$(cat .monitor.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Monitor gestopt (PID: $PID)"
    fi
    rm .monitor.pid
fi

echo ""
echo "Alle componenten gestopt!"
echo "=========================================="
