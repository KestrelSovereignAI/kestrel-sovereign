#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.kestrel.pid"
PORT=8888

if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  No PID file found - server may not be running"
    echo "   Checking for orphaned processes on port $PORT..."

    # Check if anything is running on the port
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
        echo "   Found process $PID using port $PORT"
        echo "   Kill it? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            kill "$PID"
            echo "✅ Process killed"
        fi
    else
        echo "   No process found on port $PORT"
    fi
    exit 0
fi

PID=$(cat "$PID_FILE")
echo "🛑 Stopping Kestrel server (PID: $PID)..."

if ps -p "$PID" > /dev/null 2>&1; then
    kill "$PID"

    # Wait for graceful shutdown
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done

    # Force kill if still running
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️  Graceful shutdown failed, forcing..."
        kill -9 "$PID"
    fi

    rm "$PID_FILE"
    echo "✅ Kestrel server stopped"
else
    echo "⚠️  Process $PID not found (already stopped?)"
    rm "$PID_FILE"
fi
