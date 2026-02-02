#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.kestrel.pid"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/kestrel.log"
PORT=8888

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check if server is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "❌ Kestrel server already running (PID: $PID)"
        echo "   Use ./stop_kestrel.sh to stop it first"
        exit 1
    else
        echo "⚠️  Stale PID file found, removing..."
        rm "$PID_FILE"
    fi
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "❌ Port $PORT is already in use"
    echo "   Run: lsof -i :$PORT to see what's using it"
    exit 1
fi

echo "🦅 Starting Kestrel Sovereign Agent..."
echo "   Port: $PORT"
echo "   Logs: $LOG_FILE"
echo "   PID file: $PID_FILE"
echo ""

cd "$SCRIPT_DIR"

# Source .env file if it exists
if [ -f ".env" ]; then
    echo "   Loading: .env"
    set -a
    source ".env"
    set +a
fi

# Start server
uv run uvicorn server:app --host 0.0.0.0 --port $PORT > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Save PID
echo "$SERVER_PID" > "$PID_FILE"

# Wait for server to start
echo "   Waiting for server to start..."
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo ""
        echo "✅ Kestrel server started successfully!"
        echo ""
        echo "   URL: http://localhost:$PORT"
        echo "   API Docs: http://localhost:$PORT/docs"
        echo "   Logs: tail -f $LOG_FILE"
        echo ""
        exit 0
    fi
    sleep 0.5
done

echo "⚠️  Server may have failed to start. Check logs:"
echo "   tail -f $LOG_FILE"
