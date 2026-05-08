#!/bin/bash
# Startup script for Kestrel Agent container
# Runs the web server in background and provides terminal chat interface

# Check if agent database exists, if not create one
if [ ! -f "/app/kestrel.db" ]; then
    echo "No agent database found. Creating new Kestrel agent..."
    /app/.venv/bin/python /app/scripts/init_agent_identity.py
fi

# Start the FastAPI server in background
/app/.venv/bin/uvicorn kestrel_sovereign.server:app --host 0.0.0.0 --port 8888 &
SERVER_PID=$!

# Wait for server to start and initialize
echo "Waiting for server to initialize..."
sleep 10

# Check if we're running interactively
if [ -t 0 ]; then
    # Interactive mode - run chat interface
    echo "Kestrel Agent started!"
    echo "Web UI available at http://localhost:8888"
    echo "Terminal chat interface active. Type your messages below."
    echo "Type '!quit' to exit."

    # Set database path for chat
    export KESTREL_DB_PATH=/app/kestrel.db

    # Run the terminal chat interface
    /app/.venv/bin/python -m kestrel_sovereign.main

    # When chat exits, kill the server
    kill $SERVER_PID
else
    # Non-interactive mode - just run server
    echo "Kestrel Agent server started in background"
    echo "Web UI available at http://localhost:8888"
    echo "Health check: http://localhost:8888/health"

    # Wait for server process
    wait $SERVER_PID
fi