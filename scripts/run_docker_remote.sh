#!/bin/bash
# Run Kestrel Agent in Docker (Remote LLM mode)
# Reads configuration from .env file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

# Container configuration
CONTAINER_NAME="kestrel-remote"
IMAGE_NAME="kestrel-remote:latest"
HOST_PORT="${KESTREL_PORT:-8888}"
CONTAINER_PORT=8888

# Check .env exists
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

# Load environment variables from .env
set -a
source "$ENV_FILE"
set +a

# Validate required variables
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "Error: OPENAI_API_KEY not set in .env"
    exit 1
fi

# Stop and remove existing container
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting $CONTAINER_NAME on port $HOST_PORT..."

# Detect host for Ollama access from inside Docker
# On Docker Desktop (Mac/Windows), use host.docker.internal
# On Linux, use the host gateway
if [[ "$(uname)" == "Darwin" ]]; then
    OLLAMA_HOST="http://host.docker.internal:11434"
else
    OLLAMA_HOST="http://172.17.0.1:11434"
fi

# Run container with environment variables from .env
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$HOST_PORT:$CONTAINER_PORT" \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e KESTREL_API_KEY="${KESTREL_API_KEY:-}" \
    -e KESTREL_DATA_KEY="${KESTREL_DATA_KEY:-}" \
    -e REPLICATE_API_TOKEN="${REPLICATE_API_TOKEN:-}" \
    -e TAVILY_API_KEY="${TAVILY_API_KEY:-}" \
    -e RUNPOD_API_KEY="${RUNPOD_API_KEY:-}" \
    -e XAI_API_KEY="${XAI_API_KEY:-}" \
    -e OLLAMA_HOST="$OLLAMA_HOST" \
    "$IMAGE_NAME"

echo "Waiting for server to start..."
sleep 10

# Health check
if curl -sf "http://localhost:$HOST_PORT/health" > /dev/null; then
    echo "✅ Kestrel Agent running at http://localhost:$HOST_PORT"
    echo "   Health: http://localhost:$HOST_PORT/health"
    echo "   API Docs: http://localhost:$HOST_PORT/docs"
    
    # Show API key for testing (if generated)
    if [[ -z "$KESTREL_API_KEY" ]]; then
        echo ""
        echo "⚠️  No KESTREL_API_KEY set - server generated one."
        echo "   Check logs: docker logs $CONTAINER_NAME | grep -i key"
    fi
else
    echo "❌ Health check failed"
    echo "Logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -20
    exit 1
fi
