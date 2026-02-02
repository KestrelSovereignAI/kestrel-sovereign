#!/bin/bash
# Startup script for RunPod Ollama Server
# Starts Ollama and optionally pre-pulls specified models

set -e

echo "=== Starting Kestrel Ollama Server ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"

# Check if we should use a volume for persistent models
if [ -d "/workspace" ]; then
    echo "Using /workspace for model storage"
    export OLLAMA_MODELS="/workspace/ollama_models"
    mkdir -p "$OLLAMA_MODELS"
fi

# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
MAX_RETRIES=30
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: Ollama failed to start after $MAX_RETRIES seconds"
        exit 1
    fi
    sleep 1
done

# Pre-pull models if specified in environment
# Models should be comma-separated: OLLAMA_MODELS_PULL=qwen2.5:7b,llama3.2:3b
if [ -n "$OLLAMA_MODELS_PULL" ]; then
    echo "Pre-pulling models: $OLLAMA_MODELS_PULL"
    IFS=',' read -ra MODELS <<< "$OLLAMA_MODELS_PULL"
    for model in "${MODELS[@]}"; do
        model=$(echo "$model" | xargs)  # Trim whitespace
        echo "Pulling $model..."
        ollama pull "$model" || echo "Warning: Failed to pull $model"
    done
fi

# List available models
echo ""
echo "=== Available Models ==="
ollama list

echo ""
echo "=== Ollama Server Ready ==="
echo "API endpoint: http://0.0.0.0:11434"
echo "Health check: http://0.0.0.0:11434/api/tags"
echo ""

# Wait for Ollama process (keep container running)
wait $OLLAMA_PID
