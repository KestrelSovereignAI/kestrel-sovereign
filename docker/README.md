# Kestrel Agent Docker Images

This directory contains Docker configurations for deploying Kestrel agents in different environments.

## Current Status ✅

**Mac Silicon Test Results:**
- ✅ Container builds successfully
- ✅ Agent creation works (generates DID and database)
- ✅ Web server starts on port 8888
- ✅ Web UI loads at http://localhost:8888
- ✅ Health endpoint responds
- ⚠️ Agent initialization has MCP Docker dependency issue (non-critical for basic chat)

**Known Limitations:**
- Remote version cannot use MCP tools due to Docker-in-Docker requirement
- For full MCP functionality, use standalone version with proper Docker socket mounting

## Available Images

### 1. Remote LLM (`Dockerfile.remote`) - Recommended for Mac Silicon
- **Size**: ~500MB (smallest)
- **Use Case**: Cloud deployments, external LLM APIs, Mac Silicon development
- **Features**:
  - Uses OpenAI/Anthropic APIs for LLM inference
  - Fast startup, minimal dependencies
  - No local LLM installation required
  - Perfect for development and testing on Mac

### 2. Standalone (`Dockerfile.standalone`)
- **Size**: ~1.5GB (medium)
- **Use Case**: Air-gapped environments, local inference
- **Features**:
  - Includes Ollama for local LLM serving
  - Self-contained LLM inference
  - Auto-pulls Llama2 model on startup
  - Best for offline deployments

### 3. GPU-Enabled (`Dockerfile.gpu`)
- **Size**: ~3GB+ (largest)
- **Use Case**: High-performance local inference with NVIDIA GPUs
- **Features**:
  - CUDA 11.8 runtime for GPU acceleration
  - GPU-accelerated Ollama inference
  - Optimized for larger models (13B+)
  - Requires NVIDIA GPU and drivers

## Quick Start

### For Mac Silicon (Recommended)
```bash
# Build the remote version
docker build -f docker/Dockerfile.remote -t kestrel-agent .

# Run with your API keys
docker run -p 8888:8888 \
  -e OPENAI_API_KEY=your_openai_key \
  -e ANTHROPIC_API_KEY=your_anthropic_key \
  kestrel-agent
```

#### Mac Silicon with Local Ollama (For Private Conversations)
For optimal performance on Apple Silicon with local LLM inference:

```bash
# 1. Install and run Ollama natively on your Mac
brew install ollama
ollama serve  # Runs on localhost:11434

# 2. In another terminal, run Kestrel container
docker run -p 8888:8888 \
  -e OLLAMA_HOST=host.docker.internal:11434 \
  -e KESTREL_LLM_PROVIDER=ollama \
  kestrel-agent
```

This gives you the best of both worlds: native Ollama performance on Apple Silicon + containerized Kestrel agent.

### For Standalone Deployment
```bash
# Build standalone version
docker build -f docker/Dockerfile.standalone -t kestrel-standalone .

# Run (includes Ollama)
docker run -p 8888:8888 -p 11434:11434 kestrel-standalone
```

### For GPU Deployment
```bash
# Build GPU version
docker build -f docker/Dockerfile.gpu -t kestrel-gpu .

# Run with GPU access
docker run --gpus all -p 8888:8888 -p 11434:11434 kestrel-gpu
```

## Accessing the Agent

Once running, access the Kestrel agent at:
- **Web UI**: http://localhost:8888
- **Health Check**: http://localhost:8888/health
- **Agent Info**: http://localhost:8888/agent/info

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KESTREL_DB_PATH` | Agent data directory containing `kestrel_prime.db` | `/app/agent_data` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OLLAMA_HOST` | Ollama server address | `0.0.0.0:11434` |

## Features

All images include:
- ✅ Individual agent UI (self-hosted agent UI)
- ✅ Terminal chat interface
- ✅ Web-based chat interface
- ✅ Privacy mode controls
- ✅ Agent-specific branding
- ✅ Health checks and monitoring

## Development

To modify Dockerfiles:
```bash
# Edit the appropriate Dockerfile
vim docker/Dockerfile.remote

# Rebuild
docker build -f docker/Dockerfile.remote -t kestrel-dev .
```

## Features

All images include:
- ✅ Individual agent UI (Kestrel AI companion interface)
- ✅ Agent-specific branding and messaging
- ✅ Terminal chat interface (when run interactively)
- ✅ Web-based chat interface
- ✅ Privacy mode controls (Ephemeral, Isolated, Anonymous, Normal, Public)
- ✅ Health checks and monitoring
- ✅ Automatic agent creation on first run

## Troubleshooting

### Container exits immediately
Check logs: `docker logs <container_id>`
Common issues:
- Missing API keys for remote version
- Database path issues
- Port conflicts (change host port mapping)

### Web UI not loading
- Ensure port 8888 is available
- Check container is running: `docker ps`
- Verify health endpoint: `curl http://localhost:8888/health`

### LLM not responding
- For remote: Check API keys are set correctly
- For standalone: Wait for Ollama to download models (first run takes longer)
- Check Ollama logs if using standalone/GPU versions
