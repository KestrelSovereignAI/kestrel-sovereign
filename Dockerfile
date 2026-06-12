# Dockerfile for Kestrel Agent - Standalone with Local LLM
# Includes Ollama for local LLM inference - larger image
ARG TARGETARCH
FROM --platform=linux/${TARGETARCH} python:3.11-slim

# Set environment variables
# NOTE: KESTREL_DB_PATH semantics (config.py treats it as a DIRECTORY, but the
# entrypoint/init script use /app/kestrel.db as a file) are inconsistent across
# Dockerfile / docker_entrypoint.sh / scripts/init_agent_identity.py / compose;
# reconciling them needs an end-to-end container test and is tracked separately.
ENV KESTREL_DB_PATH=/app/kestrel.db
ENV KESTREL_IDENTITY_PATH=/app/kestrel_did.json
ENV KESTREL_ENV=production
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=0.0.0.0:11434

WORKDIR /app

# Install system dependencies including Ollama requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install uv
RUN pip install uv

# Copy application source code
COPY . .

# Make startup script executable
RUN chmod +x /app/docker_entrypoint.sh

# Create the virtual environment
RUN uv venv /app/.venv

# Install the project
RUN uv pip install --python /app/.venv/bin/python .

# Create directory for agent data
RUN mkdir -p /app/agent_data

# Create Ollama startup script
RUN echo '#!/bin/bash\n\
# Start Ollama in background\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama to be ready\n\
sleep 5\n\
\n\
# Pull default model (can be overridden)\n\
ollama pull llama2 || echo "Model pull failed, will use available models"\n\
\n\
# Start the Kestrel agent\n\
/app/docker_entrypoint.sh\n\
\n\
# Cleanup\n\
kill $OLLAMA_PID' > /app/start_with_ollama.sh && chmod +x /app/start_with_ollama.sh

# Expose ports (8888 for Kestrel, 11434 for Ollama)
EXPOSE 8888 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Run with Ollama
CMD ["/app/start_with_ollama.sh"]