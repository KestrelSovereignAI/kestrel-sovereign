#!/bin/bash
# Rookery Host entrypoint for Cloud Run / Azure Container Apps
# 1. Generates rookery.toml via auto-discovery if not mounted
# 2. Bootstraps agent identities for any new agent data dirs
# 3. Starts the rookery host (host.py)

set -e

PORT="${PORT:-8080}"
ROOKERY_CONFIG="${KESTREL_ROOKERY_CONFIG:-/app/rookery.toml}"
AGENT_DATA_DIR="${KESTREL_AGENT_DATA_DIR:-/app/agent_data}"

# Generate rookery.toml from auto-discovery if not already present
if [ ! -f "$ROOKERY_CONFIG" ]; then
    echo "No rookery.toml found. Auto-discovering agents in $AGENT_DATA_DIR..."
    /app/.venv/bin/python -c "
import os
from kestrel_sovereign.rookery.config import RookeryConfig

config = RookeryConfig.auto_discover('$AGENT_DATA_DIR')
config.host.port = int(os.environ.get('PORT', '8080'))
config.save('$ROOKERY_CONFIG')
print(f'Generated rookery config: {len(config.agents)} agents on port {config.host.port}')
for name in config.agents:
    print(f'  - {name}')
"
else
    echo "Using existing rookery config: $ROOKERY_CONFIG"
fi

# Bootstrap identity for each agent data dir that lacks one
for dir in "$AGENT_DATA_DIR"/*/; do
    [ -d "$dir" ] || continue
    if ! ls "$dir"/kestrel_*.json &>/dev/null; then
        agent_name=$(basename "$dir")
        echo "Bootstrapping identity for agent '$agent_name'..."
        /app/.venv/bin/python -c "
from kestrel_sovereign.inception_service import create_kestrel_identity
creds = create_kestrel_identity('$dir', 'docs/principles/KESTREL_CONSTITUTION.md')
print(f'  Created: {creds.agent_did}')
"
    else
        agent_name=$(basename "$dir")
        echo "Agent '$agent_name' identity exists."
    fi
done

echo "Starting Kestrel Rookery Host on port $PORT..."
exec /app/.venv/bin/uvicorn host:app --host 0.0.0.0 --port "$PORT"
