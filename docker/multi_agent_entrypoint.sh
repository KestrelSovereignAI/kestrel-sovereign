#!/bin/bash
# MultiAgent Host entrypoint for Cloud Run / Azure Container Apps
# 1. Provisions agent directories from KESTREL_AGENTS env var (if set)
# 2. Generates multi_agent.toml via auto-discovery if not mounted
# 3. Bootstraps agent identities for any new agent data dirs
# 4. Starts the multi_agent host (host.py)

set -e

# Strip surrounding quotes from all env vars.
# Docker's --env-file includes quotes literally, breaking API keys and secrets.
# python-dotenv strips them natively, but Docker does not.
while IFS='=' read -r key val; do
    if [[ "$val" == \"*\" && "$val" == *\" ]]; then
        export "$key"="${val:1:-1}"
    elif [[ "$val" == \'*\' && "$val" == *\' ]]; then
        export "$key"="${val:1:-1}"
    fi
done < <(env)

PORT="${PORT:-8080}"
MULTI_AGENT_CONFIG="${KESTREL_MULTI_AGENT_CONFIG:-/app/multi_agent.toml}"
AGENT_DATA_DIR="${KESTREL_AGENT_DATA_DIR:-/app/agent_data}"

# Provision agent directories from KESTREL_AGENTS env var
# Format: comma-separated agent names, e.g. "claw,testbot,emma"
if [ -n "$KESTREL_AGENTS" ]; then
    echo "Provisioning agents from KESTREL_AGENTS: $KESTREL_AGENTS"
    IFS=',' read -ra AGENTS <<< "$KESTREL_AGENTS"
    for agent in "${AGENTS[@]}"; do
        agent=$(echo "$agent" | xargs)  # trim whitespace
        agent_dir="$AGENT_DATA_DIR/$agent"
        if [ ! -d "$agent_dir" ]; then
            echo "  Creating agent directory: $agent_dir"
            mkdir -p "$agent_dir"
        fi
    done
fi

# Generate multi_agent.toml from auto-discovery if not already present
if [ ! -f "$MULTI_AGENT_CONFIG" ]; then
    echo "No multi_agent.toml found. Auto-discovering agents in $AGENT_DATA_DIR..."
    /app/.venv/bin/python -c "
import os
from kestrel_sovereign.multi_agent.config import MultiAgentConfig

config = MultiAgentConfig.auto_discover('$AGENT_DATA_DIR', include_empty=True)
config.host.port = int(os.environ.get('PORT', '8080'))
config.save('$MULTI_AGENT_CONFIG')
print(f'Generated multi_agent config: {len(config.agents)} agents on port {config.host.port}')
for name in config.agents:
    print(f'  - {name}')
"
else
    echo "Using existing multi_agent config: $MULTI_AGENT_CONFIG"
fi

# Bootstrap identity and initialize DB for each agent data dir
for dir in "$AGENT_DATA_DIR"/*/; do
    [ -d "$dir" ] || continue
    agent_name=$(basename "$dir")

    # Bootstrap identity if missing
    if ! ls "$dir"/kestrel_*.json &>/dev/null; then
        echo "Bootstrapping identity for agent '$agent_name'..."
        /app/.venv/bin/python -c "
from kestrel_sovereign.inception_service import create_kestrel_identity
creds = create_kestrel_identity('$dir', 'docs/principles/KESTREL_CONSTITUTION.md')
print(f'  Created: {creds.agent_did}')
"
    else
        echo "Agent '$agent_name' identity exists."
    fi

    # Initialize DB if missing (required by ProcessManager validation)
    if [ ! -f "$dir/kestrel_prime.db" ]; then
        echo "Initializing database for agent '$agent_name'..."
        /app/.venv/bin/python -c "
from kestrel_sovereign.storage.async_storage import AsyncStorage
import asyncio

async def init():
    storage = AsyncStorage(db_path='$dir/kestrel_prime.db')
    await storage.initialize()
    await storage.close()

asyncio.run(init())
print('  Database initialized.')
"
    fi
done

echo "Starting Kestrel MultiAgent Host on port $PORT..."
exec /app/.venv/bin/uvicorn host:app --host 0.0.0.0 --port "$PORT"
