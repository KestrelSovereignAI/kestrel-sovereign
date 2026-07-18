#!/bin/bash
# MultiAgent Host entrypoint for Cloud Run / Azure Container Apps
# 1. Provisions agent directories from KESTREL_AGENTS env var (if set)
# 2. Generates multi_agent.toml via auto-discovery if not mounted
# 3. Bootstraps agent identities for any new agent data dirs
# 4. Starts the multi_agent host (server:app in multi-agent mode, #2382)

set -euo pipefail

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
PERSISTENCE_MODE="${KESTREL_DEPLOYMENT_PERSISTENCE:-}"

if [ "$PERSISTENCE_MODE" = "durable_sovereign" ]; then
    echo "FATAL: durable multi-agent Cloud Run needs one custody bundle and database binding per agent; refusing local inception." >&2
    exit 1
fi
if [ "$PERSISTENCE_MODE" != "ephemeral_demo" ]; then
    echo "FATAL: multi-agent containers currently support only explicit ephemeral_demo persistence." >&2
    exit 1
fi
if [ "${KESTREL_ENV:-}" = "production" ] || [ "${KESTREL_ENV:-}" = "prod" ]; then
    echo "FATAL: ephemeral_demo identity cannot run as production." >&2
    exit 1
fi

# Provision agent directories from KESTREL_AGENTS env var
# Format: comma-separated agent names, e.g. "claw,testbot,emma"
if [ -n "${KESTREL_AGENTS:-}" ]; then
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
    export KESTREL_BOOTSTRAP_AGENT_DATA_DIR="$AGENT_DATA_DIR"
    export KESTREL_BOOTSTRAP_MULTI_AGENT_CONFIG="$MULTI_AGENT_CONFIG"
    /app/.venv/bin/python - <<'PY'
import os
from kestrel_sovereign.multi_agent.config import MultiAgentConfig

data_dir = os.environ["KESTREL_BOOTSTRAP_AGENT_DATA_DIR"]
config_path = os.environ["KESTREL_BOOTSTRAP_MULTI_AGENT_CONFIG"]
config = MultiAgentConfig.auto_discover(data_dir, include_empty=True)
config.host.port = int(os.environ.get('PORT', '8080'))
config.save(config_path)
print(f'Generated multi_agent config: {len(config.agents)} agents on port {config.host.port}')
for name in config.agents:
    print(f'  - {name}')
PY
    unset KESTREL_BOOTSTRAP_AGENT_DATA_DIR KESTREL_BOOTSTRAP_MULTI_AGENT_CONFIG
else
    echo "Using existing multi_agent config: $MULTI_AGENT_CONFIG"
fi

# Bootstrap identity and initialize DB for each agent data dir
for dir in "$AGENT_DATA_DIR"/*/; do
    [ -d "$dir" ] || continue
    agent_name=$(basename "$dir")

    # Bootstrap identity if missing
    if ! ls "$dir"/kestrel_*.json &>/dev/null && ! ls "$dir"/*_did.json &>/dev/null; then
        echo "Bootstrapping identity for agent '$agent_name'..."
        export KESTREL_BOOTSTRAP_AGENT_DIR="$dir"
        export KESTREL_BOOTSTRAP_AGENT_NAME="$agent_name"
        /app/.venv/bin/python - <<'PY'
import os
from kestrel_sovereign.inception_service import create_kestrel_identity
# Do NOT pass a constitution_path. Inception defaults to the shared
# governing-constitution resolver's authoritative packaged source
# (config.CONSTITUTION_PATH = kestrel_sovereign/data/KESTREL_CONSTITUTION.md)
# — the EXACT bytes the periodic integrity audit later recomputes (#2463).
# Passing the docs copy (OKF-frontmatter-wrapped) would incept a hash the
# audit can never match, self-bricking fresh container agents into Safe Mode.
creds = create_kestrel_identity(
    os.environ["KESTREL_BOOTSTRAP_AGENT_DIR"],
    agent_name=os.environ["KESTREL_BOOTSTRAP_AGENT_NAME"],
    is_test_instance=True,
    is_demo=True,
)
print(f'  Created: {creds.agent_did}')
PY
        unset KESTREL_BOOTSTRAP_AGENT_DIR KESTREL_BOOTSTRAP_AGENT_NAME
    else
        echo "Agent '$agent_name' identity exists."
    fi

    # Initialize DB if missing (required by ProcessManager validation)
    if [ ! -f "$dir/kestrel_prime.db" ]; then
        echo "Initializing database for agent '$agent_name'..."
        export KESTREL_BOOTSTRAP_DB_PATH="$dir/kestrel_prime.db"
        /app/.venv/bin/python - <<'PY'
import asyncio
import os
from kestrel_sovereign.storage.async_storage import AsyncStorage

async def init():
    storage = AsyncStorage(db_path=os.environ["KESTREL_BOOTSTRAP_DB_PATH"])
    await storage.initialize()
    await storage.close()

asyncio.run(init())
print('  Database initialized.')
PY
        unset KESTREL_BOOTSTRAP_DB_PATH
    fi
done

echo "Starting Kestrel MultiAgent Host on port $PORT..."
# Consolidated onto server:app in multi-agent mode (#2382). The legacy
# proxy host (host:app) was retired; server:app co-hosts all agents in one
# process and mounts the host-feature runtime unconditionally.
export KESTREL_MULTI_AGENT=1
export KESTREL_MULTI_AGENT_CONFIG="$MULTI_AGENT_CONFIG"
exec /app/.venv/bin/uvicorn kestrel_sovereign.server:app --host 0.0.0.0 --port "$PORT"
