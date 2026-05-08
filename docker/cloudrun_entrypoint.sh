#!/bin/bash
# Cloud Run entrypoint for Kestrel Agent
# Bootstraps identity on first start, then runs uvicorn

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

# Single-agent Cloud Run: bootstrap DID + database in KESTREL_DB_PATH
# directly. kestrel_sovereign/server.py's lifespan calls
# get_agent_did_async(KESTREL_DB_PATH) and opens
# "${KESTREL_DB_PATH}/kestrel_prime.db", so the entrypoint must
# write identity in the same dir.
#
# (Earlier this script used a "${KESTREL_DB_PATH}/${KESTREL_AGENT_NAME}"
# subdir to align with the multi_agent image's auto-discovery layout —
# but cloudrun runs single-agent, so the subdir was vestigial and put the
# DID one level deeper than the lifespan reads. Reproduced + fixed in #1029.)
AGENT_DIR="${KESTREL_DB_PATH:-/app/agent_data}"
PORT="${PORT:-8080}"

# Bootstrap agent identity if none exists
mkdir -p "$AGENT_DIR"
if ! ls "$AGENT_DIR"/kestrel_*.json &>/dev/null; then
    echo "No agent identity found. Creating new Kestrel agent in ${AGENT_DIR}..."
    /app/.venv/bin/python -c "
import sys; sys.path.insert(0, '/app')
from kestrel_sovereign.inception_service import create_kestrel_identity
creds = create_kestrel_identity('$AGENT_DIR', 'docs/principles/KESTREL_CONSTITUTION.md')
print(f'Agent created: {creds.agent_did}')
"
else
    echo "Agent identity found in $AGENT_DIR"
fi

# Start the server
exec /app/.venv/bin/uvicorn kestrel_sovereign.server:app --host 0.0.0.0 --port "$PORT"
