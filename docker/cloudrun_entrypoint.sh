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

AGENT_DIR="${KESTREL_DB_PATH:-/app/agent_data}"
PORT="${PORT:-8080}"

# Bootstrap agent identity if none exists
if ! ls "$AGENT_DIR"/kestrel_*.json &>/dev/null; then
    echo "No agent identity found. Creating new Kestrel agent..."
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
exec /app/.venv/bin/uvicorn server:app --host 0.0.0.0 --port "$PORT"
