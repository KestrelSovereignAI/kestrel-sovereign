#!/bin/bash
# Cloud Run entrypoint for Kestrel Agent
# Restores durable identity or bootstraps an explicit demo, then runs uvicorn

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

# Single-agent Cloud Run keeps key custody under KESTREL_DB_PATH. Durable mode
# reads DID/state from PostgreSQL; demo mode keeps its disposable SQLite there.
#
# (Earlier this script used a "${KESTREL_DB_PATH}/${KESTREL_AGENT_NAME}"
# subdir to align with the multi_agent image's auto-discovery layout —
# but cloudrun runs single-agent, so the subdir was vestigial and put the
# DID one level deeper than the lifespan reads. Reproduced + fixed in #1029.)
AGENT_DIR="${KESTREL_DB_PATH:-/app/agent_data}"
PORT="${PORT:-8080}"
PERSISTENCE_MODE="${KESTREL_DEPLOYMENT_PERSISTENCE:-}"

case "$PERSISTENCE_MODE" in
    durable_sovereign)
        if [ "${KESTREL_DB_BACKEND:-}" != "postgres" ]; then
            echo "FATAL: durable_sovereign requires PostgreSQL; refusing disposable local state." >&2
            exit 1
        fi
        if [ -z "${KESTREL_DATABASE_URL:-}" ] || [ -z "${KESTREL_HOLD_EVIDENCE_DATABASE_URL:-}" ] || [ -z "${KESTREL_DATA_KEY:-}" ] || [ -z "${KESTREL_EXPECTED_DID:-}" ] || [ -z "${KESTREL_IDENTITY_BUNDLE:-}" ]; then
            echo "FATAL: durable sovereign custody or database binding is unavailable; refusing to re-incept." >&2
            exit 1
        fi
        /app/.venv/bin/python -m kestrel_sovereign.identity.custody_bundle \
            restore-env --agent-dir "$AGENT_DIR" --expected-did "$KESTREL_EXPECTED_DID"
        # Secret Manager injected the bundle only for bootstrap. The serving
        # process retains the data-key/DB credentials it needs, but never the
        # portable identity package.
        unset KESTREL_IDENTITY_BUNDLE
        ;;
    ephemeral_demo)
        if [ "${KESTREL_ENV:-}" = "production" ] || [ "${KESTREL_ENV:-}" = "prod" ]; then
            echo "FATAL: ephemeral_demo identity cannot run as production." >&2
            exit 1
        fi
        mkdir -p "$AGENT_DIR"
        if ! ls "$AGENT_DIR"/kestrel_*.json &>/dev/null && ! ls "$AGENT_DIR"/*_did.json &>/dev/null; then
            echo "No demo identity found. Creating a disposable test/demo identity..."
            export KESTREL_BOOTSTRAP_AGENT_DIR="$AGENT_DIR"
            /app/.venv/bin/python - <<'PY'
import os
from kestrel_sovereign.inception_service import create_kestrel_identity

creds = create_kestrel_identity(
    os.environ["KESTREL_BOOTSTRAP_AGENT_DIR"],
    is_test_instance=True,
    is_demo=True,
)
print(f"Disposable demo agent created: {creds.agent_did}")
PY
            unset KESTREL_BOOTSTRAP_AGENT_DIR
        else
            echo "Disposable demo identity found in $AGENT_DIR"
        fi
        ;;
    *)
        echo "FATAL: set KESTREL_DEPLOYMENT_PERSISTENCE to durable_sovereign or ephemeral_demo." >&2
        exit 1
        ;;
esac

# Start the server
exec /app/.venv/bin/uvicorn kestrel_sovereign.server:app --host 0.0.0.0 --port "$PORT"
