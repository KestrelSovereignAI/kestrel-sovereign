#!/usr/bin/env bash
# demos/run.sh <demo-name>
#
# Runs a Kestrel demo against an ISOLATED demo agent.
# Never talks to the live server on port 8888.
#
# What it does:
#   1. Creates a fresh agent DB via scripts/setup_demo_agent.py  → agent_data/demo/
#   2. Starts a dedicated uvicorn on DEMO_PORT (default 8889) against that DB
#   3. Waits for /health
#   4. Runs `cd demos/<name> && npx playwright test --config=config.cjs`
#      with KESTREL_URL pointing at the isolated server and KESTREL_API_KEY
#      unset so the demo fetches the demo agent's key via /api/auth/key
#   5. Tears the server down on exit (EXIT trap) regardless of outcome
#
# Usage:
#   demos/run.sh technical
#   demos/run.sh spawn
#   demos/run.sh falconer
#   DEMO_PORT=9001 demos/run.sh technical
#
# Preserves LLM provider keys (ANTHROPIC_API_KEY, OPENROUTER_API_KEY, etc.)
# from .env but strips KESTREL_API_KEY so the production key can't auth
# against the demo DB.

set -euo pipefail

DEMO="${1:-}"
if [ -z "$DEMO" ]; then
  cat >&2 <<EOF
Usage: $0 <demo-name>
  e.g. $0 technical
       $0 spawn
       $0 falconer
EOF
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d "demos/$DEMO" ]; then
  echo "[demo-runner] No demo at demos/$DEMO" >&2
  exit 2
fi
if [ ! -f "demos/$DEMO/config.cjs" ]; then
  echo "[demo-runner] Missing demos/$DEMO/config.cjs" >&2
  exit 2
fi

DEMO_PORT="${DEMO_PORT:-8900}"
DEMO_URL="http://localhost:$DEMO_PORT"
DEMO_DB="$ROOT/agent_data/demo"
SERVER_LOG="/tmp/kestrel-demo-server-$DEMO_PORT.log"

# Safety: if DEMO_PORT is 8888 (the live server), refuse.
if [ "$DEMO_PORT" = "8888" ]; then
  echo "[demo-runner] Refusing to use port 8888 — that's the live server. Pick another port." >&2
  exit 2
fi

# Safety: if something is already listening on DEMO_PORT, refuse.
if lsof -nP -iTCP:"$DEMO_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[demo-runner] Port $DEMO_PORT already in use. Free it or set DEMO_PORT=<free-port>." >&2
  exit 2
fi

# Load LLM provider keys from .env but scrub KESTREL_API_KEY
if [ -f "$ROOT/.env" ]; then
  # shellcheck disable=SC1091
  set -a; source "$ROOT/.env"; set +a
fi
unset KESTREL_API_KEY

echo "[demo-runner] Creating fresh demo agent DB at $DEMO_DB ..."
uv run python scripts/setup_demo_agent.py

echo "[demo-runner] Starting isolated server on $DEMO_URL (DB=$DEMO_DB) ..."
# Force standalone mode — both belt AND braces for #868:
#   * KESTREL_ROOKERY_CONFIG points at a non-existent path so the server
#     skips rookery loading.  Without this, server.py:201 auto-loads
#     rookery.toml from the project root and mounts every sibling agent
#     (Meridian, Claw, Nellie) alongside the demo agent.
#   * KESTREL_DEMO_SERVER=1 makes server.py refuse the auto-load even if
#     someone removes the explicit KESTREL_ROOKERY_CONFIG line above.
KESTREL_DB_PATH="$DEMO_DB" \
KESTREL_ROOKERY_CONFIG="$DEMO_DB/rookery-disabled.toml" \
KESTREL_DEMO_SERVER=1 \
    uv run uvicorn server:app --host 127.0.0.1 --port "$DEMO_PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  local code=$?
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[demo-runner] Stopping server (PID $SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$code" -ne 0 ]; then
    echo "[demo-runner] Server log: $SERVER_LOG"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

echo -n "[demo-runner] Waiting for $DEMO_URL/health "
for i in $(seq 1 60); do
  if curl -sS -o /dev/null -w "%{http_code}" "$DEMO_URL/health" 2>/dev/null | grep -q '^200$'; then
    echo "up."
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "[demo-runner] Server exited before becoming healthy. Log tail:" >&2
    tail -50 "$SERVER_LOG" >&2
    exit 1
  fi
  printf "."
  sleep 1
done

if ! curl -sS -o /dev/null -w "%{http_code}" "$DEMO_URL/health" 2>/dev/null | grep -q '^200$'; then
  echo ""
  echo "[demo-runner] Timed out waiting for server health. Log tail:" >&2
  tail -50 "$SERVER_LOG" >&2
  exit 1
fi

# Sanity-check (#868 acceptance criterion #3) — abort startup if the demo
# server somehow reports a non-demo agent.  The two upstream defences
# (KESTREL_ROOKERY_CONFIG override + KESTREL_DEMO_SERVER=1) should make
# this unreachable in practice, but the cost of a false negative is
# wiping a live agent — so we re-check at the boundary.
echo "[demo-runner] Verifying every loaded agent is is_demo=true ..."
AGENTS_JSON="$(curl -sS "$DEMO_URL/api/agents" || true)"
LIVE_AGENTS="$(printf '%s' "$AGENTS_JSON" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print('!!parse-error', end='')
    sys.exit(0)
live = [a.get('name') or a.get('id') or '<unnamed>' for a in (data.get('agents') or []) if a.get('is_demo') is not True]
print(','.join(live), end='')
")"
case "$LIVE_AGENTS" in
  '!!parse-error')
    echo "[demo-runner] Could not parse /api/agents response — refusing to run." >&2
    echo "$AGENTS_JSON" | head -c 400 >&2
    echo "" >&2
    exit 1
    ;;
  '')
    : # all agents are demo-scoped, proceed
    ;;
  *)
    echo "[demo-runner] Refusing to run: server reports non-demo agent(s) — $LIVE_AGENTS" >&2
    echo "[demo-runner] This is the routing precondition that wiped Meridian (#867/#868)." >&2
    exit 1
    ;;
esac

echo "[demo-runner] Running demos/$DEMO ..."
cd "demos/$DEMO"
KESTREL_URL="$DEMO_URL" npx playwright test --config=config.cjs
status=$?

echo "[demo-runner] Done (exit=$status). Artifacts in demos/$DEMO/demo-output/"
exit "$status"
