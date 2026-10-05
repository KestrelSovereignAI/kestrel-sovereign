#!/usr/bin/env bash
# Local clean-install validation — the macOS/Windows axis on your own machine.
#
# The clean-install GitHub workflow only runs the billed macOS/Windows runners
# as a pre-publish gate (and a weekly sweep). To avoid the gate failing AFTER
# you cut a release tag, run this on your dev machine BEFORE tagging — it
# reproduces the same readiness assertions the workflow runs, for free, on
# whatever OS you're sitting at.
#
# It delegates every assertion to scripts/ci/clean_install_verify.py — the same
# pure-Python checks the workflow uses — so local and CI stay in lockstep.
#
# Usage:
#   scripts/ci/clean_install_local.sh             # sync  install method (default)
#   scripts/ci/clean_install_local.sh wheel       # wheel install method
#
# Run from a clean checkout. It builds into the project venv (sync) or a
# throwaway venv (wheel), so it does not disturb an editable dev install.
set -euo pipefail

INSTALL_METHOD="${1:-sync}"
PYTHON_VERSION="${KESTREL_PY_VERSION:-3.13}"
AGENT_NAME="Kestrel"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Inception writes durable identity/config files. Refuse an existing local
# checkout rather than backing up or reusing a developer's agent, which would
# make this a dirty-state test and could silently select their cloud routes.
for state_path in .env kestrel.toml multi_agent.toml agent_data; do
  if [ -e "$state_path" ] || [ -L "$state_path" ]; then
    echo "error: clean-install rehearsal requires no existing $state_path; use a fresh checkout" >&2
    exit 2
  fi
done

# Keep host-feature state out of the operator's ~/.kestrel, and never discover
# a paid route from credentials exported in the invoking shell. A fresh
# checkout has no persisted provider keys; the quickstart chooses local Ollama
# (or its no-provider default), and genesis remains explicitly pending.
HARNESS_DIR="$(mktemp -d)"
trap 'rm -rf -- "$HARNESS_DIR"' EXIT
export KESTREL_HOST_DB_PATH="$HARNESS_DIR/host-features.db"
unset KESTREL_HOME KESTREL_DB_PATH AGENT_DATA_DIR KESTREL_DATA_DIR KESTREL_MULTI_AGENT_CONFIG
unset OPENAI_API_KEY ANTHROPIC_API_KEY OPENROUTER_API_KEY OPENROUTER_MANAGEMENT_API_KEY
unset GOOGLE_API_KEY GEMINI_API_KEY DEEPSEEK_API_KEY XAI_API_KEY MOONSHOT_API_KEY

# Mirror the workflow env so the wizard takes its non-interactive, test-instance
# path and never blocks on a prompt or pollutes a real agent DB.  The paired
# test-instance + audit-mode settings defer genesis as *pending* (never passed)
# even when this host happens to run Ollama; no model is called by the harness.
export KESTREL_AUDIT_MODE="skip"
export KESTREL_NONINTERACTIVE="1"
export KESTREL_TEST_INSTANCE="1"
export PYTHONSAFEPATH="1"

echo "==> Clean-install local validation: ${INSTALL_METHOD} install / Python ${PYTHON_VERSION} / $(uname -s)"

uv python install "${PYTHON_VERSION}"

if [ "${INSTALL_METHOD}" = "wheel" ]; then
  # Mirror `pip install kestrel-sovereign`: build a wheel and install it into a
  # genuinely disposable venv OUTSIDE the checkout. A bare `uv venv` would
  # reuse the project's ./.venv and overwrite the developer's editable install
  # with the wheel; an explicit out-of-tree path keeps the dev env untouched.
  WHEEL_VENV="$HARNESS_DIR/venv"
  uv build --wheel
  uv venv --python "${PYTHON_VERSION}" "${WHEEL_VENV}"
  uv pip install --python "${WHEEL_VENV}" dist/*.whl
  PY=("${WHEEL_VENV}/bin/python")
  KESTREL=("${WHEEL_VENV}/bin/kestrel")
elif [ "${INSTALL_METHOD}" = "sync" ]; then
  # Source-clone path: `uv sync` resolves the project's ./.venv to the lock —
  # the dev env's normal state — and `uv run` invokes through it.
  uv sync
  PY=(uv run python)
  KESTREL=(uv run kestrel)
else
  echo "error: install method must be 'sync' or 'wheel' (got '${INSTALL_METHOD}')" >&2
  exit 2
fi

verify() { "${PY[@]}" scripts/ci/clean_install_verify.py "$@"; }

echo "==> Setup wizard (kestrel setup --quickstart)"
"${KESTREL[@]}" setup --quickstart

# A developer may already have a real Kestrel host on 8888 or an agent on
# 8801. Rebind only this newly-created test config to free loopback ports.
"${PY[@]}" scripts/ci/clean_install_local_ports.py

echo "==> Readiness assertions"
verify wizard-artifacts
"${KESTREL[@]}" doctor
verify identity --agent-name "${AGENT_NAME}"
verify constitution --agent-name "${AGENT_NAME}"
verify memory --agent-name "${AGENT_NAME}"
verify test-instance --agent-name "${AGENT_NAME}"
verify start-and-health --agent-name "${AGENT_NAME}"
verify host-and-chat-503
verify did-persists --agent-name "${AGENT_NAME}"

echo "==> PASS — clean install OK on $(uname -s) (${INSTALL_METHOD}). Safe to tag."
