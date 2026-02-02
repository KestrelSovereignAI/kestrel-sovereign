#!/bin/bash
set -e

# Claude Code Agent Container Entrypoint
# Handles OAuth token setup and runs the agent

echo "=== Claude Code Agent Container ==="
echo "Starting at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Validate required environment variables
if [ -z "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
    echo "ERROR: CLAUDE_CODE_OAUTH_TOKEN is required"
    echo "Export it from your local machine:"
    echo '  export CLAUDE_CODE_OAUTH_TOKEN=$(security find-generic-password -s "Claude Code-credentials" -w | python3 -c "import sys,json; print(json.load(sys.stdin)['\''claudeAiOauth'\'']['\''accessToken'\''])")'
    exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "ERROR: GITHUB_TOKEN is required for GitHub API access"
    exit 1
fi

# Setup Claude credentials file
echo "Setting up Claude credentials..."
mkdir -p /home/agent/.claude
cat > /home/agent/.claude/.credentials.json << EOF
{
  "claudeAiOauth": {
    "accessToken": "$CLAUDE_CODE_OAUTH_TOKEN"
  }
}
EOF
chmod 600 /home/agent/.claude/.credentials.json

# Setup git credentials for pushing
echo "Setting up git credentials..."
git config --global user.email "${GIT_EMAIL:-agent@kestrel.local}"
git config --global user.name "${GIT_NAME:-Kestrel Agent}"
git config --global credential.helper "store"

# Store GitHub token for git operations
echo "https://x-access-token:${GITHUB_TOKEN}@github.com" > /home/agent/.git-credentials
chmod 600 /home/agent/.git-credentials

# Setup GitHub CLI auth
echo "$GITHUB_TOKEN" | gh auth login --with-token 2>/dev/null || true

# IMPORTANT: Always use container-local venv at /home/agent/.venv
# This avoids conflicts with host .venv and ensures we never use system Python
export UV_PROJECT_ENVIRONMENT=/home/agent/.venv

# If workspace is mounted and has a pyproject.toml, install dependencies
if [ -f "/home/agent/workspace/pyproject.toml" ]; then
    cd /home/agent/workspace
    echo "Installing dependencies to container venv ($UV_PROJECT_ENVIRONMENT)..."
    uv sync 2>&1 || echo "Note: uv sync had issues, continuing anyway"
fi

# Execute the command passed to the container
echo "=== Running command ==="
echo "Using venv: $UV_PROJECT_ENVIRONMENT"
exec "$@"
