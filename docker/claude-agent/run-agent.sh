#!/bin/bash
# Run Claude Code agent in container
# Usage: ./run-agent.sh [claim|process] [options]
#
# Examples:
#   ./run-agent.sh claim --repo owner/repo --issue 42
#   ./run-agent.sh claim --repo owner/repo --label enhancement
#   ./run-agent.sh process --repo owner/repo --assignee claude-bot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Kestrel Claude Agent ===${NC}"

# Check for required environment variables
if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${RED}ERROR: GITHUB_TOKEN not set${NC}"
    echo "Export your GitHub personal access token:"
    echo "  export GITHUB_TOKEN=ghp_..."
    exit 1
fi

# Get Claude OAuth token
if [ -z "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
    echo -e "${YELLOW}Extracting Claude OAuth token from Keychain...${NC}"
    if [[ "$(uname)" == "Darwin" ]]; then
        eval "$($SCRIPT_DIR/export-token.sh)"
    else
        echo -e "${RED}ERROR: CLAUDE_CODE_OAUTH_TOKEN not set and not on macOS${NC}"
        echo "Set it manually or run export-token.sh on macOS first"
        exit 1
    fi
fi

# Build image if needed
IMAGE_NAME="kestrel-claude-agent"
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
fi

# Run the agent
echo -e "${GREEN}Starting agent container...${NC}"
echo "Project: $PROJECT_ROOT"
echo "Command: kestrel-github $*"

docker run --rm \
    -e CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN" \
    -e GITHUB_TOKEN="$GITHUB_TOKEN" \
    -e GIT_EMAIL="${GIT_EMAIL:-agent@kestrel.local}" \
    -e GIT_NAME="${GIT_NAME:-Kestrel Agent}" \
    -v "$PROJECT_ROOT:/home/agent/workspace:rw" \
    "$IMAGE_NAME" \
    uv run kestrel-github "$@"
