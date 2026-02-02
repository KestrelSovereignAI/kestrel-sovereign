#!/bin/bash
#
# Sovereign Agent Manager
#
# Runs Kestrel agents in Docker containers with cryptographic isolation.
# The agent receives KESTREL_DATA_KEY but cannot access your filesystem
# to discover where you store it.
#
# Usage:
#   ./scripts/sovereign-agent.sh create Emma ~/emma_data
#   ./scripts/sovereign-agent.sh chat ~/emma_data
#   ./scripts/sovereign-agent.sh retire ~/emma_data
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="kestrel-sovereign"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for KESTREL_DATA_KEY
check_key() {
    if [ -z "$KESTREL_DATA_KEY" ]; then
        log_error "KESTREL_DATA_KEY is not set!"
        echo ""
        echo "Generate a key with:"
        echo "  python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        echo ""
        echo "Then set it:"
        echo "  export KESTREL_DATA_KEY=\"your-key-here\""
        echo ""
        echo "Store it safely (password manager, ~/.zshrc, etc.)"
        exit 1
    fi
}

# Build the Docker image if needed
build_image() {
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        log_info "Building $IMAGE_NAME image..."
        docker build -f "$PROJECT_DIR/docker/Dockerfile.sovereign" -t "$IMAGE_NAME" "$PROJECT_DIR"
    else
        log_info "Using existing $IMAGE_NAME image"
    fi
}

# Create a new agent
cmd_create() {
    local name="$1"
    local data_dir="$2"

    if [ -z "$name" ] || [ -z "$data_dir" ]; then
        echo "Usage: $0 create <agent-name> <data-directory>"
        echo "Example: $0 create Emma ~/emma_data"
        exit 1
    fi

    check_key
    build_image

    # Expand ~ in path
    data_dir="${data_dir/#\~/$HOME}"

    # Create data directory if it doesn't exist
    mkdir -p "$data_dir"

    log_info "Creating agent '$name' in $data_dir..."

    docker run --rm \
        -e KESTREL_DATA_KEY="$KESTREL_DATA_KEY" \
        -v "$data_dir:/data" \
        "$IMAGE_NAME" \
        inception_service.py --name "$name" --output /data

    # NOTE: "--output" is an explicit alias for "--output-dir" in inception_service.py.
    # Do not rely on argparse prefix-abbreviation behavior.

    log_info "Agent '$name' created successfully!"
    echo ""
    echo "Start chatting with:"
    echo "  $0 chat $data_dir"
}

# Chat with an existing agent
cmd_chat() {
    local data_dir="$1"

    if [ -z "$data_dir" ]; then
        echo "Usage: $0 chat <data-directory>"
        echo "Example: $0 chat ~/emma_data"
        exit 1
    fi

    check_key
    build_image

    # Expand ~ in path
    data_dir="${data_dir/#\~/$HOME}"

    if [ ! -f "$data_dir/kestrel_prime.db" ]; then
        log_error "No agent found in $data_dir"
        echo "Create one first with: $0 create <name> $data_dir"
        exit 1
    fi

    log_info "Starting chat with agent in $data_dir..."
    echo ""

    docker run -it --rm \
        -e KESTREL_DATA_KEY="$KESTREL_DATA_KEY" \
        -v "$data_dir:/data" \
        "$IMAGE_NAME" \
        main.py /data/kestrel_prime.db
}

# Retire a test agent
cmd_retire() {
    local data_dir="$1"

    if [ -z "$data_dir" ]; then
        echo "Usage: $0 retire <data-directory>"
        echo "Example: $0 retire ~/test_agent_data"
        exit 1
    fi

    check_key
    build_image

    # Expand ~ in path
    data_dir="${data_dir/#\~/$HOME}"

    if [ ! -f "$data_dir/kestrel_prime.db" ]; then
        log_error "No agent found in $data_dir"
        exit 1
    fi

    log_warn "This will retire the agent in $data_dir"
    read -p "Are you sure? (yes/no) " confirm

    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi

    docker run --rm \
        -e KESTREL_DATA_KEY="$KESTREL_DATA_KEY" \
        -v "$data_dir:/data" \
        "$IMAGE_NAME" \
        retirement_service.py /data/kestrel_prime.db
}

# Run a custom command
cmd_run() {
    local data_dir="$1"
    shift
    local command="$@"

    if [ -z "$data_dir" ] || [ -z "$command" ]; then
        echo "Usage: $0 run <data-directory> <command>"
        echo "Example: $0 run ~/emma_data python -c 'print(\"hello\")'"
        exit 1
    fi

    check_key
    build_image

    # Expand ~ in path
    data_dir="${data_dir/#\~/$HOME}"

    docker run --rm \
        -e KESTREL_DATA_KEY="$KESTREL_DATA_KEY" \
        -v "$data_dir:/data" \
        "$IMAGE_NAME" \
        $command
}

# Rebuild the image
cmd_build() {
    log_info "Rebuilding $IMAGE_NAME image..."
    docker build --no-cache -f "$PROJECT_DIR/docker/Dockerfile.sovereign" -t "$IMAGE_NAME" "$PROJECT_DIR"
    log_info "Build complete!"
}

# Show help
cmd_help() {
    echo "Sovereign Agent Manager"
    echo ""
    echo "Runs Kestrel agents in isolated Docker containers."
    echo "The agent receives your KESTREL_DATA_KEY but cannot access"
    echo "your filesystem to discover where you store it."
    echo ""
    echo "Commands:"
    echo "  create <name> <dir>  Create a new agent"
    echo "  chat <dir>           Chat with an existing agent"
    echo "  retire <dir>         Retire a test agent"
    echo "  run <dir> <cmd>      Run a custom command"
    echo "  build                Rebuild the Docker image"
    echo "  help                 Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 create Emma ~/emma_data"
    echo "  $0 chat ~/emma_data"
    echo "  $0 retire ~/test_agent"
    echo ""
    echo "Environment:"
    echo "  KESTREL_DATA_KEY     Required. Your master encryption key."
    echo ""
    echo "Security:"
    echo "  The agent runs in a container with only /data mounted."
    echo "  It cannot read ~/.zshrc, ~/.ssh, or any other host files."
    echo "  Store your KESTREL_DATA_KEY anywhere on your host - the"
    echo "  agent will never be able to find it."
}

# Main
case "${1:-help}" in
    create)  cmd_create "$2" "$3" ;;
    chat)    cmd_chat "$2" ;;
    retire)  cmd_retire "$2" ;;
    run)     shift; cmd_run "$@" ;;
    build)   cmd_build ;;
    help)    cmd_help ;;
    *)       cmd_help; exit 1 ;;
esac
