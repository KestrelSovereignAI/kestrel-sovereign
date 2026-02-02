#!/bin/bash
# Build Kestrel Agent Docker image (Remote LLM mode)
# Lightweight image without torch, spacy, chromadb (~500MB vs 32GB)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="kestrel-remote"
TAG="${1:-latest}"
PLATFORM="${2:-linux/amd64}"

echo "Building $IMAGE_NAME:$TAG for $PLATFORM..."
echo "Using lightweight dependencies (no torch, spacy, chromadb)"

cd "$PROJECT_ROOT"

docker build \
    -f Dockerfile.agent.remote \
    -t "$IMAGE_NAME:$TAG" \
    --platform "$PLATFORM" \
    .

# Show image size
SIZE=$(docker images "$IMAGE_NAME:$TAG" --format "{{.Size}}")
echo ""
echo "✅ Built $IMAGE_NAME:$TAG ($SIZE)"
echo ""
echo "To run: ./scripts/run_docker_remote.sh"
