#!/bin/bash
set -e

# Build and push Kestrel Cloud Run image to GCR
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
IMAGE_NAME="kestrel"
TAG="${1:-latest}"

echo "Building Kestrel Cloud Run image..."
echo "  Project: $PROJECT_ID"
echo "  Tag: $TAG"
echo ""

# Build from project root
cd "$(dirname "$0")/../.."

# GitHub token for private repo access (kestrel-talon)
GITHUB_TOKEN="${GITHUB_TOKEN:-$(gh auth token 2>/dev/null || true)}"
if [ -z "$GITHUB_TOKEN" ]; then
    echo "WARNING: No GITHUB_TOKEN found. Build may fail if private repos are dependencies."
fi

# Multi-arch build + push (amd64 for Cloud Run, arm64 for local/Apple Silicon)
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile.cloudrun \
    --secret id=github_token,env=GITHUB_TOKEN \
    -t "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}" \
    -t "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" \
    --push \
    .

echo ""
echo "Done! Image: gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
