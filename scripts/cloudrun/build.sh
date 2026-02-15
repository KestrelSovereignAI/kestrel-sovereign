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

docker build \
    -f docker/Dockerfile.cloudrun \
    -t "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}" \
    -t "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" \
    .

echo ""
echo "Pushing to GCR..."
docker push "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
docker push "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

echo ""
echo "Done! Image: gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
