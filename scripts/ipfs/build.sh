#!/bin/bash
set -e

# Build and push Kestrel IPFS (Kubo + GCS datastore) image to GCR
#
# This builds a custom Kubo image that stores IPFS blocks in Google Cloud Storage,
# giving us a self-hosted IPFS node with durable, cost-effective block storage.
#
# Usage:
#   ./scripts/ipfs/build.sh [tag]
#
# Requires:
#   - GCP_PROJECT_ID env var
#   - Docker with buildx (for multi-arch)
#   - Authenticated to GCR: gcloud auth configure-docker

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
IMAGE_NAME="kestrel-ipfs-gcs"
TAG="${1:-latest}"

echo "Building Kestrel IPFS (Kubo + GCS) image..."
echo "  Project: $PROJECT_ID"
echo "  Tag: $TAG"
echo "  Platform: linux/amd64 (GCE target)"
echo ""

cd "$(dirname "$0")/../.."

docker build \
    --platform linux/amd64 \
    -f docker/ipfs/Dockerfile.gcs \
    -t "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}" \
    -t "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" \
    docker/ipfs/

echo ""
echo "Pushing to GCR..."
docker push "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
docker push "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

echo ""
echo "Done! Image: gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
