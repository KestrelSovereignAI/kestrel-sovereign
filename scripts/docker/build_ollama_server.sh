#!/bin/bash
set -e

# Configuration - Using kestrel-agent-admin service account
SERVICE_ACCOUNT="kestrel-agent-admin@YOUR_PROJECT_ID.iam.gserviceaccount.com"
PROJECT_ID="YOUR_PROJECT_ID"
IMAGE_NAME="kestrel-ollama"

echo "=== Building Kestrel Ollama Server Docker image ==="
echo "Service Account: $SERVICE_ACCOUNT"
echo "Project: $PROJECT_ID"
echo "Image: gcr.io/$PROJECT_ID/$IMAGE_NAME"
echo ""

echo "Build context: $(pwd)"
echo "Dockerfile: docker/Dockerfile.ollama-server"
echo ""

echo "Building Ollama Server with:"
echo "  - Official Ollama image base"
echo "  - Persistent model storage support"
echo "  - Health check endpoint"
echo "  - Model pre-pull on startup"
echo ""

# Create cloudbuild.yaml for Ollama server build
cat > /tmp/cloudbuild.kestrel-ollama.yaml <<'EOF'
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'docker/Dockerfile.ollama-server', '-t', 'gcr.io/$PROJECT_ID/kestrel-ollama:latest', '.']
    timeout: 1200s  # 20 minutes should be enough
images:
  - 'gcr.io/$PROJECT_ID/kestrel-ollama:latest'
timeout: 1200s
EOF

# Build using Cloud Build with kestrel-agent-admin service account
echo "Starting Cloud Build..."
gcloud builds submit \
  --config=/tmp/cloudbuild.kestrel-ollama.yaml \
  --project=$PROJECT_ID \
  --account=$SERVICE_ACCOUNT \
  .

echo ""
echo "Ollama Server Docker image ready!"
echo "Image available at: gcr.io/$PROJECT_ID/$IMAGE_NAME:latest"
echo ""
echo "Next steps:"
echo "1. Create RunPod template with this image"
echo "2. Or use directly with runpod_config.toml [profiles.ollama]"
echo "3. Set OLLAMA_MODELS_PULL env var for pre-pulling models"
