#!/bin/bash
set -e

# Configuration - Using kestrel-agent-admin service account
SERVICE_ACCOUNT="kestrel-agent-admin@YOUR_PROJECT_ID.iam.gserviceaccount.com"
PROJECT_ID="YOUR_PROJECT_ID"
IMAGE_NAME="kestrel-lora"
DOCKERFILE="docker/Dockerfile.lora-trainer"

echo "🔨 Building Kestrel LoRA Trainer Docker image..."
echo ""
echo "📦 Optimized Multi-Stage Build:"
echo "  - Stage 1: CUDA devel for building wheels"
echo "  - Stage 2: CUDA runtime for final image"
echo "  - --no-cache-dir on all pip installs"
echo "  - Network volume for model caching (/workspace)"
echo "  - Target size: ~12-14GB"
echo ""
echo "Service Account: $SERVICE_ACCOUNT"
echo "Project: $PROJECT_ID"
echo "Image: gcr.io/$PROJECT_ID/$IMAGE_NAME:latest"
echo ""
echo "📁 Build context: $(pwd)"
echo "📄 Dockerfile: $DOCKERFILE"
echo ""

# Create cloudbuild.yaml for LoRA trainer build
cat > /tmp/cloudbuild.kestrel-lora.yaml <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '--no-cache', '-f', '$DOCKERFILE', '-t', 'gcr.io/$PROJECT_ID/kestrel-lora:latest', '.']
    timeout: 3600s  # 60 minutes for large PyTorch install
images:
  - 'gcr.io/$PROJECT_ID/kestrel-lora:latest'
timeout: 3600s
EOF

# Build using Cloud Build with kestrel-agent-admin service account
echo "🚀 Starting Cloud Build (this may take 20-40 minutes)..."
gcloud builds submit \
  --config=/tmp/cloudbuild.kestrel-lora.yaml \
  --project=$PROJECT_ID \
  --account=$SERVICE_ACCOUNT \
  .

echo ""
echo "🎉 LoRA Trainer Docker image ready!"
echo "📦 Image available at: gcr.io/$PROJECT_ID/$IMAGE_NAME:latest"
echo ""
echo "📝 Next steps:"
echo "1. Verify runpod_config.toml has: image_name = \"gcr.io/$PROJECT_ID/$IMAGE_NAME:latest\""
echo "2. Test LoRA training via: POST /api/companions/{id}/train-lora"
