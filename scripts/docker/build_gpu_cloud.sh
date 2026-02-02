#!/bin/bash
set -e

# Configuration - Using default Cloud Build service account
SERVICE_ACCOUNT="523805591861@cloudbuild.gserviceaccount.com"  # Default Cloud Build service account
PROJECT_ID="YOUR_PROJECT_ID"
IMAGE_NAME="kestrel-gpu"

echo "🔨 Building Kestrel GPU Docker image..."
echo "Service Account: $SERVICE_ACCOUNT (default Cloud Build service account)"
echo "Project: $PROJECT_ID"
echo "Image: gcr.io/$PROJECT_ID/$IMAGE_NAME"
echo ""

echo "📁 Build context: $(pwd)"
echo "📄 Dockerfile: docker/Dockerfile.gpu"
echo ""

echo "📦 Building GPU-enabled Kestrel with:"
echo "  - NVIDIA CUDA 11.8 runtime"
echo "  - Python 3.11 compiled from source"
echo "  - Ollama with GPU support"
echo "  - Kestrel agent framework"
echo ""

# Create cloudbuild.yaml for GPU build
cat > /tmp/cloudbuild.kestrel-gpu.yaml <<'EOF'
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'docker/Dockerfile.gpu', '-t', 'gcr.io/$PROJECT_ID/kestrel-gpu:latest', '.']
    timeout: 1800s  # 30 minutes for CUDA build
images:
  - 'gcr.io/$PROJECT_ID/kestrel-gpu:latest'
EOF

# Build using Cloud Build with default service account (no credentials needed)
echo "🚀 Starting Cloud Build (this may take 15-30 minutes due to CUDA compilation)..."
gcloud builds submit \
  --config=/tmp/cloudbuild.kestrel-gpu.yaml \
  --project=$PROJECT_ID \
  --async \
  .

# Wait for build to complete by polling status
echo "⏳ Waiting for build to complete..."
BUILD_ID=$(gcloud builds list --project=$PROJECT_ID --limit=1 --format="value(id)" --ongoing)
while [ ! -z "$BUILD_ID" ]; do
  STATUS=$(gcloud builds describe $BUILD_ID --project=$PROJECT_ID --format="value(status)" 2>/dev/null || echo "UNKNOWN")
  if [ "$STATUS" = "SUCCESS" ]; then
    echo "✅ Build completed successfully!"
    echo "📦 Image available at: gcr.io/$PROJECT_ID/kestrel-gpu:latest"
    break
  elif [ "$STATUS" = "FAILURE" ] || [ "$STATUS" = "TIMEOUT" ]; then
    echo "❌ Build failed with status: $STATUS"
    echo "📋 Check build logs:"
    echo "gcloud builds log $BUILD_ID --project=$PROJECT_ID"
    exit 1
  else
    echo "🔄 Build status: $STATUS (waiting...)"
    sleep 30
  fi
done

echo ""
echo "🎉 GPU Docker image ready for RunPod testing!"
echo "📝 Next steps:"
echo "1. Update test_runpod_gpu.py with image: gcr.io/$PROJECT_ID/kestrel-gpu:latest"
echo "2. Run: python test_runpod_gpu.py"