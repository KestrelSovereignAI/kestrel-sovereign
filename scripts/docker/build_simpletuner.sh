#!/bin/bash
set -e

# Build SimpleTuner FLUX.2 Training + Inference Image
# Uses kestrel-agent-admin service account for GCR access

SERVICE_ACCOUNT="kestrel-agent-admin@YOUR_PROJECT_ID.iam.gserviceaccount.com"
PROJECT_ID="YOUR_PROJECT_ID"
IMAGE_NAME="kestrel-lora"
DOCKERFILE="docker/Dockerfile.simpletuner"

echo "🔨 Building SimpleTuner FLUX.2 Docker image..."
echo ""
echo "📦 Image includes:"
echo "  - SimpleTuner training framework"
echo "  - FLUX.2-dev inference pipeline"
echo "  - LoRA training endpoint (/train)"
echo "  - LoRA inference endpoint (/generate)"
echo "  - LoRA listing endpoint (/loras)"
echo ""
echo "Service Account: $SERVICE_ACCOUNT"
echo "Project: $PROJECT_ID"
echo "Image: gcr.io/$PROJECT_ID/$IMAGE_NAME:latest"
echo ""
echo "📁 Build context: $(pwd)"
echo "📄 Dockerfile: $DOCKERFILE"
echo ""

# Verify we're in the right directory
if [ ! -f "$DOCKERFILE" ]; then
    echo "❌ Error: $DOCKERFILE not found"
    echo "   Run this script from the kestrel project root"
    exit 1
fi

# Verify the API file exists and has the generate endpoint
if ! grep -q "@app.post.*generate" docker/simpletuner_api.py; then
    echo "❌ Error: /generate endpoint not found in simpletuner_api.py"
    exit 1
fi

echo "✅ Verified: simpletuner_api.py has /generate endpoint"
echo ""

# Build using Cloud Build with proper service account
echo "🚀 Starting Cloud Build (this may take 15-30 minutes)..."
gcloud builds submit \
  --config=docker/cloudbuild-simpletuner.yaml \
  --project=$PROJECT_ID \
  --account=$SERVICE_ACCOUNT \
  .

echo ""
echo "🎉 SimpleTuner image ready!"
echo "📦 Image: gcr.io/$PROJECT_ID/$IMAGE_NAME:latest"
echo ""
echo "📝 Next steps:"
echo "1. Unlock the RunPod pod: l6bcivl0w96gq5"
echo "2. Stop and restart the pod to pull new image"
echo "3. Test /loras and /generate endpoints"
