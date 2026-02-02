#!/bin/bash
set -e

# Configuration - using default Cloud Build service account
SERVICE_ACCOUNT="523805591861@cloudbuild.gserviceaccount.com"
PROJECT_ID="YOUR_PROJECT_ID"
IMAGE_NAME="kestrel-test"

echo "🧪 Building test Docker image..."
echo "Service Account: $SERVICE_ACCOUNT (default Cloud Build service account)"
echo "Project: $PROJECT_ID"
echo "Image: gcr.io/$PROJECT_ID/$IMAGE_NAME"
echo ""

echo "📁 Build context: $(pwd)"
echo "📄 Dockerfile: docker/Dockerfile.test"
echo ""

# Create cloudbuild.yaml for test build
cat > /tmp/cloudbuild.test.yaml <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/$IMAGE_NAME', '-f', 'docker/Dockerfile.test', '.']
images:
- 'gcr.io/$PROJECT_ID/$IMAGE_NAME'
EOF

# Build using Cloud Build
echo "🚀 Starting Cloud Build test..."
gcloud builds submit \
  --config=/tmp/cloudbuild.test.yaml \
  --project=$PROJECT_ID \
  .

echo ""
echo "✅ Test build completed successfully!"
echo "📦 Image available at: gcr.io/$PROJECT_ID/$IMAGE_NAME"