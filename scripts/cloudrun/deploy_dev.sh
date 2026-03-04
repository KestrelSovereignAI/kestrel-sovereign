#!/bin/bash
set -e

# Deploy Kestrel to Cloud Run — DEV environment
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
SERVICE_NAME="kestrel-dev"
IMAGE_NAME="kestrel"
REGION="us-central1"

echo "Deploying Kestrel DEV to Cloud Run..."
echo "  Project: $PROJECT_ID"
echo "  Service: $SERVICE_NAME"
echo "  Region: $REGION"
echo "  Scaling: min=0 (scale to zero), max=10"
echo ""

gcloud run deploy "$SERVICE_NAME" \
    --image "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" \
    --region "$REGION" \
    --platform managed \
    --no-allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --port 8080 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 300 \
    --concurrency 80 \
    --set-secrets="OPENAI_API_KEY=kestrel-openai-key:latest,KESTREL_API_KEY=kestrel-api-key:latest,KESTREL_DATA_KEY=kestrel-data-key:latest" \
    --set-env-vars="KESTREL_ENV=development,KESTREL_DB_BACKEND=sqlite,KESTREL_DB_PATH=/app/agent_data" \
    --project="$PROJECT_ID" \
    --quiet

echo ""
echo "DEV deployment complete!"
echo ""
echo "Service URL (requires Google IAM auth):"
gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --project="$PROJECT_ID" \
    --format='value(status.url)'
echo ""
echo "Access via proxy: gcloud run services proxy $SERVICE_NAME --region=$REGION --project=$PROJECT_ID"
