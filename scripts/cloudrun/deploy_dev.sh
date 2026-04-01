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
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --port 8080 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 300 \
    --concurrency 80 \
    --set-secrets="OPENAI_API_KEY=kestrel-openai-key:latest,ANTHROPIC_API_KEY=kestrel-anthropic-key:latest,KESTREL_API_KEY=kestrel-api-key:latest,KESTREL_DATA_KEY=kestrel-data-key:latest,GOOGLE_CLIENT_ID=kestrel-google-client-id:latest,GOOGLE_CLIENT_SECRET=kestrel-google-client-secret:latest,KESTREL_SESSION_SECRET=kestrel-session-secret:latest" \
    --set-env-vars="KESTREL_ENV=development,KESTREL_DB_BACKEND=sqlite,KESTREL_DB_PATH=/app/agent_data,KESTREL_MULTI_AGENT=true,KESTREL_REQUIRE_OAUTH=true,KESTREL_ALLOWED_EMAILS=${KESTREL_ALLOWED_EMAILS:?Set KESTREL_ALLOWED_EMAILS env var (comma-separated list of authorized email addresses)}" \
    --project="$PROJECT_ID" \
    --quiet

echo ""
echo "DEV deployment complete!"
echo ""
echo "Service URL (sign in with authorized Google account):"
gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --project="$PROJECT_ID" \
    --format='value(status.url)'
