#!/bin/bash
set -e

# Deploy Kestrel Rookery (multi-agent host) to Cloud Run — DEV environment
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
SERVICE_NAME="kestrel-rookery-dev"
IMAGE_NAME="kestrel-rookery"
REGION="us-central1"

echo "Deploying Kestrel Rookery DEV to Cloud Run..."
echo "  Project: $PROJECT_ID"
echo "  Service: $SERVICE_NAME"
echo "  Region: $REGION"
echo "  Mode: Rookery (multi-agent host)"
echo "  Scaling: min=0 (scale to zero), max=10"
echo "  Resources: 4Gi memory, 4 CPU (host + agents)"
echo ""

gcloud run deploy "$SERVICE_NAME" \
    --image "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 4 \
    --port 8080 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 300 \
    --concurrency 80 \
    --set-secrets="OPENAI_API_KEY=kestrel-openai-key:latest,KESTREL_API_KEY=kestrel-api-key:latest,KESTREL_DATA_KEY=kestrel-data-key:latest,LIGHTHOUSE_API_KEY=kestrel-lighthouse-key:latest,GOOGLE_CLIENT_ID=kestrel-google-client-id:latest,GOOGLE_CLIENT_SECRET=kestrel-google-client-secret:latest,KESTREL_SESSION_SECRET=kestrel-session-secret:latest" \
    --set-env-vars="KESTREL_ENV=development,KESTREL_DB_BACKEND=sqlite,KESTREL_HOST_AUTOSTART=true,KESTREL_ALLOWED_EMAILS=jaslogic@gmail.com\,noelschulz1981@gmail.com\,gabriela.aquino@gmail.com" \
    --project="$PROJECT_ID" \
    --quiet

echo ""
echo "Rookery DEV deployment complete!"
echo ""
echo "Service URL (sign in with authorized Google account):"
gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --project="$PROJECT_ID" \
    --format='value(status.url)'
