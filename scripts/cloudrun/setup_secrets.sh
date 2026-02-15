#!/bin/bash
set -e

# One-time setup: Create GCP Secret Manager secrets for Kestrel Cloud Run
# Reads values from .env file in project root
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"

echo "Setting up GCP Secret Manager secrets for Kestrel..."
echo "  Project: $PROJECT_ID"
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.."

if [ ! -f .env ]; then
    echo "Error: .env file not found in project root"
    exit 1
fi

# Enable Secret Manager API
echo "Enabling Secret Manager API..."
gcloud services enable secretmanager.googleapis.com --project="$PROJECT_ID" 2>/dev/null || true

# Helper: create or update a secret
create_secret() {
    local SECRET_NAME="$1"
    local ENV_VAR_NAME="$2"

    # Extract value from .env (strip quotes)
    local VALUE
    VALUE=$(grep "^${ENV_VAR_NAME}=" .env | cut -d'=' -f2- | tr -d '"' | tr -d "'")

    if [ -z "$VALUE" ]; then
        echo "  SKIP: $ENV_VAR_NAME not found in .env"
        return
    fi

    # Check if secret exists
    if gcloud secrets describe "$SECRET_NAME" --project="$PROJECT_ID" &>/dev/null; then
        echo "  UPDATE: $SECRET_NAME (adding new version)"
        echo -n "$VALUE" | gcloud secrets versions add "$SECRET_NAME" \
            --data-file=- \
            --project="$PROJECT_ID"
    else
        echo "  CREATE: $SECRET_NAME"
        echo -n "$VALUE" | gcloud secrets create "$SECRET_NAME" \
            --data-file=- \
            --replication-policy="automatic" \
            --project="$PROJECT_ID"
    fi
}

# Create secrets
create_secret "kestrel-openai-key" "OPENAI_API_KEY"
create_secret "kestrel-api-key" "KESTREL_API_KEY"
create_secret "kestrel-data-key" "KESTREL_DATA_KEY"

echo ""
echo "Done! Secrets configured in project $PROJECT_ID"
echo ""
echo "To verify:"
echo "  gcloud secrets list --project=$PROJECT_ID --filter='name:kestrel'"
