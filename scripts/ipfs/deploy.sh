#!/bin/bash
set -e

# Deploy Kestrel IPFS node to GCE
#
# Creates a small VM running Kubo with GCS-backed block storage.
# Blocks are stored in gs://kestrel-ipfs, so the VM itself is stateless
# and can be recreated without data loss.
#
# Usage:
#   ./scripts/ipfs/deploy.sh [create|update|delete|status|ssh]
#
# Requires:
#   - GCP_PROJECT_ID env var
#   - Image built and pushed: ./scripts/ipfs/build.sh
#   - Service account with Storage Admin on kestrel-ipfs bucket
#
# Cost estimate:
#   e2-small (2 vCPU, 2GB RAM): ~$13/mo on-demand, ~$4/mo spot
#   10GB boot disk: ~$0.40/mo
#   GCS storage: ~$0.02/GB/mo (Standard)

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
INSTANCE_NAME="kestrel-ipfs"
ZONE="us-central1-a"
REGION="us-central1"
MACHINE_TYPE="e2-small"
IMAGE_NAME="kestrel-ipfs-gcs"
GCS_BUCKET="kestrel-ipfs"
NETWORK_TAG="kestrel-ipfs"

# Service account for IPFS node (needs Storage Admin on kestrel-ipfs bucket)
# Uses the default compute service account if not set
SA_EMAIL="${IPFS_SERVICE_ACCOUNT:-$(gcloud iam service-accounts list \
    --project="$PROJECT_ID" \
    --filter="email:compute@developer.gserviceaccount.com" \
    --format='value(email)' 2>/dev/null)}"

ACTION="${1:-status}"

print_info() {
    echo "Kestrel IPFS Node (GCE)"
    echo "  Project:  $PROJECT_ID"
    echo "  Instance: $INSTANCE_NAME"
    echo "  Zone:     $ZONE"
    echo "  Machine:  $MACHINE_TYPE"
    echo "  Bucket:   gs://$GCS_BUCKET"
    echo ""
}

create_firewall_rules() {
    echo "Setting up firewall rules..."

    # Swarm port (public — needed for IPFS peering)
    gcloud compute firewall-rules describe "${NETWORK_TAG}-swarm" \
        --project="$PROJECT_ID" &>/dev/null || \
    gcloud compute firewall-rules create "${NETWORK_TAG}-swarm" \
        --project="$PROJECT_ID" \
        --direction=INGRESS \
        --action=ALLOW \
        --rules=tcp:4001,udp:4001 \
        --source-ranges=0.0.0.0/0 \
        --target-tags="$NETWORK_TAG" \
        --description="IPFS swarm (libp2p peering)" \
        --quiet

    # Gateway port (public — HTTP access to IPFS content)
    gcloud compute firewall-rules describe "${NETWORK_TAG}-gateway" \
        --project="$PROJECT_ID" &>/dev/null || \
    gcloud compute firewall-rules create "${NETWORK_TAG}-gateway" \
        --project="$PROJECT_ID" \
        --direction=INGRESS \
        --action=ALLOW \
        --rules=tcp:8080 \
        --source-ranges=0.0.0.0/0 \
        --target-tags="$NETWORK_TAG" \
        --description="IPFS gateway (HTTP)" \
        --quiet

    # API port (restricted — only from our other GCE/Cloud Run services)
    gcloud compute firewall-rules describe "${NETWORK_TAG}-api" \
        --project="$PROJECT_ID" &>/dev/null || \
    gcloud compute firewall-rules create "${NETWORK_TAG}-api" \
        --project="$PROJECT_ID" \
        --direction=INGRESS \
        --action=ALLOW \
        --rules=tcp:5001 \
        --source-ranges=10.128.0.0/9,35.235.240.0/20 \
        --target-tags="$NETWORK_TAG" \
        --description="IPFS API (internal + IAP only)" \
        --quiet

    echo "  Firewall rules ready."
}

do_create() {
    print_info
    create_firewall_rules

    echo "Creating instance..."

    gcloud compute instances create-with-container "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --boot-disk-size=10GB \
        --boot-disk-type=pd-standard \
        --tags="$NETWORK_TAG" \
        --service-account="$SA_EMAIL" \
        --scopes=storage-full \
        --container-image="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" \
        --container-env="KUBO_GCS_BUCKET=${GCS_BUCKET}" \
        --container-mount-host-path=host-path=/var/ipfs,mount-path=/home/ipfs/.ipfs \
        --metadata=google-logging-enabled=true \
        --quiet

    EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

    echo ""
    echo "Instance created!"
    echo ""
    echo "  External IP: $EXTERNAL_IP"
    echo "  Swarm:       /ip4/${EXTERNAL_IP}/tcp/4001"
    echo "  Gateway:     http://${EXTERNAL_IP}:8080"
    echo "  API:         http://${EXTERNAL_IP}:5001 (internal only)"
    echo ""
    echo "  IPFS peer ID (wait ~60s for startup, then):"
    echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- docker exec \$(docker ps -q) ipfs id -f='<id>'"
    echo ""
    echo "  Pin a file:"
    echo "    curl -X POST 'http://${EXTERNAL_IP}:5001/api/v0/pin/add?arg=<CID>'"
}

do_update() {
    print_info
    echo "Updating container image..."

    gcloud compute instances update-container "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --container-image="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" \
        --container-env="KUBO_GCS_BUCKET=${GCS_BUCKET}" \
        --quiet

    echo "Updated! Instance will pull new image on next restart."
    echo "  Restart now: gcloud compute instances reset $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
}

do_delete() {
    print_info
    echo "WARNING: This will delete the IPFS VM instance."
    echo "  GCS blocks in gs://$GCS_BUCKET are NOT affected."
    echo ""
    read -p "Delete $INSTANCE_NAME? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud compute instances delete "$INSTANCE_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --quiet
        echo "Deleted."
    else
        echo "Cancelled."
    fi
}

do_status() {
    print_info

    if gcloud compute instances describe "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" &>/dev/null; then

        STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --format='value(status)')

        EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

        echo "  Status: $STATUS"
        echo "  IP:     $EXTERNAL_IP"
        echo ""

        if [ "$STATUS" = "RUNNING" ] && [ -n "$EXTERNAL_IP" ]; then
            echo "  Gateway check:"
            HTTP_CODE=$(curl -s --connect-timeout 5 --max-time 10 -o /dev/null -w "%{http_code}" "http://${EXTERNAL_IP}:8080/ipfs/QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn" 2>/dev/null || echo "timeout")
            echo "    http://${EXTERNAL_IP}:8080 => HTTP $HTTP_CODE"
        fi
    else
        echo "  Instance not found. Create with: $0 create"
    fi
}

do_ssh() {
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE"
}

case "$ACTION" in
    create)  do_create ;;
    update)  do_update ;;
    delete)  do_delete ;;
    status)  do_status ;;
    ssh)     do_ssh ;;
    *)
        echo "Usage: $0 [create|update|delete|status|ssh]"
        echo ""
        echo "  create  - Create IPFS VM instance"
        echo "  update  - Update container image"
        echo "  delete  - Delete VM (GCS blocks preserved)"
        echo "  status  - Check instance status"
        echo "  ssh     - SSH into the VM"
        exit 1
        ;;
esac
