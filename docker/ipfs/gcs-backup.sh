#!/bin/sh
# Sync IPFS blocks to GCS for durability
# Runs periodically via cron inside the container
#
# Uses gsutil rsync for efficient incremental backup.
# Only new/changed blocks are uploaded.

BUCKET="${GCS_BACKUP_BUCKET:-kestrel-ipfs}"
IPFS_PATH="${IPFS_PATH:-/data/ipfs}"

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "$(date -Iseconds) SKIP: GOOGLE_APPLICATION_CREDENTIALS not set"
    exit 0
fi

echo "$(date -Iseconds) Starting GCS backup to gs://${BUCKET}/blocks/"

# Sync blocks directory (the actual content-addressed data)
gsutil -m rsync -r "${IPFS_PATH}/blocks/" "gs://${BUCKET}/blocks/"

# Backup datastore and config
gsutil cp "${IPFS_PATH}/config" "gs://${BUCKET}/config"
gsutil cp "${IPFS_PATH}/version" "gs://${BUCKET}/version"

# Backup pins
ipfs pin ls --type=recursive > /tmp/pins.txt 2>/dev/null
gsutil cp /tmp/pins.txt "gs://${BUCKET}/pins.txt"
rm -f /tmp/pins.txt

echo "$(date -Iseconds) GCS backup complete"
