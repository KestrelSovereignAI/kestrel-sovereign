#!/bin/bash
set -e

IPFS_PATH="${IPFS_PATH:-/home/ipfs/.ipfs}"

# Ensure IPFS directory exists and is owned by ipfs user
mkdir -p "$IPFS_PATH"
chown ipfs:ipfs "$IPFS_PATH"

# Initialize IPFS if needed
if [ ! -f "$IPFS_PATH/config" ]; then
    gosu ipfs ipfs init --profile=server
fi

# Run init scripts
for f in /container-init.d/*.sh; do
    [ -f "$f" ] && gosu ipfs bash "$f"
done

# Restore from GCS if repo is empty and backup exists
if [ -n "$GCS_BACKUP_BUCKET" ] && [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    BLOCK_COUNT=$(find "$IPFS_PATH/blocks" -type f 2>/dev/null | head -5 | wc -l)
    if [ "$BLOCK_COUNT" -lt 2 ]; then
        echo "Empty repo — attempting GCS restore from gs://${GCS_BACKUP_BUCKET}/"
        gsutil -m rsync -r "gs://${GCS_BACKUP_BUCKET}/blocks/" "$IPFS_PATH/blocks/" 2>/dev/null || true
        chown -R ipfs:ipfs "$IPFS_PATH/blocks" 2>/dev/null || true
    fi
fi

# Start cron for periodic GCS backup
cron

# Start IPFS daemon
exec gosu ipfs ipfs "$@"
