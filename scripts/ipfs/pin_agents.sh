#!/bin/bash
set -e

# Pin all agent database snapshots to the IPFS node
#
# Takes a snapshot of each agent's SQLite DB and adds it to IPFS,
# then pins it so it persists in GCS-backed storage.
#
# Usage:
#   ./scripts/ipfs/pin_agents.sh [ipfs-api-url]
#
# Default IPFS API: http://localhost:5001 (local node)
# For remote:       ./scripts/ipfs/pin_agents.sh http://<GCE-IP>:5001

IPFS_API="${1:-http://localhost:5001}"
AGENT_DATA_DIR="${KESTREL_AGENT_DATA:-$(dirname "$0")/../../agent_data}"
AGENT_DATA_DIR="$(cd "$AGENT_DATA_DIR" && pwd)"

echo "Pinning agent snapshots to IPFS"
echo "  API:  $IPFS_API"
echo "  Data: $AGENT_DATA_DIR"
echo ""

# Check IPFS node is reachable
if ! curl -s --connect-timeout 5 "${IPFS_API}/api/v0/id" >/dev/null 2>&1; then
    echo "ERROR: Cannot reach IPFS API at $IPFS_API"
    echo "  Local:  ipfs daemon must be running"
    echo "  Remote: check firewall and instance status"
    exit 1
fi

PEER_ID=$(curl -s "${IPFS_API}/api/v0/id" | python3 -c "import sys,json; print(json.load(sys.stdin)['ID'])" 2>/dev/null)
echo "  Peer:  $PEER_ID"
echo ""

pin_agent() {
    local agent_name="$1"
    local db_path="$2"

    if [ ! -f "$db_path" ]; then
        echo "  SKIP $agent_name — no database at $db_path"
        return
    fi

    # Create consistent snapshot using sqlite3 backup
    local tmp_snapshot
    tmp_snapshot=$(mktemp -t "ipfs-${agent_name}-XXXXXX.db")
    python3 -c "
import sqlite3, sys
src = sqlite3.connect('$db_path')
dst = sqlite3.connect('$tmp_snapshot')
src.backup(dst)
dst.close()
src.close()
" 2>/dev/null

    local size
    size=$(stat -f%z "$tmp_snapshot" 2>/dev/null || stat -c%s "$tmp_snapshot" 2>/dev/null)

    # Add to IPFS
    local result
    result=$(curl -s -X POST -F "file=@${tmp_snapshot};filename=${agent_name}/kestrel_prime.db" \
        "${IPFS_API}/api/v0/add?pin=true&quieter=true" 2>&1)

    local cid
    cid=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['Hash'])" 2>/dev/null)

    rm -f "$tmp_snapshot"

    if [ -n "$cid" ]; then
        local human_size
        human_size=$(echo "$size" | python3 -c "
import sys
s = int(sys.stdin.read())
for unit in ['B','KB','MB','GB']:
    if s < 1024: print(f'{s:.1f} {unit}'); break
    s /= 1024
")
        echo "  $agent_name: $cid ($human_size)"
    else
        echo "  $agent_name: FAILED — $result"
    fi
}

# Find and pin all agent databases
for agent_dir in "$AGENT_DATA_DIR"/*/; do
    agent_name=$(basename "$agent_dir")
    db_path="${agent_dir}kestrel_prime.db"
    pin_agent "$agent_name" "$db_path"
done

echo ""
echo "Done. View pins: curl -s ${IPFS_API}/api/v0/pin/ls | python3 -m json.tool"
