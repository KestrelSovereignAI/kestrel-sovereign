#!/bin/bash
# DEPRECATED: Use 'kestrel start' instead.
# This script is a thin wrapper for backward compatibility.
echo "⚠️  DEPRECATED: start_kestrel.sh is deprecated. Use 'kestrel start' instead."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec uv run kestrel start "$@"
