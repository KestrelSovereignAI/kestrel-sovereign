#!/bin/bash
# DEPRECATED: Use 'kestrel stop' instead.
# This script is a thin wrapper for backward compatibility.
echo "⚠️  DEPRECATED: stop_kestrel.sh is deprecated. Use 'kestrel stop' instead."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec uv run kestrel stop "$@"
