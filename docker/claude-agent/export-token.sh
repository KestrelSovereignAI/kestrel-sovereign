#!/bin/bash
# Export Claude OAuth token from macOS Keychain
# Run this on your local machine before starting containers

set -e

# Check if we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script only works on macOS (uses Keychain)"
    exit 1
fi

# Extract token from Keychain
TOKEN=$(security find-generic-password -s "Claude Code-credentials" -w 2>/dev/null | \
    python3 -c "import sys,json; print(json.load(sys.stdin)['claudeAiOauth']['accessToken'])" 2>/dev/null)

if [ -z "$TOKEN" ]; then
    echo "ERROR: Could not extract Claude OAuth token from Keychain"
    echo "Make sure you're logged into Claude Code locally"
    exit 1
fi

# Verify token format
if [[ ! "$TOKEN" =~ ^sk-ant-oat ]]; then
    echo "WARNING: Token doesn't look like an OAuth token (expected sk-ant-oat...)"
fi

# Output for sourcing
echo "export CLAUDE_CODE_OAUTH_TOKEN='$TOKEN'"

# Also check expiration
EXPIRES=$(security find-generic-password -s "Claude Code-credentials" -w 2>/dev/null | \
    python3 -c "
import sys,json,datetime
creds = json.load(sys.stdin)
expires_ms = creds['claudeAiOauth'].get('expiresAt', 0)
if expires_ms:
    expires = datetime.datetime.fromtimestamp(expires_ms / 1000)
    now = datetime.datetime.now()
    if expires < now:
        print(f'WARNING: Token expired at {expires}', file=sys.stderr)
        print('Claude Code may auto-refresh it, but container might not', file=sys.stderr)
    else:
        remaining = expires - now
        print(f'Token expires in {remaining}', file=sys.stderr)
" 2>&1)

echo "$EXPIRES" >&2
