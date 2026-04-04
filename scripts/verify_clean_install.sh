#!/usr/bin/env bash
# verify_clean_install.sh — Clean install verification for kestrel-sovereign (Issue #548)
#
# Verifies the full install matrix:
#   Test 1: SDK only
#   Test 2: Core sovereign (no feature packages)
#   Test 3: Feature package from local source
#   Test 4: Feature package with SDK only (dev mode)
#   Test 5: Full stack (sovereign + wallet + intelligence)
#
# Usage:
#   ./scripts/verify_clean_install.sh           # Run all tests
#   ./scripts/verify_clean_install.sh 1         # Run only test 1
#   ./scripts/verify_clean_install.sh 1 3 5     # Run tests 1, 3, and 5
#
# Requirements:
#   - uv (https://docs.astral.sh/uv/)
#   - git
#
# Exit codes:
#   0 — All requested tests passed
#   1 — One or more tests failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TMPBASE="${TMPDIR:-/tmp}/kestrel-clean-install-$$"

PASS=0
FAIL=0
SKIP=0
RESULTS=()

cleanup() {
    rm -rf "$TMPBASE"
}
trap cleanup EXIT

log() { printf "\033[1;34m[verify]\033[0m %s\n" "$*"; }
pass() { printf "\033[1;32m[PASS]\033[0m %s\n" "$*"; PASS=$((PASS + 1)); RESULTS+=("PASS: $*"); }
fail() { printf "\033[1;31m[FAIL]\033[0m %s\n" "$*"; FAIL=$((FAIL + 1)); RESULTS+=("FAIL: $*"); }
skip() { printf "\033[1;33m[SKIP]\033[0m %s\n" "$*"; SKIP=$((SKIP + 1)); RESULTS+=("SKIP: $*"); }

# Determine which tests to run
if [ $# -eq 0 ]; then
    TESTS=(1 2 3 4 5)
else
    TESTS=("$@")
fi

should_run() {
    local n="$1"
    for t in "${TESTS[@]}"; do
        [ "$t" = "$n" ] && return 0
    done
    return 1
}

mkdir -p "$TMPBASE"

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: SDK only
# ─────────────────────────────────────────────────────────────────────────────
if should_run 1; then
    log "Test 1: SDK only install"
    TEST1_DIR="$TMPBASE/test1"
    mkdir -p "$TEST1_DIR"

    if uv venv "$TEST1_DIR/.venv" --quiet 2>/dev/null && \
       VIRTUAL_ENV="$TEST1_DIR/.venv" "$TEST1_DIR/.venv/bin/pip" install --quiet "$REPO_ROOT/sdk" 2>/dev/null; then
        if "$TEST1_DIR/.venv/bin/python" -c "from kestrel_sdk.features.base import Feature; print('SDK OK')" 2>/dev/null; then
            pass "Test 1: SDK only — import kestrel_sdk.features.base.Feature"
        else
            fail "Test 1: SDK only — import failed"
        fi
    else
        fail "Test 1: SDK only — install failed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Core sovereign (no feature packages)
# ─────────────────────────────────────────────────────────────────────────────
if should_run 2; then
    log "Test 2: Core sovereign install (no feature packages)"
    TEST2_DIR="$TMPBASE/test2"
    mkdir -p "$TEST2_DIR"

    if uv venv "$TEST2_DIR/.venv" --quiet 2>/dev/null && \
       VIRTUAL_ENV="$TEST2_DIR/.venv" "$TEST2_DIR/.venv/bin/pip" install --quiet "$REPO_ROOT" 2>/dev/null; then

        # Test import
        if "$TEST2_DIR/.venv/bin/python" -c "from kestrel_sovereign.features.base import Feature; print('Sovereign OK')" 2>/dev/null; then
            pass "Test 2: Core sovereign — import kestrel_sovereign.features.base.Feature"
        else
            fail "Test 2: Core sovereign — import failed"
        fi

        # Test health endpoint (start server briefly)
        AGENT_DIR="$TEST2_DIR/agent_data"
        mkdir -p "$AGENT_DIR"
        PORT=18548
        export KESTREL_DB_PATH="$AGENT_DIR"

        "$TEST2_DIR/.venv/bin/python" -c "
from kestrel_sovereign.inception_service import create_kestrel_identity
import os
create_kestrel_identity('$AGENT_DIR', os.path.join('$REPO_ROOT', 'docs', 'principles', 'KESTREL_CONSTITUTION.md'))
" 2>/dev/null || true

        "$TEST2_DIR/.venv/bin/uvicorn" server:app --host 127.0.0.1 --port $PORT \
            --app-dir "$REPO_ROOT" &
        SERVER_PID=$!
        sleep 3

        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            pass "Test 2: Core sovereign — /health endpoint responds"
        else
            fail "Test 2: Core sovereign — /health endpoint not responding"
        fi

        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        unset KESTREL_DB_PATH
    else
        fail "Test 2: Core sovereign — install failed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Feature package install (wallet)
# ─────────────────────────────────────────────────────────────────────────────
if should_run 3; then
    log "Test 3: Feature package install (wallet)"
    TEST3_DIR="$TMPBASE/test3"
    mkdir -p "$TEST3_DIR"

    if uv venv "$TEST3_DIR/.venv" --quiet 2>/dev/null && \
       VIRTUAL_ENV="$TEST3_DIR/.venv" "$TEST3_DIR/.venv/bin/pip" install --quiet "$REPO_ROOT" 2>/dev/null && \
       VIRTUAL_ENV="$TEST3_DIR/.venv" "$TEST3_DIR/.venv/bin/pip" install --quiet "$REPO_ROOT/kestrel_feature_wallet" 2>/dev/null; then
        if "$TEST3_DIR/.venv/bin/python" -c "from kestrel_feature_wallet import WalletFeature; print('Wallet OK')" 2>/dev/null; then
            pass "Test 3: Feature package — import WalletFeature"
        else
            fail "Test 3: Feature package — import failed"
        fi
    else
        fail "Test 3: Feature package — install failed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Feature package with SDK only (dev mode)
# ─────────────────────────────────────────────────────────────────────────────
if should_run 4; then
    log "Test 4: Feature package with SDK only (dev mode)"
    TEST4_DIR="$TMPBASE/test4"
    mkdir -p "$TEST4_DIR"

    if uv venv "$TEST4_DIR/.venv" --quiet 2>/dev/null && \
       VIRTUAL_ENV="$TEST4_DIR/.venv" "$TEST4_DIR/.venv/bin/pip" install --quiet "$REPO_ROOT/sdk" 2>/dev/null; then

        # In dev mode, install the feature package in editable mode.
        # The wallet requires kestrel-sovereign as a dependency, but in pure SDK-dev
        # mode we install with --no-deps to test that the SDK interface is sufficient.
        if VIRTUAL_ENV="$TEST4_DIR/.venv" "$TEST4_DIR/.venv/bin/pip" install --quiet --no-deps \
             -e "$REPO_ROOT/kestrel_feature_wallet" 2>/dev/null; then
            if "$TEST4_DIR/.venv/bin/python" -c "
from kestrel_sdk.features.base import Feature
from kestrel_feature_wallet.wallet_feature import WalletFeature
assert issubclass(WalletFeature, Feature), 'WalletFeature must be a Feature subclass'
print('Dev mode OK')
" 2>/dev/null; then
                pass "Test 4: SDK + feature dev mode — WalletFeature is Feature subclass"
            else
                fail "Test 4: SDK + feature dev mode — import or subclass check failed"
            fi
        else
            fail "Test 4: SDK + feature dev mode — editable install failed"
        fi
    else
        fail "Test 4: SDK + feature dev mode — SDK install failed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Full stack (sovereign + wallet + intelligence)
# ─────────────────────────────────────────────────────────────────────────────
if should_run 5; then
    log "Test 5: Full stack install (sovereign + wallet + intelligence)"
    TEST5_DIR="$TMPBASE/test5"
    mkdir -p "$TEST5_DIR"

    if uv venv "$TEST5_DIR/.venv" --quiet 2>/dev/null && \
       VIRTUAL_ENV="$TEST5_DIR/.venv" "$TEST5_DIR/.venv/bin/pip" install --quiet "$REPO_ROOT" 2>/dev/null && \
       VIRTUAL_ENV="$TEST5_DIR/.venv" "$TEST5_DIR/.venv/bin/pip" install --quiet "$REPO_ROOT/kestrel_feature_wallet" 2>/dev/null && \
       VIRTUAL_ENV="$TEST5_DIR/.venv" "$TEST5_DIR/.venv/bin/pip" install --quiet "$REPO_ROOT/kestrel-feature-intelligence" 2>/dev/null; then

        # Verify all imports work
        if "$TEST5_DIR/.venv/bin/python" -c "
from kestrel_sovereign.features.base import Feature
from kestrel_feature_wallet import WalletFeature
from kestrel_feature_intelligence import ReflectionFeature, CouncilFeature
print('Full stack OK')
" 2>/dev/null; then
            pass "Test 5: Full stack — all packages importable"
        else
            fail "Test 5: Full stack — import failed"
        fi

        # Verify entry_point discovery finds the features
        if "$TEST5_DIR/.venv/bin/python" -c "
import importlib.metadata
eps = importlib.metadata.entry_points()
group = eps.select(group='kestrel_sovereign.features')
names = {ep.name for ep in group}
assert 'WalletFeature' in names, f'WalletFeature not in entry_points: {names}'
assert 'ReflectionFeature' in names, f'ReflectionFeature not in entry_points: {names}'
assert 'CouncilFeature' in names, f'CouncilFeature not in entry_points: {names}'
print(f'Entry points discovered: {sorted(names)}')
" 2>/dev/null; then
            pass "Test 5: Full stack — entry_point discovery finds all features"
        else
            fail "Test 5: Full stack — entry_point discovery failed"
        fi
    else
        fail "Test 5: Full stack — install failed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Clean Install Verification Summary"
echo "═══════════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo "───────────────────────────────────────────────────────────────"
echo "  Passed: $PASS  |  Failed: $FAIL  |  Skipped: $SKIP"
echo "═══════════════════════════════════════════════════════════════"

[ "$FAIL" -eq 0 ]
