#!/bin/bash
# Smart Test Runner - Runs tests in phases with resume capability
# Usage: ./scripts/run_tests_smart.sh [--resume] [--phase PHASE] [--skip-unit] [--skip-integration] [--skip-e2e]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# State file to track progress
STATE_FILE="/tmp/kestrel_test_state.json"

# Parse arguments
RESUME=false
START_PHASE=""
SKIP_UNIT=false
SKIP_INTEGRATION=false
SKIP_E2E=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME=true
            shift
            ;;
        --phase)
            START_PHASE="$2"
            shift 2
            ;;
        --skip-unit)
            SKIP_UNIT=true
            shift
            ;;
        --skip-integration)
            SKIP_INTEGRATION=true
            shift
            ;;
        --skip-e2e)
            SKIP_E2E=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --resume           Resume from last failed test"
            echo "  --phase PHASE      Start from specific phase (unit|integration|e2e)"
            echo "  --skip-unit        Skip unit tests"
            echo "  --skip-integration Skip integration tests"
            echo "  --skip-e2e         Skip Playwright E2E tests"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Initialize or load state
init_state() {
    if [[ "$RESUME" == "true" && -f "$STATE_FILE" ]]; then
        echo -e "${YELLOW}Resuming from previous state...${NC}"
        cat "$STATE_FILE"
    else
        echo '{"phase": "unit", "last_failed": null, "completed_phases": []}' > "$STATE_FILE"
    fi
}

save_state() {
    local phase="$1"
    local last_failed="$2"
    local completed="$3"
    echo "{\"phase\": \"$phase\", \"last_failed\": $last_failed, \"completed_phases\": $completed}" > "$STATE_FILE"
}

get_state_value() {
    local key="$1"
    python3 -c "import json; print(json.load(open('$STATE_FILE')).get('$key', ''))"
}

# Run unit tests
run_unit_tests() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  PHASE 1: Unit Tests${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

    local last_failed=$(get_state_value "last_failed")
    local pytest_args="-x -v --tb=short"

    # If resuming with a specific failed test, start from there
    if [[ "$RESUME" == "true" && "$last_failed" != "null" && "$last_failed" != "" ]]; then
        # Extract just the test file if it's a unit test
        if [[ "$last_failed" == *"tests/unit/"* ]]; then
            pytest_args="$pytest_args --lf"  # --last-failed
            echo -e "${YELLOW}Resuming from last failed test...${NC}"
        fi
    fi

    if uv run pytest tests/unit/ $pytest_args; then
        echo -e "\n${GREEN}✓ Unit tests passed${NC}"
        return 0
    else
        local failed_test=$(uv run pytest tests/unit/ --collect-only -q 2>/dev/null | head -1)
        save_state "unit" "\"$failed_test\"" "[]"
        echo -e "\n${RED}✗ Unit tests failed${NC}"
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  PHASE 2: Integration Tests${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

    local pytest_args="-x -v --tb=short"

    if [[ "$RESUME" == "true" ]]; then
        pytest_args="$pytest_args --lf"
    fi

    if uv run pytest tests/integration/ $pytest_args; then
        echo -e "\n${GREEN}✓ Integration tests passed${NC}"
        return 0
    else
        save_state "integration" "\"integration\"" "[\"unit\"]"
        echo -e "\n${RED}✗ Integration tests failed${NC}"
        return 1
    fi
}

# Run Playwright E2E tests
run_e2e_tests() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  PHASE 3: Playwright E2E Tests${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

    # Check if server is running
    if ! curl -s http://localhost:8888/health > /dev/null 2>&1; then
        echo -e "${YELLOW}Starting Kestrel server...${NC}"
        uv run kestrel start &
        sleep 5
    fi

    local playwright_args="--project=chromium"

    if npx playwright test $playwright_args; then
        echo -e "\n${GREEN}✓ E2E tests passed${NC}"
        return 0
    else
        save_state "e2e" "\"e2e\"" "[\"unit\", \"integration\"]"
        echo -e "\n${RED}✗ E2E tests failed${NC}"
        return 1
    fi
}

# Main execution
main() {
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           Kestrel Smart Test Runner                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"

    init_state

    local current_phase=$(get_state_value "phase")

    # Override with command line phase if provided
    if [[ -n "$START_PHASE" ]]; then
        current_phase="$START_PHASE"
    fi

    local unit_passed=false
    local integration_passed=false
    local e2e_passed=false

    # Phase 1: Unit Tests
    if [[ "$SKIP_UNIT" != "true" && ("$current_phase" == "unit" || "$current_phase" == "") ]]; then
        if run_unit_tests; then
            unit_passed=true
            current_phase="integration"
        else
            echo -e "\n${RED}Unit tests failed. Run with --resume to continue from where you left off.${NC}"
            exit 1
        fi
    else
        unit_passed=true
        echo -e "${YELLOW}Skipping unit tests${NC}"
    fi

    # Phase 2: Integration Tests
    if [[ "$SKIP_INTEGRATION" != "true" && ("$current_phase" == "integration" || "$unit_passed" == "true") ]]; then
        if run_integration_tests; then
            integration_passed=true
            current_phase="e2e"
        else
            echo -e "\n${RED}Integration tests failed. Run with --resume to continue from where you left off.${NC}"
            exit 1
        fi
    else
        integration_passed=true
        echo -e "${YELLOW}Skipping integration tests${NC}"
    fi

    # Phase 3: E2E Tests
    if [[ "$SKIP_E2E" != "true" && ("$current_phase" == "e2e" || "$integration_passed" == "true") ]]; then
        if run_e2e_tests; then
            e2e_passed=true
        else
            echo -e "\n${RED}E2E tests failed. Run with --resume to continue from where you left off.${NC}"
            exit 1
        fi
    else
        e2e_passed=true
        echo -e "${YELLOW}Skipping E2E tests${NC}"
    fi

    # Summary
    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  TEST SUMMARY${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    [[ "$SKIP_UNIT" != "true" ]] && echo -e "  Unit Tests:        ${GREEN}✓ PASSED${NC}"
    [[ "$SKIP_INTEGRATION" != "true" ]] && echo -e "  Integration Tests: ${GREEN}✓ PASSED${NC}"
    [[ "$SKIP_E2E" != "true" ]] && echo -e "  E2E Tests:         ${GREEN}✓ PASSED${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

    # Clear state on success
    rm -f "$STATE_FILE"

    echo -e "\n${GREEN}All tests passed!${NC}"
}

main
