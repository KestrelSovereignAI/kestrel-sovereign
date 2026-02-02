#!/bin/bash
#
# Kestrel Public Release Audit Script
#
# This script searches for sensitive content that should not be in a public release.
# Run this on the staging directory BEFORE committing to the public repo.
#
# Usage:
#     ./scripts/audit_sensitive_content.sh [directory]
#
# If no directory is specified, defaults to ~/kestrel-public
#
# License: Apache 2.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directory to audit
AUDIT_DIR="${1:-$HOME/kestrel-public}"

echo "========================================"
echo "Kestrel Public Release Security Audit"
echo "========================================"
echo ""
echo "Auditing: $AUDIT_DIR"
echo ""

if [ ! -d "$AUDIT_DIR" ]; then
    echo -e "${RED}ERROR: Directory $AUDIT_DIR does not exist${NC}"
    exit 1
fi

cd "$AUDIT_DIR"

FOUND_ISSUES=0

# Function to search for a pattern
search_pattern() {
    local pattern="$1"
    local description="$2"
    local severity="$3"  # CRITICAL, WARNING, INFO

    echo -n "Checking: $description... "

    # Use grep with extended regex, case insensitive where appropriate
    local results=$(grep -rIl "$pattern" . 2>/dev/null || true)

    if [ -n "$results" ]; then
        if [ "$severity" = "CRITICAL" ]; then
            echo -e "${RED}FOUND - CRITICAL${NC}"
            FOUND_ISSUES=$((FOUND_ISSUES + 1))
        elif [ "$severity" = "WARNING" ]; then
            echo -e "${YELLOW}FOUND - WARNING${NC}"
        else
            echo -e "${YELLOW}FOUND - INFO${NC}"
        fi
        echo "  Files:"
        echo "$results" | sed 's/^/    /'
        echo ""
    else
        echo -e "${GREEN}OK${NC}"
    fi
}

echo "----------------------------------------"
echo "CRITICAL: Personal Identity"
echo "----------------------------------------"
# Add your own patterns here if needed
# search_pattern "your-email" "Personal email" "CRITICAL"

echo ""
echo "----------------------------------------"
echo "CRITICAL: API Keys & Secrets"
echo "----------------------------------------"
search_pattern "sk-[a-zA-Z0-9]" "OpenAI/Anthropic API keys" "CRITICAL"
search_pattern "sk-or-v1-" "OpenRouter API keys" "CRITICAL"
search_pattern "sk-ant-" "Anthropic API keys" "CRITICAL"
search_pattern "sk-proj-" "OpenAI project keys" "CRITICAL"
search_pattern "rpa_[A-Za-z0-9]" "RunPod API keys" "CRITICAL"
search_pattern "xai-[A-Za-z0-9]" "xAI API keys" "CRITICAL"
search_pattern "r8_[A-Za-z0-9]" "Replicate API tokens" "CRITICAL"
search_pattern "hf_[A-Za-z0-9]" "HuggingFace tokens" "CRITICAL"
search_pattern "tvly-" "Tavily API keys" "CRITICAL"
search_pattern "github_pat_" "GitHub PATs" "CRITICAL"
search_pattern "AIzaSy" "Google API keys" "CRITICAL"
search_pattern "GOCSPX-" "Google OAuth secrets" "CRITICAL"

echo ""
echo "----------------------------------------"
echo "CRITICAL: GCP Infrastructure"
echo "----------------------------------------"
search_pattern "YOUR_PROJECT_ID" "GCP Project ID" "CRITICAL"
search_pattern "kestrel-agent-admin@" "Service account" "CRITICAL"
search_pattern "523805591861" "GCP project number" "CRITICAL"
search_pattern "\.iam\.gserviceaccount\.com" "Service accounts" "WARNING"

echo ""
echo "----------------------------------------"
echo "CRITICAL: Absolute Paths"
echo "----------------------------------------"
search_pattern "/Volumes/data" "Mac absolute paths" "CRITICAL"
# Add your own home directory pattern if needed
# search_pattern "/Users/yourname" "Personal home directory" "CRITICAL"

echo ""
echo "----------------------------------------"
echo "WARNING: Platform References"
echo "----------------------------------------"
search_pattern "frinz" "Legacy platform references" "WARNING"

echo ""
echo "----------------------------------------"
echo "WARNING: Cloud URLs"
echo "----------------------------------------"
search_pattern "\.run\.app" "Cloud Run URLs" "WARNING"
search_pattern "7jpbsywhdq" "Specific Cloud Run service" "CRITICAL"

echo ""
echo "----------------------------------------"
echo "INFO: Sensitive File Patterns"
echo "----------------------------------------"

# Check for files that shouldn't exist
echo -n "Checking: .env files with real values... "
if find . -name ".env" -o -name ".env.production" -o -name ".env.local" | grep -q .; then
    echo -e "${RED}FOUND - CRITICAL${NC}"
    find . -name ".env" -o -name ".env.production" -o -name ".env.local"
    FOUND_ISSUES=$((FOUND_ISSUES + 1))
else
    echo -e "${GREEN}OK${NC}"
fi

echo -n "Checking: Credential files... "
if find . -name "*.json" -path "*/credentials/*" | grep -q .; then
    echo -e "${RED}FOUND - CRITICAL${NC}"
    find . -name "*.json" -path "*/credentials/*"
    FOUND_ISSUES=$((FOUND_ISSUES + 1))
else
    echo -e "${GREEN}OK${NC}"
fi

echo -n "Checking: Database files... "
if find . -name "*.db" -o -name "*.capsule" -o -name "*.sqlite" | grep -q .; then
    echo -e "${RED}FOUND - CRITICAL${NC}"
    find . -name "*.db" -o -name "*.capsule" -o -name "*.sqlite"
    FOUND_ISSUES=$((FOUND_ISSUES + 1))
else
    echo -e "${GREEN}OK${NC}"
fi

echo -n "Checking: Private directories... "
for dir in "docs/business" "docs/legal" "docs/outreach" "gabriel_workspace"; do
    if [ -d "$dir" ]; then
        echo -e "${RED}FOUND - CRITICAL${NC}"
        echo "  $dir exists and should be removed"
        FOUND_ISSUES=$((FOUND_ISSUES + 1))
    fi
done
echo -e "${GREEN}OK${NC}"

echo ""
echo "========================================"
echo "AUDIT SUMMARY"
echo "========================================"

if [ $FOUND_ISSUES -gt 0 ]; then
    echo -e "${RED}FAILED: Found $FOUND_ISSUES critical issues${NC}"
    echo ""
    echo "Please fix the issues above before making the repository public."
    exit 1
else
    echo -e "${GREEN}PASSED: No critical issues found${NC}"
    echo ""
    echo "Review any WARNING items above, then proceed with:"
    echo "  1. Add LICENSE, CONTRIBUTING.md, etc."
    echo "  2. Initialize git and commit"
    echo "  3. Push to GitHub (after LLC is ready)"
    exit 0
fi
