#!/bin/bash
# Create all feature branches for parallel development
# Usage: ./scripts/create_feature_branches.sh

set -e

echo "🌿 Creating feature branches for parallel development..."
echo ""

# Ensure we're on main and up to date
git checkout main
git pull origin main 2>/dev/null || echo "Note: Could not pull from origin"

# Define branches with descriptions
declare -A branches=(
    ["feature/selfie-e2e"]="Image generation working end-to-end"
    ["feature/mcp-dynamic"]="Dynamic MCP loading & self-expansion"
    ["feature/sovereignty-ui"]="Backup/restore UI polish"
    ["feature/gpu-on-demand"]="RunPod seamless brain upgrade"
    ["feature/privacy-visual"]="Privacy mode indicators & UX"
    ["feature/reflection"]="Nightly self-reflection"
    ["feature/wallet-real"]="Real Filecoin integration"
    ["feature/comprehensive-tests"]="Test coverage for all features"
)

# Create each branch
for branch in "${!branches[@]}"; do
    description="${branches[$branch]}"

    if git show-ref --verify --quiet "refs/heads/$branch" 2>/dev/null; then
        echo "⏭️  $branch already exists (skipping)"
    else
        echo "✨ Creating $branch - $description"
        git checkout -b "$branch" main
        git checkout main
    fi
done

echo ""
echo "✅ Feature branches ready!"
echo ""
echo "Available branches:"
git branch | grep feature/ | while read branch; do
    branch_name="${branch## }"
    desc="${branches[$branch_name]:-No description}"
    echo "  $branch_name - $desc"
done

echo ""
echo "Quick commands:"
echo "  git checkout feature/selfie-e2e      # Start working on selfies"
echo "  git checkout feature/mcp-dynamic     # Start working on MCP"
echo "  git checkout feature/comprehensive-tests  # Start writing tests"
echo ""
echo "Push branches to remote:"
echo "  git push -u origin feature/selfie-e2e"
