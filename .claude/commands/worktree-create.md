Create a new git worktree for parallel development.

Usage: /worktree-create <feature-name>

Steps:
1. Generate a clean feature branch name from the provided name
2. Create worktree directory: `../kestrel-<feature-name>/`
3. Create git worktree with branch: `feature/<feature-name>`
4. List all active worktrees
5. Provide setup instructions for the new worktree

Feature name: $1

Example: /worktree-create llm-hardening
Creates: ../kestrel-llm-hardening on branch feature/llm-hardening
