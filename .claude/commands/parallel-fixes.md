Launch multiple codeagent subagents to fix GitHub issues in parallel.

Usage: /parallel-fixes <comma-separated-issue-numbers>

Example: /parallel-fixes 50,53,55,57,58

## IMPORTANT

Only use for issues that **won't conflict** with each other (Group A issues).
Check the code review document for conflict groupings:
- **Group A**: Safe for parallel (different files)
- **Group B**: Sequential within group (shared modules)
- **Group C**: Hub files (one at a time)

## Process

### 1. Parse Issue Numbers
Split the argument by comma to get list of issue numbers.

### 2. Fetch All Issues
For each issue number:
```bash
gh issue view <num> --repo ${GITHUB_REPO:-kestrel-sovereign/kestrel-sovereign} --json number,title,body,labels
```

### 3. Create All Worktrees
For each issue:
```bash
git worktree add ../kestrel-fix-<num> -b <branch-from-issue>
```

### 4. Launch Codeagents in Parallel
**CRITICAL**: Send a SINGLE message with multiple Task tool calls.
This ensures true parallel execution.

Each codeagent receives:
- Issue number, title, and body
- Working directory (the worktree path)
- Acceptance criteria
- Files to modify

Example (for 3 issues):
```
[Task tool call 1: codeagent for issue 50 in ../kestrel-fix-50]
[Task tool call 2: codeagent for issue 53 in ../kestrel-fix-53]
[Task tool call 3: codeagent for issue 55 in ../kestrel-fix-55]
```

### 5. Monitor Progress
Use `/tasks` command to check agent status.
Wait for all agents to complete.

### 6. Validate Results
Run `/test-worktrees` to verify all worktrees pass tests.

### 7. Report Results
For each issue:
- Agent completion status
- Test results
- PR URL (if created)
- Any errors encountered

## Issue Numbers
$1

## Wave 1 Group A Issues (Safe for Parallel)
| Issue | Title | Files |
|-------|-------|-------|
| 50 | Hardcoded Credentials | docker-compose, scripts, tests |
| 53 | Decompose simpletuner_api.py | docker/simpletuner_api.py |
| 55 | Decompose Compute Managers | features/vastai/*, gcp/*, runpod/* |
| 57 | Eldercare TODOs | kestrel/endpoints/eldercare.py |
| 58 | Replicate Workarounds | features/training/adapters/replicate_adapter.py |
| 60 | Blockchain TODO | kestrel/endpoints/proxy_deposits.py |
| 61 | NotImplementedError Stubs | Multiple files |
| 62 | Naming Inconsistencies | storage/db/__init__.py |

## Notes

- This creates N worktrees and N subagents
- Ensure sufficient disk space (~500MB per worktree)
- Each agent runs independently
- Use `/merge-features` after all PRs are approved
- Reference: docs/code_reviews/CODE_REVIEW_01_03_2026.md
