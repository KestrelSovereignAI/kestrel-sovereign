Fix a GitHub issue using codeagent in an isolated git worktree.

Usage: /fix-issue <issue-number>

Example: /fix-issue 50

## Process

1. **Fetch Issue Details**
   ```bash
   gh issue view $1 --repo ${GITHUB_REPO:-kestrel-sovereign/kestrel-sovereign} --json number,title,body,labels
   ```

2. **Extract Branch Name**
   Parse the issue body for "Suggested Branch" section to get the branch name.
   Default format: `fix/issue-$1` if not specified.

3. **Create Git Worktree**
   ```bash
   git worktree add ../kestrel-fix-$1 -b <branch-name>
   ```

4. **Launch Codeagent**
   Use the Task tool with subagent_type='codeagent' providing:
   - Full issue title and body as context
   - Working directory: `../kestrel-fix-$1`
   - Acceptance criteria from the issue
   - Files to modify from the issue

5. **Run Tests**
   ```bash
   cd ../kestrel-fix-$1 && uv run pytest -x --tb=short
   ```

6. **Create Pull Request** (if tests pass)
   ```bash
   cd ../kestrel-fix-$1 && gh pr create --title "<issue-title>" --body "Fixes #$1"
   ```

7. **Report Results**
   - PR URL if created
   - Test results summary
   - Any issues encountered

## Issue Number
$1

## Notes

- This command creates an isolated worktree to avoid conflicts with other work
- The codeagent will read the issue and implement the fix
- Tests must pass before PR creation
- Use `/test-worktrees` to verify all worktrees pass tests
- Use `/merge-features` to integrate completed branches
