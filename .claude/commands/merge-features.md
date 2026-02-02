Integrate multiple feature branches into main branch.

This command orchestrates the safe integration of completed features:

Process:
1. **Verify all tests pass** in each feature branch
2. **Check for conflicts** between feature branches
3. **Review integration points** (dependencies between features)
4. **Create integration plan** with merge order
5. **Merge features** one by one
6. **Run full test suite** after each merge
7. **Create integration commit** with comprehensive message

Safety checks:
- All tests must pass before merging
- No uncommitted changes in any worktree
- Main branch is up to date
- No merge conflicts

Merge order (recommended):
1. Constitution verification (foundational)
2. Privacy settings (core feature)
3. LLM service hardening (depends on privacy)

After merging:
- Run full integration test suite
- Verify no regressions
- Update PROJECT_STATUS.md
- Clean up merged worktrees (optional)

DO NOT merge if any tests fail.
DO NOT force-merge conflicts - resolve them properly.
