Run tests across all active git worktrees in parallel.

This command:
1. Lists all active git worktrees
2. For each worktree, runs pytest with fail-fast (-x)
3. Collects test results from each worktree
4. Generates consolidated test report

Test command used: `pytest -x tests/ -v --tb=short`

Output format:
```
Worktree Test Results
=====================

📁 ./ (main)
   ✅ 45 passed, 0 failed

📁 ./-llm-service (feature/llm-service-hardening)
   ✅ 32 passed, 0 failed

📁 ./-verify-constitution (feature/verify-constitution-inception)
   ⚠️  8 passed, 1 failed
   Failed: tests/test_genesis_audit.py::test_audit_blocks_corruption

📁 ./-privacy-settings (feature/privacy-settings-ui)
   ✅ 15 passed, 0 failed

Overall: 100/101 tests passed (99%)
```

Stop at first failing worktree if any test fails.
