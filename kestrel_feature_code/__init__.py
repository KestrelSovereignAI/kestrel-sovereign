"""
Kestrel Feature Code — Agent self-modification with constitutional approval.

Extracted from kestrel-sovereign as a standalone feature package.

Enables the Kestrel Agent to edit its own source code with proper
security controls and approval flows. Sovereign-only: only agents that
own their own codebase should use this feature.

Key Principles:
1. All code edits require explicit user approval
2. Changes are tracked via git commits
3. Edits use exact text matching (no regex) for safety
4. Server restart can be signaled after changes

Tools:
    !code-read <path>           Read a source file
    !code-search <pattern>      Search for text in codebase
    !code-edit <path>           Edit a source file (requires approval)
    !code-diff <path>           Show uncommitted changes
    !code-commit <message>      Commit staged changes (requires approval)
    !code-restart               Signal server restart (requires approval)
    !code-test [path]           Run pytest tests
    !code-lint [path]           Run ruff linter
    !code-logs                  View recent application logs
    !code-rollback [commit]     Rollback to previous commit (requires approval)
"""

from .feature import CodeEditFeature

__all__ = ["CodeEditFeature"]
