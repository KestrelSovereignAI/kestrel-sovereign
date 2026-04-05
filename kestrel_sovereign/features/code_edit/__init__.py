"""
Code Edit Feature - Self-modification with constitutional approval.

Enables the Kestrel Agent to edit its own source code with proper
security controls and approval flows.

Key Principles:
1. All code edits require explicit user approval
2. Changes are tracked via git commits
3. Edits use exact text matching (no regex) for safety
4. Server restart can be signaled after changes

Usage:
    !code-read <path>           Read a source file
    !code-edit <path>           Edit a source file (requires approval)
    !code-diff <path>           Show uncommitted changes
    !code-commit <message>      Commit staged changes
    !code-restart               Signal server restart

Security:
    - All edits go through SecurityFeature approval queue
    - Changes are logged and auditable
    - Git history provides full traceability
"""

from .feature import CodeEditFeature

__all__ = ["CodeEditFeature"]
