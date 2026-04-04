"""
Kestrel Hooks - Core Types (Claude Code Aligned).

Re-exports from kestrel_sdk.hooks.base for backward compatibility.
Feature packages should import from kestrel_sdk.hooks.base directly.
"""

# Re-export everything from kestrel_sdk
from kestrel_sdk.hooks.base import (  # noqa: F401
    Hook,
    HookEvent,
    HookInput,
    HookOutput,
    PermissionDecision,
)
