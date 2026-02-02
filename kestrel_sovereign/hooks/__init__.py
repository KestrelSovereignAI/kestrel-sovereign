"""
Kestrel Hooks System - Event-driven middleware for tool and agent calls.

Aligned with Claude Code's hooks pattern for:
- PreToolUse / PostToolUse - Before/after tool execution
- PreSubagentCall / PostSubagentCall - Before/after feature subagent invocation
- SessionStart / Stop - Agent lifecycle events
- UserPromptSubmit - When user sends a message

Usage:
    from kestrel_sovereign.hooks import HooksManager, Hook, HookEvent, HookInput, HookOutput

    # Create custom hook
    class MyHook(Hook):
        async def execute(self, input: HookInput) -> HookOutput:
            if should_block(input.tool_name):
                return HookOutput.deny("Blocked by policy")
            return HookOutput.allow()

    # Register with manager
    manager = HooksManager()
    manager.register(MyHook(name="my_hook", events=[HookEvent.PRE_TOOL_USE]))
"""

from kestrel_sovereign.hooks.base import (
    Hook,
    HookEvent,
    HookInput,
    HookOutput,
    PermissionDecision,
)
from kestrel_sovereign.hooks.manager import HooksManager

__all__ = [
    "Hook",
    "HookEvent",
    "HookInput",
    "HookOutput",
    "HooksManager",
    "PermissionDecision",
]
