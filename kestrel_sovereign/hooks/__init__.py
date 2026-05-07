"""
Kestrel Hooks — HooksManager (framework implementation).

The interface types — ``Hook``, ``HookEvent``, ``HookInput``,
``HookOutput``, ``PermissionDecision`` — live in the SDK and must be
imported from there directly:

    from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput

The ``HooksManager`` class is kestrel-sovereign's internal in-memory
dispatcher / registry for hooks. It's a framework implementation
detail, not a contract feature packages should bind to.

Usage:
    from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
    from kestrel_sovereign.hooks import HooksManager

    class MyHook(Hook):
        async def execute(self, input: HookInput) -> HookOutput:
            if should_block(input.tool_name):
                return HookOutput.deny("Blocked by policy")
            return HookOutput.allow()

    manager = HooksManager()
    manager.register(MyHook(name="my_hook", events=[HookEvent.PRE_TOOL_USE]))
"""

from kestrel_sovereign.hooks.manager import HooksManager

__all__ = ["HooksManager"]
