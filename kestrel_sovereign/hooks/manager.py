"""
Kestrel Hooks Manager - Central registration and execution.

The HooksManager coordinates hook registration and execution, ensuring
hooks are called in priority order with proper timeout handling.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from kestrel_sovereign.hooks.base import (
    Hook,
    HookEvent,
    HookInput,
    HookOutput,
    PermissionDecision,
)

logger = logging.getLogger(__name__)


class HooksManager:
    """
    Central manager for hook registration and execution.

    Features:
    - Priority-based execution order (lower priority = earlier execution)
    - Per-hook timeout handling
    - Tool name regex matching
    - Chain interruption on DENY or ASK decisions

    Example:
        manager = HooksManager()
        manager.register(SecurityHook())
        manager.register(LoggingHook())

        # Execute pre-tool hooks
        result = await manager.execute_hooks(
            HookEvent.PRE_TOOL_USE,
            HookInput(
                session_id="session123",
                hook_event_name="PreToolUse",
                tool_name="web_search",
                tool_input={"query": "weather"}
            )
        )

        if not result.continue_execution:
            # Hook blocked execution
            logger.warning(f"Blocked: {result.stop_reason}")
    """

    def __init__(self):
        """Initialize the hooks manager with empty hook registry."""
        self._hooks: Dict[HookEvent, List[Hook]] = {e: [] for e in HookEvent}

    def register(self, hook: Hook) -> None:
        """
        Register a hook for its specified events.

        Args:
            hook: The Hook instance to register

        Note:
            Hooks are automatically sorted by priority after registration.
        """
        for event in hook.events:
            if hook not in self._hooks[event]:
                self._hooks[event].append(hook)
                # Sort by priority (lower = earlier)
                self._hooks[event].sort(key=lambda h: h.priority)
                logger.info(
                    f"Registered hook '{hook.name}' for event {event.value} "
                    f"(priority {hook.priority})"
                )

    def unregister(self, hook: Hook) -> None:
        """
        Unregister a hook from all events.

        Args:
            hook: The Hook instance to unregister
        """
        for event in hook.events:
            if hook in self._hooks[event]:
                self._hooks[event].remove(hook)
                logger.info(f"Unregistered hook '{hook.name}' from event {event.value}")

    def unregister_by_name(self, name: str) -> bool:
        """
        Unregister a hook by its name.

        Args:
            name: The name of the hook to unregister

        Returns:
            True if a hook was found and removed, False otherwise
        """
        found = False
        for event in HookEvent:
            hooks_to_remove = [h for h in self._hooks[event] if h.name == name]
            for hook in hooks_to_remove:
                self._hooks[event].remove(hook)
                found = True
                logger.info(f"Unregistered hook '{name}' from event {event.value}")
        return found

    def get_hooks(self, event: HookEvent) -> List[Hook]:
        """
        Get all registered hooks for an event.

        Args:
            event: The hook event type

        Returns:
            List of hooks registered for the event (sorted by priority)
        """
        return list(self._hooks[event])

    def get_enabled_hooks(self, event: HookEvent) -> List[Hook]:
        """
        Get all enabled hooks for an event.

        Args:
            event: The hook event type

        Returns:
            List of enabled hooks (sorted by priority)
        """
        return [h for h in self._hooks[event] if h.enabled]

    def set_hook_enabled(self, name: str, enabled: bool) -> bool:
        """
        Enable or disable a hook by name.

        Args:
            name: The name of the hook
            enabled: Whether to enable or disable

        Returns:
            True if a hook was found, False otherwise
        """
        found = False
        for event in HookEvent:
            for hook in self._hooks[event]:
                if hook.name == name:
                    hook.enabled = enabled
                    found = True
                    logger.info(f"Hook '{name}' {'enabled' if enabled else 'disabled'}")
        return found

    async def execute_hooks(
        self,
        event: HookEvent,
        input: HookInput,
    ) -> HookOutput:
        """
        Execute matching hooks in priority order.

        Hooks are executed sequentially in priority order. The chain is
        interrupted if a hook returns DENY or ASK decision.

        Args:
            event: The hook event type
            input: HookInput containing context for the hooks

        Returns:
            HookOutput from the chain. Returns ALLOW if no hooks block.

        Note:
            - DENY or ASK stops the chain and returns immediately
            - MODIFY updates the input for subsequent hooks
            - Timeout causes the hook to be skipped with a warning
            - Exceptions cause the hook to be skipped with an error log
        """
        matching_hooks = self._get_matching_hooks(event, input.tool_name)

        if not matching_hooks:
            return HookOutput.allow()

        logger.debug(
            f"Executing {len(matching_hooks)} hooks for {event.value}"
            f"{f' (tool: {input.tool_name})' if input.tool_name else ''}"
        )

        for hook in matching_hooks:
            try:
                output = await asyncio.wait_for(
                    hook.execute(input),
                    timeout=hook.timeout
                )

                logger.debug(
                    f"Hook '{hook.name}' returned: "
                    f"decision={output.permission_decision}, "
                    f"continue={output.continue_execution}"
                )

                # DENY or ASK stops the chain
                if output.permission_decision in (
                    PermissionDecision.DENY,
                    PermissionDecision.ASK
                ):
                    return output

                # MODIFY updates input for next hook
                if output.updated_input:
                    input.tool_input = output.updated_input

            except asyncio.TimeoutError:
                logger.warning(
                    f"Hook '{hook.name}' timed out after {hook.timeout}s, skipping"
                )
                continue

            except Exception as e:
                logger.error(f"Hook '{hook.name}' failed with error: {e}", exc_info=True)
                continue

        return HookOutput.allow()

    async def execute_hooks_parallel(
        self,
        event: HookEvent,
        input: HookInput,
    ) -> List[HookOutput]:
        """
        Execute all matching hooks in parallel (for post-hooks).

        Unlike execute_hooks(), this runs all hooks concurrently and
        returns all outputs. This is useful for post-execution hooks
        that don't need to affect the execution flow.

        Args:
            event: The hook event type
            input: HookInput containing context for the hooks

        Returns:
            List of HookOutput from all executed hooks
        """
        matching_hooks = self._get_matching_hooks(event, input.tool_name)

        if not matching_hooks:
            return []

        async def execute_with_timeout(hook: Hook) -> Optional[HookOutput]:
            try:
                return await asyncio.wait_for(
                    hook.execute(input),
                    timeout=hook.timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Hook '{hook.name}' timed out (parallel)")
                return None
            except Exception as e:
                logger.error(f"Hook '{hook.name}' failed (parallel): {e}")
                return None

        results = await asyncio.gather(
            *[execute_with_timeout(hook) for hook in matching_hooks]
        )

        return [r for r in results if r is not None]

    def _get_matching_hooks(
        self,
        event: HookEvent,
        tool_name: Optional[str]
    ) -> List[Hook]:
        """
        Get hooks that match the event and optionally the tool name.

        Args:
            event: The hook event type
            tool_name: Optional tool name to match against

        Returns:
            List of matching enabled hooks (sorted by priority)
        """
        return [
            h for h in self._hooks[event]
            if h.enabled and (not tool_name or h.matches(tool_name))
        ]

    def __repr__(self) -> str:
        hook_counts = {e.value: len(self._hooks[e]) for e in HookEvent}
        counts_str = ", ".join(f"{k}={v}" for k, v in hook_counts.items() if v > 0)
        return f"HooksManager({counts_str or 'empty'})"
