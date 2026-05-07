"""
Kestrel Security - Security Hook Implementation.

This module provides the SecurityHook that intercepts tool execution
and enforces permission policies.
"""

import json
import logging
from typing import Optional

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue

logger = logging.getLogger(__name__)


class SecurityHook(Hook):
    """
    PreToolUse hook that checks permissions and queues for approval.

    This hook runs before any tool execution and:
    1. Checks the permission level for the tool
    2. Allows/denies based on stored policy
    3. Queues for user approval if level is ASK
    4. Logs the decision to the audit log

    Example:
        hook = SecurityHook(permission_store, approval_queue)
        manager.register(hook)

        # Now all tool executions go through the security check
    """

    def __init__(
        self,
        permission_store: PermissionStore,
        approval_queue: ApprovalQueue,
        priority: int = 10,  # Run early in hook chain
    ):
        """
        Initialize the security hook.

        Args:
            permission_store: Store for permission policies
            approval_queue: Queue for pending approvals
            priority: Hook priority (default 10, runs early)
        """
        super().__init__(
            name="security_guard",
            events=[HookEvent.PRE_TOOL_USE, HookEvent.PRE_SUBAGENT_CALL],
            priority=priority,
            # SecurityHook blocks on human input. The hook manager
            # MUST NOT wrap its execute() in ``asyncio.wait_for`` —
            # that would cancel the queue's await before the user
            # could click. Was the actual driver of the "modal
            # disappears in ~5 seconds" bug: Hook.timeout defaulted
            # to 5s, and the manager's wait_for ran on every hook.
            # Setting awaits_user_input=True moves the lifecycle
            # bound off the hook timer and onto the approval queue
            # itself (which has its own staleness sweep).
            awaits_user_input=True,
        )
        self.permission_store = permission_store
        self.approval_queue = approval_queue

    async def execute(self, input: HookInput) -> HookOutput:
        """
        Check permissions and enforce policy.

        Args:
            input: HookInput with tool/feature context

        Returns:
            HookOutput with permission decision
        """
        feature_name = input.feature_name or "unknown"
        tool_name = input.tool_name or "unknown"

        logger.debug(f"Security check: {feature_name}.{tool_name}")

        # Get current permission level
        level = await self.permission_store.get_permission(feature_name, tool_name)

        # Prepare args summary for audit log (truncate for privacy)
        args_summary = self._summarize_args(input.tool_input)

        if level == PermissionLevel.ALLOW:
            await self.permission_store.log_decision(
                feature_name=feature_name,
                tool_name=tool_name,
                action="tool_execution",
                decision="auto_allowed",
                args_summary=args_summary,
            )
            logger.debug(f"Auto-allowed: {feature_name}.{tool_name}")
            return HookOutput.allow(f"Auto-approved: {feature_name}.{tool_name}")

        if level == PermissionLevel.DENY:
            await self.permission_store.log_decision(
                feature_name=feature_name,
                tool_name=tool_name,
                action="tool_execution",
                decision="auto_denied",
                args_summary=args_summary,
            )
            logger.info(f"Auto-denied: {feature_name}.{tool_name}")
            return HookOutput.deny(f"Blocked by policy: {feature_name}.{tool_name}")

        if level in (PermissionLevel.ASK, PermissionLevel.SESSION):
            # Queue for approval and wait.  As of #785 the queue itself
            # owns scope persistence and audit-row writes — we just need
            # to translate the (approved, scope) result into a HookOutput.
            # This used to live here, but every direct ``request_approval``
            # caller (code_edit, compute, keys, reflection.*) was missing
            # it, so the responsibility moved into the queue where every
            # caller benefits.
            logger.info(f"Requesting approval: {feature_name}.{tool_name}")

            approved, scope = await self.approval_queue.request_approval(
                feature_name=feature_name,
                tool_name=tool_name,
                tool_args=input.tool_input or {},
            )

            if not approved:
                logger.info(f"User denied or timeout: {feature_name}.{tool_name}")
                return HookOutput.deny(f"User denied: {feature_name}.{tool_name}")

            logger.info(f"User approved ({scope}): {feature_name}.{tool_name}")
            return HookOutput.allow(f"User approved: {scope}")

        # Fallback: allow
        return HookOutput.allow()

    def _summarize_args(
        self,
        args: Optional[dict],
        max_length: int = 200,
    ) -> Optional[str]:
        """
        Create a privacy-safe summary of tool arguments.

        Args:
            args: Tool arguments dictionary
            max_length: Maximum summary length

        Returns:
            Truncated JSON string or None
        """
        if not args:
            return None

        try:
            # Mask potentially sensitive values
            masked = self._mask_sensitive(args)
            summary = json.dumps(masked, default=str)

            if len(summary) > max_length:
                return summary[: max_length - 3] + "..."
            return summary

        except (TypeError, ValueError) as e:
            logger.debug(f"Failed to summarize args (type/value error): {e}")
            return "(args could not be summarized)"
        except Exception as e:
            logger.debug(f"Failed to summarize args: {e}", exc_info=True)
            return "(args could not be summarized)"

    def _mask_sensitive(self, data: dict) -> dict:
        """
        Mask potentially sensitive values in arguments.

        Args:
            data: Dictionary to mask

        Returns:
            Dictionary with sensitive values masked
        """
        sensitive_keys = {
            "password",
            "secret",
            "token",
            "key",
            "api_key",
            "private_key",
            "credit_card",
            "ssn",
            "social_security",
        }

        result = {}
        for key, value in data.items():
            key_lower = key.lower()

            # Mask sensitive keys
            if any(s in key_lower for s in sensitive_keys):
                result[key] = "***MASKED***"
            elif isinstance(value, dict):
                result[key] = self._mask_sensitive(value)
            elif isinstance(value, list):
                result[key] = [
                    self._mask_sensitive(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    def __repr__(self) -> str:
        return f"SecurityHook(name={self.name}, priority={self.priority})"
