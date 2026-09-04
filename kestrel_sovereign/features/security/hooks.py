"""
Kestrel Security - Security Hook Implementation.

This module provides the SecurityHook that intercepts tool execution
and enforces permission policies.
"""

import logging
from typing import Optional

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
from kestrel_sovereign.features.security.permissions import (
    SUBAGENT_DISPATCH_ACTION,
    PermissionLevel,
    PermissionStore,
)
from kestrel_sovereign.features.security.approval_queue import (
    ApprovalQueue,
    classify_denial,
)
from kestrel_sovereign.features.security.args_summary import (
    mask_sensitive,
    summarize_args,
)

logger = logging.getLogger(__name__)

# Session ids that identify a NON-INTERACTIVE hook invocation — a background
# context with no human attached to the approval queue. An ASK-gated tool in
# one of these contexts must resolve to a non-blocking no_approver denial rather
# than queue-and-wait forever (which would wedge the background loop, #2111).
# The scheduler tags every tick with session_id="scheduler"
# (features/scheduler/feature.py); test_scheduler_ask_gate pins that agreement.
NON_INTERACTIVE_SESSION_IDS = frozenset({"scheduler"})


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

        # This hook runs on PRE_SUBAGENT_CALL as well as PRE_TOOL_USE. Both
        # used to write "tool_execution", which made a feature-as-subagent
        # DISPATCH — an envelope carrying the requested task text — look
        # exactly like a tool that actually ran (#3107). Record which one this
        # is, so a reader can tell a request from an action.
        audit_action = (
            SUBAGENT_DISPATCH_ACTION
            if input.hook_event_name == HookEvent.PRE_SUBAGENT_CALL.value
            else "tool_execution"
        )

        if level == PermissionLevel.ALLOW:
            await self.permission_store.log_decision(
                feature_name=feature_name,
                tool_name=tool_name,
                action=audit_action,
                decision="auto_allowed",
                args_summary=args_summary,
            )
            logger.debug(f"Auto-allowed: {feature_name}.{tool_name}")
            return HookOutput.allow(f"Auto-approved: {feature_name}.{tool_name}")

        if level == PermissionLevel.AUTO:
            await self.permission_store.log_decision(
                feature_name=feature_name,
                tool_name=tool_name,
                action=audit_action,
                decision="auto_mode_allowed",
                user_choice="constitutional_honesty_unflagged",
                args_summary=args_summary,
            )
            logger.info(
                "Auto mode approved %s.%s after earlier policy hooks did not block",
                feature_name,
                tool_name,
            )
            return HookOutput.allow(
                "Auto mode approved after constitutional/honesty checks did not flag: "
                f"{feature_name}.{tool_name}"
            )

        if level == PermissionLevel.DENY:
            await self.permission_store.log_decision(
                feature_name=feature_name,
                tool_name=tool_name,
                action=audit_action,
                decision="auto_denied",
                args_summary=args_summary,
            )
            logger.info(f"Auto-denied: {feature_name}.{tool_name}")
            return HookOutput.deny(f"Blocked by policy: {feature_name}.{tool_name}")

        if level in (
            PermissionLevel.ASK,
            PermissionLevel.ALWAYS_ASK,
            PermissionLevel.SESSION,
        ):
            # Queue for approval and wait.  As of #785 the queue itself
            # owns scope persistence and audit-row writes — we just need
            # to translate the (approved, scope) result into a HookOutput.
            # This used to live here, but every direct ``request_approval``
            # caller (code_edit, compute, keys, reflection.*) was missing
            # it, so the responsibility moved into the queue where every
            # caller benefits.
            logger.info(f"Requesting approval: {feature_name}.{tool_name}")

            # A non-interactive caller (e.g. a scheduler tick) has no human to
            # answer the queue; ask the queue not to block-and-wait forever but
            # to return a non-blocking no_approver denial instead (#2111).
            allow_blocking = input.session_id not in NON_INTERACTIVE_SESSION_IDS
            approved, scope = await self.approval_queue.request_approval(
                feature_name=feature_name,
                tool_name=tool_name,
                tool_args=input.tool_input or {},
                allow_blocking=allow_blocking,
                # The queue writes audit rows on paths this hook never
                # returns through, so it needs the same fact. Without this the
                # ASK path — which is the DEFAULT for the feature-as-subagent
                # dispatcher — kept writing "tool_execution" and the exclusion
                # in security_audit_search missed exactly the case it was
                # added for (#3107 review round 2 fixed one door, not both).
                audit_action=audit_action,
            )

            if not approved:
                # Branch on the denial provenance — only an explicit human deny
                # is a user denial (#1542). classify_denial is the shared source
                # of truth so this hook and the direct queue consumers
                # (ComputeFeature, KeysFeature, …) can't drift.
                denial = classify_denial(scope)

                # Headless test instance with no one to answer the queue
                # (#2029). The queue short-circuited rather than blocking
                # forever; surface an honest, non-blocking "requires approval /
                # no approver available" denial instead of mislabeling it as a
                # user denial. This is what keeps a single spawn_agent call
                # from wedging the agent's request worker until restart.
                if scope == "no_approver":
                    logger.warning(
                        "No interactive approver for %s.%s; returning "
                        "requires-approval (non-blocking).",
                        feature_name,
                        tool_name,
                    )
                    return HookOutput.deny(
                        f"{feature_name}.{tool_name} requires approval, but no "
                        "interactive approver is available (headless/test "
                        "instance). Not queuing to avoid an indefinite block. "
                        "Enable non-interactive approval "
                        "(KESTREL_TEST_AUTO_APPROVE=1) or run with an attached "
                        "approver."
                    )
                if denial.is_user_denial:
                    logger.info(f"User denied: {feature_name}.{tool_name}")
                    return HookOutput.deny(f"User denied: {feature_name}.{tool_name}")
                logger.info(
                    "Not approved (%s): %s.%s",
                    denial.reason,
                    feature_name,
                    tool_name,
                )
                return HookOutput.deny(
                    f"{feature_name}.{tool_name} not approved — "
                    f"{denial.description}."
                )

            logger.info(f"User approved ({scope}): {feature_name}.{tool_name}")
            return HookOutput.allow(f"User approved: {scope}")

        # Fallback: allow
        return HookOutput.allow()

    def _summarize_args(
        self,
        args: Optional[dict],
        max_length: Optional[int] = None,
    ) -> Optional[str]:
        """Create a privacy-safe summary of tool arguments.

        Thin wrapper over the shared
        :func:`kestrel_sovereign.features.security.args_summary.summarize_args`
        so this path and :class:`ApprovalQueue` produce identical masked rows.

        That sentence used to be false. This wrapper overrode the shared cap
        with 200 while ``ApprovalQueue`` used the shared 500, so the same tool
        call was truncated to two different lengths depending on which door it
        came through — and a read-back over the column could not state one
        honest bound (#3107 review round 3). The override is gone; passing
        ``max_length`` explicitly is still allowed for a caller that means it.
        """
        if max_length is None:
            return summarize_args(args)
        return summarize_args(args, max_length=max_length)

    def _mask_sensitive(self, data: dict) -> dict:
        """Mask potentially sensitive values in arguments.

        Delegates to the shared
        :func:`kestrel_sovereign.features.security.args_summary.mask_sensitive`.
        """
        return mask_sensitive(data)

    def __repr__(self) -> str:
        return f"SecurityHook(name={self.name}, priority={self.priority})"
