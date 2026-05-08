"""
Kestrel Security Feature - Main feature class.

The SecurityFeature manages permissions and the approval queue,
providing CLI commands and API integration.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue, ApprovalRequest
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sdk.hooks.base import Hook
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)


class SecurityFeature(Feature):
    """
    Security Agent for managing permissions and approval queue.

    This feature provides:
    - Hierarchical permission management for all tools
    - Queue-based approval system for interactive policy building
    - Security hooks that intercept tool execution
    - CLI commands for permission management
    - Audit logging of all decisions

    CLI Commands:
    - !security-list: Show permission tree
    - !security-set: Set permission for a tool
    - !security-pending: Show pending approvals
    - !security-approve: Approve a pending request
    - !security-deny: Deny a pending request
    """

    def __init__(self, agent):
        """Initialize the security feature."""
        super().__init__(agent)
        self.permission_store: Optional[PermissionStore] = None
        self.approval_queue: Optional[ApprovalQueue] = None
        self.security_hook: Optional[SecurityHook] = None
        self._initialized = False
        self._init_lock: Optional[asyncio.Lock] = None

    @property
    def tool_description(self) -> str:
        """Description shown to orchestrator LLM."""
        return (
            "Manage security permissions for agent features and tools. "
            "Control which tools can run automatically, require approval, "
            "or are blocked. View pending approvals and audit logs."
        )

    async def initialize(self):
        """Initialize the security feature."""
        # Get database path from agent
        db_path = getattr(self.agent, "storage_path", "kestrel_prime.db")

        # Initialize permission store
        self.permission_store = PermissionStore(db_path)

        # Initialize approval queue with SSE callback AND a reference to the
        # permission store so the queue can persist the user's scope choice
        # ("session"/"always") and write audit rows centrally.  Without this,
        # the six features that call ``approval_queue.request_approval``
        # directly (code_edit, compute, keys, reflection.*) would each have
        # to remember to persist scope themselves — and all six historically
        # forgot, which is the bug behind #785.
        self.approval_queue = ApprovalQueue(
            on_request_added=self._emit_approval_request,
            on_request_withdrawn=self._emit_approval_withdrawn,
            permission_store=self.permission_store,
        )

        # Create security hook
        self.security_hook = SecurityHook(
            self.permission_store, self.approval_queue
        )

        # Initialize lock for async init
        self._init_lock = asyncio.Lock()

        # Complete async initialization directly (no longer need to schedule as task)
        await self._async_init()

        logger.info("SecurityFeature initialized (async setup pending)")

    async def _ensure_initialized(self):
        """Ensure async initialization is complete before operations."""
        if not self._initialized:
            await self._async_init()

    def get_hooks(self) -> List[Hook]:
        """Return the security hook for auto-registration."""
        if self.security_hook:
            return [self.security_hook]
        return []

    async def post_all_features_loaded(self, agent):
        """Register all tools with security permissions after all features are loaded."""
        await self._register_all_tools()
        logger.info("Security permissions registered for all features")

    async def _async_init(self):
        """Async initialization (database setup)."""
        async with self._init_lock:
            if self._initialized:
                return

            # Initialize database tables
            await self.permission_store.initialize()

        # NOTE: Hook registration is now handled automatically by the
        # Feature lifecycle via get_hooks(). No manual registration needed.

        # NOTE: _register_all_tools() is called by the agent AFTER all
        # features are registered, not here. If called here, features
        # registered after SecurityFeature (alphabetically later) would
        # be missing from the permission tree.

        self._initialized = True
        logger.info("SecurityFeature async initialization complete")

    async def _register_all_tools(self):
        """Register all agent tools with default ASK permission.

        Demo servers (KESTREL_DEMO_SERVER=1, set by ``kestrel demo run``) get
        ALLOW as the default instead — Playwright demos can't click
        through an interactive approval modal, and the demo agent runs
        in an isolated DB so the broader-grants are scoped correctly.
        Without this, every demo whose subject is something OTHER than
        the security flow has to chase modal-dismissal helpers in JS
        for whichever feature the LLM happens to pick (#897 review).
        """
        if not hasattr(self.agent, "features"):
            return

        is_demo_server = os.environ.get("KESTREL_DEMO_SERVER", "").lower() in (
            "1", "true", "yes",
        )
        default_level = PermissionLevel.ALLOW if is_demo_server else PermissionLevel.ASK

        for feature_name, feature in self.agent.features.items():
            # Skip registering our own tools
            if feature_name == "SecurityFeature":
                continue

            for tool_obj in feature.get_tools():
                await self.permission_store.register_tool(
                    feature_name=feature_name,
                    tool_name=tool_obj.name,
                    default_level=default_level,
                )

        if is_demo_server:
            logger.info(
                "Registered all tools with ALLOW (KESTREL_DEMO_SERVER=1)"
            )
        else:
            logger.info("Registered all tools with security permissions")

    async def _emit_approval_request(self, request: ApprovalRequest):
        """
        Emit SSE event for pending approval.

        Called by the approval queue when a new request is added.
        """
        if hasattr(self.agent, "emit_event"):
            await self.agent.emit_event(
                "approval_request",
                {
                    "id": request.id,
                    "feature": request.feature_name,
                    "tool": request.tool_name,
                    "args": request.tool_args,
                    "timestamp": request.created_at.isoformat(),
                },
            )
        else:
            logger.debug("Agent has no emit_event method, SSE not sent")

    async def _emit_approval_withdrawn(self, request: ApprovalRequest, reason: str):
        """
        Emit SSE event for an approval that was evicted from the queue
        without a user submit (timeout or task cancellation).

        The UI subscribes to ``approval_withdrawn`` and closes any modal
        showing this id, so the user doesn't end up clicking ""Approve"" on
        an entry the server has already discarded — which used to surface
        as a 404 with the misleading message ""expired"". See #877.
        """
        if hasattr(self.agent, "emit_event"):
            await self.agent.emit_event(
                "approval_withdrawn",
                {
                    "id": request.id,
                    "reason": reason,  # "timeout" | "cancelled"
                },
            )
        else:
            logger.debug("Agent has no emit_event method, withdrawal SSE not sent")

    async def shutdown(self):
        """Clean up resources."""
        # Cancel any pending approvals
        if self.approval_queue:
            self.approval_queue.cancel_all()

        # Clear session overrides
        if self.permission_store:
            self.permission_store.clear_session_overrides()

        logger.info("SecurityFeature shutdown")

    # === CLI Commands ===

    @tool(
        name="list_permissions",
        description="List all configured security permissions in tree format",
        category=ToolCategory.SYSTEM,
        command_prefix="!security-list",
    )
    async def list_permissions(self) -> ToolResult:
        """
        List all configured security permissions in tree format.

        Shows the hierarchical permission tree with rollup states
        for each feature and individual tool permissions.
        """
        try:
            tree = await self.permission_store.get_permission_tree()
        except Exception as e:
            logger.error(f"list_permissions failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not tree:
            return ToolResult.ok(
                confirmation=(
                    "Security Permissions: no permissions configured yet "
                    "(tools will be registered on first invocation)"
                ),
                data={"feature_count": 0, "tools_count": 0, "features": []},
            )

        # Format as hierarchical tree
        state_icon = {
            "allow_all": "☑",
            "deny_all": "☒",
            "ask_all": "☐",
            "session_all": "◑",
            "mixed": "◐",
        }
        level_icon = {
            "allow": "☑",
            "deny": "☒",
            "ask": "☐",
            "session": "◑",
        }

        lines = ["Security Permissions:\n"]
        features_data = []
        tools_count = 0
        for feature in tree:
            icon = state_icon.get(feature.rollup_state, "?")
            lines.append(f"  {icon} {feature.feature_name} [{feature.rollup_state}]")
            tools_struct = []
            for tool_obj in feature.tools:
                tool_icon = level_icon.get(tool_obj.level.value, "?")
                lines.append(f"    {tool_icon} {tool_obj.tool_name}")
                tools_struct.append({
                    "tool_name": tool_obj.tool_name,
                    "level": tool_obj.level.value,
                })
                tools_count += 1
            features_data.append({
                "feature_name": feature.feature_name,
                "rollup_state": feature.rollup_state,
                "tools": tools_struct,
            })

        lines.append(
            "\nLegend: ☑=Allow ☒=Deny ☐=Ask ◑=Session ◐=Mixed"
        )
        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "feature_count": len(features_data),
                "tools_count": tools_count,
                "features": features_data,
            },
        )

    @tool(
        name="set_permission",
        description="Set permission level for a tool or all tools in a feature",
        category=ToolCategory.SYSTEM,
        command_prefix="!security-set",
    )
    async def set_permission(
        self,
        feature_name: str,
        tool_name: Optional[str] = None,
        level: str = "ask",
    ) -> ToolResult:
        """
        Set permission level for a tool or all tools in a feature.

        Args:
            feature_name: Name of the feature (e.g., "WalletAgent")
            tool_name: Name of the tool (optional, sets all if omitted)
            level: Permission level - "allow", "deny", "ask", or "session"
        """
        try:
            perm_level = PermissionLevel(level)
        except ValueError:
            return ToolResult.failed(
                f"Invalid level '{level}'. Use: allow, deny, ask, session"
            )

        try:
            if tool_name:
                await self.permission_store.set_permission(
                    feature_name, tool_name, perm_level
                )
                return ToolResult.ok(
                    confirmation=f"Set {feature_name}.{tool_name} to {level}",
                    data={
                        "feature_name": feature_name,
                        "tool_name": tool_name,
                        "level": level,
                        "scope": "tool",
                    },
                )
            await self.permission_store.set_feature_permission(
                feature_name, perm_level
            )
            return ToolResult.ok(
                confirmation=f"Set all tools in {feature_name} to {level}",
                data={
                    "feature_name": feature_name,
                    "tool_name": None,
                    "level": level,
                    "scope": "feature",
                },
            )
        except Exception as e:
            logger.error(f"set_permission failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

    @tool(
        name="pending_approvals",
        description="Show pending approval requests",
        category=ToolCategory.SYSTEM,
        command_prefix="!security-pending",
    )
    async def pending_approvals(self) -> ToolResult:
        """Show pending approval requests."""
        pending = self.approval_queue.pending_requests

        if not pending:
            return ToolResult.ok(
                confirmation="No pending approvals",
                data={"count": 0, "requests": []},
            )

        lines = [f"Pending Approvals ({len(pending)}):"]
        requests_data = []
        for req in pending:
            created_at = req.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - created_at).total_seconds()
            lines.append(
                f"  [{req.id[:8]}] {req.feature_name}.{req.tool_name} "
                f"({int(age)}s ago)"
            )
            requests_data.append({
                "id": req.id,
                "feature_name": req.feature_name,
                "tool_name": req.tool_name,
                "age_seconds": int(age),
                "created_at": created_at.isoformat(),
            })

        lines.append(
            "\nUse !security-approve <id> or !security-deny <id> to respond"
        )
        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={"count": len(requests_data), "requests": requests_data},
        )

    @tool(
        name="approve",
        description="Approve a pending request",
        category=ToolCategory.SYSTEM,
        command_prefix="!security-approve",
    )
    async def approve_request(
        self,
        request_id: str,
        scope: str = "once",
    ) -> ToolResult:
        """
        Approve a pending request.

        Args:
            request_id: ID of the pending request (first 8 chars OK)
            scope: Approval scope - "once", "session", or "always"
        """
        if scope not in ("once", "session", "always"):
            return ToolResult.failed(
                f"Invalid scope '{scope}'. Use: once, session, always"
            )

        # Support partial ID matching
        full_id = self._find_request_id(request_id)
        if not full_id:
            return ToolResult.failed(
                f"Request '{request_id}' not found",
                data={"request_id": request_id},
            )

        # submit_decision returns True only if the queue accepted the
        # decision (the request is still pending). If False, the queue
        # withdrew or expired it between _find_request_id and now —
        # surface as ERROR rather than claiming approval.
        if not self.approval_queue.submit_decision(full_id, True, scope):
            return ToolResult.failed(
                f"Request '{request_id}' is no longer pending "
                "(timeout, cancellation, or already decided)",
                data={"request_id": full_id, "decision_attempted": "approved"},
            )

        # Honesty: ``ApprovalQueue.submit_decision`` only sets the
        # in-memory decision; the scope is persisted later by
        # ``request_approval()`` via ``_persist_decision``, and that
        # path swallows store failures. For ``scope="once"`` there's
        # no persistence — the immediate approval is the whole
        # action, so OK is honest. For ``scope="session"``/``"always"``
        # the durable scope may not actually be written, so we surface
        # PARTIAL and tell the LLM the next tool call may re-prompt.
        # See round 2 codex finding + #1078 follow-up ticket.
        if scope == "once":
            return ToolResult.ok(
                confirmation=f"Approved {request_id[:8]} (once)",
                data={
                    "request_id": full_id,
                    "scope": scope,
                    "decision": "approved",
                },
            )
        return ToolResult.partial(
            confirmation=(
                f"Approved {request_id[:8]} for this request "
                f"(decision submitted with scope={scope})"
            ),
            error=(
                f"scope={scope} persistence is asynchronous and store "
                "failures are not surfaced; the durable permission may "
                "not have been written. The next tool call may re-prompt"
            ),
            data={
                "request_id": full_id,
                "scope": scope,
                "decision": "approved",
                "scope_persistence": "asynchronous_and_unverified",
            },
        )

    @tool(
        name="deny",
        description="Deny a pending request",
        category=ToolCategory.SYSTEM,
        command_prefix="!security-deny",
    )
    async def deny_request(self, request_id: str) -> ToolResult:
        """
        Deny a pending request.

        Args:
            request_id: ID of the pending request (first 8 chars OK)
        """
        # Support partial ID matching
        full_id = self._find_request_id(request_id)
        if not full_id:
            return ToolResult.failed(
                f"Request '{request_id}' not found",
                data={"request_id": request_id},
            )

        if self.approval_queue.submit_decision(full_id, False, "denied"):
            return ToolResult.ok(
                confirmation=f"Denied {request_id[:8]}",
                data={"request_id": full_id, "decision": "denied"},
            )
        return ToolResult.failed(
            f"Request '{request_id}' is no longer pending "
            "(timeout, cancellation, or already decided)",
            data={"request_id": full_id, "decision_attempted": "denied"},
        )

    @tool(
        name="security_audit",
        description="Show recent security audit log",
        category=ToolCategory.SYSTEM,
        command_prefix="!security-audit",
    )
    async def security_audit(self, limit: int = 20) -> ToolResult:
        """
        Show recent security audit log.

        Args:
            limit: Maximum number of entries to show (the request — actual
                   count returned may be lower if fewer entries exist).
        """
        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(
                f"limit must be an integer, got {limit!r}"
            )
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")

        try:
            logs = await self.permission_store.get_audit_log(limit_val)
        except Exception as e:
            logger.error(f"security_audit failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        if not logs:
            return ToolResult.ok(
                confirmation="Security Audit Log: no entries yet",
                data={"count": 0, "limit_requested": limit_val, "entries": []},
            )

        lines = [f"Security Audit Log (last {len(logs)} entries):\n"]

        # Honesty + privacy: the LLM receives ``ToolResult.data`` in
        # the next turn's context. Audit rows can carry an
        # ``args_summary`` that includes tool arguments (paths,
        # request payloads, sometimes unmasked direct-ApprovalQueue
        # callers). Surfacing those into the LLM context is a
        # privacy regression versus the str-only pre-fix shape, which
        # deliberately omitted args. Filter ``entries`` to the same
        # fields the confirmation displays. (Round 1 codex finding.)
        _SAFE_AUDIT_FIELDS = ("feature", "tool", "decision", "user_choice", "timestamp")
        safe_entries = []
        for entry in logs:
            decision_icon = {
                "auto_allowed": "☑",
                "auto_denied": "☒",
                "user_approved": "✓",
                "user_denied": "✗",
                "timeout": "⏱",
            }.get(entry["decision"], "?")

            lines.append(
                f"  {decision_icon} {entry['feature']}.{entry['tool']} "
                f"[{entry['decision']}]"
            )
            if entry["user_choice"]:
                lines.append(f"     ↳ scope: {entry['user_choice']}")

            safe_entries.append({
                k: entry.get(k) for k in _SAFE_AUDIT_FIELDS if k in entry
            })

        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "count": len(logs),
                "limit_requested": limit_val,
                "entries": safe_entries,
            },
        )

    def _find_request_id(self, partial_id: str) -> Optional[str]:
        """
        Find full request ID from partial match.

        Args:
            partial_id: First few characters of the request ID

        Returns:
            Full request ID if found, None otherwise
        """
        for request in self.approval_queue.pending_requests:
            if request.id.startswith(partial_id):
                return request.id
        return None
