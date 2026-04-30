"""
Kestrel Security Feature - Main feature class.

The SecurityFeature manages permissions and the approval queue,
providing CLI commands and API integration.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue, ApprovalRequest
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.hooks.base import Hook
from kestrel_sovereign.tools.base import ToolCategory

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
        """Register all agent tools with default ASK permission."""
        if not hasattr(self.agent, "features"):
            return

        for feature_name, feature in self.agent.features.items():
            # Skip registering our own tools
            if feature_name == "SecurityFeature":
                continue

            for tool_obj in feature.get_tools():
                await self.permission_store.register_tool(
                    feature_name=feature_name,
                    tool_name=tool_obj.name,
                    default_level=PermissionLevel.ASK,
                )

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
    async def list_permissions(self) -> str:
        """
        List all configured security permissions in tree format.

        Shows the hierarchical permission tree with rollup states
        for each feature and individual tool permissions.

        Returns:
            Formatted permission tree as string
        """
        tree = await self.permission_store.get_permission_tree()

        if not tree:
            return (
                "Security Permissions:\n"
                "  No permissions configured yet.\n"
                "  Tools will be registered when first invoked.\n"
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

        for feature in tree:
            icon = state_icon.get(feature.rollup_state, "?")
            lines.append(f"  {icon} {feature.feature_name} [{feature.rollup_state}]")

            for tool_obj in feature.tools:
                icon = level_icon.get(tool_obj.level.value, "?")
                lines.append(f"    {icon} {tool_obj.tool_name}")

        lines.append(
            "\nLegend: ☑=Allow ☒=Deny ☐=Ask ◑=Session ◐=Mixed"
        )
        return "\n".join(lines)

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
    ) -> str:
        """
        Set permission level for a tool or all tools in a feature.

        Args:
            feature_name: Name of the feature (e.g., "WalletAgent")
            tool_name: Name of the tool (optional, sets all if omitted)
            level: Permission level - "allow", "deny", "ask", or "session"

        Returns:
            Confirmation message
        """
        try:
            perm_level = PermissionLevel(level)
        except ValueError:
            return f"Invalid level '{level}'. Use: allow, deny, ask, session"

        if tool_name:
            await self.permission_store.set_permission(
                feature_name, tool_name, perm_level
            )
            return f"Set {feature_name}.{tool_name} to {level}"
        else:
            await self.permission_store.set_feature_permission(
                feature_name, perm_level
            )
            return f"Set all tools in {feature_name} to {level}"

    @tool(
        name="pending_approvals",
        description="Show pending approval requests",
        category=ToolCategory.SYSTEM,
        command_prefix="!security-pending",
    )
    async def pending_approvals(self) -> str:
        """
        Show pending approval requests.

        Returns:
            List of pending requests or "No pending approvals"
        """
        pending = self.approval_queue.pending_requests

        if not pending:
            return "No pending approvals"

        lines = [f"Pending Approvals ({len(pending)}):"]
        for req in pending:
            created_at = req.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - created_at).total_seconds()
            lines.append(
                f"  [{req.id[:8]}] {req.feature_name}.{req.tool_name} "
                f"({int(age)}s ago)"
            )

        lines.append(
            "\nUse !security-approve <id> or !security-deny <id> to respond"
        )
        return "\n".join(lines)

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
    ) -> str:
        """
        Approve a pending request.

        Args:
            request_id: ID of the pending request (first 8 chars OK)
            scope: Approval scope - "once", "session", or "always"

        Returns:
            Confirmation message
        """
        if scope not in ("once", "session", "always"):
            return f"Invalid scope '{scope}'. Use: once, session, always"

        # Support partial ID matching
        full_id = self._find_request_id(request_id)
        if not full_id:
            return f"Request '{request_id}' not found"

        if self.approval_queue.submit_decision(full_id, True, scope):
            return f"✓ Approved {request_id[:8]} with scope={scope}"
        return f"Request '{request_id}' not found"

    @tool(
        name="deny",
        description="Deny a pending request",
        category=ToolCategory.SYSTEM,
        command_prefix="!security-deny",
    )
    async def deny_request(self, request_id: str) -> str:
        """
        Deny a pending request.

        Args:
            request_id: ID of the pending request (first 8 chars OK)

        Returns:
            Confirmation message
        """
        # Support partial ID matching
        full_id = self._find_request_id(request_id)
        if not full_id:
            return f"Request '{request_id}' not found"

        if self.approval_queue.submit_decision(full_id, False, "denied"):
            return f"✗ Denied {request_id[:8]}"
        return f"Request '{request_id}' not found"

    @tool(
        name="security_audit",
        description="Show recent security audit log",
        category=ToolCategory.SYSTEM,
        command_prefix="!security-audit",
    )
    async def security_audit(self, limit: int = 20) -> str:
        """
        Show recent security audit log.

        Args:
            limit: Maximum number of entries to show (default 20)

        Returns:
            Formatted audit log
        """
        logs = await self.permission_store.get_audit_log(limit)

        if not logs:
            return "Security Audit Log: No entries yet"

        lines = [f"Security Audit Log (last {len(logs)} entries):\n"]

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

        return "\n".join(lines)

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
