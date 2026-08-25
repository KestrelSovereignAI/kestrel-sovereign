"""
Kestrel Security Feature - Main feature class.

The SecurityFeature manages permissions and the approval queue,
providing CLI commands and API integration.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.enum_coerce import normalize_choice as _normalize_choice
from kestrel_sovereign.features.security.permissions import (
    SEARCH_TOOL_NAME,
    PermissionLevel,
    PermissionStore,
    assert_sdk_permission_level_parity,
    compose_restrictive_permission,
)


#: Above this many matches, ``security_audit_search`` returns the COUNT and no
#: arguments. The tool's whole justification for surfacing ``args_summary`` —
#: which ``security_audit`` deliberately withholds — is that the caller already
#: described what it wanted. A query matching most of the log is not a
#: description, and answering it with a page of arguments would rebuild the
#: unbounded disclosure by another route. This is a disclosure bound, not a
#: page-size preference, which is why ``limit`` is clamped to it rather than
#: being allowed to exceed it.
MAX_DISCLOSING_MATCHES = 25


# Per-feature default permission levels for fresh agents (#406).
#
# Keys here MUST match the registered ``Feature.name`` value (which is the
# Python class name in core features). Codex review on #1262 caught a first
# revision where these used aspirational-sounding names (`ModelFeature`,
# `TasksFeature`, `KeysFeature`, `ChannelsFeature`, `WebhooksFeature`) that
# do not match the actual class names — those entries silently fell through
# to the ASK fallback, leaving the very approval loop this map is supposed
# to prevent. Class names below are the canonical ones; cross-reference
# against ``grep -rn '^class.*Feature\\|^class.*Agent' kestrel_sovereign/features/``.
#
# When SecurityFeature registers each loaded feature's tools at startup, the
# default permission used to be ASK for everything. That left brand-new
# agents paralyzed: every core boot tool (Bootstrap, Sovereignty, Identity)
# raised an approval modal nobody was there to answer, looping the agent
# into a 20+ minute timeout cascade on first run (Meridian, #406).
#
# This map opts the core "must work to function" features into ALLOW.
# Genuinely destructive or externally-visible features stay on ASK. An SDK
# feature declaration composes with an explicit entry here: ordinary modes
# cannot weaken the Sovereign baseline, while ALWAYS_ASK or DENY may tighten
# it. Anything not listed falls through to ASK only when it has no SDK
# declaration — adding a legacy feature to the agent does NOT silently grant
# it, while an unmapped SDK feature retains its declared contract.
_DEFAULT_PERMISSION_BY_FEATURE: Dict[str, PermissionLevel] = {
    # --- Core boot / identity / memory: ALLOW (agent cannot function without) ---
    "BootstrapFeature": PermissionLevel.ALLOW,
    "IdentityFeature": PermissionLevel.ALLOW,
    "ConstitutionFeature": PermissionLevel.ALLOW,
    "MemoryFeature": PermissionLevel.ALLOW,
    "MemoryAgencyFeature": PermissionLevel.ALLOW,
    "StrategicMemoryFeature": PermissionLevel.ALLOW,
    "ContextFeature": PermissionLevel.ALLOW,
    "SovereigntyFeature": PermissionLevel.ALLOW,
    "HealthFeature": PermissionLevel.ALLOW,
    "ModelAgent": PermissionLevel.ALLOW,            # not ModelFeature
    "SaveFeature": PermissionLevel.ALLOW,
    "TaskFeature": PermissionLevel.ALLOW,           # not TasksFeature
    "StateOfMindFeature": PermissionLevel.ALLOW,
    "ResponseAuditFeature": PermissionLevel.ALLOW,
    "AuditAnchorFeature": PermissionLevel.ALLOW,
    "WellnessFeature": PermissionLevel.ALLOW,
    "WebSearchFeature": PermissionLevel.ALLOW,
    "ChannelFeature": PermissionLevel.ALLOW,        # not ChannelsFeature
    "PeersFeature": PermissionLevel.ALLOW,
    "SchedulerFeature": PermissionLevel.ALLOW,
    "ConsentFeature": PermissionLevel.ALLOW,
    "SkillsFeature": PermissionLevel.ALLOW,
    "CliFeature": PermissionLevel.ALLOW,

    # --- Externally-visible / irreversible / risky: ASK explicitly ---
    "ComputeFeature": PermissionLevel.ASK,
    "ComputerUseFeature": PermissionLevel.ASK,
    "SpawnFeature": PermissionLevel.ASK,
    "DeliveryFeature": PermissionLevel.ASK,
    "WebhookFeature": PermissionLevel.ASK,          # not WebhooksFeature
    "BridgeFeature": PermissionLevel.ASK,
    "DeployFeature": PermissionLevel.ASK,
    "InferenceLeaseFeature": PermissionLevel.ASK,
    # KeyManagementFeature is ASK at the feature level because it bundles
    # destructive operations (delete_service_key, remove_service_key) with
    # read-only listings. ALLOW on the whole feature would auto-grant
    # irreversible credential deletion to a fresh agent. Operators can
    # upgrade specific read-only tools to ALLOW post-inception if needed,
    # but the feature default stays conservative (codex review v3 #1262).
    "KeyManagementFeature": PermissionLevel.ASK,
}


def default_permission_for_feature(
    feature_name: str,
    fallback: PermissionLevel = PermissionLevel.ASK,
) -> PermissionLevel:
    """Return the default permission level for a freshly-loaded feature.

    Unmapped callers get the conservative ASK fallback. SDK declarations are
    composed by ``SecurityFeature._register_all_tools`` only when this map has
    an explicit entry; otherwise the declaration remains the feature baseline.
    To opt a legacy feature into ALLOW or to explicitly ASK, update
    ``_DEFAULT_PERMISSION_BY_FEATURE`` above.
    """
    return _DEFAULT_PERMISSION_BY_FEATURE.get(feature_name, fallback)


# Per-TOOL default-permission overrides (#2093, F203).
#
# The per-feature map above is all-or-nothing: a feature is either ALLOW or
# ASK for *every* tool it exposes. That fails for features that mix cheap
# reads with irreversible destruction — the same reason ``KeyManagementFeature``
# stays feature-level ASK (see its comment above). ``MemoryFeature`` is the
# inverse case: it MUST be feature-level ALLOW so a fresh agent can search and
# recall without an approval-modal loop (#406), yet a handful of its tools
# hard-delete conversation history with no recovery. Feature-level ALLOW let
# any loaded agent — including a demo/governed one — call
# ``purge_conversation`` and wipe every live and trashed row in a session with
# no authorization check, while the tool description promised a Sovereign gate
# that did not exist (F203).
#
# This map overrides the per-feature default for specific ``(feature, tool)``
# pairs, keyed by the registered ``Feature.name`` mapping to a ``{tool_name:
# level}`` dict. Destructive tools register at ALWAYS_ASK so ``SecurityHook``
# routes them through the Sovereign approval queue and — being ALWAYS_ASK, not
# ASK — they stay a hard rail that global auto-mode cannot promote to AUTO,
# matching the ``codex_native`` commandExecution/fileChange precedent below.
# They are also excluded from the demo-server ALLOW override (see
# ``_register_all_tools``).
_DEFAULT_PERMISSION_BY_TOOL: Dict[str, Dict[str, PermissionLevel]] = {
    "MemoryFeature": {
        "purge_conversation": PermissionLevel.ALWAYS_ASK,
        "purge_message_by_id": PermissionLevel.ALWAYS_ASK,
        "delete_messages": PermissionLevel.ALWAYS_ASK,
        "delete_conversation": PermissionLevel.ALWAYS_ASK,
        "delete_message_by_id": PermissionLevel.ALWAYS_ASK,
    },
    "SpawnFeature": {
        # The optional ``offboard_runtime`` argument irreversibly removes the
        # child and descendant hosted-runtime trees. Permissions are
        # tool-scoped, so the whole tool must remain an unpromotable approval
        # rail. Static registration also upgrades legacy persisted ALLOW rows
        # from when termination only stopped a restartable child.
        "terminate_child": PermissionLevel.ALWAYS_ASK,
    },
    "InferenceLeaseFeature": {
        # Acquisition starts billable infrastructure and remains a hard
        # approval rail. Status is read-only; release is owner-scoped and stops
        # spend, so both must remain available for unattended reconciliation.
        "inference_lease_acquire": PermissionLevel.ALWAYS_ASK,
        "inference_lease_status": PermissionLevel.ALLOW,
        "inference_lease_release": PermissionLevel.ALLOW,
    },
}


def default_permission_for_tool(
    feature_name: str,
    tool_name: str,
) -> Optional[PermissionLevel]:
    """Return a per-tool default-permission override, or ``None``.

    ``None`` means "no static rail". A non-None result is a Sovereign-owned
    migration/security floor: an SDK declaration may tighten it but cannot
    weaken it.
    """
    return _DEFAULT_PERMISSION_BY_TOOL.get(feature_name, {}).get(tool_name)
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue, ApprovalRequest
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sdk.hooks.base import Hook
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)


# Environment variables that opt a *test instance* into non-interactive
# approval (#1936). Any one set to a truthy value is sufficient. This is the
# explicit second gate on top of ``is_test_instance`` — a tagged test agent
# does NOT auto-approve unless the operator who launched it asked for it, so a
# test instance used interactively (e.g. a recorded demo) still prompts.
_TEST_AUTO_APPROVE_ENV_VARS = (
    "KESTREL_TEST_AUTO_APPROVE",
    "KESTREL_TEST_INSTANCE_AUTO_APPROVE",
)


def _test_auto_approve_opt_in() -> bool:
    """True when the operator has opted test instances into non-interactive
    approval via the environment. Never consulted for non-test agents."""
    for _var in _TEST_AUTO_APPROVE_ENV_VARS:
        if os.environ.get(_var, "").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


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
        self.auto_approve_policy = None
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

        # Build the scoped auto-approve policy from kestrel.toml's
        # [security] section + the Sovereign-curated DB allowlist. This is
        # what lets a sovereign agent close her own loop without the
        # Sovereign typing CLI approvals (epic #1290).
        try:
            from kestrel_sovereign.config import load_section
            from kestrel_sovereign.security.auto_approve import (
                AutoApprovePolicy,
            )

            self.auto_approve_policy = AutoApprovePolicy.from_config(
                load_section("security"),
                self.permission_store,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to human approval
            logger.warning(
                "SecurityFeature: auto-approve policy unavailable "
                "(falling back to human approval): %s",
                exc,
                exc_info=True,
            )
            self.auto_approve_policy = None

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
            auto_approve_policy=self.auto_approve_policy,
            agent=self.agent,
        )

        # Create security hook
        self.security_hook = SecurityHook(
            self.permission_store, self.approval_queue
        )

        # Initialize lock for async init
        self._init_lock = asyncio.Lock()

        # Complete async initialization directly (no longer need to schedule as task)
        await self._async_init()

        # #1936 — enable non-interactive approval for tagged test instances.
        await self._maybe_enable_test_instance_auto_approve()

        logger.info("SecurityFeature initialized (async setup pending)")

    async def _maybe_enable_test_instance_auto_approve(self) -> bool:
        """Enable non-interactive approval for a tagged test instance (#1936).

        A headless test agent (driven via ``kestrel ask``) has no human to
        answer the interactive approval queue, so every ASK-level tool falls
        into the ~20-minute approval timeout (#406) and is recorded as a user
        denial — exactly how the dogfooding loop's compute drive saw a clean
        script rejected ("User denied execution (scope: timeout)").

        Rather than invent a parallel approval path, reuse the framework's
        existing, audited global auto-mode: with it enabled, ``get_permission``
        promotes ASK/SESSION/default to AUTO while DENY and ALWAYS_ASK remain
        hard policy rails, and the tool flows through the established AUTO path
        in both the SecurityHook and the ApprovalQueue. Every other PreToolUse
        hook (constitutional / honesty / security) still runs and can block —
        this only resolves the *human-confirmation* gate, and each
        auto-approval is audited (``decision="auto_mode_allowed"``).

        Strictly scoped: requires BOTH ``is_test_instance`` AND an explicit
        operator opt-in (``KESTREL_TEST_AUTO_APPROVE``). A sovereign/production
        agent (``is_test_instance is False``) can never reach this; a test
        instance launched without the env opt-in stays fully interactive.

        Returns True iff auto-mode was enabled (for callers/tests).
        """
        if self.permission_store is None:
            return False
        if not bool(getattr(self.agent, "is_test_instance", False)):
            return False
        if not _test_auto_approve_opt_in():
            return False

        self.permission_store.set_global_auto_mode(True)
        agent_name = getattr(self.agent, "_agent_name", None) or getattr(
            self.agent, "did", "test-instance"
        )
        try:
            await self.permission_store.log_decision(
                feature_name="SecurityFeature",
                tool_name="non_interactive_approval",
                action="auto_mode_config",
                decision="auto_mode_allowed",
                user_choice="is_test_instance_opt_in",
            )
        except Exception:  # noqa: BLE001 - auditing is best-effort, never fatal
            logger.debug("Failed to audit test-instance auto-approve", exc_info=True)
        logger.warning(
            "SecurityFeature: NON-INTERACTIVE approval ENABLED for test "
            "instance %s (#1936) — ASK-level tools auto-approve after hooks; "
            "DENY/ALWAYS_ASK still hard-stop. This MUST never be a "
            "sovereign/production agent.",
            agent_name,
        )
        return True

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
        # One-time consolidation of legacy snake-case/alias rows into the
        # canonical PascalCase rows the orchestrator now looks up (#1427).
        # Without this, agents already in operation lose previously-granted
        # tools (e.g. ``computer_use.fs_read`` → ``ComputerUseFeature.fs_read``)
        # the next time the security hook runs.
        aliases: Dict[str, str] = {}
        for feature in agent.features.values():
            tool_alias = getattr(feature, "tool_name", None)
            if not isinstance(tool_alias, str) or not tool_alias:
                continue
            canonical = getattr(feature, "name", type(feature).__name__)
            if not canonical or canonical == tool_alias:
                continue
            aliases[tool_alias] = canonical
        try:
            await self.permission_store.migrate_legacy_feature_aliases(aliases)
        except Exception as exc:  # never fail startup on migration
            logger.warning(
                "Legacy permission alias migration skipped: %s", exc,
            )
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
        """Register all agent tools with sensible per-feature default permissions.

        Demo servers (KESTREL_DEMO_SERVER=1, set by ``kestrel demo run``) use
        ALLOW as the feature-wide baseline — Playwright demos can't click
        through an interactive approval modal, and the demo agent runs in an
        isolated DB so the broader grants are scoped correctly (#897). An SDK
        feature-wide ALWAYS_ASK or DENY declaration may tighten that baseline,
        and explicit per-tool declarations retain their normal precedence.

        Outside demo mode, an active SDK ``FeaturePermissionDefaults``
        descriptor composes with an explicit Sovereign per-feature baseline:
        ordinary declarations cannot weaken that baseline, while ALWAYS_ASK
        or DENY may tighten it. An unmapped feature retains its SDK declaration
        instead of composing against the conservative legacy ASK fallback.
        Features not yet declaring a default use
        ``default_permission_for_feature`` as the migration baseline: core boot
        features get ALLOW so a fresh agent isn't paralyzed in an approval-
        modal loop on first turn (#406, Meridian incident), while risky or
        unmapped legacy features remain ASK. Static per-tool rails compose
        independently below.
        """
        if not hasattr(self.agent, "features"):
            return

        assert_sdk_permission_level_parity()
        is_demo_server = os.environ.get("KESTREL_DEMO_SERVER", "").lower() in (
            "1", "true", "yes",
        )

        for feature_name, feature in self.agent.features.items():
            if feature_name == "SecurityFeature":
                continue
            await self.register_feature_tools(
                feature_name,
                feature,
                is_demo_server=is_demo_server,
            )

        for tool_name in ("commandExecution", "fileChange"):
            await self.permission_store.register_tool(
                feature_name="codex_native",
                tool_name=tool_name,
                default_level=PermissionLevel.ALWAYS_ASK,
            )

        if is_demo_server:
            logger.info(
                "Registered demo permission baseline ALLOW; SDK declarations "
                "and static per-tool rails may tighten it "
                "(KESTREL_DEMO_SERVER=1)"
            )
        else:
            logger.info(
                "Registered all tools with per-feature default permissions"
            )

    async def register_feature_tools(
        self,
        feature_name: str,
        feature: Feature,
        *,
        is_demo_server: Optional[bool] = None,
    ) -> None:
        """Register one feature's tool and dispatch permission rows.

        This is the shared owner-bound seam for boot's all-feature pass and a
        successful runtime enable. Runtime activation calls it after the SDK
        contribution is active but before hooks, A2A dispatch, or promoted
        direct tools become callable, so contributed hard defaults take effect
        immediately without rescanning unrelated features.
        """
        # SDK features declare permission defaults with a separate enum. Do
        # not consume even a currently familiar value unless the *complete*
        # declaration and enforcement vocabularies still agree.
        assert_sdk_permission_level_parity()
        if is_demo_server is None:
            is_demo_server = os.environ.get(
                "KESTREL_DEMO_SERVER", ""
            ).lower() in ("1", "true", "yes")

        from kestrel_sovereign.features.contribution_runtime import (
            PermissionDefaultsRegistry,
        )

        declared_defaults = None
        registry = getattr(self.agent, "permission_defaults_registry", None)
        # ``agent`` is MagicMock-backed in some legacy embedders/tests;
        # require the lifecycle registry rather than treating an arbitrary
        # dynamically-created ``.get`` result as a security declaration.
        if isinstance(registry, PermissionDefaultsRegistry):
            declared_defaults = registry.get(feature_name)

        mapped_feature = feature_name in _DEFAULT_PERMISSION_BY_FEATURE
        if is_demo_server:
            feature_default = PermissionLevel.ALLOW
            if declared_defaults is not None:
                feature_default = compose_restrictive_permission(
                    feature_default,
                    PermissionLevel(declared_defaults.feature_default.value),
                )
        elif declared_defaults is not None:
            declared_feature_default = PermissionLevel(
                declared_defaults.feature_default.value
            )
            if mapped_feature:
                feature_default = compose_restrictive_permission(
                    default_permission_for_feature(feature_name),
                    declared_feature_default,
                )
            else:
                feature_default = declared_feature_default
        else:
            feature_default = default_permission_for_feature(feature_name)

        # Register the inner @tool methods. SDK declarations compose with,
        # rather than replace, Sovereign's static rails. The static map
        # includes destructive approval floors and established ALLOW
        # migrations needed by unattended reconciliation.
        for tool_obj in feature.get_tools():
            static_override = default_permission_for_tool(
                feature_name, tool_obj.name
            )
            if declared_defaults is not None:
                declared_override = declared_defaults.tool_overrides.get(
                    tool_obj.name
                )
                tool_override = (
                    None
                    if declared_override is None
                    else PermissionLevel(declared_override.value)
                )
            else:
                tool_override = None

            if tool_override is None:
                declared_default = feature_default
            elif not is_demo_server and mapped_feature:
                # A mapped feature's already-composed Sovereign baseline also
                # clamps per-tool declarations. In particular, ALLOW cannot
                # bypass an explicit risky-feature ASK rail merely because no
                # exact static per-tool entry exists. Hard declarations may
                # still tighten through the same composition policy.
                declared_default = compose_restrictive_permission(
                    feature_default,
                    tool_override,
                )
            else:
                # Preserve the reviewed demo behavior and the SDK contract for
                # unmapped extracted features: their exact tool declaration is
                # authoritative unless an exact static per-tool rail follows.
                declared_default = tool_override

            if static_override is None:
                effective_default = declared_default
            else:
                effective_default = compose_restrictive_permission(
                    static_override,
                    declared_default,
                )
            await self.permission_store.register_tool(
                feature_name=feature_name,
                tool_name=tool_obj.name,
                default_level=effective_default,
                # Every static override retains its persisted-row upgrade
                # semantics, including ALLOW rails for unattended status
                # and release work. Ordinary contributed defaults remain
                # insert-only unless they are DENY/ALWAYS_ASK.
                hardened=(
                    static_override is not None
                    or effective_default
                    in {PermissionLevel.DENY, PermissionLevel.ALWAYS_ASK}
                ),
            )

        # Also register the feature-as-subagent dispatch entry. It carries the
        # feature baseline and therefore must use the same hardening predicate
        # as inner tools: a stale ALLOW row cannot outlive a newly-declared
        # feature-wide DENY or ALWAYS_ASK rail.
        subagent_tool_name = getattr(feature, "tool_name", None)
        if subagent_tool_name and not isinstance(subagent_tool_name, property):
            await self.permission_store.register_tool(
                feature_name=feature_name,
                tool_name=subagent_tool_name,
                default_level=feature_default,
                hardened=feature_default in {
                    PermissionLevel.DENY,
                    PermissionLevel.ALWAYS_ASK,
                },
            )

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
            "auto_all": "◆",
            "deny_all": "☒",
            "always_ask_all": "◇",
            "ask_all": "☐",
            "session_all": "◑",
            "mixed": "◐",
        }
        level_icon = {
            "allow": "☑",
            "auto": "◆",
            "deny": "☒",
            "always_ask": "◇",
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
            "\nLegend: ☑=Allow ◆=Auto ◇=Always Ask ☒=Deny ☐=Ask ◑=Session ◐=Mixed"
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
        description=(
            "Set permission level for a tool or all tools in a feature. "
            "level is one of: allow, auto, always_ask, deny, ask, session. "
            "Call list_permissions to enumerate valid feature_name/tool_name "
            "values — setting a permission on an unregistered name does not "
            "take effect."
        ),
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
            feature_name: Name of the feature (e.g., "WalletAgent"). Must be a
                registered feature — call ``list_permissions`` to enumerate
                valid names.
            tool_name: Name of the tool (optional, sets all if omitted). When
                provided, must be a registered tool of ``feature_name``.
            level: Permission level - "allow", "auto", "always_ask", "deny", "ask", or "session"
        """
        try:
            perm_level = PermissionLevel(level)
        except ValueError:
            return ToolResult.failed(
                f"Invalid level '{level}'. Use: allow, auto, always_ask, deny, ask, session"
            )

        # Validate the target against the registered permission tree before
        # writing anything. Every real feature/tool is seeded into the store at
        # startup (``_register_all_tools`` -> ``register_tool``), so the tree is
        # the authoritative registry of valid names. Writing a permission row
        # for a typo'd / nonexistent name silently succeeds but never matches a
        # real tool — a dead row that gives the agent a false "permission set"
        # with no effect (#1946). Surface it as PARTIAL and persist nothing,
        # pointing the agent at ``list_permissions`` for discovery. There is no
        # legitimate forward-grant pattern: future features get their rows from
        # startup registration, never from a pre-emptive set_permission.
        unknown = await self._unknown_target(feature_name, tool_name)
        if unknown is not None:
            return unknown

        try:
            if tool_name:
                await self.permission_store.set_permission(
                    feature_name, tool_name, perm_level
                )
                return ToolResult.ok(
                    confirmation=self._format_permission_confirmation(
                        feature_name, tool_name, level
                    ),
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
                confirmation=self._format_permission_confirmation(
                    feature_name, None, level
                ),
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
        description="Approve a pending request. scope: 'once', 'session', or 'always'.",
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
        scope = _normalize_choice(scope, {
            "single": "once", "one": "once", "this_session": "session",
            "forever": "always", "permanent": "always", "permanently": "always",
        })
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

        # submit_decision returns a result object: ``in_memory`` means the
        # queue accepted the decision, while ``persisted`` reports whether
        # the durable scope/audit write completed.
        decision = await self.approval_queue.submit_decision(full_id, True, scope)
        if not decision.in_memory:
            return ToolResult.failed(
                f"Request '{request_id}' is no longer pending "
                "(timeout, cancellation, or already decided)",
                data={"request_id": full_id, "decision_attempted": "approved"},
            )

        # The original tool-call awaiter had already been cancelled (Cloud Run
        # request timeout / SSE disconnect / abort). The scope rule was still
        # recorded for future calls, but THIS call is orphaned — report a
        # partial so the caller knows the approval didn't resume anything and
        # the action must be re-run. (#2558)
        if decision.awaiter_gone:
            return ToolResult.partial(
                confirmation=(
                    f"Recorded approval of {request_id[:8]} ({scope}), but the "
                    "original request had already been cancelled — re-run the "
                    "action to complete it now."
                ),
                error=(
                    "The tool call awaiting this approval had already ended "
                    "(timeout/disconnect); the scope rule applies to future "
                    "calls but this call is orphaned."
                ),
                data={
                    "request_id": full_id,
                    "scope": scope,
                    "decision": "approved",
                    "scope_persisted": decision.persisted,
                    "awaiter_gone": True,
                    "persistence_error": decision.error,
                },
            )

        if decision.persisted:
            return ToolResult.ok(
                confirmation=f"Approved {request_id[:8]} ({scope})",
                data={
                    "request_id": full_id,
                    "scope": scope,
                    "decision": "approved",
                    "scope_persisted": True,
                    "awaiter_gone": False,
                },
            )
        return ToolResult.partial(
            confirmation=(
                f"Approved {request_id[:8]} for this request "
                f"(scope={scope} persistence failed)"
            ),
            error=(
                f"scope={scope} decision was accepted in memory, but the "
                f"durable permission/audit write failed: {decision.error}"
            ),
            data={
                "request_id": full_id,
                "scope": scope,
                "decision": "approved",
                "scope_persisted": False,
                "awaiter_gone": False,
                "persistence_error": decision.error,
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

        # Carry distinct provenance for an explicit user denial. Operator/
        # auto policy DENY resolves through ``request_approval``'s early
        # return as ``(False, "denied")``; a human pressing deny here must
        # be distinguishable from that so downstream consumers (e.g. the
        # Talon verifier's ``classify_denial``) can surface ``blocked_by_user``
        # only for a real user denial and never mislabel a policy/sandbox
        # block as one (#1542). The resulting ``request.user_decision`` still
        # audits as ``user_denied`` via ``_persist_decision``.
        decision = await self.approval_queue.submit_decision(
            full_id, False, "user_denied"
        )
        if decision.in_memory and decision.persisted:
            return ToolResult.ok(
                confirmation=f"Denied {request_id[:8]}",
                data={
                    "request_id": full_id,
                    "decision": "user_denied",
                    "scope_persisted": True,
                    # The awaiter may already be gone (the call ended before
                    # the deny landed); harmless for a denial, but surfaced so
                    # callers can distinguish it. (#2558)
                    "awaiter_gone": decision.awaiter_gone,
                },
            )
        if decision.in_memory:
            return ToolResult.partial(
                confirmation=f"Denied {request_id[:8]}",
                error=(
                    "Denial was accepted in memory, but the audit write "
                    f"failed: {decision.error}"
                ),
                data={
                    "request_id": full_id,
                    "decision": "user_denied",
                    "scope_persisted": False,
                    "awaiter_gone": decision.awaiter_gone,
                    "persistence_error": decision.error,
                },
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
                "auto_mode_allowed": "◆",
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

    @tool(
        name=SEARCH_TOOL_NAME,
        description=(
            "Search your own recorded tool calls for ones matching a "
            "description — 'have I already filed/commented/run this?'"
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!security-audit-search",
    )
    async def security_audit_search(
        self,
        query: str,
        tool_name: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 20,
    ) -> ToolResult:
        """Find prior tool calls of your own that match ``query`` (#3107).

        ``security_audit`` lists what happened recently and deliberately omits
        the recorded arguments, because dumping every row's arguments into the
        context is disclosure nobody asked for. That filter also removed the
        only field that can answer "have I already done this?", so a later turn
        rediscovered work it had already done and filed the same issue twice.

        This asks the other question. The caller must already describe what it
        is looking for, so what comes back is a match to a description the
        caller supplied — not whatever happens to be recent.

        **That footing only holds while the query is a description.** A query
        broad enough to match most of the log is not one, and answering it with
        a page of arguments would rebuild the unbounded disclosure by another
        route. So the match is counted first: past
        ``MAX_DISCLOSING_MATCHES`` the count comes back and the arguments do
        not. Narrowing is the caller's move, and nothing leaves the database
        in the meantime.

        **A match is an AUTHORIZATION, not a completion.** The security hook
        logs at ``PRE_TOOL_USE``, so a row says the call was allowed to run —
        not that its body succeeded. Denied and timed-out attempts are rows
        too. Read ``decision``; a match is evidence you TRIED, and where a
        paired ``<tool>.outcome`` row exists that is what carries the result.

        What the disclosure ceiling actually is: ``args_summary`` was masked
        and truncated by ``summarize_args`` when it was written, and
        ``GET /api/security/audit`` already returns that same value verbatim.
        Nothing here widens what was persisted. The masking is a key-name
        heuristic, so it is a good filter and not a guarantee.

        The truncation is 500 characters, but was **200** on the
        ``SecurityHook`` path until that override was removed — so rows written
        before then are cut shorter, and a search cannot see past whatever cut
        its own row got. That is why an empty result says so rather than
        reading as proof.

        Args:
            query: What to look for — a repo, an issue number, a title
                   fragment, a command. Matched case-insensitively against the
                   recorded arguments and the tool name.
            tool_name: Restrict to one tool, e.g. "create_github_issue".
            days: Only consider the last N days.
            limit: Maximum matches to return, newest first. It cannot widen
                   disclosure — the bound is the total-match gate, not this.
        """
        needle = (query or "").strip()
        if not needle:
            return ToolResult.failed(
                "query must be a non-empty description of what to look for; "
                "use security_audit for an unfiltered recent listing"
            )

        try:
            limit_val = int(limit)
        except (TypeError, ValueError):
            return ToolResult.failed(f"limit must be an integer, got {limit!r}")
        if limit_val < 1:
            return ToolResult.failed("limit must be >= 1")
        # No clamp on ``limit``, deliberately. Mutation testing showed one had
        # no observable effect, and it was right: the disclosure bound is the
        # TOTAL-match gate below, and ``search_audit_log`` only runs when the
        # total is already within ``MAX_DISCLOSING_MATCHES``. A caller raising
        # ``limit`` therefore cannot widen what is disclosed — it can only ask
        # for a page at least as large as a set that is already bounded. A
        # clamp here would have read as the thing enforcing the bound while
        # enforcing nothing, which is worse than its absence.

        if days is not None:
            try:
                days_val: Optional[int] = int(days)
            except (TypeError, ValueError):
                return ToolResult.failed(f"days must be an integer, got {days!r}")
            if days_val < 1:
                return ToolResult.failed("days must be >= 1")
        else:
            days_val = None

        scope = [f"query={needle!r}"]
        if tool_name:
            scope.append(f"tool={tool_name}")
        if days_val is not None:
            scope.append(f"days={days_val}")
        scope_text = ", ".join(scope)

        try:
            # One statement, one snapshot: the store fetches a single row of
            # headroom past the bound and reports whether it existed. A
            # separate COUNT could pass while the page query, on its own
            # connection, saw more rows and returned their arguments anyway.
            matches, too_broad = await self.permission_store.search_audit_log(
                needle,
                tool_name=tool_name or None,
                days=days_val,
                limit=MAX_DISCLOSING_MATCHES,
            )
        except Exception as e:
            logger.error(f"security_audit_search failed: {e}", exc_info=True)
            return ToolResult.failed(str(e))

        # The gate must see the full bound, so the QUERY always asks for
        # MAX_DISCLOSING_MATCHES; the caller's smaller limit is applied after.
        # Asking the store for `limit` directly would let limit=1 report
        # "not too broad" for a query matching four hundred rows.
        matches = matches[:limit_val]

        if too_broad:
            # Deliberately no arguments: a query this broad is not a
            # description of one prior action, and returning a page of it would
            # be the unbounded dump wearing a query parameter.
            return ToolResult.ok(
                confirmation=(
                    f"More than {MAX_DISCLOSING_MATCHES} recorded calls "
                    f"matched ({scope_text}) — too broad to answer with "
                    "arguments.\n"
                    "Narrow it: add a distinctive phrase from the thing "
                    "itself, a tool_name, or a days window. Nothing was "
                    "returned from the log."
                ),
                data={
                    "count": None,
                    "too_broad": True,
                    "max_disclosing_matches": MAX_DISCLOSING_MATCHES,
                    "query": needle,
                    "tool_name": tool_name or "",
                    "days": days_val,
                    "matches": [],
                },
            )

        if not matches:
            # An empty result is NOT proof the thing was never done. Say so:
            # this searches recorded arguments, masked and truncated to 500
            # characters, so a match past that cut is invisible here.
            return ToolResult.ok(
                confirmation=(
                    f"No recorded tool call matched ({scope_text}).\n"
                    "This searches the masked argument summary, which is "
                    "truncated when written — at 500 characters now, and at "
                    "200 for rows written before that was unified, so an older "
                    "row is cut shorter than a new one. A distinguishing "
                    "detail past its own row's cut cannot match. Absence here "
                    "is weak evidence, not proof you never did it."
                ),
                data={
                    "count": 0,
                    "too_broad": False,
                    "query": needle,
                    "tool_name": tool_name or "",
                    "days": days_val,
                    "limit_requested": limit_val,
                    "matches": [],
                },
            )

        lines = [
            f"{len(matches)} prior ATTEMPT(s) matched ({scope_text}) — each row "
            "records that the call was authorized, not that it succeeded:\n"
        ]
        for entry in matches:
            lines.append(
                f"  {entry['timestamp']}  {entry['feature']}.{entry['tool']} "
                f"[{entry['decision']}]"
            )
            if entry["args_summary"]:
                lines.append(f"     {entry['args_summary']}")

        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "count": len(matches),
                "too_broad": False,
                "query": needle,
                "tool_name": tool_name or "",
                "days": days_val,
                "limit_requested": limit_val,
                "matches": matches,
            },
        )

    async def _unknown_target(
        self,
        feature_name: str,
        tool_name: Optional[str],
    ) -> Optional[ToolResult]:
        """Validate a set_permission target against the registered tree.

        Returns ``None`` when ``feature_name`` (and ``tool_name`` if given) is a
        registered feature/tool, meaning the caller may proceed to persist.
        Otherwise returns a PARTIAL ToolResult that persists nothing and points
        the agent at ``list_permissions`` for the valid names — so a typo'd or
        nonexistent name surfaces a clear non-OK result instead of writing a
        dead permission row (#1946).

        If the tree cannot be read, returns ``None`` (fail open): an unreadable
        registry must not block legitimate permission changes, and the
        downstream ``set_permission`` path will surface any real store error.
        """
        try:
            tree = await self.permission_store.get_permission_tree()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "set_permission could not validate target against permission "
                "tree (%s); proceeding without name validation",
                exc,
            )
            return None

        # Match against every casing/alias variant the store itself resolves
        # at read/write time (snake/Pascal + registered cross-form aliases),
        # so a legitimate legacy-alias write (e.g. ``computer_use`` for
        # ``ComputerUseFeature``) is accepted rather than rejected as unknown
        # (codex review #1946 P2).
        variants = self.permission_store.feature_name_variants(feature_name)
        feature = next(
            (f for f in tree if f.feature_name in variants), None
        )
        if feature is None:
            return ToolResult.partial(
                confirmation=(
                    f"No permission was changed: feature '{feature_name}' is "
                    "not a registered feature."
                ),
                error=(
                    f"Unknown feature_name '{feature_name}'. Call "
                    "list_permissions to see valid feature/tool names. Nothing "
                    "was persisted."
                ),
                data={
                    "feature_name": feature_name,
                    "tool_name": tool_name,
                    "persisted": False,
                    "reason": "unknown_feature",
                },
            )

        if tool_name and not any(
            t.tool_name == tool_name for t in feature.tools
        ):
            return ToolResult.partial(
                confirmation=(
                    f"No permission was changed: '{tool_name}' is not a "
                    f"registered tool of feature '{feature_name}'."
                ),
                error=(
                    f"Unknown tool_name '{tool_name}' for feature "
                    f"'{feature_name}'. Call list_permissions to see valid "
                    "tool names. Nothing was persisted."
                ),
                data={
                    "feature_name": feature_name,
                    "tool_name": tool_name,
                    "persisted": False,
                    "reason": "unknown_tool",
                },
            )

        return None

    def _format_permission_confirmation(
        self,
        feature_name: str,
        tool_name: Optional[str],
        level: str,
    ) -> str:
        target = (
            f"{feature_name}.{tool_name}"
            if tool_name
            else f"all tools in {feature_name}"
        )
        confirmation = f"Set {target} to {level}"
        if level == PermissionLevel.AUTO.value:
            confirmation += (
                "\n\nWarning: Auto mode skips human approval when earlier "
                "constitutional, honesty, and security hooks do not flag the "
                "call. It is not a guarantee that every risk has been detected."
            )
        return confirmation

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
