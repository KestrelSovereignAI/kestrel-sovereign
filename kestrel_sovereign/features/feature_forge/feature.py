"""FeatureForgeFeature — the feature that creates features (issue #2434).

A governed, constitutionally-gated pathway for a Kestrel agent to design and
scaffold its own features. Self-extension becomes a structured, audited primitive
instead of an ad-hoc issue→Talon loop:

    forge_feature   -> scaffold a package from a declarative spec  (draft)
    forge_validate  -> run the Iron Rule gate (narrow-only)        (validated)
    forge_register  -> queue for Sovereign approval                (pending_approval)
    list_forged / forge_status -> observe the pipeline

Security model:
    * No self-granting. The forge can only compose capabilities the agent already
      holds, narrowed (:mod:`iron_rule`). Widen attempts are rejected at
      validation time, mirroring Book III Section 3.
    * Sovereign approval is a hard gate before any forged feature loads
      (Amendment I). Forged packages are INERT until approved — scaffolded
      outside the discovery path with no entry point (:mod:`store`).
    * Every forge operation is written to the audit trail (Amendment III).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool

from .iron_rule import (
    BASELINE_CAPABILITIES,
    IronRuleVerdict,
    validate_narrowing,
)
from .scaffold import SpecError, parse_spec, render_package
from .store import (
    STATE_APPROVED,
    STATE_BLOCKED,
    STATE_DRAFT,
    STATE_PENDING,
    STATE_REJECTED,
    STATE_VALIDATED,
    ForgeRecord,
    ForgeStore,
    ForgeStoreError,
)

logger = logging.getLogger(__name__)

# Approval-queue scopes that denote an *explicit human denial* — the only
# outcomes that may mark a forged feature permanently ``rejected`` (#1542).
# ``ApprovalQueue.request_approval`` returns ``user_denied`` from the deny tool
# and ``once``/``session``/``always`` (with approved=False) from the web-UI deny
# endpoint. Everything else — ``no_approver`` (headless/no approver attached),
# ``timeout``, ``cancelled``/``cancelled_all``, and policy ``denied`` — is NOT a
# user denial and must not be laundered into a terminal rejection.
_USER_DENIAL_SCOPES = frozenset({"user_denied", "once", "session", "always"})


def _is_user_denial(scope: Optional[str]) -> bool:
    return (scope or "").strip().lower() in _USER_DENIAL_SCOPES

# Privileged capability -> the loaded feature class name(s) whose presence means
# the platform grants the agent that capability. If none of the named features is
# loaded, the capability is NOT held and requesting it is a widen attempt.
_PRIVILEGED_GRANTED_BY: Dict[str, tuple] = {
    "shell_execution": ("ComputeFeature", "ComputerUseFeature"),
    "filesystem_read": ("ComputerUseFeature",),
    "filesystem_write": ("ComputerUseFeature",),
    "spawn_agent": ("SpawnFeature",),
    "network_outbound": ("WebSearchFeature", "BridgeFeature", "WebhookFeature"),
    "wallet_spend": ("WalletFeature", "WalletAgent"),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeatureForgeFeature(Feature):
    """Governed self-extension: design, validate, and register forged features."""

    def __init__(self, agent):
        super().__init__(agent)
        self.store: Optional[ForgeStore] = None
        # Background tasks awaiting Sovereign approval decisions, keyed by name.
        self._approval_tasks: Dict[str, asyncio.Task] = {}

    @property
    def tool_description(self) -> str:
        return (
            "Design and scaffold new agent features under constitutional "
            "governance: forge a feature package from a spec, run the Iron Rule "
            "gate (narrow-only permissions), and queue it for Sovereign approval. "
            "Forged features stay inert until approved."
        )

    @property
    def promote_tools_on_startup(self) -> bool:
        # Meta/agent-management surface — expose the individual forge tools
        # directly at startup rather than a single dispatcher.
        return True

    async def initialize(self) -> None:
        self.store = ForgeStore(self._forge_root())
        logger.info("FeatureForgeFeature initialized (forge root: %s)", self.store.root)

    async def shutdown(self) -> None:
        for task in list(self._approval_tasks.values()):
            if not task.done():
                task.cancel()
        self._approval_tasks.clear()

    # =========================================================================
    # Roots & capabilities
    # =========================================================================

    def _forge_root(self) -> Path:
        """Resolve the forge root — inside the agent data dir, never the package.

        Placing forged packages here (not under ``kestrel_sovereign/features/``)
        is what keeps them inert: feature discovery only scans the package
        directory and installed entry points, neither of which reaches here.
        """
        override = os.environ.get("KESTREL_FORGE_ROOT")
        if override:
            return Path(override)

        # Derive from the agent's storage path (a DB file OR directory).
        storage_path = getattr(self.agent, "storage_path", None)
        if storage_path:
            p = Path(storage_path)
            base = p if p.is_dir() else p.parent
            return base / "forged_features"

        db_path = os.environ.get("KESTREL_DB_PATH")
        if db_path:
            return Path(db_path) / "forged_features"

        return Path.cwd() / "forged_features"

    def _loaded_feature_names(self) -> set:
        features = getattr(self.agent, "features", None)
        if isinstance(features, dict):
            return set(features.keys())
        return set()

    def granted_capabilities(self) -> List[str]:
        """The capabilities the platform currently grants this agent.

        Baseline capabilities are always held; privileged ones are held only when
        the feature that provides them is loaded. This is the ``granted`` set the
        Iron Rule gate narrows against.
        """
        loaded = self._loaded_feature_names()
        granted = set(BASELINE_CAPABILITIES)
        for capability, providers in _PRIVILEGED_GRANTED_BY.items():
            if any(name in loaded for name in providers):
                granted.add(capability)
        return sorted(granted)

    # =========================================================================
    # Audit (Amendment III)
    # =========================================================================

    async def _audit(
        self,
        operation: str,
        feature_name: str,
        decision: str,
        detail: Optional[str] = None,
    ) -> None:
        """Write a forge operation to the security audit trail (Amendment III).

        Best-effort: an audit-write failure must never abort the forge operation
        itself, but is logged loudly since a missing audit row is a governance
        gap.
        """
        security = None
        features = getattr(self.agent, "features", None)
        if isinstance(features, dict):
            security = features.get("SecurityFeature")
        store = getattr(security, "permission_store", None) if security else None
        if store is None:
            logger.debug(
                "Forge audit (no permission store): %s %s -> %s",
                operation, feature_name, decision,
            )
            return
        try:
            await store.log_decision(
                feature_name="FeatureForgeFeature",
                tool_name=operation,
                action="forge",
                decision=decision,
                user_choice=feature_name,
                args_summary=detail,
            )
        except Exception as exc:  # noqa: BLE001 - auditing never fatal
            logger.warning(
                "Forge audit write failed for %s %s: %s",
                operation, feature_name, exc, exc_info=True,
            )

    def _history_entry(self, state: str, note: str = "") -> Dict[str, str]:
        return {"state": state, "note": note, "at": _utc_now_iso()}

    # =========================================================================
    # Tools
    # =========================================================================

    @tool(
        name="forge_feature",
        description=(
            "Scaffold a complete, inert feature package from a declarative spec "
            "(name, purpose, tools with parameters, required permissions). The "
            "spec may be a JSON object or a JSON string. Produces a loadable "
            "package in the draft state; it does not load or execute."
        ),
        category=ToolCategory.AGENT_MANAGEMENT,
        command_prefix="!forge",
    )
    async def forge_feature(self, spec: Any) -> ToolResult:
        """Scaffold a forged feature package from ``spec``.

        Args:
            spec: Declarative feature spec — a JSON object (or JSON string) with
                ``name``, optional ``purpose``, a ``tools`` list (each with
                ``name``, ``description``, ``parameters``), and a ``permissions``
                list of requested capabilities.
        """
        if self.store is None:
            return ToolResult.failed("Forge store not initialized")

        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except json.JSONDecodeError as exc:
                return ToolResult.failed(f"spec is not valid JSON: {exc}")

        try:
            parsed = parse_spec(spec)
        except SpecError as exc:
            return ToolResult.failed(f"Invalid spec: {exc}")

        if self.store.exists(parsed.module_name):
            return ToolResult.failed(
                f"A forged feature named '{parsed.module_name}' already exists. "
                "Use forge_status to inspect it."
            )

        record = ForgeRecord(
            name=parsed.module_name,
            display_name=parsed.name,
            class_name=parsed.class_name,
            state=STATE_DRAFT,
            spec=parsed.to_dict(),
            history=[self._history_entry(STATE_DRAFT, "scaffolded")],
            created_at=_utc_now_iso(),
            updated_at=_utc_now_iso(),
        )
        try:
            files = self.store.save_scaffold(record, render_package(parsed))
        except (ForgeStoreError, OSError) as exc:
            return ToolResult.failed(f"Failed to write scaffold: {exc}")

        await self._audit(
            "forge_feature", parsed.class_name, "drafted",
            detail=f"{len(parsed.tools)} tool(s), perms={parsed.permissions}",
        )

        return ToolResult.ok(
            confirmation=(
                f"Forged '{parsed.class_name}' in draft state "
                f"({len(files)} files). Inert until approved. "
                f"Run forge_validate('{parsed.module_name}') next."
            ),
            data={
                "name": parsed.module_name,
                "class_name": parsed.class_name,
                "state": STATE_DRAFT,
                "files": files,
                "pkg_dir": str(self.store.pkg_dir(parsed.module_name)),
                "requested_permissions": parsed.permissions,
            },
        )

    @tool(
        name="forge_validate",
        description=(
            "Run the Iron Rule gate over a forged feature: reject any spec that "
            "requests capabilities beyond what the platform already grants the "
            "agent (narrow only, never widen). On success the feature advances to "
            "the validated state."
        ),
        category=ToolCategory.AGENT_MANAGEMENT,
        command_prefix="!forge-validate",
    )
    async def forge_validate(self, feature_name: str) -> ToolResult:
        """Validate a forged feature against the Iron Rule.

        Args:
            feature_name: The forged feature's module name (from forge_feature).
        """
        record = self._require_record(feature_name)
        if isinstance(record, ToolResult):
            return record

        requested = record.spec.get("permissions") or []
        granted = self.granted_capabilities()
        verdict: IronRuleVerdict = validate_narrowing(requested, granted)

        record.verdict = verdict.to_dict()
        record.updated_at = _utc_now_iso()

        if not verdict.valid:
            # Stay in draft (or revert to it) — a rejected validation must not
            # advance the pipeline.
            record.state = STATE_DRAFT
            record.history = (record.history or []) + [
                self._history_entry(STATE_DRAFT, f"validation rejected: {verdict.reason}")
            ]
            self.store.save(record)
            await self._audit(
                "forge_validate", record.class_name, "rejected",
                detail=verdict.reason,
            )
            return ToolResult.failed(
                verdict.reason,
                data={"name": record.name, "verdict": record.verdict, "state": record.state},
            )

        record.state = STATE_VALIDATED
        record.history = (record.history or []) + [
            self._history_entry(STATE_VALIDATED, "iron rule passed")
        ]
        self.store.save(record)
        await self._audit(
            "forge_validate", record.class_name, "validated",
            detail=verdict.reason,
        )
        return ToolResult.ok(
            confirmation=(
                f"'{record.class_name}' passed the Iron Rule gate. "
                f"Run forge_register('{record.name}') to queue for approval."
            ),
            data={"name": record.name, "verdict": record.verdict, "state": STATE_VALIDATED},
        )

    @tool(
        name="forge_register",
        description=(
            "Queue a validated forged feature for Sovereign approval via the "
            "SecurityFeature approval queue. The feature stays inert (pending "
            "approval) and advances to 'approved' — meaning its SOURCE is "
            "approved — once the Sovereign approves the request; an explicit "
            "Sovereign denial marks it rejected. Approval does not itself load "
            "or install the feature (that is a separate operator-gated step)."
        ),
        category=ToolCategory.AGENT_MANAGEMENT,
        command_prefix="!forge-register",
    )
    async def forge_register(self, feature_name: str) -> ToolResult:
        """Queue a validated forged feature for Sovereign approval.

        Args:
            feature_name: The forged feature's module name.
        """
        record = self._require_record(feature_name)
        if isinstance(record, ToolResult):
            return record

        # A ``blocked`` record (approval never resolved — no approver / timeout /
        # cancellation) is recoverable: re-registering it re-queues a fresh
        # approval request. A ``rejected`` record (explicit Sovereign denial) is
        # terminal and must be re-validated first.
        if record.state not in (STATE_VALIDATED, STATE_BLOCKED):
            return ToolResult.failed(
                f"'{record.name}' is in state '{record.state}'; it must be "
                "'validated' (or a recoverable 'blocked') before it can be "
                "registered. Run forge_validate first.",
                data={"name": record.name, "state": record.state},
            )

        queue = self._approval_queue()
        record.state = STATE_PENDING
        record.updated_at = _utc_now_iso()
        record.history = (record.history or []) + [
            self._history_entry(STATE_PENDING, "queued for Sovereign approval")
        ]
        self.store.save(record)
        await self._audit(
            "forge_register", record.class_name, "pending_approval",
            detail=f"perms={record.spec.get('permissions')}",
        )

        if queue is None:
            return ToolResult.partial(
                confirmation=(
                    f"'{record.class_name}' is queued (pending_approval) but no "
                    "approval queue is available on this agent, so it cannot be "
                    "approved here. It remains inert."
                ),
                error="SecurityFeature approval queue unavailable",
                data={"name": record.name, "state": STATE_PENDING},
            )

        # Await the Sovereign's decision in the background so this tool returns
        # immediately with the feature inert & pending. The Sovereign answers via
        # the normal SecurityFeature approve/deny flow, which unblocks the task.
        task = asyncio.ensure_future(self._await_approval(record.name))
        self._approval_tasks[record.name] = task

        return ToolResult.ok(
            confirmation=(
                f"'{record.class_name}' queued for Sovereign approval "
                "(pending_approval). It is INERT: approval authorizes its SOURCE "
                "but does not itself load or install the feature (that remains a "
                "separate, operator-gated step)."
            ),
            data={"name": record.name, "state": STATE_PENDING},
        )

    @tool(
        name="list_forged",
        description="List all forged features and their pipeline state.",
        category=ToolCategory.AGENT_MANAGEMENT,
        command_prefix="!forge-list",
    )
    async def list_forged(self) -> ToolResult:
        """List every forged feature with its current pipeline state."""
        if self.store is None:
            return ToolResult.failed("Forge store not initialized")
        records = self.store.list()
        items = [
            {
                "name": r.name,
                "class_name": r.class_name,
                "state": r.state,
                "permissions": r.spec.get("permissions", []),
                "updated_at": r.updated_at,
            }
            for r in records
        ]
        return ToolResult.ok(
            confirmation=f"{len(items)} forged feature(s)",
            data={"count": len(items), "forged": items},
        )

    @tool(
        name="forge_status",
        description=(
            "Show the full pipeline status of one forged feature: state, spec, "
            "Iron Rule verdict, scaffolded files, and state history."
        ),
        category=ToolCategory.AGENT_MANAGEMENT,
        command_prefix="!forge-status",
    )
    async def forge_status(self, feature_name: str) -> ToolResult:
        """Show a forged feature's detailed status.

        Args:
            feature_name: The forged feature's module name.
        """
        record = self._require_record(feature_name)
        if isinstance(record, ToolResult):
            return record
        return ToolResult.ok(
            confirmation=f"'{record.class_name}' is in state '{record.state}'",
            data=record.to_dict(),
        )

    # =========================================================================
    # Internals
    # =========================================================================

    def _require_record(self, feature_name: str):
        """Return the record or a failed ToolResult if missing/store absent."""
        if self.store is None:
            return ToolResult.failed("Forge store not initialized")
        name = self._resolve_name(feature_name)
        record = self.store.get(name) if name else None
        if record is None:
            return ToolResult.failed(
                f"No forged feature named '{feature_name}'. Use list_forged to "
                "see available names."
            )
        return record

    def _resolve_name(self, feature_name: str) -> Optional[str]:
        """Resolve a supplied name to a stored module name (accepts class name)."""
        if not feature_name:
            return None
        if self.store.exists(feature_name):
            return feature_name
        # Accept the PascalCase class name or a "Feature"-suffixed form.
        from .scaffold import to_snake_case

        snake = to_snake_case(feature_name)
        if snake and self.store.exists(snake):
            return snake
        return None

    def _approval_queue(self):
        features = getattr(self.agent, "features", None)
        if not isinstance(features, dict):
            return None
        security = features.get("SecurityFeature")
        return getattr(security, "approval_queue", None) if security else None

    async def _await_approval(self, name: str) -> None:
        """Await the Sovereign's approval decision and finalize the record."""
        queue = self._approval_queue()
        if queue is None:
            return
        record = self.store.get(name)
        if record is None:
            return
        try:
            approved, scope = await queue.request_approval(
                feature_name="FeatureForgeFeature",
                tool_name="approve_forged_feature",
                tool_args={
                    "forged_feature": record.class_name,
                    "capabilities": record.spec.get("permissions", []),
                    "tools": [t.get("name") for t in record.spec.get("tools", [])],
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Forge approval await failed for %s: %s", name, exc)
            return
        finally:
            self._approval_tasks.pop(name, None)

        # Re-read in case the record changed while we waited.
        record = self.store.get(name)
        if record is None:
            return
        if approved:
            record.state = STATE_APPROVED
            record.history = (record.history or []) + [
                self._history_entry(STATE_APPROVED, f"approved (scope={scope})")
            ]
            record.updated_at = _utc_now_iso()
            self.store.save(record)
            await self._audit("forge_approve", record.class_name, "approved", detail=scope)
        elif _is_user_denial(scope):
            # An explicit Sovereign denial — terminal rejection (Amendment I).
            record.state = STATE_REJECTED
            record.history = (record.history or []) + [
                self._history_entry(STATE_REJECTED, f"denied by Sovereign (scope={scope})")
            ]
            record.updated_at = _utc_now_iso()
            self.store.save(record)
            await self._audit("forge_reject", record.class_name, "rejected", detail=scope)
        else:
            # NOT a user denial (no_approver / timeout / cancellation / policy
            # denied). Do not launder a missing decision into a rejection —
            # park the record in the recoverable ``blocked`` state so it can be
            # re-registered once an approver is available (#1542).
            record.state = STATE_BLOCKED
            record.history = (record.history or []) + [
                self._history_entry(
                    STATE_BLOCKED,
                    f"approval unresolved, not a user denial (scope={scope})",
                )
            ]
            record.updated_at = _utc_now_iso()
            self.store.save(record)
            await self._audit("forge_blocked", record.class_name, "blocked", detail=scope)
