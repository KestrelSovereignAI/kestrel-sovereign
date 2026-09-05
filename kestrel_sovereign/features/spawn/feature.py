"""
Spawn Feature — Runtime agent spawning as LLM-callable tools.

Allows the orchestrator LLM to autonomously create child agents,
delegate tasks, retrieve results, and terminate children. Built on
top of the AgentManager's spawn primitives and SpawnMandate for
cryptographic delegation chains.

Architecture:
    Orchestrator LLM → SpawnFeature.spawn_agent(name, purpose, ...)
        → AgentManager.spawn_agent(name, parent_agent, mandate)
            → Inception → Child KestrelAgent running in-process
        ← Child DID returned
    Orchestrator LLM → SpawnFeature.delegate_task(child_name, task)
        → Child agent processes task via its own LLM context
        ← Result stored for later retrieval
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)

_SAFE_AGENT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def _flatten_terminal_outcomes(error: BaseException) -> list[BaseException]:
    """Flatten a lifecycle exception group without inspecting error text."""

    if isinstance(error, BaseExceptionGroup):
        leaves: list[BaseException] = []
        for nested in error.exceptions:
            leaves.extend(_flatten_terminal_outcomes(nested))
        return leaves
    return [error]


def _safe_retained_agent_name(error: object) -> str | None:
    """Return only a canonical public agent name from retained metadata."""

    metadata = getattr(error, "metadata", None)
    candidate = metadata.get("agent") if isinstance(metadata, dict) else None
    if type(candidate) is str and _SAFE_AGENT_NAME_RE.fullmatch(candidate):
        return candidate
    return None


def _safe_termination_agent_name(error: object) -> str | None:
    """Return only a canonical child name from a termination outcome."""

    candidate = getattr(error, "child_name", None)
    if type(candidate) is str and _SAFE_AGENT_NAME_RE.fullmatch(candidate):
        return candidate
    return None


def _absence_finalization_partial_result(
    *,
    child_name: str,
    offboard_runtime: bool,
) -> ToolResult:
    """Report routing finalization without inventing runtime-tree custody.

    A concurrent lifecycle operation may remove manager routing and the parent
    edge before this request receives an authoritative termination result.
    That is enough to finalize local tracking, but proves neither that runtime
    state was retained nor that a destructive request removed it.
    """

    return ToolResult.partial(
        f"Terminated child '{child_name}'.",
        (
            "The child was already absent from manager routing when this "
            "request reconciled local lifecycle tracking. Runtime retention "
            "and offboarding are unknown; reconcile runtime custody before "
            "restart or deprovisioning. Do not retry termination."
        ),
        data={
            "terminated": True,
            "child_name": child_name,
            "agent_removed": True,
            "finalized_from_absence": True,
            "runtime_offboard_requested": offboard_runtime,
            "runtime_offboarded": False,
            "runtime_cleanup_pending": False,
            "runtime_cleanup_state": "custody_unknown",
            "runtime_already_absent": False,
            "runtime_custody_known": False,
            "runtime_retention_unknown": True,
            "operator_action_required": True,
            "retry_termination": False,
        },
    )


def _termination_partial_result(
    *,
    child_name: str,
    offboard_runtime: bool,
    manager: object,
    error: BaseException,
    retained_error_type: type[BaseException],
    not_performed_error_type: type[BaseException],
    reconciliation_error_type: type[BaseException],
    termination_not_performed_error_type: type[BaseException],
    public_exception_type_name: Callable[[BaseException], str],
) -> ToolResult | None:
    """Build a safe terminal tool outcome, or decline unsupported failures.

    Only manager-typed custody/reconciliation outcomes and cancellation may be
    summarized. Arbitrary exceptions are re-raised by the caller because they
    do not prove that shutdown/unpublication completed and may represent a
    programmer or security-boundary failure.
    """

    leaves = _flatten_terminal_outcomes(error)
    retained = [item for item in leaves if isinstance(item, retained_error_type)]
    not_performed = [
        item for item in leaves if isinstance(item, not_performed_error_type)
    ]
    reconciliation = [
        item for item in leaves if isinstance(item, reconciliation_error_type)
    ]
    termination_not_performed = [
        item
        for item in leaves
        if isinstance(item, termination_not_performed_error_type)
    ]
    supported = (
        retained_error_type,
        not_performed_error_type,
        reconciliation_error_type,
        termination_not_performed_error_type,
        asyncio.CancelledError,
    )
    if any(not isinstance(item, supported) for item in leaves):
        return None
    current_task = asyncio.current_task()
    if any(isinstance(item, asyncio.CancelledError) for item in leaves) and (
        current_task is not None and current_task.cancelling()
    ):
        return None
    # Cancellation alone must retain normal task-cancellation semantics. A
    # typed reconciliation error is independently sufficient proof of removal;
    # otherwise this helper is specifically the retained-custody contract.
    if not retained and not not_performed and not termination_not_performed and (
        not reconciliation or len(reconciliation) != len(leaves)
    ):
        return None
    if termination_not_performed and len(termination_not_performed) == len(leaves):
        return None

    get_agent = getattr(manager, "get_agent", None)
    if not callable(get_agent):
        return None
    named_termination_not_performed = any(
        getattr(item, "child_name", None).casefold() == child_name.casefold()
        for item in termination_not_performed
        if isinstance(getattr(item, "child_name", None), str)
    )
    try:
        named_child_removed = (
            not named_termination_not_performed and get_agent(child_name) is None
        )
    except Exception:
        return None
    if not named_child_removed and not named_termination_not_performed:
        return None

    surviving_subtree_agents: list[str] = []
    for item in termination_not_performed:
        agent_name = _safe_termination_agent_name(item)
        if agent_name is None:
            return None
        if agent_name.casefold() == child_name.casefold():
            continue
        if agent_name not in surviving_subtree_agents:
            surviving_subtree_agents.append(agent_name)
    surviving_subtree_agents.sort(key=str.casefold)

    retained_agents: list[str] = []
    retained_states_by_agent: dict[str, set[str]] = {}
    for item in retained:
        agent_name = _safe_retained_agent_name(item)
        if agent_name is None:
            return None
        if agent_name not in retained_agents:
            retained_agents.append(agent_name)
        metadata = getattr(item, "metadata", None)
        retained_state = (
            str(metadata.get("runtime_cleanup_state", "retained"))
            if isinstance(metadata, dict)
            else "retained"
        )
        retained_states_by_agent.setdefault(agent_name, set()).add(retained_state)
    retained_agents.sort(key=str.casefold)
    pending_agents = sorted(
        (
            agent_name
            for agent_name, states in retained_states_by_agent.items()
            if states == {"pending"}
        ),
        key=str.casefold,
    )
    retained_only_agents = sorted(
        (
            agent_name
            for agent_name, states in retained_states_by_agent.items()
            if states != {"pending"}
        ),
        key=str.casefold,
    )
    named_child_retained = any(
        name.casefold() == child_name.casefold() for name in retained_agents
    )

    not_performed_agents: list[str] = []
    for item in not_performed:
        agent_name = _safe_retained_agent_name(item)
        if agent_name is None:
            return None
        if agent_name not in not_performed_agents:
            not_performed_agents.append(agent_name)
    not_performed_agents.sort(key=str.casefold)
    named_child_not_performed = any(
        name.casefold() == child_name.casefold() for name in not_performed_agents
    )

    no_op_states = {
        str(item.metadata.get("runtime_cleanup_state"))
        for item in not_performed
        if isinstance(getattr(item, "metadata", None), dict)
    }
    named_child_no_op_states = {
        str(item.metadata.get("runtime_cleanup_state"))
        for item in not_performed
        if isinstance(getattr(item, "metadata", None), dict)
        and isinstance(getattr(item, "agent_name", None), str)
        and item.agent_name.casefold() == child_name.casefold()
    }
    if len(named_child_no_op_states) == 1:
        scoped_no_op_state = next(iter(named_child_no_op_states))
    else:
        scoped_no_op_state = None
    custody_unknown = any(
        item.metadata.get("runtime_custody_known") is False
        for item in not_performed
        if isinstance(getattr(item, "metadata", None), dict)
    )
    named_child_not_hosted = "not_hosted" in named_child_no_op_states

    additional = [
        item
        for item in leaves
        if not isinstance(item, (retained_error_type, not_performed_error_type))
    ]
    additional_types = sorted(
        {public_exception_type_name(item) for item in additional}
    )
    cause_types = sorted(
        {
            str(item.metadata["cause_type"])
            for item in retained
            if isinstance(getattr(item, "metadata", None), dict)
            and type(item.metadata.get("cause_type")) is str
        }
    )
    cleanup_pending = any(
        item.metadata.get("runtime_cleanup_state") == "pending"
        for item in retained
        if isinstance(getattr(item, "metadata", None), dict)
    )
    named_child_retained_states = {
        str(item.metadata.get("runtime_cleanup_state", "retained"))
        for item in retained
        if isinstance(getattr(item, "metadata", None), dict)
        and isinstance(getattr(item, "agent_name", None), str)
        and item.agent_name.casefold() == child_name.casefold()
    }
    if len(named_child_retained_states) == 1:
        named_child_retained_state = next(iter(named_child_retained_states))
    elif named_child_retained_states:
        named_child_retained_state = "mixed"
    elif named_child_retained:
        named_child_retained_state = "retained"
    else:
        named_child_retained_state = None

    retention_witness = False
    named_retention_witness = False
    if offboard_runtime:
        # Only a retained cleanup, a subtree which was never stopped, or an
        # explicitly storage-backed agent is a positive retention witness.
        # Routing absence alone is not evidence for either retention or
        # deletion and therefore cannot produce a retention field.
        retention_witness = (
            bool(retained)
            or bool(termination_not_performed)
            or "not_hosted" in no_op_states
        )
        named_retention_witness = (
            named_child_retained
            or named_termination_not_performed
            or named_child_not_hosted
        )
        runtime_retained = retention_witness
        named_runtime_retained = named_retention_witness
        named_runtime_removed = named_child_removed and (
            not named_child_retained and not named_child_not_performed
        )
        reported_cleanup_state = (
            "termination_not_performed"
            if named_termination_not_performed
            else "not_performed"
            if not named_child_removed
            else named_child_retained_state
            if named_child_retained
            else scoped_no_op_state or "removed"
        )
    else:
        # This is the compatibility stop contract: no runtime cleanup was
        # admitted, so every named/descendant tree remains available for a
        # later restart regardless of the independent tracking failure that
        # made the overall lifecycle outcome partial.
        runtime_retained = True
        named_runtime_retained = True
        named_runtime_removed = False
        cleanup_pending = False
        reported_cleanup_state = "not_requested"

    data: Dict[str, Any] = {
        "terminated": named_child_removed,
        "child_name": child_name,
        "agent_removed": named_child_removed,
        "runtime_offboard_requested": offboard_runtime,
        "runtime_offboarded": (
            offboard_runtime
            and named_child_removed
            and not retained
            and not not_performed
            and not termination_not_performed
        ),
        "runtime_retained": runtime_retained,
        "runtime_retained_for_restart": not offboard_runtime,
        "named_child_runtime_retained": named_runtime_retained,
        "named_child_runtime_removed": named_runtime_removed,
        "runtime_cleanup_pending": cleanup_pending,
        "runtime_cleanup_state": reported_cleanup_state,
        "operator_action_required": True,
        "retry_termination": not named_child_removed,
        "retained_outcome_count": len(retained),
        "retained_agents": retained_agents,
        "additional_outcome_count": len(additional),
        "additional_outcome_types": additional_types,
    }
    if custody_unknown and not retention_witness:
        data.pop("runtime_retained")
    if custody_unknown and not named_retention_witness:
        data.pop("named_child_runtime_retained")
    if len(retained_agents) == 1:
        data["retained_agent"] = retained_agents[0]
    if retained:
        data["runtime_custody_code"] = "runtime_offboarding_retained"
        data["pending_agents"] = pending_agents
        data["retained_only_agents"] = retained_only_agents
        data["retained_cause_types"] = cause_types
        if len(cause_types) == 1:
            data["retained_cause_type"] = cause_types[0]
    if reconciliation:
        data["tracking_reconciled"] = False
    retained_custody_messages: list[str] = []
    if pending_agents:
        retained_custody_messages.append(
            "Secure runtime cleanup is still pending for "
            f"{', '.join(pending_agents)} and may complete in manager-owned "
            "cleanup."
        )
    if retained_only_agents:
        retained_custody_messages.append(
            "Secure runtime custody was retained for "
            f"{', '.join(retained_only_agents)}."
        )

    if termination_not_performed:
        data["termination_not_performed_outcome_count"] = len(
            termination_not_performed
        )
        data["surviving_subtree_agents"] = surviving_subtree_agents
        data["named_child_termination_not_performed"] = (
            named_termination_not_performed
        )
        data["retry_named_child_termination"] = named_termination_not_performed
        data["retry_descendant_termination"] = bool(surviving_subtree_agents)
        data["retry_descendant_agents"] = surviving_subtree_agents
        if offboard_runtime:
            data["runtime_custody_known"] = False
    if not_performed:
        data["not_performed_outcome_count"] = len(not_performed)
        data["not_performed_agents"] = not_performed_agents
        not_performed_code = "runtime_offboarding_not_performed"
        if retained:
            data["runtime_custody_codes"] = [
                "runtime_offboarding_retained",
                not_performed_code,
            ]
        else:
            data["runtime_custody_code"] = not_performed_code
        if scoped_no_op_state is not None:
            data["runtime_already_absent"] = (
                scoped_no_op_state == "already_absent"
            )
            if scoped_no_op_state != "custody_unknown":
                data["hosted_runtime_configured"] = (
                    scoped_no_op_state != "not_hosted"
                )
        if custody_unknown:
            data["runtime_custody_known"] = False
            data["runtime_retention_unknown"] = True
            data["finalized_from_absence"] = True

    if termination_not_performed:
        survivors = ", ".join(surviving_subtree_agents)
        if named_termination_not_performed and surviving_subtree_agents:
            custody_message = (
                f"Termination was not performed for the named child and surviving "
                f"descendants {survivors}. Runtime custody is incomplete. Retry "
                "the named child termination after operator reconciliation."
            )
        elif named_termination_not_performed:
            custody_message = (
                "Termination was not performed for the named child. Its runtime "
                "state was retained. Retry the named child termination after "
                "operator reconciliation."
            )
        elif offboard_runtime:
            custody_message = (
                f"Termination was not performed for surviving subtree {survivors}. "
                "Its runtime state remains retained and complete custody is "
                "unknown. Retry termination for the named descendant subtree "
                "after operator reconciliation."
            )
        else:
            custody_message = (
                f"Termination was not performed for surviving subtree {survivors}; "
                "runtime offboarding was not requested. Retry termination for the "
                "named descendant subtree after operator reconciliation."
            )
        if retained_custody_messages:
            custody_message = (
                f"{custody_message} {' '.join(retained_custody_messages)} "
                "An operator must also reconcile each retained tree and pending "
                "cleanup."
            )
    elif not named_child_removed:
        custody_message = (
            "Descendant cleanup had terminal outcomes, but the named child was "
            "not removed. Retry termination after operator reconciliation."
        )
    elif not offboard_runtime:
        custody_message = (
            "The child was stopped and its runtime state was retained for "
            "restart, but lifecycle bookkeeping requires operator "
            "reconciliation. Do not retry termination."
        )
    elif custody_unknown:
        custody_message = (
            "The child was already stopped and unpublished, but this request "
            "performed no secure runtime offboarding. Runtime retention and "
            "deletion are unknown until operator reconciliation confirms "
            "custody. Do not retry termination."
        )
    elif retained and not_performed:
        custody_message = (
            f"{' '.join(retained_custody_messages)} Runtime cleanup was not "
            f"performed for {', '.join(not_performed_agents)}. Do not retry "
            "termination; an operator must reconcile each retained tree, "
            "pending cleanup, and the remaining no-op custody outcomes."
        )
    elif not_performed and scoped_no_op_state == "already_absent":
        custody_message = (
            "The child was stopped, but its hosted runtime namespace was already "
            "absent and no tree was deleted. Do not retry termination."
        )
    elif not_performed and scoped_no_op_state == "not_hosted":
        custody_message = (
            "The child was stopped, but it has no hosted runtime namespace that "
            "Core can securely offboard. Its storage-backed state was not deleted. "
            "Do not retry termination."
        )
    elif not_performed:
        custody_message = (
            "The child was stopped, but the cascade produced mixed runtime "
            "custody outcomes. Do not retry termination; an operator must "
            "reconcile the named child and descendant custody states."
        )
    elif retained:
        custody_message = (
            f"{' '.join(retained_custody_messages)} Do not retry termination; "
            "an operator must reconcile each retained tree and pending cleanup."
        )
    else:
        custody_message = (
            "The child was removed, but lifecycle bookkeeping requires "
            "operator reconciliation. Do not retry termination."
        )
    return ToolResult.partial(
        (
            f"Terminated child '{child_name}'."
            if named_child_removed
            else f"Child '{child_name}' was not terminated."
        ),
        custody_message,
        data=data,
    )


def _coerce_constraint_value(value: str):
    """Coerce a spawn-constraint string value to its natural type.

    The spawn tool takes constraints as a comma-separated ``key=value`` string,
    so every value arrives as ``str``. ScopedConstitution.validate_constraints
    type-checks numeric constraints (e.g. ``max_tokens``) as ``(int, float)``, so
    a raw ``"1000"`` would be rejected. Coerce an integer-looking value to
    ``int`` and a float-looking value to ``float``; leave everything else as the
    original string (flags, tool names, behavioral text). Booleans are left as
    strings — the flag path already stores ``"true"`` and validation keys off
    presence, not a bool type.
    """
    import math

    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    # Reject non-finite: float("nan") would break verify_integrity's exact-dict
    # equality (nan != nan), and inf is not a meaningful constraint bound. Keep
    # such values as their original string.
    return f if math.isfinite(f) else value


class SpawnFeature(Feature):
    """Runtime agent spawning — create, delegate to, and manage child agents."""

    @property
    def tool_description(self) -> str:
        return (
            "Spawn and manage child agents — create new agents with specific purposes, "
            "delegate tasks, retrieve results, and terminate children"
        )

    @property
    def promote_tools_on_startup(self) -> bool:
        return True

    async def initialize(self):
        # A bounded soft-disable may have returned while a cancellation-
        # resistant delegated turn remains under this feature's detached
        # shutdown owner. Re-enable must join that owner before resetting any
        # lifecycle fields or reattaching its manager; otherwise shutdown and
        # the enabled feature can concurrently own the same runtime.
        existing_shutdown = getattr(self, "_standalone_shutdown_task", None)
        if existing_shutdown is not None:
            from kestrel_sovereign.kestrel_agent import (
                await_lifecycle_task_completion,
            )

            cancelled, failure = await await_lifecycle_task_completion(
                existing_shutdown
            )
            if self._standalone_shutdown_task is existing_shutdown:
                self._standalone_shutdown_task = None
            recorded_failure = getattr(
                self, "_standalone_shutdown_failure", None
            )
            self._standalone_shutdown_failure = None
            failure = recorded_failure if recorded_failure is not None else failure
            if failure is not None:
                raise failure
            if cancelled:
                raise asyncio.CancelledError()

        # Runtime soft-disable stops descendants but deliberately leaves finite
        # restart witnesses under their signed deadline. Preserve the exact
        # private manager that owns those timers across re-enable; manufacturing
        # a second manager would let the old timer retire authority underneath a
        # child restored into the new runtime.
        retained_manager = getattr(self, "_retained_standalone_manager", None)
        self._retained_standalone_manager = retained_manager
        self._agent_manager = retained_manager
        self._lifecycle = (
            getattr(retained_manager, "_lifecycle", None)
            if retained_manager is not None
            else None
        )
        self._standalone_restore_pending = retained_manager is not None
        self._standalone_restore_task: asyncio.Task | None = None
        self._standalone_shutdown_task: asyncio.Task | None = None
        self._standalone_shutdown_failure: BaseException | None = None
        self._standalone_manager_owned = retained_manager is not None
        self._child_results: dict[str, Any] = {}  # child_name -> latest result
        self._child_tasks: dict[str, asyncio.Task] = {}  # child_name -> running task
        # Re-delegation replaces the public per-child result slot, but an old
        # provider may suppress cancellation.  Retain every such turn until it
        # is terminal so standalone shutdown cannot close its runtime early.
        self._delegated_task_owners: set[asyncio.Task] = set()

    def get_router(self):
        """Return the Spawn panel router for dynamic mounting.

        The router exposes ``/api/spawn/children`` to the Console UI.
        Lives at ``kestrel_sovereign/endpoints/spawn.py`` (inlined
        from the archived ``kestrel-feature-spawn`` package).
        """
        from kestrel_sovereign.endpoints.spawn import router
        return router

    def get_ui_contributions(self):
        """Declare the Spawn panel frontend served from this feature package.

        The panel JS (nav tab + ``#panel-spawn`` body) lives in this package's
        ``static/`` directory rather than core ``static/`` (#2048, epic #2038).
        The server mounts the directory at ``/features/{name}/static/`` and the
        frontend boot loader ``import()``s ``spawn.js``, which self-registers the
        panel through the slot/panel registry. The ``spawn`` capability gates it
        and is derived from this feature's enabled state (#2041).
        """
        from kestrel_sovereign.features.base import UIContributions

        static_dir = str((Path(__file__).parent / "static").resolve())
        return UIContributions(
            modules=["spawn.js"],
            static_dir=static_dir,
            capability="spawn",
        )

    def _get_agent_manager(self):
        """Lazily resolve or create an AgentManager.

        In multi_agent mode the manager already exists on the agent.  In
        single-agent mode we create a lightweight one on-the-fly so
        spawn_agent works regardless of deployment mode.
        """
        manager = self._agent_manager
        manager_was_injected = manager is not None
        standalone_manager_created = False
        if manager is None:
            # Try to get from the agent's registered manager
            manager = getattr(self.agent, '_agent_manager', None)
            if manager is None:
                manager = getattr(self.agent, 'agent_manager', None)

        from kestrel_sovereign.multi_agent.agent_manager import AgentManager

        if (
            isinstance(manager, AgentManager)
            and getattr(self, "_standalone_manager_owned", False)
        ):
            # Reattach the same private manager during soft-enable before its
            # restored children can expose tools or routes.
            self.agent._agent_manager = manager

        # Single-agent mode: create a lightweight AgentManager
        if manager is None:
            storage_dir = getattr(self.agent, 'storage_path', None)
            from kestrel_sovereign.spawn.authority_registry import (
                standalone_spawn_manager_base_dir,
            )

            base_dir = standalone_spawn_manager_base_dir(storage_dir)
            manager = AgentManager(
                base_data_dir=base_dir,
                startup_roster_enabled=False,
            )
            standalone_manager_created = True
            self._standalone_manager_owned = True
            self._standalone_restore_pending = True
            logger.info("Created lightweight AgentManager for single-agent spawn")

        # Route every real manager through its canonical registration seam.
        # This is idempotent for hosted agents, restores a single-agent child's
        # durable mandate, and makes injected real managers obey the same depth
        # and parent-control rules. Test doubles remain untouched.
        if isinstance(manager, AgentManager) and standalone_manager_created:
            agent_state = vars(self.agent)
            agent_did = agent_state.get("did")
            if not isinstance(agent_did, str) or not agent_did:
                agent_did = agent_state.get("agent_id")
            names = vars(manager).get("_agent_names", {})
            agent_name = names.get(agent_did) if isinstance(names, dict) else None
            if not isinstance(agent_name, str) or not agent_name:
                for attribute in ("agent_name", "_agent_name"):
                    candidate = getattr(self.agent, attribute, None)
                    if isinstance(candidate, str) and candidate:
                        agent_name = candidate
                        break
            if not isinstance(agent_name, str) or not agent_name:
                agent_name = "default"
            already_published = (
                manager.get_agent(agent_name) is self.agent
                and vars(manager).get("_agent_names", {}).get(agent_did)
                == agent_name
            )
            if not already_published:
                manager._register_agent(agent_name, self.agent)

        # Ensure lifecycle is wired up
        if (
            isinstance(manager, AgentManager)
            and not manager_was_injected
            and not getattr(manager, '_lifecycle', None)
        ):
            from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle
            lifecycle = SpawnedAgentLifecycle(manager)
            manager._lifecycle = lifecycle
            self._lifecycle = lifecycle
            logger.info("SpawnedAgentLifecycle attached to AgentManager")

        self._agent_manager = manager
        return manager

    async def _get_ready_agent_manager(self):
        """Resolve the manager and restore a standalone host exactly once."""

        manager = self._get_agent_manager()
        if not getattr(self, "_standalone_restore_pending", False):
            return manager

        from kestrel_sovereign.multi_agent.agent_manager import AgentManager
        if not isinstance(manager, AgentManager):
            self._standalone_restore_pending = False
            return manager
        task = getattr(self, "_standalone_restore_task", None)
        if task is None:
            task = asyncio.create_task(
                manager.restore_spawn_authority_tree(self.agent),
                name="standalone_spawn_authority_restore",
            )
            self._standalone_restore_task = task
        from kestrel_sovereign.kestrel_agent import await_lifecycle_task_completion

        caller_cancelled = False
        try:
            caller_cancelled, failure = await await_lifecycle_task_completion(task)
            if failure is not None:
                raise failure
        except BaseException:
            # Keep the pending bit so a later ready hook or tool call can retry.
            raise
        else:
            # ``load_from_config`` deliberately reports per-agent startup
            # failures out of band so a multi-agent host can keep healthy
            # peers online.  A standalone parent has no lifespan handler to
            # consume that report, and treating a normal return as success
            # would permanently suppress retries for a transiently unavailable
            # child.  Only a failure-free pass consumes the pending restore.
            failures = manager.init_failures
            if failures:
                logger.warning(
                    "Standalone spawn-authority restore retained for retry: %s",
                    "; ".join(
                        f"{name}: {type(error).__name__}: {error}"
                        for name, error in failures
                    ),
                )
            else:
                self._standalone_restore_pending = False
        finally:
            if task.done() and self._standalone_restore_task is task:
                self._standalone_restore_task = None
        if caller_cancelled:
            raise asyncio.CancelledError()
        return manager

    async def on_agent_ready(self, agent=None) -> None:
        """Recover persistent standalone children after parent boot commits."""

        await self._get_ready_agent_manager()

    def _set_agent_manager(self, manager):
        """Explicitly set the AgentManager (used in testing)."""
        self._retained_standalone_manager = None
        self._agent_manager = manager
        self._standalone_manager_owned = False

    def _get_lifecycle(self, manager):
        """Return the lifecycle manager when it is fully wired."""
        from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle

        lifecycle = getattr(manager, "_lifecycle", None)
        if isinstance(lifecycle, SpawnedAgentLifecycle):
            return lifecycle
        return None

    @tool(
        name="spawn_agent",
        description=(
            "Create a new child agent with a specific purpose and constraints. "
            "The child runs in-process and is governed by a signed SpawnMandate. "
            "Returns the child's name and DID on success. "
            "constraints is a comma-separated list where each item is either "
            "'key=value' (recorded verbatim) or a bare 'flag' (recorded as "
            "flag=true), e.g. 'max_tokens=1000,no_web'. features is a "
            "comma-separated list of feature names the child may use (class "
            "name or shorthand, e.g. 'memory,web_search'); unknown names are "
            "rejected up-front. ttl is the child's time-to-live in seconds; "
            "ttl<=0 makes the child PERSISTENT (no automatic expiry) instead "
            "of ephemeral."
        ),
        category=ToolCategory.AGENT_MANAGEMENT,
    )
    async def spawn_agent(
        self,
        name: str,
        purpose: str,
        budget: float = 0.0,
        ttl: int = 3600,
        constraints: str = "",
        features: str = "",
    ) -> ToolResult:
        """
        Create a child agent with a signed mandate.

        Args:
            name: Unique name for the child agent
            purpose: What the child agent is for (stored in mandate)
            budget: Reserved for a future per-child spend ceiling. NOT YET
                 ENFORCED — pass 0 (default). A positive value is rejected
                 rather than silently accepted as a control that does nothing.
            ttl: Time-to-live in seconds (default 3600). A value <= 0 makes
                 the child PERSISTENT (registered with SpawnMode.PERSISTENT,
                 no automatic TTL expiry) rather than ephemeral.
            constraints: Comma-separated additional constraints. Each item is
                 either 'key=value' (stored verbatim) or a bare token 'flag'
                 (stored as flag=true). Example: "max_tokens=1000,no_web".
            features: Comma-separated list of allowed feature names. Each name
                 is resolved against the discoverable feature registry (class
                 name like "MemoryFeature" or shorthand like "memory"). Unknown
                 names are rejected up-front rather than silently producing a
                 non-functional mandate.

        Returns:
            ToolResult.ok with child name + DID + purpose + TTL.
            ERROR when no AgentManager is wired up (agent isn't
            running in a multi-agent host), when a requested feature name is
            unknown, or the spawn raised.
        """
        manager = await self._get_ready_agent_manager()
        if manager is None:
            return ToolResult.failed(
                error="No AgentManager available — agent is not running in a multi_agent"
            )

        # A positive budget is now ENFORCED (#2113): AgentManager holds it from
        # the parent's funded wallet and routes the child's spend through a
        # ceiling'd DelegatedWallet, releasing the unspent hold on termination. A
        # budget with no funded parent wallet is refused by
        # AgentManager._validate_budget_precondition and surfaced as the spawn
        # failure below — so no advertised-but-unenforced control ships.

        # Parse comma-separated 'key=value' / bare-flag items into the mandate's
        # additional_constraints dict. Values are coerced to their natural type:
        # a numeric string (e.g. max_tokens=1000) becomes an int/float so it
        # satisfies ScopedConstitution.validate_constraints, which type-checks
        # numeric constraints like max_tokens as (int, float) — a raw "1000"
        # string would be rejected and every documented spawn would fail (#2138).
        # Non-numeric values stay strings; a bare flag becomes "true".
        constraint_dict = {}
        if constraints:
            for item in constraints.split(","):
                item = item.strip()
                if "=" in item:
                    k, v = item.split("=", 1)
                    constraint_dict[k.strip()] = _coerce_constraint_value(v.strip())
                elif item:
                    constraint_dict[item] = "true"

        features_list = [f.strip() for f in features.split(",") if f.strip()] if features else []

        # Validate AND canonicalize requested feature names up-front. An
        # unresolvable name would be carried into the mandate's
        # ``features_allowed`` but never match a real feature — a silent,
        # non-functional grant (#1946). Reject unknowns here, and resolve
        # accepted names to their canonical CLASS name (shorthand "memory"
        # -> "MemoryFeature"), because the child's feature loader
        # (discover_features) filters the allowlist by class name.
        # ``resolve_feature_canonical_name`` covers local, entry-point AND
        # isolated-venv feature packages — the last are loaded as ProxyFeature
        # and keyed by their exact class name, so validating with the narrower
        # ``discover_feature_class_by_name`` would wrongly reject an installed
        # isolated feature the child could actually load.
        if features_list:
            from kestrel_sovereign.features import resolve_feature_canonical_name

            unknown_features = []
            canonical_features = []
            for f in features_list:
                canonical = resolve_feature_canonical_name(f)
                if canonical is None:
                    unknown_features.append(f)
                elif canonical not in canonical_features:
                    canonical_features.append(canonical)
            if unknown_features:
                return ToolResult.failed(
                    error=(
                        f"Unknown feature(s): {unknown_features}. Each entry in "
                        "'features' must resolve to a discoverable feature "
                        "(class name like 'MemoryFeature' or shorthand like "
                        "'memory'). Nothing was spawned."
                    ),
                    data={
                        "unknown_features": unknown_features,
                        "requested_features": features_list,
                    },
                )
            # Persist the resolved class names so the mandate's allowlist
            # matches what the child's loader compares against.
            features_list = canonical_features

        # Build the mandate
        from kestrel_sovereign.spawn.mandate import SpawnMandate

        mandate = SpawnMandate(
            parent_did=self.agent.agent_id,
            purpose=purpose,
            budget_allocation=budget,
            ttl_seconds=ttl,
            additional_constraints=constraint_dict,
            features_allowed=features_list,
        )

        try:
            child = await manager.spawn_agent(
                name=name,
                parent_agent=self.agent,
                mandate=mandate,
            )
            # Register with lifecycle for TTL tracking and history
            from kestrel_sovereign.spawn.lifecycle import SpawnedAgentLifecycle
            lifecycle = getattr(manager, '_lifecycle', None)
            if isinstance(lifecycle, SpawnedAgentLifecycle):
                from kestrel_sovereign.spawn.lifecycle import SpawnMode
                committed_mandate = vars(child).get(
                    "_persisted_spawn_mandate"
                )
                if not isinstance(committed_mandate, SpawnMandate):
                    raise RuntimeError(
                        "Spawned child has no committed spawn mandate"
                    )
                await lifecycle.register(
                    child_name=name,
                    child_did=child.agent_id,
                    parent_did=self.agent.agent_id,
                    ttl_seconds=committed_mandate.ttl_seconds,
                    purpose=committed_mandate.purpose,
                    mode=(
                        SpawnMode.EPHEMERAL
                        if committed_mandate.ttl_seconds > 0
                        else SpawnMode.PERSISTENT
                    ),
                    started_at=committed_mandate.created_at,
                )
            return ToolResult.ok(
                f"Spawned child '{name}' (did={child.agent_id}, ttl={ttl}s).",
                data={
                    "spawned": True,
                    "child_name": name,
                    "child_did": child.agent_id,
                    "purpose": purpose,
                    "ttl": ttl,
                },
            )
        except Exception as e:
            logger.error(f"Failed to spawn child agent '{name}': {e}")
            return ToolResult.failed(error=str(e))

    @tool(
        name="list_children",
        description="List all active child agents spawned by this agent, with their status.",
        category=ToolCategory.AGENT_MANAGEMENT,
    )
    async def list_children(self) -> ToolResult:
        """
        List active children with status.

        Returns:
            ToolResult.ok with the list of children and their status.
            ERROR when there's no AgentManager (agent isn't in a
            multi-agent host) — was the legacy
            ``{"children": [], "error": "..."}`` shape that mixed
            empty-list-of-children with the no-host-error case.
        """
        manager = await self._get_ready_agent_manager()
        if manager is None:
            return ToolResult.failed(error="No AgentManager available")

        parent_did = self.agent.agent_id
        child_names = manager.get_children(parent_did)
        lifecycle = self._get_lifecycle(manager)

        children = []
        for child_name in child_names:
            child_agent = manager.get_agent(child_name)
            status = "running" if child_agent is not None else "stopped"

            has_result = child_name in self._child_results
            has_pending_task = (
                child_name in self._child_tasks
                and not self._child_tasks[child_name].done()
            )

            child_record = {
                "name": child_name,
                "status": status,
                "has_result": has_result,
                "has_pending_task": has_pending_task,
            }
            if lifecycle is not None:
                refusal = lifecycle.get_termination_refusal(child_name)
                if refusal is not None:
                    child_record["termination_refusal"] = refusal
                    child_record["operator_action_required"] = True
            children.append(child_record)

        if not children:
            return ToolResult.ok(
                "No active children.",
                data={"children": [], "count": 0},
            )
        return ToolResult.ok(
            f"{len(children)} active child(ren): "
            + ", ".join(f"{c['name']} ({c['status']})" for c in children)
            + ".",
            data={"children": children, "count": len(children)},
        )

    @tool(
        name="delegate_task",
        description=(
            "Send a task to an existing child agent for processing. "
            "The child uses its own LLM context to execute the task. "
            "Results can be retrieved later with get_child_result."
        ),
        category=ToolCategory.AGENT_MANAGEMENT,
    )
    async def delegate_task(self, child_name: str, task: str) -> ToolResult:
        """
        Send work to an existing child agent.

        Args:
            child_name: Name of the child agent to delegate to
            task: The task description for the child to execute

        Returns:
            ToolResult.ok with the queued task info and a note that
            ``get_child_result`` is the way to retrieve the output.
            ERROR for no-AgentManager, child-not-running, and
            child-not-belonging-to-this-parent paths.
        """
        manager = await self._get_ready_agent_manager()
        if manager is None:
            return ToolResult.failed(error="No AgentManager available")

        child_agent = manager.get_agent(child_name)
        if child_agent is None:
            return ToolResult.failed(
                error=f"Child agent '{child_name}' not found or not running"
            )

        # Verify this is actually our child
        parent_did = self.agent.agent_id
        if child_name not in manager.get_children(parent_did):
            return ToolResult.failed(
                error=f"Agent '{child_name}' is not a child of this agent"
            )

        # Run the task asynchronously via the child agent's chat method.
        #
        # A completed task records its result in ``_child_results`` but must NOT
        # finalize/terminate the child (#F279). ``lifecycle.report_result``
        # cancels the TTL timer and runs ``_terminate_and_cleanup`` — calling it
        # per task killed the child after ONE delegate, breaking the documented
        # spawn → delegate → get_child_result → delegate-again flow (and making
        # a second delegate_task/terminate_child fail with "not found"). The
        # child persists until its TTL expires, an explicit terminate_child, or
        # parent shutdown — the paths that legitimately finalize it.
        async def _run_child_task():
            owner = asyncio.current_task()
            try:
                result = await child_agent.process_input(task)
                if self._child_tasks.get(child_name) is owner:
                    self._child_results[child_name] = {
                        "success": True,
                        "result": result,
                        "completed_at": time.time(),
                    }
            except Exception as e:
                logger.error(f"Child '{child_name}' task failed: {e}")
                if self._child_tasks.get(child_name) is owner:
                    self._child_results[child_name] = {
                        "success": False,
                        "error": str(e),
                        "completed_at": time.time(),
                    }

        # Cancel any existing task for this child
        if child_name in self._child_tasks and not self._child_tasks[child_name].done():
            self._child_tasks[child_name].cancel()

        delegated = asyncio.create_task(_run_child_task())
        self._child_tasks[child_name] = delegated
        owners = getattr(self, "_delegated_task_owners", None)
        if owners is None:
            owners = set()
            self._delegated_task_owners = owners
        owners.add(delegated)
        delegated.add_done_callback(owners.discard)

        return ToolResult.ok(
            f"Delegated task to '{child_name}'. Use get_child_result to retrieve the result.",
            data={
                "delegated": True,
                "child_name": child_name,
                "task": task,
                "note": "Task is running. Use get_child_result to retrieve the result.",
            },
        )

    @tool(
        name="get_child_result",
        description="Retrieve the result from a child agent's completed task.",
        category=ToolCategory.AGENT_MANAGEMENT,
    )
    async def get_child_result(self, child_name: str) -> ToolResult:
        """
        Retrieve results from a completed child task.

        Args:
            child_name: Name of the child agent to get results from

        Returns:
            ToolResult.ok with ready=True + the child's task output
            on success. PARTIAL when the task is still running OR no
            result is available — these are not errors (the LLM may
            legitimately need to poll), but the LLM must NOT report
            them as completed work; the caveat says exactly which
            "not ready" state we're in.
            ERROR when the underlying child task itself raised — the
            stored failure dict carries ``error`` and the envelope
            ERROR copies it through so the LLM speaks the failure
            instead of treating "ready=True" as success.
        """
        # Check if there's a pending task still running
        if child_name in self._child_tasks and not self._child_tasks[child_name].done():
            return ToolResult.partial(
                f"Task for '{child_name}' is still running.",
                "no result yet — try get_child_result again after the task completes.",
                data={
                    "ready": False,
                    "child_name": child_name,
                    "note": "Task is still running. Try again later.",
                },
            )

        if child_name not in self._child_results:
            return ToolResult.partial(
                f"No result available for '{child_name}'.",
                (
                    "either no task was delegated to this child, or the "
                    "previous result was already consumed by a prior "
                    "get_child_result call (this surface pops on read)."
                ),
                data={
                    "ready": False,
                    "child_name": child_name,
                    "note": "No result available. Either no task was delegated or it hasn't completed.",
                },
            )

        result = self._child_results.pop(child_name)
        # The stored result has ``success`` and either ``result`` (on
        # success) or ``error`` (on failure). Surface child-side
        # failures as ERROR so the LLM doesn't claim the delegated
        # work succeeded.
        if not result.get("success"):
            err = result.get("error") or f"Child task for '{child_name}' failed"
            return ToolResult.failed(
                error=err,
                data={"ready": True, "child_name": child_name, **result},
            )
        return ToolResult.ok(
            f"Retrieved result for '{child_name}'.",
            data={"ready": True, "child_name": child_name, **result},
        )

    @tool(
        name="terminate_child",
        description=(
            "Stop a child agent. By default its runtime state is retained for "
            "restart; offboard_runtime=true irreversibly deletes the child and "
            "descendant hosted runtime trees after shutdown."
        ),
        category=ToolCategory.AGENT_MANAGEMENT,
    )
    async def terminate_child(
        self,
        child_name: str,
        offboard_runtime: bool = False,
    ) -> ToolResult:
        """
        Terminate a child agent early.

        Args:
            child_name: Name of the child agent to terminate
            offboard_runtime: Explicitly and irreversibly delete hosted
                runtime state after stopping the child and descendants.

        Returns:
            ToolResult.ok when the child was actually terminated.
            ToolResult.partial when the child was stopped/unpublished but its
            isolated runtime cleanup is pending/retained or its manager
            bookkeeping needs operator reconciliation. It is also partial for
            either offboard value when local finalization observes only routing
            absence, because that proves neither runtime deletion nor retention.
            ERROR when there's no AgentManager, the named agent is
            not a child of this parent, or the underlying terminate
            call returned False.
        """
        manager = await self._get_ready_agent_manager()
        if manager is None:
            return ToolResult.failed(error="No AgentManager available")
        if type(offboard_runtime) is not bool:
            return ToolResult.failed(error="offboard_runtime must be a bool")

        # Verify this is our child
        parent_did = self.agent.agent_id
        if child_name not in manager.get_children(parent_did):
            return ToolResult.failed(
                error=f"Agent '{child_name}' is not a child of this agent"
            )

        # Cancel any running task
        if child_name in self._child_tasks and not self._child_tasks[child_name].done():
            self._child_tasks[child_name].cancel()
            self._child_tasks.pop(child_name, None)

        # Clean up stored results
        self._child_results.pop(child_name, None)

        # Remove from manager (handles shutdown + cascading child termination)
        from kestrel_sovereign.multi_agent.agent_manager import (
            ChildTerminationNotPerformedError,
            ChildTerminationReconciliationError,
            RuntimeOffboardingNotPerformedError,
            RuntimeOffboardingRetainedError,
            public_exception_type_name,
        )

        lifecycle = self._get_lifecycle(manager)
        try:
            if lifecycle is not None:
                if offboard_runtime:
                    result = await lifecycle.terminate(
                        child_name=child_name,
                        reason="explicit termination",
                        offboard_runtime=True,
                    )
                else:
                    result = await lifecycle.terminate(
                        child_name=child_name,
                        reason="explicit termination",
                    )
                removed = result is not None
            else:
                if offboard_runtime:
                    removed = await manager.terminate_child(
                        parent_did,
                        child_name,
                        offboard_runtime=True,
                    )
                else:
                    removed = await manager.terminate_child(parent_did, child_name)
        except BaseException as exc:
            # Manager-typed terminal outcomes prove that routing withdrawal
            # succeeded even when custody/reconciliation did not. Flatten
            # those groups into one truthful, path-free PARTIAL. Anything else
            # remains exceptional: converting an arbitrary programming or
            # namespace-security failure into a tool ERROR would mask it.
            partial = _termination_partial_result(
                child_name=child_name,
                offboard_runtime=offboard_runtime,
                manager=manager,
                error=exc,
                retained_error_type=RuntimeOffboardingRetainedError,
                not_performed_error_type=RuntimeOffboardingNotPerformedError,
                reconciliation_error_type=ChildTerminationReconciliationError,
                termination_not_performed_error_type=(
                    ChildTerminationNotPerformedError
                ),
                public_exception_type_name=public_exception_type_name,
            )
            if partial is None:
                raise
            return partial
        if removed:
            if (
                lifecycle is not None
                and getattr(result, "finalized_from_absence", False) is True
            ):
                return _absence_finalization_partial_result(
                    child_name=child_name,
                    offboard_runtime=offboard_runtime,
                )
            return ToolResult.ok(
                f"Terminated child '{child_name}'.",
                data={
                    "terminated": True,
                    "child_name": child_name,
                    "runtime_offboarded": offboard_runtime,
                    "runtime_retained_for_restart": not offboard_runtime,
                },
            )

        return ToolResult.failed(error=f"Failed to terminate '{child_name}'")

    async def _shutdown_standalone_after_delegated_tasks(
        self,
        manager,
        child_tasks: list[asyncio.Task],
    ) -> None:
        """Quiesce delegated turns before closing their standalone runtimes."""

        from kestrel_sovereign.kestrel_agent import await_lifecycle_task_completion

        outcomes: list[BaseException] = []
        caller_cancelled = False
        for task in child_tasks:
            cancelled, failure = await await_lifecycle_task_completion(task)
            caller_cancelled = caller_cancelled or cancelled
            if failure is not None and not isinstance(
                failure, asyncio.CancelledError
            ):
                outcomes.append(failure)

        descendant_shutdown = asyncio.create_task(
            manager.shutdown_spawn_authority_tree(self.agent),
            name="standalone_spawn_descendant_shutdown",
        )
        cancelled, failure = await await_lifecycle_task_completion(
            descendant_shutdown
        )
        caller_cancelled = caller_cancelled or cancelled
        if failure is not None:
            outcomes.append(failure)

        # A failed descendant drain can still own live runtimes. Keep the exact
        # manager attached so the failure is observable and a later shutdown
        # can retry instead of manufacturing a fresh, lineage-empty manager.
        if failure is None:
            self._retained_standalone_manager = manager
            if vars(self.agent).get("_agent_manager") is manager:
                self.agent._agent_manager = None
            if vars(self.agent).get("agent_manager") is manager:
                self.agent.agent_manager = None
            if self._agent_manager is manager:
                self._agent_manager = None
                self._lifecycle = None
                self._standalone_manager_owned = False

        if caller_cancelled:
            outcomes.append(asyncio.CancelledError())
        if len(outcomes) == 1:
            raise outcomes[0]
        if outcomes:
            raise BaseExceptionGroup(
                "Spawn feature shutdown had terminal failures",
                outcomes,
            )

    async def shutdown(self):
        """Join private restore and retain descendant owners before root shutdown."""
        from kestrel_sovereign.kestrel_agent import await_lifecycle_task_completion
        from kestrel_sovereign.multi_agent.agent_manager import AgentManager

        outcomes: list[BaseException] = []
        caller_cancelled = False
        existing_shutdown = getattr(self, "_standalone_shutdown_task", None)
        if existing_shutdown is not None:
            if not existing_shutdown.done():
                # The retained owner from an earlier bounded shutdown remains
                # responsible for both delegated turns and descendant cleanup.
                return
            self._standalone_shutdown_task = None
            cancelled, failure = await await_lifecycle_task_completion(
                existing_shutdown
            )
            caller_cancelled = caller_cancelled or cancelled
            recorded_failure = self._standalone_shutdown_failure
            self._standalone_shutdown_failure = None
            failure = recorded_failure if recorded_failure is not None else failure
            if failure is not None:
                raise failure
            if caller_cancelled:
                raise asyncio.CancelledError()
            return
        delegated_owners = getattr(self, "_delegated_task_owners", set())
        child_tasks = list(
            {*self._child_tasks.values(), *delegated_owners}
        )
        for task in child_tasks:
            if not task.done():
                task.cancel()
        self._child_tasks.clear()
        delegated_owners.clear()
        self._child_results.clear()

        restore_task = getattr(self, "_standalone_restore_task", None)
        if restore_task is not None:
            if not restore_task.done():
                restore_task.cancel()
            cancelled, failure = await await_lifecycle_task_completion(restore_task)
            caller_cancelled = caller_cancelled or cancelled
            if failure is not None and not isinstance(
                failure, asyncio.CancelledError
            ):
                outcomes.append(failure)
            if self._standalone_restore_task is restore_task:
                self._standalone_restore_task = None

        manager = self._agent_manager
        if (
            getattr(self, "_standalone_manager_owned", False)
            and isinstance(manager, AgentManager)
        ):
            standalone_shutdown = asyncio.create_task(
                self._shutdown_standalone_after_delegated_tasks(
                    manager,
                    child_tasks,
                ),
                name="standalone_spawn_shutdown_owner",
            )
            self._standalone_shutdown_task = standalone_shutdown
            if any(not task.done() for task in child_tasks):
                # Cancellation-resistant cognition cannot be joined inside the
                # feature's bounded shutdown slice, but its child dispatcher and
                # storage cannot be closed underneath it either.  The strongly
                # retained, cancellation-safe owner above performs that ordering
                # after this control-plane call returns.
                def retain_shutdown_outcome(completed: asyncio.Task) -> None:
                    failure: BaseException | None
                    if completed.cancelled():
                        failure = asyncio.CancelledError()
                    else:
                        failure = completed.exception()
                    self._standalone_shutdown_failure = failure
                    if failure is not None:
                        logger.error(
                            "Detached standalone descendant shutdown failed; "
                            "manager ownership retained for retry",
                            exc_info=(
                                type(failure),
                                failure,
                                failure.__traceback__,
                            ),
                        )

                standalone_shutdown.add_done_callback(retain_shutdown_outcome)
                self._standalone_restore_pending = False
                return
            cancelled, failure = await await_lifecycle_task_completion(
                standalone_shutdown
            )
            caller_cancelled = caller_cancelled or cancelled
            if failure is not None:
                outcomes.append(failure)
        self._standalone_restore_pending = False

        if caller_cancelled:
            outcomes.append(asyncio.CancelledError())
        if len(outcomes) == 1:
            raise outcomes[0]
        if outcomes:
            raise BaseExceptionGroup(
                "Spawn feature shutdown had terminal failures",
                outcomes,
            )
