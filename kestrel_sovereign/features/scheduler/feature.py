"""
Scheduler Feature -- cron-based scheduled task execution for agents.

Allows agents to create, manage, and monitor scheduled tasks that run on
cron expressions. Tasks are persisted in the database and survive restarts.
A background asyncio runner checks for due tasks every 30 seconds.

Valid scheduled task names are validated at creation time (#1618):
``schedule_add`` rejects any name that is neither a built-in cron source
nor a discoverable feature tool, so a typo / unregistered name (e.g.
``github_pr_watch`` before it was registered) fails loudly instead of
silently failing every tick with "Unknown task".

Built-in cron sources (see ``signals/sources/scheduler.py`` CRON_TASKS):
    backup_snapshot       -- snapshot the agent's data via the sync service
    signal_dispatch       -- dispatch queued work to Talon
    trash_retention       -- hard-purge soft-deleted conversation rows
                             past their per-agent retention window (#764)
    training_cycle        -- run a LoRA training cycle (ReflectionFeature)
    morning_signal        -- produce the daily briefing artifact
    sleep                 -- THE nightly memory-maintenance cycle (#1674 P3):
                             reflection (via the subscribed sleep hook) +
                             consolidation + the forgetting deletion tier, all
                             through MemorySystem.consolidate(). Activity-gated;
                             skip_export=True (backups own DR). Supersedes the
                             auto-seeded memory_consolidate + reflect crons.
    reflect               -- reflection workflow tool (ReflectionFeature). Still
                             schedulable, but no longer auto-seeded — reflection
                             subscribes to `sleep` via sleep_hooks.
    memory_consolidate    -- consolidate short-term memory into episodes + run
                             the [forgetting] deletion tier. Still schedulable,
                             but no longer auto-seeded — `sleep` runs the same
                             MemorySystem.consolidate() flow nightly (#1674).
    wait_reconcile        -- poll every MonitorableWaitable provider's
                             in-flight handles, wake on any terminal
                             transition (Wave 2 of #1860)
    restart_coordinator   -- execute pending restart requests (#1512)
    github_pr_watch       -- poll a GitHub PR/issue, wake on relevant
                             state/comment/check/merge changes (#1618)
    ecosystem_discovery_watch -- run stale-work/red-CI discovery and wake
                             only on actionable new/changed/resolved findings
                             (#2281)
    bootstrap_timeout_check -- flag agents left bootstrap_state=pending
                              past the timeout (#378)

Any loaded feature tool can also be scheduled by its tool name.

Tools:
    !schedule list                          -- list all scheduled tasks
    !schedule add <cron> <task> [args_json] -- add a new task
    !schedule remove <task_id>             -- remove a task
    !schedule pause <task_id>              -- pause a task
    !schedule resume <task_id>             -- resume a paused task
    !schedule history                      -- recent execution history
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign.features.scheduler.outcome import ScheduledTaskOutcome
from kestrel_sovereign.features.base import Feature, _serialize_tool_result, tool
from kestrel_sovereign.features.scheduler.cron import (
    CronParseError,
    get_timezone,
    next_run,
    parse,
)
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    SCHEDULER_ROLLOUT_STATE_QUIESCING,
    SchedulerProtocolVersionIncompatible,
    SchedulerRunner,
    adopt_scheduler_registration_ownership,
    scheduler_database_clock,
    scheduler_database_now_sql,
    validate_schedule_idempotency_base,
)
from kestrel_sovereign.features.storage_access import resolve_feature_database

logger = logging.getLogger(__name__)

# Built-in cron tasks that were seeded by a prior version but no longer have a
# handler/source. Their persisted scheduled_tasks rows are deleted once on
# startup so they don't fire forever as "Unknown task". Add a name here when a
# built-in cron is retired; never list user-schedulable feature tools.
_RETIRED_BUILTIN_CRON_TASKS = frozenset({
    "cognition_retention",  # #1674 — superseded by [forgetting] in memory_consolidate
    "talon_monitor",  # #1860 Wave 2 — superseded by the generic wait_reconcile
})

# Crons SUPERSEDED by the nightly `sleep` cycle (#1674 P3). These names are also
# valid TOOLS a user could schedule, so they are NOT blanket-retired; only rows
# matching the exact prior core auto-seed (cron + args) are removed on startup,
# mapped name -> (cron_expression, parsed_args). See post_all_features_loaded.
_SUPERSEDED_AUTOSEEDS = {
    "memory_consolidate": ("0 4 * * *", {}),
    "reflect": ("0 */4 * * *", {"scope": "all", "depth": "normal"}),
}

_BUILTIN_SCHEDULE_IDEMPOTENCY_PREFIX = "scheduler:builtin:v1:"


class SchedulerFeature(Feature):
    """
    Cron/scheduler system for running agent tasks on a schedule.

    On initialize(), creates DB tables and prepares a background runner.
    Standalone polling is armed only from ``on_agent_ready``; a shared
    PostgreSQL host owns one fleet runner instead.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage scheduled tasks - add, remove, pause, resume cron-based "
            "tasks for memory consolidation, wellness checks, audit anchoring, "
            "and custom operations"
        )

    @staticmethod
    def _scheduler_mutation_wrote(result: Any) -> bool:
        """Interpret real rowcounts while keeping lightweight DB doubles usable."""

        if isinstance(result, bool):
            return result
        if isinstance(result, int):
            return result > 0
        return True

    def _pending_scheduler_registration_nonce(self) -> Optional[str]:
        """Return this dynamic onboarding's private schedule ownership marker.

        The manager keeps the registration object on an agent until host
        onboarding commits.  Scheduler rows written in that window can be
        removed by its rollback; any row without this registration's nonce is
        never inferred to belong to the failed registration.
        """

        registration = getattr(
            self.agent, "_dynamic_scheduler_tenant_registration", None
        )
        nonce = getattr(registration, "registration_nonce", None)
        return nonce if isinstance(nonce, str) and nonce else None

    @staticmethod
    def _rollout_mutation_error() -> ToolResult:
        """Return the fail-closed response for an unacknowledged rollout."""

        return ToolResult.failed(
            "Scheduler protocol rollout is not active for this agent. "
            "Do not create, resume, or change schedules until legacy "
            "scheduler replicas are drained and the rollout nonce is acknowledged."
        )

    async def initialize(self):
        """Initialize the scheduler: set up DB refs, register cron sources
        with the SignalDispatcher (Phase 4 of #889), and prepare polling."""
        self._db = None
        self._agent_id = ""
        self._runner: Optional[SchedulerRunner] = None
        self._polling_managed_by_host = (
            getattr(self.agent, "_scheduler_polling_managed_by_host", False)
            is True
        )

        self._db = resolve_feature_database(self.agent)

        # Agent identity (DID is the canonical source of truth)
        self._agent_id = self.agent.did

        if self._db is None:
            logger.warning("SchedulerFeature: no database available, running in no-op mode")
            return

        # Register one source per built-in cron task with the dispatcher.
        # Done at initialize() (not post_all_features_loaded) so the
        # registry is stable before the background runner starts polling.
        # Tools that depend on other features (e.g. `reflect` requires
        # ReflectionFeature) register their sources unconditionally; if
        # the underlying tool isn't loaded the handler raises and the
        # dispatcher captures it as FAILED. Saner than dynamic registration.
        from kestrel_sovereign.signals.sources.scheduler import (
            build_cron_registrations,
        )

        registry = getattr(self.agent, "signal_registry", None)
        if registry is not None:
            cron_registrations = build_cron_registrations(
                tool_lookup=self._lookup_and_run_tool,
                builtin_handlers={
                    "backup_snapshot": self._handle_backup_snapshot,
                    "trash_retention": self._run_trash_retention,
                    "github_pr_watch": self._run_github_pr_watch,
                    "ecosystem_discovery_watch": self._run_ecosystem_discovery_watch,
                    "sleep": self._handle_sleep,
                    "wait_reconcile": self._run_wait_reconcile,
                    "bootstrap_timeout_check": self._run_bootstrap_timeout_check,
                },
            )
            from kestrel_sovereign.signals import RegistrationPolicy

            # OPTIONAL policy (#2522): a feature reload re-registering an
            # equivalent cron source is a no-op; a name clash with a DIFFERENT
            # contract is reported (logged loudly) rather than silently skipped
            # by the old ``name not in registry`` precheck. Never raises, so
            # scheduler init is not aborted by one bad source.
            cron_outcomes = registry.register_batch(
                cron_registrations, RegistrationPolicy.OPTIONAL
            )
            # Own the cron sources we newly registered so shutdown / boot
            # rollback unregisters exactly them (#2522 P2).
            self._own_signal_sources(cron_outcomes)

            # github_pr_watch (#1618) is an ACTION cron task that, on a
            # relevant change, enqueues a COGNITION github.pr_activity
            # signal. Register that downstream source here so the wake
            # has a prompt template and policy. The watcher is built-in
            # (no GitHub feature required), so the source lives with the
            # scheduler rather than an external package.
            from kestrel_sovereign.signals.sources.github_pr_watch import (
                build_github_pr_activity_registration,
            )

            self._own_signal_sources(
                registry.register_with_policy(
                    build_github_pr_activity_registration(),
                    RegistrationPolicy.OPTIONAL,
                )
            )

            from kestrel_sovereign.signals.sources.ecosystem_discovery import (
                build_ecosystem_discovery_registration,
            )

            self._own_signal_sources(
                registry.register_with_policy(
                    build_ecosystem_discovery_registration(),
                    RegistrationPolicy.OPTIONAL,
                )
            )
        else:
            logger.warning(
                "SchedulerFeature: no signal_registry on agent, "
                "cron tasks will not be dispatched as signals"
            )

        # A shared-PostgreSQL AgentManager host owns one fleet runner with
        # live authority and lifecycle locks. A second agent-scoped runner
        # would keep executing from its frozen DID after administrative
        # removal, so hosted agents deliberately expose no polling runner.
        if self._polling_managed_by_host:
            logger.info(
                "SchedulerFeature prepared for host-owned polling (%s)",
                self._agent_id,
            )
            return

        # Prepare the durable protocol and tables now so post-load can seed
        # defaults, but do not poll until on_agent_ready. Other features still
        # have post-load wiring to complete, and an overdue one-shot must not be
        # terminalized while its built-in owner is transiently unavailable.
        self._runner = SchedulerRunner(
            db=self._db,
            agent_id=self._agent_id,
            executor=self._dispatch_scheduled_task,
            misfire_grace_seconds=self._load_misfire_grace_seconds(),
            max_concurrent_tasks=self._load_max_concurrent_tasks(),
            lease_seconds=self._load_lease_seconds(),
        )
        await self._runner.start(polling=False)
        logger.info("SchedulerFeature initialized; polling awaits agent readiness")

    async def on_agent_ready(self, agent) -> None:
        """Arm standalone polling only after every feature finished post-load."""

        if self._runner is not None:
            await self._runner.arm()

    @staticmethod
    def _load_max_concurrent_tasks() -> int:
        """Read ``[scheduler] max_concurrent_tasks`` from kestrel.toml.

        Defaults to ``DEFAULT_MAX_CONCURRENT_TASKS`` (#1675). Operators set 1
        to restore the legacy strictly-serial tick behaviour."""
        from kestrel_sovereign.config import load_section
        from kestrel_sovereign.features.scheduler.runner import (
            DEFAULT_MAX_CONCURRENT_TASKS,
        )

        cfg = load_section("scheduler") or {}
        raw = cfg.get("max_concurrent_tasks", DEFAULT_MAX_CONCURRENT_TASKS)
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid scheduler.max_concurrent_tasks=%r, using %d",
                raw, DEFAULT_MAX_CONCURRENT_TASKS,
            )
            return DEFAULT_MAX_CONCURRENT_TASKS

    @staticmethod
    def _load_misfire_grace_seconds() -> int:
        """Read ``[scheduler] misfire_grace_seconds`` from kestrel.toml.

        Defaults to ``DEFAULT_MISFIRE_GRACE_SECONDS`` (host-suspend
        resilience, #1545). Operators set 0 to disable the rail and restore
        legacy fire-every-overdue-task behaviour."""
        from kestrel_sovereign.config import load_section
        from kestrel_sovereign.features.scheduler.runner import (
            DEFAULT_MISFIRE_GRACE_SECONDS,
        )

        cfg = load_section("scheduler") or {}
        raw = cfg.get("misfire_grace_seconds", DEFAULT_MISFIRE_GRACE_SECONDS)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid scheduler.misfire_grace_seconds=%r, using %ds",
                raw, DEFAULT_MISFIRE_GRACE_SECONDS,
            )
            return DEFAULT_MISFIRE_GRACE_SECONDS

    @staticmethod
    def _load_lease_seconds() -> int:
        """Read the durable claim lease interval from ``[scheduler]``.

        A lease is renewed while dispatch is active.  It is deliberately much
        longer than normal polling jitter, but finite so a dead replica's work
        can be recovered by another runner.
        """
        from kestrel_sovereign.config import load_section
        from kestrel_sovereign.features.scheduler.runner import DEFAULT_LEASE_SECONDS

        cfg = load_section("scheduler") or {}
        raw = cfg.get("lease_seconds", DEFAULT_LEASE_SECONDS)
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid scheduler.lease_seconds=%r, using %ds",
                raw, DEFAULT_LEASE_SECONDS,
            )
            return DEFAULT_LEASE_SECONDS

    async def post_all_features_loaded(self, agent):
        """Register default scheduled tasks after all features are loaded.

        Idempotent — checks for existing tasks before adding.
        Only schedules reflection/training if ReflectionFeature is available.
        """
        # Register the ``ci:`` Waitable provider so a GitHub PR merge/check
        # wait can be durably watched/re-armed across restart (#2729). Done
        # before the no-DB early return: the CI provider polls GitHub over the
        # network (via GITHUB_TOKEN) and does not need the scheduler DB, and
        # registering the kind also makes it available for wait-registration
        # ownership cross-checks.
        registry = getattr(agent, "wait_registry", None)
        if registry is not None:
            from kestrel_sovereign.features.scheduler.ci_wait_provider import (
                CIWaitable,
            )

            # Record ownership so base shutdown()/boot rollback unregisters it.
            self._register_wait_provider(registry, CIWaitable(self), replace=True)

        if not self._db:
            return

        # Check what's already scheduled
        existing = await self.schedule_list()
        existing_tasks = (existing.data or {}).get("tasks", []) if existing.data else []
        existing_names = {t["task_name"] for t in existing_tasks}

        # One-time cutover cleanup: drop persisted schedule rows for built-in
        # cron tasks that no longer exist. An agent that booted on a prior
        # version had `cognition_retention` (#1715) seeded into scheduled_tasks;
        # after #1674 removed its handler+source, that orphan row would fire
        # every tick with no CRON registration and fail tool lookup as an
        # "Unknown task". Delete such rows so the cutover is clean rather than
        # noisy. (Built-in only — user-scheduled feature tools are never touched.)
        retired = _RETIRED_BUILTIN_CRON_TASKS & existing_names
        for task in existing_tasks:
            if task["task_name"] in retired:
                # schedule_list() exposes the row identifier as "id".
                task_id = task["id"]
                await self.schedule_remove(task_id)
                logger.info(
                    "Removed retired built-in schedule '%s' (id=%s)",
                    task["task_name"], str(task_id)[:8],
                )
        existing_names -= retired

        # Cutover for crons SUPERSEDED by the nightly `sleep` cycle (#1674 P3):
        # memory_consolidate + reflect are still valid TOOLS (a user may schedule
        # them), so we can't blanket-retire by name. Instead remove ONLY rows
        # that exactly match what core used to AUTO-SEED (name + cron + args) —
        # leaving any user-customized schedule intact. Without this, an upgraded
        # agent would run both the old crons AND `sleep`, double-touching memory.
        for task in existing_tasks:
            seed = _SUPERSEDED_AUTOSEEDS.get(task["task_name"])
            if seed is None:
                continue
            cron, args = seed
            if task.get("cron_expression") == cron and task.get("args") == args:
                await self.schedule_remove(task["id"])
                existing_names.discard(task["task_name"])
                logger.info(
                    "Removed auto-seeded '%s' (superseded by nightly sleep, "
                    "id=%s)", task["task_name"], str(task["id"])[:8],
                )

        # Reflection-dependent schedules only if ReflectionFeature is loaded
        has_reflection = "ReflectionFeature" in agent.features

        from kestrel_sovereign.storage.retention import (
            DEFAULT_RETENTION_CRON,
        )

        defaults = [
            ("backup_snapshot", "0 */4 * * *", "{}"),
            ("morning_signal", "0 8 * * *", "{}"),
            ("signal_dispatch", "5 8 * * *", "{}"),
            # Trash retention sweep (#764). Hard-purges soft-deleted
            # conversation rows past the per-agent retention window.
            # Operators tune frequency via `!schedule` and the window
            # via [trash] in kestrel.toml.
            ("trash_retention", DEFAULT_RETENTION_CRON, "{}"),
        ]

        # Nightly sleep cycle (#1674 P3) — the ONE built-in memory-maintenance
        # cron. sleep() runs reflection (via the subscribed sleep hook),
        # consolidation, and the forgetting deletion tier through the single
        # MemorySystem.consolidate() chokepoint. skip_export=True: backups stay
        # on their own 4h disaster-recovery cadence (backup_snapshot), so sleep
        # doesn't double-snapshot. Replaces the old memory_consolidate + reflect
        # crons (retired below) so we don't have several crons each touching
        # memory. Seeded when MemoryFeature is present (there's memory to tend).
        if "MemoryFeature" in agent.features:
            defaults.append(
                ('sleep', "0 4 * * *", '{"skip_export": true}'),  # nightly at 4am
            )

        if has_reflection:
            # Reflection no longer has its own cron — it subscribes to the sleep
            # cycle via agent.sleep_hooks (on_pre_sleep / on_post_
            # consolidation). LoRA training stays a separate nightly job (it's
            # model training, not memory maintenance).
            defaults.append(
                ("training_cycle", "0 3 * * *", '{"iterations":3,"depth":"normal"}'),
            )

        # Generic wait reconciler (Wave 2 of #1860) — the core, always-on
        # successor to the talon-specific talon_monitor. Cheap (no LLM,
        # just polling each MonitorableWaitable provider's in-flight
        # handles) so 1/min cadence is fine. Emits one COGNITION signal
        # per terminal-state transition (provider-specific source when the
        # provider declares one, else wait.complete). Unconditional — the
        # reconciler is core and no-ops cleanly when no monitorable
        # provider is registered.
        defaults.append(("wait_reconcile", "* * * * *", "{}"))

        # Restart coordinator (#1512) — installs only if the
        # RestartCoordinatorFeature is loaded. Cheap (no LLM; idle
        # unless a request is pending) so 1/min is fine. Emits one
        # restart.completed signal per row after the agent boots
        # back up post-restart.
        if "RestartCoordinatorFeature" in agent.features:
            defaults.append(("restart_coordinator", "* * * * *", "{}"))

        for task_name, cron, args in defaults:
            # ``schedule_list`` is a snapshot used only for migration cleanup
            # above.  Every default must still take the transactional path:
            # a concurrent host can rely on a pending registration's already
            # listed row, which atomically adopts that exact row before
            # returning it.
            result = await self._ensure_builtin_schedule(
                cron_expression=cron,
                task_name=task_name,
                args_json=args,
            )
            if result.status is ToolResultStatus.OK:
                if (result.data or {}).get("existing"):
                    logger.debug(
                        "Schedule '%s' already exists after durable recheck",
                        task_name,
                    )
                else:
                    logger.info(
                        "Scheduled '%s' (%s), next: %s",
                        task_name,
                        cron,
                        (result.data or {}).get("next_run_at"),
                    )
            else:
                logger.warning("Failed to schedule '%s': %s", task_name, result.error)

    async def _ensure_builtin_schedule(
        self,
        *,
        cron_expression: str,
        task_name: str,
        args_json: str,
    ) -> ToolResult:
        """Ensure one core default under the durable rollout serialization.

        ``post_all_features_loaded`` can run concurrently on two replicas.
        Its outer list is only an optimization; this in-transaction logical
        recheck is the authority. The deterministic base identifies core's
        auto-seed without imposing a global uniqueness rule on ``task_name``:
        users may still create any number of independently keyed schedules.
        """

        if not self._db:
            return ToolResult.failed("Database not available")
        stable_idempotency = (
            f"{_BUILTIN_SCHEDULE_IDEMPOTENCY_PREFIX}{task_name}"
        )
        try:
            async with self._schedule_transaction():
                if not await self._lock_active_scheduler_rollout():
                    return self._rollout_mutation_error()
                while True:
                    existing = await self._db.fetchone(
                        """
                        SELECT id, next_run_at, idempotency_key,
                               scheduler_registration_nonce
                        FROM scheduled_tasks
                        WHERE agent_id = ?
                          AND (task_name = ? OR idempotency_key = ?)
                        ORDER BY CASE WHEN idempotency_key = ? THEN 0 ELSE 1 END,
                                 created_at ASC
                        LIMIT 1
                        """,
                        (
                            self._agent_id,
                            task_name,
                            stable_idempotency,
                            stable_idempotency,
                        ),
                    )
                    if existing is None:
                        return await self.schedule_add(
                            cron_expression=cron_expression,
                            task_name=task_name,
                            args_json=args_json,
                            idempotency_key=stable_idempotency,
                        )

                    # The prior registration owns this row only while no
                    # other host has relied on it.  Adopt the exact selected
                    # built-in before returning it to another registration
                    # (or a committed host), so that the original owner's
                    # later rollback cannot remove shared scheduler state.
                    # A retry by the same pending registration deliberately
                    # retains its marker: that registration still owns the
                    # row until onboarding commits.
                    existing_registration_nonce = existing[3]
                    pending_registration_nonce = (
                        self._pending_scheduler_registration_nonce()
                    )
                    if (
                        existing_registration_nonce is not None
                        and existing_registration_nonce
                        != pending_registration_nonce
                    ):
                        adopted = await adopt_scheduler_registration_ownership(
                            self._db,
                            task_id=existing[0],
                            agent_id=self._agent_id,
                            observed_registration_nonce=existing_registration_nonce,
                            pending_registration_nonce=pending_registration_nonce,
                        )
                        if not adopted:
                            # The first registration may have rolled the row
                            # back after our selection but before adoption.
                            # Do not report a vanished schedule as reusable.
                            # Recheck while retaining the rollout lock; if a
                            # competing host has not replaced the row, this
                            # iteration selects that replacement instead.
                            continue
                    return ToolResult.ok(
                        confirmation=(
                            f"Built-in schedule '{task_name}' already exists"
                        ),
                        data={
                            "success": True,
                            "task_id": existing[0],
                            "task_name": task_name,
                            "next_run_at": existing[1],
                            "existing": True,
                        },
                    )
        except Exception as error:
            logger.error(
                "Failed to ensure built-in schedule %s: %s",
                task_name,
                error,
            )
            return ToolResult.failed(str(error))

    async def shutdown(self):
        """Stop the background runner."""
        if self._runner:
            await self._runner.stop()
        # Unregister the cron / pr-watch / discovery sources (base #2522 P2).
        await super().shutdown()

    # ------------------------------------------------------------------
    # Task executor — dispatches via SignalDispatcher (Phase 4 of #889)
    # ------------------------------------------------------------------

    async def _dispatch_scheduled_task(self, task_name: str, args: dict) -> Any:
        """SchedulerRunner executor — builds a Signal envelope per task
        and routes through the agent's SignalDispatcher.

        Returns whatever shape the runner expects from the legacy
        executor: a string (or a (text, outcome_signal) tuple). The
        translation lives in `_translate_signal_result`.

        The cron expression and task_execution_log shape are unchanged.
        Per-task mode (ACTION/ARTIFACT) and resource locks come from the
        SourceRegistration built in `signals/sources/scheduler.py`.
        """
        from kestrel_sdk.signals import Signal, Visibility
        from kestrel_sovereign.signals.sources.scheduler import (
            CRON_TASKS,
            cron_source_name,
        )

        dispatcher = getattr(self.agent, "dispatcher", None)
        if dispatcher is None:
            # Fallback for partially-initialized agents (e.g. legacy
            # test fixtures): execute the tool directly without going
            # through the signal pipeline. Production agents always
            # have a dispatcher; this branch is defensive only.
            logger.warning(
                "SchedulerFeature: no dispatcher on agent, "
                "executing %r directly", task_name,
            )
            return await self._lookup_and_run_tool(task_name, args)

        # Look up the task's mode from the classification table. If a
        # task fires that isn't in CRON_TASKS, it has no source
        # registration — fall back to direct tool execution rather than
        # rejecting (preserves backward compat for ad-hoc tools added
        # via `!schedule add <cron> <custom_tool>`).
        mode_by_name = {name: mode for name, mode, _ in CRON_TASKS}
        mode = mode_by_name.get(task_name)
        if mode is None:
            logger.info(
                "SchedulerFeature: %r has no source registration, "
                "executing directly", task_name,
            )
            return await self._lookup_and_run_tool(task_name, args)

        signal = Signal(
            source=cron_source_name(task_name),
            kind="run",
            mode=mode,
            payload=args or {},
            target_agent=self.agent.did,
            visibility=Visibility.INTERNAL,
        )
        result = await dispatcher.dispatch_signal(signal)
        return self._translate_signal_result(result, task_name)

    @staticmethod
    def _translate_signal_result(result, task_name: str) -> Any:
        """Map a SignalResult into the runner's expected return shape
        (str | (str, float) tuple | ScheduledTaskOutcome | None).

        - OK                    → action_result if ACTION, artifact if ARTIFACT
        - FAILED, DROPPED_VALIDATION, DROPPED_CYCLE
                                → raise RuntimeError → runner records
                                  status='failed'. These represent
                                  misconfiguration (bad args, cycle in
                                  the causation chain) — silently
                                  recording them as 'success' would
                                  hide real bugs.
        - DROPPED_RATE_LIMIT, DROPPED_QUIET_HOURS, COALESCED
                                → "skipped: <status>" string (success
                                  row; benign skip the operator can
                                  grep for in result_text).
        """
        from kestrel_sdk.signals import SignalMode, Status

        if result.status == Status.OK:
            payload = (
                result.action_result if result.mode == SignalMode.ACTION
                else result.artifact
            )
            if isinstance(payload, ScheduledTaskOutcome):
                return payload
            # Tools may return a string, a (text, signal) tuple, a Dict,
            # or None. The runner JSON-stringifies non-tuple values for
            # task_execution_log.result_text. Pass through as-is so the
            # runner's existing handling (which keeps tuples for
            # outcome_signal extraction) keeps working.
            if isinstance(payload, tuple):
                return payload
            if isinstance(payload, str):
                return payload
            if payload is None:
                return None
            # Dict / other → JSON-encode here (matches the legacy
            # behavior of `json.dumps(result, default=str)` in the old
            # _execute_scheduled_task body).
            return json.dumps(payload, default=str)

        # Failure-equivalent drops — surface as exceptions so the
        # runner records status='failed'.
        if result.status in (
            Status.FAILED,
            Status.DROPPED_VALIDATION,
            Status.DROPPED_CYCLE,
        ):
            raise RuntimeError(
                f"dispatch {result.status.value} for {task_name}: "
                f"{result.error or 'unknown'}"
            )

        # Benign drops (rate limit, quiet hours, coalesced) — recorded
        # as success with a short text describing the drop.
        return f"skipped: {result.status.value} ({result.error or ''})".strip(" ()")

    # ------------------------------------------------------------------
    # Source-registration handlers
    # ------------------------------------------------------------------

    async def _run_tool_hook_gated(
        self, feature_name: str, agent_tool: Any, args: dict
    ) -> Any:
        """Run a feature tool on a scheduler tick through the SAME
        PRE_TOOL_USE / POST_TOOL_USE hook envelope the chat and subagent
        dispatch paths use (F245).

        Before this, scheduled feature-tool execution called
        ``agent_tool.execute(**args)`` directly, so ``SecurityHook``
        DENY/ASK policy and MODIFY redaction never fired — a DENY/ASK-
        gated tool executed unchecked on every tick. Now the security
        gate applies on every tick: a DENY or ASK decision becomes a
        structured blocked outcome (so the runner records the reason and
        pauses the schedule), and a MODIFY hook's redacted args are the ones
        actually passed to the tool.
        """
        hooks_manager = getattr(self.agent, "hooks_manager", None)
        effective_args = args
        if hooks_manager is not None:
            from kestrel_sdk.hooks.base import HookEvent, HookInput
            from kestrel_sovereign.hooks.decision_gate import (
                evaluate_blocking_decision,
            )

            pre_input = HookInput(
                session_id="scheduler",
                hook_event_name=HookEvent.PRE_TOOL_USE.value,
                tool_name=agent_tool.name,
                tool_input=args,
                feature_name=feature_name,
            )
            pre_output = await hooks_manager.execute_hooks(
                HookEvent.PRE_TOOL_USE, pre_input,
            )
            # Resolve post-hook (redacted) args first — a MODIFY hook may
            # run before a later DENY hook; the effective args must be the
            # redacted form even on the block path.
            mutated = getattr(pre_input, "tool_input", None)
            if isinstance(mutated, dict):
                effective_args = mutated
            updated = getattr(pre_output, "updated_input", None)
            if isinstance(updated, dict):
                effective_args = updated

            # DENY and ASK both block the scheduled run. A scheduler tick has
            # no interactive approver, so return a structured blocked outcome
            # that the runner can persist and pause. Returning instead of
            # raising also keeps this expected policy state out of the signal
            # dispatcher's ERROR/traceback path (#2430).
            blocked = evaluate_blocking_decision(pre_output)
            if blocked is not None:
                logger.warning(
                    "Scheduler: tool %r blocked (%s) on tick: %s",
                    agent_tool.name, blocked.decision.value, blocked.reason,
                )
                return ScheduledTaskOutcome.blocked(
                    task_name=agent_tool.name,
                    decision=blocked.decision.value,
                    reason=blocked.reason or blocked.error,
                )

            result = await agent_tool.execute(**effective_args)

            post_input = HookInput(
                session_id="scheduler",
                hook_event_name=HookEvent.POST_TOOL_USE.value,
                tool_name=agent_tool.name,
                tool_input=effective_args,
                feature_name=feature_name,
                tool_response=(
                    result if isinstance(result, dict) else {"result": str(result)}
                ),
            )
            await hooks_manager.execute_hooks_parallel(
                HookEvent.POST_TOOL_USE, post_input,
            )
            return result

        # No hooks manager (bare test host) — execute directly.
        return await agent_tool.execute(**effective_args)

    def _semantic_maintenance_required_for_training(self) -> bool:
        """Whether this agent has an operator-selected semantic phase.

        This mirrors the agent's sleep predicate without asking a scheduled
        consumer to infer a profile.  A validation-only maintenance
        configuration is still a governed semantic boundary and therefore
        gates corpus/training consumers too.
        """
        if (
            getattr(self.agent, "semantic_inference_configured", False) is True
            or getattr(self.agent, "semantic_maintenance_configured", False) is True
        ):
            return True
        from kestrel_sovereign.knowledge.inference import InferenceProfile

        return isinstance(
            getattr(self.agent, "semantic_inference_profile", None),
            InferenceProfile,
        )

    async def _training_cycle_semantic_maintenance_gate(
        self,
    ) -> ScheduledTaskOutcome | None:
        """Block scheduled training unless durable semantic input is current.

        ``training_cycle`` is deliberately not part of the sleep cron: it can
        be scheduled independently, including before sleep's next run.  Its
        scheduler path must consequently enforce the same semantic data
        boundary as phase-ordered post-consolidation hooks.  This is a
        non-pausing block; a later maintenance completion should make the next
        regular training tick eligible without manual schedule repair.
        """
        if not self._semantic_maintenance_required_for_training():
            return None
        storage = getattr(self.agent, "storage", None)
        readiness = getattr(
            storage, "semantic_maintenance_training_readiness", None
        )
        if not callable(readiness):
            return ScheduledTaskOutcome(
                status="blocked",
                result_text=(
                    "blocked: training_cycle requires durable semantic "
                    "maintenance, but its governed status is unavailable"
                ),
            )
        try:
            state = await readiness(
                getattr(self.agent, "semantic_inference_profile", None),
                inference_limits=getattr(
                    self.agent, "semantic_inference_limits", None
                ),
                maintenance_limits=getattr(
                    self.agent, "semantic_maintenance_limits", None
                ),
                allow_prior_verified_snapshot=(
                    getattr(
                        self.agent,
                        "semantic_maintenance_allow_prior_verified_snapshot",
                        False,
                    )
                    is True
                ),
            )
        except Exception as error:  # noqa: BLE001 - fail closed at the data boundary
            logger.warning(
                "Scheduler: semantic maintenance readiness failed for training_cycle: %s",
                type(error).__name__,
            )
            return ScheduledTaskOutcome(
                status="blocked",
                result_text=(
                    "blocked: training_cycle requires durable semantic "
                    "maintenance, but its status could not be verified"
                ),
            )
        if state.ready:
            if state.using_prior_verified_snapshot:
                logger.warning(
                    "Scheduler: training_cycle using operator-permitted prior "
                    "verified semantic snapshot"
                )
            return None
        return ScheduledTaskOutcome(
            status="blocked",
            result_text=(
                "blocked: training_cycle requires complete current semantic "
                f"maintenance ({state.reason or 'semantic_maintenance_unverified'})"
            ),
        )

    @staticmethod
    def _feature_enabled(feature: Any) -> bool:
        """Whether ``feature`` is live-enabled and safe for the scheduler to run.

        A soft-disabled feature stays in ``agent.features`` with
        ``enabled=False`` but has ALL of its live surfaces detached (tools,
        hooks, A2A agent, routes) by ``_unregister_feature_runtime``. The
        scheduler must treat it as absent so a persisted schedule can't invoke a
        disabled tool on a background tick (#2522) — the one execution path that
        never re-checks the orchestrator's own ``enabled`` gate. Mirrors that
        gate's ``getattr(feature, "enabled", True)`` exactly, so a feature that
        never set the flag (the normal booted state) stays executable.
        """
        return bool(getattr(feature, "enabled", True))

    async def _lookup_and_run_tool(self, task_name: str, args: dict) -> Any:
        """Tool-lookup body shared by every cron source handler that
        delegates to a feature tool. This is the existing executor's
        tool-search logic, lifted out so the source registrations can
        invoke it without re-entering the dispatcher (which would loop).

        Every resolved tool runs through ``_run_tool_hook_gated`` so the
        PRE_TOOL_USE/POST_TOOL_USE hooks and the SecurityHook DENY/ASK
        gate fire on each tick (F245). Disabled features are skipped so a
        persisted schedule cannot invoke a soft-disabled tool (#2522)."""
        if task_name == "training_cycle":
            blocked = await self._training_cycle_semantic_maintenance_gate()
            if blocked is not None:
                return blocked
        features = getattr(self.agent, "features", {})
        for feature in features.values():
            if not hasattr(feature, "get_tools"):
                continue
            if not self._feature_enabled(feature):
                # A disabled feature's tools are detached from every other live
                # surface; the scheduler skips it too (handled benignly below if
                # the task actually resolves to it).
                continue
            for agent_tool in feature.get_tools():
                if agent_tool.name == task_name:
                    result = await self._run_tool_hook_gated(
                        type(feature).__name__, agent_tool, args,
                    )
                    if isinstance(result, ScheduledTaskOutcome):
                        return result
                    # Preserve the legacy JSON-encode contract for
                    # downstream consumers (task_execution_log.result_text,
                    # endpoints/agent.py history view).
                    if isinstance(result, str):
                        return result
                    return json.dumps(_serialize_tool_result(result), default=str)

        # Also check our own tools (SchedulerFeature has !schedule
        # commands but they're not typically scheduled themselves).
        for agent_tool in self.get_tools():
            if agent_tool.name == task_name:
                result = await self._run_tool_hook_gated(
                    type(self).__name__, agent_tool, args,
                )
                if isinstance(result, ScheduledTaskOutcome):
                    return result
                if isinstance(result, str):
                    return result
                return json.dumps(_serialize_tool_result(result), default=str)

        # A persisted schedule that names a tool owned by a NOW-disabled feature
        # must not execute it. Skip benignly (like the startup-order race below)
        # rather than raising, so a disable doesn't spam the execution log with a
        # failure every tick; re-enabling the feature restores execution on the
        # next tick. Detected AFTER the enabled-feature search so an enabled
        # owner always wins, and distinguished from a genuinely-unknown task so
        # the operator sees the real reason.
        for feature in features.values():
            if not hasattr(feature, "get_tools") or self._feature_enabled(feature):
                continue
            try:
                disabled_tools = feature.get_tools()
            except Exception:  # noqa: BLE001 - a broken feature can't block others
                continue
            if any(getattr(t, "name", None) == task_name for t in disabled_tools):
                feature_name = getattr(feature, "name", type(feature).__name__)
                logger.info(
                    "Scheduler: task %r is owned by disabled feature %r; "
                    "skipping this tick", task_name, feature_name,
                )
                return (
                    f"skipped: {task_name} owning feature {feature_name!r} "
                    "is disabled"
                )

        # A persisted built-in cron task (e.g. restart_coordinator) can
        # fire on the first scheduler tick after a restart BEFORE its
        # owning feature has finished loading and registered the tool —
        # the runner starts polling in initialize() while feature load
        # order is not guaranteed (#1796). That is a transient startup-
        # order race, not a misconfiguration: a later tick (once the
        # feature is loaded) runs the task normally. Skip it benignly
        # this tick instead of raising "Unknown task", which would record
        # a spurious one-time failure in the execution log.
        from kestrel_sovereign.signals.sources.scheduler import CRON_TASKS

        if task_name in {name for name, _mode, _res in CRON_TASKS}:
            logger.info(
                "Scheduler: built-in cron task %r not yet resolvable "
                "(owning feature still loading); skipping this tick",
                task_name,
            )
            return (
                f"skipped: {task_name} owning feature not loaded yet "
                "(transient startup-order race)"
            )

        raise ValueError(f"Unknown task: {task_name}")

    def _scheduler_executable_task_names(self) -> set:
        """Return the set of task names the scheduler can actually run.

        Used by ``schedule_add`` to reject unknown names at creation time
        (#1618). A task is executable if it is one of:
          - a built-in cron source (``CRON_TASKS``), including the
            built-in-handler tasks like ``backup_snapshot`` and
            ``github_pr_watch`` that don't go through tool lookup;
          - a feature tool discoverable by ``_lookup_and_run_tool``
            (any loaded feature's ``get_tools()``);
          - one of this scheduler feature's own tools.

        Mirrors the resolution order of the runtime executor so the
        validation can't drift from what actually runs.
        """
        from kestrel_sovereign.signals.sources.scheduler import CRON_TASKS

        names: set = {name for name, _mode, _res in CRON_TASKS}

        features = getattr(self.agent, "features", {}) or {}
        for feature in features.values():
            if not hasattr(feature, "get_tools"):
                continue
            if not self._feature_enabled(feature):
                # A disabled feature's tools are not executable, so they are not
                # schedulable — mirrors the runtime executor's skip (#2522), so
                # schedule_add can't accept a name that would then be skipped
                # every tick.
                continue
            try:
                for agent_tool in feature.get_tools():
                    names.add(agent_tool.name)
            except Exception:
                # A misbehaving feature must not block validation of
                # everything else; skip it -- but log it, because a
                # silent skip drops that feature's task names from the
                # "every currently-valid name" set that schedule_add's
                # rejection error promises (AGENTS.md), so an operator
                # would otherwise see a legitimate name rejected with no
                # trace of why.
                logger.warning(
                    "Feature %r get_tools() raised while collecting "
                    "scheduler task names; its task names are omitted "
                    "from the valid set (a schedule_add for them will be "
                    "rejected as unknown)",
                    getattr(feature, "name", type(feature).__name__),
                    exc_info=True,
                )
                continue

        for agent_tool in self.get_tools():
            names.add(agent_tool.name)

        return names

    async def _scheduled_task_denied(self, task_name: str) -> bool:
        """Return True if ``task_name`` resolves to a feature tool the
        SecurityFeature permission store has set to ``DENY``.

        ``schedule_add`` uses this to reject a DENY-listed tool at creation
        time (F245) instead of persisting it and having the tick-path hook
        gate (``_run_tool_hook_gated``) fail every single tick. Resolution
        mirrors that executor: the owning feature is looked up by the same
        ``agent.features`` key SecurityFeature registered its permissions
        under, and the tool name is the ``@tool`` name.

        Built-in cron sources (``CRON_TASKS``) have no permission row and
        are never treated as denied here — they are operator-level scheduler
        primitives, not agent tools. On any lookup failure or a missing
        SecurityFeature this returns ``False`` (fail-open only for the
        *creation-time* check; the tick-path gate still fires at runtime).
        """
        features = getattr(self.agent, "features", {}) or {}
        security_feature = features.get("SecurityFeature")
        if security_feature is None:
            return False
        store = getattr(security_feature, "permission_store", None)
        if not store or not callable(getattr(store, "get_permission", None)):
            return False

        from kestrel_sovereign.features.security.permissions import PermissionLevel

        # Resolve the owning feature the SAME way SecurityFeature registered
        # its permissions: keyed by the agent.features name (feature.name).
        for feature_name, feature in features.items():
            if not hasattr(feature, "get_tools"):
                continue
            if not self._feature_enabled(feature):
                # A disabled feature's tool is not schedulable at all
                # (rejected by _scheduler_executable_task_names), so its
                # permission row is irrelevant here — skip it for consistency
                # with the runtime executor (#2522).
                continue
            try:
                tools = feature.get_tools()
            except Exception:
                # A misbehaving feature must not block creation validation.
                continue
            for agent_tool in tools:
                if agent_tool.name == task_name:
                    try:
                        level = await store.get_permission(
                            feature_name, agent_tool.name
                        )
                    except Exception:
                        return False
                    return level == PermissionLevel.DENY
        return False

    async def _handle_backup_snapshot(self, args: dict) -> str:
        """ACTION handler for the `backup_snapshot` cron source. Hits
        the agent's sync service directly — there is no feature tool
        for this, so it can't go through the generic tool lookup.

        Uses ``snapshot_if_changed`` (#1674 P3) so a scheduled backup on an
        idle agent skips re-dumping an unchanged DB (notably the full S3
        upload). Explicit ``!backup`` / shutdown still call ``force_snapshot``."""
        sync = getattr(self.agent, "_sync_service", None)
        if sync:
            snap = getattr(sync, "snapshot_if_changed", None) or sync.force_snapshot
            results = await snap()
            return json.dumps(
                {t: {"success": r.success, "bytes": r.bytes_synced} for t, r in results.items()},
                default=str,
            )
        return json.dumps({"error": "no sync service configured"})

    async def _handle_sleep(self, args: dict) -> str:
        """Built-in handler for the nightly ``sleep`` cron (#1674 P3).

        Runs the agent's single memory-maintenance cycle: reflection (via the
        subscribed ``sleep_hooks``), consolidation, and the forgetting
        deletion tier — all through ``MemorySystem.consolidate()``. Backups keep
        their own 4h disaster-recovery cadence, so this defaults to
        ``skip_export=True`` (override via the schedule's args JSON).

        Activity-gated: on an idle agent (no new messages since the last
        episode) the expensive reflection pass is skipped — directly addressing
        the "reflection every cycle even with no activity" waste. Consolidation
        still runs (cheap, and the time/decay-based forgetting tier rides it).
        """
        agent = self.agent
        if not hasattr(agent, "sleep"):
            return json.dumps({"skipped": True, "reason": "agent has no sleep()"})

        skip_export = bool(args.get("skip_export", True))
        skip_consolidation = bool(args.get("skip_consolidation", False))
        # Caller can force reflection on/off; otherwise gate it on activity.
        if "skip_reflection" in args:
            skip_reflection = bool(args["skip_reflection"])
        else:
            skip_reflection = not await self._sleep_had_activity()

        try:
            report = await agent.sleep(
                skip_export=skip_export,
                skip_consolidation=skip_consolidation,
                skip_reflection=skip_reflection,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[sleep] agent=%s cycle failed: %s", self._agent_id, e)
            return json.dumps({"error": str(e)})

        data = report.to_dict() if hasattr(report, "to_dict") else {}
        data["skip_reflection"] = skip_reflection
        return json.dumps(data, default=str)

    async def _sleep_had_activity(self) -> bool:
        """Best-effort: has anything happened since the last episode?

        Returns True (don't skip) on ANY uncertainty so reflection is never
        wrongly skipped — the gate may only ever skip when confidently idle.

        Compares the newest conversation message against the newest episode by
        parsing both in Python: memory_episodes.created_at is ISO-with-`T`/tz
        (consolidator writes datetime.now(utc).isoformat()) while
        conversation_history.created_at is SQLite CURRENT_TIMESTAMP (space-
        separated, UTC). A raw SQL string `>` would mis-sort those formats and
        wrongly mark an active agent idle, so we normalize both to naive-UTC."""
        storage = getattr(self.agent, "storage", None)
        db = getattr(storage, "db", None)
        if db is None:
            return True
        try:
            last_msg_raw = await db.fetchval(
                "SELECT MAX(created_at) FROM conversation_history WHERE agent_id = ? AND deleted_at IS NULL",
                (self._agent_id,),
            )
            last_msg = self._parse_ts_utc(last_msg_raw)
            if last_msg is None:
                return False  # no messages at all → genuinely idle
            last_ep = self._parse_ts_utc(
                await db.fetchval(
                    "SELECT MAX(created_at) FROM memory_episodes WHERE agent_id = ?",
                    (self._agent_id,),
                )
            )
            # New message since the last episode (or no episode yet) → active.
            return last_ep is None or last_msg > last_ep
        except Exception as e:  # noqa: BLE001 - never wrongly skip on error
            logger.debug("[sleep] activity check failed (assuming active): %s", e)
            return True

    @staticmethod
    def _parse_ts_utc(value) -> Optional[datetime]:
        """Parse a stored timestamp (ISO-with-tz, space-separated, or native
        datetime) to a naive-UTC datetime for safe cross-format comparison.
        Returns None when absent/unparseable."""
        if value is None:
            return None
        dt = value if isinstance(value, datetime) else None
        if dt is None:
            try:
                dt = datetime.fromisoformat(str(value))
            except (ValueError, TypeError):
                return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    async def _run_trash_retention(self, args: dict) -> str:
        """Built-in handler for the ``trash_retention`` scheduled task (#764).

        Reads the ``[trash]`` section from ``kestrel.toml``, resolves the
        retention window for the agent's current privacy mode, computes
        the cutoff, and asks the storage facade to hard-purge soft-
        deleted rows past that cutoff. Returns a JSON summary so the
        task-execution log carries the per-run breakdown.

        Skips with a warning when the resolved retention is zero or
        negative — purging on a same-day cutoff would scrub rows the
        user might still be reaching for. Operators can disable the
        rail entirely with ``!schedule pause <task_id>``.

        Args:
            args: Optional overrides — supports ``max_rows`` (int) for
                the per-sweep cap. Empty dict is the common case.
        """
        from datetime import datetime, timedelta, timezone
        from kestrel_sovereign.storage.retention import (
            DEFAULT_MAX_ROWS_PER_SWEEP,
            agent_privacy_mode,
            load_trash_config,
            resolve_retention_days,
        )

        # Durable signal history has a source-specific deadline rather than
        # the conversation-trash policy below.  It shares this periodic
        # maintenance rail, but never lets a delivery-ledger failure block the
        # unrelated conversation sweep (or vice versa).
        durable_purged: Optional[int] = None
        dispatcher = getattr(self.agent, "dispatcher", None)
        if dispatcher is not None:
            purge_durable = getattr(
                dispatcher, "purge_expired_durable_deliveries", None
            )
            if callable(purge_durable):
                try:
                    durable_purged = await purge_durable()
                except Exception as exc:
                    logger.warning(
                        "[retention] durable signal cleanup failed for agent=%s: %s",
                        self._agent_id,
                        exc,
                    )

        durable_summary = (
            {"durable_signal_events_purged": durable_purged}
            if durable_purged is not None
            else {}
        )

        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "purge_trash_older_than"):
            return json.dumps({
                "skipped": True,
                "reason": "storage facade missing purge_trash_older_than",
                **durable_summary,
            })

        config = load_trash_config()
        privacy_mode = agent_privacy_mode(self.agent)
        retention_days = resolve_retention_days(
            config=config, privacy_mode=privacy_mode,
        )
        if retention_days is None:
            logger.warning(
                "[retention] skipping agent=%s — config sets retention to "
                "a non-positive value (would purge instantly)",
                self._agent_id,
            )
            return json.dumps({
                "skipped": True,
                "reason": "non-positive retention window",
                "privacy_mode": privacy_mode,
                **durable_summary,
            })

        max_rows = int(args.get("max_rows") or DEFAULT_MAX_ROWS_PER_SWEEP)
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_iso = cutoff.replace(tzinfo=None).isoformat(sep=" ")

        try:
            purged = await storage.purge_trash_older_than(
                cutoff_iso, max_rows=max_rows, reason="retention-janitor",
            )
        except Exception as e:
            logger.warning(
                "[retention] agent=%s sweep failed: %s", self._agent_id, e,
            )
            return json.dumps({"error": str(e), **durable_summary})

        if purged:
            logger.info(
                "[retention.sweep] agent=%s privacy=%s window=%dd "
                "rows_purged=%d cap=%d",
                self._agent_id, privacy_mode, retention_days, purged, max_rows,
            )
        return json.dumps({
            "rows_purged": purged,
            "privacy_mode": privacy_mode,
            "retention_days": retention_days,
            "cutoff": cutoff_iso,
            "max_rows": max_rows,
            **durable_summary,
        })

    # ------------------------------------------------------------------
    # wait_reconcile (Wave 2 of #1860)
    # ------------------------------------------------------------------

    async def _run_wait_reconcile(self, args: dict) -> str:
        """Built-in handler for the ``wait_reconcile`` scheduled task.

        Delegates to the agent's singleton :class:`WaitReconciler` (built
        lazily + cached on ``agent._wait_reconciler`` so its in-memory
        pending-task map survives across ticks). This is the generic
        successor to the talon-specific ``talon_monitor`` cron — it wakes
        the agent on ANY MonitorableWaitable provider's terminal handle.

        Returns a JSON summary so the task_execution_log carries the
        per-run delivery breakdown.

        Args:
            args: Unused (the reconciler enumerates the registry itself).
        """
        from kestrel_sovereign.waits.reconciler import run_wait_reconcile

        result = await run_wait_reconcile(self.agent)
        return json.dumps(result.data or {}, default=str)

    async def _run_bootstrap_timeout_check(self, args: dict) -> str:
        """Built-in handler for the optional bootstrap timeout watchdog."""
        from kestrel_sovereign.lifecycle_checks import warn_stale_bootstrap_pending

        threshold_seconds = int(args.get("threshold_seconds") or 3600)
        stale = await warn_stale_bootstrap_pending(
            self.agent,
            threshold_seconds=threshold_seconds,
            context="scheduler",
        )
        return json.dumps(
            stale or {
                "is_stale": False,
                "status": "ok",
                "threshold_seconds": threshold_seconds,
            },
            default=str,
        )

    # ------------------------------------------------------------------
    # github_pr_watch (#1618)
    # ------------------------------------------------------------------

    async def _ensure_pr_watch_table(self) -> None:
        """Lazily create the per-agent PR-watch fingerprint table."""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS github_pr_watch_state (
                agent_id        TEXT NOT NULL,
                watch_key       TEXT NOT NULL,
                fingerprint     TEXT NOT NULL,
                normalized_json TEXT,
                updated_at      TEXT,
                PRIMARY KEY (agent_id, watch_key)
            )
            """
        )

    async def _load_pr_watch_state(self, watch_key: str):
        """Return ``(fingerprint, normalized_dict)`` for a watch, or
        ``(None, None)`` if this is the first observation."""
        await self._ensure_pr_watch_table()
        row = await self._db.fetchone(
            "SELECT fingerprint, normalized_json FROM github_pr_watch_state "
            "WHERE agent_id = ? AND watch_key = ?",
            (self._agent_id, watch_key),
        )
        if not row:
            return None, None
        normalized = None
        if row[1]:
            try:
                parsed = json.loads(row[1])
                normalized = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                normalized = None
        return row[0], normalized

    async def _save_pr_watch_state(
        self, watch_key: str, fingerprint: str, normalized: dict
    ) -> None:
        """Upsert the latest observed fingerprint for a watch."""
        await self._ensure_pr_watch_table()
        now_iso = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT INTO github_pr_watch_state
                (agent_id, watch_key, fingerprint, normalized_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, watch_key) DO UPDATE SET
                fingerprint = excluded.fingerprint,
                normalized_json = excluded.normalized_json,
                updated_at = excluded.updated_at
            """,
            (
                self._agent_id,
                watch_key,
                fingerprint,
                json.dumps(normalized, default=str),
                now_iso,
            ),
        )

    # ------------------------------------------------------------------
    # ecosystem_discovery_watch (#2281)
    # ------------------------------------------------------------------

    async def _ensure_ecosystem_discovery_table(self) -> None:
        """Lazily create the per-agent discovery-watch fingerprint table."""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS ecosystem_discovery_watch_state (
                agent_id        TEXT NOT NULL,
                watch_key       TEXT NOT NULL,
                fingerprint     TEXT NOT NULL,
                state_json      TEXT,
                updated_at      TEXT,
                PRIMARY KEY (agent_id, watch_key)
            )
            """
        )

    async def _load_ecosystem_discovery_state(self, watch_key: str):
        """Return ``(fingerprint, state_dict)`` for a discovery watch."""
        from kestrel_sovereign.signals.sources.ecosystem_discovery import (
            state_from_json,
        )

        await self._ensure_ecosystem_discovery_table()
        row = await self._db.fetchone(
            "SELECT fingerprint, state_json FROM ecosystem_discovery_watch_state "
            "WHERE agent_id = ? AND watch_key = ?",
            (self._agent_id, watch_key),
        )
        if not row:
            return None, None
        return row[0], state_from_json(row[1])

    async def _save_ecosystem_discovery_state(
        self, watch_key: str, fingerprint: str, state_json: str
    ) -> None:
        """Upsert the latest observed discovery fingerprint."""
        await self._ensure_ecosystem_discovery_table()
        now_iso = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT INTO ecosystem_discovery_watch_state
                (agent_id, watch_key, fingerprint, state_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, watch_key) DO UPDATE SET
                fingerprint = excluded.fingerprint,
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (self._agent_id, watch_key, fingerprint, state_json, now_iso),
        )

    async def _run_ecosystem_discovery_watch(self, args: dict) -> str:
        """Run stale-work/red-CI discovery and wake on actionable changes.

        Args:
            tool: discovery tool name, default ``scan_stale_work``.
            tool_args: kwargs passed to that tool. Any top-level args other
                than watcher control keys are also forwarded for convenience.
            watch_key: optional dedupe key. Defaults to tool + repo/filter args.
            notify: optional target agent DID (defaults to this agent).
            max_findings: maximum findings included in the cognition payload.
        """
        from kestrel_sovereign.signals.sources.ecosystem_discovery import (
            DEFAULT_DISCOVERY_TOOL,
            build_signal_for_discovery_findings,
            evaluate_discovery_watch,
            state_to_json,
        )

        tool_name = str(args.get("tool") or DEFAULT_DISCOVERY_TOOL)
        control_keys = {"tool", "tool_args", "watch_key", "notify", "max_findings"}
        tool_args = dict(args.get("tool_args") or {})
        for key, value in args.items():
            if key not in control_keys:
                tool_args.setdefault(key, value)

        repo = str(tool_args.get("repo") or tool_args.get("repository") or "")
        try:
            max_findings = int(args.get("max_findings") or 20)
        except (TypeError, ValueError):
            max_findings = 20

        watch_key = str(
            args.get("watch_key")
            or f"{tool_name}:{repo or json.dumps(tool_args, sort_keys=True, default=str)}"
        )

        try:
            raw_result = await self._lookup_and_run_tool(tool_name, tool_args)
        except Exception as e:
            return json.dumps({
                "signaled": False,
                "blocked": "tool_error",
                "error": f"{type(e).__name__}: {e}",
                "watch_key": watch_key,
                "tool": tool_name,
            })

        last_fp, last_state = await self._load_ecosystem_discovery_state(watch_key)
        decision = evaluate_discovery_watch(
            raw_result,
            last_fingerprint=last_fp,
            last_state=last_state,
            default_repo=repo,
            max_findings=max_findings,
        )
        state_json = state_to_json(decision.state)

        if not decision.should_signal:
            await self._save_ecosystem_discovery_state(
                watch_key, decision.state.fingerprint, state_json,
            )
            return json.dumps({
                "signaled": False,
                "reason": decision.reason,
                "watch_key": watch_key,
                "findings_count": len(decision.state.findings),
            })

        target = args.get("notify") or getattr(self.agent, "did", "") or self._agent_id
        signal = build_signal_for_discovery_findings(
            watch_key=watch_key,
            tool_name=tool_name,
            decision=decision,
            target_agent=str(target),
        )

        dispatcher = getattr(self.agent, "dispatcher", None)
        if dispatcher is None or not hasattr(dispatcher, "enqueue_signal"):
            return json.dumps({
                "signaled": False,
                "blocked": "no_dispatcher",
                "reason": "agent has no signal dispatcher",
                "watch_key": watch_key,
                "findings_count": len(decision.state.findings),
            })

        try:
            enq = dispatcher.enqueue_signal(signal)
            if hasattr(enq, "__await__"):
                await enq
        except Exception as e:
            logger.warning(
                "ecosystem_discovery_watch: enqueue_signal raised for %s: %s",
                watch_key, e,
            )
            return json.dumps({
                "signaled": False,
                "blocked": "dispatch_error",
                "error": f"{type(e).__name__}: {e}",
                "watch_key": watch_key,
                "findings_count": len(decision.state.findings),
            })

        await self._save_ecosystem_discovery_state(
            watch_key, decision.state.fingerprint, state_json,
        )
        return json.dumps({
            "signaled": True,
            "reason": decision.reason,
            "watch_key": watch_key,
            "findings_count": len(decision.state.findings),
        })

    async def _run_github_pr_watch(self, args: dict) -> str:
        """Built-in handler for the ``github_pr_watch`` cron task (#1618).

        Polls a single GitHub PR/issue, fingerprints the watched fields,
        and enqueues one ``github.pr_activity`` COGNITION signal only when
        a relevant change is detected. A no-op poll emits nothing.

        Reports ``blocked: auth`` / ``blocked: network`` distinctly from a
        no-change poll so a bad token or flaky network is never silently
        read as "nothing happened". The persisted fingerprint is NOT
        advanced on a blocked poll, so the next successful poll still sees
        the real delta.

        Args (from the scheduled task's args_json):
            repo: ``owner/name`` of the repository.
            pr / issue / number: the PR or issue number to watch.
            triggers: optional list of change categories that wake the
                agent (default: state, merge, comments, checks). Pass
                ``["any"]`` to wake on every fingerprint change.
            notify: optional target agent DID (defaults to this agent).
        """
        from kestrel_sovereign.features.strategic_memory.github_integration import (
            get_github_token,
        )
        from kestrel_sovereign.signals.sources.github_pr_watch import (
            PRWatchAuthError,
            PRWatchNetworkError,
            build_signal_for_pr_change,
            evaluate_pr_watch,
            fetch_pr_state,
        )

        repo = args.get("repo")
        # PRs and issues share one numbering space but live behind different
        # API endpoints, so we track which arg supplied the number. An
        # explicit 'issue' (without 'pr') fetches /issues/{n}; everything
        # else fetches /pulls/{n}.
        if args.get("pr"):
            number = args.get("pr")
            kind = "pr"
        elif args.get("issue"):
            number = args.get("issue")
            kind = "issue"
        else:
            number = args.get("number")
            kind = "pr"
        if not repo or not number:
            return json.dumps({
                "signaled": False,
                "error": (
                    "github_pr_watch requires 'repo' and one of "
                    "'pr'/'issue'/'number' in args"
                ),
            })

        try:
            number_int = int(number)
        except (TypeError, ValueError):
            return json.dumps({
                "signaled": False,
                "error": f"PR/issue number must be an integer, got {number!r}",
            })

        watch_key = f"{repo}#{number_int}"
        triggers = args.get("triggers")

        token = get_github_token()
        if not token:
            return json.dumps({
                "signaled": False,
                "blocked": "auth",
                "reason": "no GITHUB_TOKEN available",
                "watch_key": watch_key,
            })

        try:
            raw_state = await fetch_pr_state(repo, number_int, token=token, kind=kind)
        except PRWatchAuthError as e:
            return json.dumps({
                "signaled": False,
                "blocked": "auth",
                "error": str(e),
                "watch_key": watch_key,
            })
        except PRWatchNetworkError as e:
            return json.dumps({
                "signaled": False,
                "blocked": "network",
                "error": str(e),
                "watch_key": watch_key,
            })

        last_fp, last_normalized = await self._load_pr_watch_state(watch_key)
        decision = evaluate_pr_watch(
            raw_state,
            last_fingerprint=last_fp,
            last_normalized=last_normalized,
            triggers=triggers,
        )
        if not decision.should_signal:
            # Advance the baseline for first observations and filtered/no-op
            # changes. Signal-worthy changes advance only after enqueue
            # succeeds, otherwise a transient dispatcher failure would mark
            # the event handled and permanently drop the wake.
            await self._save_pr_watch_state(
                watch_key, decision.fingerprint, decision.normalized,
            )
            return json.dumps({
                "signaled": False,
                "reason": decision.reason,
                "watch_key": watch_key,
                "changed": sorted(decision.changed),
            })

        target = args.get("notify") or getattr(self.agent, "did", "") or self._agent_id
        signal = build_signal_for_pr_change(
            repo=repo,
            number=number_int,
            decision=decision,
            target_agent=str(target),
            html_url=str(raw_state.get("html_url", "")),
        )

        dispatcher = getattr(self.agent, "dispatcher", None)
        if dispatcher is None or not hasattr(dispatcher, "enqueue_signal"):
            return json.dumps({
                "signaled": False,
                "blocked": "no_dispatcher",
                "reason": "agent has no signal dispatcher",
                "watch_key": watch_key,
                "changed": sorted(decision.matched),
            })

        try:
            enq = dispatcher.enqueue_signal(signal)
            if hasattr(enq, "__await__"):
                await enq
        except Exception as e:
            logger.warning(
                "github_pr_watch: enqueue_signal raised for %s: %s",
                watch_key, e,
            )
            return json.dumps({
                "signaled": False,
                "blocked": "dispatch_error",
                "error": f"{type(e).__name__}: {e}",
                "watch_key": watch_key,
                "changed": sorted(decision.matched),
            })

        await self._save_pr_watch_state(
            watch_key, decision.fingerprint, decision.normalized,
        )

        return json.dumps({
            "signaled": True,
            "reason": decision.reason,
            "watch_key": watch_key,
            "changed": sorted(decision.matched),
        })

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(
        "schedule_list",
        "List all scheduled tasks for this agent",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule list",
    )
    async def schedule_list(self) -> ToolResult:
        """
        List all scheduled tasks for the current agent.
        """
        if not self._db:
            return ToolResult.failed("Database not available")

        try:
            rows = await self._db.fetchall(
                """
                SELECT id, task_name, cron_expression, args_json,
                       enabled, last_run_at, next_run_at, created_at,
                       schedule_kind, run_at, timezone_name, misfire_policy,
                       misfire_grace_seconds, idempotency_key, lease_owner,
                       lease_expires_at, attempt_count, terminal_status, terminal_at
                FROM scheduled_tasks
                WHERE agent_id = ?
                ORDER BY created_at ASC
                """,
                (self._agent_id,),
            )
        except Exception as e:
            logger.error("Failed to list scheduled tasks: %s", e)
            return ToolResult.failed(str(e))

        # Honesty: parse args_json defensively. If a row has malformed
        # JSON (legacy or hand-edited DB), record it as a load_error
        # rather than letting the exception abort the entire list and
        # tank direct in-process callers (e.g. default-schedule setup).
        tasks = []
        load_errors = []
        for row in rows:
            args: Dict[str, Any] = {}
            raw_args = row[3]
            if raw_args:
                try:
                    parsed = json.loads(raw_args)
                    args = parsed if isinstance(parsed, dict) else {}
                    if not isinstance(parsed, dict):
                        load_errors.append({
                            "task_id": row[0],
                            "error": "args_json is not a JSON object",
                        })
                except json.JSONDecodeError as e:
                    load_errors.append({
                        "task_id": row[0],
                        "error": f"args_json malformed: {e}",
                    })
            tasks.append({
                "id": row[0],
                "task_name": row[1],
                "cron_expression": row[2],
                "args": args,
                "enabled": bool(row[4]),
                "last_run_at": row[5],
                "next_run_at": row[6],
                "created_at": row[7],
                # Keep eight-column legacy rows readable during an in-place
                # upgrade (the runner applies the additive migration at boot).
                "schedule_kind": row[8] if len(row) > 8 and row[8] else "cron",
                "run_at": row[9] if len(row) > 9 else None,
                "timezone": row[10] if len(row) > 10 and row[10] else "UTC",
                "misfire_policy": row[11] if len(row) > 11 and row[11] else "skip",
                "misfire_grace_seconds": row[12] if len(row) > 12 else None,
                "idempotency_key": row[13] if len(row) > 13 else None,
                "lease_owner": row[14] if len(row) > 14 else None,
                "lease_expires_at": row[15] if len(row) > 15 else None,
                "attempt_count": row[16] if len(row) > 16 else 0,
                "terminal_status": row[17] if len(row) > 17 else None,
                "terminal_at": row[18] if len(row) > 18 else None,
            })

        data: Dict[str, Any] = {"tasks": tasks, "count": len(tasks)}
        confirmation = f"Listed {len(tasks)} scheduled task(s)"

        if load_errors:
            data["load_errors"] = load_errors
            return ToolResult.partial(
                confirmation=confirmation,
                error=(
                    f"{len(load_errors)} task(s) had unparseable args_json "
                    "and were listed with empty args; check the DB rows for "
                    "ids in load_errors"
                ),
                data=data,
            )

        return ToolResult.ok(confirmation=confirmation, data=data)

    @tool(
        "schedule_add",
        "Add a new scheduled task with a cron expression",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule add",
    )
    async def schedule_add(
        self,
        cron_expression: str,
        task_name: str,
        args_json: str = "{}",
        timezone_name: str = "UTC",
        misfire_policy: str = "skip",
        misfire_grace_seconds: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> ToolResult:
        """
        Add a new scheduled task.

        Args:
            cron_expression: Cron expression (5 fields) or alias like @daily, @hourly
            task_name: Name of a registered scheduler-executable task — a
                built-in cron source (e.g. memory_consolidate, wait_reconcile,
                github_pr_watch) or a loaded feature tool. Unknown names are
                rejected so they can't silently fail every tick (#1618).
            args_json: JSON-encoded arguments to pass to the tool (default: {})
            timezone_name: IANA timezone for local cron matching (default UTC).
                Spring DST gaps are skipped; a fall DST fold runs once at the
                earlier local occurrence.
            misfire_policy: ``skip`` (legacy default), ``fire_once``, or
                ``catch_up``.  ``skip`` honors ``misfire_grace_seconds``;
                ``fire_once`` executes one late occurrence then re-anchors;
                ``catch_up`` advances from the missed occurrence.
            misfire_grace_seconds: Per-schedule override; omit to inherit the
                operator's scheduler default.
            idempotency_key: Stable schedule key.  Omit only for backwards
                compatibility; the scheduler generates and persists one before
                the first occurrence can run.
        """
        try:
            parse(cron_expression)
            get_timezone(timezone_name)
        except CronParseError as e:
            return ToolResult.failed(f"Invalid cron expression or timezone: {e}")

        return await self._create_schedule(
            task_name=task_name,
            args_json=args_json,
            cron_expression=cron_expression,
            next_run_at=None,
            schedule_kind="cron",
            run_at=None,
            timezone_name=timezone_name,
            misfire_policy=misfire_policy,
            misfire_grace_seconds=misfire_grace_seconds,
            idempotency_key=idempotency_key,
        )

    @tool(
        "schedule_add_deadline",
        "Add a one-shot scheduled task that fires at an absolute deadline",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule deadline",
    )
    async def schedule_add_deadline(
        self,
        run_at: str,
        task_name: str,
        args_json: str = "{}",
        misfire_policy: str = "fire_once",
        misfire_grace_seconds: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> ToolResult:
        """Persist a one-shot deadline in UTC and execute it at most once.

        ``run_at`` must include an offset (for example
        ``2026-07-24T14:30:00+00:00``).  The row is disabled and receives a
        terminal status after its claimed occurrence commits.
        """
        try:
            deadline = datetime.fromisoformat(run_at)
        except (TypeError, ValueError):
            return ToolResult.failed("run_at must be an ISO-8601 timestamp with an offset")
        if deadline.tzinfo is None:
            return ToolResult.failed("run_at must include a UTC offset or Z")
        deadline_utc = deadline.astimezone(timezone.utc).isoformat()
        return await self._create_schedule(
            task_name=task_name,
            args_json=args_json,
            cron_expression="",
            next_run_at=deadline_utc,
            schedule_kind="one_shot",
            run_at=deadline_utc,
            timezone_name="UTC",
            misfire_policy=misfire_policy,
            misfire_grace_seconds=misfire_grace_seconds,
            idempotency_key=idempotency_key,
        )

    async def _create_schedule(
        self,
        *,
        task_name: str,
        args_json: str,
        cron_expression: str,
        next_run_at: Optional[str],
        schedule_kind: str,
        run_at: Optional[str],
        timezone_name: str,
        misfire_policy: str,
        misfire_grace_seconds: Optional[int],
        idempotency_key: Optional[str],
    ) -> ToolResult:
        """Validate shared schedule fields and atomically persist a row."""
        if not self._db:
            return ToolResult.failed("Database not available")
        try:
            parsed_args = json.loads(args_json)
        except json.JSONDecodeError as e:
            return ToolResult.failed(f"Invalid args_json: {e}")
        if not isinstance(parsed_args, dict):
            return ToolResult.failed("args_json must be a JSON object")
        if misfire_policy not in {"skip", "fire_once", "catch_up"}:
            return ToolResult.failed(
                "misfire_policy must be one of: skip, fire_once, catch_up"
            )
        if misfire_grace_seconds is not None:
            try:
                misfire_grace_seconds = int(misfire_grace_seconds)
            except (TypeError, ValueError):
                return ToolResult.failed("misfire_grace_seconds must be an integer")
            if misfire_grace_seconds < 0:
                return ToolResult.failed("misfire_grace_seconds must be >= 0")
        if idempotency_key is not None and not str(idempotency_key).strip():
            return ToolResult.failed("idempotency_key must not be empty")

        valid_names = self._scheduler_executable_task_names()
        if task_name not in valid_names:
            return ToolResult.failed(
                f"Unknown scheduled task '{task_name}'. It is not a registered "
                f"scheduler-executable task. Valid task names: {', '.join(sorted(valid_names))}.",
                data={"success": False, "task_name": task_name, "valid_task_names": sorted(valid_names)},
            )
        if await self._scheduled_task_denied(task_name):
            return ToolResult.failed(
                f"Scheduled task '{task_name}' is DENY-listed by security policy and cannot be scheduled.",
                data={"success": False, "task_name": task_name, "denied_by_policy": True},
            )

        task_id = str(uuid.uuid4())
        # Every persisted schedule has a base key, including legacy-style
        # callers that do not supply one.  The runner derives a deterministic
        # occurrence key from this base and exposes it to the target tool.
        base_idempotency = str(idempotency_key).strip() if idempotency_key else f"schedule:{task_id}"
        idempotency_error = validate_schedule_idempotency_base(base_idempotency)
        if idempotency_error is not None:
            return ToolResult.failed(f"idempotency_key {idempotency_error}")
        try:
            # Hold the same durable control row used by rollout activation.
            # ``EXISTS (state = active)`` alone is a snapshot check: a legacy
            # row can force active -> quiescing between that check and this
            # insert.  Updating the active control row inside the schedule
            # transaction serializes the writer with that transition on both
            # PostgreSQL and SQLite, so a newly runnable row is either fenced
            # by the transition or never inserted.
            async with self._schedule_transaction():
                if not await self._lock_active_scheduler_rollout():
                    return self._rollout_mutation_error()
                # The runner evaluates due work against database time. Read
                # that same clock while this schedule transaction owns its
                # rollout control row, so an API replica's skew cannot select
                # the wrong first cron occurrence across a minute boundary.
                schedule_now = await scheduler_database_clock(self._db)
                now_iso = schedule_now.isoformat()
                if schedule_kind == "cron":
                    try:
                        next_run_at = next_run(
                            cron_expression,
                            after=schedule_now,
                            timezone_name=timezone_name,
                        ).isoformat()
                    except CronParseError as e:
                        return ToolResult.failed(f"Cannot compute next run: {e}")
                inserted = await self._db.execute(
                    """
                    INSERT INTO scheduled_tasks
                        (id, agent_id, task_name, cron_expression, args_json, enabled,
                         last_run_at, next_run_at, created_at, schedule_kind, run_at,
                         timezone_name, misfire_policy, misfire_grace_seconds,
                         idempotency_key, attempt_count, scheduler_protocol_version,
                         scheduler_rollout_fenced, scheduler_claim_fenced,
                         scheduler_registration_nonce)
                    VALUES (?, ?, ?, ?, ?, 1, NULL, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, 0, ?)
                    """,
                    (
                        task_id, self._agent_id, task_name, cron_expression, args_json,
                        next_run_at, now_iso, schedule_kind, run_at, timezone_name,
                        misfire_policy, misfire_grace_seconds, base_idempotency,
                        SCHEDULER_PROTOCOL_VERSION,
                        self._pending_scheduler_registration_nonce(),
                    ),
                )
                if not self._scheduler_mutation_wrote(inserted):
                    # A real DB should never return zero after holding the
                    # control-row lock; fail closed instead of claiming a
                    # schedule exists when persistence did not happen.
                    return ToolResult.failed("Failed to persist scheduled task")
        except Exception as e:
            logger.error("Failed to add scheduled task: %s", e)
            return ToolResult.failed(str(e))

        description = f"deadline={run_at}" if schedule_kind == "one_shot" else f"cron={cron_expression} timezone={timezone_name}"
        logger.info("Scheduled task added: %s (%s) %s next=%s", task_id, task_name, description, next_run_at)
        return ToolResult.ok(
            confirmation=f"Scheduled '{task_name}' (id={task_id[:8]}) {description} next={next_run_at}",
            data={
                "success": True, "task_id": task_id, "task_name": task_name,
                "cron_expression": cron_expression or None, "run_at": run_at,
                "schedule_kind": schedule_kind, "timezone": timezone_name,
                "misfire_policy": misfire_policy,
                "misfire_grace_seconds": misfire_grace_seconds,
                "idempotency_key": base_idempotency,
                "next_run_at": next_run_at, "created_at": now_iso,
            },
        )

    @tool(
        "schedule_remove",
        "Remove a scheduled task by ID",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule remove",
    )
    async def schedule_remove(self, task_id: str) -> ToolResult:
        """
        Remove a scheduled task.

        Args:
            task_id: The UUID of the task to remove
        """
        if not self._db:
            return ToolResult.failed("Database not available")

        try:
            # Keep invalidating the claim and terminalizing its log in one
            # transaction. Mutate the schedule row first, matching runner
            # finalization and the pause/update paths, before acquiring the
            # execution-log row. A crash between those writes must not strand
            # an impossible-to-recover ``claimed`` record.
            async with self._schedule_transaction():
                if not await self._lock_scheduler_rollout_for_pause():
                    return self._rollout_mutation_error()
                if not await self._lock_mutable_scheduler_row(task_id):
                    return ToolResult.failed(
                        f"Task {task_id} not found",
                        data={"task_id": task_id},
                    )
                row = await self._db.fetchone(
                    "SELECT id FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                    (task_id, self._agent_id),
                )
                if not row:
                    return ToolResult.failed(
                        f"Task {task_id} not found",
                        data={"task_id": task_id},
                    )
                removed = await self._db.execute(
                    "DELETE FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                    (task_id, self._agent_id),
                )
                if not self._scheduler_mutation_wrote(removed):
                    return ToolResult.failed("Failed to remove scheduled task")
                await self._cancel_claimed_executions(
                    task_id, "schedule removed before outcome commit"
                )
        except Exception as e:
            logger.error("Failed to remove task %s: %s", task_id, e)
            return ToolResult.failed(str(e))

        logger.info("Scheduled task removed: %s", task_id)
        return ToolResult.ok(
            confirmation=f"Removed scheduled task {task_id}",
            data={"success": True, "task_id": task_id, "status": "removed"},
        )

    @tool(
        "schedule_pause",
        "Pause a scheduled task (stops it from running until resumed)",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule pause",
    )
    async def schedule_pause(self, task_id: str) -> ToolResult:
        """
        Pause a scheduled task.

        Args:
            task_id: The UUID of the task to pause
        """
        if not self._db:
            return ToolResult.failed("Database not available")

        try:
            async with self._schedule_transaction():
                # Unlike create/resume/update, pausing is safe and necessary
                # during a quiescing rollout. Lock whichever v2 control state
                # currently owns the DID before deciding whether an enabled=0
                # row is an intentional pause or a rollout fence.
                if not await self._lock_scheduler_rollout_for_pause():
                    return self._rollout_mutation_error()
                # A legacy row may be paused only after this transaction has
                # locked the current quiescing control epoch and verified that
                # the row belongs to its exact rollout fence. It stays
                # otherwise immutable: remove/resume/update use the ordinary
                # v2-only target lock.
                if not await self._lock_mutable_scheduler_row(
                    task_id,
                    pause_legacy_rollout_nonce=(
                        await self._locked_quiescing_scheduler_rollout_nonce()
                    ),
                ):
                    return ToolResult.failed(
                        f"Task {task_id} not found",
                        data={"task_id": task_id},
                    )
                row = await self._db.fetchone(
                    """
                    SELECT id, enabled, scheduler_claim_fenced,
                           scheduler_rollout_fenced
                    FROM scheduled_tasks WHERE id = ? AND agent_id = ?
                    """,
                    (task_id, self._agent_id),
                )
                if not row:
                    return ToolResult.failed(
                        f"Task {task_id} not found",
                        data={"task_id": task_id},
                    )

                claim_fenced = bool(row[2]) if len(row) > 2 else False
                rollout_fenced = bool(row[3]) if len(row) > 3 else False
                if not row[1] and not claim_fenced and not rollout_fenced:
                    return ToolResult.ok(
                        confirmation=f"Task {task_id} was already paused (no-op)",
                        data={"success": True, "task_id": task_id, "status": "already_paused"},
                    )

                paused = await self._db.execute(
                    """
                    UPDATE scheduled_tasks
                    SET enabled = 0, lease_owner = NULL, lease_expires_at = NULL,
                        claim_token = NULL, claim_execution_id = NULL,
                        claim_scheduled_for = NULL, scheduler_claim_fenced = 0,
                        scheduler_rollout_fenced = 0,
                        scheduler_rollout_fenced_at = NULL
                    WHERE id = ? AND agent_id = ?
                    """,
                    (task_id, self._agent_id),
                )
                if not self._scheduler_mutation_wrote(paused):
                    return ToolResult.failed("Failed to pause scheduled task")
                await self._cancel_claimed_executions(
                    task_id, "schedule paused before outcome commit"
                )
        except Exception as e:
            logger.error("Failed to pause task %s: %s", task_id, e)
            return ToolResult.failed(str(e))

        logger.info("Scheduled task paused: %s", task_id)
        return ToolResult.ok(
            confirmation=f"Paused scheduled task {task_id}",
            data={"success": True, "task_id": task_id, "status": "paused"},
        )

    @tool(
        "schedule_resume",
        "Resume a paused scheduled task",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule resume",
    )
    async def schedule_resume(self, task_id: str) -> ToolResult:
        """
        Resume a paused scheduled task.

        Args:
            task_id: The UUID of the task to resume
        """
        if not self._db:
            return ToolResult.failed("Database not available")

        try:
            # The state decision belongs under the active control-row lock.
            # Otherwise a concurrent pause can land between this read and the
            # resume write, and be silently undone by stale enabled=0 data.
            async with self._schedule_transaction():
                if not await self._lock_active_scheduler_rollout():
                    return self._rollout_mutation_error()
                if not await self._lock_mutable_scheduler_row(task_id):
                    return ToolResult.failed(
                        f"Task {task_id} not found",
                        data={"task_id": task_id},
                    )
                row = await self._db.fetchone(
                    """
                    SELECT id, enabled, cron_expression, schedule_kind, run_at,
                           timezone_name, terminal_status, scheduler_claim_fenced,
                           scheduler_rollout_fenced
                    FROM scheduled_tasks WHERE id = ? AND agent_id = ?
                    """,
                    (task_id, self._agent_id),
                )
                if not row:
                    return ToolResult.failed(
                        f"Task {task_id} not found",
                        data={"task_id": task_id},
                    )

                claim_fenced = bool(row[7]) if len(row) > 7 else False
                rollout_fenced = bool(row[8]) if len(row) > 8 else False
                if rollout_fenced:
                    return self._rollout_mutation_error()
                if row[1] or claim_fenced:
                    return ToolResult.ok(
                        confirmation=f"Task {task_id} was already running (no-op)",
                        data={"success": True, "task_id": task_id, "status": "already_running"},
                    )

                cron_expr = row[2]
                schedule_kind = row[3] if len(row) > 3 and row[3] else "cron"
                run_at = row[4] if len(row) > 4 else None
                timezone_name = row[5] if len(row) > 5 and row[5] else "UTC"
                terminal_status = row[6] if len(row) > 6 else None
                if schedule_kind == "one_shot" and terminal_status:
                    return ToolResult.failed(
                        f"Task {task_id} is a terminal one-shot deadline ({terminal_status}) and cannot be resumed"
                    )
                cron_now_invalid = False
                if schedule_kind == "one_shot":
                    next_run_at = run_at
                    if not next_run_at:
                        return ToolResult.failed(f"One-shot task {task_id} has no run_at deadline")
                else:
                    try:
                        schedule_now = await scheduler_database_clock(self._db)
                        nxt = next_run(
                            cron_expr,
                            after=schedule_now,
                            timezone_name=timezone_name,
                        )
                        next_run_at = nxt.isoformat()
                    except CronParseError:
                        next_run_at = None
                        cron_now_invalid = True

                resumed = await self._db.execute(
                    """
                    UPDATE scheduled_tasks
                    SET enabled = 1, next_run_at = ?, terminal_status = NULL,
                        terminal_at = NULL, lease_owner = NULL, lease_expires_at = NULL,
                        claim_token = NULL, claim_execution_id = NULL,
                        claim_scheduled_for = NULL, scheduler_claim_fenced = 0,
                        scheduler_rollout_fenced = 0,
                        scheduler_rollout_fenced_at = NULL,
                        scheduler_protocol_version = ?
                    WHERE id = ? AND agent_id = ?
                    """,
                    (
                        next_run_at, SCHEDULER_PROTOCOL_VERSION, task_id,
                        self._agent_id,
                    ),
                )
                if not self._scheduler_mutation_wrote(resumed):
                    return ToolResult.failed("Failed to resume scheduled task")
        except Exception as e:
            logger.error("Failed to resume task %s: %s", task_id, e)
            return ToolResult.failed(str(e))

        logger.info("Scheduled task resumed: %s (next_run=%s)", task_id, next_run_at)
        data = {
            "success": True,
            "task_id": task_id,
            "status": "resumed",
            "next_run_at": next_run_at,
        }

        # Honesty: a resumed task whose cron expression no longer parses
        # is enabled in the DB but has next_run_at=None — the runner will
        # never fire it. The agent must speak that the task won't actually
        # run, not just claim it was resumed.
        if cron_now_invalid:
            return ToolResult.partial(
                confirmation=f"Resumed task {task_id} (enabled flag set)",
                error=(
                    f"cron expression {cron_expr!r} no longer parses; "
                    "next_run_at is null and the runner will never fire "
                    "this task. Use schedule_update to set a valid cron."
                ),
                data=data,
            )

        return ToolResult.ok(
            confirmation=f"Resumed scheduled task {task_id} (next={next_run_at})",
            data=data,
        )

    @tool(
        "schedule_update",
        "Update the cron expression of an existing scheduled task",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule update",
    )
    async def schedule_update(
        self,
        task_id: str,
        cron_expression: str,
        timezone_name: Optional[str] = None,
    ) -> ToolResult:
        """
        Update the cron expression on an existing scheduled task and recompute
        its next_run_at. Used by dynamic/self-adjusting schedulers (e.g. the
        reflection loop) to change task cadence without removing and re-adding.

        Args:
            task_id: The UUID of the task to update
            cron_expression: New cron expression (5 fields or alias)
            timezone_name: Optional replacement IANA timezone.  Omit to retain
                the schedule's existing zone.
        """
        if not self._db:
            return ToolResult.failed("Database not available")

        try:
            parse(cron_expression)
            if timezone_name is not None:
                get_timezone(timezone_name)
        except CronParseError as e:
            return ToolResult.failed(f"Invalid cron expression: {e}")

        try:
            # Read the row only after the same control-row lock that guards
            # the write. Reading enabled outside this transaction let a
            # concurrent pause/resume be overwritten by a stale update.
            async with self._schedule_transaction():
                if not await self._lock_active_scheduler_rollout():
                    return self._rollout_mutation_error()
                if not await self._lock_mutable_scheduler_row(task_id):
                    return ToolResult.failed(
                        f"Task {task_id} not found",
                        data={"task_id": task_id},
                    )
                row = await self._db.fetchone(
                    """
                    SELECT cron_expression, enabled, schedule_kind, timezone_name,
                           scheduler_claim_fenced, scheduler_rollout_fenced
                    FROM scheduled_tasks WHERE id = ? AND agent_id = ?
                    """,
                    (task_id, self._agent_id),
                )
                if not row:
                    return ToolResult.failed(
                        f"Task {task_id} not found",
                        data={"task_id": task_id},
                    )

                old_cron = row[0]
                claim_fenced = bool(row[4]) if len(row) > 4 else False
                rollout_fenced = bool(row[5]) if len(row) > 5 else False
                if rollout_fenced:
                    # An active control row plus a fenced row is corrupt or a
                    # concurrent transition; never clear its safety fence from
                    # a stale definition update.
                    return self._rollout_mutation_error()
                enabled = bool(row[1]) or claim_fenced
                schedule_kind = row[2] if len(row) > 2 and row[2] else "cron"
                current_timezone = row[3] if len(row) > 3 and row[3] else "UTC"
                if schedule_kind != "cron":
                    return ToolResult.failed(
                        "One-shot deadlines do not have a cron expression to update"
                    )
                effective_timezone = timezone_name or current_timezone

                if old_cron == cron_expression and effective_timezone == current_timezone:
                    return ToolResult.ok(
                        confirmation=(
                            f"Task {task_id} cron unchanged ({cron_expression}); no-op"
                        ),
                        data={
                            "success": True,
                            "task_id": task_id,
                            "status": "unchanged",
                            "cron_expression": cron_expression,
                        },
                    )

                next_run_at: Optional[str] = None
                if enabled:
                    try:
                        schedule_now = await scheduler_database_clock(self._db)
                        next_run_at = next_run(
                            cron_expression,
                            after=schedule_now,
                            timezone_name=effective_timezone,
                        ).isoformat()
                    except CronParseError as e:
                        return ToolResult.failed(f"Cannot compute next run: {e}")

                updated = await self._db.execute(
                    """
                    UPDATE scheduled_tasks
                    SET cron_expression = ?, timezone_name = ?, next_run_at = ?,
                        enabled = ?, scheduler_protocol_version = ?,
                        lease_owner = NULL, lease_expires_at = NULL,
                        claim_token = NULL, claim_execution_id = NULL,
                        claim_scheduled_for = NULL, scheduler_claim_fenced = 0,
                        scheduler_rollout_fenced = 0,
                        scheduler_rollout_fenced_at = NULL
                    WHERE id = ? AND agent_id = ?
                    """,
                    (
                        cron_expression, effective_timezone, next_run_at,
                        1 if enabled else 0, SCHEDULER_PROTOCOL_VERSION,
                        task_id, self._agent_id,
                    ),
                )
                if not self._scheduler_mutation_wrote(updated):
                    return ToolResult.failed("Failed to update scheduled task")
                await self._cancel_claimed_executions(
                    task_id,
                    "schedule definition updated before outcome commit",
                    status="superseded",
                )
        except Exception as e:
            logger.error("Failed to update task %s: %s", task_id, e)
            return ToolResult.failed(str(e))

        logger.info(
            "Scheduled task updated: %s cron: %s -> %s (next=%s)",
            task_id, old_cron, cron_expression, next_run_at,
        )
        return ToolResult.ok(
            confirmation=(
                f"Updated task {task_id} cron: {old_cron} -> {cron_expression} "
                f"(next={next_run_at})"
            ),
            data={
                "success": True,
                "task_id": task_id,
                "status": "updated",
                "old_cron": old_cron,
                "cron_expression": cron_expression,
                "timezone": effective_timezone,
                "next_run_at": next_run_at,
            },
        )

    async def _cancel_claimed_executions(
        self,
        task_id: str,
        reason: str,
        *,
        status: str = "cancelled",
    ) -> None:
        """Terminalize claims invalidated by an operator schedule mutation.

        The scheduler runner still owns the CAS against its lease token.  This
        helper only closes logs after the mutation has cleared that token, so
        a stale worker cannot later overwrite the operator's intent.
        """
        if self._database_backend_type() in {"postgres", "sqlite"}:
            # The caller already owns the schedule transaction. Evaluate the
            # completion timestamp in the terminal UPDATE itself, matching
            # runner finalization so a blocked/clock-skewed API replica cannot
            # write an audit time that precedes its durable mutation.
            completed_at_sql = scheduler_database_now_sql(self._db)
            await self._db.execute(
                f"""
                UPDATE task_execution_log
                SET status = ?, result_text = ?, completed_at = {completed_at_sql}
                WHERE task_id = ? AND agent_id = ? AND status = 'claimed'
                """,
                (status, reason, task_id, self._agent_id),
            )
            return

        # Preserve the historic lightweight-adapter contract where there is
        # no durable database clock to query or express in SQL.
        await self._db.execute(
            """
            UPDATE task_execution_log
            SET status = ?, result_text = ?, completed_at = ?
            WHERE task_id = ? AND agent_id = ? AND status = 'claimed'
            """,
            (status, reason, datetime.now(timezone.utc).isoformat(), task_id, self._agent_id),
        )

    @asynccontextmanager
    async def _schedule_transaction(self):
        """Use a DB transaction while retaining lightweight legacy test DBs."""
        transaction = getattr(self._db, "transaction", None)
        if not callable(transaction):
            yield
            return
        context = transaction()
        if not hasattr(context, "__aenter__"):
            yield
            return
        async with context:
            yield

    def _database_backend_type(self) -> str:
        """Return the concrete scheduler database type."""

        backend_type = getattr(self._db, "backend_type", "")
        return backend_type.lower() if isinstance(backend_type, str) else ""

    async def _lock_mutable_scheduler_row(
        self,
        task_id: str,
        *,
        pause_legacy_rollout_nonce: Optional[str] = None,
    ) -> Optional[bool]:
        """Lock, version-check, and adopt a schedule before mutating it.

        Global and control-row compatibility alone is insufficient: a newer
        writer can create an individual future-version schedule after this
        feature starts.  Every API that changes that schedule must inspect its
        version under the same transaction before overwriting fences or claim
        state.  The locked row is also the authoritative place to adopt a
        foreign dynamic-registration marker.  ``schedule_pause`` alone may
        supply a locked quiescing activation nonce to turn a legacy row that
        is fenced by that exact epoch into an intentional durable pause.
        """

        backend_type = self._database_backend_type()
        if backend_type not in {"postgres", "sqlite"}:
            # Keep deliberately minimal legacy test adapters usable. Concrete
            # scheduler backends always take the durable path below.
            return True
        sql = """
            SELECT scheduler_protocol_version, scheduler_registration_nonce,
                   scheduler_rollout_fenced, scheduler_rollout_nonce
            FROM scheduled_tasks
            WHERE id = ? AND agent_id = ?
        """
        if backend_type == "postgres":
            sql += " FOR UPDATE"
        row = await self._db.fetchone(sql, (task_id, self._agent_id))
        if row is None:
            return None
        version = row[0] if row else None
        if version is None:
            protocol_version = None
        else:
            try:
                protocol_version = int(version)
            except (TypeError, ValueError):
                return False
        if (
            protocol_version is not None
            and protocol_version > SCHEDULER_PROTOCOL_VERSION
        ):
            raise SchedulerProtocolVersionIncompatible()
        if protocol_version != SCHEDULER_PROTOCOL_VERSION:
            # A NULL/older marker is evidence of a legacy writer. Do not
            # stamp it as v2 (or let removal erase it) before reconciliation
            # can fence that writer and require an explicit rollout ACK. The
            # sole exception is pause: after the control row and this target
            # have both been locked in this transaction, a current rollout
            # fence may be converted into an intentional durable pause. A
            # stale or absent nonce remains immutable.
            if (
                pause_legacy_rollout_nonce is not None
                and bool(row[2])
                and row[3] == pause_legacy_rollout_nonce
            ):
                return True
            return False
        return await adopt_scheduler_registration_ownership(
            self._db,
            task_id=task_id,
            agent_id=self._agent_id,
            observed_registration_nonce=row[1] if len(row) > 1 else None,
            pending_registration_nonce=self._pending_scheduler_registration_nonce(),
        )

    async def _locked_quiescing_scheduler_rollout_nonce(self) -> Optional[str]:
        """Return the locked quiescing epoch that can authorize legacy pause.

        :meth:`_lock_scheduler_rollout_for_pause` has already taken the
        control-row lock in the enclosing schedule transaction. Re-reading it
        with ``FOR UPDATE`` on PostgreSQL makes the dependency explicit at the
        point where the target-row lock is evaluated; SQLite's write
        transaction provides the corresponding serialization. Only a nonempty
        activation nonce from the current v2 quiescing control row can relax
        the legacy-row rejection for ``schedule_pause``.
        """

        backend_type = self._database_backend_type()
        if backend_type not in {"postgres", "sqlite"}:
            return None

        sql = """
            SELECT protocol_version, state, activation_nonce
            FROM scheduler_protocol_rollout
            WHERE agent_id = ?
        """
        if backend_type == "postgres":
            sql += " FOR UPDATE"
        row = await self._db.fetchone(sql, (self._agent_id,))
        if row is None or len(row) < 3:
            return None
        try:
            protocol_version = int(row[0])
        except (TypeError, ValueError):
            return None
        nonce = row[2]
        if (
            protocol_version == SCHEDULER_PROTOCOL_VERSION
            and row[1] == SCHEDULER_ROLLOUT_STATE_QUIESCING
            and isinstance(nonce, str)
            and nonce
        ):
            return nonce
        return None

    async def _lock_compatible_scheduler_protocol(
        self,
        *,
        allowed_states: set[str],
    ) -> bool:
        """Lock and validate global plus DID protocol state before mutation.

        PostgreSQL row locks linearize a live future-version transition with
        the schedule write. SQLite first executes a predicate-protected no-op
        write to acquire its transaction writer slot; a future global row
        matches nothing and is inspected without being changed. Callers must
        already own :meth:`_schedule_transaction`.
        """

        backend_type = self._database_backend_type()
        if backend_type not in {"postgres", "sqlite"}:
            # Retain the historical adapter contract for deliberately minimal
            # unit/integration doubles. Concrete persistent backends always
            # take the global+tenant compatibility path below.
            state_sql = ", ".join(
                f"'{state}'" for state in sorted(allowed_states)
            )
            locked = await self._db.execute(
                f"""
                UPDATE scheduler_protocol_rollout
                SET updated_at = updated_at
                WHERE agent_id = ? AND protocol_version = ?
                  AND state IN ({state_sql})
                """,
                (self._agent_id, SCHEDULER_PROTOCOL_VERSION),
            )
            return self._scheduler_mutation_wrote(locked)

        if backend_type == "postgres":
            schema_row = await self._db.fetchone(
                """
                SELECT protocol_version
                FROM scheduler_protocol_schema
                WHERE singleton = 1
                FOR UPDATE
                """
            )
        else:
            if backend_type == "sqlite":
                await self._db.execute(
                    """
                    UPDATE scheduler_protocol_schema
                    SET protocol_version = protocol_version
                    WHERE singleton = 1 AND protocol_version <= ?
                    """,
                    (SCHEDULER_PROTOCOL_VERSION,),
                )
            schema_row = await self._db.fetchone(
                """
                SELECT protocol_version
                FROM scheduler_protocol_schema
                WHERE singleton = 1
                """
            )

        if schema_row is None or not schema_row:
            return False
        try:
            schema_version = int(schema_row[0])
        except (TypeError, ValueError):
            return False
        if schema_version > SCHEDULER_PROTOCOL_VERSION:
            raise SchedulerProtocolVersionIncompatible()
        if schema_version != SCHEDULER_PROTOCOL_VERSION:
            return False

        rollout_sql = """
            SELECT protocol_version, state
            FROM scheduler_protocol_rollout
            WHERE agent_id = ?
        """
        if backend_type == "postgres":
            rollout_sql += " FOR UPDATE"
        rollout_row = await self._db.fetchone(
            rollout_sql,
            (self._agent_id,),
        )
        if rollout_row is None or len(rollout_row) < 2:
            return False
        try:
            rollout_version = int(rollout_row[0])
        except (TypeError, ValueError):
            return False
        if rollout_version > SCHEDULER_PROTOCOL_VERSION:
            raise SchedulerProtocolVersionIncompatible()
        compatible = (
            rollout_version == SCHEDULER_PROTOCOL_VERSION
            and rollout_row[1] in allowed_states
        )
        if not compatible:
            return False

        # A pending registration preserves its own ownership marker while it
        # seeds schedules. Any other feature instance that mutates this DID
        # adopts the control row instead, preventing the first registration's
        # rollback from deleting state a different replica now relies on.
        pending_nonce = self._pending_scheduler_registration_nonce()
        await self._db.execute(
            """
            UPDATE scheduler_protocol_rollout
            SET scheduler_registration_nonce = CASE
                WHEN scheduler_registration_nonce = ?
                THEN scheduler_registration_nonce
                ELSE NULL
            END
            WHERE agent_id = ? AND protocol_version = ?
              AND state IN ({states})
            """.format(states=", ".join("?" for _ in sorted(allowed_states))),
            (
                pending_nonce,
                self._agent_id,
                SCHEDULER_PROTOCOL_VERSION,
                *sorted(allowed_states),
            ),
        )
        return True

    async def _lock_active_scheduler_rollout(self) -> bool:
        """Lock this DID's active rollout epoch for a runnable mutation.

        The no-op assignment is intentional.  PostgreSQL takes a row lock,
        SQLite serializes the write transaction, and the predicate rejects a
        DID that is quiescing or has no v2 protocol control row.  This must be
        called *inside* :meth:`_schedule_transaction` before adding, resuming,
        or otherwise making a schedule runnable.
        """

        return await self._lock_compatible_scheduler_protocol(
            allowed_states={"active"},
        )

    async def _lock_scheduler_rollout_for_pause(self) -> bool:
        """Serialize a safe pause with active *or* quiescing rollout state.

        Pausing never makes work runnable, so it remains authorized during a
        rollout drain. The no-op control-row update makes activation wait for
        the pause's durable intent; clearing ``scheduler_rollout_fenced`` then
        prevents activation from restoring that row to enabled=1.
        """

        return await self._lock_compatible_scheduler_protocol(
            allowed_states={"active", "quiescing"},
        )

    @tool(
        "schedule_record_outcome",
        "Attach an engagement signal (0.0-1.0) to a past task execution",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule outcome",
    )
    async def schedule_record_outcome(
        self,
        execution_id: str,
        signal: float,
    ) -> ToolResult:
        """
        Record an engagement signal on a past execution. Used when the
        downstream outcome is only known later — e.g. a morning_signal
        dispatched at 8am that the user replied to at 9:15am scores 1.0
        when the reply arrives.

        Signal is clamped to [0.0, 1.0].

        Args:
            execution_id: The id returned by schedule_history
            signal: Engagement score in [0.0, 1.0]
        """
        if not self._db:
            return ToolResult.failed("Database not available")

        try:
            signal_val = float(signal)
        except (TypeError, ValueError):
            return ToolResult.failed(f"signal must be numeric, got {signal!r}")

        clamped = max(0.0, min(1.0, signal_val))
        was_clamped = clamped != signal_val

        try:
            async with self._schedule_transaction():
                if not await self._lock_scheduler_rollout_for_pause():
                    return self._rollout_mutation_error()
                row = await self._db.fetchone(
                    """
                    SELECT id, status, task_id
                    FROM task_execution_log WHERE id = ? AND agent_id = ?
                    """,
                    (execution_id, self._agent_id),
                )
                if not row:
                    return ToolResult.failed(
                        f"Execution {execution_id} not found",
                        data={"execution_id": execution_id},
                    )
                if len(row) > 1 and row[1] == "claimed":
                    return ToolResult.failed(
                        "Cannot record an outcome for a claimed execution; "
                        "the scheduler still owns its finalization",
                        data={"execution_id": execution_id},
                    )
                # Retain historic outcome recording for a terminal log whose
                # schedule was intentionally removed, while protecting any
                # extant future-version task that owns this log.
                if len(row) > 2 and isinstance(row[2], str):
                    compatible = await self._lock_mutable_scheduler_row(row[2])
                    if compatible is False:
                        return ToolResult.failed(
                            f"Task {row[2]} not found",
                            data={"task_id": row[2]},
                        )

                updated = await self._db.execute(
                    """
                    UPDATE task_execution_log SET outcome_signal = ?
                    WHERE id = ? AND agent_id = ? AND status <> 'claimed'
                    """,
                    (clamped, execution_id, self._agent_id),
                )
                if not self._scheduler_mutation_wrote(updated):
                    return ToolResult.failed("Failed to record scheduler outcome")
        except Exception as e:
            logger.error("Failed to record outcome for %s: %s", execution_id, e)
            return ToolResult.failed(str(e))

        data = {
            "success": True,
            "execution_id": execution_id,
            "signal": clamped,
            "signal_requested": signal_val,
            "signal_clamped": was_clamped,
        }

        # Honesty: silently clamping a 1.5 to 1.0 hides the over-range
        # input from the LLM. Surface as PARTIAL so the agent must
        # speak the clamping (same pattern as save_fact in #1091).
        if was_clamped:
            return ToolResult.partial(
                confirmation=(
                    f"Recorded outcome for {execution_id} "
                    f"(signal={clamped:.2f})"
                ),
                error=(
                    f"requested signal={signal_val} was outside [0.0, 1.0]; "
                    f"clamped to {clamped:.2f}"
                ),
                data=data,
            )
        return ToolResult.ok(
            confirmation=(
                f"Recorded outcome for {execution_id} (signal={clamped:.2f})"
            ),
            data=data,
        )

    @tool(
        "schedule_engagement",
        "Report aggregate engagement scores per scheduled task",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule engagement",
    )
    async def schedule_engagement(self, days: int = 7) -> ToolResult:
        """
        Aggregate recent outcome signals per task.

        Returns one row per task with execution count, signal count (non-null),
        and mean signal over the window. A task with many executions but few
        signals indicates the downstream integration isn't reporting back —
        useful for diagnosing dead loops.

        Args:
            days: Look-back window in days (1-365, default: 7)
        """
        try:
            days = int(days)
        except (TypeError, ValueError):
            return ToolResult.failed(f"days must be an integer, got {days!r}")
        if days < 1 or days > 365:
            return ToolResult.failed("days must be in [1, 365]")

        if not self._db:
            return ToolResult.failed("Database not available")

        try:
            from datetime import timedelta
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            rows = await self._db.fetchall(
                """
                SELECT
                    st.id, st.task_name, st.cron_expression,
                    COUNT(el.id) AS executions,
                    COUNT(el.outcome_signal) AS signals,
                    AVG(el.outcome_signal) AS mean_signal
                FROM scheduled_tasks st
                LEFT JOIN task_execution_log el
                    ON el.task_id = st.id AND el.executed_at >= ?
                WHERE st.agent_id = ?
                GROUP BY st.id, st.task_name, st.cron_expression
                ORDER BY st.task_name
                """,
                (since, self._agent_id),
            )
        except Exception as e:
            logger.error("Failed to aggregate engagement: %s", e)
            return ToolResult.failed(str(e))

        tasks = [
            {
                "task_id": row[0],
                "task_name": row[1],
                "cron_expression": row[2],
                "executions": row[3] or 0,
                "signals": row[4] or 0,
                "mean_signal": float(row[5]) if row[5] is not None else None,
            }
            for row in rows
        ]
        return ToolResult.ok(
            confirmation=(
                f"Engagement over last {days} day(s): {len(tasks)} task(s)"
            ),
            data={"window_days": days, "tasks": tasks, "count": len(tasks)},
        )

    @tool(
        "schedule_history",
        "Show recent task execution history",
        category=ToolCategory.UTILITY,
        command_prefix="!schedule history",
    )
    async def schedule_history(self, limit: int = 20) -> ToolResult:
        """
        Show recent task execution history.

        Args:
            limit: Maximum number of execution records to return (default: 20)
        """
        if not self._db:
            return ToolResult.failed("Database not available")

        try:
            rows = await self._db.fetchall(
                """
                SELECT el.id, el.task_id, el.status, el.result_text,
                       el.duration_ms, el.executed_at, st.task_name, el.outcome_signal
                FROM task_execution_log el
                LEFT JOIN scheduled_tasks st ON st.id = el.task_id
                WHERE el.agent_id = ?
                ORDER BY el.executed_at DESC
                LIMIT ?
                """,
                (self._agent_id, limit),
            )
        except Exception as e:
            logger.error("Failed to get execution history: %s", e)
            return ToolResult.failed(str(e))

        records = []
        for row in rows:
            records.append({
                "id": row[0],
                "task_id": row[1],
                "status": row[2],
                "result_text": row[3],
                "duration_ms": row[4],
                "executed_at": row[5],
                "task_name": row[6],
                "outcome_signal": row[7],
            })

        return ToolResult.ok(
            confirmation=f"Found {len(records)} execution record(s)",
            data={"executions": records, "count": len(records)},
        )
