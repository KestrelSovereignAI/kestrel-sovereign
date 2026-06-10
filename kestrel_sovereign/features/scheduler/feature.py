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
    reflect               -- run a reflection workflow (ReflectionFeature)
    memory_consolidate    -- consolidate short-term memory into episodes
    talon_monitor         -- poll Talon jobs, wake on completion (#1510)
    restart_coordinator   -- execute pending restart requests (#1512)
    github_pr_watch       -- poll a GitHub PR/issue, wake on relevant
                             state/comment/check/merge changes (#1618)

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
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.scheduler.cron import CronParseError, next_run, parse
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.features.storage_access import resolve_feature_database

logger = logging.getLogger(__name__)


class SchedulerFeature(Feature):
    """
    Cron/scheduler system for running agent tasks on a schedule.

    On initialize(), creates DB tables and starts a background runner that
    polls for due tasks every 30 seconds. Each due task is executed by
    invoking the registered callback (typically a feature tool).
    """

    @property
    def tool_description(self) -> str:
        return (
            "Manage scheduled tasks - add, remove, pause, resume cron-based "
            "tasks for memory consolidation, wellness checks, audit anchoring, "
            "and custom operations"
        )

    async def initialize(self):
        """Initialize the scheduler: set up DB refs, register cron sources
        with the SignalDispatcher (Phase 4 of #889), and start the
        background runner."""
        self._db = None
        self._agent_id = ""
        self._runner: Optional[SchedulerRunner] = None

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
                },
            )
            for reg in cron_registrations:
                # Idempotent against re-init: skip if already registered
                # (the registry rejects duplicates, but a feature reload
                # in tests shouldn't blow up).
                if reg.name not in registry:
                    registry.register(reg)

            # github_pr_watch (#1618) is an ACTION cron task that, on a
            # relevant change, enqueues a COGNITION github.pr_activity
            # signal. Register that downstream source here so the wake
            # has a prompt template and policy. The watcher is built-in
            # (no GitHub feature required), so the source lives with the
            # scheduler rather than an external package.
            from kestrel_sovereign.signals.sources.github_pr_watch import (
                SOURCE_NAME as _PR_ACTIVITY_SOURCE,
                build_github_pr_activity_registration,
            )

            if _PR_ACTIVITY_SOURCE not in registry:
                registry.register(build_github_pr_activity_registration())
        else:
            logger.warning(
                "SchedulerFeature: no signal_registry on agent, "
                "cron tasks will not be dispatched as signals"
            )

        # Start background runner. The new executor builds a Signal
        # envelope per task and routes through the dispatcher; cron
        # config and task_execution_log shape are unchanged.
        self._runner = SchedulerRunner(
            db=self._db,
            agent_id=self._agent_id,
            executor=self._dispatch_scheduled_task,
            misfire_grace_seconds=self._load_misfire_grace_seconds(),
        )
        await self._runner.start()
        logger.info("SchedulerFeature initialized")

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

    async def post_all_features_loaded(self, agent):
        """Register default scheduled tasks after all features are loaded.

        Idempotent — checks for existing tasks before adding.
        Only schedules reflection/training if ReflectionFeature is available.
        """
        if not self._db:
            return

        # Check what's already scheduled
        existing = await self.schedule_list()
        existing_tasks = (existing.data or {}).get("tasks", []) if existing.data else []
        existing_names = {t["task_name"] for t in existing_tasks}

        # Reflection-dependent schedules only if ReflectionFeature is loaded
        has_reflection = "ReflectionFeature" in agent.features

        from kestrel_sovereign.storage.retention import DEFAULT_RETENTION_CRON

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

        # Memory consolidation only if MemoryFeature is loaded (it owns the tool)
        if "MemoryFeature" in agent.features:
            defaults.append(
                ("memory_consolidate", "0 4 * * *", "{}"),  # nightly at 4am
            )

        if has_reflection:
            defaults.extend([
                ("reflect", "0 */4 * * *", '{"scope":"all","depth":"normal"}'),
                ("training_cycle", "0 3 * * *", '{"iterations":3,"depth":"normal"}'),
            ])

        # Talon-job monitor (#1510) — installs only if the Talon
        # coordinator feature is loaded. Cheap (no LLM, just file/PID
        # checks) so 1/min cadence is OK. Emits one
        # talon.job_complete COGNITION signal per state transition;
        # the dispatcher coalesces if two adjacent polls double-fire.
        if "TalonCoordinatorFeature" in agent.features:
            defaults.append(("talon_monitor", "* * * * *", "{}"))

        # Restart coordinator (#1512) — installs only if the
        # RestartCoordinatorFeature is loaded. Cheap (no LLM; idle
        # unless a request is pending) so 1/min is fine. Emits one
        # restart.completed signal per row after the agent boots
        # back up post-restart.
        if "RestartCoordinatorFeature" in agent.features:
            defaults.append(("restart_coordinator", "* * * * *", "{}"))

        for task_name, cron, args in defaults:
            if task_name in existing_names:
                logger.debug("Schedule '%s' already exists, skipping", task_name)
                continue
            result = await self.schedule_add(
                cron_expression=cron, task_name=task_name, args_json=args,
            )
            if result.status is ToolResultStatus.OK:
                logger.info(
                    "Scheduled '%s' (%s), next: %s",
                    task_name, cron, (result.data or {}).get("next_run_at"),
                )
            else:
                logger.warning("Failed to schedule '%s': %s", task_name, result.error)

    async def shutdown(self):
        """Stop the background runner."""
        if self._runner:
            await self._runner.stop()

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
        from kestrel_sdk.signals import Signal, SignalMode, Status, Visibility
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
        (str | (str, float) tuple | None).

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

    async def _lookup_and_run_tool(self, task_name: str, args: dict) -> str:
        """Tool-lookup body shared by every cron source handler that
        delegates to a feature tool. This is the existing executor's
        tool-search logic, lifted out so the source registrations can
        invoke it without re-entering the dispatcher (which would loop)."""
        features = getattr(self.agent, "features", {})
        for feature in features.values():
            if not hasattr(feature, "get_tools"):
                continue
            for agent_tool in feature.get_tools():
                if agent_tool.name == task_name:
                    result = await agent_tool.execute(**args)
                    # Preserve the legacy JSON-encode contract for
                    # downstream consumers (task_execution_log.result_text,
                    # endpoints/agent.py history view).
                    if isinstance(result, str):
                        return result
                    return json.dumps(result, default=str)

        # Also check our own tools (SchedulerFeature has !schedule
        # commands but they're not typically scheduled themselves).
        for agent_tool in self.get_tools():
            if agent_tool.name == task_name:
                result = await agent_tool.execute(**args)
                if isinstance(result, str):
                    return result
                return json.dumps(result, default=str)

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

    async def _handle_backup_snapshot(self, args: dict) -> str:
        """ACTION handler for the `backup_snapshot` cron source. Hits
        the agent's sync service directly — there is no feature tool
        for this, so it can't go through the generic tool lookup."""
        sync = getattr(self.agent, "_sync_service", None)
        if sync:
            results = await sync.force_snapshot()
            return json.dumps(
                {t: {"success": r.success, "bytes": r.bytes_synced} for t, r in results.items()},
                default=str,
            )
        return json.dumps({"error": "no sync service configured"})

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

        storage = getattr(self.agent, "storage", None)
        if storage is None or not hasattr(storage, "purge_trash_older_than"):
            return json.dumps({
                "skipped": True,
                "reason": "storage facade missing purge_trash_older_than",
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
            return json.dumps({"error": str(e)})

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
        })

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
        # Always advance the persisted fingerprint to the latest observed
        # state, even on a no-signal change, so the next poll compares
        # against current reality rather than a stale baseline.
        await self._save_pr_watch_state(
            watch_key, decision.fingerprint, decision.normalized,
        )

        if not decision.should_signal:
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
                       enabled, last_run_at, next_run_at, created_at
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
    ) -> ToolResult:
        """
        Add a new scheduled task.

        Args:
            cron_expression: Cron expression (5 fields) or alias like @daily, @hourly
            task_name: Name of a registered scheduler-executable task — a
                built-in cron source (e.g. memory_consolidate, talon_monitor,
                github_pr_watch) or a loaded feature tool. Unknown names are
                rejected so they can't silently fail every tick (#1618).
            args_json: JSON-encoded arguments to pass to the tool (default: {})
        """
        if not self._db:
            return ToolResult.failed("Database not available")

        try:
            parse(cron_expression)
        except CronParseError as e:
            return ToolResult.failed(f"Invalid cron expression: {e}")

        try:
            parsed_args = json.loads(args_json)
            if not isinstance(parsed_args, dict):
                return ToolResult.failed("args_json must be a JSON object")
        except json.JSONDecodeError as e:
            return ToolResult.failed(f"Invalid args_json: {e}")

        # Reject unknown task names at creation time (#1618). A name that
        # is neither a built-in cron source nor a discoverable feature
        # tool would silently enter the schedule and then fail with
        # "Unknown task" on every single tick (the github_pr_watch
        # incident). Fail loudly here instead, listing the valid names.
        valid_names = self._scheduler_executable_task_names()
        if task_name not in valid_names:
            return ToolResult.failed(
                f"Unknown scheduled task '{task_name}'. It is not a "
                f"registered scheduler-executable task, so every cron tick "
                f"would fail with 'Unknown task'. Valid task names: "
                f"{', '.join(sorted(valid_names))}.",
                data={
                    "success": False,
                    "task_name": task_name,
                    "valid_task_names": sorted(valid_names),
                },
            )

        now = datetime.now(timezone.utc)
        try:
            first_run = next_run(cron_expression, after=now)
            next_run_at = first_run.isoformat()
        except CronParseError as e:
            return ToolResult.failed(f"Cannot compute next run: {e}")

        task_id = str(uuid.uuid4())
        now_iso = now.isoformat()

        try:
            await self._db.execute(
                """
                INSERT INTO scheduled_tasks
                    (id, agent_id, task_name, cron_expression, args_json,
                     enabled, last_run_at, next_run_at, created_at)
                VALUES (?, ?, ?, ?, ?, 1, NULL, ?, ?)
                """,
                (task_id, self._agent_id, task_name, cron_expression,
                 args_json, next_run_at, now_iso),
            )
        except Exception as e:
            logger.error("Failed to add scheduled task: %s", e)
            return ToolResult.failed(str(e))

        logger.info(
            "Scheduled task added: %s (%s) cron=%s next=%s",
            task_id, task_name, cron_expression, next_run_at,
        )

        return ToolResult.ok(
            confirmation=(
                f"Scheduled '{task_name}' (id={task_id[:8]}) "
                f"cron={cron_expression} next={next_run_at}"
            ),
            data={
                "success": True,
                "task_id": task_id,
                "task_name": task_name,
                "cron_expression": cron_expression,
                "next_run_at": next_run_at,
                "created_at": now_iso,
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
            row = await self._db.fetchone(
                "SELECT id FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
            )
            if not row:
                return ToolResult.failed(
                    f"Task {task_id} not found",
                    data={"task_id": task_id},
                )

            await self._db.execute(
                "DELETE FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
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
            row = await self._db.fetchone(
                "SELECT id, enabled FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
            )
            if not row:
                return ToolResult.failed(
                    f"Task {task_id} not found",
                    data={"task_id": task_id},
                )

            if not row[1]:
                return ToolResult.ok(
                    confirmation=f"Task {task_id} was already paused (no-op)",
                    data={"success": True, "task_id": task_id, "status": "already_paused"},
                )

            await self._db.execute(
                "UPDATE scheduled_tasks SET enabled = 0 WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
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
            row = await self._db.fetchone(
                "SELECT id, enabled, cron_expression FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
            )
            if not row:
                return ToolResult.failed(
                    f"Task {task_id} not found",
                    data={"task_id": task_id},
                )

            if row[1]:
                return ToolResult.ok(
                    confirmation=f"Task {task_id} was already running (no-op)",
                    data={"success": True, "task_id": task_id, "status": "already_running"},
                )

            cron_expr = row[2]
            now = datetime.now(timezone.utc)
            cron_now_invalid = False
            try:
                nxt = next_run(cron_expr, after=now)
                next_run_at = nxt.isoformat()
            except CronParseError:
                next_run_at = None
                cron_now_invalid = True

            await self._db.execute(
                "UPDATE scheduled_tasks SET enabled = 1, next_run_at = ? WHERE id = ? AND agent_id = ?",
                (next_run_at, task_id, self._agent_id),
            )
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
    ) -> ToolResult:
        """
        Update the cron expression on an existing scheduled task and recompute
        its next_run_at. Used by dynamic/self-adjusting schedulers (e.g. the
        reflection loop) to change task cadence without removing and re-adding.

        Args:
            task_id: The UUID of the task to update
            cron_expression: New cron expression (5 fields or alias)
        """
        if not self._db:
            return ToolResult.failed("Database not available")

        try:
            parse(cron_expression)
        except CronParseError as e:
            return ToolResult.failed(f"Invalid cron expression: {e}")

        try:
            row = await self._db.fetchone(
                "SELECT cron_expression, enabled FROM scheduled_tasks WHERE id = ? AND agent_id = ?",
                (task_id, self._agent_id),
            )
            if not row:
                return ToolResult.failed(
                    f"Task {task_id} not found",
                    data={"task_id": task_id},
                )

            old_cron = row[0]
            enabled = bool(row[1])

            if old_cron == cron_expression:
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

            now = datetime.now(timezone.utc)
            next_run_at: Optional[str] = None
            if enabled:
                try:
                    next_run_at = next_run(cron_expression, after=now).isoformat()
                except CronParseError as e:
                    return ToolResult.failed(f"Cannot compute next run: {e}")

            await self._db.execute(
                """
                UPDATE scheduled_tasks
                SET cron_expression = ?, next_run_at = ?
                WHERE id = ? AND agent_id = ?
                """,
                (cron_expression, next_run_at, task_id, self._agent_id),
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
                "next_run_at": next_run_at,
            },
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
            row = await self._db.fetchone(
                "SELECT id FROM task_execution_log WHERE id = ? AND agent_id = ?",
                (execution_id, self._agent_id),
            )
            if not row:
                return ToolResult.failed(
                    f"Execution {execution_id} not found",
                    data={"execution_id": execution_id},
                )

            await self._db.execute(
                "UPDATE task_execution_log SET outcome_signal = ? WHERE id = ? AND agent_id = ?",
                (clamped, execution_id, self._agent_id),
            )
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
