"""RestartCoordinatorFeature — durable, host-mediated restart requests.

Four agent-facing @tool entry points (request / list / cancel / escalation
acknowledgement) plus an ACTION-mode
``restart_coordinator`` cron entry that scans the durable table and
spawns a detached subprocess to actually restart Kestrel once safety
checks pass. After restart, ``initialize`` sweeps any in-flight
``executing`` rows owned by this agent into ``completed`` and emits a
``restart.completed`` signal so the requesting agent wakes.

The feature owns NO direct dependency on Talon or any other feature.
The safety checks conservatively introspect the agent's existing public
surfaces (``dispatcher.has_in_flight_signals``-style checks treat missing
idleness evidence as busy). The actual restart command is spawned via the CLI
entry point (``kestrel restart``) so the executor doesn't need to know the
runtime layout.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.enum_coerce import normalize_choice as _normalize_choice
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.storage.database_clock import database_clock

from .authority import (
    RestartAuthorityError,
    require_restart_request_authority,
    verify_restart_authority,
)
from .event_store import (
    ensure_restart_status_events_table,
    list_recent_events_for_history,
    record_event as record_status_event,
)
from .events import EVENT_NAME, build_restart_status_event
from .store import (
    KNOWN_OPERATIONS,
    KNOWN_POLICIES,
    KNOWN_STATUSES,
    KNOWN_URGENCIES,
    PENDING_STATES,
    acknowledge_escalation,
    cancel_request_if_owned,
    clear_deferral_started,
    ensure_restart_requests_table,
    get_request,
    get_request_for_agent,
    insert_request,
    list_requests,
    list_requests_needing_wake,
    mark_deferral_started,
    mark_wake_delivered,
    mark_wake_dispatched,
    record_update_log,
    update_status,
)
from .update_profiles import (
    KNOWN_UPDATE_PROFILES,
    default_sovereign_repo_path,
    get_update_profile,
    is_valid_target_ref,
    repo_is_git_checkout,
)

# Cap on captured stdout/stderr per update step kept in the durable log.
_OUTPUT_TAIL_CHARS = 2000

# Active request ids older than this are treated as abandoned markers
# (endpoint cleanup never ran — client disconnect, crashed generator)
# and swept before judging agent liveness. A genuine streaming cognition
# request completes in seconds-to-minutes; 15 minutes is far longer than
# any real turn yet breaks the deadlock well inside the ~20 min window
# observed in #1558.
STALE_ACTIVE_REQUEST_SECONDS = 900

# ``idle_agents_only`` remains the cautious default, but it is bounded.  A
# request that has been continuously blocked for this long advances with an
# explicit escalation event carrying the blocker evidence.  This prevents a
# busy marker or continuously-active requester from starving the only
# coordinator capable of applying its restart forever.
MAX_IDLE_ONLY_DEFERRAL_SECONDS = 1800

# How long to wait after spawning the detached restart before concluding the
# dispatch failed. A real restart kills this process well inside the window;
# still being alive with a dead child means it did not happen (#2667).
RESTART_DISPATCH_GRACE_SECONDS = 10

# How often to check the child within that window.
_DISPATCH_POLL_SECONDS = 0.5

# Failed dispatch attempts for one request in one boot before the coordinator
# stops retrying and rejects it. A permanently broken ``kestrel restart`` would
# otherwise spawn a doomed subprocess every cron tick indefinitely.
MAX_RESTART_DISPATCH_ATTEMPTS = 3

# An ``executing`` row stamped with THIS boot older than this never had its
# restart happen — the process it was going to kill is still running it. The
# in-dispatch check catches the common case; this is the backstop for a row
# whose verification never ran (feature reloaded, task cancelled, crash
# between the status write and the spawn) and which would otherwise sit in
# ``executing`` forever with no path back (#2667).
STALE_EXECUTING_SECONDS = 600

# Per-process boot identifier (#1796). Generated once at import, so it is
# stable for the lifetime of THIS host process and differs from any prior
# process. It is stamped onto a restart row at the moment the row crosses
# into ``executing`` and read by the post-restart sweep: a row whose stamp
# differs from this id was left ``executing`` by a PRIOR process — proof
# the restart already happened, so wake the requester. A row stamped with
# THIS id is a restart still in flight (or a detached restart that failed
# to kill the parent) within the live process and must NOT be falsely
# terminalized as ``completed``; it stays visibly ``executing``.
_PROCESS_BOOT_ID = uuid.uuid4().hex

logger = logging.getLogger(__name__)


# Synonyms LLMs reliably reach for that map cleanly onto the canonical
# enum sets. The middle urgency is ``normal``, but every model defaults to
# the universal low/medium/high taxonomy — accept ``medium`` rather than
# hard-failing the call. ``urgent``/``emergency`` likewise map to the
# nearest canonical rung. Normalization is case-insensitive.
_URGENCY_ALIASES = {
    "medium": "normal",
    "med": "normal",
    "moderate": "normal",
    "default": "normal",
    "urgent": "high",
    "important": "high",
    "emergency": "critical",
    "highest": "critical",
    "p0": "critical",
    "p1": "high",
    "p2": "normal",
    "p3": "low",
}
# Policy synonyms: the canonical values are verbose, so map the obvious
# shorthands onto them.
_POLICY_ALIASES = {
    "idle_only": "idle_agents_only",
    "idle": "idle_agents_only",
    "when_idle": "idle_agents_only",
    "timeout": "allow_busy_after_timeout",
    "after_timeout": "allow_busy_after_timeout",
    "busy_after_timeout": "allow_busy_after_timeout",
    "manual": "manual_only",
    "explicit": "manual_only",
}
# Status synonyms for the ``list_restart_requests`` filter.
_STATUS_ALIASES = {
    "cancelled": "canceled",
    "complete": "completed",
    "done": "completed",
    "running": "executing",
    "in_progress": "executing",
    "in-progress": "executing",
    "denied": "rejected",
}


def _tail(raw: Any) -> str:
    """Decode subprocess output bytes and keep the trailing tail only."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    text = text.strip()
    if len(text) > _OUTPUT_TAIL_CHARS:
        return text[-_OUTPUT_TAIL_CHARS:]
    return text


# Background-task name prefixes for *infrastructure* work that must never
# hold off an idle restart (#1626). Three shapes all wedged
# ``idle_agents_only`` forever by being counted as "busy":
#   - ``durable_signal_log`` — fire-and-forget log writes that complete in
#     well under a second but are minted continuously by heartbeats/scheduler
#     ticks, so one is almost always alive when the idle check runs. Covers
#     both ``durable_signal_log:`` and ``durable_signal_log_writer:``.
#     NOTE these are currently dispatcher-owned (``_outcome_log_tasks``) and
#     so do not reach the agent set this predicate reads — the entry is kept
#     because the exclusion is a statement of INTENT about this class of task,
#     and because the old ``signal_log:`` spelling listed here had been dead
#     since #2713 renamed them, which is the same list-drifts-from-reality
#     failure #2665 is about.
#   - ``a2a_question_expiry_sweep`` — an intentionally permanent ``while
#     True`` maintenance daemon (peers feature) that never completes.
#   - ``a2a_question_supervisor:`` — the sender-side SSE subscription
#     supervisor spawned by ``send_a2a_question`` (covers both the live
#     ``a2a_question_supervisor:<recipient>:<task_id>`` name and the
#     ``a2a_question_supervisor:replay:...`` startup-replay variant). It is
#     a passive, deadline-bounded wait for a peer's answer, NOT held
#     cognition: the correlation lives durably in ``pending_a2a_questions``
#     and the startup replay re-arms an in-flight supervisor (or terminalizes
#     a past-deadline row) after a restart, so killing one to restart loses
#     nothing. Counting it wedged ``idle_agents_only`` with a phantom "N
#     background tasks in flight" that survived restarts while the task store
#     read empty — an unanswered question during a peer/model outage pinned
#     the count immortally (#2666). Safe to exclude only BECAUSE the
#     supervisor is deadline-bounded (it exits at ``timeout_seconds`` and
#     replay expires past-deadline rows); do not exclude any peer wait that
#     lacks that guarantee.
# None is user/signal work; real work (``signal_dispatch:*``) still
# defers a restart. The name is already stamped on the task at creation —
# it was just never read here. New long-lived/bookkeeping daemons must be
# named with a prefix listed here (or excluded from ``_background_tasks``).
_INFRA_TASK_PREFIXES = (
    "durable_signal_log",
    "a2a_question_expiry_sweep",
    "a2a_question_supervisor:",
)


def _is_infra_background_task(task) -> bool:
    """True if ``task`` is fire-and-forget infrastructure bookkeeping that
    must not gate an idle restart (e.g. ``signal_log:*`` writes)."""
    try:
        name = task.get_name() or ""
    except Exception:
        return False
    return name.startswith(_INFRA_TASK_PREFIXES)


# How many task KINDS to describe individually in a deferral reason before
# summarising the rest. The bound exists because a deferring request writes a
# status-event row every cron tick (``* * * * *``) with no dedupe, so an
# unbounded reason accumulates in the event store for as long as the restart
# stays wedged.
_MAX_NAMED_BUSY_KINDS = 5

# Single task names are capped so one pathological name cannot dominate.
_MAX_BUSY_NAME_CHARS = 80


def _task_age_seconds(task, now: float) -> Optional[float]:
    """How long ``task`` has been running, or ``None`` if unstamped."""
    started = getattr(task, "_kestrel_started_at", None)
    if not isinstance(started, (int, float)):
        return None
    return max(0.0, now - started)


def _format_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "age unknown"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    return f"{int(seconds // 3600)}h"


def _describe_background_tasks(tasks, now: Optional[float] = None) -> str:
    """Describe the tasks blocking an idle restart, not just how many (#2665).

    A bare count ("2 background tasks in flight") cannot be reconciled against
    anything: the live report of two in-flight tasks alongside a task store
    returning zero rows was undiagnosable precisely because the coordinator
    never said WHICH handles it meant.

    Grouped by KIND and ordered OLDEST FIRST, both deliberately:

    - Truncating a flat list sorted by name hides whatever sorts late. Six
      ``a2a_*`` tasks would push a wedged ``signal_dispatch:*`` past the bound
      and out of the string — the bound doing the opposite of its job at
      exactly the moment it engages. Collapsing by kind means no kind can be
      truncated away by the volume of another.
    - Age is what separates "busy" from "wedged", and #2665's symptom was a
      duration symptom. Oldest first puts the likely culprit at the front.
    """
    now = time.monotonic() if now is None else now
    kinds: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        try:
            name = task.get_name() or "<unnamed>"
        except Exception:
            name = "<unnamed>"
        name = name[:_MAX_BUSY_NAME_CHARS]
        # Kind is the stable leading segment; the tail is the per-instance id.
        kind = name.split(":", 1)[0] if ":" in name else name
        age = _task_age_seconds(task, now)
        entry = kinds.setdefault(
            kind, {"count": 0, "oldest": None, "example": name},
        )
        entry["count"] += 1
        if age is not None and (
            entry["oldest"] is None or age > entry["oldest"]
        ):
            entry["oldest"] = age
            entry["example"] = name

    def _sort_key(item):
        # Oldest first; unstamped last but still ahead of nothing.
        oldest = item[1]["oldest"]
        return (0 if oldest is not None else 1, -(oldest or 0.0), item[0])

    ordered = sorted(kinds.items(), key=_sort_key)
    shown = ordered[:_MAX_NAMED_BUSY_KINDS]
    parts = []
    for _kind, entry in shown:
        label = entry["example"]
        if entry["count"] > 1:
            label = f"{label} x{entry['count']}"
        parts.append(f"{label} ({_format_age(entry['oldest'])})")
    remaining = len(ordered) - len(shown)
    if remaining > 0:
        parts.append(f"+{remaining} more kind(s)")
    return ", ".join(parts)


class RestartCoordinatorFeature(Feature):
    """Durable restart-request surface for agents (#1512)."""

    @property
    def tool_description(self) -> str:
        return (
            "Request a safe Kestrel host restart and track its outcome. "
            "Agent files a request; host coordinator executes when safe. "
            "A plain restart NEVER updates code. To pull/install a new ref "
            "before restarting, set operation='update_then_restart' with an "
            "explicit, allowlisted update profile — that step is always "
            "explicit and audited, never an implicit side effect of restart."
        )

    async def initialize(self):
        # Request ids whose restart.completed wake is currently being
        # supervised by a background ack task in THIS process. Guards the
        # cron-tick retry from re-enqueuing a duplicate wake while a long
        # cognition turn is still in flight (#1796).
        self._inflight_restart_acks: set = set()
        # request_id -> monotonic time THIS process crossed it into
        # ``executing``. Used to tell a dispatch still in flight from one that
        # silently failed, without a schema column: a row stamped with this
        # boot id but absent here has no dispatch behind it (#2667).
        self._executing_since: Dict[str, float] = {}
        # Failed dispatch attempts per request in THIS boot, so a permanently
        # broken restart stops being retried rather than flapping forever.
        self._dispatch_failures: Dict[str, int] = {}
        # When THIS feature instance came up. The boot id is module-scoped but
        # this map is per-instance, so a reload must not treat the previous
        # instance's in-flight dispatches as orphans the moment it starts.
        self._instance_started_at = time.monotonic()
        # Instance-level so a host (or a test) can tune how long to wait
        # before concluding a dispatched restart never happened.
        self._restart_dispatch_grace = RESTART_DISPATCH_GRACE_SECONDS
        self._db = resolve_feature_database(self.agent)
        if self._db is not None:
            try:
                await ensure_restart_requests_table(self._db)
                logger.info(
                    "RestartCoordinatorFeature: restart_requests table ready"
                )
            except Exception as e:
                # Fail CLOSED. Every read projects the full column list, so a
                # half-migrated table makes each one raise — and merely
                # warning here left the feature reporting itself enabled while
                # waking nobody for the whole boot, which is exactly what the
                # store's post-migration verification exists to prevent.
                # Dropping the handle degrades the coordinator explicitly:
                # its tools report storage unavailable and the sweep no-ops,
                # instead of looking healthy and doing nothing.
                logger.error(
                    "RestartCoordinatorFeature: restart_requests schema is "
                    "not usable; disabling coordinator storage for this "
                    "boot: %s", e,
                )
                self._db = None
        if self._db is not None:
            try:
                # #1562 — typed restart-status event records. Additive
                # CREATE TABLE IF NOT EXISTS, no existing column touched.
                await ensure_restart_status_events_table(self._db)
                logger.info(
                    "RestartCoordinatorFeature: "
                    "restart_status_events table ready"
                )
            except Exception as e:
                logger.warning(
                    "RestartCoordinatorFeature: "
                    "status-event table init failed: %s", e,
                )

        # Self-register the restart.completed signal source on the
        # agent's signal registry. Owning the registration here keeps
        # it co-located with the feature so when this package eventually
        # extracts to an external feature, the registration travels
        # with it.
        registry = getattr(self.agent, "signal_registry", None)
        if registry is not None and hasattr(registry, "register_with_policy"):
            from kestrel_sovereign.signals import RegistrationPolicy
            from kestrel_sovereign.signals.sources.restart import (
                build_restart_completed_registration,
            )
            # OPTIONAL policy (#2522): idempotent on a second initialize(); an
            # existing restart.completed with a DIFFERENT contract is reported
            # rather than swallowed by a broad except. Never raises.
            # Registered AS THIS FEATURE, so the base-class shutdown / boot
            # rollback releases it (#2522 P2, #3074).
            self._register_signal_sources(
                build_restart_completed_registration(),
                RegistrationPolicy.OPTIONAL,
            )

        # A successful restart orphans its child's stderr file (this process
        # dies before it can clean up), so boot is the only place that can.
        self._sweep_orphaned_restart_stderr()

        # Recover any row left in ``updating`` by a host that went down
        # mid-update (operator restart, crash) BEFORE the executing
        # sweep — such a row never reached the restart and must be
        # retried, not reported as a completed update-and-restart.
        await self._reset_interrupted_updates()

        # NOTE: the post-restart wake sweep does NOT run here. ``initialize``
        # runs during the feature-load phase — BEFORE the agent builds its
        # memory system + context manager — so a COGNITION wake dispatched now
        # cannot run a turn (it would defer/retry, the slow path the Sovereign
        # flagged). The sweep is fired from ``on_agent_ready`` instead, which the
        # agent calls at the END of initialize() once everything is up, so the
        # FIRST wake attempt succeeds immediately. The cron tick remains the
        # backstop for undelivered wakes (#1809).

    async def shutdown(self) -> None:
        """Stop owned wake acknowledgements before shared storage shuts down.

        The agent owns the dispatcher task and the shared database; this
        feature owns only the acknowledgement task that observes delivery and
        writes ``wake_delivered``.  Releasing the in-flight guard after the
        owned tasks have been cancelled keeps a later enable/retry from seeing
        a stale in-process acknowledgement.
        """
        await super().shutdown()
        inflight = getattr(self, "_inflight_restart_acks", None)
        if inflight is not None:
            inflight.clear()

    def get_router(self):
        """Expose the restart status-event API for chat-history reload.

        Lets the Console repaint the restart status-bubble trail when a
        conversation is loaded (#1816). Mounted dynamically so the route
        only exists when this feature is loaded.
        """
        from kestrel_sovereign.endpoints.restart_events import router
        return router

    async def on_agent_ready(self, agent=None) -> List[asyncio.Task[Any]]:
        """Agent fully initialized (memory + context manager + dispatcher up).

        The agent calls this at the very end of ``initialize()``. Running the
        post-restart wake sweep here — rather than in ``initialize`` — means the
        first ``restart.completed`` COGNITION dispatch happens against a
        ready-to-think agent, so it wakes the requester immediately instead of
        deferring and waiting up to a full cron interval (#1809). Best-effort:
        the cron tick is the backstop, so a transient failure here never wedges
        the wake.
        """
        try:
            return await self._reap_post_restart_rows()
        except Exception as e:  # never let readiness wiring break boot
            logger.warning("post-restart wake sweep on_agent_ready failed: %s", e)
            return []

    @tool(
        name="request_restart",
        description=(
            "File a sovereign-authorized durable whole-host restart request. "
            "This tool succeeds only inside a turn authenticated by the "
            "sovereign API key; agent identity, peer status, causation, and "
            "generic ASK/AUTO approval do not confer this authority. The exact "
            "operation/update bounds are sealed durably and re-verified by "
            "the host coordinator. "
            "The host coordinator "
            "evaluates safety and executes when conditions are met.\n\n"
            "urgency: one of low|normal|high|critical (default 'normal'); "
            "common synonyms are accepted ('medium'→normal, 'urgent'→high, "
            "'emergency'→critical). Higher urgency is executed first.\n"
            "policy: one of idle_agents_only|allow_busy_after_timeout|"
            "manual_only (default 'idle_agents_only'):\n"
            "  - idle_agents_only: wait for every co-hosted agent to become "
            "idle; after a bounded continuous deferral, emit an audited "
            "escalation and proceed so one blocker cannot starve the host.\n"
            "  - allow_busy_after_timeout: prefer idle, but execute anyway "
            "once the request has aged past the busy timeout even if the "
            "agent is still busy.\n"
            "  - manual_only: never auto-execute; the row waits for an "
            "explicit dispatch.\n\n"
            "operation='restart_only' (default) restarts the current code "
            "and NEVER updates it. operation='update_then_restart' first "
            "runs an explicit, allowlisted update profile (e.g. "
            "'sovereign_local_uv_sync': git fetch + checkout target_ref + "
            "uv sync) against a local checkout, then restarts into the new "
            "code. Update mode requires update_profile and target_ref; "
            "repo_path defaults to the local Sovereign checkout. "
            "Updating/installing is always explicit and audited — it is "
            "never an implicit side effect of a plain restart.\n\n"
            "Returns: data={created: bool, request: <public dict>}. The "
            "filed request's id is at data.request.id (NOT a top-level "
            "request_id) — pass it to list_restart_requests or "
            "cancel_restart_request."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!restart request",
    )
    async def request_restart(
        self,
        reason: str,
        urgency: str = "normal",
        policy: str = "idle_agents_only",
        desired_window: str = "",
        operation: str = "restart_only",
        update_profile: str = "",
        target_ref: str = "",
        repo_path: str = "",
        allow_migrations: bool = False,
    ) -> ToolResult:
        if not reason or not reason.strip():
            return ToolResult.failed(
                "reason is required",
                data={"created": False},
            )
        urgency = _normalize_choice(urgency, _URGENCY_ALIASES)
        if urgency not in KNOWN_URGENCIES:
            return ToolResult.failed(
                f"urgency must be one of {sorted(KNOWN_URGENCIES)}; "
                f"got {urgency!r}",
                data={"created": False},
            )
        policy = _normalize_choice(policy, _POLICY_ALIASES)
        if policy not in KNOWN_POLICIES:
            return ToolResult.failed(
                f"policy must be one of {sorted(KNOWN_POLICIES)}; "
                f"got {policy!r}",
                data={"created": False},
            )
        if operation not in KNOWN_OPERATIONS:
            return ToolResult.failed(
                f"operation must be one of {sorted(KNOWN_OPERATIONS)}; "
                f"got {operation!r}",
                data={"created": False},
            )

        # Authority is checked before update-mode path discovery or checkout
        # inspection. A caller who cannot request a whole-host mutation must
        # not be able to use its validation errors as a filesystem oracle.
        try:
            require_restart_request_authority()
        except RestartAuthorityError as error:
            return ToolResult.failed(
                str(error),
                data={"created": False, "authority": "required"},
            )

        # Validate and normalise the update-mode parameters up front so an
        # unsafe/unknown profile never reaches the durable table.
        update_repo_path = ""
        update_target_ref = ""
        if operation == "update_then_restart":
            if update_profile not in KNOWN_UPDATE_PROFILES:
                return ToolResult.failed(
                    "update_then_restart requires a known update_profile; "
                    f"got {update_profile!r}. Allowed: "
                    f"{sorted(KNOWN_UPDATE_PROFILES)}",
                    data={"created": False},
                )
            update_target_ref = (target_ref or "").strip()
            if not is_valid_target_ref(update_target_ref):
                return ToolResult.failed(
                    "update_then_restart requires a valid target_ref "
                    "(branch/tag/sha); "
                    f"got {target_ref!r}",
                    data={"created": False},
                )
            update_repo_path = (repo_path or "").strip()
            if not update_repo_path:
                update_repo_path = default_sovereign_repo_path()
            if not repo_is_git_checkout(update_repo_path):
                return ToolResult.failed(
                    "update_then_restart requires repo_path to be a local "
                    "git checkout; "
                    f"got {update_repo_path!r}. Pass repo_path explicitly.",
                    data={"created": False},
                )

        if self._db is None:
            return ToolResult.failed(
                "Restart coordinator storage unavailable",
                data={"created": False},
            )

        agent_id = getattr(self.agent, "did", "") or ""
        # Record the in-flight chat/agent turn that filed this request so
        # the coordinator can ignore the requester's own active-request
        # marker when judging idleness — that marker should not block the
        # very restart it asked for (#1561).
        requester_request_id = (
            getattr(self.agent, "_current_request_id", "") or ""
        )
        # Capture the chat session this request was filed from so the
        # post-restart wake lands in the SAME window (#1809, #2928). The
        # lifecycle accessor proves the calling task owns the live turn before
        # exposing its effective session. Reading the agent-global
        # ``_active_session_id`` directly can cross-wire unattended work into a
        # concurrent chat, while the logging ContextVar only covers an optional
        # query/header value and is not the turn's routing authority.
        # CLI/system/session-less requests remain explicitly unbound.
        origin_session_id = self._turn_session_id() or ""
        try:
            req = await insert_request(
                self._db,
                requested_by_agent=str(agent_id),
                reason=reason.strip(),
                urgency=urgency,
                policy=policy,
                desired_window=desired_window,
                operation=operation,
                update_repo_path=update_repo_path,
                update_target_ref=update_target_ref,
                update_profile=(
                    update_profile if operation == "update_then_restart" else ""
                ),
                update_allow_migrations=bool(allow_migrations),
                requester_request_id=str(requester_request_id),
                origin_session_id=origin_session_id,
            )
        except RestartAuthorityError as error:
            return ToolResult.failed(
                str(error),
                data={"created": False, "authority": "required"},
            )
        logger.info(
            "Restart request filed: id=%s op=%s urgency=%s policy=%s "
            "profile=%s ref=%s reason=%s",
            req.id, operation, urgency, policy, req.update_profile,
            req.update_target_ref, reason[:80],
        )
        # Surface the filed request as a chat-visible status event so the
        # Sovereign sees it without relying on agent prose (#1551).
        await self._emit_status_event(req, state="pending")
        return ToolResult.ok(
            confirmation=(
                f"Filed {operation} request {req.id} ({urgency}, {policy})"
            ),
            data={"created": True, "request": req.to_public_dict()},
        )

    @tool(
        name="list_restart_requests",
        description=(
            "List restart requests filed by this agent, optionally filtered "
            "by status. Other agents' requests are never visible. Valid "
            "statuses: pending|approved|updating|executing|completed|"
            "rejected|canceled (omit status for all). An unknown status is "
            "rejected with the valid set rather than silently returning no "
            "rows.\n\n"
            "Returns: data={count: int, requests: [<public dict>, ...]}."
        ),
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!restart list",
    )
    async def list_restart_requests(
        self, status: str = "",
    ) -> ToolResult:
        if self._db is None:
            return ToolResult.failed(
                "Restart coordinator storage unavailable",
                data={"requests": []},
            )
        status_filter = status.strip()
        if status_filter:
            status_filter = _normalize_choice(status_filter, _STATUS_ALIASES)
            if status_filter not in KNOWN_STATUSES:
                return ToolResult.failed(
                    f"status must be one of {sorted(KNOWN_STATUSES)}; "
                    f"got {status!r}",
                    data={"count": 0, "requests": []},
                )
        requester = self._agent_requester_id()
        if requester is None:
            return ToolResult.failed(
                "Restart request access requires this agent's durable identity",
                data={"count": 0, "requests": []},
            )
        rows = await list_requests(
            self._db,
            status=(status_filter or None),
            agent_id=requester,
        )
        return ToolResult.ok(
            confirmation=f"{len(rows)} restart request(s)",
            data={
                "count": len(rows),
                "requests": [r.to_public_dict() for r in rows],
            },
        )

    @tool(
        name="list_restart_status_events",
        description=(
            "List this agent's recent restart_status lifecycle events for "
            "chat-history reload and its pre-turn snapshot. Other agents' "
            "events are never visible. Newest "
            "first; uses the typed event records persisted alongside "
            "each SSE emit (#1562)."
        ),
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!restart events",
    )
    async def list_restart_status_events(
        self, limit: int = 100, since: str = "",
    ) -> ToolResult:
        if self._db is None:
            return ToolResult.failed(
                "Restart coordinator storage unavailable",
                data={"events": []},
            )
        try:
            limit_int = int(limit)
        except (TypeError, ValueError):
            limit_int = 100
        limit_int = max(1, min(1000, limit_int))
        requester = self._agent_requester_id()
        if requester is None:
            return ToolResult.failed(
                "Restart event access requires this agent's durable identity",
                data={"count": 0, "events": []},
            )
        rows = await list_recent_events_for_history(
            self._db,
            limit=limit_int,
            since=(since.strip() or None),
            agent_id=requester,
        )
        return ToolResult.ok(
            confirmation=f"{len(rows)} restart status event(s)",
            data={
                "count": len(rows),
                "events": [e.to_public_dict() for e in rows],
            },
        )

    @tool(
        name="acknowledge_restart_escalation",
        description=(
            "Explicitly re-authorize and acknowledge the bounded host-wide "
            "escalation policy for one pending restart request filed by this "
            "agent and migrated from an older release. Requests filed by "
            "another agent cannot be acknowledged. Sovereign-key authority "
            "is required. This is required once for legacy rows before a continuous busy "
            "deferral may override fleet quiescence. Pass request_id from "
            "list_restart_requests."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!restart acknowledge-escalation",
    )
    async def acknowledge_restart_escalation(self, request_id: str) -> ToolResult:
        if self._db is None:
            return ToolResult.failed(
                "Restart coordinator storage unavailable",
                data={"acknowledged": False},
            )
        normalized = (request_id or "").strip()
        if not normalized:
            return ToolResult.failed(
                "request_id is required",
                data={"acknowledged": False},
            )
        requester = self._agent_requester_id()
        if requester is None:
            return ToolResult.failed(
                "Restart request access requires this agent's durable identity",
                data={"acknowledged": False, "request_id": normalized},
            )
        try:
            require_restart_request_authority()
            acknowledged = await acknowledge_escalation(
                self._db,
                normalized,
                requested_by_agent=requester,
            )
        except RestartAuthorityError as error:
            return ToolResult.failed(
                str(error),
                data={
                    "acknowledged": False,
                    "request_id": normalized,
                    "authority": "required",
                },
            )
        if not acknowledged:
            row = await get_request_for_agent(self._db, normalized, requester)
            if row is not None and row.status not in PENDING_STATES:
                return ToolResult.failed(
                    f"Cannot acknowledge a request in state {row.status!r}",
                    data={
                        "acknowledged": False,
                        "request_id": normalized,
                        "current_status": row.status,
                    },
                )
            return ToolResult.failed(
                "Restart request is unavailable or not authorized",
                data={"acknowledged": False, "request_id": normalized},
            )
        return ToolResult.ok(
            confirmation=f"Acknowledged bounded escalation for {normalized}",
            data={"acknowledged": True, "request_id": normalized},
        )

    @tool(
        name="cancel_restart_request",
        description=(
            "Cancel this agent's still-pending restart request (status "
            "pending or approved). Another agent's request cannot be "
            "canceled. Rows already updating/executing/completed/rejected/"
            "canceled cannot be canceled. Pass request_id from "
            "data.request.id of request_restart (or data.requests[].id of "
            "list_restart_requests).\n\n"
            "Returns: data={canceled: bool, request_id: str} (plus "
            "current_status when the cancel is refused)."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!restart cancel",
    )
    async def cancel_restart_request(
        self, request_id: str, reason: str = "",
    ) -> ToolResult:
        if self._db is None:
            return ToolResult.failed(
                "Restart coordinator storage unavailable",
                data={"canceled": False},
            )
        if not request_id or not request_id.strip():
            return ToolResult.failed(
                "request_id is required",
                data={"canceled": False},
            )
        normalized = request_id.strip()
        requester = self._agent_requester_id()
        if requester is None:
            return ToolResult.failed(
                "Restart request access requires this agent's durable identity",
                data={"canceled": False, "request_id": normalized},
            )
        ok = await cancel_request_if_owned(
            self._db,
            normalized,
            requested_by_agent=requester,
            status_reason=(reason.strip() or "canceled by agent"),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        if not ok:
            # A scoped re-read may explain an owned lifecycle race. Missing and
            # foreign rows intentionally share one public result.
            fresh = await get_request_for_agent(self._db, normalized, requester)
            if fresh is None:
                return ToolResult.failed(
                    "Restart request is unavailable or not authorized",
                    data={"canceled": False, "request_id": normalized},
                )
            current = fresh.status
            return ToolResult.failed(
                f"Race: request transitioned to {current!r} before "
                f"cancel landed",
                data={
                    "canceled": False,
                    "request_id": normalized,
                    "current_status": current,
                },
            )
        row = await get_request_for_agent(self._db, normalized, requester)
        if row is None or row.status != "canceled":
            raise RuntimeError(
                "Canceled restart request did not remain durably visible"
            )
        await self._emit_status_event(
            row, state="canceled",
            status_reason=(reason.strip() or "canceled by agent"),
        )
        return ToolResult.ok(
            confirmation=f"Canceled restart request {normalized}",
            data={"canceled": True, "request_id": normalized},
        )

    def _agent_requester_id(self) -> Optional[str]:
        """Return the trusted durable principal bound to this feature instance."""

        value = getattr(self.agent, "did", None)
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()

    @tool(
        name="restart_coordinator",
        description=(
            "ACTION cron task — scan restart_requests, run safety "
            "checks, and execute pending requests by spawning a "
            "detached restart subprocess. No LLM cost."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!restart coordinator",
    )
    async def restart_coordinator(self) -> ToolResult:
        if self._db is None:
            return ToolResult.failed(
                "Restart coordinator storage unavailable",
                data={"executed": False, "pending": 0},
            )

        # Retry backstop (#1796): re-sweep any ``executing`` row whose
        # post-restart wake was not delivered on the init sweep (or whose
        # supervising ack task was lost to a later restart). The init
        # sweep alone leaves such a row stranded until the NEXT reboot;
        # re-running it here makes an undelivered wake retry on the normal
        # 1/min cron cadence. The in-flight guard + signal coalescing keep
        # this from re-waking a row whose turn is already running.
        await self._reap_post_restart_rows()

        # Recover rows this process crossed into ``executing`` whose restart
        # never happened. Without this the sweep only ever scans pending rows,
        # ``cancel_restart_request`` refuses executing ones, and the row has
        # no path back at all (#2667).
        await self._reconcile_stranded_executing_rows()

        pending = await list_requests(self._db, status="pending")
        approved = await list_requests(self._db, status="approved")
        candidates = pending + approved
        if not candidates:
            return ToolResult.ok(
                confirmation="No pending restart requests",
                data={"executed": False, "pending": 0},
            )

        # Highest-urgency / earliest-requested wins.
        urgency_rank = {
            "critical": 0, "high": 1, "normal": 2, "low": 3,
        }
        candidates.sort(
            key=lambda r: (
                urgency_rank.get(r.urgency, 4), r.requested_at,
            )
        )

        executed: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        for req in candidates:
            # Reject unsigned legacy, forged, tampered, or key-revoked rows
            # before policy/safety can defer them indefinitely. This is the
            # explicit legacy migration policy: no automatic authority
            # backfill and no inference from historical approval state. The
            # sovereign acknowledgement tool is the only re-authorization door.
            if await self._reject_invalid_authority(
                req,
                expected_current_status=req.status,
            ):
                continue
            req, decision = await self._evaluate_and_track_safety(req)
            if not decision["safe"]:
                if decision.get("deferable", True):
                    deferred.append({
                        "request_id": req.id,
                        "reason": decision["reason"],
                        "request_age_seconds": decision.get("request_age_seconds"),
                        "deferral_age_seconds": decision.get("deferral_age_seconds"),
                        "blocker": decision.get("blocker"),
                    })
                    if decision.get("lost_race"):
                        continue
                    # Surface the deferred attempt + its reason (#1551).
                    await self._emit_status_event(
                        req, state="pending",
                        deferral_reason=decision["reason"],
                        blocker=decision.get("blocker"),
                        request_age_seconds=decision.get("request_age_seconds"),
                        deferral_age_seconds=decision.get("deferral_age_seconds"),
                    )
                    continue
                # Hard reject.
                rejected = await update_status(
                    self._db, req.id,
                    status="rejected",
                    status_reason=decision["reason"],
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    expected_current_status=req.status,
                    expected_authority_signature=req.authority_signature,
                )
                if rejected:
                    await self._emit_status_event(
                        req, state="rejected", status_reason=decision["reason"],
                    )
                continue

            # Safety checks can await fleet state. Re-verify immediately before
            # crossing into update/execution so key rotation during that wait
            # revokes the request before any host mutation begins.
            if await self._reject_invalid_authority(
                req,
                expected_current_status=req.status,
            ):
                continue

            # Move out of the pending state BEFORE doing work. A plain
            # restart goes straight to ``executing`` (the spawn window is
            # milliseconds). An ``update_then_restart`` first goes to
            # ``updating`` — the git fetch/checkout + ``uv sync`` can take
            # minutes, and a row in ``updating`` must NOT be mistaken by
            # the post-restart sweep for a completed restart if the host
            # reboots for an unrelated reason mid-update. Only once the
            # update is done do we move to ``executing`` (right before the
            # spawn), so the sweep's "executing → completed" wake fires
            # only for restarts we actually performed.
            initial_state = (
                "updating" if req.operation == "update_then_restart"
                else "executing"
            )
            if initial_state == "executing":
                # Record BEFORE the transition commits. ``update_status``
                # awaits, so recording after it leaves a window where the row
                # is durably ``executing`` under this boot id with no entry
                # here — and ``_reconcile_stranded_executing_rows`` treats
                # exactly that as an orphan and resets a dispatch that is
                # very much alive. Popped again below if the transition loses
                # its race.
                self._executing_since[req.id] = time.monotonic()
            moved = await update_status(
                self._db, req.id,
                status=initial_state,
                status_reason=(
                    "running update profile before restart"
                    if initial_state == "updating"
                    else "dispatched to detached restart subprocess"
                ),
                expected_current_status=req.status,
                expected_authority_signature=req.authority_signature,
                # Stamp the live process only when crossing straight to
                # ``executing`` (#1796); an ``updating`` row has not yet
                # reached the restart, so it carries no boot stamp.
                executing_boot_id=(
                    _PROCESS_BOOT_ID if initial_state == "executing" else None
                ),
            )
            if not moved:
                if initial_state == "executing":
                    self._executing_since.pop(req.id, None)
                deferred.append({
                    "request_id": req.id,
                    "reason": "lost race against another transition",
                })
                continue
            req.status = initial_state

            if decision.get("escalated"):
                await self._emit_status_event(
                    req,
                    state="escalated",
                    deferral_reason=decision["reason"],
                    blocker=decision.get("blocker"),
                    request_age_seconds=decision.get("request_age_seconds"),
                    deferral_age_seconds=decision.get("deferral_age_seconds"),
                    escalated=True,
                )

            # Surface the transition out of pending — ``updating`` (update
            # profile running) or ``executing`` (restart dispatched) (#1551).
            await self._emit_status_event(req, state=initial_state)

            # update_then_restart: run the allowlisted update profile
            # against the local checkout BEFORE restarting. A failure
            # here records the audit log and decides retryable vs
            # terminal; only a clean update proceeds to the spawn.
            if req.operation == "update_then_restart":
                # The transition and status event both await external work.
                # Re-verify at the actual mutation boundary so key rotation
                # during either await revokes the update as well as restart.
                if await self._reject_invalid_authority(
                    req,
                    expected_current_status="updating",
                ):
                    continue
                handled = await self._handle_update_then_restart(req)
                if handled is not None:
                    # Either deferred (retryable) or rejected (terminal).
                    deferred.append(handled)
                    # Reflect the post-update outcome the helper landed
                    # the row on — fetch the fresh status so a rejected
                    # update reads as rejected, a retryable one as a
                    # deferred pending (#1551).
                    fresh = await get_request(self._db, req.id)
                    if fresh is not None:
                        if fresh.status == "rejected":
                            await self._emit_status_event(
                                fresh, state="rejected",
                                status_reason=fresh.status_reason,
                            )
                        else:
                            await self._emit_status_event(
                                fresh, state=fresh.status,
                                deferral_reason=handled.get("reason", ""),
                            )
                    continue
                # The update is the longest awaited mutation in this path.
                # Recheck the seal before any safety-state write: if the key
                # rotated, trying to reseal a new deferral timestamp would
                # lose its CAS and strand the row in ``updating`` forever.
                if await self._reject_invalid_authority(
                    req,
                    expected_current_status="updating",
                ):
                    continue
                # Re-run the safety gate before the restart now that the
                # (possibly slow) update has completed.
                req, decision = await self._evaluate_and_track_safety(req)
                if not decision["safe"]:
                    if decision.get("lost_race"):
                        deferred.append({
                            "request_id": req.id,
                            "reason": decision["reason"],
                        })
                        continue
                    moved = await update_status(
                        self._db, req.id,
                        status="pending",
                        status_reason=(
                            "update succeeded but agent became unsafe to "
                            f"restart: {decision['reason']}"
                        ),
                        expected_current_status="updating",
                        expected_authority_signature=req.authority_signature,
                    )
                    if not moved:
                        deferred.append({
                            "request_id": req.id,
                            "reason": "lost race after update safety check",
                        })
                        continue
                    deferred.append({
                        "request_id": req.id,
                        "reason": (
                            "update ok; restart deferred — "
                            f"{decision['reason']}"
                        ),
                        "request_age_seconds": decision.get("request_age_seconds"),
                        "deferral_age_seconds": decision.get("deferral_age_seconds"),
                        "blocker": decision.get("blocker"),
                    })
                    await self._emit_status_event(
                        req, state="pending",
                        deferral_reason=(
                            "update ok; restart deferred — "
                            f"{decision['reason']}"
                        ),
                        blocker=decision.get("blocker"),
                        request_age_seconds=decision.get("request_age_seconds"),
                        deferral_age_seconds=decision.get("deferral_age_seconds"),
                    )
                    continue
                # Update done and still safe — NOW cross into ``executing``
                # right before the spawn so the post-restart sweep
                # recognizes the restart we are about to perform.
                # Recorded before the write for the same reason as the
                # straight-to-executing path above.
                self._executing_since[req.id] = time.monotonic()
                moved = await update_status(
                    self._db, req.id,
                    status="executing",
                    status_reason="update complete; dispatching restart",
                    expected_current_status="updating",
                    expected_authority_signature=req.authority_signature,
                    executing_boot_id=_PROCESS_BOOT_ID,
                )
                if not moved:
                    self._executing_since.pop(req.id, None)
                    deferred.append({
                        "request_id": req.id,
                        "reason": "lost race after update before restart",
                    })
                    continue
                req.status = "executing"
                if decision.get("escalated"):
                    await self._emit_status_event(
                        req,
                        state="escalated",
                        deferral_reason=decision["reason"],
                        blocker=decision.get("blocker"),
                        request_age_seconds=decision.get("request_age_seconds"),
                        deferral_age_seconds=decision.get("deferral_age_seconds"),
                        escalated=True,
                    )
                await self._emit_status_event(req, state="executing")

            # Re-verify at the actual restart boundary. This catches key
            # rotation/revocation during safety waits or a long update and
            # prevents a check/use split from turning stale evidence into a
            # fleet-wide process interruption.
            if await self._reject_invalid_authority(
                req,
                expected_current_status="executing",
            ):
                self._executing_since.pop(req.id, None)
                continue

            try:
                proc = self._spawn_restart_subprocess()
            except Exception as e:
                logger.error(
                    "restart_coordinator: spawn failed: %s", e,
                )
                await update_status(
                    self._db, req.id,
                    status="pending",
                    status_reason=f"spawn failed: {e}",
                    expected_current_status="executing",
                    expected_authority_signature=req.authority_signature,
                )
                await self._emit_status_event(
                    req, state="pending",
                    deferral_reason=f"spawn failed: {e}",
                )
                continue

            # Popen returning does not mean the restart happened. Watch the
            # child in the background: if we are still alive after the grace
            # window and it is not, the row must NOT be left ``executing`` — a
            # later unrelated restart would find a prior-boot executing row
            # and terminalize it as "completed", reporting a restart that
            # never occurred (#2667).
            self._arm_restart_dispatch_watch(proc, req.id)

            executed.append({"request_id": req.id})
            # Only execute one per poll — the host process is about
            # to die. Anything else in the queue is the next boot's
            # problem.
            break

        return ToolResult.ok(
            confirmation=(
                f"restart_coordinator: pending={len(candidates)} "
                f"executed={len(executed)} deferred={len(deferred)}"
            ),
            data={
                "pending": len(candidates),
                "executed": executed,
                "deferred": deferred,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _reject_invalid_authority(
        self,
        req,
        *,
        expected_current_status: str,
    ) -> bool:
        """Stop processing unless the exact durable sovereign seal is current.

        Safety-state writes reseal a row without changing its lifecycle status.
        Re-reading only to verify the stale caller object therefore leaves a
        check/use gap: another coordinator can clear or start a deferral clock
        after safety evaluation and before execution.  A changed, valid seal is
        not rejected; pending work is deferred for a fresh evaluation, while a
        row already crossed into a mutation state is returned to ``pending``.
        """

        fresh = await get_request(self._db, req.id)
        if fresh is None or fresh.status != expected_current_status:
            return True

        verified, reason = verify_restart_authority(fresh)
        if verified:
            if fresh.authority_signature == req.authority_signature:
                return False
            if expected_current_status not in PENDING_STATES:
                recovered = await update_status(
                    self._db,
                    fresh.id,
                    status="pending",
                    status_reason=(
                        "signed safety state changed during restart dispatch; "
                        "returned to pending for reevaluation"
                    ),
                    expected_current_status=expected_current_status,
                    expected_authority_signature=fresh.authority_signature,
                )
                if recovered:
                    fresh.status = "pending"
                    await self._emit_status_event(
                        fresh,
                        state="pending",
                        deferral_reason=(
                            "signed safety state changed during restart dispatch; "
                            "reevaluating"
                        ),
                    )
            return True
        landed = await update_status(
            self._db,
            fresh.id,
            status="rejected",
            status_reason=f"authority denied: {reason}",
            completed_at=datetime.now(timezone.utc).isoformat(),
            expected_current_status=expected_current_status,
            expected_authority_signature=fresh.authority_signature,
        )
        if landed:
            rejected = await get_request(self._db, fresh.id)
            await self._emit_status_event(
                rejected or fresh,
                state="rejected",
                status_reason=f"authority denied: {reason}",
            )
        return True

    async def _emit_status_event(
        self,
        req,
        *,
        state: str,
        deferral_reason: str = "",
        status_reason: str = "",
        blocker: Optional[Dict[str, Any]] = None,
        request_age_seconds: Optional[float] = None,
        deferral_age_seconds: Optional[float] = None,
        escalated: bool = False,
    ) -> None:
        """Surface a chat-visible ``restart_status`` event (#1551).

        Best-effort: a restart is an audited deployment primitive, so
        the Sovereign should see it in chat — but a missing/raising
        ``emit_event`` (headless host, test agent) must never break the
        request lifecycle. Failures are logged at debug and swallowed.

        Every emit is also persisted to ``restart_status_events`` so
        the lifecycle trail survives reload/history navigation, the
        frontend can dedupe by stable ``dedupe_signature``, and the
        agent's pre-turn state block can render restart context as
        non-instructional state (#1562). The persistence is the audit
        primary; the SSE emit is the live-paint side-channel.
        """
        agent_did = getattr(self.agent, "did", "") or ""
        requested_by_agent = str(
            getattr(req, "requested_by_agent", "") or agent_did
        )
        payload = build_restart_status_event(
            req,
            state=state,
            deferral_reason=deferral_reason,
            status_reason=status_reason,
            agent_did=str(agent_did),
            requested_by_agent_name=self._resolve_requesting_agent_name(
                requested_by_agent
            ),
            blocker=blocker,
            request_age_seconds=request_age_seconds,
            deferral_age_seconds=deferral_age_seconds,
            escalated=escalated,
        )

        # Audit row first. If the persist fails AND a DB is available,
        # skip the SSE emit too — a UI bubble with no durable backing
        # row would reappear differently on reload (codex P2 r1).
        # When no DB is configured at all (headless host, test stub),
        # the SSE emit is still safe because there's no audit promise
        # to break.
        persist_ok = True
        if self._db is not None:
            try:
                await record_status_event(
                    self._db,
                    request_id=str(getattr(req, "id", "")),
                    state=str(state),
                    agent_id=requested_by_agent,
                    payload=payload,
                    operation=str(
                        getattr(req, "operation", "restart_only")
                    ),
                    urgency=str(getattr(req, "urgency", "normal")),
                    policy=str(
                        getattr(req, "policy", "idle_agents_only")
                    ),
                )
            except Exception as e:  # pragma: no cover - defensive
                persist_ok = False
                logger.warning(
                    "restart_status persist failed for %s, skipping "
                    "SSE emit to avoid phantom bubble: %s",
                    getattr(req, "id", "?"), e,
                )

        if not persist_ok:
            return

        emit = getattr(self.agent, "emit_event", None)
        if emit is None:
            return
        try:
            await emit(EVENT_NAME, payload)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(
                "restart_status emit failed for %s: %s",
                getattr(req, "id", "?"), e,
            )

    def _resolve_requesting_agent_name(self, requested_by_agent: str) -> str:
        """Resolve a restart requester DID to a co-hosted agent name."""
        requester_did = str(requested_by_agent or "")
        if not requester_did:
            return ""
        own_name = self._agent_name_if_did_matches(self.agent, requester_did)
        if own_name:
            return own_name
        manager_name = self._agent_manager_name_for_did(requester_did)
        if manager_name:
            return manager_name
        agents = self._resolve_cohosted_agents()
        if not agents:
            return ""
        for agent in agents:
            name = self._agent_name_if_did_matches(agent, requester_did)
            if name:
                return name
        return ""

    @staticmethod
    def _agent_name_if_did_matches(agent: Any, requested_by_agent: str) -> str:
        if str(getattr(agent, "did", "") or "") != requested_by_agent:
            return ""
        for attr in ("name", "_agent_name", "agent_name"):
            name = str(getattr(agent, attr, "") or "")
            if name and name != requested_by_agent:
                return name
        return ""

    def _agent_manager_name_for_did(self, requested_by_agent: str) -> str:
        for attr in ("_agent_manager", "agent_manager"):
            manager = getattr(self.agent, attr, None)
            get_agent_name = getattr(manager, "get_agent_name", None)
            if get_agent_name is None:
                continue
            try:
                name = str(get_agent_name(requested_by_agent) or "")
            except Exception:  # pragma: no cover - defensive
                continue
            if name and name != requested_by_agent:
                return name
        return ""

    async def _evaluate_and_track_safety(self, req):
        """Evaluate safety while maintaining the durable busy interval."""

        database_now = await database_clock(self._db)
        decision = self._evaluate_safety(req, database_now=database_now)
        if (
            not decision["safe"]
            and decision.get("deferable", True)
            and req.policy == "idle_agents_only"
            and not getattr(req, "first_blocked_at", "")
        ):
            original_status = req.status
            refreshed = await mark_deferral_started(
                self._db,
                req.id,
                expected_current_status=original_status,
            )
            if refreshed is None or refreshed.status != original_status:
                current_status = (
                    refreshed.status if refreshed is not None else "missing"
                )
                return req, {
                    "safe": False,
                    "deferable": True,
                    "lost_race": True,
                    "reason": (
                        "lost race while recording restart deferral: "
                        f"expected {original_status!r}, found {current_status!r}"
                    ),
                    "blocker": None,
                    "request_age_seconds": self._request_age_seconds(
                        req, database_now
                    ),
                    "deferral_age_seconds": self._deferral_age_seconds(
                        req, database_now
                    ),
                }
            req = refreshed
            database_now = await database_clock(self._db)
            decision = self._evaluate_safety(req, database_now=database_now)
        elif (
            decision["safe"]
            and not decision.get("escalated")
            and getattr(req, "first_blocked_at", "")
        ):
            # A genuinely idle observation breaks the continuous-deferral
            # interval. An escalation that proceeds while busy deliberately
            # retains its evidence if dispatch later fails and the row retries.
            cleared = await clear_deferral_started(
                self._db,
                req.id,
                expected_current_status=req.status,
            )
            if cleared is not None:
                # The clear reseals authority evidence. Continue with that
                # exact durable row, never the pre-clear in-memory signature.
                req = cleared
            else:
                refreshed = await get_request(self._db, req.id)
                verified = (
                    verify_restart_authority(refreshed)[0]
                    if refreshed is not None
                    else False
                )
                if (
                    not verified
                    or refreshed is None
                    or refreshed.status != req.status
                    or refreshed.first_blocked_at
                ):
                    current_status = (
                        refreshed.status if refreshed is not None else "missing"
                    )
                    return req, {
                        "safe": False,
                        "deferable": True,
                        "lost_race": True,
                        "reason": (
                            "lost race while clearing restart deferral: "
                            f"expected {req.status!r}, found "
                            f"{current_status!r}"
                        ),
                        "blocker": None,
                        "request_age_seconds": self._request_age_seconds(
                            req,
                            database_now,
                        ),
                        "deferral_age_seconds": self._deferral_age_seconds(
                            req,
                            database_now,
                        ),
                    }
                req = refreshed
        return req, decision

    def _evaluate_safety(
        self, req, *, database_now: datetime,
    ) -> Dict[str, Any]:
        """Return ``{safe, reason, deferable}`` for one request.

        ``deferable=True`` means "try again next poll"; ``False`` means
        "reject permanently" (e.g. unknown policy — should never
        happen since insert_request gates this, but defensive).
        """
        policy = req.policy
        if policy not in KNOWN_POLICIES:
            return {
                "safe": False,
                "deferable": False,
                "reason": f"unknown policy {policy!r}",
                "blocker": None,
                "request_age_seconds": self._request_age_seconds(
                    req, database_now
                ),
                "deferral_age_seconds": self._deferral_age_seconds(
                    req, database_now
                ),
                "escalated": False,
            }
        if policy == "manual_only":
            return {
                "safe": False,
                "deferable": True,
                "reason": "policy=manual_only; awaiting explicit dispatch",
                "blocker": None,
                "request_age_seconds": self._request_age_seconds(
                    req, database_now
                ),
                "deferral_age_seconds": self._deferral_age_seconds(
                    req, database_now
                ),
                "escalated": False,
            }

        # The dispatched command is ``kestrel restart`` with no agent name, so
        # it stops every agent in this host process. Normal dispatch therefore
        # requires whole-fleet quiescence (#F235). The requesting turn's own
        # marker is still excluded only on its owning agent (#1561).
        idle = self._fleet_idle(
            ignore_request_id=getattr(req, "requester_request_id", "") or "",
        )
        request_age = self._request_age_seconds(req, database_now)
        deferral_age = self._deferral_age_seconds(req, database_now)
        if idle["idle"]:
            return {
                "safe": True,
                "deferable": True,
                "reason": "",
                "blocker": None,
                "request_age_seconds": request_age,
                "deferral_age_seconds": deferral_age,
                "escalated": False,
            }

        if policy == "allow_busy_after_timeout":
            if self._request_aged_past_timeout(req, database_now):
                return {
                    "safe": True,
                    "deferable": True,
                    "reason": "timeout policy expired",
                    "blocker": idle.get("blocker"),
                    "request_age_seconds": request_age,
                    "deferral_age_seconds": deferral_age,
                    "escalated": False,
                }
            return {
                "safe": False,
                "deferable": True,
                "reason": (
                    f"host busy ({idle['reason']}); waiting for "
                    f"timeout to elapse"
                ),
                "blocker": idle.get("blocker"),
                "request_age_seconds": request_age,
                "deferral_age_seconds": deferral_age,
                "escalated": False,
            }

        # idle_agents_only
        if (
            deferral_age is not None
            and deferral_age >= MAX_IDLE_ONLY_DEFERRAL_SECONDS
        ):
            if not bool(getattr(req, "escalation_acknowledged", False)):
                return {
                    "safe": False,
                    "deferable": True,
                    "reason": (
                        "idle_agents_only deferral limit reached for a "
                        "pre-upgrade request; explicit escalation "
                        "acknowledgement required"
                    ),
                    "blocker": idle.get("blocker"),
                    "request_age_seconds": request_age,
                    "deferral_age_seconds": deferral_age,
                    "escalated": False,
                }
            return {
                "safe": True,
                "deferable": True,
                "reason": (
                    "idle_agents_only deferral limit reached; escalating with "
                    f"host blocker ({idle['reason']})"
                ),
                "blocker": idle.get("blocker"),
                "request_age_seconds": request_age,
                "deferral_age_seconds": deferral_age,
                "escalated": True,
            }
        return {
            "safe": False,
            "deferable": True,
            "reason": f"host busy ({idle['reason']})",
            "blocker": idle.get("blocker"),
            "request_age_seconds": request_age,
            "deferral_age_seconds": deferral_age,
            "escalated": False,
        }

    def _resolve_cohosted_agents(self) -> Optional[List[Any]]:
        """Every agent sharing this host process, or None for a single-agent host.

        Primary source is ``_cohosted_agents_provider`` (installed by
        AgentManager at registration). As a backstop, resolve the agent's
        attached ``_agent_manager``/``agent_manager`` — an agent registered
        outside ``AgentManager._load_one`` (e.g. the SpawnFeature
        lightweight-manager path) may lack the provider but still carry a
        manager backref, and its spawned children must still gate a whole-host
        restart (#F235).
        """
        provider = getattr(self.agent, "_cohosted_agents_provider", None)
        if provider is not None:
            try:
                return list(provider() or [self.agent])
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("restart_coordinator: fleet provider failed: %s", e)
        for attr in ("_agent_manager", "agent_manager"):
            mgr = getattr(self.agent, attr, None)
            if mgr is not None and hasattr(mgr, "list_agents"):
                try:
                    return list(mgr.list_agents().values()) or [self.agent]
                except Exception:  # pragma: no cover - defensive
                    continue
        return None

    def _fleet_idle(self, ignore_request_id: str = "") -> Dict[str, Any]:
        """Idleness across all agents before a whole-host restart (#F235).

        The multi-agent host installs ``agent._cohosted_agents_provider`` (a
        callable returning every co-hosted agent) at load time. When absent
        (single-agent host, or unwired), this degrades to the requester-only
        check — behaviour-preserving for the single-agent case. Only the
        REQUESTING agent excludes its own requester marker; a sibling defers
        the restart on ANY active request.
        """
        agents = self._resolve_cohosted_agents()
        if agents is None:
            # No fleet view — genuine single-agent host. Requester-only check
            # (behaviour-preserving).
            return self._agent_appears_idle(ignore_request_id=ignore_request_id)
        for other in agents:
            excl = ignore_request_id if other is self.agent else ""
            # Name tasks only for OUR agent. This reason is persisted to the
            # coordinator agent's event store and pushed on its SSE stream, so
            # enumerating a sibling's tasks would publish that agent's
            # topology — peer counterparties, active integrations, DIDs — to a
            # different tenant. On a multi-tenant host that is a disclosure,
            # and #2665 is a self-diagnosis: nothing here needs sibling task
            # identity, only that the sibling is busy.
            state = self._agent_appears_idle(
                ignore_request_id=excl,
                agent=other,
                name_tasks=(other is self.agent),
            )
            if not state["idle"]:
                name = getattr(other, "name", None) or getattr(other, "did", "?")
                blocker = dict(state.get("blocker") or {})
                blocker["scope"] = (
                    "requesting_agent" if other is self.agent
                    else "cohosted_agent"
                )
                return {
                    "idle": False,
                    "reason": f"co-hosted agent {name} busy ({state['reason']})",
                    "blocker": blocker,
                }
        return {"idle": True, "reason": "", "blocker": None}

    def _agent_appears_idle(
        self, ignore_request_id: str = "", agent: Any = None,
        name_tasks: bool = True,
    ) -> Dict[str, Any]:
        """Idle check against an agent's in-flight surface.

        ``agent`` defaults to ``self.agent`` (the requesting agent). Passing a
        co-hosted sibling agent lets the fleet-idleness gate (#F235) evaluate
        every agent sharing this host process, not just the requester.

        The default ``KestrelAgent`` exposes ``_active_request_ids``
        (set of in-flight cognition request IDs) and
        ``_background_tasks`` (asyncio tasks the dispatcher started).
        Either non-empty → not idle.

        ``ignore_request_id`` is the chat/agent turn that filed the
        restart being evaluated. That turn's own active-request marker
        must not block the very restart it requested, so it is excluded
        from the active-request count for that specific row (#1561). All
        other active requests still count as busy.

        Optional dispatcher hooks ``in_flight_signals`` /
        ``active_count`` are consulted first if present (gives feature
        tests + extracted-feature deployments a cheap override).

        If NONE of these surfaces are available the agent's idleness
        is genuinely unknown — conservatively report busy so
        ``idle_agents_only`` policy does not silently lose its safety
        gate (codex P1 on PR #1512 round 1). Hosts that want eager
        restarts can use ``allow_busy_after_timeout``.
        """
        if agent is None:
            agent = self.agent
        any_surface_seen = False
        dispatcher = getattr(agent, "dispatcher", None)
        if dispatcher is not None:
            for attr in ("in_flight_signals", "active_count"):
                if hasattr(dispatcher, attr):
                    any_surface_seen = True
                    try:
                        val = getattr(dispatcher, attr)
                        if callable(val):
                            val = val()
                        if val:
                            return {
                                "idle": False,
                                "reason": (
                                    f"dispatcher reports {attr}={val}"
                                    if name_tasks
                                    else "dispatcher reports work in flight"
                                ),
                                "blocker": {
                                    "scope": "requesting_agent",
                                    "kind": "dispatcher",
                                    "surface": attr if name_tasks else None,
                                    "count": (
                                        int(val)
                                        if name_tasks and isinstance(val, int)
                                        else None
                                    ),
                                    "oldest_age_seconds": None,
                                },
                            }
                    except Exception:
                        # Treat introspection failure as "unknown" —
                        # the catch-all conservative fallback below
                        # handles it.
                        continue

        active_ids = getattr(agent, "_active_request_ids", None)
        if active_ids is not None:
            any_surface_seen = True
            # A finished/abandoned stream should have been cleared by the
            # endpoint's `finally` (`_cleanup_cancelled_request`). A client
            # disconnect or crashed generator can leave a request id
            # registered forever, permanently blocking `idle_agents_only`
            # restarts (#1558). Sweep ids older than the staleness window
            # before counting so a stale marker can never deadlock us.
            pruner = getattr(agent, "prune_stale_active_requests", None)
            if callable(pruner):
                try:
                    pruned = pruner(STALE_ACTIVE_REQUEST_SECONDS)
                    if pruned:
                        logger.info(
                            "restart_coordinator: swept %d stale active "
                            "request id(s) (older than %ds): %s",
                            len(pruned), STALE_ACTIVE_REQUEST_SECONDS,
                            pruned,
                        )
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug(
                        "restart_coordinator: stale-request sweep "
                        "failed: %s", e,
                    )
            # Count blockers EXCLUDING the requester's own turn — that
            # marker should not defer the restart it filed (#1561). The
            # stale-request sweep above is the backstop for abandoned
            # markers; this is the normal path for requester-self restarts.
            try:
                blockers = [
                    rid for rid in active_ids if rid != ignore_request_id
                ]
                n = len(blockers)
            except TypeError:
                n = 0
            if n:
                ages_fn = getattr(agent, "active_request_ages", None)
                blocker_ages: Dict[str, float] = {}
                if name_tasks and callable(ages_fn):
                    try:
                        blocker_ages = {
                            str(rid): float(age)
                            for rid, age in ages_fn().items()
                            if rid != ignore_request_id
                        }
                    except (AttributeError, TypeError, ValueError):
                        blocker_ages = {}
                return {
                    "idle": False,
                    "reason": (
                        (
                            f"{n} active request id(s)"
                            f"{self._active_request_age_suffix(ignore_request_id, agent=agent)}"
                        )
                        if name_tasks
                        else "active request(s) in flight"
                    ),
                    "blocker": {
                        "scope": "requesting_agent",
                        "kind": "active_requests",
                        "count": n if name_tasks else None,
                        "oldest_age_seconds": (
                            max(blocker_ages.values()) if blocker_ages else None
                        ),
                    },
                }

        bg_tasks = getattr(agent, "_background_tasks", None)
        if bg_tasks is not None:
            any_surface_seen = True
            try:
                # Exclude fire-and-forget infrastructure bookkeeping
                # (signal_log: writes). It churns continuously under
                # heartbeat/scheduler load and would otherwise keep the
                # agent "busy" forever, never letting an idle restart
                # through (#1626). Real agent work (e.g. signal_dispatch:)
                # still counts.
                alive = [
                    t for t in bg_tasks
                    if not t.done() and not _is_infra_background_task(t)
                ]
            except (TypeError, AttributeError):
                alive = []
            if alive:
                ages = [
                    age for age in (
                        _task_age_seconds(task, time.monotonic()) for task in alive
                    )
                    if age is not None
                ]
                detail = (
                    f": {_describe_background_tasks(alive)}"
                    if name_tasks else ""
                )
                return {
                    "idle": False,
                    "reason": (
                        f"{len(alive)} background task(s) in flight{detail}"
                        if name_tasks
                        else "background task(s) in flight"
                    ),
                    "blocker": {
                        "scope": "requesting_agent",
                        "kind": "background_tasks",
                        "count": len(alive) if name_tasks else None,
                        "oldest_age_seconds": (
                            max(ages) if name_tasks and ages else None
                        ),
                        "summary": (
                            _describe_background_tasks(alive)
                            if name_tasks else None
                        ),
                    },
                }

        if not any_surface_seen:
            # No introspection available — conservatively defer. The
            # operator can still force progress with the timeout
            # policy.
            return {
                "idle": False,
                "reason": "no idleness introspection on agent",
                "blocker": {
                    "scope": "requesting_agent",
                    "kind": "unknown",
                    "count": None,
                    "oldest_age_seconds": None,
                },
            }
        return {"idle": True, "reason": "", "blocker": None}

    def _active_request_age_suffix(
        self, ignore_request_id: str = "", agent: Any = None,
    ) -> str:
        """Append the oldest active-request age to a busy deferral reason.

        Observability for #1558: when a restart defers on ``agent busy``,
        the operator can see how old the in-flight request markers are
        relative to the staleness sweep window, so a near-stale id is
        visible before it ages out. The requester's own turn is excluded
        so the reported age reflects only the requests still blocking the
        restart (#1561).
        """
        if agent is None:
            agent = self.agent
        ages_fn = getattr(agent, "active_request_ages", None)
        if not callable(ages_fn):
            return ""
        try:
            ages = ages_fn()
        except Exception:
            return ""
        ages = {
            rid: age for rid, age in ages.items() if rid != ignore_request_id
        }
        if not ages:
            return ""
        oldest = max(ages.values())
        return (
            f"; oldest {int(oldest)}s of "
            f"{STALE_ACTIVE_REQUEST_SECONDS}s stale window"
        )

    @classmethod
    def _request_aged_past_timeout(
        cls, req, database_now: datetime,
    ) -> bool:
        """Has the request sat in pending/approved longer than 5 min?"""

        age = cls._request_age_seconds(req, database_now)
        return age is not None and age > 300

    @staticmethod
    def _age_seconds(value: Any, database_now: datetime) -> Optional[float]:
        """Return a non-negative UTC age for one persisted ISO timestamp."""

        try:
            requested = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
        if requested.tzinfo is None:
            requested = requested.replace(tzinfo=timezone.utc)
        return max(0.0, (database_now - requested).total_seconds())

    @classmethod
    def _request_age_seconds(
        cls, req, database_now: datetime,
    ) -> Optional[float]:
        """Age since the restart request was filed."""

        return cls._age_seconds(getattr(req, "requested_at", ""), database_now)

    @classmethod
    def _deferral_age_seconds(
        cls, req, database_now: datetime,
    ) -> Optional[float]:
        """Age of the current uninterrupted busy interval, if one exists."""

        return cls._age_seconds(
            getattr(req, "first_blocked_at", ""), database_now
        )

    async def _handle_update_then_restart(
        self, req,
    ) -> Optional[Dict[str, Any]]:
        """Run the request's update profile before its restart.

        Called while the row is in ``updating`` (set by the caller before
        the slow update begins). Returns ``None`` when the update
        succeeded and the caller should proceed to spawn the restart.
        Returns a ``{request_id, reason}`` dict when the request was
        handled here — moved back to ``pending`` (retryable) or to
        ``rejected`` (terminal) — and the caller should NOT restart.
        """
        now = lambda: datetime.now(timezone.utc).isoformat()  # noqa: E731

        profile = get_update_profile(req.update_profile)
        # Defensive re-validation: a row could have been inserted
        # outside request_restart. Unknown profile / bad inputs are
        # unsafe and unfixable by retry → terminal reject.
        if profile is None:
            await update_status(
                self._db, req.id,
                status="rejected",
                status_reason=(
                    f"unknown update profile {req.update_profile!r}"
                ),
                completed_at=now(),
                expected_current_status="updating",
                expected_authority_signature=req.authority_signature,
            )
            return {
                "request_id": req.id,
                "reason": f"rejected: unknown update profile "
                          f"{req.update_profile!r}",
            }
        if not is_valid_target_ref(req.update_target_ref) or \
                not repo_is_git_checkout(req.update_repo_path):
            await update_status(
                self._db, req.id,
                status="rejected",
                status_reason=(
                    "invalid update target_ref/repo_path: "
                    f"ref={req.update_target_ref!r} "
                    f"repo={req.update_repo_path!r}"
                ),
                completed_at=now(),
                expected_current_status="updating",
                expected_authority_signature=req.authority_signature,
            )
            return {
                "request_id": req.id,
                "reason": "rejected: invalid update target_ref/repo_path",
            }

        update = await self._run_update(req, profile)
        try:
            await record_update_log(
                self._db, req.id, json.dumps(update),
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "restart_coordinator: failed to persist update log for "
                "%s: %s", req.id, e,
            )

        if not update["ok"]:
            # Fetch/checkout/install failed before any restart. Leave the
            # request retryable — the next poll re-runs the idempotent
            # profile — with a clear reason naming the failed step.
            await update_status(
                self._db, req.id,
                status="pending",
                status_reason=(
                    f"update failed at step {update.get('failed_step')!r}; "
                    "left retryable (see update_log)"
                ),
                expected_current_status="updating",
                expected_authority_signature=req.authority_signature,
            )
            return {
                "request_id": req.id,
                "reason": (
                    f"update failed at step {update.get('failed_step')!r}; "
                    "retryable"
                ),
            }
        return None

    async def _run_update(self, req, profile) -> Dict[str, Any]:
        """Execute a profile's update steps, capturing each outcome.

        Returns a JSON-serialisable audit dict: per-step results, the
        resolved commit, and the migration outcome. Stops at the first
        mutating step that fails.
        """
        steps = profile.build_steps(
            repo_path=req.update_repo_path,
            target_ref=req.update_target_ref,
            allow_migrations=bool(req.update_allow_migrations),
        )
        results: List[Dict[str, Any]] = []
        resolved_ref = ""
        ok = True
        failed_step: Optional[str] = None
        for step in steps:
            outcome = await self._run_update_step(step)
            results.append(outcome)
            if step.name == "resolve_ref" and outcome.get("ok"):
                resolved_ref = (outcome.get("stdout_tail") or "").strip()
            if not outcome.get("ok") and not (
                step.read_only or step.allow_failure
            ):
                ok = False
                failed_step = step.name
                break

        if not profile.supports_migrations:
            migration = {
                "ran": False,
                "reason": (
                    f"profile {profile.name!r} defines no explicit "
                    "migration step; sovereign schema migrates additively "
                    "on the next boot"
                ),
            }
        elif not req.update_allow_migrations:
            migration = {
                "ran": False,
                "reason": "allow_migrations=false on the request",
            }
        else:
            migration = {"ran": True, "reason": "profile-defined migration"}

        return {
            "ok": ok,
            "profile": profile.name,
            "repo_path": req.update_repo_path,
            "target_ref": req.update_target_ref,
            "resolved_ref": resolved_ref,
            "allow_migrations": bool(req.update_allow_migrations),
            "steps": results,
            "migration": migration,
            "failed_step": failed_step,
        }

    async def _run_update_step(self, step) -> Dict[str, Any]:
        """Run one allowlisted argv step; capture rc + truncated output.

        Uses ``create_subprocess_exec`` (argv list, never a shell) so a
        crafted ref/path can never inject a command.
        """
        if getattr(step, "native", None) == "reattach_branch":
            return await self._reattach_branch_step(step)
        try:
            proc = await asyncio.create_subprocess_exec(
                *step.argv,
                cwd=step.cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            rc = proc.returncode
        except Exception as e:
            return {
                "step": step.name,
                "argv": list(step.argv),
                "returncode": None,
                "ok": False,
                "error": str(e),
                "stdout_tail": "",
                "stderr_tail": "",
            }
        return {
            "step": step.name,
            "argv": list(step.argv),
            "returncode": rc,
            "ok": rc == 0,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        }

    async def _reattach_branch_step(self, step) -> Dict[str, Any]:
        """Native reattach: land the local branch on the fetched commit.

        Runs after the profile's ``checkout --detach FETCH_HEAD``. A single
        argv command cannot express the required guard — a name can exist
        as BOTH a tag and a branch, and ``git fetch <name>`` lands on the
        TAG commit, so attaching to ``origin/<name>`` could install a
        different commit than the one fetched (codex P2 on the reattach
        change). Intent is decided from what the fetch actually selected —
        the ``FETCH_HEAD`` file records ``tag '<ref>'`` / ``branch
        '<ref>'`` per fetched ref — never from local tag state (a stale
        local tag shadowing a branch must not force tag intent, codex
        round-2). Plain plumbing throughout (each call
        ``create_subprocess_exec``, never a shell):

        1. fetch selected ``tag '<ref>'``    → skip, stay detached.
        2. fetch selected no ``branch '<ref>'`` (sha target) → skip.
        3. ``origin/<ref>`` != FETCH_HEAD    → skip (defensive).
        4. otherwise ``checkout -B <ref> FETCH_HEAD`` and set the branch
           upstream to ``origin/<ref>`` — a NEW local branch without
           ``@{u}`` would fail the next ``kestrel update``'s bare
           ``git pull --ff-only`` (codex round-2).

        Skips report ``ok=True`` with the reason in ``stdout_tail``; the
        step is additionally ``allow_failure`` so even unexpected git
        errors never abort the update.
        """
        repo, ref = step.native_args
        # A caller may pass the fully-qualified form (refs/heads/main —
        # accepted by is_valid_target_ref); FETCH_HEAD records the SHORT
        # name ("branch 'main'"), and checkout/upstream want it too
        # (codex round-3).
        if ref.startswith("refs/heads/"):
            ref = ref[len("refs/heads/"):]

        async def _git(*args: str):
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", repo, *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return proc.returncode, _tail(stdout), _tail(stderr)

        def _outcome(ok: bool, rc, out: str, err: str = "") -> Dict[str, Any]:
            return {
                "step": step.name,
                "argv": list(step.argv),
                "returncode": rc,
                "ok": ok,
                "stdout_tail": out,
                "stderr_tail": err,
            }

        try:
            # What did the fetch select for <ref>? FETCH_HEAD lines look
            # like: "<sha>\t(not-for-merge)?\tbranch 'x' of <url>".
            rc, fetch_head_rel, _err = await _git(
                "rev-parse", "--git-path", "FETCH_HEAD"
            )
            if rc != 0:
                return _outcome(
                    True, 0, "skip: no FETCH_HEAD; staying detached"
                )
            fh = Path(fetch_head_rel.strip())
            if not fh.is_absolute():
                fh = Path(repo) / fh
            try:
                fetch_lines = fh.read_text().splitlines()
            except OSError:
                return _outcome(
                    True, 0, "skip: unreadable FETCH_HEAD; staying detached"
                )
            # Only the FOR-MERGE line(s) (empty second tab-field) name the
            # ref the fetch selected for the requested refspec; ``--tags``
            # also writes incidental ``not-for-merge`` tag lines which
            # must not decide intent (codex round-4: an origin tag merely
            # sharing the branch's short name would otherwise force a
            # skip even though the branch was selected).
            selected = []
            for line in fetch_lines:
                parts = line.split("\t")
                if len(parts) >= 3 and not parts[1].strip():
                    selected.append(parts[2])
            if any(f"tag '{ref}'" in desc for desc in selected):
                return _outcome(
                    True, 0,
                    f"skip: fetch selected tag {ref!r}; staying detached",
                )
            if not any(f"branch '{ref}'" in desc for desc in selected):
                return _outcome(
                    True, 0,
                    f"skip: fetch selected no branch {ref!r}; "
                    "staying detached",
                )
            rc, branch_sha, _err = await _git(
                "rev-parse", "--verify", "--quiet",
                f"refs/remotes/origin/{ref}^{{commit}}",
            )
            if rc != 0:
                return _outcome(
                    True, 0,
                    f"skip: no origin branch {ref!r}; staying detached",
                )
            rc, fetch_sha, _err = await _git(
                "rev-parse", "--verify", "FETCH_HEAD^{commit}"
            )
            if rc != 0 or branch_sha.strip() != fetch_sha.strip():
                return _outcome(
                    True, 0,
                    f"skip: origin/{ref} != FETCH_HEAD; staying detached",
                )
            rc, out, err = await _git("checkout", "-B", ref, fetch_sha.strip())
            if rc != 0:
                return _outcome(False, rc, out, err)
            rc, up_out, up_err = await _git(
                "branch", f"--set-upstream-to=refs/remotes/origin/{ref}", ref
            )
            return _outcome(rc == 0, rc, out + "\n" + up_out, err + up_err)
        except Exception as e:
            return {
                "step": step.name,
                "argv": list(step.argv),
                "returncode": None,
                "ok": False,
                "error": str(e),
                "stdout_tail": "",
                "stderr_tail": "",
            }

    def _spawn_restart_subprocess(self) -> subprocess.Popen:
        """Spawn a detached ``kestrel restart`` subprocess.

        ``start_new_session=True`` dissociates the child from the
        Kestrel host's process group so the restart survives our
        impending shutdown. ``close_fds=True`` ensures we leak no
        file descriptors into the new session.

        Returns the handle so the caller can verify the child actually stayed
        up. ``Popen`` only raises when the binary cannot be exec'd at all — a
        child that starts and immediately dies raises nothing, and discarding
        the handle made that outcome indistinguishable from success (#2667).

        ``stderr`` goes to a FILE, not ``DEVNULL`` and deliberately not a pipe.
        DEVNULL threw away the only evidence of why a dispatch failed. A pipe
        would be worse than either: this child must OUTLIVE us, and a pipe's
        read end dies with us, so a successful restart would leave the child
        taking EPIPE on its next stderr write — we would be breaking the very
        restart we are trying to perform. A chatty child would also block on a
        full pipe buffer that nobody is reading. A file has neither problem
        and still survives for us to read.
        """
        cmd: List[str]
        kestrel_bin = shutil.which("kestrel")
        if kestrel_bin:
            cmd = [kestrel_bin, "restart"]
        else:
            cmd = [sys.executable, "-m", "kestrel_sovereign.cli", "restart"]
        logger.info(
            "restart_coordinator: spawning detached restart %s", cmd,
        )
        stderr_file = tempfile.NamedTemporaryFile(
            prefix="kestrel-restart-", suffix=".err", delete=False,
        )
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            # Our copy of the descriptor is not needed once the child holds
            # one; the path is what we read back.
            stderr_file.close()
        # Carried on the handle so the watchdog can find it without threading
        # a second value through every call site.
        proc._kestrel_stderr_path = stderr_file.name  # type: ignore[attr-defined]
        logger.info(
            "restart_coordinator: restart subprocess pid=%s (stderr: %s)",
            proc.pid, stderr_file.name,
        )
        return proc

    def _restart_dispatch_failure(
        self, proc: subprocess.Popen,
    ) -> Optional[str]:
        """Why the restart dispatch failed, or ``None`` if it looks healthy.

        A successful restart kills THIS process, so still being alive while
        the child has already exited means the restart did not happen.

        Only an integer exit status counts as evidence. Anything else —
        a still-running child, or a handle that cannot report a status —
        returns ``None``: claiming a failure we cannot demonstrate would
        bounce a restart that is actually in flight.
        """
        try:
            returncode = proc.poll()
        except Exception:  # pragma: no cover - defensive
            return None
        if not isinstance(returncode, int):
            return None
        # The child is gone and we are not. Read what it said on the way out.
        detail = self._read_restart_stderr_tail(proc)
        reason = (
            f"restart subprocess (pid {proc.pid}) exited {returncode} "
            "without restarting the host"
        )
        return f"{reason}: {detail}" if detail else reason

    @staticmethod
    def _sweep_orphaned_restart_stderr(max_age_seconds: int = 86400) -> int:
        """Delete restart stderr files left behind by SUCCESSFUL restarts.

        The failure path cleans up its own file when it reads the tail. The
        success path cannot: the restart kills this process mid-flight, so the
        file is orphaned by definition. Without a sweep that is one small file
        per restart, forever. Best-effort — a restart must never fail because
        housekeeping did.
        """
        removed = 0
        cutoff = time.time() - max_age_seconds
        try:
            candidates = glob.glob(
                os.path.join(tempfile.gettempdir(), "kestrel-restart-*.err")
            )
        except OSError:  # pragma: no cover - defensive
            return 0
        for path in candidates:
            try:
                if os.path.getmtime(path) < cutoff:
                    os.unlink(path)
                    removed += 1
            except OSError:
                continue
        if removed:
            logger.info(
                "restart_coordinator: swept %d orphaned restart stderr "
                "file(s)", removed,
            )
        return removed

    @staticmethod
    def _read_restart_stderr_tail(proc) -> str:
        """Last stderr line of a dead restart child, and clean up its file."""
        path = getattr(proc, "_kestrel_stderr_path", None)
        if not path:
            return ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                # Bounded: a failing child can be arbitrarily chatty and this
                # string ends up in a status reason and a log line.
                lines = fh.read(_OUTPUT_TAIL_CHARS).strip().splitlines()
            return lines[-1] if lines else ""
        except OSError:
            return ""
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _arm_restart_dispatch_watch(self, proc, request_id: str) -> bool:
        """Start the dispatch watchdog, if this host can carry one.

        A host with no background-task machinery (embedded runtime, test
        double) cannot supervise the child. That must not fail the restart
        itself — it only means this row's recovery falls to
        ``_reconcile_stranded_executing_rows``, which needs no task at all.
        Returns whether the watch was armed.
        """
        if not callable(getattr(self.agent, "_track_background_task", None)):
            logger.debug(
                "restart_coordinator: no background-task support; restart "
                "dispatch for %s falls back to the stranded-row sweep",
                request_id,
            )
            return False
        self._track_owned_background_task(
            self._watch_restart_dispatch(proc, request_id),
            name=f"restart_dispatch_watch:{request_id}",
        )
        return True

    async def _watch_restart_dispatch(self, proc, request_id: str) -> None:
        """Recover the row if the detached restart dies instead of restarting.

        Runs as a background task rather than inline: the coordinator tick
        must not block for the grace window, and on the happy path this
        process is killed mid-wait and the task simply never finishes.

        Without this the failure had no record at all — ``Popen`` returning is
        not evidence the restart happened, so the row sat ``executing``
        forever with ``completed_at`` null and no error event, while the host
        kept running old code with the update's new dependencies already
        installed underneath it (#2667).
        """
        # POLL to the deadline rather than checking once at the end. The
        # realistic failure — ``os.kill`` refused (EPERM: host under another
        # uid, pid-file mismatch) — has ``cmd_stop`` burn ~5.5s on
        # SIGTERM/poll/SIGKILL before ``cmd_start`` fails on the port, so the
        # child dies around 7-10s. A single check at 10.0s is a coin flip
        # against that, and losing it means the watchdog declares the dispatch
        # healthy and recovery silently falls through to the 600s reconciler.
        # We are alive for the whole window by definition, so polling is free.
        grace = getattr(
            self, "_restart_dispatch_grace", RESTART_DISPATCH_GRACE_SECONDS,
        )
        deadline = time.monotonic() + grace
        reason = None
        while True:
            reason = self._restart_dispatch_failure(proc)
            if reason is not None:
                break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(min(_DISPATCH_POLL_SECONDS, grace))
        if reason is None:
            return
        logger.error("restart_coordinator: %s", reason)
        row = await get_request(self._db, request_id)
        if row is None or row.status != "executing":
            return

        # Returning the row to ``pending`` means the next tick re-dispatches.
        # For a permanently broken restart (missing binary, a uid that cannot
        # signal the host) that turns "stuck forever" into "flaps forever":
        # a doomed subprocess every minute, each with its own status event and
        # error log. After a few identical failures in this boot, stop and say
        # so terminally instead of retrying into the same wall.
        attempts = self._dispatch_failures.get(request_id, 0) + 1
        self._dispatch_failures[request_id] = attempts
        give_up = attempts >= MAX_RESTART_DISPATCH_ATTEMPTS
        next_status = "rejected" if give_up else "pending"
        next_reason = (
            f"{reason}; giving up after {attempts} dispatch attempts this boot"
            if give_up else reason
        )

        moved = await update_status(
            self._db, request_id,
            status=next_status,
            status_reason=next_reason,
            completed_at=(
                datetime.now(timezone.utc).isoformat() if give_up else None
            ),
            expected_current_status="executing",
        )
        if not moved:
            return
        self._executing_since.pop(request_id, None)
        if give_up:
            self._dispatch_failures.pop(request_id, None)
            logger.error(
                "restart_coordinator: rejecting restart %s after %d failed "
                "dispatch attempts", request_id, attempts,
            )
        await self._emit_status_event(
            row, state=next_status,
            **(
                {"status_reason": next_reason} if give_up
                else {"deferral_reason": next_reason}
            ),
        )

    async def _reconcile_stranded_executing_rows(self) -> List[str]:
        """Return rows this boot stranded in ``executing`` to ``pending``.

        A row stamped with THIS process's boot id is a restart that was
        dispatched but never happened — a restart that HAD happened would be
        running a different process with a different id. Past
        ``STALE_EXECUTING_SECONDS`` it is not "still in flight", it is stuck.

        Before this, nothing could move such a row: the coordinator scans only
        pending/approved, and ``cancel_restart_request`` refuses executing
        rows. It sat there permanently, and worse, the NEXT unrelated restart
        would see a row whose ``executing_boot_id`` no longer matches the new
        process and terminalize it as "completed — post-restart sweep observed
        agent re-init", reporting success for a restart that never ran (#2667).

        Returns the ids reset, for the caller's audit trail.
        """
        if self._db is None:
            return []
        reset: List[str] = []
        now = time.monotonic()
        for row in await list_requests(self._db, status="executing"):
            if row.executing_boot_id != _PROCESS_BOOT_ID:
                # A prior boot's row: the restart provably happened, so this
                # belongs to the post-restart wake sweep, not here.
                continue
            # Age is measured from when THIS process crossed the row into
            # ``executing``, not from ``requested_at`` — a row that queued for
            # hours before dispatch would otherwise look instantly stale.
            started = self._executing_since.get(row.id)
            if started is not None and (now - started) < STALE_EXECUTING_SECONDS:
                continue
            if started is None:
                # Stamped by this process but absent from the in-flight map:
                # the dispatch that owned it is gone (feature reload, cancelled
                # task) and nothing is waiting on it.
                #
                # This branch must still wait. ``_PROCESS_BOOT_ID`` is
                # module-scoped but the map is per-INSTANCE, so a feature
                # reload inside the same process starts with an empty map and
                # would otherwise take this branch — with no age check at all —
                # and reset a dispatch the previous instance started moments
                # ago. Requiring this instance to have been up for the same
                # window closes that, since a genuinely stranded row is going
                # nowhere and can wait.
                if (now - self._instance_started_at) < STALE_EXECUTING_SECONDS:
                    continue
                reason = (
                    "restart row is executing under this process with no "
                    "dispatch in flight; the restart did not happen"
                )
            else:
                reason = (
                    "restart dispatched but this process is still running "
                    f"after {STALE_EXECUTING_SECONDS}s; the restart did not "
                    "happen"
                )
            moved = await update_status(
                self._db, row.id,
                status="pending",
                status_reason=reason,
                expected_current_status="executing",
            )
            if not moved:
                continue
            self._executing_since.pop(row.id, None)
            logger.error(
                "restart_coordinator: recovered stranded executing row %s "
                "(%s)", row.id, reason,
            )
            await self._emit_status_event(
                row, state="pending", deferral_reason=reason,
            )
            reset.append(row.id)
        return reset

    async def _reset_interrupted_updates(self) -> None:
        """Reset rows stuck in ``updating`` back to ``pending`` for retry.

        A row reaches ``updating`` only while an ``update_then_restart``
        profile runs (git fetch/checkout + ``uv sync``). At boot, any
        such row is necessarily a leftover from a previous process whose
        host went down mid-update — the update never finished and we
        never restarted into the new code. Unlike an ``executing`` row,
        this must NOT be reported as a completed restart (its
        ``resolved_ref`` would be empty/stale and the checkout may be
        half-applied). The profile steps are idempotent, so the safe
        recovery is to make the request retryable; the next coordinator
        poll re-runs the update cleanly.
        """
        if self._db is None:
            return
        agent_id = getattr(self.agent, "did", "") or ""
        if not agent_id:
            return
        stuck = await list_requests(
            self._db, status="updating", agent_id=str(agent_id),
        )
        for row in stuck:
            await update_status(
                self._db, row.id,
                status="pending",
                status_reason=(
                    "host restarted mid-update before the restart could "
                    "run; update incomplete — reset to pending for retry"
                ),
                expected_current_status="updating",
            )

    async def _reap_post_restart_rows(self) -> List[asyncio.Task[Any]]:
        """Sweep ``executing`` rows this agent filed and wake the
        requesting agent with one ``restart.completed`` COGNITION signal
        per row.

        A prior-boot ``executing`` row is terminalized to ``completed`` (with
        ``completed_at``) IMMEDIATELY — the restart provably happened, since a
        different process is running this sweep — and only THEN is the wake
        dispatched (#1819). This is the fix for the inconsistency where the wake
        turn observed the row still ``executing`` with no ``completed_at`` while
        the wake payload claimed completion, and the ``completed`` status bubble
        appeared AFTER the wake message.

        Wake delivery is tracked separately via the durable ``wake_delivered``
        flag, preserving the #1796 retry guarantee without overloading the row's
        status: the wake is (re)dispatched while ``wake_delivered`` is 0, and the
        flag is set only once the COGNITION dispatch returns ``Status.OK`` (NOT on
        mere enqueue, NOT on ``COALESCED``). An undelivered wake leaves the row
        ``completed`` with ``wake_delivered=0`` so the next sweep (the
        ``restart_coordinator`` cron tick, or the next boot) retries it.

        This sweep runs both at ``initialize`` (post-reboot) AND on every
        live ``restart_coordinator`` cron tick (the retry backstop). To
        keep the cron tick from falsely completing a restart that has NOT
        happened — a row this same process just crossed to ``executing``
        whose detached ``kestrel restart`` is still in flight or failed to
        kill the parent — only rows stamped by a DIFFERENT process are
        woken. A row stamped with the current ``_PROCESS_BOOT_ID`` stays
        visibly ``executing`` (a real restart would have replaced this
        process and given it a fresh id).
        """
        if self._db is None:
            return []
        agent_id = getattr(self.agent, "did", "") or ""
        if not agent_id:
            return []
        needing_wake = await list_requests_needing_wake(
            self._db, agent_id=str(agent_id),
        )
        if not needing_wake:
            return []

        dispatcher = getattr(self.agent, "dispatcher", None)
        dispatcher_usable = (
            dispatcher is not None
            and hasattr(dispatcher, "enqueue_signal")
        )
        wake_tasks: List[asyncio.Task[Any]] = []
        for row in needing_wake:
            # A row stamped by THIS process is a restart still in flight
            # (or a detached restart that failed to kill the parent), not
            # one that already completed — skip it so a failed restart is
            # not silently masked as success (#1796).
            if row.executing_boot_id == _PROCESS_BOOT_ID:
                continue
            # Terminalize FIRST (the restart is provably done), so the wake
            # turn and the completed bubble both see a consistent completed
            # row (#1819). Already-completed rows here are wake retries — skip
            # re-terminalizing (and re-emitting the bubble).
            if row.status == "executing":
                terminalized = await self._terminalize_completed(
                    row, datetime.now(timezone.utc).isoformat(),
                )
                if not terminalized:
                    # The durable ``executing`` → ``completed`` write did not
                    # land (a concurrent sweep beat us to it). Re-read the
                    # authoritative row and only deliver the wake once it is
                    # genuinely ``completed`` — never fire a completion wake
                    # against a row still ``executing`` in the DB (#1801).
                    fresh = await get_request(self._db, row.id)
                    if fresh is None or fresh.status != "completed":
                        continue
                    row = fresh
            wake_task = await self._deliver_restart_completed(
                row, str(agent_id), dispatcher, dispatcher_usable,
            )
            if wake_task is not None:
                wake_tasks.append(wake_task)
        return wake_tasks

    async def _deliver_restart_completed(
        self, row, agent_id: str, dispatcher, dispatcher_usable: bool,
    ) -> Optional[asyncio.Task[Any]]:
        """Dispatch one ``restart.completed`` wake for an already-terminalized
        row and gate ``wake_delivered`` on the wake actually landing (#1819).

        The row is ``completed`` before this runs (the sweep terminalized it),
        so this only concerns NOTIFYING the agent. ``wake_delivered`` is flipped
        to 1 once the COGNITION dispatch returns ``Status.OK``; until then the
        row is re-swept and the wake retried.
        """
        # The wake must report when the restart actually COMPLETED, not when
        # this (possibly retried) dispatch runs — otherwise a cron-retried wake
        # would claim a later "landed at" time than the row/status event (#1819
        # codex P3). The sweep terminalizes before dispatch, so completed_at is
        # set; fall back to now only if it's somehow missing.
        completed_at = getattr(row, "completed_at", None) or datetime.now(
            timezone.utc
        ).isoformat()

        # No dispatcher to wake (headless host, test stub): nothing to deliver
        # to and no point retrying forever, so mark the wake delivered.
        if not dispatcher_usable:
            await self._mark_wake_delivered(row)
            return None

        # A supervisor from this process is already awaiting this row's
        # wake — don't enqueue a duplicate (avoids a wake storm while a
        # long cognition turn is still running).
        if row.id in self._inflight_restart_acks:
            return None

        # Stamp BEFORE the signal is handed over. ``enqueue_signal`` starts
        # the dispatch immediately, so every await after it — this write
        # included — is a point at which the woken turn may already be running
        # and reading this row. A stamp written afterwards races the very turn
        # it exists to inform: the same "recorded too late to be read" shape
        # as ``wake_delivered``, which is the whole of #2774.
        #
        # It therefore records the ATTEMPT. An enqueue that raises below
        # leaves the stamp standing, and that is the honest reading: '' still
        # means this sweep never tried to wake the row, which is the negative
        # evidence the ticket needs, while a failed handoff is logged and
        # leaves the row undelivered for the next sweep to retry.
        await self._mark_wake_dispatched(row)

        try:
            from kestrel_sovereign.signals.sources.restart import (
                build_signal_for_restart_completed,
            )
            signal = build_signal_for_restart_completed(
                row, target_agent=str(agent_id), completed_at=completed_at,
            )
            handle = dispatcher.enqueue_signal(signal)
            if asyncio.iscoroutine(handle):
                handle = await handle
        except Exception as e:
            # Enqueue raised synchronously — leave wake_delivered=0 so a
            # later sweep retries (codex P2 on PR #1512 round 1).
            logger.warning(
                "restart sweep: failed to enqueue signal for %s: %s; "
                "wake stays undelivered for next-sweep retry", row.id, e,
            )
            return None

        waiter = getattr(handle, "wait", None)
        if not callable(waiter):
            # Legacy/stub dispatcher whose enqueue_signal returns no
            # awaitable handle — the signal was accepted onto the queue, so
            # treat the wake as delivered.
            await self._mark_wake_delivered(row)
            return None

        # Delivery-gated: supervise the wake and only flag wake_delivered
        # once the COGNITION dispatch actually lands.
        self._inflight_restart_acks.add(row.id)
        return self._spawn_ack_supervisor(row, handle)

    def _spawn_ack_supervisor(self, row, handle) -> asyncio.Task[Any]:
        """Await the ``restart.completed`` dispatch and flag ``wake_delivered``
        only once the wake has actually been DELIVERED (``Status.OK``). A failed
        or dropped wake leaves ``wake_delivered=0`` so a later sweep retries it
        (#1796/#1819). The row is already ``completed`` either way.

        ``COALESCED`` is deliberately NOT treated as delivered: the
        coalescing key is recorded BEFORE ``process_input`` runs, so a
        wake that fails inside the resuming turn still suppresses a fast
        retry as ``COALESCED``. Acking on that would falsely mark a
        wake that never produced a turn — so a coalesced retry just leaves
        ``wake_delivered=0`` until the coalescing window elapses and a real
        re-dispatch lands ``OK``.
        """

        async def _await_and_ack() -> None:
            from kestrel_sdk.signals import Status

            try:
                result = await handle.wait()
            except Exception as e:
                logger.warning(
                    "restart sweep: restart.completed wake for %s raised "
                    "(%s); wake stays undelivered for retry", row.id, e,
                )
                return
            finally:
                self._inflight_restart_acks.discard(row.id)

            status = getattr(result, "status", None)
            if status is not Status.OK:
                logger.warning(
                    "restart sweep: restart.completed wake for %s did not "
                    "land (status=%s); wake stays undelivered for retry",
                    row.id, getattr(status, "value", status),
                )
                return
            await self._mark_wake_delivered(row)

        # FEATURE-owned so ``Feature.shutdown()`` (runtime disable / boot
        # rollback / soft disable) cancels this in-flight ack. The agent-global
        # task set is only reaped at FULL agent shutdown, so an agent-only task
        # would outlive a disabled feature and later flag ``wake_delivered``
        # against torn-down state (kestrel-sovereign#2522 P2). Still
        # agent-tracked underneath, so full shutdown reaps it too.
        return self._track_owned_background_task(
            _await_and_ack(), name=f"restart_completed_ack:{row.id}",
        )

    async def _terminalize_completed(self, row, now: str) -> bool:
        """Mark a swept restart row ``completed`` and mirror the COGNITION
        wake with a chat-visible status event (#1551). Returns ``True`` only
        when the durable ``executing`` → ``completed`` write actually landed.

        Called BEFORE the wake is dispatched (#1819): the restart provably
        finished (a different process is running the sweep), so the row reaches
        a consistent ``completed`` state — with ``completed_at`` — before the
        wake turn can observe it, and the ``completed`` bubble paints first.

        The ``expected_current_status="executing"`` guard can legitimately
        fail to update a row — a concurrent sweep (``on_agent_ready`` racing
        the first cron tick) already terminalized it. Honour that result
        (#1801): only mutate the in-memory row and emit the ``completed``
        status event when the write landed. Mutating/emitting on a failed
        write would deliver a ``restart.completed`` wake claiming a terminal
        state the durable row never reached, leaving the request stuck
        ``executing`` with no ``completed`` status event — the exact
        inconsistency reported in #1801.
        """
        ok = await update_status(
            self._db, row.id,
            status="completed",
            status_reason="post-restart sweep observed agent re-init",
            completed_at=now,
            expected_current_status="executing",
        )
        if not ok:
            return False
        row.completed_at = now
        row.status = "completed"
        await self._emit_status_event(
            row, state="completed",
            status_reason="post-restart sweep observed agent re-init",
        )
        return True

    async def _mark_wake_delivered(self, row) -> None:
        """Flag the post-restart wake as delivered so it isn't re-dispatched.

        A failure here is NOT cosmetic and is not logged at DEBUG. The wake has
        already been consumed, so leaving ``wake_delivered = 0`` makes every
        later sweep rediscover the row and re-emit a completion wake the agent
        has already handled — the one-per-minute storm in #2738 — while the
        row reports negative evidence that contradicts its own consumer
        (#2774). Both the raising case and the silent no-op (a write that
        matched no row) are surfaced so the condition leaves a record rather
        than only symptoms. #2660 documents 2,045 signal_log writes lost to
        SQLite lock contention, so this is not a hypothetical trigger.
        """
        request_id = getattr(row, "id", "?")
        try:
            landed = await mark_wake_delivered(self._db, row.id)
        except Exception as e:
            logger.warning(
                "restart sweep: failed to flag wake_delivered for %s: %s; "
                "until this write lands the next sweep will re-emit this "
                "completion wake", request_id, e,
            )
            return
        if not landed:
            logger.warning(
                "restart sweep: wake_delivered write for %s matched no row; "
                "until this write lands the next sweep will re-emit this "
                "completion wake", request_id,
            )
            return
        row.wake_delivered = True

    async def _mark_wake_dispatched(self, row) -> None:
        """Record the wake dispatch, before the signal is handed over (#2774).

        ``wake_delivered`` cannot answer "was a wake sent for this row" during
        the woken turn, because it is only set once that same turn returns OK.

        Best-effort by design: failing to record observability must never stop
        the wake itself, so this logs and continues. It does log, though —
        these columns exist to BE the record of a dispatch, and losing the
        write silently is the same asymmetry that made ``wake_delivered``
        untrustworthy in the first place.
        """
        request_id = getattr(row, "id", "?")
        dispatched_at = datetime.now(timezone.utc).isoformat()
        try:
            stamped = await mark_wake_dispatched(
                self._db, row.id,
                dispatched_at=dispatched_at,
                boot_id=_PROCESS_BOOT_ID,
            )
        except Exception as e:
            logger.warning(
                "restart sweep: failed to record wake dispatch for %s: %s; "
                "the wake still fires, but this dispatch leaves no record",
                request_id, e,
            )
            return
        if not stamped:
            logger.warning(
                "restart sweep: wake dispatch stamp for %s matched no row; "
                "the wake still fires, but this dispatch leaves no record",
                request_id,
            )
            return
        row.wake_dispatched_at = dispatched_at
        row.wake_dispatch_boot_id = _PROCESS_BOOT_ID
        # Keep the in-memory row consistent with the durable count: a stale 0
        # here is what ``to_public_dict`` and the status events built from this
        # object mid-sweep would publish.
        row.wake_dispatch_count = (getattr(row, "wake_dispatch_count", 0) or 0) + 1
