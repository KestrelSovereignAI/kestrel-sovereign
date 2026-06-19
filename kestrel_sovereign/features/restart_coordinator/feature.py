"""RestartCoordinatorFeature — durable, host-mediated restart requests.

Three @tool entry points (request / list / cancel) plus an ACTION-mode
``restart_coordinator`` cron entry that scans the durable table and
spawns a detached subprocess to actually restart Kestrel once safety
checks pass. After restart, ``initialize`` sweeps any in-flight
``executing`` rows owned by this agent into ``completed`` and emits a
``restart.completed`` signal so the requesting agent wakes.

The feature owns NO direct dependency on Talon or any other feature.
The safety checks are best-effort introspections of the agent's
existing public surfaces (``dispatcher.has_in_flight_signals``-style
checks gracefully degrade to "assume idle" if those surfaces aren't
present). The actual restart command is spawned via the CLI entry
point (``kestrel restart``) so the executor doesn't need to know the
runtime layout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.storage_access import resolve_feature_database

from .event_store import (
    ensure_restart_status_events_table,
    list_recent_events_for_history,
    record_event as record_status_event,
)
from .events import EVENT_NAME, build_restart_status_event
from .store import (
    KNOWN_OPERATIONS,
    KNOWN_POLICIES,
    KNOWN_URGENCIES,
    PENDING_STATES,
    TERMINAL_STATES,
    ensure_restart_requests_table,
    get_request,
    insert_request,
    list_requests,
    list_requests_needing_wake,
    mark_wake_delivered,
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
# hold off an idle restart (#1626). Two shapes both wedged
# ``idle_agents_only`` forever by being counted as "busy":
#   - ``signal_log:`` — fire-and-forget log writes that complete in well
#     under a second but are minted continuously by heartbeats/scheduler
#     ticks, so one is almost always alive when the idle check runs.
#   - ``a2a_question_expiry_sweep`` — an intentionally permanent ``while
#     True`` maintenance daemon (peers feature) that never completes.
# Neither is user/signal work; real work (``signal_dispatch:*``) still
# defers a restart. The name is already stamped on the task at creation —
# it was just never read here. New long-lived/bookkeeping daemons must be
# named with a prefix listed here (or excluded from ``_background_tasks``).
_INFRA_TASK_PREFIXES = ("signal_log:", "a2a_question_expiry_sweep")


def _is_infra_background_task(task) -> bool:
    """True if ``task`` is fire-and-forget infrastructure bookkeeping that
    must not gate an idle restart (e.g. ``signal_log:*`` writes)."""
    try:
        name = task.get_name() or ""
    except Exception:
        return False
    return name.startswith(_INFRA_TASK_PREFIXES)


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
        self._db = resolve_feature_database(self.agent)
        if self._db is not None:
            try:
                await ensure_restart_requests_table(self._db)
                logger.info(
                    "RestartCoordinatorFeature: restart_requests table ready"
                )
            except Exception as e:
                logger.warning(
                    "RestartCoordinatorFeature: table init failed: %s", e,
                )
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
        if registry is not None and hasattr(registry, "register"):
            try:
                from kestrel_sovereign.signals.sources.restart import (
                    build_restart_completed_registration,
                )
                registry.register(build_restart_completed_registration())
            except Exception as e:
                logger.warning(
                    "RestartCoordinatorFeature: signal-source register "
                    "failed: %s", e,
                )

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

    async def on_agent_ready(self, agent=None) -> None:
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
            await self._reap_post_restart_rows()
        except Exception as e:  # never let readiness wiring break boot
            logger.warning("post-restart wake sweep on_agent_ready failed: %s", e)

    @tool(
        name="request_restart",
        description=(
            "File a durable restart request. The host coordinator "
            "evaluates safety and executes when conditions are met. "
            "Returns a request_id you can pass to list_restart_requests "
            "or cancel_restart_request.\n\n"
            "operation='restart_only' (default) restarts the current code "
            "and NEVER updates it. operation='update_then_restart' first "
            "runs an explicit, allowlisted update profile (e.g. "
            "'sovereign_local_uv_sync': git fetch + checkout target_ref + "
            "uv sync) against a local checkout, then restarts into the new "
            "code. Update mode requires update_profile and target_ref; "
            "repo_path defaults to the local Sovereign checkout. "
            "Updating/installing is always explicit and audited — it is "
            "never an implicit side effect of a plain restart."
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
        if urgency not in KNOWN_URGENCIES:
            return ToolResult.failed(
                f"urgency must be one of {sorted(KNOWN_URGENCIES)}; "
                f"got {urgency!r}",
                data={"created": False},
            )
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
        # post-restart wake lands in the SAME window (#1809). Prefer the agent's
        # authoritative per-turn ``_active_session_id`` (set by both the
        # streaming and non-streaming turn bodies from the effective session,
        # incl. the JSON-body session the primary chat path uses). Fall back to
        # the logging ``session_id_var`` (set only from a query param / header).
        # Empty for CLI/system-filed requests with no session — those wake
        # system-initiated, as before.
        origin_session_id = getattr(self.agent, "_active_session_id", "") or ""
        if not origin_session_id:
            try:
                from kestrel_sovereign.logging_config import session_id_var
                origin_session_id = session_id_var.get() or ""
            except Exception:
                origin_session_id = ""
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
            update_profile=(update_profile if operation == "update_then_restart"
                            else ""),
            update_allow_migrations=bool(allow_migrations),
            requester_request_id=str(requester_request_id),
            origin_session_id=origin_session_id,
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
            "List restart requests, optionally filtered by status "
            "(pending|approved|executing|completed|rejected|canceled)."
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
        rows = await list_requests(
            self._db, status=(status.strip() or None),
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
            "List recent restart_status lifecycle events for chat-"
            "history reload and the agent's pre-turn snapshot. Newest "
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
        rows = await list_recent_events_for_history(
            self._db,
            limit=limit_int,
            since=(since.strip() or None),
        )
        return ToolResult.ok(
            confirmation=f"{len(rows)} restart status event(s)",
            data={
                "count": len(rows),
                "events": [e.to_public_dict() for e in rows],
            },
        )

    @tool(
        name="cancel_restart_request",
        description=(
            "Cancel a still-pending restart request. Rows already "
            "executing/completed/rejected cannot be canceled."
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
        row = await get_request(self._db, request_id.strip())
        if row is None:
            return ToolResult.failed(
                f"No restart request with id {request_id!r}",
                data={"canceled": False, "request_id": request_id},
            )
        if row.status not in PENDING_STATES:
            return ToolResult.failed(
                f"Cannot cancel a request in state {row.status!r} — "
                f"only pending/approved can be canceled",
                data={
                    "canceled": False,
                    "request_id": request_id,
                    "current_status": row.status,
                },
            )
        ok = await update_status(
            self._db, row.id,
            status="canceled",
            status_reason=(reason.strip() or "canceled by agent"),
            completed_at=datetime.now(timezone.utc).isoformat(),
            expected_current_status=row.status,
        )
        if not ok:
            # Lost the race — someone else moved the row first.
            fresh = await get_request(self._db, row.id)
            current = fresh.status if fresh else "missing"
            return ToolResult.failed(
                f"Race: request transitioned to {current!r} before "
                f"cancel landed",
                data={
                    "canceled": False,
                    "request_id": request_id,
                    "current_status": current,
                },
            )
        await self._emit_status_event(
            row, state="canceled",
            status_reason=(reason.strip() or "canceled by agent"),
        )
        return ToolResult.ok(
            confirmation=f"Canceled restart request {row.id}",
            data={"canceled": True, "request_id": row.id},
        )

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
            decision = self._evaluate_safety(req)
            if not decision["safe"]:
                if decision.get("deferable", True):
                    deferred.append({
                        "request_id": req.id,
                        "reason": decision["reason"],
                    })
                    # Surface the deferred attempt + its reason (#1551).
                    await self._emit_status_event(
                        req, state="pending",
                        deferral_reason=decision["reason"],
                    )
                    continue
                # Hard reject.
                await update_status(
                    self._db, req.id,
                    status="rejected",
                    status_reason=decision["reason"],
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    expected_current_status=req.status,
                )
                await self._emit_status_event(
                    req, state="rejected", status_reason=decision["reason"],
                )
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
            moved = await update_status(
                self._db, req.id,
                status=initial_state,
                status_reason=(
                    "running update profile before restart"
                    if initial_state == "updating"
                    else "dispatched to detached restart subprocess"
                ),
                expected_current_status=req.status,
                # Stamp the live process only when crossing straight to
                # ``executing`` (#1796); an ``updating`` row has not yet
                # reached the restart, so it carries no boot stamp.
                executing_boot_id=(
                    _PROCESS_BOOT_ID if initial_state == "executing" else None
                ),
            )
            if not moved:
                deferred.append({
                    "request_id": req.id,
                    "reason": "lost race against another transition",
                })
                continue

            # Surface the transition out of pending — ``updating`` (update
            # profile running) or ``executing`` (restart dispatched) (#1551).
            await self._emit_status_event(req, state=initial_state)

            # update_then_restart: run the allowlisted update profile
            # against the local checkout BEFORE restarting. A failure
            # here records the audit log and decides retryable vs
            # terminal; only a clean update proceeds to the spawn.
            if req.operation == "update_then_restart":
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
                # Re-run the safety gate before the restart now that the
                # (possibly slow) update has completed.
                decision = self._evaluate_safety(req)
                if not decision["safe"]:
                    await update_status(
                        self._db, req.id,
                        status="pending",
                        status_reason=(
                            "update succeeded but agent became unsafe to "
                            f"restart: {decision['reason']}"
                        ),
                        expected_current_status="updating",
                    )
                    deferred.append({
                        "request_id": req.id,
                        "reason": (
                            "update ok; restart deferred — "
                            f"{decision['reason']}"
                        ),
                    })
                    await self._emit_status_event(
                        req, state="pending",
                        deferral_reason=(
                            "update ok; restart deferred — "
                            f"{decision['reason']}"
                        ),
                    )
                    continue
                # Update done and still safe — NOW cross into ``executing``
                # right before the spawn so the post-restart sweep
                # recognizes the restart we are about to perform.
                moved = await update_status(
                    self._db, req.id,
                    status="executing",
                    status_reason="update complete; dispatching restart",
                    expected_current_status="updating",
                    executing_boot_id=_PROCESS_BOOT_ID,
                )
                if not moved:
                    deferred.append({
                        "request_id": req.id,
                        "reason": "lost race after update before restart",
                    })
                    continue
                await self._emit_status_event(req, state="executing")

            try:
                self._spawn_restart_subprocess()
            except Exception as e:
                logger.error(
                    "restart_coordinator: spawn failed: %s", e,
                )
                await update_status(
                    self._db, req.id,
                    status="pending",
                    status_reason=f"spawn failed: {e}",
                    expected_current_status="executing",
                )
                await self._emit_status_event(
                    req, state="pending",
                    deferral_reason=f"spawn failed: {e}",
                )
                continue

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

    async def _emit_status_event(
        self,
        req,
        *,
        state: str,
        deferral_reason: str = "",
        status_reason: str = "",
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
        payload = build_restart_status_event(
            req,
            state=state,
            deferral_reason=deferral_reason,
            status_reason=status_reason,
            agent_did=str(agent_did),
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
                    agent_id=str(
                        getattr(req, "requested_by_agent", "") or agent_did
                    ),
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

    def _evaluate_safety(self, req) -> Dict[str, Any]:
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
            }
        if policy == "manual_only":
            return {
                "safe": False,
                "deferable": True,
                "reason": "policy=manual_only; awaiting explicit dispatch",
            }

        # The chat/agent turn that filed this request is itself an active
        # request marker. It must NOT block the restart it requested when
        # it is the only thing in flight — ignore the requester's own
        # marker for this specific row (#1561). Other active requests are
        # still respected so a busy agent stays protected.
        idle = self._agent_appears_idle(
            ignore_request_id=getattr(req, "requester_request_id", "") or "",
        )
        if idle["idle"]:
            return {"safe": True, "deferable": True, "reason": ""}

        if policy == "allow_busy_after_timeout":
            if self._request_aged_past_timeout(req):
                return {
                    "safe": True,
                    "deferable": True,
                    "reason": "timeout policy expired",
                }
            return {
                "safe": False,
                "deferable": True,
                "reason": (
                    f"agent busy ({idle['reason']}); waiting for "
                    f"timeout to elapse"
                ),
            }

        # idle_agents_only
        return {
            "safe": False,
            "deferable": True,
            "reason": f"agent busy ({idle['reason']})",
        }

    def _agent_appears_idle(
        self, ignore_request_id: str = "",
    ) -> Dict[str, Any]:
        """Idle check against the agent's own in-flight surface.

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
        any_surface_seen = False
        dispatcher = getattr(self.agent, "dispatcher", None)
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
                                "reason": f"dispatcher reports {attr}={val}",
                            }
                    except Exception:
                        # Treat introspection failure as "unknown" —
                        # the catch-all conservative fallback below
                        # handles it.
                        continue

        active_ids = getattr(self.agent, "_active_request_ids", None)
        if active_ids is not None:
            any_surface_seen = True
            # A finished/abandoned stream should have been cleared by the
            # endpoint's `finally` (`_cleanup_cancelled_request`). A client
            # disconnect or crashed generator can leave a request id
            # registered forever, permanently blocking `idle_agents_only`
            # restarts (#1558). Sweep ids older than the staleness window
            # before counting so a stale marker can never deadlock us.
            pruner = getattr(self.agent, "prune_stale_active_requests", None)
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
                return {
                    "idle": False,
                    "reason": (
                        f"{n} active request id(s)"
                        f"{self._active_request_age_suffix(ignore_request_id)}"
                    ),
                }

        bg_tasks = getattr(self.agent, "_background_tasks", None)
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
                return {
                    "idle": False,
                    "reason": f"{len(alive)} background task(s) in flight",
                }

        if not any_surface_seen:
            # No introspection available — conservatively defer. The
            # operator can still force progress with the timeout
            # policy.
            return {
                "idle": False,
                "reason": "no idleness introspection on agent",
            }
        return {"idle": True, "reason": ""}

    def _active_request_age_suffix(self, ignore_request_id: str = "") -> str:
        """Append the oldest active-request age to a busy deferral reason.

        Observability for #1558: when a restart defers on ``agent busy``,
        the operator can see how old the in-flight request markers are
        relative to the staleness sweep window, so a near-stale id is
        visible before it ages out. The requester's own turn is excluded
        so the reported age reflects only the requests still blocking the
        restart (#1561).
        """
        ages_fn = getattr(self.agent, "active_request_ages", None)
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

    @staticmethod
    def _request_aged_past_timeout(req) -> bool:
        """Has the request sat in pending/approved longer than 5 min?"""
        try:
            requested = datetime.fromisoformat(req.requested_at)
        except ValueError:
            return False
        now = datetime.now(timezone.utc)
        return (now - requested).total_seconds() > 300

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
            if not outcome.get("ok") and not step.read_only:
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

    def _spawn_restart_subprocess(self) -> None:
        """Spawn a detached ``kestrel restart`` subprocess.

        ``start_new_session=True`` dissociates the child from the
        Kestrel host's process group so the restart survives our
        impending shutdown. ``close_fds=True`` ensures we leak no
        file descriptors into the new session.
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
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

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

    async def _reap_post_restart_rows(self) -> None:
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
            return
        agent_id = getattr(self.agent, "did", "") or ""
        if not agent_id:
            return
        needing_wake = await list_requests_needing_wake(
            self._db, agent_id=str(agent_id),
        )
        if not needing_wake:
            return

        dispatcher = getattr(self.agent, "dispatcher", None)
        dispatcher_usable = (
            dispatcher is not None
            and hasattr(dispatcher, "enqueue_signal")
        )
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
            await self._deliver_restart_completed(
                row, str(agent_id), dispatcher, dispatcher_usable,
            )

    async def _deliver_restart_completed(
        self, row, agent_id: str, dispatcher, dispatcher_usable: bool,
    ) -> None:
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
            return

        # A supervisor from this process is already awaiting this row's
        # wake — don't enqueue a duplicate (avoids a wake storm while a
        # long cognition turn is still running).
        if row.id in self._inflight_restart_acks:
            return

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
            return

        waiter = getattr(handle, "wait", None)
        if not callable(waiter):
            # Legacy/stub dispatcher whose enqueue_signal returns no
            # awaitable handle — the signal was accepted onto the queue, so
            # treat the wake as delivered.
            await self._mark_wake_delivered(row)
            return

        # Delivery-gated: supervise the wake and only flag wake_delivered
        # once the COGNITION dispatch actually lands.
        self._inflight_restart_acks.add(row.id)
        self._spawn_ack_supervisor(row, handle)

    def _spawn_ack_supervisor(self, row, handle) -> None:
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

        tracker = getattr(self.agent, "_track_background_task", None)
        if callable(tracker):
            tracker(
                _await_and_ack(), name=f"restart_completed_ack:{row.id}",
            )
        else:  # pragma: no cover - production agents always expose tracker
            asyncio.ensure_future(_await_and_ack())

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
        """Flag the post-restart wake as delivered so it isn't re-dispatched."""
        try:
            await mark_wake_delivered(self._db, row.id)
            row.wake_delivered = True
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(
                "restart sweep: failed to flag wake_delivered for %s: %s",
                getattr(row, "id", "?"), e,
            )
