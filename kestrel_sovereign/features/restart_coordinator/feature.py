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

        # Sweep — any ``executing`` row owned by this agent that
        # survived a restart needs to land in ``completed`` and wake
        # the agent so it can verify the post-restart state. The
        # restart we executed brought us back; mark and wake.
        await self._reap_post_restart_rows()

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

        idle = self._agent_appears_idle()
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

    def _agent_appears_idle(self) -> Dict[str, Any]:
        """Idle check against the agent's own in-flight surface.

        The default ``KestrelAgent`` exposes ``_active_request_ids``
        (set of in-flight cognition request IDs) and
        ``_background_tasks`` (asyncio tasks the dispatcher started).
        Either non-empty → not idle.

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
            try:
                n = len(active_ids)
            except TypeError:
                n = 0
            if n:
                return {
                    "idle": False,
                    "reason": (
                        f"{n} active request id(s)"
                        f"{self._active_request_age_suffix()}"
                    ),
                }

        bg_tasks = getattr(self.agent, "_background_tasks", None)
        if bg_tasks is not None:
            any_surface_seen = True
            try:
                alive = [t for t in bg_tasks if not t.done()]
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

    def _active_request_age_suffix(self) -> str:
        """Append the oldest active-request age to a busy deferral reason.

        Observability for #1558: when a restart defers on ``agent busy``,
        the operator can see how old the in-flight request markers are
        relative to the staleness sweep window, so a near-stale id is
        visible before it ages out.
        """
        ages_fn = getattr(self.agent, "active_request_ages", None)
        if not callable(ages_fn):
            return ""
        try:
            ages = ages_fn()
        except Exception:
            return ""
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
        """Sweep ``executing`` rows this agent filed, mark complete,
        and emit one ``restart.completed`` COGNITION signal per row.
        """
        if self._db is None:
            return
        agent_id = getattr(self.agent, "did", "") or ""
        if not agent_id:
            return
        executing = await list_requests(
            self._db, status="executing", agent_id=str(agent_id),
        )
        if not executing:
            return

        dispatcher = getattr(self.agent, "dispatcher", None)
        dispatcher_usable = (
            dispatcher is not None
            and hasattr(dispatcher, "enqueue_signal")
        )
        for row in executing:
            now = datetime.now(timezone.utc).isoformat()
            signal_delivered: Optional[bool] = None  # True/False if attempted

            if dispatcher_usable:
                try:
                    from kestrel_sovereign.signals.sources.restart import (
                        build_signal_for_restart_completed,
                    )
                    signal = build_signal_for_restart_completed(
                        row, target_agent=str(agent_id), completed_at=now,
                    )
                    enq = dispatcher.enqueue_signal(signal)
                    if asyncio.iscoroutine(enq):
                        await enq
                    signal_delivered = True
                except Exception as e:
                    signal_delivered = False
                    logger.warning(
                        "restart sweep: failed to enqueue signal for "
                        "%s: %s; row stays executing for next-boot retry",
                        row.id, e,
                    )

            # Only terminalize when EITHER the signal delivered, OR
            # there is no dispatcher to deliver to (a headless/test
            # agent — nothing to wake, no point looping). If the
            # dispatcher was present but enqueue raised, leave the
            # row in executing so the next boot's sweep retries
            # (codex P2 on PR #1512 round 1).
            if signal_delivered is False:
                continue
            await update_status(
                self._db, row.id,
                status="completed",
                status_reason="post-restart sweep observed agent re-init",
                completed_at=now,
                expected_current_status="executing",
            )
            # Mirror the COGNITION wake with a chat-visible status event so
            # the completed restart shows in chat, not only as an agent
            # turn (#1551).
            row.completed_at = now
            await self._emit_status_event(
                row, state="completed",
                status_reason="post-restart sweep observed agent re-init",
            )
