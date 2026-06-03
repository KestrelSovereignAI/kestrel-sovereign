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

from .store import (
    KNOWN_POLICIES,
    KNOWN_URGENCIES,
    PENDING_STATES,
    TERMINAL_STATES,
    ensure_restart_requests_table,
    get_request,
    insert_request,
    list_requests,
    update_status,
)

logger = logging.getLogger(__name__)


class RestartCoordinatorFeature(Feature):
    """Durable restart-request surface for agents (#1512)."""

    @property
    def tool_description(self) -> str:
        return (
            "Request a safe Kestrel host restart and track its outcome. "
            "Agent files a request; host coordinator executes when safe."
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
            "or cancel_restart_request."
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
        )
        logger.info(
            "Restart request filed: id=%s urgency=%s policy=%s reason=%s",
            req.id, urgency, policy, reason[:80],
        )
        return ToolResult.ok(
            confirmation=(
                f"Filed restart request {req.id} ({urgency}, {policy})"
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
                    continue
                # Hard reject.
                await update_status(
                    self._db, req.id,
                    status="rejected",
                    status_reason=decision["reason"],
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    expected_current_status=req.status,
                )
                continue

            # Move to executing BEFORE spawning the subprocess. If
            # the spawn fails we move back to pending so the next
            # poll retries.
            moved = await update_status(
                self._db, req.id,
                status="executing",
                status_reason="dispatched to detached restart subprocess",
                expected_current_status=req.status,
            )
            if not moved:
                deferred.append({
                    "request_id": req.id,
                    "reason": "lost race against another transition",
                })
                continue

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
            try:
                n = len(active_ids)
            except TypeError:
                n = 0
            if n:
                return {
                    "idle": False,
                    "reason": f"{n} active request id(s)",
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

    @staticmethod
    def _request_aged_past_timeout(req) -> bool:
        """Has the request sat in pending/approved longer than 5 min?"""
        try:
            requested = datetime.fromisoformat(req.requested_at)
        except ValueError:
            return False
        now = datetime.now(timezone.utc)
        return (now - requested).total_seconds() > 300

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
