"""Waitable provider for outbound A2A tasks/questions (``a2a:<task_id>``).

An agent that dispatches work to a peer (``send_a2a_task`` /
``send_a2a_question``) gets back a sender-owned task id. The peer's progress
lives in the *peer's* task store, not this agent's, so the local ``task:``
provider — which reads this agent's own ``a2a_tasks`` inbox — cannot see it.
Registering ``task:<outbound-id>`` therefore mis-routes: the local provider
reads "not found" and emits a false terminal ``wait.complete`` failure while
the real peer task is still running (#2729).

This provider is the correct home for outbound A2A handles. It routes a
status check through the sender-side outbound audit row (which retains the
stable recipient identity) and the peer proxy, so a durable
``wait("a2a:<task_id>", mode="signal")`` survives restart and completes once
when the peer reaches a terminal state.

Poll classification is deliberately conservative about *terminal* verdicts:
a peer that is momentarily unreachable, or an outbound row that has not yet
settled, stays :class:`Outcome.PENDING` so the reconciler keeps the watch
armed — the whole point of #2729 is to never convert a live peer task into a
false terminal failure. Only a peer-reported terminal state, or a terminal
state already stamped on the local audit row, ends the wait.
"""

from __future__ import annotations

from typing import ClassVar, Optional

from kestrel_sdk.tools import Outcome, WaitStatus
from kestrel_sdk.tools.result import ToolResultStatus

# Peer/audit lifecycle tokens that end the wait.
_TERMINAL_DONE = frozenset({"completed"})
_TERMINAL_FAIL = frozenset(
    {"failed", "canceled", "cancelled", "dispatch_failed"}
)


class A2AWaitable:
    """Polls an outbound A2A task by its sender-owned id."""

    kind: ClassVar[str] = "a2a"
    # The generic reconciler routes terminal transitions through the
    # ``wait.complete`` source (there is no bespoke a2a wait-completion
    # source); the sender-side ``a2a.question_answered`` rail remains a
    # separate, complementary resumption path for question lineages.
    signal: ClassVar[Optional[str]] = None

    def __init__(self, feature: "object") -> None:
        # The owning PeersFeature; provides the outbound audit DB and the
        # peer-proxy ``get_peer_task_result`` fetch.
        self._feature = feature

    def _agent_id(self) -> str:
        feature = self._feature
        return str(
            getattr(getattr(feature, "agent", None), "did", None)
            or getattr(feature, "_own_name", "")
        )

    async def _get_outbound_row(self, handle: str):
        """Return the sender-side outbound audit row for ``handle`` or None.

        Raises the outbound store's ambiguity error to the caller so callers
        can decide how to treat a known-unsafe duplicate route.
        """
        feature = self._feature
        db = getattr(feature, "_db", None)
        if db is None or not getattr(feature, "_outbound_route_store_ready", False):
            return None
        from kestrel_sovereign.a2a.outbound_store import get_outbound_task

        return await get_outbound_task(
            db, agent_id=self._agent_id(), task_id=handle
        )

    async def owns_handle(self, handle: str) -> Optional[bool]:
        """Whether ``handle`` is a tracked outbound A2A TASK of this agent.

        Ownership check used at watch registration (#2729): ``True`` when a
        sender-side outbound audit row exists for the id AND it is a
        ``verb="task"`` dispatch; ``False`` when the route store is ready but
        has no such row, the id resolves to a known-ambiguous duplicate route
        (which must never be trusted), or the row is a NON-task dispatch
        (question/message); and ``None`` (unverifiable → caller fails open)
        when no outbound store is wired.

        A2A **questions** are deliberately rejected here (``False``): a
        ``send_a2a_question`` already has its own durable resumption rail — the
        pending-question supervisor wakes the asker with an
        ``a2a.question_answered`` signal. Arming a generic wait watch on the
        same id would wake the agent TWICE for one terminal answer, because the
        two rails have separate dedup stores (#2729 P2). Messages are
        fire-and-forget with no terminal lifecycle, so they are rejected too.
        """
        feature = self._feature
        db = getattr(feature, "_db", None)
        if db is None or not getattr(feature, "_outbound_route_store_ready", False):
            return None
        from kestrel_sovereign.a2a.outbound_store import (
            OutboundTaskRouteAmbiguousError,
        )

        try:
            row = await self._get_outbound_row(handle)
        except OutboundTaskRouteAmbiguousError:
            # A duplicate historical route is affirmatively unsafe, not a
            # no-record legacy case — refuse to arm a watch on it.
            return False
        except Exception:
            return None
        if row is None:
            # Not a tracked outbound dispatch of ours.
            return False
        # Only durable outbound TASKS are watchable via ``a2a:``. Questions and
        # messages resume via their own rails (or not at all), so claiming them
        # here would double-wake / never-terminate.
        if str(row.verb or "").strip().lower() != "task":
            return False
        return True

    async def poll(self, handle: str) -> WaitStatus:
        feature = self._feature
        from kestrel_sovereign.a2a.outbound_store import (
            OutboundTaskRouteAmbiguousError,
        )

        try:
            row = await self._get_outbound_row(handle)
        except OutboundTaskRouteAmbiguousError:
            return WaitStatus(
                Outcome.FAILED,
                f"a2a task {handle[:8]} has an ambiguous outbound route",
                data={"task_id": handle, "state": "ambiguous"},
            )
        except Exception as exc:
            # Audit store unreadable right now — stay pending rather than
            # inventing a terminal failure for a possibly-live peer task.
            return WaitStatus(
                Outcome.PENDING,
                f"a2a task {handle[:8]} route lookup unavailable: {exc}",
                data={"task_id": handle},
            )

        # Defense-in-depth (#2729 P2): a non-task outbound row must NEVER emit
        # a terminal ``wait.complete`` here. ``owns_handle`` already rejects a
        # question/message at registration, but a watch that slipped past it
        # (e.g. the outbound store was not ready THEN and the check failed
        # open) stays PENDING so the question's own ``a2a.question_answered``
        # rail is the sole resumption path — exactly one terminal wake.
        if row is not None and str(row.verb or "").strip().lower() != "task":
            return WaitStatus(
                Outcome.PENDING,
                f"a2a {handle[:8]} is a {row.verb}; it resumes via its own "
                f"rail, not this watch",
                data={"task_id": handle, "verb": row.verb},
            )

        # A terminal state already recorded on the local audit row (e.g. a
        # transport-level dispatch failure, or a prior terminal fetch) is
        # authoritative — no need to round-trip the peer again.
        if row is not None and row.terminal_state:
            return self._classify(
                row.terminal_state, handle, recipient=row.recipient
            )

        recipient = row.recipient if row is not None else None
        if not recipient:
            # No retained route yet (reserved dispatch not activated, or the
            # row was pruned). Keep watching — never a false terminal.
            return WaitStatus(
                Outcome.PENDING,
                f"a2a task {handle[:8]} has no routable recipient yet",
                data={"task_id": handle},
            )

        try:
            result = await feature.get_peer_task_result(recipient, handle)
        except Exception as exc:
            return WaitStatus(
                Outcome.PENDING,
                f"a2a task {handle[:8]} peer fetch raised: {exc}",
                data={"task_id": handle, "recipient": recipient},
            )

        if getattr(result, "status", None) is not ToolResultStatus.OK:
            # Peer unreachable / directory miss right now — transient, keep
            # the watch armed instead of converting it to a terminal failure.
            return WaitStatus(
                Outcome.PENDING,
                f"a2a task {handle[:8]} to {recipient} not fetchable yet: "
                f"{getattr(result, 'error', '') or 'unavailable'}",
                data={"task_id": handle, "recipient": recipient},
            )

        payload = dict(result.data or {})
        state = str(payload.get("state", "") or "unknown")
        return self._classify(state, handle, recipient=recipient, extra=payload)

    def _classify(
        self,
        state: str,
        handle: str,
        *,
        recipient: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> WaitStatus:
        norm = str(state or "").strip().lower()
        data = dict(extra or {})
        data.update({"task_id": handle, "state": norm})
        if recipient:
            data.setdefault("recipient", recipient)
        label = handle[:8]
        if norm in _TERMINAL_DONE:
            return WaitStatus(
                Outcome.DONE,
                f"A2A task {label} completed"
                + (f" (recipient {recipient})" if recipient else ""),
                data=data,
            )
        if norm in _TERMINAL_FAIL:
            return WaitStatus(
                Outcome.FAILED,
                f"A2A task {label} ended in '{norm}'"
                + (f" (recipient {recipient})" if recipient else ""),
                data=data,
            )
        return WaitStatus(
            Outcome.PENDING,
            f"A2A task {label} status: {norm}",
            data=data,
        )
