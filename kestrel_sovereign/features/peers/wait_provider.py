"""Durable waits over sender-owned native dispatch IDs (``a2a:<dispatch_id>``).

Only task dispatches use this rail. Questions already own a durable question
wake and notifications request no reply, so accepting either would double-wake
or leave an unfinishable watch. Local audit state is authoritative for delivery;
a peer read may update it but cannot choose the wait engine's event identity.
"""

from typing import ClassVar

from kestrel_sdk.tools import Outcome, WaitStatus
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.a2a.stores.unified.outbound_store import DispatchState
from kestrel_sovereign.waits.engine import (
    TERMINAL_EVENT_AT_KEY,
    TERMINAL_EVENT_DETAIL_KEY,
    TERMINAL_EVENT_FINAL_KEY,
    TERMINAL_EVENT_KEY,
    TERMINAL_EVENT_VIEW_KEY,
)

_TERMINAL_DONE = frozenset({"completed"})
_TERMINAL_FAIL = frozenset({"failed", "canceled", "rejected", "not_sent"})


class A2AWaitable:
    kind: ClassVar[str] = "a2a"
    signal: ClassVar[str | None] = None

    def __init__(self, feature):
        self._feature = feature

    async def owns_handle(self, handle: str) -> bool:
        row = await self._feature._native_outbound_store().get(handle)
        return row is not None and row.verb == "task"

    async def poll(self, handle: str) -> WaitStatus:
        row = await self._feature._native_outbound_store().get(handle)
        if row is None:
            return WaitStatus(
                Outcome.FAILED,
                "No sender-owned native dispatch exists",
                data={"dispatch_id": handle},
            )
        if row.verb != "task":
            return WaitStatus(
                Outcome.PENDING,
                f"Native {row.verb} dispatch uses its own resumption policy",
                data={"dispatch_id": handle, "verb": row.verb},
            )
        if row.state is DispatchState.NOT_SENT:
            return self._classify(
                "not_sent", handle, recipient=row.recipient_id, final=True
            )
        if row.state is DispatchState.COMPLETE:
            state = row.remote_task_state or (
                "completed" if row.response_kind == "message" else "unknown"
            )
            return self._classify(state, handle, recipient=row.recipient_id, final=True)
        if row.state is DispatchState.RESERVED:
            return self._classify("reserved", handle, recipient=row.recipient_id)
        result = await self._feature.get_peer_task_result(handle)
        if result.status is not ToolResultStatus.OK:
            return WaitStatus(
                Outcome.PENDING,
                result.error or "Native peer result is not yet available",
                data={"dispatch_id": handle, "recipient_id": row.recipient_id},
            )
        return self._classify(
            result.data["state"], handle, recipient=row.recipient_id, extra=result.data
        )

    def _classify(
        self,
        state: str,
        handle: str,
        *,
        recipient: str | None = None,
        extra: dict | None = None,
        final: bool = False,
    ) -> WaitStatus:
        norm = str(state or "").strip().lower()
        data = dict(extra or {})
        # ``extra`` is the peer's own task result. This provider names no
        # terminal event, and a peer must not name one for it: the reconciler
        # dedups wakes on these keys, so a forged value would let the peer
        # choose when this agent is woken (#3399), or date a fresh completion
        # early enough to have it announced as a replay (#3390).
        for key in (
            TERMINAL_EVENT_KEY,
            TERMINAL_EVENT_DETAIL_KEY,
            TERMINAL_EVENT_VIEW_KEY,
            TERMINAL_EVENT_FINAL_KEY,
            TERMINAL_EVENT_AT_KEY,
        ):
            data.pop(key, None)
        data.update({"dispatch_id": handle, "state": norm})
        if final:
            # Only from this agent's own stamped audit row, never the peer.
            data[TERMINAL_EVENT_FINAL_KEY] = True
        if recipient:
            data.setdefault("recipient_id", recipient)
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
