"""Waitable provider for A2A background tasks (``task:<task_id>``).

Wraps :meth:`TaskFeature._get_task_status_data` — the same single-poll
status read the legacy ``wait_for_task`` loop used — and classifies it
onto the generic :class:`Outcome` vocabulary. The poll loop, cap, and
ToolResult mapping live in :mod:`kestrel_sovereign.waits.engine`.
"""

from __future__ import annotations

from typing import ClassVar, Optional

from kestrel_sdk.tools import Outcome, WaitStatus


class TaskWaitable:
    """Polls a Kestrel A2A background task by id."""

    kind: ClassVar[str] = "task"
    # The sender-side resumption rail for tasks already exists as the
    # a2a.question_answered / a2a.task_complete signals; Wave 2 wires the
    # generic reconciler to those, so this stays None until then.
    signal: ClassVar[Optional[str]] = None

    def __init__(self, feature: "object") -> None:
        # The owning TaskFeature; provides _get_task_status_data.
        self._feature = feature

    async def poll(self, handle: str) -> WaitStatus:
        data = await self._feature._get_task_status_data(handle)
        if not data.get("ok"):
            return WaitStatus(
                Outcome.FAILED,
                data.get("error") or f"task {handle} status unavailable",
                data={"task_id": handle},
            )

        status = data.get("status")
        payload = {
            "task_id": data.get("task_id", handle),
            "status": status,
            "task_type": data.get("task_type"),
            "artifacts": data.get("artifacts", []),
            "message": data.get("message"),
        }

        if status == "completed":
            n = len(data.get("artifacts") or [])
            return WaitStatus(
                Outcome.DONE,
                f"Task {handle[:8]} completed ({n} artifact(s))",
                data=payload,
            )
        if status in ("failed", "canceled"):
            return WaitStatus(
                Outcome.FAILED,
                data.get("message") or f"Task {handle[:8]} {status}",
                data=payload,
            )
        return WaitStatus(
            Outcome.PENDING,
            f"Task {handle[:8]} status: {status}",
            data=payload,
        )
