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

    async def owns_handle(self, handle: str) -> Optional[bool]:
        """Whether ``handle`` is a LOCAL background task in this agent's store.

        Ownership check used at watch registration (#2729): the ``task``
        provider is the local background-TaskStore, NOT an outbound-A2A
        provider. An outbound A2A task id (``a2a:``) does not live in the
        local ``a2a_tasks`` store, so ``task_manager.get_task`` returns None
        and this returns ``False`` — rejecting a ``task:<outbound-a2a-id>``
        watch synchronously instead of letting a later poll read "not found"
        and emit a false terminal ``wait.complete`` failure.

        Returns ``None`` (unverifiable → caller fails open) when no task
        manager is wired or the lookup raises, so a transient backend hiccup
        never blocks an otherwise-valid watch.
        """
        manager = getattr(self._feature, "task_manager", None)
        if manager is None:
            return None
        try:
            task = await manager.get_task(handle)
        except Exception:
            return None
        return task is not None

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
