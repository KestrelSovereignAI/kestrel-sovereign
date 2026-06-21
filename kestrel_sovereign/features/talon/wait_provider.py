"""Waitable provider for Talon jobs (``talon:<job_id>``).

Wraps the coordinator's durable job registry and the same reap/reconcile
single-step the legacy ``talon_wait`` loop ran per iteration, classified
onto the generic :class:`Outcome` vocabulary. Looping, the cap, and the
ToolResult mapping live in :mod:`kestrel_sovereign.waits.engine`.

Note the legacy terminal vocabulary (``complete`` / ``failed`` /
``reject`` / ``finished_unknown``) collapses here: ``complete`` -> DONE,
the other three -> FAILED.
"""

from __future__ import annotations

from typing import ClassVar, Optional

from kestrel_sdk.tools import Outcome, WaitStatus

_TERMINAL_FAIL = ("failed", "reject", "finished_unknown")


class TalonWaitable:
    """Polls a dispatched Talon job by id against the durable registry."""

    kind: ClassVar[str] = "talon"
    signal: ClassVar[Optional[str]] = "talon.job_complete"

    def __init__(self, feature: "object") -> None:
        # The owning TalonCoordinatorFeature; provides the job registry
        # and reap/reconcile/persist helpers.
        self._feature = feature
        self._host_url_cache: Optional[str] = None
        self._host_url_resolved = False

    def _host_url(self) -> Optional[str]:
        # Resolve once per provider lifetime — the host URL is stable for
        # the process, and the legacy loop also resolved it just once.
        if not self._host_url_resolved:
            self._host_url_cache = self._feature._discover_host_url()
            self._host_url_resolved = True
        return self._host_url_cache

    async def poll(self, handle: str) -> WaitStatus:
        feature = self._feature
        # Pick up jobs persisted before a restart so a freshly reloaded
        # feature can still observe them.
        feature._reload_persisted_jobs()
        info = feature._jobs.get(handle)
        if info is None:
            return WaitStatus(
                Outcome.FAILED,
                f"Unknown job_id: {handle}",
                data={"job_id": handle},
            )

        changed = feature._reap_cli_job(info)
        if info.get("method") == "a2a":
            host_url = self._host_url()
            if host_url and await feature._reconcile_a2a_job(handle, info, host_url):
                changed = True
        if changed:
            feature._persist_jobs()

        status = info.get("status")
        rc = info.get("returncode")
        payload = {
            "job_id": handle,
            "status": status,
            "returncode": rc,
            "log_tail": feature._tail_job_log(info.get("log_path"), lines=20),
        }

        if status == "complete":
            return WaitStatus(
                Outcome.DONE,
                f"Talon job {handle[:8]} completed (rc={rc})",
                data=payload,
            )
        if status in _TERMINAL_FAIL:
            return WaitStatus(
                Outcome.FAILED,
                f"Talon job {handle[:8]} ended in '{status}' (rc={rc})",
                data=payload,
            )
        return WaitStatus(
            Outcome.PENDING,
            f"Talon job {handle[:8]} status: {status}",
            data=payload,
        )
