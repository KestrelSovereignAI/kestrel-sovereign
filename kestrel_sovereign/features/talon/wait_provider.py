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

from typing import ClassVar, List, Optional

from kestrel_sdk.tools import Outcome, WaitStatus

_TERMINAL_FAIL = ("failed", "reject", "finished_unknown")
# The terminal vocabulary across both dispatch methods: a job in any of
# these is finished and the reconciler should NOT re-enumerate it as
# in-flight. ``complete`` -> DONE; the rest -> FAILED in ``poll``.
_TERMINAL_STATES = ("complete",) + _TERMINAL_FAIL


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
        # Enrich with the fields the talon.job_complete prompt template
        # indexes (repo/issue/label/started_at/completed_at/test_evidence/
        # ci_status), read straight off the durable job record. The generic
        # reconciler spreads WaitStatus.data into the signal payload, so the
        # talon template still renders fully now that the bespoke
        # build_signal_for_completed_job builder is gone (Wave 2 of #1860).
        payload = {
            "job_id": handle,
            "status": status,
            "returncode": rc,
            "log_path": info.get("log_path", ""),
            "log_tail": feature._tail_job_log(info.get("log_path"), lines=20),
            "repo": info.get("repo", ""),
            "issue": info.get("issue", ""),
            "label": info.get("label", ""),
            "started_at": info.get("started_at", ""),
            "completed_at": info.get("completed_at", ""),
            "test_evidence": info.get("test_evidence", ""),
            "ci_status": info.get("ci_status", ""),
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

    async def active_handles(self) -> List[str]:
        """Return the job ids the reconciler should poll for a wake.

        Implements :class:`~kestrel_sdk.tools.MonitorableWaitable`. Reloads
        the durable registry so a freshly-restarted feature sees jobs from a
        prior process, then returns the ids of cli_background jobs not yet in
        a terminal state. Cheap (a JSON reload + dict scan) — the reconciler
        calls it every cron tick. Classifying + signaling are the
        reconciler's job; this only enumerates.

        Scoped to ``cli_background`` jobs: those are the ones the retired
        talon_monitor cron drove, and the a2a path has its own resumption
        rail (a2a.task_complete). The reconciler still polls each returned
        handle to detect the actual terminal transition.
        """
        feature = self._feature
        feature._reload_persisted_jobs()
        active: List[str] = []
        for job_id, info in feature._jobs.items():
            if info.get("method") != "cli_background":
                continue
            if info.get("status") in _TERMINAL_STATES:
                continue
            active.append(job_id)
        return active
