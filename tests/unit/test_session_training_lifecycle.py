"""Deterministic race contracts for the session-backed training adapters (#2524).

RunPod, Vast.ai, and GCP Compute share one lifecycle owner. Every scenario
here runs against all three adapters with a fake provider whose remote side
is observable: which sessions are still billing, which remote jobs are still
running, and which provider job IDs were ever issued. After every scenario the
same invariants must hold — no running session or remote job, no lost
provider ID, no pending asyncio task, and no un-retrieved task exception.
"""

from __future__ import annotations

import asyncio
import gc
import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from google.api_core.exceptions import NotFound

from kestrel_sovereign.features.training.adapters._session_training_lifecycle import (
    TERMINAL_PHASES,
    IllegalSessionJobTransition,
    ReleaseIntent,
    SessionJobPhase,
    SessionLifecycleTimeouts,
    SessionReleaseError,
    SessionTrainingRecord,
    phase_status,
)
from kestrel_sovereign.features.training.adapters.gcp_compute_adapter import (
    GCPComputeTrainingAdapter,
)
from kestrel_sovereign.features.training.adapters.runpod_adapter import (
    RunPodTrainingAdapter,
)
from kestrel_sovereign.features.training.adapters.vastai_adapter import (
    VastAITrainingAdapter,
)
from kestrel_sovereign.features.training.factory import TrainingProviderFactory
from kestrel_sovereign.features.training.protocol import (
    TrainingProviderError,
    TrainingStatusError,
    TrainingSubmissionError,
)
from kestrel_sovereign.features.training.types import TrainingConfig, TrainingState


_FAST = SessionLifecycleTimeouts(submission_drain=0.5, release=0.5, close=2.0)
_ids = itertools.count(1)
_remotes: list[FakeRemote] = []


@dataclass
class FakeRemote:
    """The provider as observed from outside: sessions, jobs, and barriers."""

    sessions: list = field(default_factory=list)
    jobs: dict = field(default_factory=dict)  # provider job id -> job dict
    submit_calls: int = 0
    submit_started: asyncio.Event = field(default_factory=asyncio.Event)
    submit_gate: asyncio.Event = field(default_factory=asyncio.Event)
    submit_error: Exception | None = None
    # Once set, the remote has accepted the job but its response is still in
    # flight until this gate opens (None: the response arrives immediately).
    accepted: asyncio.Event = field(default_factory=asyncio.Event)
    response_gate: asyncio.Event | None = None
    # A cancelled response read that the provider code swallows, returning the ID.
    return_despite_cancel: bool = False
    # The submission ignores cancellation until this gate opens.
    ignore_cancel_until: asyncio.Event | None = None
    # Once that gate opens, the stuck submission completes and returns an ID.
    accept_after_ignored_cancel: bool = False
    # The submission swallows the first cancellation, then honours the next.
    swallow_first_cancel: bool = False
    acquire_gate: asyncio.Event | None = None
    acquire_started: asyncio.Event = field(default_factory=asyncio.Event)
    acquired: asyncio.Event = field(default_factory=asyncio.Event)
    acquire_returns_despite_cancel: bool = False
    release_gate: asyncio.Event | None = None
    release_started: asyncio.Event = field(default_factory=asyncio.Event)
    release_error: Exception | None = None
    # How many upcoming releases fail with ``release_error`` (None: all of them).
    release_failures: int | None = None
    release_calls: int = 0
    cancel_job_calls: list = field(default_factory=list)
    cancel_job_error: Exception | None = None
    upload_gate: asyncio.Event | None = None
    upload_started: asyncio.Event = field(default_factory=asyncio.Event)
    backend_base_url: str | None = "https://backend.example"
    loop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_running_loop)

    def __post_init__(self) -> None:
        _remotes.append(self)

    # -- remote side -----------------------------------------------------

    def new_session(self):
        number = next(_ids)
        session = SimpleNamespace(
            pod_id=f"pod-{number}",
            instance_id=number,
            instance_name=f"instance-{number}",
            zone="us-central1-a",
            backend_base_url=self.backend_base_url,
            state="running",
            profile=SimpleNamespace(persistent_pod_id=None),
        )
        self.sessions.append(session)
        return session

    def accept_job(self, session) -> str:
        job_id = f"remote-{next(_ids)}"
        self.jobs[job_id] = {"session": session, "state": "running"}
        return job_id

    def session_where(self, **identity):
        (session,) = [
            s
            for s in self.sessions
            if all(getattr(s, key) == value for key, value in identity.items())
        ]
        return session

    def running_sessions(self) -> list:
        return [s for s in self.sessions if s.state == "running"]

    def running_jobs(self) -> list:
        return [
            job_id
            for job_id, job in self.jobs.items()
            if job["state"] == "running" and job["session"].state == "running"
        ]

    # -- operations the adapters' managers call -------------------------

    async def acquire(self):
        self.acquire_started.set()
        if self.acquire_gate is not None:
            try:
                await self.acquire_gate.wait()
            except asyncio.CancelledError:
                if self.acquire_returns_despite_cancel:
                    return self.new_session()
                raise
        # Set in the same step that registers the job, before its submission
        # task is scheduled: a waiter woken here runs ahead of that task.
        self.acquired.set()
        return self.new_session()

    async def upload(self):
        self.upload_started.set()
        if self.upload_gate is not None:
            await self.upload_gate.wait()

    async def submit(self, session) -> str:
        self.submit_calls += 1
        self.submit_started.set()
        try:
            await self.submit_gate.wait()  # readiness / upload
        except asyncio.CancelledError:
            if self.swallow_first_cancel:
                await asyncio.Event().wait()  # only a second cancel ends this
            while self.ignore_cancel_until is not None:
                try:
                    await self.ignore_cancel_until.wait()
                    break
                except asyncio.CancelledError:
                    continue  # a stuck task ignores every cancellation
            if not self.accept_after_ignored_cancel:
                raise
        if self.submit_error is not None:
            raise self.submit_error
        job_id = self.accept_job(session)
        self.accepted.set()
        if self.response_gate is None:
            return job_id
        try:
            await self.response_gate.wait()
        except asyncio.CancelledError:
            if self.return_despite_cancel:
                return job_id
            raise
        return job_id

    async def release(self, session) -> None:
        self.release_calls += 1
        self.release_started.set()
        if self.release_gate is not None:
            await self.release_gate.wait()
        if self.release_error is not None and self.release_failures != 0:
            if self.release_failures is not None:
                self.release_failures -= 1
            raise self.release_error
        session.state = "released"

    def release_from_thread(self, session) -> None:
        """A blocking provider SDK call, run by the adapter in a worker thread."""
        asyncio.run_coroutine_threadsafe(self.release(session), self.loop).result(
            timeout=10
        )

    async def cancel_job(self, session, job_id):
        self.cancel_job_calls.append(job_id)
        if self.cancel_job_error is not None:
            raise self.cancel_job_error
        self.jobs[job_id]["state"] = "cancelled"
        return {"status": "cancelled"}


async def _swallowing_terminate(remote: FakeRemote, session) -> None:
    """The real managers' ``terminate_session``: a failure is logged, not raised."""
    try:
        await remote.release(session)
    except Exception:
        logging.getLogger(__name__).exception("terminate failed (swallowed)")


class FakeRunPodProvider:
    def __init__(self, remote: FakeRemote):
        self.remote = remote

    def stop_pod(self, pod_id):
        self.remote.release_from_thread(self.remote.session_where(pod_id=pod_id))
        return {"id": pod_id, "desiredStatus": "EXITED"}


class FakeRunPodManager:
    def __init__(self, remote: FakeRemote):
        self.remote = remote
        self.provider = FakeRunPodProvider(remote)
        self._session = None

    async def start_training_pod(self, companion_id):
        self._session = await self.remote.acquire()
        return self._session

    async def submit_training_job(self, *, session, **_kwargs):
        return await self.remote.submit(session)

    async def cancel_training_job(self, session, job_id):
        return await self.remote.cancel_job(session, job_id)

    async def stop_session(self):
        # Like the real manager: forget the pod, then let a failed stop raise.
        session, self._session = self._session, None
        await self.remote.release(session)

    async def terminate_session(self, session):
        await _swallowing_terminate(self.remote, session)

    async def poll_training_status(self, *, session, job_id):
        return {"status": "running", "progress": 0.5}


class FakeVastSDK:
    server_url = "https://vast.example"


class FakeVastManager:
    api_key = "vast-test-key"

    def __init__(self, remote: FakeRemote):
        self.remote = remote
        self._sdk = FakeVastSDK()
        self._session = None
        # A 200 reply that does not destroy the instance (None: destroy it).
        self.destroy_reply: dict | None = None
        self.transport = httpx.MockTransport(self._vast_api)

    def _get_sdk(self):
        return self._sdk

    async def _vast_api(self, request: httpx.Request) -> httpx.Response:
        """The Vast.ai REST API the adapter destroys instances through."""
        assert request.method == "DELETE"
        assert str(request.url).startswith("https://vast.example/api/v0/instances/")
        assert request.headers["Authorization"] == f"Bearer {self.api_key}"
        instance_id = int(request.url.path.rstrip("/").rsplit("/", 1)[1])
        session = self.remote.session_where(instance_id=instance_id)
        if session.state != "running":
            return httpx.Response(404, json={"success": False, "error": "not_found"})
        if self.destroy_reply is not None:
            return httpx.Response(200, json=self.destroy_reply)
        try:
            await self.remote.release(session)
        except Exception as error:
            return httpx.Response(500, json={"success": False, "msg": str(error)})
        return httpx.Response(200, json={"success": True})

    async def start_session(self, **_kwargs):
        self._session = await self.remote.acquire()

    async def submit_training_job_http(self, *, session, **_kwargs):
        return await self.remote.submit(session)

    async def terminate_session(self, session):
        await _swallowing_terminate(self.remote, session)

    async def poll_training_status_http(self, *, session, job_id):
        return {"status": "running", "progress": 0.5}


class FakeGCPInstancesClient:
    def __init__(self, remote: FakeRemote):
        self.remote = remote

    def delete(self, *, project, zone, instance):
        if self.remote.session_where(instance_name=instance).state != "running":
            raise NotFound(f"instance {instance} not found")
        return SimpleNamespace(name=f"delete:{instance}")


class FakeGCPManager:
    disk_config: dict = {}
    project_id = "kestrel-test"

    def __init__(self, remote: FakeRemote):
        self.remote = remote
        self._client = FakeGCPInstancesClient(remote)
        self._session = None

    def _get_instances_client(self):
        return self._client

    async def _wait_for_operation(self, operation_name, zone):
        instance = operation_name.removeprefix("delete:")
        await self.remote.release(self.remote.session_where(instance_name=instance))

    async def start_session(self, **_kwargs):
        self._session = await self.remote.acquire()

    async def _ssh_command(self, session, command):
        return ""

    async def _scp_upload(self, session, local_path, remote_path):
        await self.remote.upload()

    async def submit_training_job(self, *, session, **_kwargs):
        return await self.remote.submit(session)

    async def terminate_session(self, session):
        await _swallowing_terminate(self.remote, session)

    async def poll_training_status(self, *, session, job_id):
        return {"status": "running", "progress": 0.5}


_PROVIDERS = {
    "runpod": (RunPodTrainingAdapter, FakeRunPodManager),
    "vastai": (VastAITrainingAdapter, FakeVastManager),
    "gcp_compute": (GCPComputeTrainingAdapter, FakeGCPManager),
}


@pytest.fixture(params=sorted(_PROVIDERS))
def provider(request):
    return request.param


@pytest.fixture(autouse=True)
async def _unstick_remotes():
    """Open every barrier a test left closed once it ends.

    A failing assertion would otherwise leave a submission, acquisition, or
    release blocked on its barrier, and the event loop's teardown would wait
    on it forever instead of reporting the failure.
    """

    yield
    while _remotes:
        remote = _remotes.pop()
        for gate in (
            remote.ignore_cancel_until,
            remote.acquire_gate,
            remote.response_gate,
            remote.release_gate,
            remote.upload_gate,
        ):
            if gate is not None:
                gate.set()
        remote.submit_gate.set()
    await _settle()


@pytest.fixture
async def loop_errors():
    """Every loop-level error, including 'Task exception was never retrieved'."""

    loop = asyncio.get_running_loop()
    errors: list[dict] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    yield errors
    loop.set_exception_handler(previous)


def _build(provider: str, remote: FakeRemote, timeouts=_FAST):
    adapter_cls, manager_cls = _PROVIDERS[provider]
    manager = manager_cls(remote)
    extra = {"http_transport": manager.transport} if provider == "vastai" else {}
    return adapter_cls(manager=manager, lifecycle_timeouts=timeouts, **extra), manager


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _assert_no_orphans(adapter, remote: FakeRemote, loop_errors) -> None:
    await _settle()
    assert remote.running_sessions() == [], "a provider session is still billing"
    assert remote.running_jobs() == [], "a remote training job is still running"
    assert adapter._active_jobs == {}
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    assert pending == [], f"owned tasks left pending: {pending}"
    gc.collect()
    await _settle()
    assert loop_errors == [], f"un-retrieved task outcomes: {loop_errors}"


async def _start(adapter, companion_id: str = "companion-1"):
    return await adapter.start_training(companion_id, b"avatar", TrainingConfig())


# ----------------------------------------------------------------------
# 1-5. Cancellation at every phase
# ----------------------------------------------------------------------


async def test_cancel_before_submission_begins(provider, loop_errors):
    remote = FakeRemote()
    adapter, _ = _build(provider, remote)

    async def cancel_on_registration():
        await remote.acquired.wait()
        (job_id,) = adapter._active_jobs
        return await adapter.cancel(job_id)

    canceller = asyncio.create_task(cancel_on_registration())
    await _start(adapter)
    assert await canceller is True

    assert remote.submit_calls == 0
    assert remote.release_calls == 1
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_cancel_while_submission_blocked(provider, loop_errors):
    remote = FakeRemote()
    adapter, _ = _build(provider, remote)

    job = await _start(adapter)
    await remote.submit_started.wait()
    status = await adapter.get_status(job.job_id)
    assert status.state is TrainingState.PREPARING

    assert await adapter.cancel(job.job_id) is True

    assert remote.jobs == {}
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_cancel_while_gcp_upload_blocked(loop_errors):
    remote = FakeRemote(upload_gate=asyncio.Event())
    adapter, _ = _build("gcp_compute", remote)

    job = await _start(adapter)
    await remote.upload_started.wait()

    assert await adapter.cancel(job.job_id) is True

    assert remote.submit_calls == 0
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_remote_accepts_exactly_as_cancel_wins_local_race(provider, loop_errors):
    """The provider accepts the job, but the cancel lands before our await resumes.

    The ID dies inside the provider call. The session it runs on is the
    retained identity, and releasing it stops the remote job.
    """

    remote = FakeRemote(response_gate=asyncio.Event())
    adapter, _ = _build(provider, remote)
    job = await _start(adapter)
    remote.submit_gate.set()
    await remote.accepted.wait()  # accepted remotely; the response is in flight

    assert await adapter.cancel(job.job_id) is True

    (accepted,) = remote.jobs
    assert remote.jobs[accepted]["session"].state == "released"
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_remote_submission_returns_despite_cancel_keeps_provider_id(
    provider, loop_errors, caplog
):
    remote = FakeRemote(response_gate=asyncio.Event(), return_despite_cancel=True)
    adapter, _ = _build(provider, remote)
    job = await _start(adapter)
    remote.submit_gate.set()
    await remote.accepted.wait()

    with caplog.at_level(logging.WARNING):
        assert await adapter.cancel(job.job_id) is True

    (provider_job_id,) = remote.jobs
    assert f"accepted job {provider_job_id}" in caplog.text
    if provider == "runpod":
        assert remote.cancel_job_calls == [provider_job_id]
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_cancel_after_provider_id_published(provider, loop_errors):
    remote = FakeRemote()
    adapter, _ = _build(provider, remote)
    job = await _start(adapter)
    remote.submit_gate.set()
    await _settle()

    record = adapter._active_jobs[job.job_id]
    assert record.phase is SessionJobPhase.TRAINING
    provider_job_id = record.provider_job_id
    assert provider_job_id in remote.jobs
    assert (await adapter.get_status(job.job_id)).state is TrainingState.TRAINING

    assert await adapter.cancel(job.job_id) is True

    if provider == "runpod":
        assert remote.cancel_job_calls == [provider_job_id]
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_repeated_and_concurrent_cancel_share_one_teardown(provider, loop_errors):
    remote = FakeRemote(release_gate=asyncio.Event())
    adapter, _ = _build(provider, remote)
    job = await _start(adapter)
    await remote.submit_started.wait()

    callers = [asyncio.create_task(adapter.cancel(job.job_id)) for _ in range(3)]
    cleanup = asyncio.create_task(adapter.cleanup(job.job_id))
    await remote.release_started.wait()

    # A cancelled caller must not abandon the teardown it joined.
    callers[0].cancel()
    await _settle()
    assert not callers[0].done()
    remote.release_gate.set()

    with pytest.raises(asyncio.CancelledError):
        await callers[0]
    assert [await caller for caller in callers[1:]] == [True, True]
    await cleanup
    assert remote.release_calls == 1
    assert await adapter.cancel(job.job_id) is True  # idempotent once released
    assert await adapter.cancel("never-started") is False
    await _assert_no_orphans(adapter, remote, loop_errors)


# ----------------------------------------------------------------------
# 6. Failure paths
# ----------------------------------------------------------------------


async def test_submission_failure_is_recorded_and_session_released(provider, loop_errors):
    remote = FakeRemote(submit_error=ConnectionError("pod unreachable"))
    adapter, _ = _build(provider, remote)
    job = await _start(adapter)
    remote.submit_gate.set()
    await _settle()

    status = await adapter.get_status(job.job_id)
    assert status.state is TrainingState.FAILED
    assert "pod unreachable" in status.error

    await adapter.cleanup(job.job_id)
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_session_release_failure_retains_custody_until_retry(provider, loop_errors):
    remote = FakeRemote(release_error=RuntimeError("provider API down"))
    adapter, _ = _build(provider, remote)
    job = await _start(adapter)
    await remote.submit_started.wait()

    assert await adapter.cancel(job.job_id) is False
    record = adapter._active_jobs[job.job_id]
    assert record.phase is SessionJobPhase.RELEASE_FAILED
    assert record.error.startswith("session release failed")
    if provider != "vastai":  # the Vast.ai SDK swallows the underlying error
        assert "provider API down" in record.error

    # The session may still be billing: never a terminal (let alone CANCELLED)
    # status, or a caller polling for completion would stop watching it.
    status = await adapter.get_status(job.job_id)
    assert status.state is TrainingState.RELEASE_FAILED
    assert not status.state.is_terminal()
    session_id = adapter._lifecycle._hooks.session_id(record.session)
    assert session_id in status.message
    # That status query retried the release; it failed again, so custody stays.
    assert await record.release_task is False
    assert record.phase is SessionJobPhase.RELEASE_FAILED
    assert remote.running_sessions() == [record.session]
    assert not (await adapter.get_status(job.job_id)).state.is_terminal()
    await record.release_task

    remote.release_error = None
    assert await adapter.cancel(job.job_id) is True
    status = await adapter.get_status(job.job_id)
    assert status.state is TrainingState.CANCELLED  # only now: release verified
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_status_query_retries_a_failed_release(provider, loop_errors):
    """A failed release is retried on the next status query, not only at close()."""

    remote = FakeRemote(release_error=RuntimeError("provider API down"))
    adapter, _ = _build(provider, remote)
    job = await _start(adapter)
    await remote.submit_started.wait()
    assert await adapter.cancel(job.job_id) is False
    record = adapter._active_jobs[job.job_id]
    calls_before = remote.release_calls

    remote.release_error = None
    status = await adapter.get_status(job.job_id)
    assert status.state is TrainingState.RELEASE_FAILED  # the last verified outcome
    assert record.phase is SessionJobPhase.RELEASING
    assert await record.release_task is True

    assert remote.release_calls == calls_before + 1
    assert (await adapter.get_status(job.job_id)).state is TrainingState.CANCELLED
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_release_is_verified_not_inferred_from_a_swallowing_manager(
    provider, loop_errors
):
    """The managers' ``terminate_session`` logs a failed stop and returns.

    Releasing through it would report success while the instance keeps
    billing. Custody may only drop once the provider itself confirmed the
    stop, so a failure must fail the cancel and keep the job. Two jobs make
    the cancelled one not the manager's current session, which is exactly
    where a swallowing terminate path would otherwise be taken.
    """

    remote = FakeRemote()
    adapter, _ = _build(provider, remote)
    first = await _start(adapter, "companion-1")
    await _start(adapter, "companion-2")
    first_session = adapter._active_jobs[first.job_id].session
    remote.release_error = RuntimeError("provider refused the stop")

    assert await adapter.cancel(first.job_id) is False

    assert first_session.state == "running"
    record = adapter._active_jobs[first.job_id]
    assert record.phase is SessionJobPhase.RELEASE_FAILED
    assert not record.session_released

    remote.release_error = None
    await adapter.close()
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_vast_unconfirmed_destroy_is_a_release_failure(loop_errors):
    remote = FakeRemote()
    adapter, manager = _build("vastai", remote)
    job = await _start(adapter)
    manager.destroy_reply = {"success": False, "msg": "busy"}

    assert await adapter.cancel(job.job_id) is False
    record = adapter._active_jobs[job.job_id]
    assert "did not confirm" in record.error
    assert record.session.state == "running"

    manager.destroy_reply = None
    assert await adapter.cancel(job.job_id) is True
    await _assert_no_orphans(adapter, remote, loop_errors)


@pytest.mark.parametrize("provider_name", ["gcp_compute", "vastai"])
async def test_instance_already_deleted_counts_as_released(provider_name, loop_errors):
    remote = FakeRemote()
    adapter, manager = _build(provider_name, remote)
    job = await _start(adapter)
    session = adapter._active_jobs[job.job_id].session
    session.state = "released"  # an earlier delete completed out of band

    assert await adapter.cancel(job.job_id) is True
    assert remote.release_calls == 0
    assert manager._session is None
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_release_timeout_retains_custody(provider, loop_errors):
    remote = FakeRemote(release_gate=asyncio.Event())
    timeouts = SessionLifecycleTimeouts(submission_drain=0.5, release=0.05, close=2.0)
    adapter, _ = _build(provider, remote, timeouts)
    job = await _start(adapter)

    assert await adapter.cancel(job.job_id) is False
    assert adapter._active_jobs[job.job_id].phase is SessionJobPhase.RELEASE_FAILED

    remote.release_gate.set()
    assert await adapter.cancel(job.job_id) is True
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_runpod_job_cancel_failure_is_secondary_to_session_release(loop_errors):
    remote = FakeRemote(cancel_job_error=RuntimeError("/cancel returned 500"))
    adapter, _ = _build("runpod", remote)
    job = await _start(adapter)
    remote.submit_gate.set()
    await _settle()
    record = adapter._active_jobs[job.job_id]

    assert await adapter.cancel(job.job_id) is True

    assert record.compensation_error == "/cancel returned 500"
    assert record.phase is SessionJobPhase.RELEASED
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_submission_ignoring_cancel_is_compensated_then_retained(provider, loop_errors):
    remote = FakeRemote(ignore_cancel_until=asyncio.Event())
    timeouts = SessionLifecycleTimeouts(submission_drain=0.05, release=0.5, close=2.0)
    adapter, _ = _build(provider, remote, timeouts)
    job = await _start(adapter)
    await remote.submit_started.wait()

    assert await adapter.cancel(job.job_id) is False
    record = adapter._active_jobs[job.job_id]
    assert record.session.state == "released"  # compensated despite the stuck task
    assert record.phase is SessionJobPhase.RELEASE_FAILED

    remote.ignore_cancel_until.set()
    assert await adapter.cancel(job.job_id) is True
    assert remote.release_calls == 1  # the session is not released twice
    await _assert_no_orphans(adapter, remote, loop_errors)


@pytest.mark.parametrize("provider_name", ["runpod", "vastai"])
async def test_session_rejected_after_acquisition_is_released(provider_name, loop_errors):
    remote = FakeRemote(backend_base_url=None)
    adapter, _ = _build(provider_name, remote)

    with pytest.raises(TrainingSubmissionError, match="no backend URL"):
        await _start(adapter)

    assert remote.release_calls == 1
    await _assert_no_orphans(adapter, remote, loop_errors)


@pytest.mark.parametrize("provider_name", ["runpod", "vastai"])
async def test_rejected_session_whose_release_fails_is_retained_for_close(
    provider_name, loop_errors
):
    remote = FakeRemote(
        backend_base_url=None, release_error=RuntimeError("provider API down")
    )
    adapter, _ = _build(provider_name, remote)

    with pytest.raises(TrainingSubmissionError, match="no backend URL") as raised:
        await _start(adapter)

    (orphan_id,) = adapter._active_jobs
    assert any(orphan_id in note for note in raised.value.__notes__)
    orphan = adapter._active_jobs[orphan_id]
    assert orphan.phase is SessionJobPhase.RELEASE_FAILED
    assert orphan.session.state == "running"

    with pytest.raises(SessionReleaseError) as close_error:
        await adapter.close()
    assert close_error.value.retained_job_ids == [orphan_id]
    assert remote.release_calls == 2  # close() re-attempted the orphan

    remote.release_error = None
    await adapter.close()
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_close_releases_an_orphan_registered_while_it_drains(provider, loop_errors):
    """close() cancels an acquisition that returns a session anyway.

    Its immediate release fails, so the session is registered as an orphan
    after close() began; the same close must release it rather than exit
    with it still billing.
    """

    remote = FakeRemote(
        acquire_gate=asyncio.Event(),
        acquire_returns_despite_cancel=True,
        release_error=RuntimeError("transient provider error"),
        release_failures=1,
    )
    adapter, _ = _build(provider, remote)
    starter = asyncio.create_task(_start(adapter))
    await remote.acquire_started.wait()

    await adapter.close()

    with pytest.raises(TrainingSubmissionError, match="closed"):
        await starter
    assert len(remote.sessions) == 1
    assert remote.release_calls == 2
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_start_caller_cancelled_during_acquisition(provider, loop_errors):
    remote = FakeRemote(acquire_gate=asyncio.Event(), acquire_returns_despite_cancel=True)
    adapter, _ = _build(provider, remote)

    starter = asyncio.create_task(_start(adapter))
    await remote.acquire_started.wait()
    starter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starter

    # The session arrived anyway, but no caller holds its job ID: released.
    assert len(remote.sessions) == 1
    assert remote.submit_calls == 0
    await _assert_no_orphans(adapter, remote, loop_errors)


# ----------------------------------------------------------------------
# 7. Adapter close with pending work
# ----------------------------------------------------------------------


async def test_close_drains_pending_submissions_and_releases_sessions(provider, loop_errors):
    remote = FakeRemote()
    adapter, _ = _build(provider, remote)
    await _start(adapter, "companion-1")
    await _start(adapter, "companion-2")
    await remote.submit_started.wait()

    await adapter.close()

    assert len(remote.sessions) == 2
    with pytest.raises(TrainingSubmissionError, match="closed"):
        await _start(adapter)
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_close_cancels_in_flight_acquisition(provider, loop_errors):
    remote = FakeRemote(acquire_gate=asyncio.Event())
    adapter, _ = _build(provider, remote)

    starter = asyncio.create_task(_start(adapter))
    await remote.acquire_started.wait()
    await adapter.close()

    with pytest.raises(TrainingSubmissionError, match="closed"):
        await starter
    assert remote.sessions == []
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_close_reports_retained_custody(provider, loop_errors):
    remote = FakeRemote(release_error=RuntimeError("provider API down"))
    adapter, _ = _build(provider, remote)
    job = await _start(adapter)

    with pytest.raises(SessionReleaseError) as raised:
        await adapter.close()

    assert raised.value.retained_job_ids == [job.job_id]
    assert job.job_id in adapter._active_jobs

    remote.release_error = None
    await adapter.close()
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_close_survives_caller_cancellation(provider, loop_errors):
    remote = FakeRemote(release_gate=asyncio.Event())
    adapter, _ = _build(provider, remote)
    await _start(adapter)

    closer = asyncio.create_task(adapter.close())
    await remote.release_started.wait()
    closer.cancel()
    await _settle()
    assert not closer.done()
    remote.release_gate.set()

    with pytest.raises(asyncio.CancelledError):
        await closer
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_factory_close_providers_closes_session_adapters(provider, loop_errors):
    remote = FakeRemote()
    adapter, _ = _build(provider, remote)
    await _start(adapter)
    TrainingProviderFactory._instances[provider] = adapter
    try:
        await TrainingProviderFactory.close_providers()
        assert provider not in TrainingProviderFactory._instances
    finally:
        TrainingProviderFactory._instances.pop(provider, None)
    await _assert_no_orphans(adapter, remote, loop_errors)


# ----------------------------------------------------------------------
# State machine and provider-specific release
# ----------------------------------------------------------------------


async def test_illegal_transition_is_rejected(provider, loop_errors):
    remote = FakeRemote()
    adapter, _ = _build(provider, remote)
    job = await _start(adapter)
    record = adapter._active_jobs[job.job_id]

    with pytest.raises(IllegalSessionJobTransition):
        record.transition(SessionJobPhase.RELEASED)

    await adapter.cancel(job.job_id)
    await _assert_no_orphans(adapter, remote, loop_errors)


@pytest.mark.parametrize("intent", ["cancel", "cleanup"])
async def test_status_of_job_being_released_is_not_terminal(
    provider, intent, loop_errors
):
    remote = FakeRemote(release_gate=asyncio.Event())
    adapter, _ = _build(provider, remote)
    job = await _start(adapter)
    release = asyncio.create_task(getattr(adapter, intent)(job.job_id))
    await remote.release_started.wait()

    status = await adapter.get_status(job.job_id)
    assert status.state is TrainingState.RELEASING
    assert not status.state.is_terminal()

    remote.release_gate.set()
    await release
    if intent == "cleanup":
        with pytest.raises(TrainingStatusError, match="was cleaned up"):
            await adapter.get_status(job.job_id)
    else:
        assert (await adapter.get_status(job.job_id)).state is TrainingState.CANCELLED
    await _assert_no_orphans(adapter, remote, loop_errors)


def _record_in(phase: SessionJobPhase, **fields) -> SessionTrainingRecord:
    now = datetime.now(timezone.utc)
    record = SessionTrainingRecord(
        job_id="job-1",
        companion_id="companion-1",
        trigger_word="TOK",
        config=TrainingConfig(),
        session=SimpleNamespace(),
        created_at=now,
        started_at=now,
    )
    record.phase = phase
    for name, value in fields.items():
        setattr(record, name, value)
    return record


def _phase_variants(phase: SessionJobPhase) -> list[dict]:
    """Every record shape a phase can be reported from."""

    return [
        {"release_intent": intent, "provider_job_id": provider_job_id, "error": error}
        for intent in (None, *ReleaseIntent)
        for provider_job_id in (None, "remote-1")
        for error in (None, "boom")
    ]


@pytest.mark.parametrize("phase", list(SessionJobPhase), ids=lambda phase: phase.value)
def test_every_phase_maps_to_a_status_no_further_ahead_than_verified(phase):
    """The guard against reported state running ahead of provider state.

    Parametrised over every phase, so a phase added without a mapping fails
    here (``phase_status`` ends in ``assert_never``) instead of falling
    through to some other state.
    """

    for fields in _phase_variants(phase):
        record = _record_in(phase, **fields)
        try:
            status = phase_status(
                record,
                provider_name="fake",
                session_id="session-1",
                preparing_message="preparing",
            )
        except TrainingStatusError:
            # Only a cleaned-up job has no status left to report.
            assert phase is SessionJobPhase.RELEASED
            assert fields["release_intent"] is ReleaseIntent.CLEANUP
            continue
        if status is None:
            # Polling the provider: only while a provider job may be running.
            assert phase in (SessionJobPhase.TRAINING, SessionJobPhase.STOPPING)
            assert phase is SessionJobPhase.TRAINING or fields["provider_job_id"]
            continue
        assert status.job_id == record.job_id
        if status.state.is_terminal():
            assert phase in TERMINAL_PHASES, (phase, status.state)
        if status.state is TrainingState.CANCELLED:
            assert phase is SessionJobPhase.RELEASED
        if phase in (SessionJobPhase.RELEASING, SessionJobPhase.RELEASE_FAILED):
            assert not status.state.is_terminal()
            assert "session-1" in status.message
        if phase is SessionJobPhase.RELEASE_FAILED:
            assert status.state is TrainingState.RELEASE_FAILED
            assert status.error


def test_terminal_phases_are_the_verified_ones():
    assert TERMINAL_PHASES == {SessionJobPhase.FAILED, SessionJobPhase.RELEASED}
    assert not TrainingState.RELEASING.is_terminal()
    assert not TrainingState.RELEASE_FAILED.is_terminal()


async def test_runpod_stops_its_own_pod_not_the_managers_current_one(loop_errors):
    remote = FakeRemote()
    adapter, manager = _build("runpod", remote)
    first = await _start(adapter, "companion-1")
    second = await _start(adapter, "companion-2")
    first_session = adapter._active_jobs[first.job_id].session
    assert manager._session is adapter._active_jobs[second.job_id].session

    assert await adapter.cancel(first.job_id) is True

    assert first_session.state == "released"
    assert manager._session is not None  # the second job's pod is untouched
    assert manager._session.state == "running"
    await adapter.cancel(second.job_id)
    assert manager._session is None
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_runpod_cancel_training_keeps_pod_and_stops_submission(loop_errors):
    remote = FakeRemote()
    adapter, _ = _build("runpod", remote)
    job = await _start(adapter)
    await remote.submit_started.wait()

    result = await adapter.cancel_training(job.job_id)

    assert result["message"] == "Job cancelled before submission"
    record = adapter._active_jobs[job.job_id]
    assert record.phase is SessionJobPhase.FAILED
    assert record.submission_task.cancelled()
    assert record.session.state == "running"
    status = await adapter.get_status(job.job_id)
    assert status.state is TrainingState.FAILED

    await adapter.cleanup(job.job_id)
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_runpod_cancel_training_waits_for_a_stuck_submission(loop_errors):
    """A submission that outlives the drain bound must not read as stopped.

    The task ignores cancellation past ``submission_drain`` and then gets a
    job accepted. ``cancel_training()`` may not report success while that
    task is live, and the late-published ID must be cancelled on the
    provider, because the caller already asked for the job to stop.
    """

    remote = FakeRemote(
        ignore_cancel_until=asyncio.Event(), accept_after_ignored_cancel=True
    )
    timeouts = SessionLifecycleTimeouts(submission_drain=0.05, release=0.5, close=2.0)
    adapter, _ = _build("runpod", remote, timeouts)
    job = await _start(adapter)
    await remote.submit_started.wait()

    with pytest.raises(TrainingProviderError, match="did not stop"):
        await adapter.cancel_training(job.job_id)
    record = adapter._active_jobs[job.job_id]
    assert not record.submission_task.done()
    assert record.phase is SessionJobPhase.STOPPING
    status = await adapter.get_status(job.job_id)
    assert status.state not in (TrainingState.CANCELLED, TrainingState.FAILED)

    remote.ignore_cancel_until.set()  # the stuck task now lands a remote job
    await remote.accepted.wait()
    # Not settled until the provider cancel issued for the late ID finishes.
    assert record.phase is SessionJobPhase.STOPPING
    await record.submission_task
    (provider_job_id,) = remote.jobs
    assert record.provider_job_id == provider_job_id
    assert remote.cancel_job_calls == [provider_job_id]
    assert remote.jobs[provider_job_id]["state"] == "cancelled"
    assert record.phase is SessionJobPhase.FAILED
    assert record.session.state == "running"  # cancel_training keeps the pod

    # A retried stop is now complete and does not cancel the job twice.
    result = await adapter.cancel_training(job.job_id)
    assert result == {"status": "cancelled"}
    assert remote.cancel_job_calls == [provider_job_id]

    await adapter.cleanup(job.job_id)
    assert remote.cancel_job_calls == [provider_job_id]
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_runpod_late_acceptance_whose_cancel_fails_is_reported(loop_errors):
    remote = FakeRemote(
        ignore_cancel_until=asyncio.Event(),
        accept_after_ignored_cancel=True,
        cancel_job_error=RuntimeError("/cancel returned 500"),
    )
    timeouts = SessionLifecycleTimeouts(submission_drain=0.05, release=0.5, close=2.0)
    adapter, _ = _build("runpod", remote, timeouts)
    job = await _start(adapter)
    await remote.submit_started.wait()
    with pytest.raises(TrainingProviderError, match="did not stop"):
        await adapter.cancel_training(job.job_id)
    record = adapter._active_jobs[job.job_id]

    remote.ignore_cancel_until.set()
    await record.submission_task
    (provider_job_id,) = remote.jobs
    assert record.compensation_error == "/cancel returned 500"
    # Not reported as stopped: the job's status comes from the provider.
    assert record.phase is SessionJobPhase.STOPPING
    assert (await adapter.get_status(job.job_id)).state is TrainingState.TRAINING

    # The pod-releasing cleanup retries the cancel, then stops the pod.
    remote.cancel_job_error = None
    await adapter.cleanup(job.job_id)
    assert remote.cancel_job_calls == [provider_job_id, provider_job_id]
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_runpod_cancel_training_reports_a_failed_provider_cancel(loop_errors):
    remote = FakeRemote(cancel_job_error=RuntimeError("/cancel returned 500"))
    adapter, _ = _build("runpod", remote)
    job = await _start(adapter)
    remote.submit_gate.set()
    await _settle()
    record = adapter._active_jobs[job.job_id]
    provider_job_id = record.provider_job_id

    with pytest.raises(TrainingProviderError, match="/cancel returned 500"):
        await adapter.cancel_training(job.job_id)
    assert remote.jobs[provider_job_id]["state"] == "running"
    assert record.phase is SessionJobPhase.STOPPING
    assert (await adapter.get_status(job.job_id)).state is TrainingState.TRAINING

    remote.cancel_job_error = None
    assert await adapter.cancel_training(job.job_id) == {"status": "cancelled"}
    assert remote.cancel_job_calls == [provider_job_id, provider_job_id]

    await adapter.cleanup(job.job_id)
    assert remote.cancel_job_calls == [provider_job_id, provider_job_id]
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_runpod_retried_stop_recancels_a_submission_that_swallowed_one(loop_errors):
    remote = FakeRemote(swallow_first_cancel=True)
    timeouts = SessionLifecycleTimeouts(submission_drain=0.05, release=0.5, close=2.0)
    adapter, _ = _build("runpod", remote, timeouts)
    job = await _start(adapter)
    await remote.submit_started.wait()

    with pytest.raises(TrainingProviderError, match="did not stop"):
        await adapter.cancel_training(job.job_id)
    record = adapter._active_jobs[job.job_id]
    assert record.phase is SessionJobPhase.STOPPING

    result = await adapter.cancel_training(job.job_id)

    assert result["message"] == "Job cancelled before submission"
    assert record.submission_task.cancelled()
    assert record.phase is SessionJobPhase.FAILED
    assert remote.jobs == {}
    await adapter.cleanup(job.job_id)
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_release_after_a_released_pod_sends_no_job_cancel(loop_errors):
    """A pod already stopped cannot run the job; do not call its /cancel."""

    remote = FakeRemote(
        ignore_cancel_until=asyncio.Event(), accept_after_ignored_cancel=True
    )
    timeouts = SessionLifecycleTimeouts(submission_drain=0.05, release=0.5, close=2.0)
    adapter, _ = _build("runpod", remote, timeouts)
    job = await _start(adapter)
    await remote.submit_started.wait()
    assert await adapter.cancel(job.job_id) is False  # pod stopped, task stuck
    record = adapter._active_jobs[job.job_id]
    assert record.session_released

    remote.ignore_cancel_until.set()  # the stuck task now publishes an ID
    await asyncio.wait({record.submission_task})
    assert record.provider_job_id is not None

    assert await adapter.cancel(job.job_id) is True
    assert remote.cancel_job_calls == []
    await _assert_no_orphans(adapter, remote, loop_errors)


async def test_runpod_stop_on_a_failed_release_reports_a_failed_job_cancel(loop_errors):
    remote = FakeRemote(
        release_error=RuntimeError("stop_pod returned 500"),
        cancel_job_error=RuntimeError("/cancel returned 500"),
    )
    adapter, _ = _build("runpod", remote)
    job = await _start(adapter)
    remote.submit_gate.set()
    await _settle()
    record = adapter._active_jobs[job.job_id]
    await adapter.cleanup(job.job_id)
    assert record.phase is SessionJobPhase.RELEASE_FAILED
    assert not record.session_released

    with pytest.raises(TrainingProviderError, match="/cancel returned 500"):
        await adapter.cancel_training(job.job_id)

    remote.release_error = None
    remote.cancel_job_error = None
    assert await adapter.cancel(job.job_id) is True
    await _assert_no_orphans(adapter, remote, loop_errors)
