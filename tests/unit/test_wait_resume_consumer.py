"""Durable resume on a wait handle's wake (#3295).

A held wait is capped at ``MAX_HANDLE_WAIT_SECONDS``, so work that outlives
it — a workflow stage waiting on the ``talon:<job_id>`` a dispatch stage
returned — must park and resume on the handle's wake instead.
``register_wait_resume_consumer`` is that seam: it subscribes a durable
consumer to exactly the wake announcing one ``"<kind>:<handle>"`` reference.

The dispatcher, durable store, source registration, and reconciler here are
all production code; only the agent body and the wait providers are doubles.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import (
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Trust,
)
from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.wait import (
    PROMPT_TEMPLATE as WAIT_COMPLETE_TEMPLATE,
    build_wait_complete_registration,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import (
    WaitReconciler,
    register_wait_resume_consumer,
    wake_source,
)

AGENT_DID = "did:test:3295"
JOB_SOURCE = "example.job_complete"


class _Agent:
    did = AGENT_DID

    def __init__(self):
        self.tasks: list[asyncio.Task] = []
        self._privacy_transition_lock = asyncio.Lock()

    async def process_input(self, prompt: str, **kwargs):
        return "wake turn ran"

    def _get_privacy_transition_lock(self):
        return self._privacy_transition_lock

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        return task


class _JobProvider:
    """A MonitorableWaitable with its own wake source, like talon's."""

    kind = "job"
    signal = JOB_SOURCE

    def __init__(self):
        self.states: dict[str, tuple[Outcome, dict]] = {}

    def set(self, handle, outcome, data=None):
        self.states[handle] = (outcome, data or {})

    async def active_handles(self):
        return list(self.states)

    async def poll(self, handle):
        outcome, data = self.states.get(handle, (Outcome.PENDING, {}))
        return WaitStatus(outcome, f"{handle} is {outcome.value}", data=dict(data))

    async def owns_handle(self, handle):
        return not handle.startswith("foreign-")


class _PollOnlyProvider:
    """A poll-only Waitable on the generic ``wait.complete`` source."""

    kind = "run"

    def __init__(self):
        self.states: dict[str, Outcome] = {}

    async def poll(self, handle):
        outcome = self.states.get(handle, Outcome.PENDING)
        return WaitStatus(outcome, f"{handle} is {outcome.value}", data={})


def _job_complete_registration() -> SourceRegistration:
    return SourceRegistration(
        name=JOB_SOURCE,
        schema=lambda payload: payload,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=WAIT_COMPLETE_TEMPLATE,
        trust=Trust.TRUSTED,
        log_redaction=RedactionPolicy(summarize=lambda payload: "<redacted>"),
        retention_days=7,
    )


@pytest.fixture
async def rig(tmp_path, sqlite_database_factory):
    backend = SQLiteBackend(str(tmp_path / "signals.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    sources = SourceRegistry()
    sources.register(build_wait_complete_registration())
    sources.register(_job_complete_registration())
    inner = _Agent()
    dispatcher = SignalDispatcher(
        agent=inner,
        registry=sources,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    await dispatcher.initialize_durable_delivery()

    jobs = _JobProvider()
    runs = _PollOnlyProvider()
    waits = WaitRegistry()
    waits.register(jobs)
    waits.register(runs)
    db = await sqlite_database_factory(tmp_path / "agent.db")
    agent = SimpleNamespace(
        did=AGENT_DID,
        agent_id=AGENT_DID,
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=waits,
        dispatcher=dispatcher,
    )
    reconciler = WaitReconciler(agent)
    agent._wait_reconciler = reconciler

    async def tick():
        await reconciler.reconcile()
        pending = [task for task in inner.tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    yield SimpleNamespace(
        agent=agent, dispatcher=dispatcher, jobs=jobs, runs=runs, tick=tick
    )
    pending = [task for task in inner.tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


async def _claim(dispatcher, consumer_id):
    return await dispatcher.claim_durable_delivery(
        consumer_id=consumer_id, executor_id="workflow-runner"
    )


@pytest.mark.asyncio
async def test_registration_targets_the_providers_own_wake(rig):
    registration = await register_wait_resume_consumer(
        rig.agent, "job:42", consumer_id="workflows:wait:run-1"
    )

    assert registration.source == wake_source(rig.jobs) == JOB_SOURCE
    assert registration.agent_id == AGENT_DID
    assert registration.correlation_selector == "payload.ref=job:42"
    assert registration.max_attempts == 0


@pytest.mark.asyncio
async def test_terminal_handle_resumes_its_parked_consumer(rig):
    await register_wait_resume_consumer(
        rig.agent, "job:42", consumer_id="workflows:wait:run-1"
    )
    rig.jobs.set("42", Outcome.PENDING)
    await rig.tick()
    assert await _claim(rig.dispatcher, "workflows:wait:run-1") is None

    rig.jobs.set("42", Outcome.DONE, {"status": "complete"})
    await rig.tick()

    delivery = await _claim(rig.dispatcher, "workflows:wait:run-1")
    assert delivery is not None
    assert delivery.event.source == JOB_SOURCE
    assert delivery.event.payload["ref"] == "job:42"
    assert delivery.event.payload["outcome"] == Outcome.DONE.value
    assert await rig.dispatcher.ack_durable_delivery(
        consumer_id="workflows:wait:run-1",
        delivery_id=delivery.delivery_id,
        lease_token=delivery.lease_token,
    )


@pytest.mark.asyncio
async def test_another_handles_wake_does_not_resume_it(rig):
    await register_wait_resume_consumer(
        rig.agent, "job:42", consumer_id="workflows:wait:run-1"
    )
    rig.jobs.set("43", Outcome.DONE)
    await rig.tick()

    assert await _claim(rig.dispatcher, "workflows:wait:run-1") is None


@pytest.mark.asyncio
async def test_poll_data_cannot_redirect_the_wake(rig):
    """A provider forwarding third-party data must not be able to complete
    another handle's parked work by putting a ``ref`` in its poll data."""
    await register_wait_resume_consumer(
        rig.agent, "job:42", consumer_id="workflows:wait:run-1"
    )
    rig.jobs.set("99", Outcome.DONE, {"ref": "job:42"})
    await rig.tick()

    assert await _claim(rig.dispatcher, "workflows:wait:run-1") is None


@pytest.mark.asyncio
async def test_wake_committed_before_registration_is_backfilled(rig):
    """The dispatch stage returns before the verify stage parks; a job that
    finishes in between must still resume the parked stage."""
    rig.jobs.set("42", Outcome.DONE)
    await rig.tick()

    await register_wait_resume_consumer(
        rig.agent, "job:42", consumer_id="workflows:wait:run-1"
    )

    delivery = await _claim(rig.dispatcher, "workflows:wait:run-1")
    assert delivery is not None
    assert delivery.event.payload["ref"] == "job:42"


@pytest.mark.asyncio
async def test_poll_only_provider_is_armed_and_resumes(rig):
    """A provider with no ``active_handles`` is only reconciled when watched;
    the resume registration arms that watch itself."""
    registration = await register_wait_resume_consumer(
        rig.agent, "run:7", consumer_id="workflows:wait:run-2"
    )
    assert registration.source == "wait.complete"

    rig.runs.states["7"] = Outcome.FAILED
    await rig.tick()

    delivery = await _claim(rig.dispatcher, "workflows:wait:run-2")
    assert delivery is not None
    assert delivery.event.payload["ref"] == "run:7"
    assert delivery.event.payload["outcome"] == Outcome.FAILED.value


@pytest.mark.asyncio
async def test_shared_wait_complete_source_is_correlated_per_kind(rig, monkeypatch):
    """Two kinds on ``wait.complete`` with the same handle string must not
    resume each other's consumers."""
    monkeypatch.setattr(rig.jobs, "signal", None)
    await register_wait_resume_consumer(
        rig.agent, "run:7", consumer_id="workflows:wait:run-2"
    )
    rig.jobs.set("7", Outcome.DONE)
    await rig.tick()

    assert await _claim(rig.dispatcher, "workflows:wait:run-2") is None


@pytest.mark.asyncio
async def test_unknown_kind_fails_before_anything_is_registered(rig):
    with pytest.raises(ValueError, match="no wait provider for kind 'nope'"):
        await register_wait_resume_consumer(
            rig.agent, "nope:1", consumer_id="workflows:wait:run-3"
        )
    assert await rig.dispatcher.list_durable_deliveries() == []


@pytest.mark.asyncio
async def test_foreign_handle_is_rejected(rig):
    with pytest.raises(ValueError, match="does not own it"):
        await register_wait_resume_consumer(
            rig.agent, "job:foreign-1", consumer_id="workflows:wait:run-4"
        )


@pytest.mark.asyncio
async def test_missing_durable_dispatcher_fails_closed(rig):
    rig.agent.dispatcher = SimpleNamespace()
    with pytest.raises(ValueError, match="durable signal delivery unavailable"):
        await register_wait_resume_consumer(
            rig.agent, "job:42", consumer_id="workflows:wait:run-5"
        )
