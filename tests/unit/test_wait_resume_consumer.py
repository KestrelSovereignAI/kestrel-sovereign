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

from kestrel_sovereign.privacy import get_privacy_preset
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

    def __init__(self, kind: str = "run"):
        self.kind = kind
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
    checks = _PollOnlyProvider("ci")
    waits = WaitRegistry()
    waits.register(jobs)
    waits.register(runs)
    waits.register(checks)
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
        agent=agent,
        dispatcher_agent=inner,
        dispatcher=dispatcher,
        jobs=jobs,
        runs=runs,
        checks=checks,
        tick=tick,
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
    result = await register_wait_resume_consumer(
        rig.agent, "job:42", consumer_id="workflows:wait:run-1"
    )
    registration = result.registration

    assert result.already_terminal is None
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
    finishes in between must still resume the parked stage — reported by the
    registration itself and, under normal storage, delivered exactly once."""
    rig.jobs.set("42", Outcome.DONE, {"status": "complete"})
    await rig.tick()

    result = await register_wait_resume_consumer(
        rig.agent, "job:42", consumer_id="workflows:wait:run-1"
    )

    assert result.already_terminal is not None
    assert result.already_terminal.outcome is Outcome.DONE
    delivery = await _claim(rig.dispatcher, "workflows:wait:run-1")
    assert delivery is not None
    assert delivery.event.payload["ref"] == "job:42"
    assert await rig.dispatcher.ack_durable_delivery(
        consumer_id="workflows:wait:run-1",
        delivery_id=delivery.delivery_id,
        lease_token=delivery.lease_token,
    )

    await rig.tick()
    assert await _claim(rig.dispatcher, "workflows:wait:run-1") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["ephemeral", "isolated"])
async def test_elided_wake_before_registration_reports_terminal(rig, preset):
    """Under a payload-eliding privacy mode the stored wake carries only a
    marker, so backfill cannot match it and the reconciler has already
    deduplicated the transition. The registration's post-registration poll
    is then the only thing that tells the parked work its handle finished."""
    rig.dispatcher_agent.privacy_config = get_privacy_preset(preset)
    rig.jobs.set("42", Outcome.DONE, {"status": "complete"})
    await rig.tick()

    result = await register_wait_resume_consumer(
        rig.agent, "job:42", consumer_id="workflows:wait:run-1"
    )

    await rig.tick()
    assert await _claim(rig.dispatcher, "workflows:wait:run-1") is None
    assert result.already_terminal is not None
    assert result.already_terminal.outcome is Outcome.DONE
    assert result.already_terminal.data == {"status": "complete"}


@pytest.mark.asyncio
async def test_post_registration_poll_failure_is_not_reported_as_running(
    rig, monkeypatch
):
    """A poll that raises after registering leaves the handle's state
    unknown; it must surface rather than read as "still running"."""

    async def broken_poll(handle):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(rig.jobs, "poll", broken_poll)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await register_wait_resume_consumer(
            rig.agent, "job:42", consumer_id="workflows:wait:run-1"
        )


@pytest.mark.asyncio
async def test_poll_only_provider_is_armed_and_resumes(rig):
    """A provider with no ``active_handles`` is only reconciled when watched;
    the resume registration arms that watch itself."""
    result = await register_wait_resume_consumer(
        rig.agent, "run:7", consumer_id="workflows:wait:run-2"
    )
    assert result.registration.source == "wait.complete"
    assert result.already_terminal is None

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


async def _watching(agent, kind, handle) -> bool:
    state = await agent._wait_reconciler._store.get(kind, handle)
    return state is not None and state.watching == 1


@pytest.mark.asyncio
async def test_watch_is_armed_before_the_consumer_is_written(rig, monkeypatch):
    """A consumer with no watch behind it is an unrecoverable stall for a
    poll-only provider, so the watch must already be durable when the
    consumer row is written."""
    real_register = rig.dispatcher.register_durable_consumer
    watched_at_register: list[bool] = []

    async def observing_register(registration):
        watched_at_register.append(await _watching(rig.agent, "run", "7"))
        return await real_register(registration)

    monkeypatch.setattr(
        rig.dispatcher, "register_durable_consumer", observing_register
    )
    await register_wait_resume_consumer(
        rig.agent, "run:7", consumer_id="workflows:wait:run-2"
    )

    assert watched_at_register == [True]


@pytest.mark.asyncio
async def test_interrupted_consumer_write_leaves_the_watch_and_retry_recovers(
    rig, monkeypatch
):
    """An interruption at the consumer write leaves only a harmless watch;
    retrying the idempotent registration completes it and the parked work
    still resumes."""
    real_register = rig.dispatcher.register_durable_consumer

    async def interrupted_register(registration):
        raise RuntimeError("process stopped")

    monkeypatch.setattr(
        rig.dispatcher, "register_durable_consumer", interrupted_register
    )
    with pytest.raises(RuntimeError, match="process stopped"):
        await register_wait_resume_consumer(
            rig.agent, "run:7", consumer_id="workflows:wait:run-2"
        )
    assert await _watching(rig.agent, "run", "7")

    monkeypatch.setattr(rig.dispatcher, "register_durable_consumer", real_register)
    result = await register_wait_resume_consumer(
        rig.agent, "run:7", consumer_id="workflows:wait:run-2"
    )
    assert result.already_terminal is None

    rig.runs.states["7"] = Outcome.DONE
    await rig.tick()
    delivery = await _claim(rig.dispatcher, "workflows:wait:run-2")
    assert delivery is not None
    assert delivery.event.payload["ref"] == "run:7"


@pytest.mark.asyncio
async def test_missing_durable_dispatcher_arms_no_watch(rig):
    rig.agent.dispatcher = SimpleNamespace()
    with pytest.raises(ValueError, match="durable signal delivery unavailable"):
        await register_wait_resume_consumer(
            rig.agent, "run:7", consumer_id="workflows:wait:run-5"
        )
    assert not await _watching(rig.agent, "run", "7")


async def _ack(dispatcher, consumer_id, delivery):
    assert await dispatcher.ack_durable_delivery(
        consumer_id=consumer_id,
        delivery_id=delivery.delivery_id,
        lease_token=delivery.lease_token,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ref, provider_name",
    [("ci:owner/repo#12345", "checks"), ("job:job-12345", "jobs")],
)
async def test_anonymized_storage_keeps_the_wake_correlatable(
    rig, ref, provider_name
):
    """ANONYMOUS storage persists a PII-anonymized payload, and selectors
    match that stored projection. A 5-digit run in a handle reads as a ZIP
    code, so an anonymized ``payload.ref`` would never match the consumer's
    raw selector and the parked work would silently never resume."""
    rig.dispatcher_agent.privacy_config = get_privacy_preset("anonymous")
    kind, handle = ref.split(":", 1)
    consumer_id = f"workflows:wait:{kind}"
    result = await register_wait_resume_consumer(
        rig.agent, ref, consumer_id=consumer_id
    )
    assert result.already_terminal is None

    provider = getattr(rig, provider_name)
    if provider_name == "jobs":
        provider.set(handle, Outcome.DONE, {"status": "complete"})
    else:
        provider.states[handle] = Outcome.DONE
    await rig.tick()

    delivery = await _claim(rig.dispatcher, consumer_id)
    assert delivery is not None
    assert delivery.event.payload["ref"] == ref
    # Only the correlation key is exempt: the rest is still anonymized.
    assert delivery.event.payload["handle"] != handle
    assert "[ZIP_REDACTED]" in delivery.event.payload["handle"]
    await _ack(rig.dispatcher, consumer_id, delivery)

    await rig.tick()
    assert await _claim(rig.dispatcher, consumer_id) is None


@pytest.mark.asyncio
async def test_anonymized_wake_before_registration_is_backfilled(rig):
    """Backfill reads the stored, anonymized event, so it too needs the
    correlation key to survive the projection."""
    rig.dispatcher_agent.privacy_config = get_privacy_preset("anonymous")
    rig.jobs.set("job-12345", Outcome.DONE, {"status": "complete"})
    await rig.tick()

    result = await register_wait_resume_consumer(
        rig.agent, "job:job-12345", consumer_id="workflows:wait:run-6"
    )

    assert result.already_terminal is not None
    delivery = await _claim(rig.dispatcher, "workflows:wait:run-6")
    assert delivery is not None
    assert delivery.event.payload["ref"] == "job:job-12345"
    await _ack(rig.dispatcher, "workflows:wait:run-6", delivery)

    await rig.tick()
    assert await _claim(rig.dispatcher, "workflows:wait:run-6") is None
