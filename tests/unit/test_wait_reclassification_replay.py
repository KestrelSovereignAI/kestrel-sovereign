"""A provider's classifier change must not replay delivered wakes as news (#3390).

On 2026-09-29 kestrel-feature-talon 0.2.8 began classifying a clean exit by
the run's disposition, so jobs whose ledger rows held ``done:complete`` polled
as ``partial:complete``. The reconciler compared the whole outcome token, read
each as a new transition, and re-delivered weeks-old completions with
``delivery_attempt=1`` and empty prior-delivery fields — the exact payload of
a first delivery. Two of them carried Talon's same-turn instructions (answer
the clarification, re-dispatch the claim) for work finished five weeks
earlier. Nothing acted on them only because each was checked by hand.

Three guarantees, each pinned here against the real reconciler and store:

* a delivered row whose event is unchanged — same native status, or same
  event identity — but whose outcome class changed is re-keyed silently and
  audited, for any provider;
* a wake re-emitted for a handle that was already dispatched carries that
  dispatch's real attempt count, status and time, never the empty fields of
  a first delivery;
* a wake whose event the provider dates before the handle's last delivered
  wake is announced on ``wait.replay``, which says it is a replay and carries
  none of the provider's act-now instructions.

The replay prompt is checked as rendered by a real ``SignalDispatcher``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import (
    AttentionPolicy,
    RateLimit,
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Status,
    Trust,
)
from kestrel_sdk.tools import Outcome

from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.wait_replay import (
    SOURCE_NAME as WAIT_REPLAY_SOURCE,
    build_wait_replay_registration,
)
from kestrel_sovereign.storage.async_wait_signal_store import (
    REKEY_RECLASSIFIED,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.waits.engine import (
    TERMINAL_EVENT_AT_KEY,
    TERMINAL_EVENT_KEY,
    WaitRegistry,
)
from kestrel_sovereign.waits.reconciler import WaitReconciler
from tests.unit import test_wait_reconciler as doubles

AGENT_DID = "did:test:agent"
TALON_SOURCE = "talon.job_complete"
# A job that exited on 2026-08-24, long before any wake in these tests.
EXITED_AT = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


class _Jobs(doubles._FakeProvider):
    """A monitorable provider shaped like Talon: one handle per job, its own
    wake source, and a native ``status`` read from the job record."""

    kind = "talon"

    def __init__(self):
        super().__init__(signal=TALON_SOURCE)


def _job(status: str, **extra) -> dict:
    return {"job_id": "job", "status": status, **extra}


async def _build_agent(db, *providers, dispatcher=None):
    registry = WaitRegistry()
    for provider in providers:
        registry.register(provider)
    agent = SimpleNamespace(
        did=AGENT_DID,
        agent_id=AGENT_DID,
        _raw_storage=SimpleNamespace(db=db),
        wait_registry=registry,
        dispatcher=dispatcher or doubles._CapturingDispatcher(),
    )
    agent._wait_reconciler = WaitReconciler(agent)
    return agent


@pytest.fixture
def make_agent(tmp_path, sqlite_database_factory):
    async def create(*providers, dispatcher=None, db_path=None):
        db = await sqlite_database_factory(db_path or tmp_path / "agent.db")
        return await _build_agent(db, *providers, dispatcher=dispatcher)

    return create


async def _settle(agent):
    """Two ticks: detect + enqueue, then harvest. Returns the harvest tick."""
    await agent._wait_reconciler.reconcile()
    return await agent._wait_reconciler.reconcile()


def _wakes(agent):
    return agent.dispatcher.signals


# ---------------------------------------------------------------------------
# Silent, audited reclassification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("delivered", "relabelled", "tokens"), [
    # Talon 0.2.8: a clean exit is now classified by the run's disposition.
    (
        (Outcome.DONE, _job("complete")),
        (Outcome.PARTIAL, _job("complete")),
        ("done:complete", "partial:complete"),
    ),
    (
        (Outcome.FAILED, _job("failed")),
        (Outcome.PARTIAL, _job("failed")),
        ("failed:failed", "partial:failed"),
    ),
])
async def test_an_outcome_only_reclassification_wakes_nobody_and_is_audited(
    make_agent, delivered, relabelled, tokens,
):
    jobs = _Jobs()
    jobs.set("j1", delivered[0], data=delivered[1])
    agent = await make_agent(jobs)
    store = agent._wait_reconciler._store
    await _settle(agent)
    assert len(_wakes(agent)) == 1

    jobs.set("j1", relabelled[0], data=relabelled[1])
    tick = await agent._wait_reconciler.reconcile()

    assert len(_wakes(agent)) == 1, "a re-labelled event is not news"
    assert tick.data["signals_rekeyed"] == 1
    assert tick.data["signals_enqueued"] == 0
    row = await store.get("talon", "j1")
    assert row.last_signaled_outcome == tokens[1], "the ledger follows the label"
    assert row.pending_signal_id is None
    [rekey] = await store.list_rekeys("talon", "j1")
    assert (rekey.previous_token, rekey.token, rekey.reason) == (
        tokens[0], tokens[1], REKEY_RECLASSIFIED,
    )

    # Re-keyed once: later polls of the same label are plain dedup.
    for _ in range(3):
        await _settle(agent)
    assert len(_wakes(agent)) == 1
    assert len(await store.list_rekeys("talon", "j1")) == 1


@pytest.mark.asyncio
async def test_a_reclassified_named_event_is_audited_too(make_agent):
    """For a provider that names its events the outcome was already no part
    of the identity (#3399); the re-label is now recorded, not just absorbed."""
    jobs = _Jobs()
    jobs.set("j1", Outcome.FAILED, data=_job("blocked", **{TERMINAL_EVENT_KEY: "job@t0"}))
    agent = await make_agent(jobs)
    await _settle(agent)

    jobs.set(
        "j1", Outcome.PARTIAL,
        data=_job("clarifying", **{TERMINAL_EVENT_KEY: "job@t0"}),
    )
    await _settle(agent)

    assert len(_wakes(agent)) == 1
    [rekey] = await agent._wait_reconciler._store.list_rekeys()
    assert rekey.reason == REKEY_RECLASSIFIED


@pytest.mark.asyncio
async def test_a_reclassification_of_an_undelivered_wake_sends_no_second_one(
    make_agent,
):
    """The wake for the old label is still in flight when the label changes.
    It is the same event, so it is not emitted again; once delivered, the row
    is re-keyed to the new label."""
    jobs = _Jobs()
    jobs.set("j1", Outcome.DONE, data=_job("complete"))
    dispatcher = doubles._CapturingDispatcher(pending=True)
    agent = await make_agent(jobs, dispatcher=dispatcher)
    store = agent._wait_reconciler._store
    await agent._wait_reconciler.reconcile()

    jobs.set("j1", Outcome.PARTIAL, data=_job("complete"))
    await agent._wait_reconciler.reconcile()
    dispatcher.release()
    await asyncio.sleep(0)
    await _settle(agent)
    await _settle(agent)

    assert len(_wakes(agent)) == 1
    row = await store.get("talon", "j1")
    assert row.last_signaled_outcome == "partial:complete"
    assert [r.reason for r in await store.list_rekeys()] == [REKEY_RECLASSIFIED]


@pytest.mark.asyncio
async def test_without_a_native_status_an_outcome_change_is_still_news(make_agent):
    """The outcome class is set aside only when a native status shows the
    record did not change. A provider exposing none has nothing else to go
    on, so a new outcome stays a new transition."""
    jobs = _Jobs()
    jobs.set("j1", Outcome.DONE, data={"job_id": "j1"})
    agent = await make_agent(jobs)
    await _settle(agent)

    jobs.set("j1", Outcome.FAILED, data={"job_id": "j1"})
    await _settle(agent)

    assert [w.payload["outcome"] for w in _wakes(agent)] == ["done", "failed"]
    assert await agent._wait_reconciler._store.list_rekeys() == []


# ---------------------------------------------------------------------------
# A genuinely new terminal event still wakes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "then"), [
    # A late exit sidecar corrects the native status (codex Wave 2 P2).
    ((Outcome.FAILED, _job("finished_unknown")), (Outcome.FAILED, _job("failed"))),
    # ...even when it is also reclassified.
    ((Outcome.FAILED, _job("finished_unknown")), (Outcome.PARTIAL, _job("complete"))),
    # A re-run that fails again names a new event with the same outcome.
    (
        (Outcome.FAILED, _job("failed", **{TERMINAL_EVENT_KEY: "run-1"})),
        (Outcome.FAILED, _job("failed", **{TERMINAL_EVENT_KEY: "run-2"})),
    ),
])
async def test_a_genuinely_new_terminal_event_on_the_same_handle_still_wakes(
    make_agent, first, then,
):
    jobs = _Jobs()
    jobs.set("j1", first[0], data=first[1])
    agent = await make_agent(jobs)
    await _settle(agent)

    jobs.set("j1", then[0], data=then[1])
    await _settle(agent)

    assert len(_wakes(agent)) == 2
    assert await agent._wait_reconciler._store.list_rekeys() == []


@pytest.mark.asyncio
async def test_after_a_silent_reclassification_the_next_real_event_wakes(
    make_agent,
):
    jobs = _Jobs()
    jobs.set("j1", Outcome.FAILED, data=_job("finished_unknown"))
    agent = await make_agent(jobs)
    await _settle(agent)
    jobs.set("j1", Outcome.PARTIAL, data=_job("finished_unknown"))
    await _settle(agent)
    assert len(_wakes(agent)) == 1

    jobs.set("j1", Outcome.FAILED, data=_job("failed"))
    await _settle(agent)

    assert len(_wakes(agent)) == 2
    assert _wakes(agent)[1].payload["status"] == "failed"


# ---------------------------------------------------------------------------
# Honest delivery metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reemitted_wake_carries_the_real_prior_attempt_count_and_status(
    make_agent,
):
    """The first transition takes three dispatches to deliver. A wake for a
    later transition of the same handle is attempt 1 of ITS transition — the
    retry cap counts per transition (#3105) — but it reports that earlier
    delivery as it was, not as the empty fields of a first delivery."""
    jobs = _Jobs()
    jobs.set("j1", Outcome.FAILED, data=_job("finished_unknown"))
    agent = await make_agent(jobs)
    store = agent._wait_reconciler._store
    agent.dispatcher._status = Status.DROPPED_RATE_LIMIT
    await agent._wait_reconciler.reconcile()   # attempt 1
    await agent._wait_reconciler.reconcile()   # soft fail, attempt 2
    agent.dispatcher._status = Status.OK
    await agent._wait_reconciler.reconcile()   # soft fail, attempt 3
    await agent._wait_reconciler.reconcile()   # delivered
    delivered = await store.get("talon", "j1")
    assert delivered.last_signaled_outcome == "failed:finished_unknown"
    assert delivered.last_delivery_attempts == 3

    jobs.set("j1", Outcome.FAILED, data=_job("failed"))
    agent.dispatcher._status = Status.DROPPED_QUIET_HOURS
    await agent._wait_reconciler.reconcile()

    news = _wakes(agent)[-1].payload
    assert news["delivery_attempt"] == 1
    assert news["delivery_retry"] is False
    assert news["delivery_previous_attempts"] == 3
    assert news["delivery_previous_status"] == "ok_unbound"
    assert news["delivery_previous_attempt_at"] == str(
        delivered.last_attempt_started_at
    )
    assert news["delivery_last_delivered_at"] == (
        delivered.delivered_at_utc().isoformat()
    )

    agent.dispatcher._status = Status.OK
    await agent._wait_reconciler.reconcile()   # soft fail harvested, attempt 2

    retry = _wakes(agent)[-1].payload
    assert retry["delivery_attempt"] == 2
    assert retry["delivery_retry"] is True
    assert retry["delivery_previous_attempts"] == 1
    assert retry["delivery_previous_status"] == "dropped_quiet_hours"
    assert retry["delivery_last_delivered_at"] == news["delivery_last_delivered_at"], (
        "a later transition's attempts never move the last delivery"
    )


# ---------------------------------------------------------------------------
# Replays: an event older than the handle's last delivered wake
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_event_older_than_the_last_delivery_is_announced_as_a_replay(
    make_agent,
):
    jobs = _Jobs()
    jobs.set("j1", Outcome.FAILED, data=_job(
        "finished_unknown", **{TERMINAL_EVENT_AT_KEY: EXITED_AT.isoformat()},
    ))
    agent = await make_agent(jobs)
    await _settle(agent)
    assert _wakes(agent)[0].source == TALON_SOURCE, (
        "an event newer than any delivery is not a replay"
    )

    # The exit sidecar lands late: a new transition, but the exit it reports
    # happened before the wake already delivered for this job.
    agent.dispatcher._status = Status.DROPPED_RATE_LIMIT
    jobs.set("j1", Outcome.PARTIAL, data=_job(
        "complete", disposition="clarifying",
        **{TERMINAL_EVENT_AT_KEY: EXITED_AT.isoformat()},
    ))
    await _settle(agent)                        # attempt 1, soft-failed; attempt 2

    replays = _wakes(agent)[1:]
    assert len(replays) == 2
    for wake in replays:
        assert wake.source == WAIT_REPLAY_SOURCE
        assert wake.payload["delivery_replay"] is True
        assert wake.payload[TERMINAL_EVENT_AT_KEY] == EXITED_AT.isoformat()
        assert wake.payload["delivery_last_delivered_at"]
        assert wake.payload["disposition"] == "clarifying", (
            "the facts travel; only the instructions are withheld"
        )
    assert replays[1].payload["delivery_attempt"] == 2, "still a replay on retry"


@pytest.mark.asyncio
async def test_an_event_after_the_last_delivery_is_not_a_replay(make_agent):
    jobs = _Jobs()
    jobs.set("j1", Outcome.FAILED, data=_job(
        "finished_unknown", **{TERMINAL_EVENT_AT_KEY: EXITED_AT.isoformat()},
    ))
    agent = await make_agent(jobs)
    await _settle(agent)

    later = datetime.now(timezone.utc) + timedelta(seconds=1)
    jobs.set("j1", Outcome.FAILED, data=_job(
        "failed", **{TERMINAL_EVENT_AT_KEY: later.isoformat()},
    ))
    await _settle(agent)

    news = _wakes(agent)[1]
    assert news.source == TALON_SOURCE
    assert news.payload["delivery_replay"] is False


@pytest.mark.asyncio
async def test_a_datetime_event_time_is_accepted_and_carried_as_iso(make_agent):
    jobs = _Jobs()
    jobs.set("j1", Outcome.FAILED, data=_job("finished_unknown"))
    agent = await make_agent(jobs)
    await _settle(agent)

    naive_utc = EXITED_AT.replace(tzinfo=None)
    jobs.set("j1", Outcome.FAILED, data=_job(
        "failed", **{TERMINAL_EVENT_AT_KEY: naive_utc},
    ))
    await agent._wait_reconciler.reconcile()

    replay = _wakes(agent)[1]
    assert replay.source == WAIT_REPLAY_SOURCE
    assert replay.payload[TERMINAL_EVENT_AT_KEY] == EXITED_AT.isoformat(), (
        "naive is UTC, and the durable payload needs a string"
    )


@pytest.mark.asyncio
async def test_an_unreadable_event_time_judges_nothing(make_agent, caplog):
    jobs = _Jobs()
    jobs.set("j1", Outcome.FAILED, data=_job("finished_unknown"))
    agent = await make_agent(jobs)
    await _settle(agent)

    jobs.set("j1", Outcome.FAILED, data=_job(
        "failed", **{TERMINAL_EVENT_AT_KEY: "last tuesday"},
    ))
    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.waits.reconciler"):
        await agent._wait_reconciler.reconcile()

    news = _wakes(agent)[1]
    assert news.source == TALON_SOURCE
    assert news.payload["delivery_replay"] is False
    assert any("last tuesday" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_handle_never_delivered_has_no_replay(make_agent):
    jobs = _Jobs()
    jobs.set("j1", Outcome.DONE, data=_job(
        "complete", **{TERMINAL_EVENT_AT_KEY: EXITED_AT.isoformat()},
    ))
    agent = await make_agent(jobs)
    await agent._wait_reconciler.reconcile()

    first = _wakes(agent)[0]
    assert first.source == TALON_SOURCE
    assert first.payload["delivery_replay"] is False
    assert first.payload["delivery_last_delivered_at"] == ""


# ---------------------------------------------------------------------------
# The rendered prompt, through a real dispatcher
# ---------------------------------------------------------------------------

ACT_NOW = "ACT NOW: answer the question and re-dispatch the claim in this same turn."


class _PromptAgent(EventManagerMixin):
    """A dispatcher agent that records the prompt each COGNITION wake renders."""

    did = AGENT_DID
    agent_name = "kestrel"

    def __init__(self):
        self._event_listeners: list = []
        self._pending_task_notifications: list = []
        self.background_tasks: list[asyncio.Task] = []
        self.prompts: list[str] = []

    async def process_input(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        return "Wake turn ran."

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


def _talon_like_registration(template: Path) -> SourceRegistration:
    return SourceRegistration(
        name=TALON_SOURCE,
        schema=lambda payload: payload,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=template,
        trust=Trust.TRUSTED,
        rate_limit=RateLimit(),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        log_redaction=RedactionPolicy(
            summarize=lambda payload: "talon-like wake",
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=1,
    )


@pytest.fixture
async def dispatch_rig(tmp_path, sqlite_database_factory):
    backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()

    template = tmp_path / "talon_like.md"
    template.write_text(
        "[JOB_COMPLETE] Job `{payload[handle]}` finished: {payload[summary]}\n\n"
        f"{ACT_NOW}\n\npayload={{payload}}\n",
        encoding="utf-8",
    )
    sources = SourceRegistry()
    sources.register(_talon_like_registration(template))
    sources.register(build_wait_replay_registration())

    agent = _PromptAgent()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=sources,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    jobs = _Jobs()
    db = await sqlite_database_factory(tmp_path / "agent.db")
    reconcile_agent = await _build_agent(db, jobs, dispatcher=dispatcher)

    async def cycle():
        for _ in range(2):
            await reconcile_agent._wait_reconciler.reconcile()
            pending = [t for t in agent.background_tasks if not t.done()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    yield SimpleNamespace(agent=agent, jobs=jobs, cycle=cycle)

    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


@pytest.mark.asyncio
async def test_a_replay_renders_without_the_providers_act_now_instructions(
    dispatch_rig,
):
    rig = dispatch_rig
    rig.jobs.set("j1", Outcome.FAILED, summary="exit status unknown", data=_job(
        "finished_unknown", **{TERMINAL_EVENT_AT_KEY: EXITED_AT.isoformat()},
    ))
    await rig.cycle()
    [first] = rig.agent.prompts
    assert ACT_NOW in first, "a fresh event renders through the provider's prompt"

    rig.jobs.set("j1", Outcome.PARTIAL, summary="stopped to ask", data=_job(
        "complete", disposition="clarifying",
        **{TERMINAL_EVENT_AT_KEY: EXITED_AT.isoformat()},
    ))
    await rig.cycle()

    replay = rig.agent.prompts[1]
    assert ACT_NOW not in replay
    assert "[WAIT_REPLAY]" in replay
    assert "This is a REPLAY, not news." in replay
    assert EXITED_AT.isoformat() in replay
    assert "have been withheld" in replay

    # A genuinely later event on the same handle is news again.
    later = datetime.now(timezone.utc) + timedelta(seconds=1)
    rig.jobs.set("j1", Outcome.FAILED, summary="re-run failed", data=_job(
        "failed", **{TERMINAL_EVENT_AT_KEY: later.isoformat()},
    ))
    await rig.cycle()

    assert len(rig.agent.prompts) == 3
    assert ACT_NOW in rig.agent.prompts[2]


# ---------------------------------------------------------------------------
# Existing ledger rows after the upgrade
# ---------------------------------------------------------------------------

# The shape of Emma's ledger on 2026-09-29: 112 ``done:complete`` rows, the
# 34 replayed ones now ``partial:complete``, and 52 ``failed:failed``.
_LEGACY_ROWS = {
    "done-job": "done:complete",
    "partial-job": "partial:complete",
    "failed-job": "failed:failed",
}


def _write_pre_3390_ledger(db_path: Path) -> None:
    import sqlite3

    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE wait_signal_state (
            agent_id TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            handle TEXT NOT NULL,
            last_signaled_outcome TEXT,
            last_delivery_status TEXT,
            last_surface_status TEXT,
            last_delivery_error TEXT,
            last_delivery_attempts INTEGER NOT NULL DEFAULT 0,
            last_delivery_attempt_at TIMESTAMP,
            attempts_signaled_target TEXT NOT NULL DEFAULT '',
            last_attempt_started_at TIMESTAMP,
            delivery_deferred_until TIMESTAMP,
            delivery_deferrals INTEGER NOT NULL DEFAULT 0,
            pending_signal_id TEXT,
            pending_signaled_target TEXT,
            pending_signal_enqueued_at TIMESTAMP,
            watching INTEGER NOT NULL DEFAULT 0,
            watch_baseline TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (agent_id, kind, handle)
        );
        """
    )
    for handle, token in _LEGACY_ROWS.items():
        legacy.execute(
            "INSERT INTO wait_signal_state (agent_id, kind, handle, "
            "last_signaled_outcome, last_delivery_status, "
            "last_delivery_attempts, last_delivery_attempt_at, "
            "attempts_signaled_target, last_attempt_started_at) "
            "VALUES (?, 'talon', ?, ?, 'ok_queued', 1, "
            "'2026-09-29 14:30:00', ?, '2026-09-29 14:27:00')",
            (AGENT_DID, handle, token, token),
        )
    legacy.commit()
    legacy.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("relabel", [Outcome.DONE, Outcome.PARTIAL, Outcome.FAILED])
async def test_existing_ledger_rows_do_not_replay_after_the_change(
    tmp_path, make_agent, relabel,
):
    """Every delivered row, polled under whatever outcome class a classifier
    assigns it next, with its native status unchanged: no wake, one audited
    re-key per real change of label."""
    db_path = tmp_path / "legacy.db"
    _write_pre_3390_ledger(db_path)
    jobs = _Jobs()
    for handle, token in _LEGACY_ROWS.items():
        native = token.split(":", 1)[1]
        jobs.set(handle, relabel, data=_job(
            native, **{TERMINAL_EVENT_AT_KEY: EXITED_AT.isoformat()},
        ))
    agent = await make_agent(jobs, db_path=db_path)
    store = agent._wait_reconciler._store

    for _ in range(3):
        await _settle(agent)

    assert _wakes(agent) == []
    relabelled = {
        handle for handle, token in _LEGACY_ROWS.items()
        if not token.startswith(f"{relabel.value}:")
    }
    rekeys = await store.list_rekeys("talon")
    assert {r.handle for r in rekeys} == relabelled
    assert all(r.reason == REKEY_RECLASSIFIED for r in rekeys)
    for handle, token in _LEGACY_ROWS.items():
        row = await store.get("talon", handle)
        assert row.last_signaled_outcome == f"{relabel.value}:{token.split(':', 1)[1]}"
        assert row.last_delivered_at == "2026-09-29 14:30:00", (
            "the migration dated each delivered row from its harvest"
        )
