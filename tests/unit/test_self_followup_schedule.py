"""Agent-authored one-shot follow-up turns (#3101).

An intention formed inside a turn ("verify PR N once CI settles, then merge")
used to die with the turn: nothing carried it across the boundary. These tests
cover the substrate that does, end to end through the real SchedulerFeature,
SchedulerRunner and SignalDispatcher rather than through mocks — the defect
being fixed is precisely a path that *looks* wired and produces no turn, which
a mock-shaped test would happily reproduce.

Two properties carry the issue's constraint that an accept which does nothing
is worse than an explicit refusal:

* a persisted follow-up really fires a turn, exactly once, with the intention
  text in it; and
* every way of persisting one that *could not* fire a turn is refused at
  schedule time, and a turn dropped at dispatch is recorded ``missed`` rather
  than filed alongside genuine successes.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from kestrel_sdk.signals import Signal, SignalMode, Status, Visibility
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.agent.sleep import SleepMixin
from kestrel_sovereign.features.scheduler.constants import (
    MISSED_COGNITION_STATUS,
)
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.scheduler import (
    build_cron_registrations,
    cron_source_name,
)
from kestrel_sovereign.signals.sources.self_followup import (
    TASK_NAME as SELF_FOLLOWUP,
    MAX_INTENT_CHARS,
    SelfFollowupIntentError,
    normalize_intent,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend

SOURCE = cron_source_name(SELF_FOLLOWUP)

# A distinctive string so "the intention reached the turn" is a real
# observation about prompt content, not an argument-identity coincidence.
SENTINEL = "verify-PR-3096-CI-then-merge-XYZZY"


class _FakeAgent(SleepMixin):
    """Minimal agent that records the turns a dispatch actually produced."""

    did = "did:test:self-followup"
    agent_name = "followup-test"

    def __init__(self):
        self.background_tasks = []
        self.sleep_hooks = []
        self.turn_prompts: list[str] = []
        self.turn_kwargs: list[dict] = []
        self.turn_session_id: str | None = None

    async def process_input(self, prompt, **kwargs):
        self.turn_prompts.append(prompt)
        self.turn_kwargs.append(kwargs)
        return "follow-up handled"

    def get_turn_bound_session_id(self):
        return self.turn_session_id

    def _track_background_task(self, coro, *, name):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest_asyncio.fixture
async def followup_env(tmp_path):
    """Real dispatcher + real scheduler runner + real SQLite, wired together."""
    backend = SQLiteBackend(str(tmp_path / "self_followup.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()

    registry = SourceRegistry()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    agent.dispatcher = dispatcher
    agent.signal_registry = registry

    async def _lookup(name, args):  # no cron tool is exercised here
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(tool_lookup=_lookup):
        registry.register(registration)

    db = AsyncDatabase(backend)
    feature = SchedulerFeature(agent)
    feature._db = db
    feature._agent_id = agent.did

    runner = SchedulerRunner(db, agent.did, feature._dispatch_scheduled_task)
    await runner._ensure_tables()

    yield agent, feature, runner, db, backend

    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


async def _drain(agent):
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _schedule(feature, *, intent=SENTINEL, seconds_ago=5, **kwargs):
    """Persist a follow-up already due, through the real tool."""
    run_at = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    ).isoformat()
    return await feature.schedule_add_deadline(
        run_at=run_at,
        task_name=SELF_FOLLOWUP,
        args_json=json.dumps({"intent": intent}),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Requirement 5 — it fires, exactly once, carrying the intention
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduled_follow_up_fires_exactly_once_with_intent_in_the_turn(
    followup_env,
):
    """The headline contract: enqueue an intention, get a real turn with it.

    Asserts all three halves the issue names — a genuine cognition turn (not a
    ping, not an echo), the intention text actually inside that turn's prompt,
    and exactly once even though the runner ticks again afterwards.
    """
    agent, feature, runner, db, backend = followup_env

    created = await _schedule(feature)
    assert created.status is ToolResultStatus.OK
    schedule_id = created.data["task_id"]

    await runner._tick()
    await _drain(agent)

    # A real turn ran, and the intention reached it.
    assert len(agent.turn_prompts) == 1, "the scheduled follow-up did not fire"
    assert SENTINEL in agent.turn_prompts[0]

    # It was a COGNITION dispatch, logged as such.
    signal_row = await backend.fetch_one(
        "SELECT source, mode, status FROM signal_log WHERE source = ?",
        (SOURCE,),
    )
    assert signal_row is not None
    assert signal_row[0] == SOURCE
    assert signal_row[1] == SignalMode.COGNITION.value
    assert signal_row[2] == Status.OK.value

    # The occurrence is recorded as a success, and the one-shot row is spent.
    execution = await db.fetchone(
        "SELECT status FROM task_execution_log WHERE task_id = ?",
        (schedule_id,),
    )
    assert execution is not None and execution[0] == "success"
    schedule_row = await db.fetchone(
        "SELECT enabled, terminal_status FROM scheduled_tasks WHERE id = ?",
        (schedule_id,),
    )
    assert schedule_row[0] == 0, "a one-shot follow-up must not stay enabled"

    # Exactly once: a second tick must not re-fire a spent one-shot.
    await runner._tick()
    await _drain(agent)
    assert len(agent.turn_prompts) == 1


@pytest.mark.asyncio
async def test_intent_is_not_stored_raw_in_the_signal_audit(followup_env):
    """An intention routinely names unreleased work; the audit keeps a digest.

    The row still has to prove *which* intention it was, so the digest and
    length are present even though the body is not.
    """
    agent, feature, runner, _db, backend = followup_env

    await _schedule(feature)
    await runner._tick()
    await _drain(agent)

    row = await backend.fetch_one(
        "SELECT payload_redacted, payload_digest FROM signal_log WHERE source = ?",
        (SOURCE,),
    )
    assert row is not None
    redacted = row[0] or ""
    assert SENTINEL not in redacted
    assert "intent_sha256_12=" in redacted
    assert f"intent_len={len(SENTINEL)}" in redacted
    assert row[1], "the audit row must still carry a payload digest"


# ---------------------------------------------------------------------------
# Requirement 4 — a dropped turn is visible, not silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dropped_cognition_occurrence_is_recorded_missed_not_success(
    followup_env,
):
    """A wake the dispatcher drops promised a turn and produced none.

    Filing that as ``success`` next to genuine ones is the silent no-op this
    feature exists to prevent, so it gets its own terminal status.
    """
    agent, feature, runner, db, _backend = followup_env

    created = await _schedule(feature)
    schedule_id = created.data["task_id"]

    async def _dropped(signal):
        from kestrel_sdk.signals import SignalResult

        return SignalResult(
            signal_id=signal.id,
            status=Status.DROPPED_RATE_LIMIT,
            mode=SignalMode.COGNITION,
            duration_ms=0,
        )

    agent.dispatcher.dispatch_signal = _dropped

    await runner._tick()
    await _drain(agent)

    assert agent.turn_prompts == [], "no turn should have run"
    execution = await db.fetchone(
        "SELECT status, result_text FROM task_execution_log WHERE task_id = ?",
        (schedule_id,),
    )
    assert execution is not None
    assert execution[0] == MISSED_COGNITION_STATUS
    assert execution[0] != "success"
    assert "missed" in (execution[1] or "").lower()


@pytest.mark.asyncio
async def test_dropped_action_occurrence_is_still_a_benign_skip(followup_env):
    """The `missed` status is COGNITION-only.

    An ACTION drop (rate limit on a maintenance sweep) genuinely is benign;
    widening `missed` to cover it would make the new signal meaningless.
    """
    _agent, feature, _runner, _db, _backend = followup_env

    from kestrel_sdk.signals import SignalResult

    result = SignalResult(
        signal_id="sig-1",
        status=Status.DROPPED_RATE_LIMIT,
        mode=SignalMode.ACTION,
        duration_ms=0,
    )
    translated = feature._translate_signal_result(result, "trash_retention")
    assert isinstance(translated, str)
    assert translated.startswith("skipped:")


# ---------------------------------------------------------------------------
# Session binding — bound renders, unattended stays internal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_follow_up_scheduled_in_a_chat_turn_returns_to_that_window(
    followup_env,
):
    """Bound wakes need session_id AND USER_VISIBLE or they render nowhere."""
    agent, feature, runner, _db, _backend = followup_env
    agent.turn_session_id = "session-abc"

    created = await _schedule(feature)
    assert created.data["self_followup"]["session_bound"] is True
    assert created.data["self_followup"]["delivery"] == "user_visible"

    captured: list[Signal] = []
    real_dispatch = agent.dispatcher.dispatch_signal

    async def _capture(signal):
        captured.append(signal)
        return await real_dispatch(signal)

    agent.dispatcher.dispatch_signal = _capture

    # The follow-up fires outside the originating turn, so the live accessor
    # no longer answers: the session must come from the persisted row.
    agent.turn_session_id = None
    await runner._tick()
    await _drain(agent)

    assert len(captured) == 1
    assert captured[0].session_id == "session-abc"
    assert captured[0].visibility == Visibility.USER_VISIBLE
    assert agent.turn_kwargs[0].get("session_id") == "session-abc"


@pytest.mark.asyncio
async def test_unattended_follow_up_stays_internal(followup_env):
    """No chat window to return to: INTERNAL and log-only, never a guess."""
    agent, feature, runner, _db, _backend = followup_env
    agent.turn_session_id = None

    created = await _schedule(feature)
    assert created.data["self_followup"]["session_bound"] is False
    assert created.data["self_followup"]["delivery"] == "internal_unattended"

    captured: list[Signal] = []
    real_dispatch = agent.dispatcher.dispatch_signal

    async def _capture(signal):
        captured.append(signal)
        return await real_dispatch(signal)

    agent.dispatcher.dispatch_signal = _capture
    await runner._tick()
    await _drain(agent)

    assert len(captured) == 1
    assert captured[0].session_id is None
    assert captured[0].visibility == Visibility.INTERNAL


# ---------------------------------------------------------------------------
# Refusals — every way of persisting a follow-up that could not fire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recurring_self_followup_is_refused(followup_env):
    """A cron self-followup is a standing order to spend on turns forever."""
    _agent, feature, _runner, db, _backend = followup_env

    result = await feature.schedule_add(
        cron_expression="@daily",
        task_name=SELF_FOLLOWUP,
        args_json=json.dumps({"intent": SENTINEL}),
    )
    assert result.status is ToolResultStatus.ERROR
    assert "one-shot" in result.error.lower()
    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?", (SELF_FOLLOWUP,)
    )
    assert not rows, "a refused follow-up must not leave a row behind"


@pytest.mark.asyncio
async def test_a_follow_up_turn_cannot_schedule_another_follow_up(followup_env):
    """Single hop (#3101 Q3).

    A persisted row starts a fresh causation chain, so the registration's
    ``allow_self_loops=False`` cannot see this case; without the schedule-time
    refusal a follow-up could queue a follow-up without bound.
    """
    _agent, feature, _runner, db, _backend = followup_env

    from kestrel_sovereign.signals.context import (
        reset_current_signal,
        set_current_signal,
    )

    inside = Signal(
        source=SOURCE,
        kind="run",
        mode=SignalMode.COGNITION,
        payload={"intent": SENTINEL},
        target_agent="did:test:self-followup",
    )
    token = set_current_signal(inside)
    try:
        result = await _schedule(feature, intent="and then another thing")
    finally:
        reset_current_signal(token)

    assert result.status is ToolResultStatus.ERROR
    assert result.data.get("refused") == "self_followup_chain"
    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?", (SELF_FOLLOWUP,)
    )
    assert not rows


@pytest.mark.asyncio
async def test_caller_supplied_origin_session_is_refused(followup_env):
    """Rule 2: the wake's target window is resolved locally, never supplied.

    Refused rather than silently overwritten, so an attempt to aim a follow-up
    at somebody else's chat window is visible instead of quietly corrected.
    """
    _agent, feature, _runner, _db, _backend = followup_env

    result = await feature.schedule_add_deadline(
        run_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        task_name=SELF_FOLLOWUP,
        args_json=json.dumps(
            {"intent": SENTINEL, "origin_session_id": "someone-elses-window"}
        ),
    )
    assert result.status is ToolResultStatus.ERROR
    assert "origin_session_id" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, "intent"),
        ({"intent": "   "}, "intent"),
        ({"intent": 17}, "intent"),
        ({"intent": SENTINEL, "priority": "high"}, "unexpected keys"),
    ],
)
async def test_unusable_followup_args_are_refused(followup_env, args, expected):
    """An empty or malformed intention would fire a turn with nothing to act on."""
    _agent, feature, _runner, db, _backend = followup_env

    result = await feature.schedule_add_deadline(
        run_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        task_name=SELF_FOLLOWUP,
        args_json=json.dumps(args),
    )
    assert result.status is ToolResultStatus.ERROR
    assert expected in result.error.lower() or expected in result.error
    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?", (SELF_FOLLOWUP,)
    )
    assert not rows


@pytest.mark.asyncio
async def test_bound_follow_up_is_refused_when_it_could_not_surface(
    followup_env, monkeypatch
):
    """#2877/#2922: session_id without result_summary renders nowhere.

    Fail loudly at schedule time rather than firing into a blank pane.
    """
    agent, feature, _runner, _db, _backend = followup_env
    agent.turn_session_id = "session-abc"

    registration = agent.signal_registry.get(SOURCE)
    monkeypatch.setattr(registration, "result_summary", None, raising=False)

    result = await _schedule(feature)
    assert result.status is ToolResultStatus.ERROR
    assert result.data.get("refused") == "bound_wake_cannot_surface"


@pytest.mark.asyncio
async def test_unregistered_source_is_refused_not_persisted(followup_env):
    """No registration means the row would come due and produce nothing."""
    agent, feature, _runner, db, _backend = followup_env

    agent.signal_registry = None
    result = await _schedule(feature)

    assert result.status is ToolResultStatus.ERROR
    assert result.data.get("refused") == "source_not_registered"
    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?", (SELF_FOLLOWUP,)
    )
    assert not rows


# ---------------------------------------------------------------------------
# Relative deadlines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delay_seconds_resolves_against_the_database_clock(followup_env):
    """"in 20 minutes" without the caller needing to know what time it is."""
    _agent, feature, _runner, db, _backend = followup_env

    before = datetime.now(timezone.utc)
    result = await feature.schedule_add_deadline(
        task_name=SELF_FOLLOWUP,
        args_json=json.dumps({"intent": SENTINEL}),
        delay_seconds=1200,
    )
    assert result.status is ToolResultStatus.OK

    row = await db.fetchone(
        "SELECT run_at, next_run_at, schedule_kind FROM scheduled_tasks WHERE id = ?",
        (result.data["task_id"],),
    )
    assert row[2] == "one_shot"
    assert row[0] == row[1]
    due = datetime.fromisoformat(row[0])
    delta = (due - before).total_seconds()
    assert 1150 < delta < 1260, f"deadline landed at {delta}s, expected ~1200"


@pytest.mark.asyncio
async def test_deadline_needs_exactly_one_of_run_at_or_delay(followup_env):
    """Two ways to say when is an ambiguity, not a convenience."""
    _agent, feature, _runner, _db, _backend = followup_env
    args = json.dumps({"intent": SENTINEL})

    both = await feature.schedule_add_deadline(
        run_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        task_name=SELF_FOLLOWUP,
        args_json=args,
        delay_seconds=60,
    )
    assert both.status is ToolResultStatus.ERROR
    assert "not both" in both.error

    neither = await feature.schedule_add_deadline(
        task_name=SELF_FOLLOWUP, args_json=args
    )
    assert neither.status is ToolResultStatus.ERROR
    assert "delay_seconds" in neither.error

    bad = await feature.schedule_add_deadline(
        task_name=SELF_FOLLOWUP, args_json=args, delay_seconds=0
    )
    assert bad.status is ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# Requirement 4 — the inspectable surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_projection_reports_pending_then_fired(followup_env):
    """schedule_self_followups joins the pending row with its outcome."""
    agent, feature, runner, _db, _backend = followup_env

    await _schedule(feature)

    pending = await feature.schedule_self_followups()
    assert pending.status is ToolResultStatus.OK
    assert pending.data["count"] == 1
    assert pending.data["pending_count"] == 1
    entry = pending.data["followups"][0]
    assert entry["state"] == "pending"
    assert entry["intent"] == SENTINEL
    assert entry["delivery"] == "internal_unattended"

    await runner._tick()
    await _drain(agent)

    fired = await feature.schedule_self_followups()
    assert fired.data["followups"][0]["state"] == "fired"
    assert fired.data["missed_count"] == 0
    assert fired.data["pending_count"] == 0


@pytest.mark.asyncio
async def test_projection_counts_a_dropped_turn_as_missed(followup_env):
    """A dropped self-scheduled turn must be visible rather than silent."""
    agent, feature, runner, _db, _backend = followup_env

    await _schedule(feature)

    async def _dropped(signal):
        from kestrel_sdk.signals import SignalResult

        return SignalResult(
            signal_id=signal.id,
            status=Status.DROPPED_QUIET_HOURS,
            mode=SignalMode.COGNITION,
            duration_ms=0,
        )

    agent.dispatcher.dispatch_signal = _dropped
    await runner._tick()
    await _drain(agent)

    projected = await feature.schedule_self_followups()
    assert projected.data["missed_count"] == 1
    assert projected.data["followups"][0]["state"] == "missed"
    assert "missed" in projected.confirmation


@pytest.mark.asyncio
async def test_projection_does_not_round_an_unknown_status_up_to_fired(
    followup_env,
):
    """A state the projection has not been taught about is reported verbatim.

    Folding it into `fired` would make a future failure mode look like a
    success — the exact "absence reported as fact" shape this issue names.
    """
    _agent, feature, _runner, _db, _backend = followup_env

    state = feature._self_followup_state(
        disablement_state="terminal", execution_status="some_future_status"
    )
    assert state == "some_future_status"
    assert feature._self_followup_state(
        disablement_state="enabled", execution_status=None
    ) == "pending"


# ---------------------------------------------------------------------------
# Intent normalization
# ---------------------------------------------------------------------------


def test_normalize_intent_rejects_empty_and_non_string():
    with pytest.raises(SelfFollowupIntentError):
        normalize_intent("")
    with pytest.raises(SelfFollowupIntentError):
        normalize_intent("   \n\t ")
    with pytest.raises(SelfFollowupIntentError):
        normalize_intent(None)


def test_normalize_intent_strips_control_characters_but_keeps_layout():
    cleaned = normalize_intent("check CI\x00 then\x07 merge\nsecond line")
    assert "\x00" not in cleaned and "\x07" not in cleaned
    assert cleaned == "check CI then merge\nsecond line"


def test_normalize_intent_caps_length():
    cleaned = normalize_intent("x" * (MAX_INTENT_CHARS + 500))
    assert len(cleaned) <= MAX_INTENT_CHARS + len("...(truncated)")
    assert cleaned.endswith("...(truncated)")
