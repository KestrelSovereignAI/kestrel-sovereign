"""Gate-5 review findings on #3112 — verified, then fixed.

Four findings came back from a full-diff review against `origin/main`. Three
were confirmed by reading the code; the fourth's stated mechanism did not hold
and is deliberately NOT "fixed" here (see the note at the bottom of this
module). Each test is written so that reverting its guard makes it fail.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from kestrel_sovereign.features.scheduler.feature import SELF_FOLLOWUP_TASK_NAME
from kestrel_sovereign.privacy import PrivacyConfig

SENTINEL = "gate5-intent-XYZZY"


class _RecordingLock:
    """A stand-in for the agent's ReentrantTransitionLock that records use.

    Real `asyncio.Lock` semantics — so a second acquirer genuinely waits —
    plus a record of whether it was held, which is what the wiring test needs.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self.held = False
        self.acquisitions = 0

    async def __aenter__(self):
        await self._lock.acquire()
        self.held = True
        self.acquisitions += 1
        return self

    async def __aexit__(self, *exc):
        self.held = False
        self._lock.release()
        return False



async def _queue_followup(agent, feature, intent=SENTINEL, task_name=None):
    """Create a follow-up the way production does: from inside a live turn.

    `self_followup` is refused with `no_in_turn_origin` outside one — the row
    must be agent-authored. Creating it any other way makes every "the intent
    is not visible" assertion below pass vacuously on an empty list, which is
    exactly what the positive-control test caught.
    """
    async with agent._turn_lifecycle():
        return await feature._create_schedule(
            task_name=task_name or SELF_FOLLOWUP_TASK_NAME,
            args_json=json.dumps({"intent": intent}),
            cron_expression="",
            next_run_at=None,
            schedule_kind="one_shot",
            run_at="2099-01-01T00:00:00+00:00",
            timezone_name="UTC",
            misfire_policy="skip",
            misfire_grace_seconds=None,
            idempotency_key=None,
        )

# ---------------------------------------------------------------------------
# P2: a COGNITION row must never take a direct-tool fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cognition_without_a_dispatcher_reports_a_missed_turn(followup_env):
    """It used to be recorded as SUCCESS and the one-shot terminalized.

    `self_followup` has no tool by construction, so `_lookup_and_run_tool`
    found none, saw the name IS in CRON_TASKS, and returned the benign
    "skipped: owning feature not loaded yet" string meant for the startup-order
    race. The runner logged that as success and consumed the row — the agent's
    follow-up intention discarded with no turn and no error anywhere.
    """
    agent, feature, _runner, _db, _backend = followup_env
    agent.dispatcher = None  # the supported no-dispatcher fallback

    outcome = await feature._dispatch_scheduled_task(
        SELF_FOLLOWUP_TASK_NAME, {"intent": SENTINEL}
    )

    assert getattr(outcome, "status", None) == "failed", (
        f"a COGNITION row with no dispatcher must not report success, got {outcome!r}"
    )
    assert "skipped" not in str(outcome).lower(), (
        "must not reuse the startup-order-race 'skipped' string, which the "
        "runner treats as success"
    )
    assert not agent.turn_prompts, "no turn was produced, and none should be claimed"


@pytest.mark.asyncio
async def test_a_non_cognition_task_still_uses_the_direct_tool_fallback(followup_env):
    """The fallback is load-bearing for ACTION tasks — do not break it.

    Without this, 'fix' the P2 by deleting the fallback entirely and every
    test above still passes.
    """
    agent, feature, _runner, _db, _backend = followup_env
    agent.dispatcher = None
    called = {}

    async def _lookup(task_name, args):
        called["name"] = task_name
        return "ran directly"

    feature._lookup_and_run_tool = _lookup
    result = await feature._dispatch_scheduled_task("backup_snapshot", {})
    assert called.get("name") == "backup_snapshot"
    assert result == "ran directly"


# ---------------------------------------------------------------------------
# P1: persisted intent must not be readable in a volatile privacy mode
# ---------------------------------------------------------------------------


def _volatile(feature, storage="none"):
    feature.agent.privacy_config = PrivacyConfig(storage=storage)


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["none", "temp", "deidentified"])
async def test_schedule_self_followups_redacts_intent_in_a_volatile_mode(
    followup_env, storage
):
    agent, feature, _runner, db, _backend = followup_env
    result = await _queue_followup(agent, feature)
    assert result.status.value == "ok", f"setup failed: {result.error}"
    _volatile(feature, storage)

    result = await feature.schedule_self_followups()

    assert SENTINEL not in json.dumps(result.data or {}), (
        f"{storage}: intent queued under durable storage must not be readable "
        f"after switching to a volatile mode"
    )


@pytest.mark.asyncio
async def test_schedule_list_redacts_the_same_rows(followup_env):
    """The OLDER reader. It predates this feature and reads the same column.

    Guarding only `schedule_self_followups` leaves this door open on exactly
    the same rows — and this one is easier to miss precisely because nobody
    thinks of it as new.
    """
    agent, feature, _runner, _db, _backend = followup_env
    result = await _queue_followup(agent, feature)
    assert result.status.value == "ok", f"setup failed: {result.error}"
    _volatile(feature)

    result = await feature.schedule_list()

    assert SENTINEL not in json.dumps(result.data or {}), (
        "schedule_list returns args_json for the same rows and must redact too"
    )


@pytest.mark.asyncio
async def test_a_durable_mode_still_returns_the_intent(followup_env):
    """Never assert emptiness without proving the non-empty case fires.

    Redaction that always redacts would pass both tests above and destroy the
    feature.
    """
    agent, feature, _runner, _db, _backend = followup_env
    result = await _queue_followup(agent, feature)
    assert result.status.value == "ok", f"setup failed: {result.error}"
    result = await feature.schedule_self_followups()
    assert SENTINEL in json.dumps(result.data or {}), (
        "under normal privacy the intent must still be visible"
    )


@pytest.mark.asyncio
async def test_other_task_kinds_are_not_redacted(followup_env):
    """Only conversation-derived rows are sensitive; cron args are not."""
    agent, feature, _runner, _db, _backend = followup_env
    await feature._create_schedule(
        task_name="backup_snapshot",
        args_json=json.dumps({"intent": SENTINEL}),
        cron_expression="0 */4 * * *",
        next_run_at=None,
        schedule_kind="cron",
        run_at=None,
        timezone_name="UTC",
        misfire_policy="skip",
        misfire_grace_seconds=None,
        idempotency_key=None,
    )
    _volatile(feature)
    result = await feature.schedule_list()
    assert SENTINEL in json.dumps(result.data or {}), (
        "redaction keyed on task_name must not blanket every schedule"
    )


# ---------------------------------------------------------------------------
# P1: the privacy check must hold across the write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creation_holds_the_privacy_transition_lock(followup_env):
    """Wiring: the lock must actually be held when the INSERT runs.

    The check lives in a synchronous `_prepare_self_followup_args`; the INSERT
    happens several awaits later. Nothing in the scheduler feature referenced
    `privacy_transition_lock` at all, so a transition could land in that gap.
    """
    agent, feature, _runner, _db, _backend = followup_env
    lock = _RecordingLock()
    agent._get_privacy_transition_lock = lambda: lock

    held_during_write = {}
    original = feature._create_schedule_locked

    async def _spy(**kwargs):
        held_during_write["held"] = lock.held
        return await original(**kwargs)

    feature._create_schedule_locked = _spy

    result = await _queue_followup(agent, feature)
    assert result.status.value == "ok", f"setup failed: {result.error}"

    assert lock.acquisitions == 1, "the transition lock was never taken"
    assert held_during_write.get("held") is True, (
        "the lock must still be held when the row is persisted, or the "
        "creation-time privacy check means nothing at the moment of the write"
    )


@pytest.mark.asyncio
async def test_other_schedule_kinds_do_not_contend_for_the_lock(followup_env):
    """Only the follow-up path carries conversation content.

    A blanket lock would pass the test above and make every schedule creation
    contend with privacy transitions.
    """
    agent, feature, _runner, _db, _backend = followup_env
    lock = _RecordingLock()
    agent._get_privacy_transition_lock = lambda: lock

    await feature._create_schedule(
        task_name="backup_snapshot",
        args_json=json.dumps({}),
        cron_expression="0 */4 * * *",
        next_run_at=None,
        schedule_kind="cron",
        run_at=None,
        timezone_name="UTC",
        misfire_policy="skip",
        misfire_grace_seconds=None,
        idempotency_key=None,
    )
    assert lock.acquisitions == 0


@pytest.mark.asyncio
async def test_a_transition_cannot_interleave_with_the_write(followup_env):
    """The concurrency reproduction the review asked for.

    A privacy transition is started while a follow-up creation is mid-flight,
    at a point where the creation has already passed its synchronous check and
    is awaiting. With the lock spanning check-and-persist, the transition must
    wait for the write to finish rather than landing in the gap.
    """
    agent, feature, _runner, _db, _backend = followup_env
    lock = _RecordingLock()
    agent._get_privacy_transition_lock = lambda: lock

    order: list[str] = []
    creation_reached_the_gap = asyncio.Event()
    release_creation = asyncio.Event()
    original = feature._create_schedule_locked

    async def _slow_create(**kwargs):
        creation_reached_the_gap.set()
        await release_creation.wait()   # sit in the window a transition wants
        result = await original(**kwargs)
        order.append("write")
        return result

    feature._create_schedule_locked = _slow_create

    async def _transition():
        await creation_reached_the_gap.wait()
        async with lock:                 # what a real transition does
            feature.agent.privacy_config = PrivacyConfig(storage="none")
            order.append("transition")

    creator = asyncio.create_task(
        feature._create_schedule(
            task_name=SELF_FOLLOWUP_TASK_NAME,
            args_json=json.dumps({"intent": SENTINEL}),
            cron_expression="",
            next_run_at=None,
            schedule_kind="one_shot",
            run_at="2099-01-01T00:00:00+00:00",
            timezone_name="UTC",
            misfire_policy="skip",
            misfire_grace_seconds=None,
            idempotency_key=None,
        )
    )
    transitioner = asyncio.create_task(_transition())

    await creation_reached_the_gap.wait()
    await asyncio.sleep(0)               # give the transition every chance
    assert order == [], "neither side should have completed yet"
    release_creation.set()
    await asyncio.gather(creator, transitioner)

    assert order == ["write", "transition"], (
        "the transition must not land between the privacy check and the "
        f"write; got {order}"
    )


# ---------------------------------------------------------------------------
# The fourth finding is deliberately not "fixed"
# ---------------------------------------------------------------------------


def test_fire_time_check_and_dispatch_have_no_await_between_them():
    """The review's second P1 claimed the mode can change between the fire-time
    check and `dispatch_signal`. It cannot: on the branch that dispatches, the
    code from the guard to the call is straight-line, so in single-threaded
    asyncio nothing can interleave there. The only awaits between them belong
    to the two early-return branches.

    Adding a re-check immediately before `dispatch_signal` would guard against
    something that cannot happen. This test pins the property the argument
    rests on, so if someone later introduces an await in that stretch the
    claim becomes live and this fails loudly rather than silently.
    """
    import inspect

    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature

    src = inspect.getsource(SchedulerFeature._dispatch_scheduled_task)
    guard = src.index("hides_persisted_user_content(self.agent)")
    dispatch = src.index("dispatch_signal(")
    # Cut at the START of the dispatching statement — the `await dispatcher.`
    # that performs it is the boundary, not something between the two.
    stmt_start = src.rindex("\n", 0, dispatch)
    between = src[guard:stmt_start]

    # Awaits belonging to the early-return fallbacks are not on the dispatch
    # path; anything else would be.
    offenders = [
        line.strip()
        for line in between.splitlines()
        if "await " in line and "_lookup_and_run_tool" not in line
    ]
    assert not offenders, (
        "an await now sits between the fire-time privacy check and dispatch, "
        "so the mode CAN change in between and the check must be repeated: "
        f"{offenders}"
    )



# ---------------------------------------------------------------------------
# Gate 6: findings from reviewing the gate-5 fixes themselves
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stored_follow_up_RESULT_is_redacted_in_a_volatile_mode(followup_env):
    """The gate-5 fix redacted the input and left the output.

    `task_execution_log.result_text` holds the follow-up's complete cognition
    RESPONSE — strictly more conversation-derived than the intent that produced
    it. Redacting `args_json.intent` and not this hid the question and returned
    the answer: the same two-doors mistake the input guard was written to fix,
    one field along.
    """
    agent, feature, _runner, db, _backend = followup_env
    result = await _queue_followup(agent, feature)
    assert result.status.value == "ok", f"setup failed: {result.error}"

    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?",
        (SELF_FOLLOWUP_TASK_NAME,),
    )
    task_id = rows[0][0]
    await db.execute(
        """INSERT INTO task_execution_log
           (id, task_id, agent_id, status, result_text, executed_at,
            attempt_count, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("exec-1", task_id, agent.did, "success", SENTINEL,
         "2026-08-31T00:00:00+00:00", 1, 0),
    )

    _volatile(feature)
    out = await feature.schedule_self_followups()
    assert SENTINEL not in json.dumps(out.data or {}), (
        "the stored cognition RESULT must be redacted too, not just the intent"
    )


@pytest.mark.asyncio
async def test_schedule_history_redacts_the_same_result(followup_env):
    """The older reader returns the same column."""
    agent, feature, _runner, db, _backend = followup_env
    result = await _queue_followup(agent, feature)
    assert result.status.value == "ok"

    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?",
        (SELF_FOLLOWUP_TASK_NAME,),
    )
    await db.execute(
        """INSERT INTO task_execution_log
           (id, task_id, agent_id, status, result_text, executed_at,
            attempt_count, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("exec-2", rows[0][0], agent.did, "success", SENTINEL,
         "2026-08-31T00:00:00+00:00", 1, 0),
    )

    _volatile(feature)
    out = await feature.schedule_history()
    assert SENTINEL not in json.dumps(out.data or {}), (
        "schedule_history exposes result_text for the same rows"
    )


@pytest.mark.asyncio
async def test_a_durable_mode_still_returns_the_result(followup_env):
    """Never assert emptiness without proving the non-empty case fires."""
    agent, feature, _runner, db, _backend = followup_env
    result = await _queue_followup(agent, feature)
    assert result.status.value == "ok"

    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?",
        (SELF_FOLLOWUP_TASK_NAME,),
    )
    await db.execute(
        """INSERT INTO task_execution_log
           (id, task_id, agent_id, status, result_text, executed_at,
            attempt_count, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("exec-3", rows[0][0], agent.did, "success", SENTINEL,
         "2026-08-31T00:00:00+00:00", 1, 0),
    )

    out = await feature.schedule_history()
    assert SENTINEL in json.dumps(out.data or {}), (
        "under normal privacy the stored result must still be visible"
    )


@pytest.mark.asyncio
async def test_a_transition_during_dispatch_refuses_delivery(followup_env):
    """The finding I refused once, with the repro I said was missing.

    The stated mechanism — a mode change BETWEEN the fire-time check and
    `dispatch_signal` — really is unreachable; that stretch is straight-line.
    But the payload is built from persisted intent BEFORE the await, and
    `dispatch_signal` itself awaits persistence, locking and turn admission. A
    transition landing inside THAT window delivers conversation content into a
    cognition turn under a mode that forbids it.

    A finding's mechanism can be wrong while its claim is right.

    The transition is modelled as flipping the mode while the transition lock
    is being ACQUIRED, which is what a real one does — it holds that lock. An
    earlier draft flipped the mode before the call, so the outer fire-time
    guard caught it and the re-check under the lock was never exercised; the
    mutant for removing that re-check survived.
    """
    agent, feature, _runner, _db, _backend = followup_env

    class _TransitionOnAcquire:
        """Flips privacy to volatile exactly when the lock is taken."""

        def __init__(self):
            self._lock = asyncio.Lock()
            self.acquisitions = 0

        async def __aenter__(self):
            await self._lock.acquire()
            self.acquisitions += 1
            # A privacy transition got here first and completed.
            feature.agent.privacy_config = PrivacyConfig(storage="none")
            return self

        async def __aexit__(self, *exc):
            self._lock.release()
            return False

    lock = _TransitionOnAcquire()
    agent._get_privacy_transition_lock = lambda: lock

    class _RefusingDispatcher:
        async def dispatch_signal(self, signal):
            raise AssertionError(
                "delivery proceeded under a volatile mode — the re-check "
                "inside the lock did not fire"
            )

    agent.dispatcher = _RefusingDispatcher()

    # Mode is DURABLE here, so the outer fire-time guard passes and the only
    # thing that can stop delivery is the re-check under the lock.
    outcome = await feature._dispatch_scheduled_task(
        SELF_FOLLOWUP_TASK_NAME, {"intent": SENTINEL}
    )

    assert lock.acquisitions == 1, "the transition lock was never taken"
    assert getattr(outcome, "status", None) == "failed", (
        f"delivery must be refused once the mode is volatile, got {outcome!r}"
    )
    assert SENTINEL not in str(outcome)
    assert not agent.turn_prompts, "no cognition turn may be produced"


@pytest.mark.asyncio
async def test_a_durable_mode_still_delivers(followup_env):
    """The converse: the lock must not refuse a follow-up that is still legal.

    Without this, 'fix' the finding by refusing every delivery.
    """
    agent, feature, _runner, _db, _backend = followup_env

    class _PlainLock:
        def __init__(self):
            self._lock = asyncio.Lock()

        async def __aenter__(self):
            await self._lock.acquire()
            return self

        async def __aexit__(self, *exc):
            self._lock.release()
            return False

    agent._get_privacy_transition_lock = lambda: _PlainLock()

    outcome = await feature._dispatch_scheduled_task(
        SELF_FOLLOWUP_TASK_NAME, {"intent": SENTINEL}
    )
    assert getattr(outcome, "status", None) != "failed", (
        f"a durable-mode follow-up must still be delivered, got {outcome!r}"
    )
    assert agent.turn_prompts, "the cognition turn should have been produced"
