"""Fire-time privacy revalidation for a queued follow-up (#3112 gate-4 P1).

The creation-time refusal in ``_create_schedule`` is evaluated once, against
the mode in force when the row was written. Privacy mode is mutable, so a
follow-up queued under full storage and fired after a transition to a volatile
mode reads conversation-derived text back out of the raw scheduler database
and into a cognition turn -- the thing the mode forbids.

Each test here is written so that reverting the guard makes it fail. The two
bypass tests exist because ``_dispatch_scheduled_task`` has two branches that
return through ``_lookup_and_run_tool`` before reaching the signal path; a
guard placed after them would pass a naive test and still be reachable.
"""

from __future__ import annotations

import pytest
from kestrel_sovereign.features.scheduler.feature import (
    SELF_FOLLOWUP_TASK_NAME,
)
from kestrel_sovereign.privacy import PrivacyConfig

SENTINEL = "fire-time-privacy-XYZZY"


def _volatile(monkeypatch, feature, storage):
    """Put the agent into a volatile privacy mode, via the REAL PrivacyConfig.

    A stub with ``is_ephemeral`` hardcoded True would assert the predicate
    under test instead of exercising it; the real dataclass makes
    ``storage="none"`` mean what production means by it.
    """
    feature.agent.privacy_config = PrivacyConfig(storage=storage)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "storage,mode_name",
    [
        ("none", "EPHEMERAL"),
        ("temp", "ISOLATED"),
        ("deidentified", "DEIDENTIFIED"),
    ],
)
async def test_a_queued_follow_up_is_refused_after_a_volatile_transition(
    followup_env, monkeypatch, storage, mode_name
):
    """Queued under durable storage, fired under a volatile mode -> refused."""
    agent, feature, _runner, _db, _backend = followup_env

    _volatile(monkeypatch, feature, storage)

    outcome = await feature._dispatch_scheduled_task(
        SELF_FOLLOWUP_TASK_NAME, {"intent": SENTINEL}
    )

    assert getattr(outcome, "status", None) == "failed", (
        f"{mode_name}: a follow-up queued before the transition must be "
        f"refused at fire time, got {outcome!r}"
    )
    assert SENTINEL not in str(outcome), (
        f"{mode_name}: the persisted intent must not be read back out"
    )
    assert not agent.turn_prompts, (
        f"{mode_name}: no cognition turn may be produced"
    )


@pytest.mark.asyncio
async def test_a_durable_mode_still_fires_the_queued_follow_up(followup_env):
    """The opposite direction: the guard must not refuse a legitimate fire.

    A refusal that also blocks the durable-mode path would make the feature
    inert rather than safe -- an accept that produces no turn is the failure
    #3101 exists to prevent.
    """
    agent, feature, _runner, _db, _backend = followup_env

    feature.agent.privacy_config = PrivacyConfig(storage="full")

    await feature._dispatch_scheduled_task(
        SELF_FOLLOWUP_TASK_NAME, {"intent": SENTINEL}
    )

    assert agent.turn_prompts, (
        "a durable privacy mode must still produce the follow-up turn"
    )
    assert any(SENTINEL in p for p in agent.turn_prompts), (
        "the intention text must reach the turn"
    )


@pytest.mark.asyncio
async def test_the_no_dispatcher_fallback_cannot_bypass_the_guard(
    followup_env, monkeypatch
):
    """Bypass 1: the partially-initialized-agent branch.

    ``_dispatch_scheduled_task`` returns through ``_lookup_and_run_tool`` when
    the agent has no dispatcher. That branch reads the same persisted intent,
    so a guard placed after it is a guard with a way around it.
    """
    agent, feature, _runner, _db, _backend = followup_env

    _volatile(monkeypatch, feature, "none")
    monkeypatch.setattr(agent, "dispatcher", None, raising=False)

    outcome = await feature._dispatch_scheduled_task(
        SELF_FOLLOWUP_TASK_NAME, {"intent": SENTINEL}
    )

    assert getattr(outcome, "status", None) == "failed", (
        "the no-dispatcher fallback must not bypass the fire-time guard, "
        f"got {outcome!r}"
    )
    assert not agent.turn_prompts


# ---------------------------------------------------------------------------
# The transition can land INSIDE the dispatch (#3101 review P1)
# ---------------------------------------------------------------------------
#
# The fire-time check above and ``await dispatcher.dispatch_signal(...)`` have
# no await between them, and that was mistaken for safety. ``dispatch_signal``
# is itself a suspension point -- durable admission, event persistence and lock
# acquisition all await before the COGNITION route reaches ``process_input`` --
# so a transition only had to land inside it. These tests drive the real
# dispatcher and fail against the pre-fix branch.


async def _flip_during_dispatch(agent, feature, storage="none"):
    """Fire a follow-up while a transition lands inside the dispatch.

    The flip is scheduled as a task BEFORE the dispatch is awaited, so it runs
    at the dispatch's first true suspension point -- i.e. after the fire-time
    check has already passed under durable storage. No internals are patched;
    the interleaving is the one single-threaded asyncio actually produces.
    """
    import asyncio

    agent.privacy_config = PrivacyConfig(storage="full")

    async def flip():
        agent.privacy_config = PrivacyConfig(storage=storage)

    flipper = asyncio.create_task(flip())
    try:
        outcome = await feature._dispatch_scheduled_task(
            SELF_FOLLOWUP_TASK_NAME, {"intent": SENTINEL}
        )
    except RuntimeError as exc:
        # DROPPED_VALIDATION is surfaced as an exception so the runner records
        # 'failed' rather than filing a turn that never happened as success.
        outcome = exc
    finally:
        await flipper
    return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["none", "temp", "deidentified"])
async def test_a_transition_inside_the_dispatch_still_stops_the_turn(
    followup_env, storage
):
    """The headline race: no turn, and the intent is not in the refusal."""
    agent, feature, _runner, _db, _backend = followup_env

    outcome = await _flip_during_dispatch(agent, feature, storage)

    assert not agent.turn_prompts, (
        f"{storage}: the persisted intent reached a cognition turn after the "
        "mode became volatile mid-dispatch"
    )
    assert SENTINEL not in str(outcome), (
        f"{storage}: the refusal must not quote the intent"
    )


@pytest.mark.asyncio
async def test_a_turn_stopped_mid_dispatch_is_recorded_failed_not_success(
    followup_env,
):
    """A refused turn must be visible, never filed alongside real successes.

    This is the #3101 constraint applied to the race: an accept that produces
    no turn is worse than an explicit refusal, so the occurrence has to land in
    the execution log as a non-success an operator can find.
    """
    agent, feature, runner, db, _backend = followup_env
    import json
    from datetime import datetime, timedelta, timezone

    agent.privacy_config = PrivacyConfig(storage="full")
    async with agent._turn_lifecycle():
        created = await feature.schedule_add_deadline(
            run_at=(
                datetime.now(timezone.utc) - timedelta(seconds=5)
            ).isoformat(),
            task_name=SELF_FOLLOWUP_TASK_NAME,
            args_json=json.dumps({"intent": SENTINEL}),
        )
    schedule_id = created.data["task_id"]

    # Flip while the runner's dispatch is in flight.
    original = feature._dispatch_scheduled_task

    async def _flip_then_dispatch(task_name, args):
        agent.privacy_config = PrivacyConfig(storage="none")
        return await original(task_name, args)

    feature._dispatch_scheduled_task = _flip_then_dispatch
    runner._executor = _flip_then_dispatch
    await runner._tick()

    assert not agent.turn_prompts

    row = await db.fetchone(
        "SELECT status, result_text FROM task_execution_log WHERE task_id = ?",
        (schedule_id,),
    )
    assert row is not None, "the refused occurrence left no execution record"
    assert row[0] != "success", (
        f"a follow-up that produced no turn was recorded {row[0]!r}"
    )
    assert SENTINEL not in (row[1] or "")

    projected = await feature.schedule_self_followups()
    states = [f["state"] for f in projected.data["followups"]]
    assert "fired" not in states, (
        f"the projection reported a turn that never ran: {states}"
    )


@pytest.mark.asyncio
async def test_a_raising_pre_turn_guard_refuses_rather_than_admitting(
    followup_env, monkeypatch
):
    """A guard exists to refuse; one that raises has answered nothing.

    Treating a broken guard as consent would convert it into an absent one,
    which is the silent-accept shape this feature is written to avoid.
    """
    agent, feature, _runner, _db, _backend = followup_env
    from kestrel_sovereign.signals.sources.scheduler import cron_source_name

    agent.privacy_config = PrivacyConfig(storage="full")
    registration = agent.signal_registry.get(
        cron_source_name(SELF_FOLLOWUP_TASK_NAME)
    )

    def _boom(_signal, _agent):
        raise RuntimeError("guard is broken")

    monkeypatch.setattr(registration, "pre_turn_guard", _boom, raising=False)

    with pytest.raises(RuntimeError):
        await feature._dispatch_scheduled_task(
            SELF_FOLLOWUP_TASK_NAME, {"intent": SENTINEL}
        )
    assert not agent.turn_prompts, (
        "a guard that raised was treated as consent and the turn ran"
    )


def test_the_source_actually_registers_the_pre_turn_guard():
    """Pin the wiring: without it every test above passes for the wrong reason."""
    from kestrel_sovereign.signals.sources.self_followup import (
        build_self_followup_registration,
        refuse_followup_under_volatile_privacy,
    )

    registration = build_self_followup_registration()
    assert (
        getattr(registration, "pre_turn_guard", None)
        is refuse_followup_under_volatile_privacy
    )


@pytest.mark.asyncio
async def test_the_unregistered_task_fallback_cannot_bypass_the_guard(
    followup_env, monkeypatch
):
    """Bypass 2: the no-source-registration branch.

    A task absent from ``CRON_TASKS`` also returns through
    ``_lookup_and_run_tool``. Emptying the classification table reproduces
    that branch for the self_followup name specifically.
    """
    from kestrel_sovereign.signals.sources import scheduler as scheduler_sources

    agent, feature, _runner, _db, _backend = followup_env

    _volatile(monkeypatch, feature, "none")
    monkeypatch.setattr(scheduler_sources, "CRON_TASKS", (), raising=False)

    outcome = await feature._dispatch_scheduled_task(
        SELF_FOLLOWUP_TASK_NAME, {"intent": SENTINEL}
    )

    assert getattr(outcome, "status", None) == "failed", (
        "the unregistered-task fallback must not bypass the fire-time "
        f"guard, got {outcome!r}"
    )
    assert not agent.turn_prompts
