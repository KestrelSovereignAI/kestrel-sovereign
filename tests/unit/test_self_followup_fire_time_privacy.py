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
