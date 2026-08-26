"""Gate tests for the two P1 refusals in 7621694b (#3112 gate-3).

Both gates were committed with their tests deferred; these are those tests.
Each one is written so that reverting the guard makes it fail — verified by
mutation, not by reading, because a guard test that passes with the guard
removed is the failure mode this branch has already produced twice.

Gate 0 (volatile privacy modes) uses the REAL
:class:`kestrel_sovereign.privacy.PrivacyConfig` rather than a stub with
``is_ephemeral`` hardcoded True. A stub would assert the predicate the test
is supposed to exercise; the real dataclass makes ``storage="none"`` mean
what production means by it.

Gate 2b (causation ancestry) uses the REAL
:class:`kestrel_sdk.signals.CausationFrame` that
``SignalDispatcher._compute_frame_and_check_cycle`` appends, so the shape
under test is the shape production builds.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from kestrel_sdk.signals import CausationFrame, Signal, SignalMode
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.privacy import PrivacyConfig
from kestrel_sovereign.signals.sources.scheduler import cron_source_name
from kestrel_sovereign.signals.sources.self_followup import (
    TASK_NAME as SELF_FOLLOWUP,
)

from tests.unit.test_self_followup_schedule import (  # noqa: F401
    SENTINEL,
    SOURCE,
    _schedule,
    followup_env,
)

TARGET = "did:test:self-followup"


async def _no_rows(db) -> bool:
    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?", (SELF_FOLLOWUP,)
    )
    return not rows


# ---------------------------------------------------------------------------
# Gate 0 — volatile privacy modes forbid a durable intent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "storage,mode_name",
    [
        ("none", "EPHEMERAL"),
        ("temp", "ISOLATED"),
        ("deidentified", "DEIDENTIFIED"),
    ],
)
async def test_volatile_privacy_mode_refuses_a_follow_up(
    followup_env, storage, mode_name
):
    """The intent is conversation-derived text written durably to args_json.

    EPHEMERAL / ISOLATED / DEIDENTIFIED each promise conversation content does
    not outlive the session. ``scheduled_tasks.args_json`` is the raw
    persistent DB and the signal-log redactor does not reach that column, so
    accepting here would break the promise the mode makes — silently, in the
    direction of persisting more than the Sovereign asked for.
    """
    agent, feature, _runner, db, _backend = followup_env
    agent.privacy_config = PrivacyConfig(storage=storage)

    result = await _schedule(feature, intent=SENTINEL)

    assert result.status is ToolResultStatus.ERROR, (
        f"{mode_name} must refuse a durable follow-up, got {result.status}"
    )
    assert result.data.get("refused") == "volatile_privacy_mode"
    assert await _no_rows(db), "a refused follow-up must not leave a row behind"


@pytest.mark.asyncio
async def test_durable_privacy_mode_still_accepts_a_follow_up(followup_env):
    """Negative control: the gate must refuse volatile modes, not all modes.

    Without this, a guard that returned the refusal unconditionally would pass
    every test above while breaking the feature entirely.
    """
    agent, feature, _runner, db, _backend = followup_env
    agent.privacy_config = PrivacyConfig(storage="full")

    result = await _schedule(feature, intent=SENTINEL)

    assert result.status is not ToolResultStatus.ERROR, (
        f"a durable mode must still accept: {result.error!r}"
    )
    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?", (SELF_FOLLOWUP,)
    )
    assert rows, "an accepted follow-up must persist exactly one row"


# ---------------------------------------------------------------------------
# Gate 2b — the single-hop bound follows causation ancestry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_descendant_of_a_follow_up_cannot_schedule_another(followup_env):
    """self_followup -> A2A completion -> self_followup is one hop out of sight.

    The live source of the woken turn is ``a2a.task_complete``, not
    ``cron.self_followup``, so a bare equality test on the live source lets the
    descendant queue another follow-up. ``_dispatch_scheduled_task`` builds the
    next signal with a FRESH chain, so the dispatcher's ``allow_self_loops``
    cycle check cannot see the ancestry either — this schedule-time walk is the
    only thing standing between here and an unbounded self-wake chain.
    """
    _agent, feature, _runner, db, _backend = followup_env

    from kestrel_sovereign.signals.context import (
        reset_current_signal,
        set_current_signal,
    )

    descendant = Signal(
        source="a2a.task_complete",
        kind="completed",
        mode=SignalMode.COGNITION,
        payload={},
        target_agent=TARGET,
        causation_chain=[
            CausationFrame(
                agent_id=TARGET,
                source=SOURCE,
                signal_id="sig-followup-parent",
                turn_id="turn-parent",
                depth=1,
                emitted_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            )
        ],
    )

    token = set_current_signal(descendant)
    try:
        result = await _schedule(feature, intent="and one more thing")
    finally:
        reset_current_signal(token)

    assert result.status is ToolResultStatus.ERROR, (
        "a turn descended from a follow-up must not queue another"
    )
    assert result.data.get("refused") == "self_followup_chain_ancestor"
    assert result.data.get("live_source") == "a2a.task_complete", (
        "the refusal must name the live source so the reason is diagnosable"
    )
    assert await _no_rows(db), "a refused follow-up must not leave a row behind"


@pytest.mark.asyncio
async def test_an_unrelated_chain_ancestor_does_not_block_a_follow_up(followup_env):
    """Negative control: ancestry refuses self_followup ancestors, not any chain.

    A guard that refused whenever a causation chain existed at all would pass
    the test above and break every legitimate follow-up scheduled from a woken
    turn — which is most of them.
    """
    _agent, feature, _runner, db, _backend = followup_env

    from kestrel_sovereign.signals.context import (
        reset_current_signal,
        set_current_signal,
    )

    unrelated = Signal(
        source="a2a.task_complete",
        kind="completed",
        mode=SignalMode.COGNITION,
        payload={},
        target_agent=TARGET,
        causation_chain=[
            CausationFrame(
                agent_id=TARGET,
                source="cron.backup_snapshot",
                signal_id="sig-unrelated",
                turn_id="turn-unrelated",
                depth=1,
                emitted_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            )
        ],
    )

    token = set_current_signal(unrelated)
    try:
        result = await _schedule(feature, intent=SENTINEL)
    finally:
        reset_current_signal(token)

    assert result.status is not ToolResultStatus.ERROR, (
        f"an unrelated ancestor must not block a follow-up: {result.error!r}"
    )
    rows = await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?", (SELF_FOLLOWUP,)
    )
    assert rows, "an accepted follow-up must persist exactly one row"
