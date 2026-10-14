"""A follow-up runs only through the contract the scheduler registers (#3131).

The scheduler registers ``cron.self_followup`` OPTIONAL. When another feature
already holds that name with a DIFFERENT contract, the incumbent is kept and
the clash is reported as a MISMATCH. Before #3131 the follow-up path then
checked only that the incumbent was COGNITION-shaped (allows COGNITION, has a
prompt, has a summary when session-bound), accepted the row, and fired it
through the foreign schema, prompt, trust policy and pre-turn guard.
Reproduced against the pre-fix code with the real SchedulerFeature, runner and
dispatcher: the row was persisted, the turn rendered the foreign prompt, the
audit used the foreign redactor, and the occurrence was filed ``success``.

These tests drive the same real stack (``followup_env``). Each fails when the
guard it covers is removed.
"""

from __future__ import annotations

import dataclasses
import logging
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import Trust
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.signals import (
    CLAIM_IMPERATIVE,
    RateLimit,
    RegistrationPolicy,
    RegistrationState,
)
from kestrel_sovereign.signals.sources.self_followup import (
    TASK_NAME as SELF_FOLLOWUP,
    build_self_followup_registration,
)

from tests.unit.test_self_followup_schedule import (
    SENTINEL,
    SOURCE,
    _drain,
    _schedule,
)
from tests.utils.scheduler_ticks import tick_and_settle

FOREIGN_MARKER = "FOREIGN-CONTRACT-PROMPT-3131"

# The scheduler's own registration with exactly one part of the contract
# changed. Each still passes the COGNITION / prompt / summary checks that
# accepted it before #3131, so only a whole-contract comparison refuses it.
FOREIGN_VARIANTS = ["prompt", "schema", "trust", "pre_turn_guard", "rate_limit"]


def _foreign_registration(agent, variant, tmp_path):
    own = build_self_followup_registration(agent)
    if variant == "prompt":
        prompt = tmp_path / "foreign_followup.md"
        prompt.write_text(f"{FOREIGN_MARKER} {{payload}}\n")
        return dataclasses.replace(own, prompt_template=prompt)
    if variant == "schema":
        return dataclasses.replace(own, schema=lambda payload: dict(payload))
    if variant == "trust":
        return dataclasses.replace(
            own, trust=Trust.UNTRUSTED, sanitizer=lambda payload: dict(payload)
        )
    if variant == "pre_turn_guard":
        # Drops the #3310 in-turn privacy guard along with the contract.
        return dataclasses.replace(own, pre_turn_guard=None)
    if variant == "rate_limit":
        return dataclasses.replace(own, rate_limit=RateLimit())
    raise AssertionError(variant)


def _take_name_first(agent, registration, holder_name="rival_feature"):
    """Replace the fixture's source with one a named feature registered first."""
    registry = agent.signal_registry
    registry.unregister(SOURCE)
    holder = SimpleNamespace(name=holder_name)
    outcome = registry.register_with_policy(
        registration,
        RegistrationPolicy.OPTIONAL,
        owner=holder,
        role=CLAIM_IMPERATIVE,
    )
    assert outcome.state is RegistrationState.REGISTERED
    return holder


def _scheduler_registers(feature):
    """What ``SchedulerFeature.initialize`` does for this source."""
    (outcome,) = feature._register_signal_sources(
        [build_self_followup_registration(feature.agent)],
        RegistrationPolicy.OPTIONAL,
    )
    return outcome


async def _rows(db):
    return await db.fetchall(
        "SELECT id FROM scheduled_tasks WHERE task_name = ?", (SELF_FOLLOWUP,)
    )


# ---------------------------------------------------------------------------
# 1. Foreign contract registered first -> refused, nothing persisted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", FOREIGN_VARIANTS)
async def test_foreign_contract_registered_first_refuses_the_follow_up(
    followup_env, tmp_path, variant
):
    agent, feature, _runner, db, _backend = followup_env
    foreign = _foreign_registration(agent, variant, tmp_path)
    _take_name_first(agent, foreign)

    # The existing mismatch report stays: the scheduler's registration is
    # kept out and reported, and the incumbent keeps the name.
    outcome = _scheduler_registers(feature)
    assert outcome.state is RegistrationState.MISMATCH
    assert agent.signal_registry.get(SOURCE) is foreign

    result = await _schedule(feature)

    assert result.status is ToolResultStatus.ERROR, (
        f"{variant}: a foreign contract was accepted"
    )
    assert result.data["refused"] == "foreign_source_contract"
    assert result.data["held_by"] == "rival_feature"
    assert "rival_feature" in result.error
    assert SOURCE in result.error
    assert not await _rows(db), f"{variant}: a refused follow-up was persisted"


@pytest.mark.asyncio
async def test_foreign_contract_held_by_the_host_is_named_as_such(
    followup_env, tmp_path
):
    agent, feature, _runner, db, _backend = followup_env
    registry = agent.signal_registry
    registry.unregister(SOURCE)
    registry.register(_foreign_registration(agent, "prompt", tmp_path))

    result = await _schedule(feature)

    assert result.status is ToolResultStatus.ERROR
    assert result.data["refused"] == "foreign_source_contract"
    assert result.data["held_by"] == "the host"
    assert not await _rows(db)


# ---------------------------------------------------------------------------
# 2. Scheduler-owned or equivalent registration -> scheduled and fired
# ---------------------------------------------------------------------------


async def _schedule_and_fire(agent, feature, runner, db):
    created = await _schedule(feature)
    assert created.status is ToolResultStatus.OK, created.error
    await tick_and_settle(runner)
    await _drain(agent)
    assert len(agent.turn_prompts) == 1, "the follow-up did not fire"
    prompt = agent.turn_prompts[0]
    assert SENTINEL in prompt
    assert prompt.startswith("[SELF_FOLLOWUP]"), "fired through another prompt"
    execution = await db.fetchone(
        "SELECT status FROM task_execution_log WHERE task_id = ?",
        (created.data["task_id"],),
    )
    assert execution is not None and execution[0] == "success"


@pytest.mark.asyncio
async def test_scheduler_owned_registration_schedules_and_fires(followup_env):
    agent, feature, runner, db, _backend = followup_env
    agent.signal_registry.unregister(SOURCE)

    outcome = _scheduler_registers(feature)
    assert outcome.state is RegistrationState.REGISTERED
    assert agent.signal_registry.owners_of(SOURCE) == (feature,)

    await _schedule_and_fire(agent, feature, runner, db)


@pytest.mark.asyncio
async def test_equivalent_incumbent_schedules_and_fires(followup_env):
    """Another feature holding an EQUIVALENT contract is not a conflict."""
    agent, feature, runner, db, _backend = followup_env
    _take_name_first(agent, build_self_followup_registration(agent))

    outcome = _scheduler_registers(feature)
    assert outcome.state is RegistrationState.ALREADY_EQUIVALENT

    await _schedule_and_fire(agent, feature, runner, db)


# ---------------------------------------------------------------------------
# 3. A row queued before the incumbent changed is refused at fire time
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", FOREIGN_VARIANTS)
async def test_row_queued_before_a_foreign_contract_is_refused_at_fire_time(
    followup_env, tmp_path, caplog, variant
):
    """Queued under the scheduler's contract; a rival holds the name at fire.

    The production shape is a row persisted on one boot and a newly installed
    feature registering ``cron.self_followup`` ahead of the scheduler on the
    next. The schedule-time refusal cannot see that, so the fire must.
    """
    agent, feature, runner, db, backend = followup_env
    created = await _schedule(feature)
    assert created.status is ToolResultStatus.OK, created.error

    _take_name_first(agent, _foreign_registration(agent, variant, tmp_path))
    assert _scheduler_registers(feature).state is RegistrationState.MISMATCH

    with caplog.at_level(logging.WARNING):
        await tick_and_settle(runner)
        await _drain(agent)

    assert not agent.turn_prompts, f"{variant}: fired through a foreign contract"
    signal_row = await backend.fetch_one(
        "SELECT source FROM signal_log WHERE source = ?", (SOURCE,)
    )
    assert signal_row is None, "the row reached the dispatcher"

    execution = await db.fetchone(
        "SELECT status, result_text FROM task_execution_log WHERE task_id = ?",
        (created.data["task_id"],),
    )
    assert execution is not None and execution[0] == "failed"
    assert "rival_feature" in execution[1]
    assert SENTINEL not in execution[1]

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "#3131" in record.getMessage()
    ]
    assert warnings, "the fire-time refusal must be logged at WARNING"
    message = warnings[0].getMessage()
    assert "rival_feature" in message
    assert SENTINEL not in message
