"""Unit tests for ``ResourceCustody``: release only on a truthful confirmation (#3522)."""

import pytest

from kestrel_sovereign.agent import custody as custody_module
from kestrel_sovereign.agent.custody import (
    FEATURE_OWNER,
    TRUTHFUL_CLOSE_OWNERS,
    ReleaseOutcome,
    ResourceCustody,
    feature_resource_name,
    reported_outcome,
)


def _step(outcome, calls=None):
    async def step():
        if calls is not None:
            calls.append(outcome)
        return outcome

    return step


def test_only_the_task_manager_is_trusted_to_report_its_release():
    """Every other owner's close still swallows failures (#3559, #3560)."""
    assert TRUTHFUL_CLOSE_OWNERS == frozenset({"task_manager"})


@pytest.mark.asyncio
async def test_an_unconverted_owner_that_reports_released_stays_held():
    custody = ResourceCustody()
    custody.acquire("llm_service")
    calls: list = []

    outcome = await custody.release(
        "llm_service", _step(ReleaseOutcome.RELEASED, calls)
    )

    assert calls == [ReleaseOutcome.RELEASED], "the release step still runs"
    assert outcome is ReleaseOutcome.UNKNOWN
    assert custody.held == ("llm_service",)


@pytest.mark.asyncio
async def test_the_task_manager_reporting_released_leaves_custody():
    """Its close raises on a failed store close, so it is believed (#3558)."""
    custody = ResourceCustody()
    custody.acquire("task_manager")

    outcome = await custody.release(
        "task_manager", _step(ReleaseOutcome.RELEASED)
    )

    assert outcome is ReleaseOutcome.RELEASED
    assert custody.held == ()


@pytest.mark.asyncio
async def test_a_truthful_owner_that_reports_released_leaves_custody():
    custody = ResourceCustody(truthful_owners={"storage"})
    custody.acquire("storage")
    custody.acquire("dispatcher")
    assert custody.held == ("storage", "dispatcher")

    assert (
        await custody.release("storage", _step(ReleaseOutcome.RELEASED))
        is ReleaseOutcome.RELEASED
    )
    assert custody.held == ("dispatcher",)


@pytest.mark.asyncio
async def test_the_module_allow_list_is_read_at_each_release(monkeypatch):
    custody = ResourceCustody()
    custody.acquire("storage")
    monkeypatch.setattr(custody_module, "TRUTHFUL_CLOSE_OWNERS", frozenset({"storage"}))

    assert (
        await custody.release("storage", _step(ReleaseOutcome.RELEASED))
        is ReleaseOutcome.RELEASED
    )
    assert custody.held == ()


@pytest.mark.parametrize(
    "outcome, reported",
    [
        (ReleaseOutcome.RETAINED, ReleaseOutcome.RETAINED),
        (ReleaseOutcome.UNKNOWN, ReleaseOutcome.UNKNOWN),
        (None, ReleaseOutcome.UNKNOWN),
        (True, ReleaseOutcome.UNKNOWN),
        ("released", ReleaseOutcome.UNKNOWN),
    ],
    ids=["retained", "unknown", "returns-nothing", "returns-truthy", "returns-str"],
)
@pytest.mark.asyncio
async def test_a_truthful_owner_reporting_anything_else_stays_held(outcome, reported):
    custody = ResourceCustody(truthful_owners={"worker"})
    custody.acquire("worker")

    assert await custody.release("worker", _step(outcome)) is reported
    assert custody.is_held("worker")


@pytest.mark.asyncio
async def test_a_step_that_raises_leaves_the_resource_held_and_propagates():
    custody = ResourceCustody(truthful_owners={"worker"})
    custody.acquire("worker")

    async def fails():
        raise RuntimeError("stop failed")

    with pytest.raises(RuntimeError, match="stop failed"):
        await custody.release("worker", fails)
    assert custody.is_held("worker")


@pytest.mark.asyncio
async def test_a_step_runs_for_a_resource_never_acquired():
    """Teardown still releases what was set up without custody."""
    custody = ResourceCustody()
    calls: list = []

    outcome = await custody.release(
        "memory_system", _step(ReleaseOutcome.RELEASED, calls)
    )

    assert outcome is ReleaseOutcome.RELEASED
    assert calls == [ReleaseOutcome.RELEASED]
    assert custody.held == ()


def test_an_owner_is_named_at_acquisition_and_kept_on_reacquiring():
    custody = ResourceCustody(truthful_owners={FEATURE_OWNER})
    name = feature_resource_name("Talon")
    custody.acquire(name, FEATURE_OWNER)
    custody.acquire(name, "something-else")

    assert custody.settle(name, ReleaseOutcome.RELEASED) is ReleaseOutcome.RELEASED
    assert custody.held == ()


def test_the_owner_defaults_to_the_resource_name():
    custody = ResourceCustody(truthful_owners={FEATURE_OWNER})
    custody.acquire("storage")

    assert custody.settle("storage", ReleaseOutcome.RELEASED) is ReleaseOutcome.UNKNOWN
    assert custody.held == ("storage",)


def test_feature_resource_names_are_prefixed():
    assert feature_resource_name("SecurityFeature") == "feature:SecurityFeature"


@pytest.mark.parametrize("outcome", list(ReleaseOutcome))
def test_a_close_that_returns_an_outcome_reports_it(outcome):
    assert reported_outcome(outcome) is outcome


@pytest.mark.parametrize("result", [None, True, False, "retained", 0])
def test_a_close_that_returns_anything_else_reports_released(result):
    """Custody believes that only from a truthful owner."""
    assert reported_outcome(result) is ReleaseOutcome.RELEASED
