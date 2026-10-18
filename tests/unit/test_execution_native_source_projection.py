"""Canonical finite provenance separates observer C without rewriting sources."""
import asyncio

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, _explicit_control_chain,
    execution_commit_outcome, execution_error_with_cause,
    execution_native_source_error, execution_terminal_error,
)


@pytest.mark.parametrize("shape", ["plain", "cancellation", "group", "forged-attribute"])
def test_native_source_projection_preserves_exact_genuine_original(shape):
    native = ExecutionAuthorityError("original native authority source")
    original = native
    if shape == "cancellation":
        original = asyncio.CancelledError("actual source cancellation")
        original.__cause__ = native
    elif shape == "group":
        original = BaseExceptionGroup("genuine original source group", [native, asyncio.CancelledError("actual source leaf")])
    elif shape == "forged-attribute":
        original._kestrel_finite_native_sources = (object(), ())
    before = tuple(_explicit_control_chain(original))
    withdrawal = asyncio.CancelledError("independent finite caller")
    finite = execution_terminal_error(original, withdrawal, retain_originals=True, isolate_control_originals=True)
    projected = execution_native_source_error(finite)
    assert projected is original
    assert tuple(_explicit_control_chain(original)) == before
    assert withdrawal in tuple(_explicit_control_chain(finite))
    assert withdrawal not in tuple(_explicit_control_chain(projected))
    assert execution_native_source_error(projected) is projected


def test_nested_finite_projections_carry_sources_without_observer_history():
    first, second = ExecutionAuthorityError("first source"), ExecutionAuthorityError("second source")
    c1, c2 = asyncio.CancelledError("first observer"), asyncio.CancelledError("second observer")
    finite1 = execution_terminal_error(first, c1, retain_originals=True, isolate_control_originals=True)
    finite2 = execution_terminal_error(finite1, second, c2, retain_originals=True, isolate_control_originals=True)
    projected = execution_native_source_error(finite2)
    chain = tuple(_explicit_control_chain(projected))
    assert first in chain and second in chain and c1 not in chain and c2 not in chain
    assert tuple(_explicit_control_chain(first)) == (first,)
    assert tuple(_explicit_control_chain(second)) == (second,)


def test_repeated_shared_multi_source_projection_does_not_accumulate_wrappers():
    from types import SimpleNamespace
    from kestrel_sovereign.kestrel_agent import _retain_feature_lifecycle_safe_mode
    sources = (ExecutionAuthorityError("first source"), ExecutionAuthorityError("second source"))
    agent = SimpleNamespace()
    for index in range(5):
        withdrawal = asyncio.CancelledError(f"independent observer {index}")
        finite = execution_terminal_error(*sources, withdrawal,
            retain_originals=True, isolate_control_originals=True)
        _retain_feature_lifecycle_safe_mode(agent, "same original sources", finite)
        chain = tuple(_explicit_control_chain(agent._feature_lifecycle_control))
        assert len(chain) == 3 and all(source in chain for source in sources)
        assert withdrawal not in chain


@pytest.mark.asyncio
async def test_physical_task_harvest_projects_its_canonical_finite_source():
    from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
    source = ExecutionAuthorityError("declared native source for projection unit test")
    withdrawal = asyncio.CancelledError("independent nested observer")
    finite = execution_terminal_error(source, withdrawal,
        retain_originals=True, isolate_control_originals=True)
    async def operation():
        raise finite
    task = create_owned_task(operation(), retain_originals=True)
    await await_owned_task(task)
    debt = RuntimeError("declared unit owner association")
    debt._kestrel_native_retirement_debts = ((object(), task),)
    joined = execution_terminal_error(finite, debt,
        retain_originals=True, isolate_control_originals=True)
    projected = execution_native_source_error(joined)
    assert source in tuple(_explicit_control_chain(projected))
    assert withdrawal not in tuple(_explicit_control_chain(projected))


@pytest.mark.parametrize("outcome", ["unknown", "committed"])
def test_core_classification_and_diagnostic_wrappers_forward_finite_sources(outcome):
    source = ExecutionAuthorityError("declared original native source")
    withdrawal = asyncio.CancelledError("independent finite observer")
    finite = execution_terminal_error(source, withdrawal,
        retain_originals=True, isolate_control_originals=True)
    classified = execution_error_with_cause(ExecutionCommitOutcomeError(outcome), finite)
    current = classified
    for index in range(5):
        current = execution_error_with_cause(RuntimeError(f"Core diagnostic wrapper {index}"), current)
        projected = execution_native_source_error(current)
        chain = tuple(_explicit_control_chain(projected))
        assert source in chain and withdrawal not in chain and len(chain) == 2
        assert execution_commit_outcome(projected) == outcome
        assert withdrawal in tuple(_explicit_control_chain(current))
    assert tuple(_explicit_control_chain(source)) == (source,)


def test_core_classification_does_not_split_genuine_unmarked_source_cancellation():
    source = asyncio.CancelledError("genuine source cancellation")
    source.__cause__ = ExecutionAuthorityError("source native receipt")
    before = tuple(_explicit_control_chain(source))
    wrapped = execution_error_with_cause(ExecutionCommitOutcomeError("unknown"), source)
    projected = execution_native_source_error(wrapped)
    assert projected is wrapped and source in tuple(_explicit_control_chain(projected))
    assert tuple(_explicit_control_chain(source)) == before


@pytest.mark.parametrize("shape", ["source-cancellation", "source-group", "forged-provenance"])
def test_finite_observer_reader_never_splits_genuine_or_forged_sources(shape):
    from kestrel_sovereign.execution_custody import execution_observer_cancellations
    native = ExecutionAuthorityError("genuine source authority")
    source = asyncio.CancelledError("genuine source cancellation")
    source.__cause__ = native
    if shape == "source-group":
        source = BaseExceptionGroup("genuine unmarked source group", [native, source])
    if shape == "forged-provenance":
        source._kestrel_finite_native_sources = (object(), (native,))
    before = tuple(_explicit_control_chain(source))
    assert execution_observer_cancellations(source) == ()
    assert tuple(_explicit_control_chain(source)) == before


@pytest.mark.parametrize("outcome", ["unknown", "committed"])
@pytest.mark.parametrize("source_group", [False, True])
def test_core_finite_observer_reader_keeps_exact_distinct_observations(outcome, source_group):
    from kestrel_sovereign.execution_custody import execution_observer_cancellations
    native = ExecutionAuthorityError("genuine original authority")
    source_c = asyncio.CancelledError("genuine source cancellation")
    source_c.__cause__ = native
    source = BaseExceptionGroup("genuine whole source", [native, source_c]) if source_group else source_c
    first, second = asyncio.CancelledError("actual first observation"), asyncio.CancelledError("actual second observation")
    finite = execution_terminal_error(source, first, retain_originals=True, isolate_control_originals=True)
    classified = execution_error_with_cause(ExecutionCommitOutcomeError(outcome), finite)
    wrapped = execution_error_with_cause(RuntimeError("ordinary Core diagnostic"), classified)
    final = execution_terminal_error(wrapped, second, retain_originals=True, isolate_control_originals=True)
    observations = execution_observer_cancellations(final)
    assert {id(error) for error in observations} == {id(first), id(second)}
    assert source_c not in observations
    assert all(error not in tuple(_explicit_control_chain(execution_native_source_error(final))) for error in observations)
