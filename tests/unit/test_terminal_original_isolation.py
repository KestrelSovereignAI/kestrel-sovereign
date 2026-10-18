"""Pure terminal-composition contracts; native I/O is tested separately."""

import asyncio

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    _explicit_control_chain,
    execution_commit_outcome,
    execution_terminal_error,
    is_execution_control_error,
)


def native_receipt(kind):
    if kind == "authority":
        return ExecutionAuthorityError("declared native authority receipt")
    marker = ExecutionCommitOutcomeError(
        "committed" if kind == "committed" else "unknown"
    )
    if kind == "native-cancel":
        error = asyncio.CancelledError("declared native writer withdrawal")
        error.__cause__ = marker
        return error
    return marker


KINDS = ("authority", "unknown", "committed", "native-cancel")


@pytest.mark.parametrize("kind", KINDS)
def test_finite_native_projection_does_not_mutate_shared_writer_original(kind):
    writer = native_receipt(kind)
    withdrawal = asyncio.CancelledError("finite caller withdrawal")
    before = tuple(_explicit_control_chain(writer))
    terminal = execution_terminal_error(
        writer,
        withdrawal,
        retain_originals=True,
        isolate_control_originals=True,
    )
    assert terminal is not writer
    chain = tuple(_explicit_control_chain(terminal))
    assert writer in chain and withdrawal in chain
    assert is_execution_control_error(terminal)
    assert execution_commit_outcome(terminal) == execution_commit_outcome(writer)
    assert tuple(_explicit_control_chain(writer)) == before
    assert withdrawal not in tuple(_explicit_control_chain(writer))
    original_terminal = terminal
    for _ in range(20):
        terminal = execution_terminal_error(
            terminal,
            writer,
            withdrawal,
            retain_originals=True,
            isolate_control_originals=True,
        )
        assert terminal is original_terminal
        assert tuple(_explicit_control_chain(terminal)) == chain


@pytest.mark.parametrize("kind", KINDS)
def test_single_original_and_duplicate_observation_keep_identity(kind):
    original = native_receipt(kind)
    assert (
        execution_terminal_error(
            original,
            original,
            None,
            retain_originals=True,
            isolate_control_originals=True,
        )
        is original
    )


@pytest.mark.parametrize("kind", KINDS)
def test_original_enclosing_carrier_is_not_replaced_by_its_native_leaf(kind):
    original = native_receipt(kind)
    withdrawal = asyncio.CancelledError("enclosed caller")
    group = BaseExceptionGroup("actual finite original carrier", [original, withdrawal])
    assert (
        execution_terminal_error(
            original,
            group,
            withdrawal,
            retain_originals=True,
            isolate_control_originals=True,
        )
        is group
    )


@pytest.mark.parametrize("kind", KINDS)
def test_default_strongest_receipt_reader_still_retains_bounded_identity(kind):
    original = native_receipt(kind)
    withdrawal = asyncio.CancelledError("not a retained hot reader history")
    before = tuple(_explicit_control_chain(original))
    assert execution_terminal_error(original, withdrawal) is original
    assert tuple(_explicit_control_chain(original)) == before


def test_ordinary_finite_compatibility_and_cancellation_preference_stay_unchanged():
    original = OSError("ordinary opening failure")
    withdrawal = asyncio.CancelledError("actual caller")
    terminal = execution_terminal_error(
        original,
        withdrawal,
        retain_originals=True,
        prefer_cancellation=True,
        isolate_control_originals=True,
    )
    assert terminal is withdrawal and original in tuple(
        _explicit_control_chain(terminal)
    )
    assert not is_execution_control_error(terminal)
