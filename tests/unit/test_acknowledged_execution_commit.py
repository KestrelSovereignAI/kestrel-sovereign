"""Receipt classification only; actual transaction proofs live in integration."""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    AcknowledgedExecutionCommit,
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    execution_commit_outcome,
    execution_denial_scopes,
    execution_terminal_error,
    is_execution_control_error,
    retain_acknowledged_execution_commit,
)


@pytest.mark.asyncio
async def test_settled_fact_cannot_hide_combined_unscoped_control():
    receipt = await acknowledged_receipt()
    reset = create_owned_task(asyncio.sleep(0))
    await await_owned_task(reset)
    receipt.retirement_owner = reset
    original = retain_acknowledged_execution_commit(
        asyncio.CancelledError("original"), receipt
    )
    unscoped = ExecutionAuthorityError("original unscoped denial")
    selected = execution_terminal_error(original, unscoped)
    assert selected is original
    assert execution_commit_outcome(selected) == "committed"
    assert execution_denial_scopes(selected) is None
    # Every aggregation is another boundary: a selected cancellation is an
    # explicit provenance carrier even though it is not an AuthorityError.
    replacement = retain_acknowledged_execution_commit(
        asyncio.CancelledError("different original observer"), receipt
    )
    selected_again = execution_terminal_error(replacement, selected)
    assert selected_again is replacement
    assert execution_denial_scopes(selected_again) is None


@pytest.mark.asyncio
async def test_selected_cancel_keeps_scoped_provenance_across_reaggregation():
    receipt = await acknowledged_receipt()
    reset = create_owned_task(asyncio.sleep(0))
    await await_owned_task(reset)
    receipt.retirement_owner = reset
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("retired-original"))
    scope.revoke("original denial")
    with pytest.raises(ExecutionAuthorityError) as caught:
        scope.require_work()
    original = retain_acknowledged_execution_commit(asyncio.CancelledError(), receipt)
    selected = execution_terminal_error(original, caught.value)
    assert execution_denial_scopes(selected) == (scope,)
    replacement = retain_acknowledged_execution_commit(
        asyncio.CancelledError(), receipt
    )
    assert execution_denial_scopes(execution_terminal_error(replacement, selected)) == (
        scope,
    )


@pytest.mark.asyncio
async def test_settled_fact_never_localizes_legacy_unresolved_committed_carrier():
    receipt = await acknowledged_receipt()
    reset = create_owned_task(asyncio.sleep(0))
    await await_owned_task(reset)
    receipt.retirement_owner = reset
    selected = retain_acknowledged_execution_commit(asyncio.CancelledError(), receipt)
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("unresolved original"))
    scope.preserve_commit_uncertainty("committed")
    with pytest.raises(ExecutionCommitOutcomeError) as caught:
        scope.require_work()
    aggregate = execution_terminal_error(selected, caught.value)
    assert aggregate is selected
    assert execution_commit_outcome(aggregate) == "committed"
    assert execution_denial_scopes(aggregate) is None
    # Repeated future denials must not accumulate fresh error/traceback
    # histories just to retain this one irreversible unresolved fact.
    for _ in range(100):
        with pytest.raises(ExecutionCommitOutcomeError) as repeated:
            scope.require_work()
        assert execution_terminal_error(aggregate, repeated.value) is aggregate
        assert aggregate.__cause__ is None
        assert vars(aggregate)["_kestrel_unresolved_commit_outcome"] == "committed"
        assert execution_denial_scopes(aggregate) is None
    replacement = retain_acknowledged_execution_commit(
        asyncio.CancelledError(), receipt
    )
    assert (
        execution_denial_scopes(execution_terminal_error(replacement, aggregate))
        is None
    )


async def acknowledged_receipt():
    commit = create_owned_task(asyncio.sleep(0))
    assert (await await_owned_task(commit)).error is None
    return AcknowledgedExecutionCommit(commit)


def test_native_original_retention_keeps_enclosing_cancellation_without_cause_cycle():
    from kestrel_sovereign.execution_custody import _explicit_control_chain

    cause = ValueError("original explicit cause")
    cancellation = asyncio.CancelledError("original body cancellation")
    cancellation.__cause__ = cause
    closer = OSError("actual rollback failed")
    terminal = execution_terminal_error(cancellation, closer, retain_originals=True)
    chain = tuple(_explicit_control_chain(terminal))
    assert cancellation in chain and cause in chain and closer in chain
    assert terminal not in tuple(_explicit_control_chain(terminal.__cause__))


@pytest.mark.asyncio
async def test_observer_receipt_is_local_only_after_original_retirement():
    receipt = await acknowledged_receipt()
    original = asyncio.CancelledError("original observer")
    assert retain_acknowledged_execution_commit(original, receipt) is original
    assert execution_commit_outcome(original) == "committed"
    assert is_execution_control_error(original)
    assert execution_denial_scopes(original) is None
    reset = create_owned_task(asyncio.sleep(0))
    receipt.retirement_owner = reset
    assert execution_denial_scopes(original) is None
    assert (await await_owned_task(reset)).error is None
    assert receipt.retired
    assert execution_denial_scopes(original) == ()
    assert original.args == ("original observer",)


@pytest.mark.asyncio
async def test_settled_commit_does_not_localize_unknown_or_actual_failed_closer():
    receipt = await acknowledged_receipt()
    reset = create_owned_task(asyncio.sleep(0))
    await await_owned_task(reset)
    receipt.retirement_owner = reset
    original = retain_acknowledged_execution_commit(
        asyncio.CancelledError("known"), receipt
    )
    unknown = ExecutionCommitOutcomeError("unknown")
    grouped = BaseExceptionGroup("original facts", [original, unknown])
    assert execution_commit_outcome(grouped) == "unknown"
    assert execution_denial_scopes(grouped) is None
    assert execution_terminal_error(original, unknown) is unknown

    failure = OSError("original native reset failure")

    async def failed_closer():
        raise failure

    close_owner = create_owned_task(failed_closer())
    assert (await await_owned_task(close_owner)).error is failure
    original._kestrel_native_retirement_debts = ((object(), close_owner),)
    assert execution_commit_outcome(original) == "committed"
    assert execution_denial_scopes(original) is None
    assert execution_terminal_error(original, asyncio.CancelledError()) is original


@pytest.mark.asyncio
async def test_only_denied_scope_retains_acknowledged_commit_after_retirement():
    receipt = await acknowledged_receipt()
    reset = create_owned_task(asyncio.sleep(0))
    await await_owned_task(reset)
    receipt.retirement_owner = reset
    healthy = ExecutionCustody(ProcessRuntimeExecutionFence("healthy"))
    retired = ExecutionCustody(ProcessRuntimeExecutionFence("retired"))
    retired.revoke("original local denial")
    retired.retain_acknowledged_commit(receipt)
    for _ in range(2):
        with pytest.raises(ExecutionAuthorityError) as caught:
            retired.require_work()
        assert execution_commit_outcome(caught.value) == "committed"
        assert execution_denial_scopes(caught.value) == (retired,)
        healthy.require_work()
    unscoped = retain_acknowledged_execution_commit(
        ExecutionAuthorityError("unscoped"), receipt
    )
    assert execution_denial_scopes(unscoped) is None
