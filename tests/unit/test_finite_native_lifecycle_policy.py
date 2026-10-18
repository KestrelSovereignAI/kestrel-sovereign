"""Native finite callers and manager retention have different lifetimes."""
import ast
import asyncio
from itertools import permutations
from pathlib import Path

import pytest

from kestrel_sovereign.execution_custody import (
    AcknowledgedExecutionCommit, ExecutionAuthorityError, ExecutionCommitOutcomeError,
    _explicit_control_chain, execution_cleanup_debts, execution_commit_outcome,
    execution_denial_scopes, execution_finite_error, execution_native_source_error,
    execution_observation_error,
)
from kestrel_sovereign.multi_agent.agent_manager import _merge_manager_cleanup_control
from tests.utils.original_failure import OriginalFailure


def source(kind):
    if kind == "denied":
        return ExecutionAuthorityError("original scope denied")
    if kind in ("committed", "unknown"):
        return ExecutionCommitOutcomeError(kind)
    if kind == "source-cancel":
        error = asyncio.CancelledError("genuine original source cancellation")
        error.__cause__ = ExecutionCommitOutcomeError("unknown")
        return error
    return BaseExceptionGroup("genuine foreign native group", [asyncio.CancelledError("source group member"), ExecutionCommitOutcomeError("unknown")])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["denied", "committed", "unknown", "source-cancel", "source-group"])
async def test_finite_native_composer_keeps_actual_current_c_and_immutable_sources(kind):
    native = OriginalFailure.capture(source(kind))
    entered, released, withdrawals = asyncio.Event(), asyncio.Event(), []
    async def caller():
        entered.set()
        try:
            await released.wait()
        except asyncio.CancelledError as error:
            withdrawals.append(error)
            return execution_finite_error(native.error, error)
    task = asyncio.create_task(caller())
    try:
        await entered.wait()
        task.cancel("actual current finite caller withdrew")
        finite = await task
        native.assert_retained(finite)
        assert withdrawals[0] in tuple(_explicit_control_chain(finite))
        projected = execution_native_source_error(finite)
        native.assert_retained(projected)
        assert withdrawals[0] not in tuple(_explicit_control_chain(projected))
    finally:
        released.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("retain", [False, True])
def test_finite_nonnative_caused_cancellation_keeps_generic_default(retain):
    error, ordinary = asyncio.CancelledError("ordinary source"), ValueError("ordinary source cause")
    error.__cause__ = ordinary
    assert execution_finite_error(error, retain_originals=retain) is (error if retain else ordinary)


@pytest.mark.parametrize("kind", ["ordinary", "denied", "unknown", "source-cancel", "source-group"])
def test_observation_of_already_enclosing_original_does_not_add_envelope_history(kind):
    original = OriginalFailure.capture(OSError("original ordinary error") if kind == "ordinary" else source(kind))
    sibling = OriginalFailure.capture(ValueError("independent original sibling"))
    enclosing = execution_observation_error(original.error, sibling.error)
    assert execution_observation_error(enclosing, original.error, sibling.error) is enclosing
    original.assert_retained(enclosing)
    sibling.assert_retained(enclosing)


@pytest.mark.parametrize("composer", ["observation", "finite"])
@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("kind", ["denied", "committed", "unknown", "source-cancel", "source-group"])
def test_finite_composition_keeps_cyclic_component_beside_independent_original(composer, order, kind):
    # Pure explicit-graph contract, not native I/O/authority acceptance.
    native = OriginalFailure.capture(source(kind))
    first = RuntimeError("original cyclic outer carrier")
    second = BaseExceptionGroup("original cyclic source carrier", [native.error])
    first.__cause__ = second
    second.__cause__ = first
    originals = [OriginalFailure.capture(error) for error in
                 (first, second, OSError("independent original outside the cycle"))]
    errors = [originals[index].error for index in order]
    terminal = (execution_observation_error(*errors) if composer == "observation" else
                execution_finite_error(*errors, retain_originals=True, isolate_control_originals=True))
    for original in (*originals, native):
        original.assert_retained(terminal)
    assert execution_commit_outcome(terminal) == execution_commit_outcome(native.error)
    # Re-observation must remain bounded even when one component is cyclic.
    assert execution_observation_error(terminal, *errors) is terminal


def test_all_three_lifecycle_consumers_use_explicit_finite_policy():
    root = Path(__file__).resolve().parents[2]
    for relative in ("server.py", "signals/dispatcher.py", "multi_agent/agent_manager.py"):
        tree = ast.parse((root / "kestrel_sovereign" / relative).read_text())
        calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        assert "execution_finite_error" in calls
        assert "execution_terminal_error" not in calls
        assert not any(isinstance(node, ast.ImportFrom) and any(alias.name == "execution_terminal_error" for alias in node.names) for node in ast.walk(tree))
    # Structural guard only: not a runtime enforcement/all-alias proof.


@pytest.mark.parametrize("order", [
    ["denied", "committed", "unknown", "denied"],
    ["unknown", "committed", "denied", "unknown"],
    ["source-cancel", "source-group", "unknown", "committed"],
    ["source-group", "source-cancel", "unknown", "denied"],
])
def test_manager_summary_monotonic_priority_and_exact_selected_source(order):
    summary, selected = None, None
    priority = {None: 0, "committed": 1, "unknown": 2}
    for kind in order:
        original = OriginalFailure.capture(source(kind))
        if selected is None or priority[execution_commit_outcome(original.error)] > priority[execution_commit_outcome(selected.error)]:
            selected = original
        summary = _merge_manager_cleanup_control(summary, original.error)
        selected.assert_retained(summary)
        original.assert_retained(original.error)
        assert summary.__cause__ is selected.error
        projected = execution_native_source_error(summary)
        assert execution_commit_outcome(projected) == execution_commit_outcome(summary)
        selected.assert_retained(projected)


def test_many_distinct_completed_sources_do_not_form_manager_lifetime_history():
    summary, first = None, None
    for index in range(260):
        error = ExecutionCommitOutcomeError("unknown")
        error._source_probe = index
        first = first or error
        summary = _merge_manager_cleanup_control(summary, error)
    assert summary.__cause__ is first
    assert [error._source_probe for error in _explicit_control_chain(summary) if hasattr(error, "_source_probe")] == [0]
    assert execution_commit_outcome(execution_native_source_error(summary)) == "unknown"
    # This measures explicit cause/group/provenance retention, not global GC.


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["live", "failed", "cancelled", "refused", "accepted-unstarted", "successful-unrelated"])
async def test_manager_summary_never_infers_generic_resource_retirement_from_future(state):
    selected = source("unknown")
    resource = object()
    owner = None if state == "refused" else asyncio.get_running_loop().create_future()
    if state in ("failed", "accepted-unstarted"):
        owner.set_exception(OSError("original close failed or gate never entered"))
    elif state == "cancelled":
        owner.cancel()
    elif state == "successful-unrelated":
        owner.set_result(None)
    debt = RuntimeError("actual retained resource has no justified retirement ACK")
    debt._kestrel_native_retirement_debts = ((resource, owner),)
    snapshot = OriginalFailure.capture(debt)
    summary = _merge_manager_cleanup_control(None, selected)
    summary = _merge_manager_cleanup_control(summary, debt)
    snapshot.assert_retained(debt)
    assert any(value is resource and task is owner for value, task in execution_cleanup_debts(summary))
    assert any(value is resource and task is owner for value, task in execution_cleanup_debts(execution_native_source_error(summary)))
    if owner is not None:
        if not owner.done():
            owner.set_result(None)
        if not owner.cancelled():
            owner.exception()


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", ["unpublished", "pending", "failed", "retired"])
async def test_manager_summary_keeps_exact_mutable_ack_reset_ownership(reset):
    loop = asyncio.get_running_loop()
    commit = loop.create_future()
    commit.set_result(None)
    receipt = AcknowledgedExecutionCommit(commit)
    owner = None
    if reset != "unpublished":
        owner = loop.create_future()
        receipt.retirement.owner = owner
        if reset == "failed":
            owner.set_exception(OSError("actual reset failed"))
        elif reset == "retired":
            owner.set_result(None)
    error = RuntimeError("committed source observer")
    error._kestrel_acknowledged_commits = (receipt,)
    summary = _merge_manager_cleanup_control(None, source("unknown"))
    summary = _merge_manager_cleanup_control(summary, error)
    retained = vars(summary).get("_kestrel_acknowledged_commits", ())
    assert (receipt in retained) is (reset != "retired")
    assert execution_commit_outcome(execution_native_source_error(summary)) == "unknown"
    if reset == "unpublished":
        owner = loop.create_future()
        receipt.retirement.owner = owner
        assert retained[0].retirement_owner is owner
    if owner is not None and not owner.done():
        owner.set_result(None)
    if owner is not None:
        owner.exception()
    # Scalar committed history survives exact reset retirement, not a new ACK.
    next_summary = _merge_manager_cleanup_control(summary, source("denied"))
    assert (receipt in vars(next_summary).get("_kestrel_acknowledged_commits", ())) is (reset == "failed")
    assert execution_commit_outcome(next_summary) == "unknown"


def test_compacted_scopes_do_not_masquerade_as_only_selected_tenant():
    from kestrel_sovereign.execution_custody import ExecutionCustody
    from tests.integration.test_execution_authority_postgres import GenerationFence
    scopes = [ExecutionCustody(GenerationFence()), ExecutionCustody(GenerationFence())]
    errors = []
    for scope in scopes:
        error = ExecutionAuthorityError("one original tenant denied")
        error._kestrel_denied_execution_scopes = (scope,)
        errors.append(OriginalFailure.capture(error))
    summary = _merge_manager_cleanup_control(None, errors[0].error)
    assert execution_denial_scopes(summary) == (scopes[0],)
    summary = _merge_manager_cleanup_control(summary, errors[1].error)
    assert execution_denial_scopes(summary) is None
    forwarded = _merge_manager_cleanup_control(None, summary)
    assert execution_denial_scopes(forwarded) is None
    assert forwarded.__cause__ is errors[0].error
    for index, snapshot in enumerate(errors):
        snapshot.assert_retained(snapshot.error)
        assert execution_denial_scopes(snapshot.error) == (scopes[index],)
