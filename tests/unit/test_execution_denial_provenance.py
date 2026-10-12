"""Original typed scope denial never masks runtime/commit/cleanup control."""
import pytest
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ProcessRuntimeExecutionFence,
    bind_execution_custody, execution_denial_scopes, execution_terminal_error,
)
from kestrel_sovereign.features.scheduler.runner import _LeaseRenewalState


@pytest.mark.parametrize("kind", ["occurrence", "runtime"])
def test_exact_native_scope_is_retained_across_terminal_selection(kind):
    fence = _LeaseRenewalState() if kind == "occurrence" else ProcessRuntimeExecutionFence("did:test:original")
    with bind_execution_custody(fence) as original:
        pass
    with pytest.raises(ExecutionAuthorityError) as caught:
        original.require_work()
    assert execution_denial_scopes(caught.value) == (original,)
    assert execution_denial_scopes(execution_terminal_error(caught.value, OSError("unrelated noise"))) == (original,)
    assert execution_denial_scopes(execution_terminal_error(caught.value,
        ExecutionAuthorityError("unscoped authority loss"))) is None


def test_two_original_scope_denials_and_commit_uncertainty_are_never_reduced():
    with bind_execution_custody(_LeaseRenewalState()) as occurrence:
        pass
    with bind_execution_custody(ProcessRuntimeExecutionFence("did:test:runtime")) as runtime:
        pass
    errors = []
    for scope in (occurrence, runtime):
        with pytest.raises(ExecutionAuthorityError) as caught:
            scope.require_work()
        errors.append(caught.value)
    assert execution_denial_scopes(execution_terminal_error(*errors)) == (occurrence, runtime)
    assert execution_denial_scopes(execution_terminal_error(*errors, ExecutionCommitOutcomeError("unknown"))) is None
