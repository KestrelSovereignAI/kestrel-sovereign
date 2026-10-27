"""Assertions for a refused opaque factory with no exact native carrier."""

from kestrel_sovereign._async_ownership import OwnedTaskAdmission
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)


def assert_unknown_task_admission(
    error, original, *, admission=None, source=None, commit_outcome=None,
):
    """Require original refusal plus its actual unresolved acceptance receipt.

    Closing the submitted coroutine is not proof that a custom constructor
    accepted no other carrier. The receipt must retain that uncertainty without
    inventing a Task, a backend COMMIT, or debt on the shared source exception.
    A refusal already carrying native control requires its immutable BEFORE
    snapshot and explicit expected COMMIT outcome; prior debts must survive.
    """
    assert any(carrier is original for carrier in _explicit_control_chain(error))
    assert is_execution_control_error(error)
    if source is None:
        assert commit_outcome is None, "native source requires its BEFORE snapshot"
        assert not execution_cleanup_debts(original), "decorated the shared refusal"
        prior_debts = ()
    else:
        assert source.error is original, "snapshot belongs to another source"
        source.assert_retained(error)
        prior_debts = source.debts
    assert execution_commit_outcome(original) == commit_outcome
    assert execution_commit_outcome(error) == commit_outcome
    terminal_debts = execution_cleanup_debts(error)
    assert all(
        any(resource is retained and task is owner for retained, owner in terminal_debts)
        for resource, task in prior_debts
    ), "lost original native debt while retaining Task admission"
    receipts = [
        (resource, task)
        for resource, task in terminal_debts
        if type(resource) is OwnedTaskAdmission
        and not any(resource is prior and task is owner for prior, owner in prior_debts)
    ]
    assert len(receipts) == 1, "lost or duplicated original Task admission debt"
    receipt, task = receipts[0]
    assert all(receipt is not resource for resource, _ in prior_debts), "reused original debt resource as a fresh admission"
    if admission is not None:
        assert receipt is admission, "substituted a different admission receipt"
    assert receipt._acceptance_unknown
    assert task is None and receipt.task is None
    assert receipt._reconciled_tasks == ()
    assert receipt._unpublished_wrapper is None
    return receipt
