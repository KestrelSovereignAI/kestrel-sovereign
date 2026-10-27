"""Assertions for a refused opaque factory with no exact native carrier."""

from kestrel_sovereign._async_ownership import OwnedTaskAdmission
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)


def assert_unknown_task_admission(error, original, *, admission=None):
    """Require original refusal plus its actual unresolved acceptance receipt.

    Closing the submitted coroutine is not proof that a custom constructor
    accepted no other carrier. The receipt must retain that uncertainty without
    inventing a Task, a backend COMMIT, or debt on the shared source exception.
    """
    assert original in tuple(_explicit_control_chain(error))
    assert is_execution_control_error(error)
    assert execution_commit_outcome(error) is None
    receipts = [
        (resource, task)
        for resource, task in execution_cleanup_debts(error)
        if type(resource) is OwnedTaskAdmission
    ]
    assert len(receipts) == 1, "lost or duplicated original Task admission debt"
    receipt, task = receipts[0]
    if admission is not None:
        assert receipt is admission, "substituted a different admission receipt"
    assert receipt._acceptance_unknown
    assert task is None and receipt.task is None
    assert receipt._reconciled_tasks == ()
    assert receipt._unpublished_wrapper is None
    assert not execution_cleanup_debts(original), "decorated the shared refusal"
    return receipt
