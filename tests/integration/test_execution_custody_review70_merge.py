"""Upstream after-commit callbacks must retain real custody/control outcomes."""

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot,
    execution_commit_outcome,
    execution_denial_scopes,
    is_execution_control_error,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.storage.async_database import AsyncDatabase

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_after_commit_native_authority_loss_is_committed_control_not_logged_success(native_pg):
    backend, observer, schema = native_pg
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original-after-commit-runtime"))
    receipts, later_callbacks = [], []

    def callback():
        scope.revoke("original runtime retired after actual database commit")
        try:
            scope.require_work()
        except BaseException as receipt:
            receipts.append(receipt)
            raise

    with bind_execution_custody_snapshot((scope,)):
        with pytest.raises(BaseException) as caught:
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (70, 'native committed write')")
                backend.call_after_commit(callback)
                backend.call_after_commit(lambda: later_callbacks.append(True))
    assert await observer.fetchval(f'SELECT value FROM "{schema}".effects WHERE id=70') == "native committed write"
    assert execution_commit_outcome(caught.value) == "committed"
    assert execution_denial_scopes(receipts[0]) == (scope,)
    assert caught.value.__cause__ is receipts[0]
    assert not later_callbacks
    fatal = []
    runner = SchedulerRunner(AsyncDatabase(backend), "original-host-runtime",
        lambda *args: None, on_protocol_failure=fatal.append)
    runner._latch_protocol_failure(caught.value)
    assert runner.readiness_failure is caught.value and fatal == [caught.value]
    assert runner._execution_control_error is caught.value
    with pytest.raises(BaseException) as retained:
        scope.require_work()
    assert is_execution_control_error(retained.value)
