"""Typed local receipt retention cannot weaken shared-runtime failures."""

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCustody, ProcessRuntimeExecutionFence, bind_execution_custody_snapshot,
    execution_denial_scopes,
)
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, _LeaseRenewalState
from kestrel_sovereign.storage.async_database import AsyncDatabase


def expired_receipt():
    state = _LeaseRenewalState()
    state.lost.set()
    scope = ExecutionCustody(state)
    with pytest.raises(BaseException) as caught:
        scope.require_work()
    assert execution_denial_scopes(caught.value) == (scope,)
    return caught.value


@pytest.mark.asyncio
async def test_completed_local_history_retains_first_original_receipts_with_bound(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "local-history.db"))
    notifications = []
    runner = SchedulerRunner(db, "original", lambda *args: None, on_protocol_failure=notifications.append)
    originals = [expired_receipt() for _ in range(64)]
    try:
        for receipt in originals:
            runner._latch_protocol_failure(receipt)
        assert runner._occurrence_control_receipts == originals[:32]
        assert runner._execution_control_error is None
        assert runner.readiness_failure is None and not notifications
        for receipt in originals:
            [scope] = execution_denial_scopes(receipt)
            with pytest.raises(BaseException):
                scope.require_work()
    finally:
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_group_with_original_runtime_denial_is_not_localized(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "runtime-control.db"))
    notifications = []
    runner = SchedulerRunner(db, "original", lambda *args: None, on_protocol_failure=notifications.append)
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original-runtime"))
    scope.fence.retire()
    with pytest.raises(BaseException) as caught:
        scope.require_work()
    error = BaseExceptionGroup("original occurrence and runtime losses", [expired_receipt(), caught.value])
    try:
        runner._latch_protocol_failure(error)
        assert runner.readiness_failure is error and notifications == [error]
        assert runner._execution_control_error is error
        assert not runner._occurrence_control_receipts
    finally:
        with pytest.raises(BaseException):
            await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_native_unsupported_backend_denial_still_fails_protocol(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "unsupported.db"))
    notifications = []
    runner = SchedulerRunner(db, "original", lambda *args: None, on_protocol_failure=notifications.append)
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original-runtime"))
    try:
        with bind_execution_custody_snapshot((scope,)):
            with pytest.raises(BaseException) as caught:
                await db.fetchone("SELECT 1")
        assert execution_denial_scopes(caught.value) is None
        runner._latch_protocol_failure(caught.value)
        assert runner.readiness_failure is caught.value and notifications == [caught.value]
        assert not runner._occurrence_control_receipts
    finally:
        with pytest.raises(BaseException):
            await runner.stop()
        await db.close()
