"""Typed local receipt retention cannot weaken shared-runtime failures."""

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCustody, ProcessRuntimeExecutionFence, bind_execution_custody_snapshot,
    execution_denial_scopes, execution_terminal_error,
)
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, _LeaseRenewalState
from kestrel_sovereign.storage.async_database import AsyncDatabase


def expired_receipt(*, runner=None, with_state=False):
    state = _LeaseRenewalState(occurrence_owner=runner)
    state.lost.set()
    scope = ExecutionCustody(state)
    state.occurrence_scopes = (scope,)
    with pytest.raises(BaseException) as caught:
        scope.require_work()
    assert execution_denial_scopes(caught.value) == (scope,)
    return (caught.value, state) if with_state else caught.value


@pytest.mark.asyncio
async def test_completed_local_history_retains_first_original_receipts_with_bound(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "local-history.db"))
    notifications = []
    runner = SchedulerRunner(db, "original", lambda *args: None, on_protocol_failure=notifications.append)
    original_pairs = [expired_receipt(runner=runner, with_state=True) for _ in range(64)]
    originals = [receipt for receipt, _state in original_pairs]
    try:
        for receipt, state in original_pairs:
            runner._latch_protocol_failure(receipt, occurrence_state=state)
        assert runner._occurrence_control_receipts == originals[:32]
        for receipt in originals:
            runner._latch_protocol_failure(receipt)
            assert receipt._kestrel_original_scheduler_receipt.owner is runner
        assert runner._occurrence_control_receipts == originals[:32]
        assert runner._execution_control_error is None
        assert runner.readiness_failure is None and not notifications
        for receipt in originals:
            [scope] = execution_denial_scopes(receipt)
            with pytest.raises(BaseException):
                scope.require_work()
    finally:
        try:
            await runner.stop()
        finally:
            await db.close()

@pytest.mark.asyncio
async def test_other_runner_cannot_adopt_an_original_occurrence_state(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "foreign-state.db"))
    notifications = []
    runner = SchedulerRunner(db, "original", lambda *args: None, on_protocol_failure=notifications.append)
    foreign = SchedulerRunner(db, "foreign", lambda *args: None)
    error, state = expired_receipt(runner=foreign, with_state=True)
    try:
        runner._latch_protocol_failure(error, occurrence_state=state)
        assert runner.readiness_failure is error and notifications == [error]
        assert not runner._occurrence_control_receipts
    finally:
        with pytest.raises(BaseException):
            await runner.stop()
        await foreign.stop()
        await db.close()


@pytest.mark.asyncio
async def test_original_local_receipt_remains_local_after_occurrence_context_reset(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "retained-local-state.db"))
    notifications = []
    runner = SchedulerRunner(db, "original", lambda *args: None, on_protocol_failure=notifications.append)
    error, state = expired_receipt(runner=runner, with_state=True)
    try:
        runner._latch_protocol_failure(error, occurrence_state=state)
        runner._latch_protocol_failure(error)
        assert runner._occurrence_control_receipts == [error]
        assert runner.readiness_failure is None and notifications == []
    finally:
        try:
            await runner.stop()
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_original_receipt_identity_cannot_hide_added_foreign_scope(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "mixed-receipt-state.db"))
    notifications = []
    runner = SchedulerRunner(db, "original", lambda *args: None, on_protocol_failure=notifications.append)
    original, state = expired_receipt(runner=runner, with_state=True)
    foreign = expired_receipt()
    try:
        runner._latch_protocol_failure(original, occurrence_state=state)
        mixed = execution_terminal_error(original, foreign, retain_originals=True)
        assert mixed is original and len(execution_denial_scopes(mixed)) == 2
        runner._latch_protocol_failure(mixed)
        assert runner.readiness_failure is original and notifications == [original], (
            "an earlier receipt's identity hid a newly aggregated foreign denial scope"
        )
    finally:
        try:
            await runner.stop()
        except BaseException as error:
            assert error is original
        finally:
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
