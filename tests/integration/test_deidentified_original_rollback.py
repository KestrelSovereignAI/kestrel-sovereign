"""Withdrawn real PG saves keep failed native rollback and privacy ownership.

Synthetic de-identification inputs, real storage/files/PG transactions. No
provider calls, shared/catalog rows or full Agent READY acceptance.
"""

import asyncio
import inspect

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import privacy_wrapper as module
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage, PrivacyViolationError
from tests.integration.native_memory_original_control import native_memory_owners as native_memory_owners
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_deidentified_storage_gate import _result
from tests.unit.test_cleanup_task_admission import _operation_name
from tests.shared.original_task_admission import assert_unknown_task_admission


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback_failure", [False, True])
@pytest.mark.parametrize("cancel_again", [False, True])
async def test_withdrawn_deidentified_save_keeps_actual_rollback_and_observers(
    native_memory_owners, monkeypatch, rollback_failure, cancel_again
):
    n = native_memory_owners
    wrapper = PrivacyEnforcingStorage(n.pg_storage, PrivacyMode.DEIDENTIFIED)
    writing, rollback_entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_exit = asyncpg.transaction.Transaction.__aexit__
    original_store = n.pg_storage.store_file
    original_join = module.await_owned_task
    failures, child_errors, withdrawals, owners = [], [], [], []
    rollback_fault = OSError("actual native PG rollback acknowledgement lost")

    async def store_then_hold(*args, **kwargs):
        result = await original_store(*args, **kwargs)
        writing.set()
        await asyncio.Event().wait()
        return result

    async def exit_then_ack(transaction, error_type, error, traceback):
        result = await original_exit(transaction, error_type, error, traceback)
        if isinstance(error, asyncio.CancelledError) and not transaction._nested:
            failures.append(error)
            rollback_entered.set()
            await release.wait()
            if rollback_failure:
                raise rollback_fault
        return result

    async def observed_join(task, *args, **kwargs):
        if task.get_name().startswith("deidentified-save:"):
            owners.append(task)
        result = await original_join(task, *args, **kwargs)
        if task in owners:
            child_errors.append(result.error)
            if result.cancellation is not None:
                withdrawals.append(result.cancellation)
        return result

    monkeypatch.setattr(n.pg_storage, "store_file", store_then_hold)
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", exit_then_ack)
    monkeypatch.setattr(module, "await_owned_task", observed_join)
    operation = create_owned_task(wrapper.store_deidentified_records(_result()), retain_originals=True)
    try:
        await asyncio.wait_for(writing.wait(), 5)
        operation.cancel("actual save observer withdrawal")
        await asyncio.wait_for(rollback_entered.wait(), 5)
        if cancel_again:
            operation.cancel("actual repeated save observer withdrawal")
            await asyncio.sleep(0)
        assert not operation.done()
        assert wrapper._active_deidentified_save_leases == 1
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(operation), 5)
        assert outcome.error is not None and owners and all(owner.done() for owner in owners)
        assert await n.observer.fetchval(f'SELECT COUNT(*) FROM "{n.schema}".files') == 0
        chain = tuple(_explicit_control_chain(outcome.error))
        assert withdrawals and all(error in chain for error in withdrawals)
        if rollback_failure:
            assert child_errors and all(error in chain for error in child_errors), (
                "withdrawn save discarded its actual failed rollback carrier"
            )
            assert rollback_fault in chain and failures[0] in chain
            assert is_execution_control_error(outcome.error)
            assert execution_cleanup_debts(outcome.error)
            assert wrapper._active_deidentified_save_leases == 1, (
                "failed rollback released the original privacy transition fence"
            )
            with pytest.raises(PrivacyViolationError, match="de-identified record save"):
                wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
        else:
            assert isinstance(outcome.error, asyncio.CancelledError)
            assert not execution_cleanup_debts(outcome.error)
            assert wrapper._active_deidentified_save_leases == 0
            wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    finally:
        release.set()
        if not operation.done():
            operation.cancel()
        await await_owned_task(operation)


@pytest.mark.asyncio
async def test_refused_deidentified_save_does_not_acquire_native_transaction(
    native_memory_owners
):
    n = native_memory_owners
    wrapper = PrivacyEnforcingStorage(n.pg_storage, PrivacyMode.DEIDENTIFIED)
    loop, factory = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refusal, refused = RuntimeError("actual deidentified writer Task refused"), []

    def refuse(loop, coroutine, **kwargs):
        if _operation_name(coroutine).endswith("._commit_deidentified_documents"):
            refused.append(coroutine)
            raise refusal
        return factory(loop, coroutine, **kwargs) if factory else asyncio.Task(coroutine, loop=loop, **kwargs)

    try:
        loop.set_task_factory(refuse)
        with pytest.raises(RuntimeError) as caught:
            await wrapper.store_deidentified_records(_result())
        loop.set_task_factory(factory)
        assert refused
        assert_unknown_task_admission(caught.value, refusal)
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused)
        assert wrapper._active_deidentified_save_leases == 0
        # No native write ran, but closing the unstarted payload cannot certify
        # that this opaque factory accepted no other native carrier.
        assert await n.observer.fetchval(f'SELECT COUNT(*) FROM "{n.schema}".files') == 0
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    finally:
        loop.set_task_factory(factory)
        for coroutine in refused:
            coroutine.close()
