"""Caller cancellation must join native PG epilogues, not arbitrary bodies."""

import asyncio

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot,
    execution_commit_outcome,
    execution_denial_scopes,
)
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("boundary", ["commit", "release"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("retire_scope", [False, True])
async def test_caller_cancel_joins_original_native_ack_without_poisoning_runtime(
    native_pg,
    monkeypatch,
    boundary,
    explicit,
    retire_scope,
):
    backend, observer, schema = native_pg
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original-native-ack"))
    parent = ExecutionCustody(ProcessRuntimeExecutionFence("healthy-native-ack-parent"))
    entered, release = asyncio.Event(), asyncio.Event()
    original_commit = asyncpg.transaction.Transaction.__aexit__
    original_release = asyncpg.pool.PoolAcquireContext.__aexit__
    acknowledgments = []

    async def held_commit(transaction, error_type, error, traceback):
        result = await original_commit(transaction, error_type, error, traceback)
        if boundary == "commit" and error is None and not transaction._nested:
            if retire_scope:
                scope.revoke("original participant retired during accepted ACK")
            acknowledgments.append(transaction)
            entered.set()
            await release.wait()
        return result

    async def held_release(checkout, error_type, error, traceback):
        result = await original_release(checkout, error_type, error, traceback)
        if boundary == "release" and error is None:
            if retire_scope:
                scope.revoke("original participant retired during accepted reset")
            acknowledgments.append(checkout)
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_commit)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", held_release)

    async def write():
        with bind_execution_custody_snapshot((parent, scope)):
            if explicit:
                async with backend.transaction():
                    await backend.execute(
                        "INSERT INTO effects VALUES (76, 'original native commit')"
                    )
            else:
                await backend.execute(
                    "INSERT INTO effects VALUES (76, 'original native commit')"
                )

    owner = create_owned_task(write())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        owner.cancel("original caller stopped observing")
        await asyncio.sleep(0)
        owner.cancel("repeated caller cancellation")
        await asyncio.sleep(0)
        assert not owner.done(), "caller cancellation abandoned the original native ACK"
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(owner), 3)
        assert isinstance(outcome.error, asyncio.CancelledError)
        assert outcome.error.args == ("original caller stopped observing",)
        assert execution_commit_outcome(outcome.error) == "committed"
        assert execution_denial_scopes(outcome.error) == (
            (scope,) if retire_scope else ()
        )
        receipts = vars(outcome.error)["_kestrel_acknowledged_commits"]
        assert len(receipts) == 1 and receipts[0].retired
        assert receipts[0].commit_owner is not owner
        assert receipts[0].retirement_owner is not owner
        parent.require_work()
        if retire_scope:
            with pytest.raises(ExecutionAuthorityError):
                scope.require_work()
        else:
            scope.require_work()
        assert len(acknowledgments) == 1
        assert (
            await observer.fetchval(f'SELECT value FROM "{schema}".effects WHERE id=76')
            == "original native commit"
        )
    finally:
        release.set()
        await await_owned_task(owner)


@pytest.mark.parametrize("boundary", ["commit", "release"])
async def test_acknowledged_local_retirement_preserves_healthy_parent(
    native_pg, monkeypatch, boundary
):
    backend, observer, schema = native_pg
    parent = ExecutionCustody(ProcessRuntimeExecutionFence("healthy-original-parent"))
    local = ExecutionCustody(ProcessRuntimeExecutionFence("original-child-admission"))
    native_commit = asyncpg.transaction.Transaction.__aexit__
    native_release = asyncpg.pool.PoolAcquireContext.__aexit__
    entered, release = asyncio.Event(), asyncio.Event()
    acknowledgments = []

    async def retire_original():
        local.revoke("original local admission retired after actual COMMIT")
        acknowledgments.append(True)
        entered.set()
        await release.wait()

    async def commit(transaction, error_type, error, traceback):
        result = await native_commit(transaction, error_type, error, traceback)
        if boundary == "commit" and error is None and not transaction._nested:
            await retire_original()
        return result

    async def reset(checkout, error_type, error, traceback):
        result = await native_release(checkout, error_type, error, traceback)
        if boundary == "release" and error is None:
            await retire_original()
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", commit)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", reset)

    async def write():
        with bind_execution_custody_snapshot((parent, local)):
            await backend.execute(
                "INSERT INTO effects VALUES (78, 'committed original child')"
            )

    owner = create_owned_task(write())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert not owner.done()
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(owner), 3)
        assert isinstance(outcome.error, BaseException)
        assert execution_commit_outcome(outcome.error) == "committed"
        assert execution_denial_scopes(outcome.error) == (local,)
        assert len(acknowledgments) == 1
        parent.require_work()
        with pytest.raises(ExecutionAuthorityError):
            local.require_work()
        monkeypatch.undo()
        with (
            pytest.raises(ExecutionAuthorityError),
            bind_execution_custody_snapshot((parent, local)),
        ):
            await backend.execute(
                "INSERT INTO effects VALUES (79, 'forbidden copied child')"
            )
        with bind_execution_custody_snapshot((parent,)):
            await backend.execute(
                "INSERT INTO effects VALUES (80, 'healthy independent parent')"
            )
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 2
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=79'
            )
            == 0
        )
    finally:
        release.set()
        await await_owned_task(owner)


@pytest.mark.parametrize("boundary", ["commit", "release"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_actual_native_ack_failure_still_fences_original_runtime(
    native_pg,
    monkeypatch,
    boundary,
    cancelled,
):
    backend, observer, schema = native_pg
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original-failed-ack"))
    parent = ExecutionCustody(
        ProcessRuntimeExecutionFence("original-failed-ack-parent")
    )
    native_commit = asyncpg.transaction.Transaction.__aexit__
    native_release = asyncpg.pool.PoolAcquireContext.__aexit__
    failure = (
        asyncio.CancelledError("actual native ACK failure")
        if cancelled
        else OSError("actual native ACK failure")
    )
    failures = []

    async def commit(transaction, error_type, error, traceback):
        result = await native_commit(transaction, error_type, error, traceback)
        if boundary == "commit" and error is None and not transaction._nested:
            failures.append(failure)
            raise failure
        return result

    async def reset(checkout, error_type, error, traceback):
        result = await native_release(checkout, error_type, error, traceback)
        if boundary == "release" and error is None:
            failures.append(failure)
            raise failure
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", commit)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", reset)

    async def write():
        with bind_execution_custody_snapshot((parent, scope)):
            await backend.execute(
                "INSERT INTO effects VALUES (81, 'real durable row before ACK fault')"
            )

    owner = create_owned_task(write())
    outcome = await asyncio.wait_for(await_owned_task(owner), 3)
    expected = "unknown" if boundary == "commit" else "committed"
    assert execution_commit_outcome(outcome.error) == expected
    assert execution_denial_scopes(outcome.error) is None
    assert failures == [failure]
    for original_scope in (parent, scope):
        with pytest.raises(ExecutionAuthorityError) as retained:
            original_scope.require_work()
        assert execution_commit_outcome(retained.value) == expected
    assert (
        await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=81')
        == 1
    )


async def test_actual_rollback_failure_retains_original_debt_not_commit(
    native_pg, monkeypatch
):
    from kestrel_sovereign._async_ownership import owned_task_error
    from kestrel_sovereign.execution_custody import execution_cleanup_debts

    backend, observer, schema = native_pg
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original-rollback-failure"))
    original = RuntimeError("original operation failure")
    rollback_fault = OSError("native rollback ACK failed")
    native_exit = asyncpg.transaction.Transaction.__aexit__

    async def rollback(transaction, error_type, error, traceback):
        result = await native_exit(transaction, error_type, error, traceback)
        if error is original and not transaction._nested:
            raise rollback_fault
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", rollback)

    async def write():
        with bind_execution_custody_snapshot((scope,)):
            async with backend.transaction():
                await backend.execute(
                    "INSERT INTO effects VALUES (82, 'rolled back original row')"
                )
                raise original

    owner = create_owned_task(write())
    outcome = await asyncio.wait_for(await_owned_task(owner), 3)
    assert execution_commit_outcome(outcome.error) is None
    assert execution_denial_scopes(outcome.error) is None
    debts = execution_cleanup_debts(outcome.error)
    assert len(debts) == 1
    assert owned_task_error(debts[0][1]) is rollback_fault
    with pytest.raises(ExecutionAuthorityError):
        scope.require_work()
    assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 0


async def test_operational_checkout_bounds_facts_but_settles_escaped_original(
    native_pg, monkeypatch
):
    backend, observer, schema = native_pg
    scope = ExecutionCustody(
        ProcessRuntimeExecutionFence("original-long-operational-owner")
    )
    native_exit = asyncpg.transaction.Transaction.__aexit__
    seen, owner = [], None

    async def original_commit(transaction, error_type, error, traceback):
        result = await native_exit(transaction, error_type, error, traceback)
        if error is None and not transaction._nested and not seen:
            seen.append(transaction)
            owner.cancel("original operational observer stopped")
            await asyncio.sleep(0)
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", original_commit)

    async def write():
        escaped = None
        with bind_execution_custody_snapshot((scope,)):
            async with backend.operational_session():
                state = backend._current_operational_lease().commit_state
                for index in range(65):
                    try:
                        await backend.execute(
                            "INSERT INTO effects VALUES (?, ?)",
                            (index, "original acknowledged row"),
                        )
                    except asyncio.CancelledError as error:
                        assert index == 0
                        escaped = error
                        assert execution_commit_outcome(error) == "committed"
                        assert execution_denial_scopes(error) is None
                    assert len(state.acknowledgments) == 1, (
                        "checkout retained every historical native task"
                    )
                    scope.require_work()
                receipt = vars(escaped)["_kestrel_acknowledged_commits"][0]
                assert receipt.retirement is state.retirement
                assert not receipt.retired
            assert receipt.retired
            assert execution_denial_scopes(escaped) == ()
            assert escaped.args == ("original operational observer stopped",)
        return escaped

    owner = create_owned_task(write())
    outcome = await asyncio.wait_for(await_owned_task(owner), 5)
    assert outcome.error is None
    assert execution_commit_outcome(outcome.result) == "committed"
    scope.require_work()
    assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 65


async def test_native_transaction_body_remains_cancellable_and_rolls_back(native_pg):
    backend, observer, schema = native_pg
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original-cancellable-body"))
    entered = asyncio.Event()

    async def write():
        with bind_execution_custody_snapshot((scope,)):
            async with backend.transaction():
                await backend.execute(
                    "INSERT INTO effects VALUES (77, 'uncommitted body')"
                )
                entered.set()
                await asyncio.Event().wait()

    owner = create_owned_task(write())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        owner.cancel()
        outcome = await asyncio.wait_for(await_owned_task(owner), 3)
        assert isinstance(outcome.error, asyncio.CancelledError)
        assert execution_commit_outcome(outcome.error) is None
        scope.require_work()
        assert await observer.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 0
    finally:
        if not owner.done():
            owner.cancel()
        await await_owned_task(owner)
