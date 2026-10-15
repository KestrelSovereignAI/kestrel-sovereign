"""Caller cancellation must join native PG epilogues, not arbitrary bodies."""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot,
    execution_cleanup_debts,
    execution_commit_outcome,
    execution_denial_scopes,
)
from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("abort_fails", [False, True])
async def test_asyncpg_internal_reset_task_refusal_keeps_actual_debt(
    native_pg, monkeypatch, abort_fails
):
    backend, observer, schema = native_pg
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    original_enter = asyncpg.pool.PoolAcquireContext.__aenter__
    original_abort = asyncpg.Connection.terminate
    refusal = RuntimeError("asyncpg internal reset task refused")
    abort_error = OSError("original connection termination refused")
    checkouts = []
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("inner-factory-original"))

    async def enter(checkout):
        connection = await original_enter(checkout)
        checkouts.append((checkout, connection._con))
        return connection

    def abort(connection):
        if abort_fails:
            raise abort_error
        original_abort(connection)

    def factory(loop, coro, **kwargs):
        if getattr(coro, "__qualname__", "").endswith("PoolConnectionHolder.release"):
            # The refusing factory owns its unaccepted coroutine. No fake
            # release task or successful pool reset exists.
            coro.close()
            raise refusal
        if original_factory is not None:
            return original_factory(loop, coro, **kwargs)
        return asyncio.Task(coro, loop=loop, **kwargs)

    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aenter__", enter)
    monkeypatch.setattr(asyncpg.Connection, "terminate", abort)

    async def write():
        with bind_execution_custody_snapshot((scope,)):
            await backend.execute(
                "INSERT INTO effects VALUES (789, 'before refused reset')"
            )

    owner = create_owned_task(write())
    loop.set_task_factory(factory)
    try:
        outcome = await await_owned_task(owner)
        assert execution_commit_outcome(outcome.error) == "committed"
        assert execution_denial_scopes(outcome.error) is None
        ((resource, closer),) = execution_cleanup_debts(outcome.error)
        assert resource.context_manager is checkouts[0][0]
        assert resource.native_connection is checkouts[0][1]
        assert closer.done() and closer.exception() is refusal
        with pytest.raises(ExecutionAuthorityError):
            scope.require_work()
        assert (
            await observer.fetchval(
                f'SELECT count(*) FROM "{schema}".effects WHERE id=789'
            )
            == 1
        )
        if abort_fails:
            assert abort_error is refusal.__cause__
            assert not checkouts[0][1].is_closed()
        else:
            assert backend._pool.get_idle_size() == backend._pool.get_size()
    finally:
        loop.set_task_factory(original_factory)
        monkeypatch.setattr(asyncpg.Connection, "terminate", original_abort)
        for _, connection in checkouts:
            if not connection.is_closed():
                connection.terminate()


@pytest.mark.parametrize("surface", ["legacy", "callback", "cleanup"])
async def test_unfenced_aborted_transaction_never_mints_known_commit(
    native_pg, surface
):
    backend, observer, schema = native_pg
    callbacks = []
    if surface == "cleanup":
        with pytest.raises(asyncpg.InFailedSQLTransactionError) as caught:
            async with backend._tracked_pool_checkout(
                backend._pool, admit_work=False
            ) as (connection, state):
                async with backend._tracked_native_transaction(connection, (), state):
                    await connection.execute(
                        "INSERT INTO effects VALUES (788, 'rolled back')"
                    )
                    with pytest.raises(asyncpg.DivisionByZeroError):
                        await connection.execute("SELECT 1 / 0")
        assert execution_commit_outcome(caught.value) is None
    else:

        async def write():
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (788, 'rolled back')")
                if surface == "callback":
                    backend.call_after_commit(lambda: callbacks.append("must not run"))
                with pytest.raises(Exception, match="division by zero"):
                    await backend.fetch_val("SELECT 1 / 0")

        outcome = await await_owned_task(create_owned_task(write()))
        assert (outcome.error is None) == (surface == "legacy")
        assert execution_commit_outcome(outcome.error) is None
    assert callbacks == []
    assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0


async def test_rolled_back_savepoint_does_not_poison_proven_outer_commit(native_pg):
    backend, observer, schema = native_pg
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("savepoint-original"))
    with bind_execution_custody_snapshot((scope,)):
        async with backend.transaction():
            await backend.execute("INSERT INTO effects VALUES (787, 'outer committed')")
            with pytest.raises(Exception, match="division by zero"):
                async with backend.transaction():
                    await backend.fetch_val("SELECT 1 / 0")
    assert (
        await observer.fetchval(f'SELECT value FROM "{schema}".effects WHERE id=787')
        == "outer committed"
    )
    scope.require_work()


async def test_acquire_wrapper_keeps_original_context_tokens(native_pg):
    backend, observer, schema = native_pg
    pool = backend._pool
    marker = ContextVar("original-pool-wrapper", default=None)
    caller_token = marker.set("before wrapper")
    body_observations = []

    @asynccontextmanager
    async def wrapper(native):
        token = marker.set("original acquired context")
        try:
            async with native as connection:
                yield connection
        finally:
            marker.reset(token)

    backend._pool = PostgresPoolAdapter(pool, wrapper)
    try:
        scope = ExecutionCustody(ProcessRuntimeExecutionFence("context-pool"))
        with bind_execution_custody_snapshot((scope,)):
            async with backend.transaction():
                body_observations.append(marker.get())
                body_token = marker.set("body context")
                await backend.execute(
                    "INSERT INTO effects VALUES (790, 'original wrapper')"
                )
                marker.reset(body_token)
        assert marker.get() == "before wrapper"
        assert body_observations == ["original acquired context"]
        assert (
            await observer.fetchval(
                f'SELECT value FROM "{schema}".effects WHERE id=790'
            )
            == "original wrapper"
        )
        scope.require_work()
    finally:
        backend._pool = pool
        marker.reset(caller_token)


@pytest.mark.parametrize(
    "admission",
    ["ordinary", "before_checkout", "operational", "operational_transaction"],
)
async def test_opaque_wrapper_preserves_context_but_cannot_claim_native_custody(
    native_pg, admission
):
    backend, observer, schema = native_pg
    pool = backend._pool
    marker = ContextVar("opaque-caller-context", default=None)
    entries = []

    class OpaquePool:
        def __getattr__(self, name):
            return getattr(pool, name)

        @asynccontextmanager
        async def acquire(self):
            entries.append(asyncio.current_task())
            token = marker.set("opaque acquired")
            try:
                async with pool.acquire() as connection:
                    yield connection
            finally:
                marker.reset(token)

    backend._pool = OpaquePool()
    scope = ExecutionCustody(
        ProcessRuntimeExecutionFence("opaque-not-a-native-capability")
    )
    try:
        if admission == "ordinary":
            async with backend.transaction():
                assert marker.get() == "opaque acquired"
                await backend.execute(
                    "INSERT INTO effects VALUES (792, 'legacy context')"
                )
            assert entries == [asyncio.current_task()]
            assert (
                await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
            )
        else:
            with pytest.raises(Exception, match="opaque pool checkout"):
                if admission == "before_checkout":
                    with bind_execution_custody_snapshot((scope,)):
                        await backend.execute(
                            "INSERT INTO effects VALUES (792, 'refused')"
                        )
                else:
                    async with backend.operational_session():
                        with bind_execution_custody_snapshot((scope,)):
                            if admission == "operational_transaction":
                                async with backend.transaction():
                                    pytest.fail(
                                        "opaque checkout admitted a fenced body"
                                    )
                            else:
                                await backend.execute(
                                    "INSERT INTO effects VALUES (792, 'refused')"
                                )
            assert len(entries) == (0 if admission == "before_checkout" else 1)
            assert (
                await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0
            )
        assert marker.get() is None
        assert pool.get_idle_size() == pool.get_size()
    finally:
        backend._pool = pool


@pytest.mark.parametrize("suppress", [False, True])
async def test_adapter_cannot_abandon_or_suppress_original_native_lease(
    native_pg, suppress
):
    backend, observer, schema = native_pg
    pool = backend._pool
    body_error = ExecutionAuthorityError("original adapter body denial")
    marker = ContextVar("adapter-unwind", default=None)

    @asynccontextmanager
    async def wrapper(native):
        token = marker.set("original wrapper")
        try:
            if suppress:
                try:
                    async with native as connection:
                        yield connection
                except ExecutionAuthorityError as error:
                    assert error is body_error
            else:
                connection = await native.__aenter__()
                try:
                    yield connection
                finally:
                    # A broken wrapper omits its managed native exit. Core
                    # still owns the registered lease, not this projection.
                    pass
        finally:
            marker.reset(token)

    backend._pool = PostgresPoolAdapter(pool, wrapper)
    try:
        with pytest.raises(ExecutionAuthorityError) as caught:
            async with backend._tracked_pool_checkout(backend._pool, admit_work=True):
                raise body_error
        assert caught.value is body_error
        assert marker.get() is None
        assert pool.get_idle_size() == pool.get_size()
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0
    finally:
        backend._pool = pool


@pytest.mark.parametrize("boundary", ["transaction-exit", "pool-reset"])
@pytest.mark.parametrize("body_fails", [False, "ordinary", "cancel", "authority"])
async def test_task_factory_refusal_retains_original_resource_and_body(
    native_pg,
    boundary,
    body_fails,
):
    backend, observer, schema = native_pg
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("factory-original"))
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    refusal = RuntimeError("native epilogue task factory refused")
    body_error = (
        {"cancel": asyncio.CancelledError, "authority": ExecutionAuthorityError}.get(
            body_fails, ValueError
        )
    )("original transaction body failed")
    refusals = []

    def reject_native_exit(loop, coro, **kwargs):
        operation = getattr(coro, "cr_frame", None)
        operation = (
            operation.f_locals.get("operation") if operation is not None else None
        )
        name = getattr(operation, "__qualname__", "")
        target = (
            "Transaction.__aexit__"
            if boundary == "transaction-exit"
            else "PoolAcquireContext.__aexit__"
        )
        if name.endswith(target):
            refusals.append(name)
            raise refusal
        if original_factory is not None:
            return original_factory(loop, coro, **kwargs)
        return asyncio.Task(coro, loop=loop, **kwargs)

    async def write():
        with bind_execution_custody_snapshot((scope,)):
            async with backend.transaction():
                await backend.execute(
                    "INSERT INTO effects VALUES (791, 'factory original')"
                )
                if body_fails:
                    raise body_error

    owner = create_owned_task(write())
    loop.set_task_factory(reject_native_exit)
    try:
        outcome = await await_owned_task(owner)
    finally:
        loop.set_task_factory(original_factory)
    assert refusals
    assert execution_cleanup_debts(outcome.error), (
        "refused native ownership lost its resource"
    )
    assert execution_denial_scopes(outcome.error) is None
    with pytest.raises(ExecutionAuthorityError):
        scope.require_work()
    # Commit was never sent at the refused transaction boundary. At refused
    # pool reset it did succeed, unless the original body required rollback.
    expected_row = boundary == "pool-reset" and not body_fails
    assert await observer.fetchval(
        f'SELECT count(*) FROM "{schema}".effects WHERE id=791'
    ) == int(expected_row)
    assert backend._pool.get_idle_size() == backend._pool.get_size()
    if body_fails:

        def explicit_chain(error):
            if error is None:
                return
            yield error
            yield from explicit_chain(error.__cause__)
            if isinstance(error, BaseExceptionGroup):
                for child in error.exceptions:
                    yield from explicit_chain(child)

        assert body_error in tuple(explicit_chain(outcome.error)), (
            "original body exception was replaced"
        )


@pytest.mark.parametrize("cancel_observer", [False, True])
async def test_swallowed_statement_error_is_not_a_successful_commit_fact(
    native_pg,
    monkeypatch,
    cancel_observer,
):
    backend, observer, schema = native_pg
    parent = ExecutionCustody(ProcessRuntimeExecutionFence("aborted-original"))
    exited, release = asyncio.Event(), asyncio.Event()
    original_exit = asyncpg.transaction.Transaction.__aexit__

    async def held_exit(transaction, error_type, error, traceback):
        result = await original_exit(transaction, error_type, error, traceback)
        if not transaction._nested:
            exited.set()
            await release.wait()
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_exit)

    async def write():
        with bind_execution_custody_snapshot((parent,)):
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (79, 'rolled back')")
                # PostgreSQL has now aborted the ORIGINAL transaction. asyncpg
                # COMMIT of this state returns successfully but means ROLLBACK.
                with pytest.raises(Exception, match="division by zero"):
                    await backend.fetch_val("SELECT 1 / 0")

    owner = create_owned_task(write())
    try:
        await asyncio.wait_for(exited.wait(), 3)
        if cancel_observer:
            owner.cancel("observer cancelled during aborted exit")
            await asyncio.sleep(0)
        release.set()
        outcome = await asyncio.wait_for(await_owned_task(owner), 3)
        assert outcome.error is not None, "aborted transaction was reported successful"
        assert execution_commit_outcome(outcome.error) is None
        assert not vars(outcome.error).get("_kestrel_acknowledged_commits")
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0
        parent.require_work()
    finally:
        release.set()
        await await_owned_task(owner)


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
