"""Observe ACTUAL native PG ACK/reset owners, never fabricated receipts."""

import asyncio
import gc
import weakref
from contextvars import ContextVar

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_custody_snapshot,
    execution_cleanup_debts,
    execution_commit_outcome,
    execution_denial_scopes,
    is_execution_control_error,
)
from kestrel_sovereign.native_retirement import (
    NativeExecutionObserver,
    bind_native_execution_observer,
)
from kestrel_sovereign.storage.db import postgres
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg

native_pg = _native_pg
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("wait_kind", ["checkout", "operational_lock"])
@pytest.mark.parametrize("cutoff", ["custody", "observer"])
async def test_transaction_revalidates_after_actual_native_entry_wait(
    native_pg, monkeypatch, wait_kind, cutoff,
):
    backend, control, schema = native_pg
    observer = NativeExecutionObserver()
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original waiting transaction"))
    marked = ContextVar("original waiting native entry", default=False)
    entered, starts, original_enter = asyncio.Event(), [], asyncpg.transaction.Transaction.__aenter__
    original_acquire = asyncpg.pool.PoolAcquireContext.__aenter__
    candidate = None
    reserves = []

    async def native_enter(transaction):
        if marked.get():
            starts.append(transaction)
        return await original_enter(transaction)

    async def native_acquire(checkout):
        if marked.get():
            entered.set()
        return await original_acquire(checkout)

    async def original_work():
        token = marked.set(True)
        try:
            with bind_execution_custody_snapshot((scope,)):
                if wait_kind == "operational_lock":
                    entered.set()
                async with backend.transaction():
                    await backend.execute("INSERT INTO effects VALUES (1001, 'forbidden late BEGIN')")
        finally:
            marked.reset(token)

    async def retire_waiter():
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.sleep(0)
        assert not candidate.done()
        if cutoff == "custody":
            scope.revoke("retired during original native entry wait")
        else:
            observer.close_admission()

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aenter__", native_enter)
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aenter__", native_acquire)
    try:
        if wait_kind == "checkout":
            reserves = [await backend._pool.acquire(), await backend._pool.acquire()]
            with bind_native_execution_observer(observer):
                candidate = create_owned_task(original_work())
                await retire_waiter()
                for reserved in reserves:
                    await backend._pool.release(reserved)
                reserves.clear()
                outcome = await await_owned_task(candidate)
        else:
            try:
                with bind_native_execution_observer(observer):
                    async with backend.operational_session():
                        lease = backend._current_operational_lease()
                        await lease.lock.acquire()
                        try:
                            candidate = create_owned_task(original_work())
                            await retire_waiter()
                            assert lease.lock._waiters, "candidate did not reach the actual operational mutex"
                        finally:
                            lease.lock.release()
                        outcome = await await_owned_task(candidate)
            except BaseException as error:
                # The original checkout may preserve the same retired scope
                # at reset. It is not a new transaction admission or a waiver.
                assert is_execution_control_error(error)
        assert outcome.error is not None and is_execution_control_error(outcome.error)
        assert not starts, "retirement cutoff occurred before native BEGIN but BEGIN was still submitted"
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 0
        # Positive control: the physical backend remains usable for an actual
        # independent admission, not a permanently broken test connection.
        async with backend.transaction():
            await backend.execute("INSERT INTO effects VALUES (1002, 'fresh native admission')")
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".effects') == 1
    finally:
        for reserved in reserves:
            await backend._pool.release(reserved)
        if candidate is not None:
            await await_owned_task(candidate)

async def test_inherited_operational_transaction_roots_actual_child_body(native_pg):
    backend, _control, _schema = native_pg
    observer = NativeExecutionObserver()
    entered = asyncio.Event()
    owner = None
    retained = None

    async def child_body():
        async with backend.transaction():
            await backend.execute("INSERT INTO effects VALUES (915, 'inherited native transaction body')")
            entered.set()
            await asyncio.get_running_loop().create_future()

    with bind_native_execution_observer(observer):
        async with backend.operational_session():
            record = next(iter(observer._resources))
            owner = create_owned_task(child_body())
            try:
                await asyncio.wait_for(entered.wait(), 3)
                # Prove actual native body is rooted BEFORE dropping the safe
                # test reference; a failing proof still cancels/reaps it.
                assert owner in getattr(record, "body_owners", ()), "original child transaction body is not physically rooted"
                retained = weakref.ref(owner)
                owner = None
                gc.collect()
                assert retained() in record.body_owners
            finally:
                original = owner if owner is not None else retained()
                original.cancel()
                await await_owned_task(original)
            assert not record.body_owners
    assert not observer._resources


@pytest.mark.parametrize("lease_kind", ["operational", "nested_transaction"])
async def test_late_observer_is_refused_before_native_transaction_entry(native_pg, monkeypatch, lease_kind):
    backend, _control, _schema = native_pg
    observer = NativeExecutionObserver()
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original unobserved parent transaction"))
    marked = ContextVar("original late observed transaction entry", default=False)
    entries, errors = [], []
    native_enter = asyncpg.transaction.Transaction.__aenter__

    async def enter(transaction):
        if marked.get():
            entries.append(transaction)
        return await native_enter(transaction)

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aenter__", enter)
    with bind_execution_custody_snapshot((scope,)):
        parent = backend.operational_session() if lease_kind == "operational" else backend.transaction()
        async with parent:
            token = marked.set(True)
            try:
                with bind_native_execution_observer(observer):
                    async with backend.transaction():
                        pass
            except BaseException as error:
                errors.append(error)
            finally:
                marked.reset(token)
            scope.require_work()
    assert errors and is_execution_control_error(errors[0]), "late observer admitted an unobserved native transaction"
    assert not entries, "refusal followed native BEGIN/savepoint admission"
    assert not observer._resources


@pytest.mark.parametrize("cancel_observer", [False, True])
async def test_join_tracks_real_commit_then_reset_not_arbitrary_provider_tail(
    native_pg, monkeypatch, cancel_observer,
):
    backend, control, schema = native_pg
    observer = NativeExecutionObserver()
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original native observed caller"))
    ack_entered, ack_release = asyncio.Event(), asyncio.Event()
    reset_entered, reset_release = asyncio.Event(), asyncio.Event()
    tail_release = asyncio.Event()
    native_exit, native_reset = asyncpg.transaction.Transaction.__aexit__, asyncpg.Connection.reset
    interrupted, hidden = [], []

    async def hold_ack(transaction, kind, error, traceback):
        result = await native_exit(transaction, kind, error, traceback)
        if kind is None:
            ack_entered.set()
            try:
                await ack_release.wait()
            except asyncio.CancelledError:
                interrupted.append("ack")
                raise
        return result

    async def hold_reset(connection, *args, **kwargs):
        await native_reset(connection, *args, **kwargs)
        reset_entered.set()
        try:
            await reset_release.wait()
        except asyncio.CancelledError:
            interrupted.append("reset")
            raise

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", hold_ack)
    monkeypatch.setattr(asyncpg.Connection, "reset", hold_reset)

    async def provider():
        with bind_execution_custody_snapshot((scope,)), bind_native_execution_observer(observer):
            try:
                await backend.execute("INSERT INTO effects VALUES (910, 'actual accepted native commit')")
            except BaseException as error:
                hidden.append(error)
            await tail_release.wait()

    producer, join = create_owned_task(provider()), None
    try:
        await asyncio.wait_for(ack_entered.wait(), 3)
        scope.revoke("original observed caller retired")
        observer.close_admission()
        join = create_owned_task(observer.join_accepted())
        if cancel_observer:
            await asyncio.sleep(0)
            join.cancel("independent native observer cancellation")
        await asyncio.sleep(.03)
        assert not join.done(), "actual COMMIT ACK was not in the retirement inventory"
        ack_release.set()
        await asyncio.wait_for(reset_entered.wait(), 3)
        await asyncio.sleep(.03)
        assert not join.done(), "an empty ACK task snapshot erased the original reset owner"
        reset_release.set()
        completed = await asyncio.wait_for(await_owned_task(join), 3)
        assert completed.error is None
        outcome = completed.result
        assert execution_commit_outcome(outcome.error) == "committed"
        assert execution_denial_scopes(outcome.error) == (scope,)
        assert not execution_cleanup_debts(outcome.error) and not interrupted
        assert (outcome.cancellation is not None) is cancel_observer
        if cancel_observer:
            assert "independent native observer" in str(outcome.cancellation)
        assert hidden and not producer.done(), "observer waited for arbitrary provider tail"
        assert not observer._resources
        assert await control.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
    finally:
        ack_release.set()
        reset_release.set()
        tail_release.set()
        await await_owned_task(producer)
        if join is not None:
            await await_owned_task(join)


async def test_open_body_is_retained_debt_not_an_unbounded_native_join(native_pg):
    backend, control, schema = native_pg
    observer = NativeExecutionObserver()
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original held body"))
    entered, release = asyncio.Event(), asyncio.Event()

    async def body():
        with bind_execution_custody_snapshot((scope,)), bind_native_execution_observer(observer):
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (911, 'uncommitted body')")
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()

    producer = create_owned_task(body())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        scope.revoke("original body owner retired")
        producer.cancel()
        outcome = await asyncio.wait_for(observer.join_accepted(), 1)
        debts = execution_cleanup_debts(outcome.error)
        assert len(debts) == 1 and debts[0][1] is None
        record = debts[0][0]
        assert type(record.resource) is asyncpg.pool.PoolAcquireContext
        assert record.native_connection is not None and not record.native_connection.is_closed()
        assert record in observer._resources and not producer.done()
        assert await control.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0
    finally:
        release.set()
        await await_owned_task(producer)
    assert not observer._resources


async def test_late_observer_on_inherited_operational_lease_cannot_miss_accepted_commit(native_pg, monkeypatch):
    backend, control, schema = native_pg
    observer = NativeExecutionObserver()
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original inherited operational lease"))
    entered, release = asyncio.Event(), asyncio.Event()
    original_exit = asyncpg.transaction.Transaction.__aexit__

    async def held_exit(transaction, kind, error, traceback):
        result = await original_exit(transaction, kind, error, traceback)
        if kind is None:
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", held_exit)

    async def callback():
        with bind_native_execution_observer(observer):
            await backend.execute("INSERT INTO effects VALUES (915, 'inherited original native lease')")

    owner = signal = join = None
    with bind_execution_custody_snapshot((scope,)):
        async with backend.operational_session():
            owner, signal = create_owned_task(callback()), create_owned_task(entered.wait())
            try:
                done, _ = await asyncio.wait({owner, signal}, timeout=3, return_when=asyncio.FIRST_COMPLETED)
                assert done
                if owner.done():
                    outcome = await await_owned_task(owner)
                    assert is_execution_control_error(outcome.error)
                    assert await control.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0
                    scope.require_work()  # Refusal does not revoke the borrowed host.
                else:
                    join = create_owned_task(observer.join_accepted())
                    await asyncio.sleep(.03)
                    assert not join.done(), "late observer missed the inherited lease's actual accepted COMMIT"
            finally:
                release.set()
                signal.cancel()
                await await_owned_task(signal)
                await await_owned_task(owner)
                if join is not None:
                    await await_owned_task(join)


@pytest.mark.parametrize("raw_transaction", [False, True])
async def test_script_observation_cannot_hide_raw_transaction_control(native_pg, raw_transaction):
    backend, control, schema = native_pg
    observer = NativeExecutionObserver()
    statement = "INSERT INTO effects VALUES (916, 'observed native script');"
    script = f"BEGIN; {statement} COMMIT;" if raw_transaction else statement

    async def execute():
        with bind_native_execution_observer(observer):
            await backend.execute_script(script)

    outcome = await await_owned_task(create_owned_task(execute()))
    assert (outcome.error is not None) is raw_transaction
    assert await control.fetchval(f'SELECT count(*) FROM "{schema}".effects') == (0 if raw_transaction else 1)


async def test_native_resource_roots_its_actual_body_task_until_physical_retirement(native_pg):
    backend, _control, _schema = native_pg
    observer = NativeExecutionObserver()
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original retained native body task"))
    entered = asyncio.Event()

    async def body():
        with bind_execution_custody_snapshot((scope,)), bind_native_execution_observer(observer):
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (917, 'original rooted body')")
                entered.set()
                await asyncio.Future()

    owner = create_owned_task(body())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        record, = observer._resources
        # Assert the strong link BEFORE dropping the test's link, so a
        # failing guard still has a safe original task to cancel and reap.
        assert getattr(record, "body_owner", None) is owner
        retained = weakref.ref(owner)
        owner = None
        gc.collect()
        assert retained() is record.body_owner
    finally:
        original = owner if owner is not None else retained()
        original.cancel()
        await await_owned_task(original)
    assert not observer._resources


@pytest.mark.parametrize("refusal", ["transaction", "reset", "reset_failure"])
async def test_provider_cannot_hide_original_native_close_failure(native_pg, monkeypatch, refusal):
    backend, _control, _schema = native_pg
    observer = NativeExecutionObserver()
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original failed native phase"))
    release, entered = asyncio.Event(), asyncio.Event()
    hidden = []
    original_create, native_reset = postgres.create_owned_task, asyncpg.Connection.reset

    def refuse(operation, **kwargs):
        ending = (":postgres-original-transaction-exit" if refusal == "transaction"
                  else ":postgres-original-pool-reset")
        if kwargs.get("name", "").endswith(ending):
            operation.close()
            raise RuntimeError("original native epilogue factory refused")
        return original_create(operation, **kwargs)

    async def fail_reset(connection, *args, **kwargs):
        await native_reset(connection, *args, **kwargs)
        raise OSError("original native reset failed")

    if refusal == "reset_failure":
        monkeypatch.setattr(asyncpg.Connection, "reset", fail_reset)
    else:
        monkeypatch.setattr(postgres, "create_owned_task", refuse)

    async def provider():
        with bind_execution_custody_snapshot((scope,)), bind_native_execution_observer(observer):
            try:
                await backend.execute("INSERT INTO effects VALUES (912, 'original native fault')")
            except BaseException as error:
                hidden.append(error)
            entered.set()
            await release.wait()

    producer = create_owned_task(provider())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        outcome = await asyncio.wait_for(observer.join_accepted(), 1)
        assert outcome.error is not None and hidden and not producer.done()
        assert execution_cleanup_debts(outcome.error)
        assert not observer._resources
    finally:
        release.set()
        await await_owned_task(producer)


async def test_closed_observer_cannot_admit_new_native_resource(native_pg):
    backend, control, schema = native_pg
    observer = NativeExecutionObserver()
    observer.close_admission()
    with bind_native_execution_observer(observer):
        with pytest.raises(BaseException):
            await backend.execute("INSERT INTO effects VALUES (913, 'refused new checkout')")
    assert not observer._resources
    assert await control.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0


@pytest.mark.parametrize("lost_ack", [False, True])
async def test_open_operational_body_preserves_commit_or_hidden_unknown_with_resource_debt(
    native_pg, monkeypatch, lost_ack,
):
    backend, control, schema = native_pg
    observer = NativeExecutionObserver()
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original operational-session body"))
    entered, release = asyncio.Event(), asyncio.Event()
    original_exit = asyncpg.transaction.Transaction.__aexit__

    async def lose_ack(transaction, kind, error, traceback):
        result = await original_exit(transaction, kind, error, traceback)
        if kind is None:
            raise OSError("actual COMMIT happened but its native ACK was lost")
        return result

    if lost_ack:
        monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", lose_ack)

    async def provider():
        with bind_execution_custody_snapshot((scope,)), bind_native_execution_observer(observer):
            async with backend.operational_session():
                try:
                    await backend.execute("INSERT INTO effects VALUES (914, 'operational native commit')")
                except BaseException:
                    pass  # Deliberate host suppression; Core inventory must not lose it.
                entered.set()
                await release.wait()

    producer = create_owned_task(provider())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        scope.revoke("original operational body retired")
        outcome = await asyncio.wait_for(observer.join_accepted(), 1)
        assert not producer.done() and execution_cleanup_debts(outcome.error)
        assert execution_commit_outcome(outcome.error) == ("unknown" if lost_ack else "committed")
        assert await control.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
    finally:
        release.set()
        await await_owned_task(producer)
    assert not observer._resources
