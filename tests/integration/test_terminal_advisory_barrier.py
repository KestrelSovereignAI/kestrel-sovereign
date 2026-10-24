"""Original-denial-safe physical barriers; no business cleanup is authorized.

Actual isolated PostgreSQL sessions, not a provider or asset erasure claim.
"""

import asyncio
import contextvars
import os
from uuid import uuid4

import asyncpg
import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    AdvisoryExecutionFence,
    ExecutionAuthorityError,
    _explicit_control_chain,
    bind_execution_custody,
    current_execution_custody,
    execution_commit_outcome,
    is_execution_control_error,
    require_execution_work,
)
from kestrel_sovereign.storage.db.interface import QueryError
from kestrel_sovereign.storage.db.postgres import (
    PostgresBackend,
    TerminalAdvisoryBarrier,
    TerminalAdvisoryLease,
)
from tests.integration.test_execution_authority_postgres import (
    GenerationFence,
    native_pg,
)
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.utils.original_failure import OriginalFailure

__all__ = ["native_pg"]

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _key():
    return 3739, uuid4().int % (2**31)


@pytest.mark.parametrize("control", ["none", "denied", "unknown"])
@pytest.mark.parametrize("eager", [False, True])
async def test_terminal_barrier_retains_original_denial_and_cannot_admit_work(
    native_pg, control, eager
):
    backend, observer, _ = native_pg
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("eager task factory requires Python 3.12+")
    key, captured, reached = _key(), [], []
    loop, prior = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )

    async def body():
        async with backend.advisory_locks(TerminalAdvisoryBarrier((key,))) as lease:
            captured.append(
                (lease, contextvars.copy_context(), current_execution_custody())
            )
            assert isinstance(lease, TerminalAdvisoryLease)
            lease.require_physical()
            reached.append(True)
            assert await observer.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=$1 AND locktype='advisory' AND granted AND mode='ExclusiveLock')",
                lease.backend_pid,
            )
            for work in (
                require_execution_work,
                lease.require_live,
                AdvisoryExecutionFence(lease).require_work,
            ):
                with pytest.raises(ExecutionAuthorityError):
                    work()
            with pytest.raises(ExecutionAuthorityError):
                await AdvisoryExecutionFence(lease).lock_and_validate(observer)
            with pytest.raises(ExecutionAuthorityError):
                await backend.execute(
                    "INSERT INTO effects VALUES (1, 'terminal work escape')"
                )
            child = create_owned_task(_child_denies_work(lease))
            assert (await await_owned_task(child)).error is None

    try:
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        if control == "none":
            outcome = await await_owned_task(
                create_owned_task(body(), retain_originals=True)
            )
        else:
            with bind_execution_custody(GenerationFence()) as original:
                if control == "unknown":
                    original.preserve_commit_uncertainty("unknown")
                else:
                    original.revoke("original host scope denied")
                outcome = await await_owned_task(
                    create_owned_task(body(), retain_originals=True)
                )
                with pytest.raises(ExecutionAuthorityError):
                    original.require_work()
        assert reached == [True], (
            "retained original denial refused the fixed physical barrier"
        )
        assert execution_commit_outcome(outcome.error) == (
            "unknown" if control == "unknown" else None
        )
        assert (outcome.error is None) is (control == "none")
        lease, copied, scopes = captured[0]
        with pytest.raises(ExecutionAuthorityError, match="retired"):
            lease.require_physical()
        assert all(
            scope._denial is not None
            for scope in scopes
            if isinstance(scope.fence, TerminalAdvisoryBarrier)
        )
        escaped = asyncio.create_task(
            _child_denies_work(lease, retired=True), context=copied
        )
        assert (await await_owned_task(escaped)).error is None
        assert await backend.fetch_val("SELECT count(*) FROM effects") == 0
    finally:
        loop.set_task_factory(prior)


async def _child_denies_work(lease, retired=False):
    with pytest.raises(ExecutionAuthorityError):
        require_execution_work()
    with pytest.raises(ExecutionAuthorityError):
        lease.require_live()
    if retired:
        with pytest.raises(ExecutionAuthorityError, match="retired"):
            lease.require_physical()
    else:
        lease.require_physical()


async def test_terminal_barrier_exclusive_and_connected_backend_only(native_pg):
    backend, _, _ = native_pg
    request = TerminalAdvisoryBarrier((_key(),))
    with pytest.raises(ValueError, match="exclusive"):
        async with backend.advisory_locks(request, shared=True):
            pytest.fail("shared terminal barrier admitted")
    disconnected = PostgresBackend(
        "postgresql://unused:unused@127.0.0.1:1/never-connect"
    )
    with pytest.raises(Exception, match="original connected backend"):
        async with disconnected.advisory_locks(request):
            pytest.fail("terminal barrier initialized a replacement backend")


async def test_terminal_barrier_session_loss_never_revives(native_pg):
    backend, observer, _ = native_pg
    notifications, lost, key = [], asyncio.Event(), _key()

    def on_loss(lease):
        notifications.append(lease)
        lost.set()

    with pytest.raises(ExecutionAuthorityError, match="advisory.*lost"):
        async with backend.advisory_locks(
            TerminalAdvisoryBarrier((key,)), on_loss=on_loss
        ) as lease:
            assert await observer.fetchval(
                "SELECT pg_terminate_backend($1)", lease.backend_pid
            )
            await asyncio.wait_for(lost.wait(), 5)
            with pytest.raises(ExecutionAuthorityError):
                lease.require_physical()
    assert notifications == [lease]
    async with backend.advisory_locks(TerminalAdvisoryBarrier((key,))) as fresh:
        fresh.require_physical()
        with pytest.raises(ExecutionAuthorityError):
            lease.require_physical()


@pytest.mark.parametrize("control", ["none", "denied", "unknown"])
@pytest.mark.parametrize("eager", [False, True])
async def test_reused_ordinary_session_loss_retains_terminal_callback_custody(
    native_pg, control, eager
):
    backend, observer, schema = native_pg
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("eager task factory requires Python 3.12+")
    # asyncpg transports retain their connection-creation context, which here
    # deliberately predates every terminal/denied scope.
    async with backend.advisory_locks((_key(),)) as warm:
        warm_pid = warm.backend_pid
    callbacks, children, outcomes = [], [], []
    notified = asyncio.Event()
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )

    def on_loss(_lease):
        callbacks.append(current_execution_custody(backend))
        children.append(
            create_owned_task(
                backend.execute(
                    "INSERT INTO effects VALUES (37391, 'reused-session escape')"
                ),
                retain_originals=True,
            )
        )
        notified.set()

    async def operation():
        async with backend.advisory_locks(
            TerminalAdvisoryBarrier((_key(),)), on_loss=on_loss
        ) as lease:
            assert lease.backend_pid == warm_pid, (
                "test must reuse the actual warmed session"
            )
            assert await observer.fetchval("SELECT pg_terminate_backend($1)", warm_pid)
            await asyncio.wait_for(notified.wait(), 5)
            outcomes.append(await await_owned_task(children[0]))

    try:
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        original = None
        if control == "none":
            result = await await_owned_task(
                create_owned_task(operation(), retain_originals=True)
            )
        else:
            with bind_execution_custody(GenerationFence()) as original:
                if control == "unknown":
                    original.preserve_commit_uncertainty("unknown")
                else:
                    original.revoke("original denied before session reuse")
                result = await await_owned_task(
                    create_owned_task(operation(), retain_originals=True)
                )
        assert len(callbacks) == len(outcomes) == 1
        assert any(
            isinstance(scope.fence, TerminalAdvisoryBarrier) for scope in callbacks[0]
        ), "native reused-session callback lost its original terminal scope"
        if original is not None:
            assert original in callbacks[0], (
                "native loss callback lost the original denied/unknown scope"
            )
        assert is_execution_control_error(outcomes[0].error)
        assert execution_commit_outcome(outcomes[0].error) == (
            "unknown" if control == "unknown" else None
        )
        assert is_execution_control_error(result.error)
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0
    finally:
        loop.set_task_factory(previous)
        for child in children:
            await await_owned_task(child)


@pytest.mark.parametrize("phase", ["init", "on_loss", "reset"])
@pytest.mark.parametrize("eager", [False, True])
async def test_terminal_denial_covers_native_callbacks_before_and_after_body(
    native_pg,
    monkeypatch,
    phase,
    eager,
):
    backend, observer, schema = native_pg
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("eager task factory requires Python 3.12+")
    children, results, pids = [], [], []
    loop, prior = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    actual_reset = asyncpg.Connection.reset

    async def ordinary_write():
        await backend.execute(
            "INSERT INTO effects VALUES (37390, 'terminal callback ordinary work')"
        )

    async def init(_connection):
        if phase == "init":
            children.append(create_owned_task(ordinary_write(), retain_originals=True))
            results.append(await await_owned_task(children[-1]))

    def on_loss(_lease):
        if phase == "on_loss":
            children.append(create_owned_task(ordinary_write(), retain_originals=True))

    async def reset(connection, *args, **kwargs):
        if phase == "reset" and connection.get_server_pid() in pids:
            children.append(create_owned_task(ordinary_write(), retain_originals=True))
            results.append(await await_owned_task(children[-1]))
        return await actual_reset(connection, *args, **kwargs)

    local = PostgresBackend.from_pool(
        backend.operational_pool, advisory_connect_kwargs={"init": init}
    )
    monkeypatch.setattr(asyncpg.Connection, "reset", reset)
    try:
        if eager:
            loop.set_task_factory(asyncio.eager_task_factory)

        async def operation():
            async with local.advisory_locks(
                TerminalAdvisoryBarrier((_key(),)), on_loss=on_loss
            ) as lease:
                pids.append(lease.backend_pid)
                if phase == "on_loss":
                    assert await observer.fetchval(
                        "SELECT pg_terminate_backend($1)", lease.backend_pid
                    )
                    async with asyncio.timeout(5):
                        while not children:
                            await asyncio.sleep(0.01)
                    results.append(await await_owned_task(children[-1]))

        outcome = await await_owned_task(
            create_owned_task(operation(), retain_originals=True)
        )
        if phase == "on_loss":
            assert is_execution_control_error(outcome.error)
        else:
            assert outcome.error is None, outcome.error
        assert len(children) == len(results) == 1
        assert is_execution_control_error(results[0].error), (
            f"terminal {phase} callback admitted ordinary work"
        )
        assert (
            await observer.fetchval(
                f'SELECT count(*) FROM "{schema}".effects WHERE id=37390'
            )
            == 0
        )
    finally:
        loop.set_task_factory(prior)
        for child in children:
            await await_owned_task(child)
        monkeypatch.undo()
        await local.close()


@pytest.mark.parametrize("control", ["denied", "unknown"])
async def test_terminal_pre_yield_cluster_refusal_preserves_original_control(
    native_pg, control
):
    backend, observer, _ = native_pg
    key = _key()
    with bind_execution_custody(GenerationFence()) as original:
        if control == "unknown":
            original.preserve_commit_uncertainty("unknown")
        else:
            original.revoke("original runtime denied before acquisition")
        with pytest.raises((ExecutionAuthorityError, BaseExceptionGroup)) as caught:
            async with backend.advisory_locks(
                TerminalAdvisoryBarrier((key,)),
                expected_cluster_identity="not-the-original-native-cluster",
            ):
                pytest.fail("wrong native cluster admitted a terminal physical body")
        assert execution_commit_outcome(caught.value) == (
            "unknown" if control == "unknown" else None
        )
        assert is_execution_control_error(caught.value)
        assert any(
            isinstance(error, QueryError)
            and "validated PostgreSQL cluster" in str(error)
            for error in _explicit_control_chain(caught.value)
        )
    assert await observer.fetchval("SELECT pg_try_advisory_lock($1,$2)", *key)
    assert await observer.fetchval("SELECT pg_advisory_unlock($1,$2)", *key)


async def test_terminal_barrier_does_not_borrow_last_operational_connection(native_pg):
    _backend, observer, schema = native_pg
    pool = await asyncpg.create_pool(
        os.environ["TEST_POSTGRES_URL"],
        min_size=1,
        max_size=1,
        server_settings={"search_path": schema},
    )
    local = PostgresBackend.from_pool(pool, advisory_max_pool_size=1)
    try:
        async with (
            pool.acquire() as occupied,
            asyncio.timeout(5),
            local.advisory_locks(TerminalAdvisoryBarrier((_key(),))) as lease,
        ):
            assert occupied.get_server_pid() != lease.backend_pid
            assert (
                await observer.fetchval(
                    "SELECT count(*) FROM pg_locks WHERE pid=$1 AND locktype='advisory' AND granted",
                    lease.backend_pid,
                )
                == 1
            )
    finally:
        await local.close()
        await pool.close()


async def test_terminal_barrier_cannot_replace_externally_closed_borrowed_pool(
    native_pg,
):
    _backend, _observer, schema = native_pg
    pool = await asyncpg.create_pool(
        os.environ["TEST_POSTGRES_URL"],
        min_size=1,
        max_size=1,
        server_settings={"search_path": schema},
    )
    local = PostgresBackend.from_pool(pool, advisory_max_pool_size=1)
    entered = []
    await pool.close()
    try:
        with pytest.raises(Exception, match="original connected backend"):
            async with local.advisory_locks(TerminalAdvisoryBarrier((_key(),))):
                entered.append(True)
        assert entered == [], (
            "closed borrowed operational pool admitted a replacement physical barrier"
        )
    finally:
        await local.close()


async def test_terminal_waiter_cannot_outlive_original_delegated_backend(native_pg):
    backend, observer, _ = native_pg
    child = PostgresBackend.from_pool(
        backend.operational_pool, advisory_backend=backend
    )
    key, entered = _key(), []

    async def waiter():
        async with child.advisory_locks(TerminalAdvisoryBarrier((key,))):
            entered.append(True)

    owner = None
    async with backend.advisory_locks([_key()]):
        owner = create_owned_task(waiter(), retain_originals=True)
        try:
            # Pool bound is one; close the exact child while its original
            # acquisition waits on the host's bounded advisory checkout.
            async with asyncio.timeout(5):
                while not backend._advisory_pool._queue._getters:
                    await asyncio.sleep(0)
            await child.close()
        finally:
            if owner.done():
                await await_owned_task(owner)
    outcome = await await_owned_task(owner)
    assert entered == [], (
        "terminal acquisition substituted a retired child backend lifecycle"
    )
    assert outcome.error is not None
    assert await observer.fetchval("SELECT pg_try_advisory_lock($1,$2)", *key)
    assert await observer.fetchval("SELECT pg_advisory_unlock($1,$2)", *key)


async def test_terminal_creation_refusal_retires_exact_new_native_pool(
    native_pg, monkeypatch
):
    backend, observer, _ = native_pg
    local = PostgresBackend.from_pool(
        backend.operational_pool, advisory_max_pool_size=1
    )
    original_create = asyncpg.create_pool
    created, pids = [], []
    entered, release = asyncio.Event(), asyncio.Event()

    async def create(*args, **kwargs):
        pool = await original_create(*args, **kwargs)
        created.append(pool)
        # Establish one actual native session so this is physical ownership,
        # not an initialized-but-empty Python pool object.
        async with pool.acquire() as connection:
            pids.append(connection.get_server_pid())
        entered.set()
        await release.wait()
        return pool

    async def operation():
        async with local.advisory_locks(TerminalAdvisoryBarrier((_key(),))):
            pytest.fail("terminal pool creation substituted a retired backend")

    monkeypatch.setattr(asyncpg, "create_pool", create)
    task = create_owned_task(operation(), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await local.close()
        release.set()
        outcome = await await_owned_task(task)
        assert outcome.error is not None
        assert len(created) == 1 and created[0].is_closing(), (
            "refused terminal creation left its original native pool live"
        )
        assert local._advisory_pool is None
        assert not await observer.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1)", pids[0]
        )
    finally:
        release.set()
        await await_owned_task(task)
        monkeypatch.undo()
        await local.close()
        for pool in created:
            await pool.close()


@pytest.mark.parametrize("phase", ["acquire", "body", "release"])
@pytest.mark.parametrize("control", ["denied", "unknown"])
@pytest.mark.parametrize("late_failure", [False, True])
async def test_terminal_original_controls_survive_repeated_withdrawal_and_native_tails(
    native_pg,
    monkeypatch,
    phase,
    control,
    late_failure,
):
    from kestrel_sovereign.storage.db import postgres as postgres_module

    backend, observer, schema = native_pg
    key, sessions, owners, withdrawals = _key(), [], [], []
    entered, release, body_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    actual_execute, actual_close = asyncpg.Connection.execute, asyncpg.Connection.close
    actual_release, actual_join = asyncpg.Pool.release, postgres_module.await_owned_task
    late = OriginalFailure.capture(
        OSError("actual terminal native retirement acknowledgement failed")
    )
    original = None
    acquired = False

    async def execute(connection, query, *params, **kwargs):
        if query == "SELECT pg_advisory_lock($1, $2)" and params == key:
            sessions.append(connection)
        return await actual_execute(connection, query, *params, **kwargs)

    async def close(connection, *args, **kwargs):
        if sessions and connection is sessions[0] and phase in {"acquire", "body"}:
            owners.append(asyncio.current_task())
            entered.set()
            await release.wait()
            await actual_close(connection, *args, **kwargs)
            if late_failure:
                raise late.error
            return
        return await actual_close(connection, *args, **kwargs)

    async def pool_release(pool, *args, **kwargs):
        if pool is backend._advisory_pool and phase == "release":
            owners.append(asyncio.current_task())
            entered.set()
            await release.wait()
            await actual_release(pool, *args, **kwargs)
            if late_failure:
                raise late.error
            return
        return await actual_release(pool, *args, **kwargs)

    async def join(task, *args, **kwargs):
        outcome = await actual_join(task, *args, **kwargs)
        if task in owners and outcome.cancellation is not None:
            withdrawals.append(OriginalFailure.capture(outcome.cancellation))
        return outcome

    async def invoke():
        async with backend.advisory_locks(TerminalAdvisoryBarrier((key,))):
            body_entered.set()
            if original is not None:
                raise original.error
            if phase == "body":
                await asyncio.Event().wait()

    task = None
    try:
        with bind_execution_custody(GenerationFence()) as ancestor:
            if control == "unknown":
                with monkeypatch.context() as fault:
                    receipts = lose_native_ack(
                        fault,
                        boundary="commit",
                        original=OSError(
                            "original genuine COMMIT acknowledgement lost"
                        ),
                    )
                    with pytest.raises(QueryError) as caught:
                        await backend.execute(
                            "INSERT INTO effects VALUES (3739, 'actual original native effect')"
                        )
                    original = OriginalFailure.capture(caught.value)
                assert receipts == [True] and original is not None
                assert (
                    await observer.fetchval(
                        f'SELECT count(*) FROM "{schema}".effects WHERE id=3739'
                    )
                    == 1
                )
            else:
                ancestor.revoke(
                    "original ancestor admission denied before terminal acquisition"
                )
            monkeypatch.setattr(asyncpg.Connection, "execute", execute)
            monkeypatch.setattr(asyncpg.Connection, "close", close)
            monkeypatch.setattr(asyncpg.Pool, "release", pool_release)
            monkeypatch.setattr(postgres_module, "await_owned_task", join)
            if phase == "acquire":
                await observer.fetchval("SELECT pg_advisory_lock($1,$2)", *key)
                acquired = True
            task = create_owned_task(invoke(), retain_originals=True)
            if phase == "acquire":
                async with asyncio.timeout(5):
                    while not sessions or not await observer.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=$1 AND locktype='advisory' AND NOT granted)",
                        sessions[0].get_server_pid(),
                    ):
                        assert not task.done(), (
                            "terminal acquisition refused before its original native wait"
                        )
                        await asyncio.sleep(0.01)
                task.cancel("first actual withdrawal while acquiring terminal barrier")
            elif phase == "body":
                await asyncio.wait_for(body_entered.wait(), 5)
                task.cancel("first actual terminal body withdrawal")
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel("actual terminal native-tail observer withdrew")
            await asyncio.sleep(0)
            task.cancel("repeated terminal native-tail observer withdrew")
            await asyncio.sleep(0)
            assert not task.done(), (
                "terminal barrier returned before the original native retirement owner joined"
            )
            assert owners and all(not owner.done() for owner in owners)
            release.set()
            outcome = await await_owned_task(task)
            assert all(owner.done() for owner in owners)
            assert execution_commit_outcome(outcome.error) == (
                "unknown" if control == "unknown" else None
            )
            assert isinstance(outcome.error, BaseException)
            assert any(
                isinstance(error, asyncio.CancelledError)
                for error in _explicit_control_chain(outcome.error)
            )
            assert withdrawals, (
                "original repeated-cancellation observer was not retained"
            )
            for withdrawal in withdrawals:
                withdrawal.assert_retained(outcome.error)
            if original is not None and phase != "acquire":
                original.assert_retained(outcome.error)
            if late_failure:
                late.assert_retained(outcome.error)
            assert body_entered.is_set() is (phase != "acquire")
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        monkeypatch.undo()
        # A deliberately broken reset-owner probe must not leave its exact
        # fixture checkout wedged after the original assertion has failed.
        # This is teardown, never the barrier's physical completion evidence.
        pool = backend._advisory_pool
        if pool is not None:
            for holder in pool._holders:
                if holder._in_use is not None and holder._proxy is not None:
                    await actual_release(pool, holder._proxy)
        if acquired:
            await observer.fetchval("SELECT pg_advisory_unlock($1,$2)", *key)
        # Independently verify the exact key is free after actual native joins.
        assert await observer.fetchval("SELECT pg_try_advisory_lock($1,$2)", *key)
        assert await observer.fetchval("SELECT pg_advisory_unlock($1,$2)", *key)
