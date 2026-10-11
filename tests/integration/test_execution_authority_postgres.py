"""Exact session death and transaction-bound generation fences (#3569).

Uses only TEST_POSTGRES_URL and owns a unique schema; no shared rows are reset.
"""

import asyncio
import os
import uuid

import asyncpg
import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    bind_execution_custody,
    require_execution_work,
)
from kestrel_sovereign.storage.db.postgres import PostgresBackend


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.fixture
async def native_pg():
    dsn = os.environ.get("TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("TEST_POSTGRES_URL required")
    schema = "custody_" + uuid.uuid4().hex
    control = await asyncpg.connect(dsn)
    await control.execute(f'CREATE SCHEMA "{schema}"')
    pool = await asyncpg.create_pool(
        dsn, min_size=0, max_size=2,
        server_settings={"search_path": schema},
    )
    backend = PostgresBackend.from_pool(pool, advisory_max_pool_size=1)
    try:
        await backend.execute("CREATE TABLE authority (id INTEGER PRIMARY KEY, generation INTEGER)")
        await backend.execute("INSERT INTO authority VALUES (1, 1)")
        await backend.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY, value TEXT)")
        yield backend, control, schema
    finally:
        try:
            async with asyncio.timeout(10):
                await backend.close()
                await pool.close()
        finally:
            await control.execute(f'DROP SCHEMA "{schema}" CASCADE')
            await control.close()


class GenerationFence:
    backend_type = "postgres"

    def __init__(self, generation=1, lease=None):
        self.generation = generation
        self.lease = lease

    def require_work(self):
        if self.lease is not None:
            self.lease.require_live()

    async def lock_and_validate(self, connection):
        row = await connection.fetchrow(
            "SELECT generation FROM authority WHERE id=1 FOR SHARE"
        )
        if row is None or row[0] != self.generation:
            raise ExecutionAuthorityError("runtime generation changed")


async def test_exact_advisory_session_loss_irrevocably_denies_other_connections(native_pg):
    backend, control, _ = native_pg
    lost = asyncio.Event()
    observed = []

    def on_loss(lease):
        observed.append(lease)
        lost.set()

    with pytest.raises(Exception, match="authority|advisory|session"):
        async with backend.advisory_locks([(3569, 1)], on_loss=on_loss) as lease:
            lease.require_live()
            with bind_execution_custody(GenerationFence(lease=lease)):
                assert await control.fetchval("SELECT pg_terminate_backend($1)", lease.backend_pid)
                await asyncio.wait_for(lost.wait(), timeout=5)
                with pytest.raises(Exception, match="authority|advisory|session"):
                    await backend.execute("INSERT INTO effects VALUES (1, 'stale')")
                lease.require_live()
    assert len(observed) == 1
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0
    async with backend.advisory_locks([(3569, 1)]) as replacement:
        replacement.require_live()
        with pytest.raises(Exception, match="authority|advisory|session"):
            lease.require_live()


async def test_normal_retirement_invalidates_handle_without_loss_notification(native_pg):
    backend, _, _ = native_pg
    observed = []
    async with backend.advisory_locks([(3569, 2)], on_loss=observed.append) as lease:
        lease.require_live()
    with pytest.raises(Exception, match="retired|released|authority|advisory"):
        lease.require_live()
    assert observed == []
    async with backend.advisory_locks([]) as empty:
        assert empty is None
    with pytest.raises(ValueError, match="empty"):
        async with backend.advisory_locks([], on_loss=observed.append):
            pytest.fail("an empty exclusion set fabricated an authority handle")


async def test_unlock_loss_does_not_replace_original_cancellation(native_pg, monkeypatch):
    backend, control, _ = native_pg
    execute = asyncpg.Connection.execute
    original = asyncio.CancelledError("cancelled during exact-session unlock")

    async def lose_before_unlock(connection, query, *params, **kwargs):
        if query.startswith("SELECT pg_advisory_unlock("):
            assert await control.fetchval("SELECT pg_terminate_backend($1)", connection.get_server_pid())
            async with asyncio.timeout(5):
                while not connection.is_closed():
                    await asyncio.sleep(0.01)
            raise original
        return await execute(connection, query, *params, **kwargs)

    monkeypatch.setattr(asyncpg.Connection, "execute", lose_before_unlock)
    with pytest.raises(asyncio.CancelledError) as caught:
        async with backend.advisory_locks([(3569, 5)]):
            pass
    assert caught.value is original
    monkeypatch.setattr(asyncpg.Connection, "execute", execute)
    async with backend.advisory_locks([(3569, 5)]) as replacement:
        replacement.require_live()


@pytest.mark.parametrize("explicit,scoped", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_lost_commit_acknowledgement_requires_reconciliation(native_pg, monkeypatch, explicit, scoped, cancelled):
    from contextlib import nullcontext
    from kestrel_sovereign.execution_custody import execution_commit_outcome

    backend, _, _ = native_pg
    exit_transaction = asyncpg.transaction.Transaction.__aexit__

    async def lose_ack(transaction, error_type, error, traceback):
        result = await exit_transaction(transaction, error_type, error, traceback)
        if error_type is None:
            if cancelled:
                raise asyncio.CancelledError("injected cancelled acknowledgement after native commit")
            raise OSError("injected lost acknowledgement after native commit")
        return result

    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", lose_ack)
    with bind_execution_custody(GenerationFence()) if scoped else nullcontext():
        with pytest.raises(Exception) as caught:
            if explicit:
                async with backend.transaction():
                    await backend.execute("INSERT INTO effects VALUES (1, 'committed without ack')")
            else:
                await backend.execute("INSERT INTO effects VALUES (1, 'committed without ack')")
        assert execution_commit_outcome(caught.value) == "unknown"
    monkeypatch.setattr(asyncpg.transaction.Transaction, "__aexit__", exit_transaction)
    assert await backend.fetch_val("SELECT value FROM effects") == "committed without ack"


async def test_session_dies_while_acquiring_and_body_is_never_admitted(native_pg):
    backend, control, _ = native_pg
    key = (3569, 3)
    await control.execute("SELECT pg_advisory_lock($1, $2)", *key)
    observed = []

    async def waiter():
        async with backend.advisory_locks([key], on_loss=observed.append):
            pytest.fail("a lost acquisition admitted target work")

    task = asyncio.create_task(waiter())
    try:
        async with asyncio.timeout(5):
            while True:
                pid = await backend.fetch_val(
                    "SELECT pid FROM pg_locks WHERE locktype='advisory' "
                    "AND classid=? AND objid=? AND NOT granted",
                    key,
                )
                if pid is not None:
                    break
                await asyncio.sleep(0.01)
        assert await control.fetchval("SELECT pg_terminate_backend($1)", pid)
        with pytest.raises(Exception):
            await asyncio.wait_for(task, timeout=5)
        await asyncio.sleep(0)  # Deliver any queued physical-loss callback.
        assert len(observed) == 1
        assert observed[0].backend_pid == pid
        assert observed[0].lost
    finally:
        await control.execute("SELECT pg_advisory_unlock($1, $2)", *key)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_body_cancellation_retires_without_false_fatal_notification(native_pg):
    backend, _, _ = native_pg
    entered = asyncio.Event()
    observed = []
    held = None

    async def invocation():
        nonlocal held
        async with backend.advisory_locks([(3569, 4)], on_loss=observed.append) as held:
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(invocation())
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert observed == []
    with pytest.raises(ExecutionAuthorityError, match="retired"):
        held.require_live()
    # Normal cancellation terminated only this lease's session; the bounded
    # pool remains usable for a separate, newly admitted operation.
    async with backend.advisory_locks([(3569, 4)]) as replacement:
        replacement.require_live()


async def test_already_queued_callback_cannot_revive_or_falsely_fail_retirement(native_pg):
    backend, _, _ = native_pg
    observed = []
    async with backend.advisory_locks([(3569, 5)], on_loss=observed.append) as lease:
        # Model asyncpg's already queued public callback, which removing the
        # listener cannot retract. No await occurs before body retirement.
        asyncio.get_running_loop().call_soon(lease._connection_terminated, None)
    await asyncio.sleep(0)
    assert observed == []
    with pytest.raises(ExecutionAuthorityError, match="retired"):
        lease.require_live()


@pytest.mark.parametrize("method", ["execute", "execute_many", "fetch_one", "fetch_all", "fetch_val", "execute_script", "transaction"])
async def test_stale_generation_rejects_every_native_sql_surface(native_pg, method):
    backend, _, _ = native_pg
    fence = GenerationFence()
    await backend.execute("UPDATE authority SET generation=2 WHERE id=1")
    with bind_execution_custody(fence):
        with pytest.raises(Exception, match="runtime generation changed"):
            if method == "execute_many":
                await backend.execute_many("INSERT INTO effects VALUES (?, ?)", [(1, "stale")])
            elif method == "execute_script":
                await backend.execute_script("INSERT INTO effects VALUES (1, 'stale');")
            elif method == "transaction":
                async with backend.transaction():
                    await backend.execute("INSERT INTO effects VALUES (1, 'stale')")
            else:
                await getattr(backend, method)("INSERT INTO effects VALUES (1, 'stale') RETURNING id")
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0


async def test_guarded_mutation_serializes_before_generation_transfer(native_pg):
    backend, control, schema = native_pg
    guard_held = asyncio.Event()
    release = asyncio.Event()

    async def mutate():
        with bind_execution_custody(GenerationFence()):
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (1, 'authorized')")
                guard_held.set()
                await release.wait()

    mutation = asyncio.create_task(mutate())
    await asyncio.wait_for(guard_held.wait(), timeout=5)
    transfer = asyncio.create_task(control.execute(
        f'UPDATE "{schema}".authority SET generation=2 WHERE id=1'
    ))
    try:
        # PostgreSQL reports the exact blocked lock, not a timing assumption.
        async with asyncio.timeout(5):
            while not await backend.fetch_val(
                "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE pid=? AND wait_event_type='Lock')",
                (control.get_server_pid(),),
            ):
                await asyncio.sleep(0.01)
        assert not transfer.done()
        release.set()
        await mutation
        await transfer
    finally:
        release.set()
        await asyncio.gather(mutation, transfer, return_exceptions=True)
    assert await backend.fetch_val("SELECT generation FROM authority") == 2
    assert await backend.fetch_val("SELECT value FROM effects") == "authorized"


async def test_revocation_inside_transaction_rolls_back_and_sql_cannot_escape(native_pg):
    backend, _, _ = native_pg
    with bind_execution_custody(GenerationFence()) as scope:
        with pytest.raises(Exception, match="revoked while awaiting"):
            async with backend.transaction():
                await backend.execute("INSERT INTO effects VALUES (1, 'stale')")
                scope.revoke("revoked while awaiting")
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0
    with bind_execution_custody(GenerationFence()):
        with pytest.raises(Exception, match="transaction|Transaction"):
            await backend.execute_script("COMMIT; INSERT INTO effects VALUES (1, 'escape');")
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0


@pytest.mark.parametrize("explicit", [False, True])
async def test_postcommit_revocation_does_not_claim_rollback(native_pg, explicit):
    backend, _, _ = native_pg

    class LossAfterCommit(GenerationFence):
        connection = None

        def require_work(self):
            if self.connection is not None and not self.connection.is_in_transaction():
                raise ExecutionAuthorityError("authority notification raced commit")

        async def lock_and_validate(self, connection):
            await super().lock_and_validate(connection)
            self.connection = connection

    with bind_execution_custody(LossAfterCommit()):
        with pytest.raises(Exception, match="may have committed") as caught:
            if explicit:
                async with backend.transaction():
                    await backend.execute("INSERT INTO effects VALUES (1, 'committed')")
            else:
                await backend.execute("INSERT INTO effects VALUES (1, 'committed')")
        from kestrel_sovereign.execution_custody import execution_commit_outcome

        assert execution_commit_outcome(caught.value) == "committed"
    assert await backend.fetch_val("SELECT value FROM effects") == "committed"


async def test_alternate_sqlite_path_cannot_silently_drop_postgres_fence(tmp_path):
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "custody.db"))
    await backend.connect()
    try:
        with bind_execution_custody(GenerationFence()):
            with pytest.raises(ExecutionAuthorityError, match="backend"):
                await backend.execute("CREATE TABLE forbidden (id INTEGER)")
    finally:
        await backend.close()


@pytest.mark.parametrize("surface", ["graph", "file", "conversation"])
async def test_native_storage_cannot_commit_after_owner_generation_changes(native_pg, surface):
    from kestrel_sovereign.storage import AsyncStorage, GraphNode

    backend, _, _ = native_pg
    storage = AsyncStorage.from_backend(backend, agent_id="did:example:custody-test")
    await storage.initialize()
    await backend.execute("UPDATE authority SET generation=2 WHERE id=1")
    with bind_execution_custody(GenerationFence()):
        with pytest.raises(Exception, match="runtime generation changed"):
            if surface == "graph":
                await storage.add_node(GraphNode("stale", "note", "stale", {}))
            elif surface == "file":
                await storage.store_file(b"stale", "stale.txt")
            else:
                await storage.add_conversation("user", "stale")
    assert await backend.fetch_val("SELECT count(*) FROM graph_nodes WHERE node_id='stale'") == 0
    assert await backend.fetch_val("SELECT count(*) FROM files") == 0
    assert await backend.fetch_val("SELECT count(*) FROM conversation_history") == 0


@pytest.mark.parametrize("nested_savepoint", [False, True])
async def test_transaction_refuses_late_admission_before_native_work(native_pg, nested_savepoint):
    backend, _, _ = native_pg
    # The authority/graph lock order is fixed at transaction entry. A later
    # child must be refused before its validator can acquire another row.
    with pytest.raises(Exception, match="cannot be added"):
        async with backend.transaction():
            with bind_execution_custody(GenerationFence()):
                if nested_savepoint:
                    async with backend.transaction():
                        await backend.execute("INSERT INTO effects VALUES (1, 'stale')")
                else:
                    await backend.execute("INSERT INTO effects VALUES (1, 'stale')")
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0


async def test_transaction_keeps_all_entry_admissions_after_child_pop(native_pg):
    backend, _, _ = native_pg
    with bind_execution_custody(GenerationFence()):
        with bind_execution_custody(GenerationFence()):
            # Both authorities are established before graph locking. Capture
            # the transaction in a child, then retire its parent admissions.
            mutated = asyncio.Event()
            proceed = asyncio.Event()

            async def child():
                async with backend.transaction():
                    await backend.execute("INSERT INTO effects VALUES (1, 'stale')")
                    mutated.set()
                    await proceed.wait()

            task = asyncio.create_task(child())
            await asyncio.wait_for(mutated.wait(), timeout=5)
    try:
        proceed.set()
        with pytest.raises(Exception, match="execution admission retired"):
            await task
    finally:
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0


@pytest.mark.parametrize("explicit", [False, True])
async def test_retained_backend_fence_guards_a_task_with_no_ambient_scope(native_pg, explicit):
    backend, _, _ = native_pg
    runtime = ExecutionCustody(GenerationFence())
    guarded = PostgresBackend.from_pool(backend.operational_pool, execution_custody=runtime)
    proceed = asyncio.Event()

    async def foreign_task():
        await proceed.wait()
        with pytest.raises(Exception, match="runtime generation changed"):
            if explicit:
                async with guarded.transaction():
                    await guarded.execute("INSERT INTO effects VALUES (1, 'stale')")
            else:
                await guarded.execute("INSERT INTO effects VALUES (1, 'stale')")

    task = asyncio.create_task(foreign_task())
    try:
        await backend.execute("UPDATE authority SET generation=2 WHERE id=1")
        proceed.set()
        await task
    finally:
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)
        await guarded.close()
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0


@pytest.mark.parametrize("abandon", [False, True])
async def test_denied_work_can_only_settle_its_exact_existing_stop_identity(native_pg, abandon):
    from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.stop.invocation import DistributedInvocationStore

    backend, _, _ = native_pg
    store = DistributedInvocationStore(AsyncDatabase(backend))
    await store.ensure_schema()
    for generation, owner in (("generation-a", "owner-a"), ("generation-b", "owner-b")):
        assert await store.register(
            generation_id=generation, agent_id="did:example:terminal-test",
            turn_id=generation, owner_id=owner, request_generation=1,
        )
    with bind_execution_custody(GenerationFence()) as scope:
        scope.revoke("work has been revoked")
        # A wrong owner cannot affect either existing generation.
        await store.settle("generation-a", "owner-b", RequestCompletionDisposition.COMPLETED)
        disposition = RequestCompletionDisposition.ABANDONED if abandon else RequestCompletionDisposition.COMPLETED
        await store.settle("generation-a", "owner-a", disposition)
        with pytest.raises(ExecutionAuthorityError, match="revoked"):
            await backend.execute("INSERT INTO effects VALUES (1, 'forbidden cleanup')")
    assert await backend.fetch_all("SELECT generation_id FROM stop_active_invocations") == [("generation-b",)]
    assert await backend.fetch_all("SELECT generation_id FROM stop_unresolved_invocations") == ([("generation-a",)] if abandon else [])
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0


async def test_native_invocation_preserves_completed_effect_as_unresolved_after_denial(native_pg):
    from kestrel_sovereign.agent.invocation import bind_async_invocation, mark_current_invocation_effect_completed
    from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.stop.invocation import DistributedInvocationStore

    backend, _, _ = native_pg
    store = DistributedInvocationStore(AsyncDatabase(backend))
    await store.ensure_schema()
    custody = ExecutionCustody(GenerationFence())
    cleanup = []

    class Owner:
        _execution_custody = custody

        def register_active_request(self, request_id, *, nested=False):
            pass

        async def await_durable_request_admission(self, request_id):
            return await store.register(
                generation_id="effect-generation", owner_id="effect-owner",
                agent_id="did:example:effect-owner", turn_id=request_id, request_generation=1,
            )

        async def _persist_completed_tool_stop_checkpoint(self, **kwargs):
            # Ordinary checkpoint persistence must NOT regain write access.
            await backend.execute("INSERT INTO effects VALUES (2, 'forbidden checkpoint')")

        def _cleanup_cancelled_request(self, request_id, disposition=RequestCompletionDisposition.COMPLETED):
            cleanup.append(asyncio.create_task(store.settle("effect-generation", "effect-owner", disposition)))

        @bind_async_invocation("request_id", track_request_lifecycle=True)
        async def invoke(self, request_id=None):
            await backend.execute("INSERT INTO effects VALUES (1, 'completed effect')")
            mark_current_invocation_effect_completed("session")
            custody.revoke("lost after completed effect")
            return "must not escape"

    owner = Owner()
    with pytest.raises(ExecutionAuthorityError, match="lost after completed effect") as caught:
        await owner.invoke(request_id="effect-turn")
    await asyncio.gather(*cleanup)
    assert "checkpoint remains unresolved" in " ".join(caught.value.__notes__)
    assert await backend.fetch_all("SELECT id FROM effects") == [(1,)]
    assert await backend.fetch_val("SELECT count(*) FROM stop_active_invocations") == 0
    assert await backend.fetch_all("SELECT generation_id FROM stop_unresolved_invocations") == [("effect-generation",)]


async def test_native_scheduler_shared_gate_loses_original_advisory_authority(native_pg):
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, _LeaseRenewalState, _rollout_renewal_state
    from kestrel_sovereign.execution_custody import current_execution_custody, require_execution_work
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    backend, control, _ = native_pg

    async def unused_dispatch(*args, **kwargs):
        pytest.fail("this gate test must not dispatch a provider")

    runner = SchedulerRunner(AsyncDatabase(backend), "did:example:gate", unused_dispatch)
    state = _LeaseRenewalState()
    token = _rollout_renewal_state.set(state)
    try:
        with pytest.raises(Exception, match="authority|session|advisory"):
            async with runner._postgres_rollout_effect_gate("did:example:gate"):
                lease = current_execution_custody()[-1].fence.lease
                assert state.custody == current_execution_custody()
                assert await control.fetchval("SELECT pg_terminate_backend($1)", lease.backend_pid)
                await asyncio.wait_for(state.lost.wait(), timeout=5)
                from types import SimpleNamespace
                assert await runner._renew_live_claim_once(SimpleNamespace(claim_execution_id="lost-claim")) is False
                with pytest.raises(Exception, match="authority|session|advisory"):
                    require_execution_work()
                with pytest.raises(Exception, match="authority|session|advisory"):
                    await backend.execute("INSERT INTO effects VALUES (1, 'stale scheduler')")
        assert await backend.fetch_val("SELECT count(*) FROM effects") == 0
        # A replacement lock does not repair the old handle or revive work.
        async with runner._postgres_rollout_transition_gate("did:example:gate"):
            with pytest.raises(Exception, match="authority|session|advisory"):
                lease.require_live()
    finally:
        _rollout_renewal_state.reset(token)


async def test_native_scheduler_renewal_rebinds_original_host_generation(native_pg):
    from types import SimpleNamespace
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, _LeaseRenewalState, _rollout_renewal_state
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    backend, _, _ = native_pg

    class Runner(SchedulerRunner):
        async def _renew_lease_once(self, task):
            # Uses the actual native mutation path; this test isolates the
            # renewal worker's generation propagation from claim SQL details.
            await self._db.execute("INSERT INTO effects VALUES (1, 'stale renewal')")
            return True

    async def unused_dispatch(*args, **kwargs):
        pytest.fail("renewal must not dispatch a provider")

    runner = Runner(AsyncDatabase(backend), "did:example:renewal", unused_dispatch)
    state = _LeaseRenewalState(custody=(ExecutionCustody(GenerationFence()),))
    token = _rollout_renewal_state.set(state)
    try:
        await backend.execute("UPDATE authority SET generation=2 WHERE id=1")
        with pytest.raises(Exception, match="runtime generation changed"):
            await runner._renew_live_claim_once(SimpleNamespace(claim_execution_id="old-claim"))
        assert await backend.fetch_val("SELECT count(*) FROM effects") == 0
        assert "old-claim" not in runner._live_claim_deadlines
    finally:
        _rollout_renewal_state.reset(token)


async def test_custody_bound_native_signal_boot_builds_exact_sequence_index(native_pg):
    from kestrel_sovereign.signals.durable import DurableSignalStore

    backend, _, _ = native_pg
    store = DurableSignalStore(backend)
    with bind_execution_custody(GenerationFence()):
        await store.initialize()
        assert store._postgres_source_sequence_index_catalog_valid(
            await store._postgres_source_sequence_index_catalog()
        )


async def test_scheduler_session_loss_cannot_repeat_completed_external_effect(native_pg):
    from datetime import datetime, timedelta, timezone
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, ScheduledTask, SCHEDULER_PROTOCOL_VERSION, _rollout_renewal_state
    from kestrel_sovereign.execution_custody import current_execution_custody
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    backend, control, _ = native_pg
    db = AsyncDatabase(backend)
    agent_id = "did:example:no-effect-replay"
    seen = []

    async def effect(name, args):
        seen.append(True)
        await backend.execute("INSERT INTO effects VALUES (1, 'irreversible external effect')")
        assert await db.fetchval("SELECT status FROM task_execution_log WHERE task_id='no-repeat'") == "executing"
        lease = current_execution_custody()[-1].fence.lease
        assert await control.fetchval("SELECT pg_terminate_backend($1)", lease.backend_pid)
        await asyncio.wait_for(_rollout_renewal_state.get().lost.wait(), timeout=5)
        return "completed external effect"

    runner = SchedulerRunner(db, agent_id, effect, owner_id="original-owner")
    await runner._ensure_tables()
    due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    await db.execute("""INSERT INTO scheduled_tasks
        (id, agent_id, task_name, cron_expression, enabled, next_run_at, created_at, idempotency_key, scheduler_protocol_version)
        VALUES ('no-repeat', ?, 'effect', '* * * * *', 1, ?, ?, 'no-repeat-base', ?)""",
        (agent_id, due, due, SCHEDULER_PROTOCOL_VERSION))
    task = ScheduledTask.from_row((await runner._due_rows(datetime.now(timezone.utc)))[0])
    claimed = await runner._claim(task, datetime.now(timezone.utc))
    assert claimed is not None
    from kestrel_sdk.storage.database import QueryError
    try:
        await runner._execute_claim(claimed)
    except (ExecutionAuthorityError, QueryError):
        pass  # The exact-session exclusion boundary must remain failed closed.
    assert seen == [True]
    assert await db.fetchval("SELECT status FROM task_execution_log WHERE task_id='no-repeat'") == "executing"
    await db.execute("UPDATE scheduled_tasks SET lease_expires_at=? WHERE id='no-repeat'", (due,))
    replacement = SchedulerRunner(db, agent_id, effect, owner_id="replacement-owner")
    retry = ScheduledTask.from_row((await replacement._due_rows(datetime.now(timezone.utc)))[0])
    assert await replacement._claim(retry, datetime.now(timezone.utc)) is None
    assert await db.fetchone("SELECT enabled, terminal_status FROM scheduled_tasks WHERE id='no-repeat'") == (0, "unresolved_effect")
    assert seen == [True]
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 1


async def test_failed_generation_validation_irrevocably_retires_original_admission(native_pg):
    backend, control, schema = native_pg
    with bind_execution_custody(GenerationFence()):
        await control.execute(f'UPDATE "{schema}".authority SET generation=2 WHERE id=1')
        with pytest.raises(Exception, match="generation changed"):
            await backend.execute("INSERT INTO effects VALUES (1, 'old generation')")
        await control.execute(f'UPDATE "{schema}".authority SET generation=1 WHERE id=1')
        with pytest.raises(ExecutionAuthorityError, match="generation changed"):
            await backend.execute("INSERT INTO effects VALUES (2, 'revived generation')")
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 0


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("operational", [False, True])
@pytest.mark.parametrize("failure", ["error", "cancel"])
@pytest.mark.parametrize("surface", ["query", "batch", "script"])
async def test_pool_release_failure_preserves_committed_outcome(native_pg, explicit, operational, failure, surface):
    from contextlib import asynccontextmanager
    from kestrel_sovereign.execution_custody import execution_commit_outcome
    backend, _, _ = native_pg
    pool = backend._pool
    class ReleaseFailurePool:
        def __getattr__(self, name):
            return getattr(pool, name)
        @asynccontextmanager
        async def acquire(self):
            async with pool.acquire() as connection:
                yield connection
            if failure == "cancel":
                raise asyncio.CancelledError("cancelled during pool reset")
            raise OSError("pool reset acknowledgement lost")
    backend._pool = ReleaseFailurePool()
    @asynccontextmanager
    async def checkout():
        if operational:
            async with backend.operational_session():
                yield
        else:
            yield
    try:
        async def write():
            if surface == "batch":
                await backend.execute_many("INSERT INTO effects VALUES (?, ?)", [(1, "committed before release")])
            elif surface == "script":
                await backend.execute_script("INSERT INTO effects VALUES (1, 'committed before release')")
            else:
                await backend.execute("INSERT INTO effects VALUES (1, 'committed before release')")
        with bind_execution_custody(GenerationFence()):
            with pytest.raises(Exception, match="may have committed") as caught:
                async with checkout():
                    if explicit:
                        async with backend.transaction():
                            await write()
                    else:
                        await write()
            assert execution_commit_outcome(caught.value) == "committed"
            with pytest.raises(Exception) as denied:
                require_execution_work()
            assert execution_commit_outcome(denied.value) == "committed"
    finally:
        backend._pool = pool
    assert await backend.fetch_val("SELECT value FROM effects") == "committed before release"


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("release", ["error", "cancel", "revoke"])
async def test_unbound_operational_checkout_retains_participating_admission(native_pg, explicit, release):
    from contextlib import asynccontextmanager
    from kestrel_sovereign.execution_custody import execution_commit_outcome

    backend, _, _ = native_pg
    pool = backend._pool
    from types import SimpleNamespace
    from kestrel_sovereign.execution_custody import bind_execution_runtime
    scope = ExecutionCustody(GenerationFence())
    owner = SimpleNamespace(_execution_custody=scope)

    class ReleaseBoundaryPool:
        def __getattr__(self, name):
            return getattr(pool, name)

        @asynccontextmanager
        async def acquire(self):
            async with pool.acquire() as connection:
                yield connection
                await asyncio.sleep(0)
                if release == "revoke":
                    scope.revoke("retired during pool release")
            if release == "error":
                raise OSError("late release failed")
            if release == "cancel":
                raise asyncio.CancelledError("late release cancelled")

    backend._pool = ReleaseBoundaryPool()
    try:
        with pytest.raises(Exception) as caught:
            # Shared operational sessions are deliberately unbound at entry.
            async with backend.operational_session():
                with bind_execution_runtime(owner):
                    if explicit:
                        async with backend.transaction():
                            await backend.execute("INSERT INTO effects VALUES (1, 'committed')")
                    else:
                        await backend.execute("INSERT INTO effects VALUES (1, 'committed')")
        assert execution_commit_outcome(caught.value) == "committed"
        with pytest.raises(Exception) as denied:
            scope.require_work()
        assert execution_commit_outcome(denied.value) == "committed"
    finally:
        backend._pool = pool
    assert await backend.fetch_val("SELECT value FROM effects") == "committed"


@pytest.mark.parametrize("surface", ["query", "batch", "script", "read"])
async def test_successful_release_revalidates_before_publishing(native_pg, surface):
    from contextlib import asynccontextmanager
    from kestrel_sovereign.execution_custody import execution_commit_outcome

    backend, _, _ = native_pg
    pool = backend._pool
    from types import SimpleNamespace
    from kestrel_sovereign.execution_custody import bind_execution_runtime
    scope = ExecutionCustody(GenerationFence())
    owner = SimpleNamespace(_execution_custody=scope)

    class RevokingReleasePool:
        def __getattr__(self, name):
            return getattr(pool, name)

        @asynccontextmanager
        async def acquire(self):
            async with pool.acquire() as connection:
                yield connection
                await asyncio.sleep(0)
                scope.revoke("revoked while resetting physical connection")

    backend._pool = RevokingReleasePool()
    try:
        with bind_execution_runtime(owner):
            with pytest.raises(Exception) as caught:
                if surface == "batch":
                    await backend.execute_many("INSERT INTO effects VALUES (?, ?)", [(1, "effect")])
                elif surface == "script":
                    await backend.execute_script("INSERT INTO effects VALUES (1, 'effect')")
                elif surface == "read":
                    await backend.fetch_val("SELECT generation FROM authority")
                else:
                    await backend.execute("INSERT INTO effects VALUES (1, 'effect')")
            assert execution_commit_outcome(caught.value) == "committed"
    finally:
        backend._pool = pool


async def test_scheduler_commit_uncertainty_remains_unresolved(native_pg):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, ScheduledTask, SCHEDULER_PROTOCOL_VERSION
    from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    backend, _, _ = native_pg
    db = AsyncDatabase(backend)
    agent_id = "did:example:commit-uncertain"
    calls = []
    async def effect(name, args):
        calls.append(True)
        await backend.execute("INSERT INTO effects VALUES (1, 'committed effect')")
        try:
            raise ExecutionCommitOutcomeError("unknown")
        except Exception as error:
            raise RuntimeError("wrapped commit uncertainty") from error
    runner = SchedulerRunner(db, agent_id, effect, owner_id="original-owner")
    await runner._ensure_tables()
    due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    await db.execute("""INSERT INTO scheduled_tasks
        (id, agent_id, task_name, cron_expression, enabled, next_run_at, created_at, idempotency_key, scheduler_protocol_version)
        VALUES ('uncertain', ?, 'effect', '* * * * *', 1, ?, ?, 'uncertain-base', ?)""", (agent_id, due, due, SCHEDULER_PROTOCOL_VERSION))
    task = ScheduledTask.from_row((await runner._due_rows(datetime.now(timezone.utc)))[0])
    claimed = await runner._claim(task, datetime.now(timezone.utc))
    await runner._execute_claim(claimed)
    assert await db.fetchval("SELECT status FROM task_execution_log WHERE task_id='uncertain'") == "executing"
    identity = await db.fetchone("SELECT claim_execution_id, claim_scheduled_for FROM scheduled_tasks WHERE id='uncertain'")
    owner = SimpleNamespace(did=agent_id, agent_id=agent_id, _raw_storage=SimpleNamespace(db=db), features={})
    feature = SchedulerFeature(owner)
    feature._db, feature._agent_id = db, agent_id
    paused = await feature.schedule_pause("uncertain")
    assert paused.status.value == "ok"
    assert await db.fetchone("SELECT claim_execution_id, claim_scheduled_for FROM scheduled_tasks WHERE id='uncertain'") == identity
    resumed = await feature.schedule_resume("uncertain", acknowledge_ambiguous_effect=True)
    assert resumed.status.value == "error"
    assert resumed.data["disabled_reason"] == "unresolved_effect"
    updated = await feature.schedule_update("uncertain", "*/2 * * * *")
    assert updated.status.value == "error"
    assert calls == [True]
    assert await backend.fetch_val("SELECT count(*) FROM effects") == 1


@pytest.mark.parametrize("outcome", ["denied", "unknown"])
async def test_scheduler_preparation_control_error_preserves_original_occurrence(native_pg, outcome):
    from contextlib import asynccontextmanager
    from datetime import datetime, timedelta, timezone
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, ScheduledTask, SCHEDULER_PROTOCOL_VERSION
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    backend, _, _ = native_pg
    db = AsyncDatabase(backend)
    prepared, dispatched = [], []
    agent_id = "did:test:preparation-custody"

    class Executor:
        @asynccontextmanager
        async def prepare_scheduled(self, execution):
            prepared.append(execution.id)
            if outcome == "unknown":
                await backend.execute("INSERT INTO effects VALUES (1, 'boot commit')")
                try:
                    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
                    raise ExecutionCommitOutcomeError("unknown")
                except Exception as error:
                    raise RuntimeError("cold boot acknowledgement lost") from error
            raise ExecutionAuthorityError("original warm runtime retired")
            yield lambda: dispatched.append(True)

    runner = SchedulerRunner(db, agent_id, Executor(), owner_id="original-preparation-owner")
    await runner._ensure_tables()
    due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    await db.execute("""INSERT INTO scheduled_tasks
        (id, agent_id, task_name, cron_expression, enabled, next_run_at, created_at, idempotency_key, scheduler_protocol_version)
        VALUES ('prepare-uncertain', ?, 'effect', '* * * * *', 1, ?, ?, 'preparation-base', ?)""",
        (agent_id, due, due, SCHEDULER_PROTOCOL_VERSION))
    task = ScheduledTask.from_row((await runner._due_rows(datetime.now(timezone.utc)))[0])
    claimed = await runner._claim(task, datetime.now(timezone.utc))
    await runner._execute_claim(claimed)
    assert await db.fetchval("SELECT status FROM task_execution_log WHERE task_id='prepare-uncertain'") == "executing"
    assert await db.fetchone("SELECT claim_execution_id, claim_scheduled_for, next_run_at FROM scheduled_tasks WHERE id='prepare-uncertain'") == (claimed.claim_execution_id, due, due)
    await db.execute("UPDATE scheduled_tasks SET lease_expires_at = ? WHERE id='prepare-uncertain'", ((datetime.now(timezone.utc)-timedelta(seconds=10)).isoformat(),))
    replacement = SchedulerRunner(db, agent_id, Executor(), owner_id="replacement-preparation-owner")
    recovered = ScheduledTask.from_row((await replacement._due_rows(datetime.now(timezone.utc)))[0])
    assert await replacement._claim(recovered, datetime.now(timezone.utc)) is None
    assert await db.fetchval("SELECT terminal_status FROM scheduled_tasks WHERE id='prepare-uncertain'") == "unresolved_effect"
    assert prepared == [claimed.claim_execution_id]
    assert not dispatched


@pytest.mark.parametrize("deferred", [False, True])
async def test_scheduler_marker_rechecks_expiry_after_untouched_row_lock(native_pg, deferred):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, ScheduledTask, SCHEDULER_PROTOCOL_VERSION
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    backend, control, schema = native_pg
    db = AsyncDatabase(backend)
    runner = SchedulerRunner(db, "marker-expiry", lambda *args: None, owner_id="marker-owner")
    await runner._ensure_tables()
    due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    await db.execute("""INSERT INTO scheduled_tasks
        (id, agent_id, task_name, cron_expression, enabled, next_run_at, created_at, idempotency_key, scheduler_protocol_version)
        VALUES ('marker', 'marker-expiry', 'effect', '* * * * *', 1, ?, ?, 'marker-base', ?)""", (due, due, SCHEDULER_PROTOCOL_VERSION))
    task = ScheduledTask.from_row((await runner._due_rows(datetime.now(timezone.utc)))[0])
    claimed = await runner._claim(task, datetime.now(timezone.utc))
    before = "executing" if deferred else "claimed"
    await db.execute("UPDATE task_execution_log SET status=? WHERE id=?", (before, claimed.claim_execution_id))
    await db.execute("UPDATE scheduled_tasks SET lease_expires_at=(clock_timestamp()+interval '0.6 seconds')::text WHERE id='marker'")
    blocker = control.transaction()
    await blocker.start()
    pending = None
    try:
        await control.fetchrow(f'SELECT id FROM "{schema}".scheduled_tasks WHERE id=\'marker\' FOR UPDATE')
        pending = asyncio.create_task(runner._mark_effect_started(claimed, SimpleNamespace(id=claimed.claim_execution_id), preparation_deferred=deferred))
        async with asyncio.timeout(5):
            while not await control.fetchval("SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE '%scheduled_tasks%' AND pid<>pg_backend_pid())"):
                await asyncio.sleep(0.01)
        # Real PostgreSQL evaluates the stale UPDATE predicate before this wait.
        # Holding an unchanged row is crucial: EPQ cannot refresh it afterward.
        await asyncio.sleep(0.65)
        await blocker.rollback()
        with pytest.raises(Exception, match="claim lost") as caught:
            await pending
        from kestrel_sovereign.execution_custody import is_execution_control_error
        assert is_execution_control_error(caught.value)
        assert await db.fetchval("SELECT status FROM task_execution_log WHERE id=?", (claimed.claim_execution_id,)) == before
    finally:
        if control.is_in_transaction():
            await blocker.rollback()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("phase", ["effect", "preparation"])
async def test_stalled_native_renewal_cancels_and_joins_expired_effect(native_pg, phase):
    from datetime import datetime, timedelta, timezone
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner, ScheduledTask, SCHEDULER_PROTOCOL_VERSION
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    backend, _, _ = native_pg
    db = AsyncDatabase(backend)
    started, stalled, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release = asyncio.Event()
    effects = []

    async def effect(*args):
        started.set()
        try:
            await release.wait()
            require_execution_work()
            effects.append("stale effect")
        except asyncio.CancelledError:
            # Cancellation-resistant code must also be denied by the native fence.
            with pytest.raises(ExecutionAuthorityError):
                require_execution_work()
            cancelled.set()
            return "cancellation resisted"

    executor = effect
    if phase == "preparation":
        from contextlib import asynccontextmanager
        class PreparedExecutor:
            @asynccontextmanager
            async def prepare_scheduled(self, execution):
                await effect()
                async def dispatch():
                    effects.append("stale dispatch")
                yield dispatch
            async def execute_scheduled(self, execution):
                raise AssertionError("prepared path required")
        executor = PreparedExecutor()
    runner = SchedulerRunner(db, "stalled-renewal", executor, owner_id="stalled-owner", lease_seconds=1)
    await runner._ensure_tables()
    due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    await db.execute("""INSERT INTO scheduled_tasks
        (id, agent_id, task_name, cron_expression, enabled, next_run_at, created_at, idempotency_key, scheduler_protocol_version)
        VALUES ('stalled', 'stalled-renewal', 'effect', '* * * * *', 1, ?, ?, 'stalled-base', ?)""", (due, due, SCHEDULER_PROTOCOL_VERSION))
    task = ScheduledTask.from_row((await runner._due_rows(datetime.now(timezone.utc)))[0])
    claimed = await runner._claim(task, datetime.now(timezone.utc))
    renew_once = runner._renew_lease_once
    attempts = []
    async def stalled_renewal(task):
        attempts.append(True)
        if len(attempts) == 1:
            return await renew_once(task)
        stalled.set()
        await asyncio.Event().wait()  # Bounded test owner always joins this worker.
    runner._renew_lease_once = stalled_renewal
    pending = asyncio.create_task(runner._execute_claim(claimed))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.wait_for(stalled.wait(), timeout=1)
        await asyncio.wait_for(cancelled.wait(), timeout=1.5)
        await asyncio.wait_for(pending, timeout=1)
        assert not effects
        assert await db.fetchval("SELECT status FROM task_execution_log WHERE id=?", (claimed.claim_execution_id,)) == "executing"
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
