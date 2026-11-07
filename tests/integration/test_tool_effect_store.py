"""Actual native database checks for operation admission and effect claims."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
import os
from types import SimpleNamespace

from cryptography.fernet import Fernet
import pytest
import pytest_asyncio

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.tool_effect_store import (
    ToolEffectAttemptClosed, ToolEffectAttemptSuperseded, ToolEffectClaimConflict, ToolEffectOperationConflict,
    ToolEffectIntegrityError, ToolEffectReconciliationRequired, ToolEffectStore,
)
from kestrel_sdk.storage.database.interface import TransactionError
from kestrel_sdk.resources import ResourceConflict


@pytest_asyncio.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.native_postgres)])
async def journal(tmp_path, monkeypatch, request):
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())

    async def no_core_schema(db):
        pass

    async with AsyncExitStack() as cleanup:
        if request.param == "postgres":
            from tests.utils.postgres_schema import (
                disposable_postgres_schema, postgres_test_url, with_search_path,
            )
            dsn = postgres_test_url()
            if not dsn:
                pytest.skip("PostgreSQL test URL is not configured")
            admin = await AsyncDatabase.postgres(dsn, min_pool_size=1, max_pool_size=2, schema_initializer=no_core_schema)
            cleanup.push_async_callback(admin.close)
            schema = await cleanup.enter_async_context(disposable_postgres_schema(admin, "tool_effect_test"))

            async def reopen():
                return await AsyncDatabase.postgres(with_search_path(dsn, schema), min_pool_size=1, max_pool_size=2, schema_initializer=no_core_schema)
        else:
            async def reopen():
                return await AsyncDatabase.sqlite(str(tmp_path / "journal.db"), schema_initializer=no_core_schema)

        db = await reopen()
        cleanup.push_async_callback(db.close)
        store = ToolEffectStore(db, scope=NativeRecipientScope("did:test:owned-journal", "tenant"))
        await store.initialize()
        yield store, reopen, cleanup


async def admit(store, *, key="owned-operation", request=None, namespace=None):
    return await store.admit(
        namespace=["caller", "user", "test", "owned-person"] if namespace is None else namespace,
        client_key=key, request={"input": "owned-private-input", "route": "chat"} if request is None else request,
    )


@pytest.mark.asyncio
async def test_observe_actual_postgres_journal_catalog(journal):
    import json
    from kestrel_sovereign.storage.db.tool_effect_relation import _tool_effect_program_identity

    store, _, _ = journal
    if not store.is_postgres:
        pytest.skip('Actual PostgreSQL catalog observation')
    observed = []
    async with store.backend._pool.acquire() as connection:
        relation = await connection.fetchrow("SELECT c.oid::bigint AS relation_oid, c.relnamespace::bigint AS namespace_oid FROM pg_catalog.pg_class c WHERE c.oid=pg_catalog.to_regclass('tool_effect_operations')")
        class ObservedCatalog:
            async def fetch(self, query, *arguments):
                rows = await connection.fetch(query, *arguments)
                observed.append([dict(row) for row in rows])
                return rows
        identity = await _tool_effect_program_identity(ObservedCatalog(), dict(relation))
    if identity is None:
        pytest.fail('ACTUAL_NATIVE_CATALOG=' + json.dumps(observed,
            default=lambda value: {'bytes_hex': value.hex()} if isinstance(value, bytes) else str(value)), pytrace=False)


@pytest.mark.asyncio
async def test_native_terminal_closes_original_without_admitting_cleanup_work(journal):
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError, ExecutionCustody, ProcessRuntimeExecutionFence,
        bind_execution_cleanup,
    )

    store, _, _ = journal
    first = await admit(store)
    successor = await admit(store)
    scope = ExecutionCustody(ProcessRuntimeExecutionFence('owned-native-terminal'))
    with bind_execution_cleanup(store.database, (scope,)):
        with pytest.raises(ToolEffectAttemptSuperseded):
            await store.finish(first)
        await store.finish(successor)
        with pytest.raises(ExecutionAuthorityError, match='cleanup-only'):
            await store.claim(successor, registration='forbidden', arguments={})
    assert await store.database.fetchone('SELECT attempt_id,state FROM tool_effect_operations') == (successor.attempt_id, 'CLOSED')
    assert await store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)


@pytest.mark.asyncio
@pytest.mark.parametrize('table', ['tool_effect_operations', 'tool_effect_claims'])
async def test_native_terminal_refuses_added_program_before_effect(journal, table):
    from kestrel_sovereign.execution_custody import is_execution_control_error

    store, _, _ = journal
    attempt = await admit(store)
    db = store.database
    await db.execute('CREATE TABLE terminal_effect_witness(value BIGINT)')
    if store.is_sqlite:
        await db.execute(f'CREATE TRIGGER terminal_effect BEFORE UPDATE ON {table} BEGIN INSERT INTO terminal_effect_witness VALUES(1); END')
    else:
        await db.execute('CREATE FUNCTION terminal_effect() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN INSERT INTO terminal_effect_witness VALUES(1); RETURN NEW; END $$')
        await db.execute(f'CREATE TRIGGER terminal_effect BEFORE UPDATE ON {table} FOR EACH ROW EXECUTE FUNCTION terminal_effect()')
    with pytest.raises(BaseExceptionGroup) as denied:
        await store.finish(attempt)
    assert is_execution_control_error(denied.value)
    assert await db.fetchone('SELECT COUNT(*) FROM terminal_effect_witness') == (0,)
    assert await db.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN',)


@pytest.mark.asyncio
async def test_terminal_deadline_retires_original_when_postgres_pool_is_full(journal):
    store, _, _ = journal
    if not store.is_postgres:
        pytest.skip('Native PostgreSQL pool checkout contract')
    store.terminal_timeout_seconds = 0.1
    attempt = await admit(store)
    await store.claim(attempt, registration='owned-before-pool-exhaustion', arguments={})
    async with AsyncExitStack() as holds:
        for _ in range(store.backend._pool.get_max_size()):
            await holds.enter_async_context(store.backend._pool.acquire())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(store.finish(attempt), timeout=2)
        assert not store.backend._original_result_holds
    assert await store.database.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN',)
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(store)


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup_only', [False, True])
async def test_sqlite_auxiliary_generation_uses_the_original_primary_admission(journal, cleanup_only):
    from contextlib import nullcontext
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody, ProcessRuntimeExecutionFence, bind_execution_cleanup,
        is_execution_control_error,
    )

    store, reopen, cleanup = journal
    if not store.is_sqlite:
        pytest.skip('SQLite native data_version contract')
    await store.database.execute('CREATE TABLE unrelated_native_witness(value BIGINT)')
    other = await reopen()
    cleanup.push_async_callback(other.close)
    attempt = await admit(store)
    await other.execute('INSERT INTO unrelated_native_witness VALUES(1)')
    scope = ExecutionCustody(ProcessRuntimeExecutionFence('owned-generation-check'))
    with bind_execution_cleanup(store.database, (scope,)) if cleanup_only else nullcontext():
        if cleanup_only:
            with pytest.raises(BaseExceptionGroup) as refused:
                await store.finish(attempt)
            assert is_execution_control_error(refused.value)
        else:
            await store.finish(attempt)
    assert await store.database.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN' if cleanup_only else 'CLOSED',)
    assert await store.database.fetchone('SELECT COUNT(*) FROM unrelated_native_witness') == (1,)


@pytest.mark.asyncio
async def test_sqlite_terminal_deadline_joins_native_lock_retirement(journal):
    store, reopen, cleanup = journal
    if not store.is_sqlite:
        pytest.skip('SQLite native lock retirement contract')
    store.terminal_timeout_seconds = 0.03
    other = await reopen()
    cleanup.push_async_callback(other.close)
    attempt = await admit(store)
    await store.claim(attempt, registration='owned-before-lock', arguments={})
    finishing = None
    try:
        async with other.backend.transaction():
            await other.execute('UPDATE tool_effect_operations SET revision = revision + 1')
            finishing = asyncio.create_task(store.finish(attempt))
            await asyncio.sleep(0.1)
            # The budget expired, but the original SQLite worker is still
            # physically queued behind this native lock. No early retirement.
            assert not finishing.done()
            assert store.backend._original_result_holds
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(finishing, timeout=2)
        assert not store.backend._original_result_holds
        assert await store.database.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN',)
        with pytest.raises(ToolEffectReconciliationRequired):
            await admit(store)
    finally:
        if finishing is not None:
            await asyncio.gather(finishing, return_exceptions=True)


@pytest.mark.asyncio
async def test_replay_after_actual_effect_and_clean_database_restart_is_fenced(journal, tmp_path):
    store, reopen, cleanup = journal
    attempt = await admit(store)
    claim = await store.claim(attempt, registration=["mcp", "owned-connector", "record_marker"], arguments={"marker": "owned-private-marker"})
    witness = tmp_path / "effect.txt"
    witness.write_text("one actual effect")
    await store.settle(claim, state="UNCONFIRMED")
    await store.finish(attempt)
    await store.database.close()
    db = await reopen()
    cleanup.push_async_callback(db.close)
    restarted = ToolEffectStore(db, scope=NativeRecipientScope("did:test:owned-journal", "tenant"))
    await restarted.initialize()
    with pytest.raises(ToolEffectReconciliationRequired) as error:
        await admit(restarted)
    assert error.value.counts == dict(returned_ok=0, returned_error=0, unconfirmed=1, unsettled=0, not_executed=0)
    assert witness.read_text() == "one actual effect"
    fresh = await admit(restarted, key="new-intent")
    await restarted.claim(fresh, registration=["mcp", "owned-connector", "record_marker"], arguments={"marker": "owned-private-marker"})
    # Only keyed digests and server identifiers are stored, never effect content.
    for table in ("tool_effect_operations", "tool_effect_claims"):
        persisted = repr(await db.fetchall(f"SELECT * FROM {table}"))
        for private in ("owned-private-input", "owned-private-marker", "record_marker", "owned-person", "owned-operation", "owned-connector"):
            assert private not in persisted


@pytest.mark.asyncio
async def test_zero_effect_supersession_fences_the_old_owner_and_changed_input_conflicts(journal):
    store, _, _ = journal
    old = await admit(store)
    current = await admit(store)
    assert old.attempt_id != current.attempt_id
    with pytest.raises(ToolEffectAttemptSuperseded):
        await store.claim(old, registration="owned-tool", arguments={})
    with pytest.raises(ToolEffectOperationConflict):
        await admit(store, request={"input": "different"})
    await store.claim(current, registration="owned-tool", arguments={})


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["RETURNED_OK", "RETURNED_ERROR", "UNCONFIRMED", "STARTED"])
async def test_every_effect_capable_claim_blocks_transport_replay(journal, terminal):
    store, _, _ = journal
    attempt = await admit(store)
    claim = await store.claim(attempt, registration="owned-tool", arguments={})
    if terminal != "STARTED":
        await store.settle(claim, state=terminal)
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(store)


@pytest.mark.asyncio
async def test_not_executed_allows_new_attempt_but_terminal_claim_cannot_be_reopened(journal):
    store, _, _ = journal
    first = await admit(store)
    claim = await store.claim(first, registration="owned-tool", arguments={})
    await store.settle(claim, state="NOT_EXECUTED")
    with pytest.raises(ToolEffectClaimConflict):
        await store.settle(claim, state="RETURNED_OK")
    second = await admit(store)
    with pytest.raises(ToolEffectAttemptSuperseded):
        await store.claim(first, registration="owned-tool", arguments={})
    await store.claim(second, registration="owned-tool", arguments={})


@pytest.mark.asyncio
async def test_unknown_latch_stops_new_claims_while_original_parallel_claim_can_settle(journal):
    store, _, _ = journal
    attempt = await admit(store)
    first = await store.claim(attempt, registration="owned-first", arguments={})
    second = await store.claim(attempt, registration="owned-second", arguments={})
    await store.settle(first, state="UNCONFIRMED")
    with pytest.raises(ToolEffectAttemptClosed):
        await store.claim(attempt, registration="owned-third", arguments={})
    await store.settle(second, state="RETURNED_OK")
    with pytest.raises(ToolEffectReconciliationRequired) as error:
        await admit(store)
    assert error.value.counts["returned_ok"] == error.value.counts["unconfirmed"] == 1


@pytest.mark.asyncio
async def test_independent_connections_serialize_supersession_against_claim(journal):
    first, reopen, cleanup = journal
    db = await reopen()
    cleanup.push_async_callback(db.close)
    second = ToolEffectStore(db, scope=NativeRecipientScope("did:test:owned-journal", "tenant"))
    await second.initialize()
    original = await admit(first)
    results = await asyncio.gather(
        first.claim(original, registration="owned-tool", arguments={}),
        admit(second), return_exceptions=True,
    )
    # Either the original claim commits first, or the new attempt wins. Both
    # owners must never obtain acknowledged admission for that logical effect.
    assert sum(isinstance(value, (ToolEffectAttemptSuperseded, ToolEffectReconciliationRequired)) for value in results) == 1
    if not isinstance(results[1], Exception):
        await second.claim(results[1], registration="owned-tool", arguments={})
    count, = await db.fetchone("SELECT COUNT(*) FROM tool_effect_claims")
    assert count == 1


@pytest.mark.asyncio
async def test_borrowed_store_close_leaves_primary_database_usable(journal):
    store, _, _ = journal
    assert store.initialized is True
    await store.close()
    assert store.initialized is False
    assert await store.database.fetchone("SELECT 1") == (1,)


@pytest.mark.asyncio
async def test_closed_attempt_cannot_erase_an_unsettled_effect_as_not_executed(journal):
    store, _, _ = journal
    attempt = await admit(store)
    claim = await store.claim(attempt, registration="owned-tool", arguments={})
    await store.finish(attempt)
    with pytest.raises(ToolEffectAttemptClosed):
        await store.settle(claim, state="NOT_EXECUTED")
    with pytest.raises(ToolEffectReconciliationRequired) as error:
        await admit(store)
    assert error.value.counts["unsettled"] == 1


@pytest.mark.asyncio
async def test_missing_root_with_operations_never_creates_a_replacement(journal):
    store, _, _ = journal
    await admit(store)
    await store.database.execute("DELETE FROM tool_effect_keys WHERE scope_digest = ?", (store.scope_digest,))
    restarted = ToolEffectStore(store.database, scope=NativeRecipientScope("did:test:owned-journal", "tenant"))
    with pytest.raises(ToolEffectIntegrityError, match="original tool effect envelope key is missing"):
        await restarted.initialize()
    assert restarted._root is None
    assert await store.database.fetchone("SELECT COUNT(*) FROM tool_effect_keys") == (0,)


@pytest.mark.asyncio
async def test_wrong_wrapping_key_preserves_the_original_envelope(journal, monkeypatch):
    store, _, _ = journal
    before = await store.database.fetchone("SELECT wrapped_key FROM tool_effect_keys")
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    restarted = ToolEffectStore(store.database, scope=NativeRecipientScope("did:test:owned-journal", "tenant"))
    with pytest.raises(ToolEffectIntegrityError, match="restore its original KESTREL_DATA_KEY"):
        await restarted.initialize()
    assert restarted._root is None
    assert await store.database.fetchone("SELECT wrapped_key FROM tool_effect_keys") == before


@pytest.mark.asyncio
async def test_agent_scopes_and_authenticated_caller_namespaces_are_distinct(journal):
    store, _, _ = journal
    original = await admit(store)
    await store.claim(original, registration="owned-tool", arguments={})
    other_caller = await admit(store, namespace=["caller", "user", "test", "another-owned-person"])
    assert original.operation_digest != other_caller.operation_digest
    other_scope = ToolEffectStore(store.database, scope=NativeRecipientScope("did:test:other-owned-journal", "tenant"))
    await other_scope.initialize()
    other_attempt = await admit(other_scope)
    assert original.scope_digest != other_attempt.scope_digest
    with pytest.raises(ResourceConflict):
        await other_scope.claim(original, registration="owned-tool", arguments={})
    keys = await store.database.fetchall("SELECT key_id FROM tool_effect_keys")
    assert len(keys) == len(set(row[0] for row in keys)) == 2


@pytest.mark.asyncio
async def test_real_master_key_rotation_keeps_replay_fencing_across_restart(journal, monkeypatch):
    from kestrel_sovereign.security.key_rotation import (
        KeyRotationService, RotationRecord, RotationStatus, _master_bytes_from_key,
    )

    store, reopen, cleanup = journal
    attempt = await admit(store)
    claim = await store.claim(attempt, registration="owned-tool", arguments={})
    await store.settle(claim, state="RETURNED_OK")
    # Exercise distinct globally unique key IDs in the actual rotation registry.
    other = ToolEffectStore(store.database, scope=NativeRecipientScope("did:test:other-owned-journal", "tenant"))
    await other.initialize()
    old_key = os.environ["KESTREL_DATA_KEY"]
    new_key = Fernet.generate_key().decode()
    db = store.database
    service = KeyRotationService(SimpleNamespace(database=db))
    await service.initialize()
    cleanup.push_async_callback(service.shutdown)
    await db.execute("CREATE TABLE conversation_history (id INTEGER PRIMARY KEY, content TEXT, rendered_content TEXT, agent_id TEXT, metadata TEXT)")
    binary = "BYTEA" if store.is_postgres else "BLOB"
    await db.execute(f"CREATE TABLE files (content_hash TEXT PRIMARY KEY, content {binary}, metadata TEXT)")
    rotation = RotationRecord(id="owned-journal-rotation", started_at=datetime.now(UTC), old_key_hash="old", new_key_hash="new", status=RotationStatus.IN_PROGRESS)
    await service._execute_rotation(rotation, _master_bytes_from_key(old_key), _master_bytes_from_key(new_key))
    assert rotation.status == RotationStatus.COMPLETED
    assert rotation.records_processed == 2
    monkeypatch.setenv("KESTREL_DATA_KEY", new_key)
    reopened = await reopen()
    cleanup.push_async_callback(reopened.close)
    restored = ToolEffectStore(reopened, scope=NativeRecipientScope("did:test:owned-journal", "tenant"))
    await restored.initialize()
    with pytest.raises(ToolEffectReconciliationRequired) as error:
        await admit(restored)
    assert error.value.counts["returned_ok"] == 1
    await other.close()
    assert await db.fetchone("SELECT 1") == (1,)


@pytest.mark.asyncio
async def test_uncommitted_caller_transaction_cannot_receive_claim_admission(journal):
    store, _, _ = journal
    with pytest.raises(TransactionError) as error:
        async with store.database.transaction():
            await admit(store)
    assert isinstance(error.value.__cause__, ResourceConflict)
    assert await store.database.fetchone("SELECT COUNT(*) FROM tool_effect_operations") == (0,)


@pytest.mark.asyncio
async def test_noncanonical_arguments_fail_before_a_claim_exists(journal):
    import rfc8785

    store, _, _ = journal
    attempt = await admit(store)
    with pytest.raises(rfc8785.CanonicalizationError):
        await store.claim(attempt, registration="owned-tool", arguments={"invalid": float("nan")})
    assert await store.database.fetchone("SELECT COUNT(*) FROM tool_effect_claims") == (0,)


@pytest.mark.asyncio
async def test_lost_claim_commit_ack_never_returns_admission_or_runs_effect(journal, monkeypatch, tmp_path):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError, execution_commit_outcome

    store, _, _ = journal
    attempt = await admit(store)
    original = store.backend.transaction

    @asynccontextmanager
    async def lost_ack(**options):
        async with original(**options):
            yield
        # The real native transaction committed. Lose only its acknowledgement.
        raise ExecutionCommitOutcomeError("unknown")

    witness = tmp_path / "not-admitted.txt"
    with monkeypatch.context() as patch:
        patch.setattr(store.backend, "transaction", lost_ack)
        with pytest.raises(ExecutionCommitOutcomeError) as error:
            await store.claim(attempt, registration="owned-tool", arguments={})
            witness.write_text("must never execute")
        assert execution_commit_outcome(error.value) == "unknown"
    assert not witness.exists()
    with pytest.raises(ToolEffectReconciliationRequired) as error:
        await admit(store)
    assert error.value.counts["unsettled"] == 1


@pytest.mark.asyncio
async def test_sql_refusal_rolls_back_claim_and_does_not_dispatch(journal, tmp_path):
    store, _, _ = journal
    attempt = await admit(store)
    db = store.database
    if store.is_sqlite:
        await db.execute("CREATE TRIGGER refuse_claim BEFORE INSERT ON tool_effect_claims BEGIN SELECT RAISE(ABORT, 'owned claim refusal'); END")
    else:
        await db.execute("CREATE FUNCTION refuse_claim() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'owned claim refusal'; END $$")
        await db.execute("CREATE TRIGGER refuse_claim BEFORE INSERT ON tool_effect_claims FOR EACH ROW EXECUTE FUNCTION refuse_claim()")
    witness = tmp_path / "not-dispatched.txt"
    with pytest.raises(TransactionError, match="owned claim refusal"):
        await store.claim(attempt, registration="owned-tool", arguments={})
        witness.write_text("must never execute")
    assert not witness.exists()
    assert await db.fetchone("SELECT COUNT(*) FROM tool_effect_claims") == (0,)
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    with pytest.raises(ExecutionAuthorityError):
        await admit(store)
    await db.execute('DROP TRIGGER refuse_claim' if store.is_sqlite
        else 'DROP TRIGGER refuse_claim ON tool_effect_claims')
    replacement = await admit(store)
    assert replacement.attempt_id != attempt.attempt_id


@pytest.mark.asyncio
async def test_unavailable_encryption_fails_without_changing_original_rows(journal, monkeypatch):
    store, _, _ = journal
    before = await store.database.fetchall("SELECT * FROM tool_effect_keys")
    monkeypatch.delenv("KESTREL_DATA_KEY")
    restarted = ToolEffectStore(store.database, scope=NativeRecipientScope("did:test:owned-journal", "tenant"))
    with pytest.raises(ToolEffectIntegrityError, match="requires encryption"):
        await restarted.initialize()
    assert restarted._root is None
    assert await store.database.fetchall("SELECT * FROM tool_effect_keys") == before


@pytest.mark.asyncio
async def test_unknown_schema_revision_is_not_migrated_or_marked_initialized(journal):
    store, _, _ = journal
    await store.database.execute("UPDATE tool_effect_schema SET revision = 2 WHERE singleton = 1")
    restarted = ToolEffectStore(store.database, scope=NativeRecipientScope("did:test:owned-journal", "tenant"))
    with pytest.raises(ToolEffectIntegrityError, match="unsupported tool effect schema revision"):
        await restarted.initialize()
    assert restarted._root is None
    assert await store.database.fetchone("SELECT revision FROM tool_effect_schema WHERE singleton = 1") == (2,)
