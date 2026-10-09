"""Native concurrent entry/anchor transactions retain one safety/lock order."""

import asyncio
import hashlib
from contextlib import suppress
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.boot import BootPhaseState
from kestrel_sovereign.agent.constitution import ConstitutionMixin, SafeModeCause
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from tests.integration.test_constitution_refusal_races import _agent
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files
from tests.utils.postgres_schema import (
    disposable_postgres_schema,
    pgvector_schema,
    quoted_search_path,
    with_search_path,
)


@pytest.fixture
async def fresh_anchor_schema(db_backend):
    if db_backend.backend_type != "postgres":
        yield None
        return

    async def no_schema_ddl(_db):
        pass

    admin = await AsyncDatabase.postgres(
        db_backend._dsn, schema_initializer=no_schema_ddl
    )
    try:
        vector_schema = await pgvector_schema(admin)
        async with disposable_postgres_schema(
            admin, "constitution_anchor_order"
        ) as schema:
            # Boot ONLY the owned schema first, so visible shared tables cannot
            # substitute for its empty file inventory. Then expose pgvector.
            boot = await AsyncDatabase.postgres(
                with_search_path(db_backend._dsn, schema)
            )
            await boot.close()
            yield with_search_path(
                db_backend._dsn, quoted_search_path(schema, vector_schema)
            )
    finally:
        await admin.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_consent_wait_cannot_downgrade_intervening_lifecycle_refusal(db_backend):
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:consent-refusal:" + uuid4().hex
    )
    await storage.initialize()
    entry_task = None
    proceed = asyncio.Event()
    try:
        agent = await _agent(storage)
        agent._boot_state = BootPhaseState.READY
        await agent._record_successful_constitution_audit(source="ready fixture")
        before = await agent._constitution_state_store.load(agent.agent_id)
        events_before = await agent._constitution_state_store.list_events(
            agent.agent_id
        )
        waiting = asyncio.Event()

        class ConsentBarrier:
            async def request_consent(self, *args, **kwargs):
                waiting.set()
                await proceed.wait()

        # Isolate the provider-bearing consent seam, never its transaction,
        # state lock, lifecycle publication, or actual native SQL behavior.
        agent.features["ConsentFeature"] = ConsentBarrier()
        entry_task = asyncio.create_task(
            agent.enter_safe_mode("earlier integrity reason")
        )
        await asyncio.wait_for(waiting.wait(), 5)
        async with storage.transaction():
            assert (
                await agent.enter_safe_mode(
                    "later feature quarantine",
                    cause=SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value,
                )
                is False
            )
        proceed.set()
        assert await asyncio.wait_for(entry_task, 5) is False
        assert agent._safe_mode_cause == SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value
        assert agent._safe_mode_reason == "later feature quarantine"
        assert agent._feature_lifecycle_integrity_uncertain is True
        assert agent._feature_lifecycle_repair_verified is False
        assert agent._constitution_state_persistence_pending is True
        assert await agent._constitution_state_store.load(agent.agent_id) == before
        assert (
            await agent._constitution_state_store.list_events(agent.agent_id)
            == events_before
        )
        # A fresh explicitly owned retry can persist the stronger restriction;
        # only real lifecycle repair can permit its subsequent authorized exit.
        assert await agent.enter_safe_mode(
            "persist feature quarantine",
            cause=SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value,
        )
        durable = await agent._constitution_state_store.load(agent.agent_id)
        assert (
            durable.safe_mode_cause == SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value
        )
        agent.verify_feature_lifecycle_integrity = lambda: False
        result = await agent.exit_safe_mode(authorization="test sovereign")
        assert "feature lifecycle repair verification failed" in result, result
    finally:
        proceed.set()
        if entry_task is not None and not entry_task.done():
            entry_task.cancel()
        if entry_task is not None:
            with suppress(asyncio.CancelledError):
                await entry_task
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_native_postgres_bootstrap_and_signed_writer_share_graph_before_file_order(
    db_backend,
    fresh_anchor_schema,
    tmp_path,
    monkeypatch,
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL cross-replica file/graph locking")
    identity = "did:test:anchor-lock-order:" + uuid4().hex
    first = AsyncStorage(backend="postgres", dsn=fresh_anchor_schema, agent_id=identity)
    second = AsyncStorage(
        backend="postgres", dsn=fresh_anchor_schema, agent_id=identity
    )
    await first.initialize()
    await second.initialize()
    signed_task = automatic_task = None
    allow_signed = asyncio.Event()
    try:
        signed = await _agent(first)
        automatic = await _agent(second)
        content = resolve_governing_constitution_bytes(None)
        digest = hashlib.sha256(content).hexdigest()
        # This run MUST start without this blob; a reused canonical file
        # conceals the insert-vs-graph deadlock the regression exercises.
        assert (
            await first._backend.fetch_one(
                "SELECT content_hash FROM files WHERE content_hash = ?", (digest,)
            )
            is None
        )
        await first.add_node(
            GraphNode(
                node_id=identity, node_type="agent", label="unanchored", properties={}
            )
        )
        artifact, root = _write_authority_files(tmp_path, content)
        signed._sovereign_trust_root_path = root
        blob_inserted = asyncio.Event()
        automatic_locking = asyncio.Event()
        native_store = AsyncFileStore.store_file
        native_fetch_all = type(second._backend).fetch_all
        automatic_pid = None

        async def paused_signed_store(files, data, name, *args, **kwargs):
            result = await native_store(files, data, name, *args, **kwargs)
            if files.db is first.db and data == content and not blob_inserted.is_set():
                # Actual uncommitted INSERT holds the file uniqueness lock.
                blob_inserted.set()
                await allow_signed.wait()
            return result

        async def observed_automatic_locks(backend, query, params=()):
            nonlocal automatic_pid
            if (
                backend is second._backend
                and asyncio.current_task() is automatic_task
                and "pg_advisory_xact_lock(lock_key)" in query
            ):
                automatic_pid = await backend.fetch_val("SELECT pg_backend_pid()")
                automatic_locking.set()
            return await native_fetch_all(backend, query, params)

        monkeypatch.setattr(AsyncFileStore, "store_file", paused_signed_store)
        monkeypatch.setattr(
            type(second._backend), "fetch_all", observed_automatic_locks
        )
        signed_task = asyncio.create_task(
            ConstitutionMixin.reanchor_constitution(
                signed, amendment_artifact_path=str(artifact)
            )
        )
        await asyncio.wait_for(blob_inserted.wait(), 5)
        automatic_task = asyncio.create_task(
            ConstitutionMixin._get_governing_constitution(automatic)
        )
        await asyncio.wait_for(automatic_locking.wait(), 5)
        async with asyncio.timeout(5):
            while True:
                row = await first._backend.fetch_one(
                    "SELECT wait_event_type FROM pg_stat_activity WHERE pid = ?",
                    (automatic_pid,),
                )
                if row is not None and row[0] == "Lock":
                    break
                await asyncio.sleep(0.02)
        allow_signed.set()
        signed_result, automatic_result = await asyncio.wait_for(
            asyncio.gather(signed_task, automatic_task), 5
        )
        assert "deadlock" not in (signed_result + automatic_result).lower(), (
            signed_result,
            automatic_result,
        )
        assert not signed_result.startswith("Error:"), signed_result
        # Signed repair won the fence. Bootstrap must refuse stale custody,
        # not retry by adopting that writer's new generation/revision.
        assert automatic_result.startswith("Error:"), automatic_result
        assert any(
            reason in automatic_result
            for reason in ("anchor changed", "custody changed")
        ), automatic_result
        assert (await first.get_node(identity)).properties[
            "constitution_hash"
        ] == digest
        assert (await signed._verify_constitution_integrity())[0] is True
    finally:
        allow_signed.set()
        for task in (signed_task, automatic_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (signed_task, automatic_task):
            if task is not None:
                with suppress(asyncio.CancelledError):
                    await task
        await second.close()
        await first.close()
