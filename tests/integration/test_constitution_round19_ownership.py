"""Native regressions for boot creation, target custody, and inception ownership."""
import asyncio
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kestrel_sovereign import inception_service
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage import AsyncStorage, GraphNode
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.interface import ConnectionError, QueryError, TransactionError
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("mode", ["normal", "ephemeral"])
async def test_boot_compare_create_retains_concurrent_terminal_identity(db_backend, monkeypatch, mode):
    identity = "did:test:round19-boot:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    governed = PrivacyEnforcingStorage(storage, mode)
    winner = GraphNode(node_id=identity, node_type="agent", label="Actual inception winner", properties={
        "constitution_hash": "a" * 64,
        "genesis_audit": {"status": "failed", "audited": True, "risk_level": 3, "constitution_hash": "a" * 64},
        "constitution_reanchor": {"old_hash": "b" * 64, "new_hash": "a" * 64, "signed_artifact_hash": "c" * 64},
        "constitution_reanchor_history": [{"old_hash": "none", "new_hash": "b" * 64, "signed_artifact_hash": "d" * 64}],
    })
    original_get = governed.get_node
    absent_reads = 0

    async def concurrently_published(node_id):
        nonlocal absent_reads
        row = await original_get(node_id)
        if row is None and absent_reads == 0:
            absent_reads += 1
            # Only the schedule is controlled. The winning row and ownership
            # are published through the real native graph writer.
            await storage.add_node(winner)
        return row

    monkeypatch.setattr(governed, "get_node", concurrently_published)
    agent = SimpleNamespace(agent_id=identity, identity=None, storage=governed)
    agent._refuse_if_birth_record_in_another_database = lambda: KestrelAgent._refuse_if_birth_record_in_another_database(agent)
    try:
        returned = await KestrelAgent._ensure_agent_node_present(agent)
        actual = await original_get(identity)
        assert actual.properties == winner.properties
        assert actual.label == winner.label
        assert returned.properties == winner.properties
        assert returned.label == winner.label
        assert absent_reads == 1
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [None, "did:web:round19.invalid:separate"])
@pytest.mark.parametrize("boundary", ["before-acquire", "after-acquire"])
async def test_separate_sqlite_target_fenced_without_local_anchor(tmp_path, identity, boundary):
    target_path = tmp_path / "target.db"
    displaced = tmp_path / "displaced.db"
    output = tmp_path / "identity"
    output.mkdir()
    database = await AsyncDatabase.sqlite(str(target_path))
    replacement = b"unrelated replacement bytes"
    def replace():
        target_path.rename(displaced)
        target_path.write_bytes(replacement)
    try:
        if boundary == "before-acquire":
            replace()
        with pytest.raises((ConnectionError, TransactionError), match="connected database"):
            async with inception_service._inception_publication_custody(database, output, identity):
                if boundary == "after-acquire":
                    replace()
                assert await database.fetchone("SELECT 1") == (1,)
        assert target_path.read_bytes() == replacement
        assert displaced.exists()
        # Native target is borrowed, never closed by the custody context.
        assert database.backend._connection is not None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_inception_rechecks_separate_target_before_active_key_publication(tmp_path, monkeypatch):
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
    target = tmp_path / "target.db"
    displaced = tmp_path / "displaced.db"
    output = tmp_path / "identity"
    database = await AsyncDatabase.sqlite(str(target))
    original_edge = AsyncGraphStore.add_edge
    replacement = b"unrelated target must survive"
    changed = False

    async def published_edge_then_displaced(self, *args, **kwargs):
        nonlocal changed
        result = await original_edge(self, *args, **kwargs)
        if args[2] == "governed_by" and not changed:
            changed = True
            target.rename(displaced)
            target.write_bytes(replacement)
        return result

    monkeypatch.setenv("KESTREL_DATA_KEY", "round19-native-test-only-key")
    monkeypatch.setattr(AsyncGraphStore, "add_edge", published_edge_then_displaced)
    try:
        with pytest.raises((ConnectionError, TransactionError), match="connected database"):
            await inception_service.create_kestrel_identity_async(
                output_dir=str(output), database=database, is_test_instance=True,
                identity_method="did:web", did_web_domain="round19.invalid",
                did_web_slug="separate-" + uuid4().hex,
            )
        assert changed
        assert target.read_bytes() == replacement
        assert not list(output.glob("*.key.enc"))
        assert not list(output.glob("*.pem"))
        assert not list(output.glob("*.json"))
        assert (await database.fetchone("SELECT count(*) FROM graph_nodes WHERE node_type='agent'"))[0] == 0
        assert database.backend._connection is not None
    finally:
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("failure", ["cancel", "sql"])
async def test_post_commit_inception_failure_settles_only_owned_database(tmp_path, monkeypatch, borrowed, failure):
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
    original_sqlite = AsyncDatabase.sqlite
    captured = []
    reached, release = asyncio.Event(), asyncio.Event()
    task = None

    async def observe_acquisition(*args, **kwargs):
        db = await original_sqlite(*args, **kwargs)
        captured.append((db, db.backend._connection))
        return db

    async def interrupted_native_index(self, *args, **kwargs):
        assert (await self.db.fetchone("SELECT count(*) FROM graph_nodes WHERE node_type='agent'"))[0] == 1
        assert (await self.db.fetchone("SELECT count(*) FROM graph_edges WHERE label='governed_by'"))[0] == 1
        reached.set()
        if failure == "sql":
            await self.db.execute("INSERT INTO nonexistent_post_commit_table VALUES (?)", (1,))
        await release.wait()
        return 0

    monkeypatch.setenv("KESTREL_DATA_KEY", "round19-native-test-only-key")
    monkeypatch.setattr(AsyncDatabase, "sqlite", observe_acquisition)
    monkeypatch.setattr(AsyncRAGStore, "chunk_document", interrupted_native_index)
    supplied = await observe_acquisition(str(tmp_path / "borrowed.db")) if borrowed else None
    output = tmp_path / "identity"
    try:
        task = asyncio.create_task(inception_service.create_kestrel_identity_async(
            output_dir=str(output), database=supplied, is_test_instance=True,
            identity_method="did:web", did_web_domain="round19.invalid",
            did_web_slug="lifetime-" + uuid4().hex,
        ))
        await asyncio.wait_for(reached.wait(), 5)
        if failure == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        else:
            with pytest.raises(QueryError):
                await asyncio.wait_for(task, 5)
        assert len(captured) == 1
        db, connection = captured[0]
        if borrowed:
            assert db.backend._connection is connection
            assert getattr(connection, "_thread", connection).is_alive()
            inspection = db
        else:
            assert db.backend._connection is None
            assert not db.connection_retirement_pending
            assert not getattr(connection, "_thread", connection).is_alive()
            inspection = await original_sqlite(str(output / "kestrel_prime.db"))
        try:
            assert (await inspection.fetchone("SELECT count(*) FROM graph_nodes WHERE node_type='agent'"))[0] == 1
            assert (await inspection.fetchone("SELECT count(*) FROM graph_edges WHERE label='governed_by'"))[0] == 1
        finally:
            if not borrowed:
                await inspection.close()
        assert list(output.glob("*.key.enc"))
        assert list(output.glob("*.json"))
    finally:
        release.set()
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, QueryError):
                await task
        for db, connection in captured:
            await db.close()
            assert not getattr(connection, "_thread", connection).is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["repeated-cancellation", "uncertain-close"])
async def test_post_commit_retirement_retains_custody_and_failure_owner(tmp_path, monkeypatch, failure):
    from kestrel_sovereign.storage.async_database import DatabaseRetirementError
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore
    from kestrel_sovereign.private_storage import PrivateStorageError, exclusive_private_file_lock

    original_sqlite = AsyncDatabase.sqlite
    captured = []
    closing, release = asyncio.Event(), asyncio.Event()
    task = None
    output = tmp_path / "identity"

    async def observe_owned_connection(*args, **kwargs):
        db = await original_sqlite(*args, **kwargs)
        connection, real_close = db.backend._connection, db.close
        captured.append((db, connection, real_close))
        async def controlled_close():
            closing.set()
            await release.wait()
            if failure == "uncertain-close":
                raise RuntimeError("Injected missing native retirement acknowledgement")
            await real_close()
        monkeypatch.setattr(db, "close", controlled_close)
        return db

    async def native_post_commit_error(self, *args, **kwargs):
        assert (await self.db.fetchone("SELECT count(*) FROM graph_nodes WHERE node_type='agent'"))[0] == 1
        await self.db.execute("INSERT INTO nonexistent_post_commit_table VALUES (?)", (1,))

    monkeypatch.setenv("KESTREL_DATA_KEY", "round19-native-test-only-key")
    monkeypatch.setattr(AsyncDatabase, "sqlite", observe_owned_connection)
    monkeypatch.setattr(AsyncRAGStore, "chunk_document", native_post_commit_error)
    try:
        task = asyncio.create_task(inception_service.create_kestrel_identity_async(
            output_dir=str(output), is_test_instance=True, identity_method="did:web",
            did_web_domain="round19.invalid", did_web_slug="retirement-" + uuid4().hex,
        ))
        await asyncio.wait_for(closing.wait(), 5)
        with pytest.raises(PrivateStorageError):
            with exclusive_private_file_lock(output / ".inception.lock", label="competing-native-creator", blocking=False):
                pytest.fail("Directory custody released before owned retirement")
        if failure == "repeated-cancellation":
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done(), "Repeated cancellation abandoned native retirement"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            db, connection, _ = captured[0]
            assert db.backend._connection is None
            assert not db.connection_retirement_pending
            assert not getattr(connection, "_thread", connection).is_alive()
        else:
            release.set()
            with pytest.raises(DatabaseRetirementError) as raised:
                await asyncio.wait_for(task, 5)
            db, connection, _ = captured[0]
            assert raised.value.retirement_owner is db
            assert isinstance(raised.value.operation_error, QueryError)
            assert isinstance(raised.value.cleanup_error, RuntimeError)
            assert getattr(connection, "_thread", connection).is_alive()
        assert list(output.glob("*.key.enc"))
        assert list(output.glob("*.json"))
        assert (output / "kestrel_prime.db").exists()
    finally:
        release.set()
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, QueryError, DatabaseRetirementError):
                await task
        for db, connection, real_close in captured:
            await real_close()
            assert not getattr(connection, "_thread", connection).is_alive()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("mode", ["normal", "ephemeral"])
async def test_native_boot_compare_create_new_identity(db_backend, mode):
    identity = "did:test:round19-new-boot:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    wrapped = PrivacyEnforcingStorage(storage, mode)
    agent = SimpleNamespace(agent_id=identity, identity=None, storage=wrapped)
    agent._refuse_if_birth_record_in_another_database = lambda: KestrelAgent._refuse_if_birth_record_in_another_database(agent)
    try:
        created = await KestrelAgent._ensure_agent_node_present(agent)
        assert created.properties == {"initialBalance": "100.0"}
        assert created.label == f"Agent {identity}"
        assert (await storage.get_node(identity)) == created
        assert (await KestrelAgent._ensure_agent_node_present(agent)) == created
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("mode", ["normal", "ephemeral"])
async def test_native_boot_compare_create_refuses_foreign_owner(db_backend, mode):
    from kestrel_sovereign.identity.runtime_identity import IdentityReadinessError
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

    identity = "did:test:round19-foreign-boot:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    foreign = AsyncGraphStore(storage.db, agent_id="did:test:other:" + uuid4().hex)
    # A supported foreign concept may reserve this global node id. Root boot
    # must neither disclose it nor relabel it as its own provisional identity.
    winner = GraphNode(node_id=identity, node_type="concept", label="Other owner's record", properties={"private_metadata": "native other-owner record"})
    wrapped = PrivacyEnforcingStorage(storage, mode)
    agent = SimpleNamespace(agent_id=identity, identity=None, storage=wrapped)
    agent._refuse_if_birth_record_in_another_database = lambda: KestrelAgent._refuse_if_birth_record_in_another_database(agent)
    try:
        await foreign.add_node(winner)
        assert await storage.get_node(identity) is None
        with pytest.raises(IdentityReadinessError) as refused:
            await KestrelAgent._ensure_agent_node_present(agent)
        assert refused.value.failure == "birth_record"
        assert await foreign.get_node(identity) == winner
        assert await storage.get_node(identity) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_other_retirement_error_cannot_bypass_inception_database_owner(tmp_path, monkeypatch):
    from kestrel_sovereign.storage.async_database import DatabaseRetirementError, _close_owned_database
    from kestrel_sovereign.storage.async_rag_store import AsyncRAGStore

    original_sqlite = AsyncDatabase.sqlite
    other = await original_sqlite(str(tmp_path / "other-owner.db"))
    other_real_close = other.close
    captured = []

    async def observe_owned_connection(*args, **kwargs):
        db = await original_sqlite(*args, **kwargs)
        captured.append((db, db.backend._connection))
        return db

    async def other_close_without_acknowledgement():
        raise RuntimeError("Other owner's missing native retirement acknowledgement")

    async def unrelated_retirement_failure(self, *args, **kwargs):
        assert (await self.db.fetchone("SELECT count(*) FROM graph_nodes WHERE node_type='agent'"))[0] == 1
        await _close_owned_database(other, QueryError("Independent post-commit subsystem failure"))

    monkeypatch.setenv("KESTREL_DATA_KEY", "round19-native-test-only-key")
    monkeypatch.setattr(AsyncDatabase, "sqlite", observe_owned_connection)
    monkeypatch.setattr(other, "close", other_close_without_acknowledgement)
    monkeypatch.setattr(AsyncRAGStore, "chunk_document", unrelated_retirement_failure)
    try:
        with pytest.raises(DatabaseRetirementError) as failure:
            await inception_service.create_kestrel_identity_async(
                output_dir=str(tmp_path / "identity"), is_test_instance=True,
                identity_method="did:web", did_web_domain="round19.invalid",
                did_web_slug="other-owner-" + uuid4().hex,
            )
        assert failure.value.retirement_owner is other
        assert len(captured) == 1
        db, connection = captured[0]
        assert db.backend._connection is None
        assert not db.connection_retirement_pending
        assert not getattr(connection, "_thread", connection).is_alive()
        assert list((tmp_path / "identity").glob("*.key.enc"))
    finally:
        for db, connection in captured:
            await db.close()
            assert not getattr(connection, "_thread", connection).is_alive()
        await other_real_close()
