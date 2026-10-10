"""Native proofs for refused key replacement, SQLite snapshots and history bounds."""

from copy import deepcopy
from contextlib import asynccontextmanager
from hashlib import sha256
import sqlite3
from uuid import uuid4
import os

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.storage import AsyncStorage, GraphNode
from tests.integration.test_constitution_refusal_races import _agent
from tests.integration.test_constitution_turn_admission import _ready_turn
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("deleted_root", [False, True])
async def test_refused_force_inception_preserves_active_identity_keys(db_backend, tmp_path, monkeypatch, deleted_root):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:force-fixture:" + uuid4().hex)
    await storage.initialize()
    directory = tmp_path / "active"
    slug = "retained-" + uuid4().hex
    kwargs = dict(database=storage.db, is_test_instance=True, identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test", did_web_slug=slug)
    try:
        first = await create_kestrel_identity_async(output_dir=str(directory), **kwargs)
        original = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        root = await storage.db.fetchone("SELECT properties FROM graph_nodes WHERE node_id=?", (first.agent_did,))
        if deleted_root:
            native = AsyncStorage(backend=db_backend, agent_id=first.agent_did)
            await native.initialize()
            agent = await _agent(native)
            assert await agent.enter_safe_mode("consumed identity lifetime")
            await storage.db.execute_commit("DELETE FROM graph_nodes WHERE node_id=?", (first.agent_did,))
        with pytest.raises(Exception, match="existing identity|existing constitutional lifetime|birth"):
            await create_kestrel_identity_async(output_dir=str(directory), force=True, **kwargs)
        assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == original
        assert await storage.db.fetchone("SELECT properties FROM graph_nodes WHERE node_id=?", (first.agent_did,)) == (None if deleted_root else root)
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_late_inception_refusal_never_publishes_staged_replacement_keys(db_backend, tmp_path, monkeypatch):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.identity.did_web import build_did

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:late-fixture:" + uuid4().hex)
    await storage.initialize()
    directory = tmp_path / "active"
    # Another active identity is explicitly replaceable with force, but a late
    # DB refusal must leave even that old identity's active files unchanged.
    initial = dict(database=storage.db, is_test_instance=True, identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test")
    try:
        await create_kestrel_identity_async(output_dir=str(directory), did_web_slug="old-" + uuid4().hex, **initial)
        original = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        slug = "late-" + uuid4().hex
        identity = build_did(initial["did_web_domain"], [slug])

        async def auditor(prompt):
            from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

            assert not storage.db.owns_open_transaction
            graph = AsyncGraphStore(storage.db, agent_id=identity)
            await graph.add_node(GraphNode(node_id=identity, node_type="agent", label="Concurrent winner", properties={"retained": True}))
            return {"risk_level": 1, "reasoning": "Synthetic late-commit native fixture"}

        with pytest.raises(Exception, match="existing identity"):
            await create_kestrel_identity_async(output_dir=str(directory), did_web_slug=slug, force=True, genesis_auditor=auditor, **initial)
        assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == original
        assert (await storage.db.fetchone("SELECT properties FROM graph_nodes WHERE node_id=?", (identity,))) is not None
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted_root", [False, True])
@pytest.mark.parametrize("remove_keys", [False, True])
async def test_owned_force_inception_retains_original_lifetime_and_keys(tmp_path, monkeypatch, deleted_root, remove_keys):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    directory = tmp_path / "owned"
    kwargs = dict(output_dir=str(directory), is_test_instance=True, identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test", did_web_slug="owned-retained")
    first = await create_kestrel_identity_async(**kwargs)
    storage = AsyncStorage(str(directory / "kestrel_prime.db"), backend="sqlite", agent_id=first.agent_did)
    await storage.initialize()
    try:
        agent = await _agent(storage)
        assert await agent.enter_safe_mode("retained native owned lifetime")
        if deleted_root:
            await storage.db.execute_commit("DELETE FROM graph_nodes WHERE node_id=?", (first.agent_did,))
        state = await storage.db.fetchone("SELECT * FROM constitution_runtime_state WHERE agent_id=?", (first.agent_did,))
        events = await storage.db.fetchall("SELECT * FROM constitution_runtime_events WHERE agent_id=? ORDER BY id", (first.agent_did,))
    finally:
        await storage.close()
    if remove_keys:
        from kestrel_sovereign.inception_service import _born_hybrid_identity_paths

        for path in _born_hybrid_identity_paths(directory, kwargs["did_web_slug"]):
            path.unlink()  # only synthetic fixture keys; isolate DB authority
    original_keys = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file() and p.name != "kestrel_prime.db"}
    with pytest.raises(Exception, match="existing identity|existing constitutional lifetime|birth|previous identity"):
        await create_kestrel_identity_async(force=True, **kwargs)
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file() and p.name != "kestrel_prime.db"} == original_keys
    assert not list(directory.glob("*.backup-*"))
    check = AsyncStorage(str(directory / "kestrel_prime.db"), backend="sqlite", agent_id=first.agent_did)
    await check.initialize()
    try:
        assert await check.db.fetchone("SELECT * FROM constitution_runtime_state WHERE agent_id=?", (first.agent_did,)) == state
        assert await check.db.fetchall("SELECT * FROM constitution_runtime_events WHERE agent_id=? ORDER BY id", (first.agent_did,)) == events
        root = await check.db.fetchone("SELECT node_id FROM graph_nodes WHERE node_id=?", (first.agent_did,))
        assert root == (None if deleted_root else (first.agent_did,))
    finally:
        await check.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_keys", [False, True])
@pytest.mark.parametrize("deleted_root", [False, True])
@pytest.mark.parametrize("reuse_external", [False, True])
async def test_owned_force_cannot_reuse_a_lifetime_from_an_archived_database(tmp_path, monkeypatch, remove_keys, deleted_root, reuse_external):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    directory = tmp_path / "chained"
    kwargs = dict(output_dir=str(directory), is_test_instance=True, identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test")
    first = await create_kestrel_identity_async(did_web_slug="old-lifetime", **kwargs)
    old = AsyncStorage(str(directory / "kestrel_prime.db"), backend="sqlite", agent_id=first.agent_did)
    await old.initialize()
    try:
        assert await (await _agent(old)).enter_safe_mode("retained archived native lifetime")
        if deleted_root:
            await old.db.execute_commit("DELETE FROM graph_nodes WHERE node_id=?", (first.agent_did,))
    finally:
        await old.close()
    if remove_keys:
        from kestrel_sovereign.inception_service import _born_hybrid_identity_paths

        for path in _born_hybrid_identity_paths(directory, "old-lifetime"):
            path.unlink()
    second = await create_kestrel_identity_async(did_web_slug="new-lifetime", force=True, **kwargs)
    assert second.agent_did != first.agent_did
    assert list(directory.glob("kestrel_prime.db.backup-*"))
    before = {p.name: sha256(p.read_bytes()).digest() for p in directory.iterdir() if p.is_file()}
    external = None
    if reuse_external:
        from kestrel_sovereign.storage.async_database import AsyncDatabase

        external = await AsyncDatabase.sqlite(str(tmp_path / "fresh-external.db"))
    try:
        with pytest.raises(Exception, match="previous identity|existing identity|constitutional lifetime|birth replay"):
            await create_kestrel_identity_async(did_web_slug="old-lifetime", force=True, database=external, **kwargs)
        if external is not None:
            assert await external.fetchone("SELECT node_id FROM graph_nodes WHERE node_id=?", (first.agent_did,)) is None
    finally:
        if external is not None:
            await external.close()
    assert {p.name: sha256(p.read_bytes()).digest() for p in directory.iterdir() if p.is_file()} == before


@pytest.mark.asyncio
async def test_owned_concurrent_creator_cannot_delete_the_committed_winner(tmp_path, monkeypatch):
    import asyncio
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    directory = tmp_path / "concurrent"
    kwargs = dict(output_dir=str(directory), is_test_instance=True, force=True,
                  identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test", did_web_slug="one-winner")
    native_sqlite = AsyncDatabase.sqlite
    first_opening, release_first, second_opening, winner_done = (asyncio.Event() for _ in range(4))
    opened = []
    calls = 0

    async def scheduled_open(path, **options):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_opening.set()
            await release_first.wait()
        else:
            second_opening.set()
            await winner_done.wait()
        db = await native_sqlite(path, **options)
        opened.append(db)
        return db

    monkeypatch.setattr(AsyncDatabase, "sqlite", scheduled_open)
    first_task = asyncio.create_task(create_kestrel_identity_async(**kwargs))
    second_task = second_signal = None
    try:
        await asyncio.wait_for(first_opening.wait(), 10)
        second_task = asyncio.create_task(create_kestrel_identity_async(**kwargs))
        second_signal = asyncio.create_task(second_opening.wait())
        await asyncio.wait_for(asyncio.wait([second_task, second_signal], return_when=asyncio.FIRST_COMPLETED), 10)
        release_first.set()
        winner = await asyncio.wait_for(first_task, 30)
        winner_done.set()
        with pytest.raises(Exception, match="existing identity|inception.*lock|cannot lock"):
            await asyncio.wait_for(second_task, 30)
        monkeypatch.setattr(AsyncDatabase, "sqlite", native_sqlite)
        assert (directory / "kestrel_prime.db").is_file(), "loser erased the winner's native SQLite database"
        check = await native_sqlite(str(directory / "kestrel_prime.db"))
        try:
            assert await check.fetchone("SELECT node_id FROM graph_nodes WHERE node_id=?", (winner.agent_did,)) == (winner.agent_did,)
            assert await check.fetchone("SELECT target_id FROM graph_edges WHERE source_id=? AND label='governed_by'", (winner.agent_did,))
        finally:
            await check.close()
    finally:
        release_first.set()
        winner_done.set()
        for task in (first_task, second_task, second_signal):
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        for db in opened:
            await db.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("deleted_root", [False, True])
async def test_external_target_cannot_bypass_current_local_consumed_lifetime(db_backend, tmp_path, monkeypatch, deleted_root):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async, _born_hybrid_identity_paths

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    directory = tmp_path / "current-local"
    slug = "retained-current-" + uuid4().hex
    kwargs = dict(output_dir=str(directory), is_test_instance=True, identity_method="did:web",
                  did_web_domain="agents.kestrel-sovereign.test", did_web_slug=slug)
    first = await create_kestrel_identity_async(**kwargs)
    original = AsyncStorage(str(directory / "kestrel_prime.db"), backend="sqlite", agent_id=first.agent_did)
    await original.initialize()
    try:
        assert await (await _agent(original)).enter_safe_mode("current local lifetime cannot be bypassed")
        if deleted_root:
            await original.db.execute_commit("DELETE FROM graph_nodes WHERE node_id=?", (first.agent_did,))
        state = await original.db.fetchone("SELECT * FROM constitution_runtime_state WHERE agent_id=?", (first.agent_did,))
        events = await original.db.fetchall("SELECT * FROM constitution_runtime_events WHERE agent_id=? ORDER BY id", (first.agent_did,))
    finally:
        await original.close()
    for path in _born_hybrid_identity_paths(directory, kwargs["did_web_slug"]):
        path.unlink()
    external = AsyncStorage(backend=db_backend, agent_id="did:test:fresh-external:" + uuid4().hex)
    await external.initialize()
    try:
        with pytest.raises(Exception, match="existing identity|constitutional lifetime|birth replay"):
            await create_kestrel_identity_async(database=external.db, force=True, **kwargs)
        assert await external.db.fetchone("SELECT node_id FROM graph_nodes WHERE node_id=?", (first.agent_did,)) is None
        assert not list(directory.glob(slug + "_*"))
    finally:
        await external.close()
    check = AsyncStorage(str(directory / "kestrel_prime.db"), backend="sqlite", agent_id=first.agent_did)
    await check.initialize()
    try:
        assert await check.db.fetchone("SELECT * FROM constitution_runtime_state WHERE agent_id=?", (first.agent_did,)) == state
        assert await check.db.fetchall("SELECT * FROM constitution_runtime_events WHERE agent_id=? ORDER BY id", (first.agent_did,)) == events
    finally:
        await check.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("deleted_root", [False, True])
async def test_external_publication_rechecks_current_local_lifetime_after_auditor(db_backend, tmp_path, monkeypatch, deleted_root):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.identity.did_web import build_did

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    directory = tmp_path / "late-current"
    directory.mkdir()
    slug = "late-current-" + uuid4().hex
    domain = "agents.kestrel-sovereign.test"
    identity = build_did(domain, [slug])
    local = AsyncStorage(str(directory / "kestrel_prime.db"), backend="sqlite", agent_id=identity)
    external = AsyncStorage(backend=db_backend, agent_id="did:test:external:" + uuid4().hex)
    await local.initialize()
    await external.initialize()
    audited = False

    async def auditor(prompt):
        nonlocal audited
        # Providers must not hold either database's native writer custody.
        assert not local.db.owns_open_transaction and not external.db.owns_open_transaction
        await local.add_node(GraphNode(node_id=identity, node_type="agent", label="Winner", properties={}))
        assert await (await _agent(local)).enter_safe_mode("late current-local consumed lifetime")
        if deleted_root:
            await local.db.execute_commit("DELETE FROM graph_nodes WHERE node_id=?", (identity,))
        audited = True
        return {"risk_level": 1, "reasoning": "Synthetic native competing lifetime"}

    try:
        with pytest.raises(Exception, match="existing identity|constitutional lifetime|birth replay"):
            await create_kestrel_identity_async(
                output_dir=str(directory), database=external.db, is_test_instance=True,
                identity_method="did:web", did_web_domain=domain, did_web_slug=slug,
                genesis_auditor=auditor,
            )
        assert audited
        assert await external.db.fetchone("SELECT node_id FROM graph_nodes WHERE node_id=?", (identity,)) is None
        assert await local.db.fetchone("SELECT agent_id FROM constitution_runtime_state WHERE agent_id=?", (identity,)) == (identity,)
        assert not list(directory.glob(slug + "_*"))
        assert not list(directory.glob(".inception-*"))
    finally:
        await external.close()
        await local.close()


@pytest.mark.asyncio
async def test_external_same_local_sqlite_target_joins_its_native_publication_custody(tmp_path, monkeypatch):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    directory = tmp_path / "same-local-target"
    directory.mkdir()
    db = await AsyncDatabase.sqlite(str(directory / "kestrel_prime.db"))
    try:
        result = await create_kestrel_identity_async(
            output_dir=str(directory), database=db, is_test_instance=True,
            identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test", did_web_slug="same-target",
        )
        assert db._backend.is_connected
        assert await db.fetchone("SELECT node_id FROM graph_nodes WHERE node_id=?", (result.agent_did,)) == (result.agent_did,)
        assert (directory / "same-target_did.json").is_file()
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "cancel"])
@pytest.mark.parametrize("stage", ["connect", "schema"])
async def test_owned_sqlite_initialization_failure_settles_before_cleanup_and_retry(tmp_path, monkeypatch, failure, stage):
    import asyncio
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    directory = tmp_path / "failed-initialization"
    kwargs = dict(output_dir=str(directory), is_test_instance=True, identity_method="did:web",
                  did_web_domain="agents.kestrel-sovereign.test", did_web_slug="init-retry")
    opened = []
    native_connect, native_schema = SQLiteBackend.connect, AsyncDatabase._init_schema
    reached, release = asyncio.Event(), asyncio.Event()

    async def capture_connection(backend):
        await native_connect(backend)
        opened.append(backend)
        if stage == "connect":
            await refuse_initialization()

    async def refuse_initialization():
        reached.set()
        if failure == "cancel":
            await release.wait()
        raise ConnectionError("native initialization refused")

    async def refuse_schema(db):
        await refuse_initialization()

    monkeypatch.setattr(SQLiteBackend, "connect", capture_connection)
    monkeypatch.setattr(AsyncDatabase, "_init_schema", refuse_schema)
    task = asyncio.create_task(create_kestrel_identity_async(**kwargs))
    try:
        await asyncio.wait_for(reached.wait(), 10)
        if failure == "cancel":
            task.cancel()
            task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else ConnectionError):
            await asyncio.wait_for(task, 30)
        assert all(backend._connection is None and not backend.connection_retirement_pending for backend in opened)
        assert not (directory / "kestrel_prime.db").exists(), "failed initialization stranded its exclusively reserved SQLite inode"
        assert not list(directory.glob("init-retry_*"))
        monkeypatch.setattr(AsyncDatabase, "_init_schema", native_schema)
        monkeypatch.setattr(SQLiteBackend, "connect", native_connect)
        fresh = await create_kestrel_identity_async(**kwargs)
        assert fresh.agent_did.endswith(":init-retry")
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for backend in opened:
            await backend.close()


@pytest.mark.asyncio
async def test_owned_initialization_uncertain_retirement_retains_reserved_inode(tmp_path, monkeypatch):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.storage.async_database import AsyncDatabase, DatabaseInitializationCleanupError
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    directory = tmp_path / "uncertain-initialization"
    opened = []
    native_close = SQLiteBackend.close

    async def fail_schema(db):
        opened.append(db._backend)
        raise RuntimeError("synthetic initialization failure on actual SQLite")

    async def uncertain_close(backend):
        raise RuntimeError("synthetic close acknowledgement unavailable")

    monkeypatch.setattr(AsyncDatabase, "_init_schema", fail_schema)
    monkeypatch.setattr(SQLiteBackend, "close", uncertain_close)
    try:
        with pytest.raises(DatabaseInitializationCleanupError, match="retain recovery evidence"):
            await create_kestrel_identity_async(
                output_dir=str(directory), is_test_instance=True, identity_method="did:web",
                did_web_domain="agents.kestrel-sovereign.test", did_web_slug="uncertain-init",
            )
        assert (directory / "kestrel_prime.db").is_file()
        assert not list(directory.glob("uncertain-init_*"))
        assert opened and opened[0]._connection is not None
    finally:
        for backend in opened:
            await native_close(backend)
            assert backend._connection is None and not backend.connection_retirement_pending


@pytest.mark.asyncio
async def test_owned_initialization_failure_never_unlinks_replacement_inode(tmp_path, monkeypatch):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    directory = tmp_path / "replaced-initialization"
    captured = []

    async def replace_before_failure(db):
        path = directory / "kestrel_prime.db"
        captured.append(db._backend)
        path.rename(directory / "retained-original.db")
        with sqlite3.connect(path) as replacement:
            replacement.execute("CREATE TABLE retained_winner (value TEXT)")
            replacement.execute("INSERT INTO retained_winner VALUES ('do not erase')")
        raise RuntimeError("synthetic initialization failed after replacement")

    monkeypatch.setattr(AsyncDatabase, "_init_schema", replace_before_failure)
    try:
        with pytest.raises(RuntimeError, match="lost exclusive creation ownership"):
            await create_kestrel_identity_async(
                output_dir=str(directory), is_test_instance=True, identity_method="did:web",
                did_web_domain="agents.kestrel-sovereign.test", did_web_slug="replaced-init",
            )
        with sqlite3.connect(directory / "kestrel_prime.db") as replacement:
            assert replacement.execute("SELECT value FROM retained_winner").fetchone() == ("do not erase",)
        assert (directory / "retained-original.db").is_file()
        assert all(backend._connection is None and not backend.connection_retirement_pending for backend in captured)
    finally:
        for backend in captured:
            await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["prepare", "publication"])
async def test_owned_failure_retirement_uncertainty_never_erases_database(tmp_path, monkeypatch, stage):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
    from kestrel_sovereign.constitution import anchored_bytes

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    directory = tmp_path / "failed-retirement"
    opened = []
    native_close = SQLiteBackend.close

    async def fail_audit(prompt):
        raise RuntimeError("synthetic prepare failure")

    async def fail_publication(*args, **kwargs):
        raise RuntimeError("synthetic native publication failure")

    async def uncertain_close(backend):
        opened.append(backend)
        raise RuntimeError("synthetic retirement acknowledgement unavailable")

    monkeypatch.setattr(SQLiteBackend, "close", uncertain_close)
    if stage == "publication":
        monkeypatch.setattr(anchored_bytes, "_store_exact_native_file", fail_publication)
    try:
        with pytest.raises(RuntimeError, match="retirement acknowledgement unavailable"):
            await create_kestrel_identity_async(
                output_dir=str(directory), is_test_instance=True, identity_method="did:web",
                did_web_domain="agents.kestrel-sovereign.test", did_web_slug="retired-init",
                genesis_auditor=fail_audit if stage == "prepare" else None,
            )
        assert (directory / "kestrel_prime.db").is_file(), "failed retirement erased a still-owned database"
        assert opened and opened[-1]._connection is not None
        assert not list(directory.glob("retired-init_*"))
        assert not list(directory.glob(".inception-*"))
    finally:
        for backend in opened:
            await native_close(backend)
            assert backend._connection is None and not backend.connection_retirement_pending


@pytest.mark.asyncio
async def test_native_sqlite_connection_file_custody_rejects_a_replaced_path(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db.interface import ConnectionError as NativeConnectionError

    path = tmp_path / "authority.db"
    original = await AsyncDatabase.sqlite(str(path))
    winner = None
    try:
        retained = original._backend.connected_file_identity
        original._backend.assert_connected_file_still_valid()
        for suffix in ("", "-wal", "-shm"):
            part = str(path) + suffix
            if os.path.exists(part):
                os.rename(part, str(tmp_path / "retained-original.db") + suffix)
        winner = await AsyncDatabase.sqlite(str(path))
        assert winner._backend.connected_file_identity != retained
        with pytest.raises(NativeConnectionError, match="inode changed"):
            original._backend.assert_connected_file_still_valid()
        winner._backend.assert_connected_file_still_valid()
        await winner.close()
        winner = None
        digest = sha256(path.read_bytes()).digest()
        await original.close()
        assert sha256(path.read_bytes()).digest() == digest
    finally:
        if winner is not None:
            await winner.close()
        await original.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("evidence", ["symlink", "hardlink", "stranded_wal", "corrupt", "overflow", "valid"])
async def test_owned_archive_inspection_is_bounded_read_only_and_fail_closed(tmp_path, evidence):
    from pathlib import Path
    from kestrel_sovereign.inception_service import _assert_local_inception_lifetimes
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    archive_path = tmp_path / "kestrel_prime.db.backup-fixture"
    native = await AsyncDatabase.sqlite(str(archive_path))
    await native.close()
    if evidence == "symlink":
        target = tmp_path / "retained-authority"
        archive_path.rename(target)
        archive_path.symlink_to(target)
    elif evidence == "hardlink":
        os.link(archive_path, tmp_path / "shared-authority")
    elif evidence == "stranded_wal":
        (tmp_path / "kestrel_prime.db-wal.backup-fixture").write_bytes(b"unread committed evidence")
    elif evidence == "corrupt":
        archive_path.write_bytes(b"not a database")
    elif evidence == "overflow":
        for number in range(128):
            (tmp_path / f"kestrel_prime.db.backup-{number}").write_bytes(b"must not be opened")
    before = {p.name: (p.lstat().st_ino, p.lstat().st_mtime_ns, sha256(p.read_bytes()).digest()) for p in tmp_path.iterdir()}
    if evidence == "valid":
        await _assert_local_inception_lifetimes(Path(tmp_path), "did:test:genuinely-new")
    else:
        with pytest.raises(Exception, match="regular database|sidecar evidence|not a database|inspection bound"):
            await _assert_local_inception_lifetimes(Path(tmp_path), "did:test:genuinely-new")
    assert {p.name: (p.lstat().st_ino, p.lstat().st_mtime_ns, sha256(p.read_bytes()).digest()) for p in tmp_path.iterdir()} == before


@pytest.mark.asyncio
@pytest.mark.parametrize("link", ["symlink", "hardlink"])
async def test_owned_force_refuses_linked_existing_database_before_open(tmp_path, link):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    target = tmp_path / "other-authority.db"
    native = await AsyncDatabase.sqlite(str(target))
    await native.close()
    directory = tmp_path / "birth"
    directory.mkdir()
    path = directory / "kestrel_prime.db"
    if link == "symlink":
        path.symlink_to(target)
    else:
        os.link(target, path)
    before = (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes())
    with pytest.raises(ValueError, match="exclusively retained regular database"):
        await create_kestrel_identity_async(output_dir=str(directory), force=True, is_test_instance=True,
                                            identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test", did_web_slug="new-linked")
    assert (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()) == before
    assert path.exists()
    assert not list(directory.glob("*.backup-*"))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_partial_key_publication_restores_original_active_files(db_backend, tmp_path, monkeypatch):
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.identity.did_web import build_did

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    monkeypatch.setenv("KESTREL_AUDIT_MODE", "skip")
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:partial-fixture:" + uuid4().hex)
    await storage.initialize()
    directory = tmp_path / "active"
    kwargs = dict(database=storage.db, is_test_instance=True, identity_method="did:web", did_web_domain="agents.kestrel-sovereign.test")
    try:
        await create_kestrel_identity_async(output_dir=str(directory), did_web_slug="old-" + uuid4().hex, **kwargs)
        original = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        slug = "partial-" + uuid4().hex
        identity = build_did(kwargs["did_web_domain"], [slug])
        native_link = os.link
        publications = []

        def fail_second_staged_link(source, destination, **options):
            if source.parent.name.startswith(".inception-"):
                publications.append(source.name)
                if len(publications) == 2:
                    raise OSError("injected second key publication failure")
            return native_link(source, destination, **options)

        monkeypatch.setattr(os, "link", fail_second_staged_link)
        with pytest.raises(Exception, match="second key publication failure"):
            await create_kestrel_identity_async(output_dir=str(directory), did_web_slug=slug, force=True, **kwargs)
        assert len(publications) == 2
        assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == original
        assert not list(directory.glob(".inception-*"))
        assert await storage.db.fetchone("SELECT node_id FROM graph_nodes WHERE node_id=?", (identity,)) is None
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["turn", "audit", "exit"])
async def test_sqlite_governance_custody_reserves_writer_before_first_read(tmp_path, monkeypatch, phase):
    path = tmp_path / "writer-reservation.db"
    storage = AsyncStorage(str(path), backend="sqlite", agent_id="did:test:sqlite-reservation:" + uuid4().hex)
    await storage.initialize()
    peer = sqlite3.connect(path, timeout=0, isolation_level=None)
    attempts = []
    try:
        turn, _ = await _ready_turn(storage)
        agent = await _agent(storage, is_new_identity=False) if phase != "turn" else turn
        if phase == "exit":
            assert await agent.enter_safe_mode("explicit exit fixture")
        await storage.db.execute_commit("CREATE TABLE unrelated_writer (id INTEGER)")
        read = storage.get_node

        async def raced_read(identity):
            node = await read(identity)
            if not attempts and storage.db.owns_open_transaction and identity == storage.agent_id:
                try:
                    peer.execute("INSERT INTO unrelated_writer VALUES (1)")
                    attempts.append("committed")
                except sqlite3.OperationalError as exc:
                    assert "locked" in str(exc)
                    attempts.append("reserved")
            return node

        monkeypatch.setattr(storage, "get_node", raced_read)
        if phase == "turn":
            async with agent._turn_lifecycle():
                result = await ConstitutionMixin._genesis_audit_cognition_block(agent, "ordinary input")
            assert result is None, result
        elif phase == "audit":
            assert await agent._record_successful_constitution_audit(source="native reservation proof")
        else:
            result = await agent.exit_safe_mode(authorization="explicit fixture owner")
            assert not result.startswith("Safe Mode remains active:"), result
        assert attempts == ["reserved"], "An unrelated write must not stale the governance snapshot"
        peer.execute("INSERT INTO unrelated_writer VALUES (2)")
        assert await storage.db.fetchone("SELECT COUNT(*) FROM unrelated_writer") == (1,)
        assert not (await agent._constitution_state_store.load(storage.agent_id)).safe_mode
    finally:
        peer.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["runtime", "offline"])
@pytest.mark.parametrize("history_size", [127, 128, 129])
async def test_signed_repair_history_limit_is_atomic_and_runtime_compatible(db_backend, tmp_path, monkeypatch, writer, history_size):
    from kestrel_sovereign.setup import constitution_reanchor as offline
    from kestrel_sovereign.constitution.genesis_audit import reconcile_genesis_receipt

    identity = "did:test:history-limit:" + uuid4().hex
    storage = AsyncStorage(str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id=identity) if db_backend.backend_type == "sqlite" else AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    try:
        agent = await _agent(storage)
        content = resolve_governing_constitution_bytes(None)
        old = content + b"\nHistorical governing revision for bounded-history proof.\n"
        old_hash = await storage.store_file(old, "old.md")
        def receipt(digest):
            return {"constitution_hash": digest, "status": "passed", "audited": True, "risk_level": 1, "completed_at": "2026-10-09T00:00:00Z", "reasoning": "Synthetic native bounded history"}
        history = [{"receipt": receipt(sha256((identity + str(i)).encode()).hexdigest())} for i in range(history_size)]
        await storage.add_node(GraphNode(node_id=identity, node_type="agent", label="History boundary", properties={"constitution_hash": old_hash, "genesis_audit": receipt(old_hash), "genesis_audit_history": history}))
        await agent._anchor_constitution_governance(old_hash)
        before = deepcopy((await storage.get_node(identity)).properties)
        artifact, root = _write_authority_files(tmp_path, content)
        agent._sovereign_trust_root_path = root
        if writer == "runtime":
            result = await ConstitutionMixin.reanchor_constitution(agent, amendment_artifact_path=str(artifact))
            error = result if result.startswith("Error:") else None
        else:
            target = offline.ReanchorTarget(tmp_path / "kestrel_prime.db", "sqlite", identity) if db_backend.backend_type == "sqlite" else offline.ReanchorTarget(None, "postgres", identity, db_backend._dsn)
            async def exact_target(*args, **kwargs):
                return target
            @asynccontextmanager
            async def no_embedding(*args, **kwargs):
                yield None
            monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
            monkeypatch.setattr(offline, "_agent_embedding", no_embedding)
            result = await offline.reanchor_constitution(agent_name="history proof", agent_dir=tmp_path if target.anchor_path else None, force=True, sovereign_trust_root_path=root, amendment_artifact_path=artifact, runtime_backend=target.backend, runtime_dsn=target.dsn, hosted_agent_did=identity if target.backend == "postgres" else None, environ={})
            error = result.error
        fresh = (await storage.get_node(identity)).properties
        if history_size >= 128:
            assert error is not None, "Repair cannot publish an unreadable 129-entry history"
            assert fresh == before
            assert not await storage.db.fetchone("SELECT agent_id FROM file_owners WHERE content_hash=? AND agent_id=?", (sha256(artifact.read_bytes()).hexdigest(), identity))
        else:
            assert error is None, error
            assert len(fresh["genesis_audit_history"]) == 128
            assert reconcile_genesis_receipt(fresh, fresh["constitution_hash"])["status"] == "pending"
    finally:
        await storage.close()
