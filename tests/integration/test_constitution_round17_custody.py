"""Native proofs for bounded reanchor evidence and connected SQLite adapters."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from hashlib import sha256
import sqlite3
import threading
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.storage import AsyncStorage, GraphNode
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.interface import ConnectionError
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from kestrel_sovereign.storage.sqla import make_session_factory
from tests.integration.test_constitution_refusal_races import _agent
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["runtime", "offline"])
@pytest.mark.parametrize("shape", ["not-list", "null-history", "bad-entry", "missing-destination", "bad-current", "null-current", "bad-current-old", "bad-artifact", "bad-supersession", "overflow", "at-limit", "last-slot"])
async def test_signed_same_hash_repair_preserves_complete_reanchor_evidence(db_backend, tmp_path, monkeypatch, writer, shape):
    from kestrel_sovereign.setup import constitution_reanchor as offline
    from kestrel_sovereign.constitution import anchored_bytes

    identity = "did:test:reanchor-history:" + uuid4().hex
    storage = AsyncStorage(str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id=identity) if db_backend.backend_type == "sqlite" else AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    try:
        agent = await _agent(storage)
        content = resolve_governing_constitution_bytes(None)
        digest = await storage.store_file(content, "constitution.md")
        current = {"old_hash": "none", "new_hash": digest, "signed_artifact_hash": "a" * 64, "timestamp": "2026-10-10T00:00:00Z"}
        entry = {"receipt": deepcopy(current), "superseded_by_constitution_hash": digest, "superseded_by_artifact_hash": "b" * 64, "superseded_at": "2026-10-10T00:00:00Z", "provenance": "native-test"}
        history = [deepcopy(entry)]
        if shape == "not-list": history = {"retained": entry}
        elif shape == "null-history": history = None
        elif shape == "bad-entry": history = ["retained unreadable receipt"]
        elif shape == "missing-destination": history = [{"receipt": {"old_hash": digest}, "superseded_by_constitution_hash": digest}]
        elif shape == "bad-current": current = "retained unreadable receipt"
        elif shape == "null-current": current = None
        elif shape == "bad-current-old": current["old_hash"] = "invalid"
        elif shape == "bad-artifact": history[0]["receipt"]["signed_artifact_hash"] = "invalid"
        elif shape == "bad-supersession": history[0]["superseded_by_constitution_hash"] = "invalid"
        elif shape in {"overflow", "at-limit", "last-slot"}: history = [deepcopy(entry) for _ in range({"overflow": 129, "at-limit": 128, "last-slot": 127}[shape])]
        await storage.add_node(GraphNode(node_id=identity, node_type="agent", label="Reanchor boundary", properties={"constitution_hash": digest, "constitution_reanchor": current, "constitution_reanchor_history": history, "genesis_audit": {"constitution_hash": digest, "status": "pending", "audited": False}}))
        await agent._anchor_constitution_governance(digest)
        artifact, root = _write_authority_files(tmp_path, content)
        agent._sovereign_trust_root_path = root
        calls = []
        actual_store = anchored_bytes.store_verified_governing_file
        async def observed_store(*args, **kwargs):
            calls.append(True)
            return await actual_store(*args, **kwargs)
        monkeypatch.setattr(anchored_bytes, "store_verified_governing_file", observed_store)
        target = offline.ReanchorTarget(tmp_path / "kestrel_prime.db", "sqlite", identity) if db_backend.backend_type == "sqlite" else offline.ReanchorTarget(None, "postgres", identity, db_backend._dsn)
        async def exact_target(*args, **kwargs): return target
        @asynccontextmanager
        async def no_embedding(*args, **kwargs): yield None
        monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
        monkeypatch.setattr(offline, "_agent_embedding", no_embedding)
        async def repair():
            if writer == "runtime":
                result = await ConstitutionMixin.reanchor_constitution(agent, amendment_artifact_path=str(artifact))
                return result if result.startswith("Error:") else None
            result = await offline.reanchor_constitution(agent_name="receipt proof", agent_dir=tmp_path if target.anchor_path else None, force=True, sovereign_trust_root_path=root, amendment_artifact_path=artifact, runtime_backend=target.backend, runtime_dsn=target.dsn, hosted_agent_did=identity if target.backend == "postgres" else None, environ={})
            return result.error
        if shape == "last-slot":
            assert await repair() is None
            fresh = (await storage.get_node(identity)).properties
            assert len(fresh["constitution_reanchor_history"]) == 128
            assert fresh["constitution_reanchor_history"][:127] == history
            assert fresh["constitution_reanchor_history"][-1]["receipt"] == current
        before = deepcopy((await storage.get_node(identity)).properties)
        calls.clear()
        error = await repair()
        assert error is not None, "Signed repair must refuse malformed/overflowing reanchor evidence"
        assert (await storage.get_node(identity)).properties == before
        assert not calls, "Reanchor evidence admission must precede native file/ownership mutation"
    finally:
        await storage.close()


def _rows(path, value):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE retained (value TEXT)")
        conn.execute("INSERT INTO retained VALUES (?)", (value,))


@pytest.mark.asyncio
async def test_sqla_and_scheduler_share_original_native_sqlite_target(tmp_path, monkeypatch):
    original = tmp_path / "original"
    other = tmp_path / "other"
    original.mkdir(); other.mkdir()
    _rows(original / "authority.db", "original")
    _rows(other / "authority.db", "decoy")
    monkeypatch.chdir(original)
    backend = SQLiteBackend("authority.db")
    monkeypatch.chdir(other)
    await backend.connect()
    db = AsyncDatabase(backend)
    try:
        assert backend.db_path == str(original / "authority.db")
        with pytest.raises(AttributeError):
            backend.db_path = str(other / "authority.db")
        factory = make_session_factory(db)
        async with factory.read_session() as session:
            assert (await session.execute(text("SELECT value FROM retained"))).scalars().all() == ["original"]
        runner = SchedulerRunner(db, "did:test:path", AsyncMock())
        sidecar = runner._sqlite_rollout_lock_path("did:test:path")
        assert sidecar.startswith(str(original / "authority.db") + ".scheduler-rollout-")
        monkeypatch.chdir(original)
        assert runner._sqlite_rollout_lock_path("did:test:path") == sidecar
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_yielded_sqla_connection_refuses_post_admission_inode_replacement(tmp_path):
    path = tmp_path / "authority.db"
    _rows(path, "original")
    db = await AsyncDatabase.sqlite(str(path), schema_initializer=lambda _db: asyncio.sleep(0))
    try:
        factory = make_session_factory(db)
        async with factory.engine.connect() as connection:
            assert (await connection.execute(text("SELECT value FROM retained"))).scalar_one() == "original"
            for suffix in ("", "-wal", "-shm"):
                source = tmp_path / (path.name + suffix)
                if source.exists(): source.rename(tmp_path / ("retained.db" + suffix))
            _rows(path, "replacement")
            with pytest.raises(ConnectionError, match="inode"):
                await connection.execute(text("SELECT value FROM retained"))
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sqla_driver_never_creates_a_missing_replacement_between_checks(tmp_path):
    path = tmp_path / "authority.db"
    _rows(path, "original")
    db = await AsyncDatabase.sqlite(str(path), schema_initializer=lambda _db: asyncio.sleep(0))
    try:
        factory = make_session_factory(db)
        def remove_original(*args):
            for suffix in ("", "-wal", "-shm"):
                source = tmp_path / (path.name + suffix)
                if source.exists(): source.rename(tmp_path / ("retained.db" + suffix))
        # This runs after our pre-open custody hook, immediately before the
        # actual installed SQLAlchemy dialect acquires its native driver.
        event.listen(factory.engine.sync_engine, "do_connect", remove_original)
        with pytest.raises(OperationalError, match="unable to open database"):
            async with factory.engine.connect():
                pytest.fail("Missing file must not produce a new SQLAlchemy database")
        assert not path.exists()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_scheduler_rechecks_connected_custody_after_blocking_sidecar_acquisition(tmp_path, monkeypatch):
    from kestrel_sovereign.features.scheduler import runner as runner_module

    path = tmp_path / "authority.db"
    _rows(path, "original")
    db = await AsyncDatabase.sqlite(str(path), schema_initializer=lambda _db: asyncio.sleep(0))
    runner = SchedulerRunner(db, "did:test:sidecar", AsyncMock())
    reached = asyncio.Event()
    proceed = threading.Event()
    loop = asyncio.get_running_loop()
    actual_acquire = runner_module._SQLiteRolloutFileLock.acquire
    def blocked_acquire(lock):
        loop.call_soon_threadsafe(reached.set)
        if not proceed.wait(10): raise TimeoutError("Fixture acquisition was not released")
        actual_acquire(lock)
    monkeypatch.setattr(runner_module._SQLiteRolloutFileLock, "acquire", blocked_acquire)
    async def enter_gate():
        async with runner._sqlite_rollout_gate("did:test:sidecar"):
            pytest.fail("Replaced native target must not admit a scheduler effect")
    task = asyncio.create_task(enter_gate())
    try:
        await asyncio.wait_for(reached.wait(), 10)
        for suffix in ("", "-wal", "-shm"):
            source = tmp_path / (path.name + suffix)
            if source.exists(): source.rename(tmp_path / ("retained.db" + suffix))
        _rows(path, "replacement")
        proceed.set()
        with pytest.raises(ConnectionError, match="inode"):
            await task
    finally:
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_storage_audit_sidecar_uses_native_construction_target(tmp_path, monkeypatch, configured):
    original = tmp_path / "original"
    other = tmp_path / "other"
    original.mkdir(); other.mkdir()
    monkeypatch.chdir(original)
    storage = AsyncStorage(config={"backend": "sqlite", "db_path": "authority.db"}) if configured else AsyncStorage("authority.db", backend="sqlite")
    monkeypatch.chdir(other)
    try:
        await storage.initialize()
        assert storage.db_path == storage.db.backend.db_path == str(original / "authority.db")
        assert storage.destructive_audit.db_path.parent == original
        assert not (other / "authority.db").exists()
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["factory", "cached-factory", "session", "engine", "sidecar"])
async def test_sqlite_adapter_and_sidecar_refuse_replaced_connected_inode(tmp_path, boundary):
    path = tmp_path / "authority.db"
    _rows(path, "original")
    db = await AsyncDatabase.sqlite(str(path), schema_initializer=lambda _db: asyncio.sleep(0))
    factory = None
    try:
        if boundary != "factory":
            factory = make_session_factory(db)
            async with factory.read_session() as session:
                assert (await session.execute(text("SELECT value FROM retained"))).scalar_one() == "original"
        # Preserve the original connection's DB and WAL together. Only this
        # fixture's replacement is created; no retained agent data is touched.
        old = tmp_path / "retained.db"
        for suffix in ("", "-wal", "-shm"):
            source = tmp_path / (path.name + suffix)
            if source.exists(): source.rename(tmp_path / (old.name + suffix))
        _rows(path, "replacement")
        winner = sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ConnectionError, match="changed|inode"):
            if boundary in {"factory", "cached-factory"}: make_session_factory(db)
            elif boundary == "session":
                async with factory.read_session() as session:
                    await session.execute(text("SELECT value FROM retained"))
            elif boundary == "engine":
                async with factory.engine.connect() as conn:
                    await conn.execute(text("SELECT value FROM retained"))
            else: SchedulerRunner(db, "did:test:path", AsyncMock())._sqlite_rollout_lock_path("did:test:path")
        assert sha256(path.read_bytes()).hexdigest() == winner
    finally:
        await db.close()
