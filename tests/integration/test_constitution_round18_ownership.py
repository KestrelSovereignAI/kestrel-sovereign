"""Native regressions for final-review metadata and acquisition ownership."""
import asyncio
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from kestrel_sovereign.storage import AsyncStorage, GraphNode
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("producer", ["payer", "graduation"])
async def test_metadata_producers_preserve_late_terminal_governance(db_backend, tmp_path, monkeypatch, producer):
    from kestrel_sovereign import graduate_service
    from kestrel_sovereign.services.payer_resolver import FoundationPayerResolver
    from kestrel_sovereign.constitution.genesis_audit import pending_genesis_audit, utc_timestamp

    identity = "did:test:round18-metadata:" + uuid4().hex
    digest = "a" * 64
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    proceed, reached = asyncio.Event(), asyncio.Event()
    task = None
    try:
        initial = {"constitution_hash": digest, "genesis_audit": pending_genesis_audit(digest, provenance="native-test"), "is_test_instance": True}
        await storage.add_node(GraphNode(node_id=identity, node_type="agent", label="Metadata race", properties=initial))

        async def pause():
            reached.set()
            await asyncio.wait_for(proceed.wait(), 5)

        class BarrierDatabase:
            def __getattr__(self, name):
                return getattr(storage.db, name)
            async def fetchall(self, query, params=()):
                rows = await storage.db.fetchall(query, params)
                if query.startswith("SELECT properties FROM graph_nodes"):
                    await pause()
                return rows
            @asynccontextmanager
            async def transaction(self, *args, **kwargs):
                # Corrected writers pause BEFORE native custody, not while
                # holding it; the competing owner can commit first.
                await pause()
                async with storage.db.transaction(*args, **kwargs):
                    yield

        if producer == "payer":
            resolver = object.__new__(FoundationPayerResolver)
            resolver._db = BarrierDatabase()
            task = asyncio.create_task(resolver._persist_openrouter_key_hash(identity, "native-public-credential-handle", require_row=True))
        else:
            @asynccontextmanager
            async def borrowed_storage(*args, **kwargs):
                yield storage
            async def validated_metadata_seam(*args, **kwargs):
                # Validation policy is independent of this native writer race.
                # The real public graduation producer remains under test.
                await pause()
                checklist = graduate_service.ValidationChecklist()
                checklist.add_check("synthetic metadata seam", True, "native write ownership test")
                return checklist
            monkeypatch.setattr(graduate_service, "Storage", borrowed_storage)
            monkeypatch.setattr(graduate_service, "validate_agent", validated_metadata_seam)
            marker = tmp_path / "graduation-target.db"
            marker.touch()
            task = asyncio.create_task(graduate_service.graduate_agent(str(marker)))

        await asyncio.wait_for(reached.wait(), 5)
        async with storage.transaction(immediate=True):
            fresh = await storage.get_node(identity)
            fresh.properties["genesis_audit"] = {"status": "failed", "constitution_hash": digest, "risk_level": 3, "audited": True, "completed_at": utc_timestamp(), "reasoning": "Native concurrent terminal refusal"}
            fresh.properties["constitution_reanchor"] = {"old_hash": "none", "new_hash": digest, "signed_artifact_hash": "b" * 64}
            fresh.properties["constitution_reanchor_history"] = []
            await storage.add_node(fresh)
        committed = deepcopy((await storage.get_node(identity)).properties)
        proceed.set()
        result = await asyncio.wait_for(task, 5)
        actual = (await storage.get_node(identity)).properties
        for field in ("constitution_hash", "genesis_audit", "constitution_reanchor", "constitution_reanchor_history"):
            assert actual[field] == committed[field]
        if producer == "payer":
            assert actual["openrouter_key_hash"] == "native-public-credential-handle"
        else:
            assert result is True
            assert actual["is_test_instance"] is False
            assert actual["graduated_at"]
    finally:
        proceed.set()
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["rename", "replacement"])
async def test_post_acquisition_path_failure_retires_native_connection(tmp_path, monkeypatch, shape):
    from kestrel_sovereign import inception_service

    original = AsyncDatabase.sqlite
    captured = []
    expected = b"unrelated replacement must survive"
    moved = tmp_path / "retained-original.db"

    async def acquired_then_changed(path, *args, **kwargs):
        database = await original(path, *args, **kwargs)
        captured.append((database, database.backend._connection))
        Path(path).rename(moved)
        if shape == "replacement":
            Path(path).write_bytes(expected)
        return database

    monkeypatch.setattr(AsyncDatabase, "sqlite", acquired_then_changed)
    try:
        with pytest.raises((FileNotFoundError, RuntimeError)):
            await inception_service.create_kestrel_identity_async(
                output_dir=str(tmp_path / "agent"), identity_method="did:web",
                did_web_domain="round18.invalid", did_web_slug="native-" + uuid4().hex,
                is_test_instance=True,
            )
        assert len(captured) == 1
        database, connection = captured[0]
        assert database.backend._connection is None
        assert not database.connection_retirement_pending
        worker = getattr(connection, "_thread", connection)
        assert not worker.is_alive()
        assert moved.exists()
        if shape == "replacement":
            assert (tmp_path / "agent" / "kestrel_prime.db").read_bytes() == expected
    finally:
        # Before-fix failure still joins every actual acquired worker.
        for database, connection in captured:
            await database.close()
            assert not getattr(connection, "_thread", connection).is_alive()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["runtime", "offline"])
@pytest.mark.parametrize("shape", ["repair", "active", "ambiguous", "unreadable", "missing-blob"])
async def test_signed_corrupt_pointer_repair_uses_retained_bytes_not_raw_pointer(db_backend, tmp_path, monkeypatch, writer, shape):
    from hashlib import sha256
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    from kestrel_sovereign.constitution import anchored_bytes
    from kestrel_sovereign.constitution.emancipation import EmancipationContract
    from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
    from kestrel_sovereign.setup import constitution_reanchor as offline
    from tests.integration.test_constitution_refusal_races import _agent
    from tests.integration.test_constitution_reanchor_e2e import _write_authority_files

    identity = "did:test:round18-pointer:" + uuid4().hex
    storage = AsyncStorage(
        str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id=identity,
    ) if db_backend.backend_type == "sqlite" else AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    calls = []
    native_publish = anchored_bytes.store_verified_governing_file
    async def observed_publish(*args, **kwargs):
        calls.append(True)
        return await native_publish(*args, **kwargs)
    monkeypatch.setattr(anchored_bytes, "store_verified_governing_file", observed_publish)
    try:
        agent = await _agent(storage)
        contract = EmancipationContract(enabled=True, terms="Native protected contract " + uuid4().hex) if shape == "active" else None
        # Unique content avoids touching any other fixture's shared digest.
        old_content = ("<!-- native pointer proof " + uuid4().hex + " -->\n").encode() + resolve_governing_constitution_bytes(contract)
        prior_hash = await storage.store_file(old_content, "prior-governing.md")
        await storage.add_node(GraphNode(
            node_id=identity, node_type="agent", label="Native pointer proof",
            properties={"constitution_hash": prior_hash, "genesis_audit": {"status": "pending", "audited": False, "constitution_hash": prior_hash}},
        ))
        await agent._anchor_constitution_governance(prior_hash)
        fresh = await storage.get_node(identity)
        fresh.properties["constitution_hash"] = "db-writer-replaced-hash"
        await storage.add_node(fresh)
        if shape == "ambiguous":
            alternate = await storage.store_file(b"separate native evidence " + uuid4().hex.encode(), "alternate.md")
            await storage.add_node(GraphNode(node_id=alternate, node_type="document", label="Other governing evidence", properties={"hash": alternate, "type": "Constitution"}))
            await storage.add_edge(identity, alternate, "governed_by")
        elif shape == "unreadable":
            await storage.db.execute_commit("UPDATE files SET content = ? WHERE content_hash = ?", (b"corrupted native fixture bytes", prior_hash))
        elif shape == "missing-blob":
            await storage.db.execute_commit("DELETE FROM file_owners WHERE content_hash = ?", (prior_hash,))
            await storage.db.execute_commit("DELETE FROM files WHERE content_hash = ?", (prior_hash,))

        before_properties = deepcopy((await storage.get_node(identity)).properties)
        before_edges = await storage.db.fetchall("SELECT target_id FROM graph_edges WHERE source_id = ? AND label = 'governed_by' ORDER BY target_id", (identity,))
        before_files = await storage.db.fetchall("SELECT content_hash, content, metadata FROM files ORDER BY content_hash")
        new_content = resolve_governing_constitution_bytes(None)
        digest = sha256(new_content).hexdigest()
        artifact, root = _write_authority_files(tmp_path, new_content)
        agent._sovereign_trust_root_path = root
        if writer == "runtime":
            result = await ConstitutionMixin.reanchor_constitution(agent, amendment_artifact_path=str(artifact))
            succeeded = not result.startswith("Error:")
        else:
            target = offline.ReanchorTarget(Path(storage.db_path), "sqlite", identity) if db_backend.backend_type == "sqlite" else offline.ReanchorTarget(None, "postgres", identity, db_backend._dsn)
            async def exact_target(*args, **kwargs):
                return target
            @asynccontextmanager
            async def no_embedding(*args, **kwargs):
                yield None
            monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
            monkeypatch.setattr(offline, "_agent_embedding", no_embedding)
            result = await offline.reanchor_constitution(
                agent_name="native pointer proof", agent_dir=tmp_path if target.anchor_path else None,
                force=True, sovereign_trust_root_path=root, amendment_artifact_path=artifact,
                runtime_backend=target.backend, runtime_dsn=target.dsn,
                hosted_agent_did=identity if target.backend == "postgres" else None, environ={},
            )
            succeeded = result.error is None
        actual = (await storage.get_node(identity)).properties
        if shape == "repair":
            assert succeeded, result
            assert calls == [True]
            assert actual["constitution_hash"] == digest
            receipt = actual["constitution_reanchor"]
            assert receipt["old_hash"] == prior_hash
            assert receipt["repaired_constitution_pointer"] == "db-writer-replaced-hash"
            assert await storage.files.retrieve_file(prior_hash) == old_content
        else:
            assert not succeeded, result
            assert calls == []
            assert actual == before_properties
            assert await storage.db.fetchall("SELECT target_id FROM graph_edges WHERE source_id = ? AND label = 'governed_by' ORDER BY target_id", (identity,)) == before_edges
            assert await storage.db.fetchall("SELECT content_hash, content, metadata FROM files ORDER BY content_hash") == before_files
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["uncertain-close", "repeated-cancellation"])
async def test_path_inspection_failure_settles_close_and_retains_uncertain_evidence(tmp_path, monkeypatch, mode):
    from kestrel_sovereign import inception_service
    from kestrel_sovereign.storage.async_database import DatabaseInitializationCleanupError

    original = AsyncDatabase.sqlite
    acquired = []
    closing, release = asyncio.Event(), asyncio.Event()
    displaced = tmp_path / "displaced.db"
    task = None
    async def acquire_then_displace(path, *args, **kwargs):
        database = await original(path, *args, **kwargs)
        real_close = database.close
        connection = database.backend._connection
        acquired.append((database, connection, real_close))
        Path(path).rename(displaced)
        async def controlled_close():
            closing.set()
            await release.wait()
            if mode == "uncertain-close":
                raise RuntimeError("Synthetic missing retirement acknowledgement")
            await real_close()
        monkeypatch.setattr(database, "close", controlled_close)
        return database
    monkeypatch.setattr(AsyncDatabase, "sqlite", acquire_then_displace)
    try:
        task = asyncio.create_task(inception_service.create_kestrel_identity_async(
            output_dir=str(tmp_path / "agent"), identity_method="did:web",
            did_web_domain="round18.invalid", did_web_slug="close-" + uuid4().hex,
            is_test_instance=True,
        ))
        await asyncio.wait_for(closing.wait(), 5)
        if mode == "repeated-cancellation":
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            database, connection, _ = acquired[0]
            assert database.backend._connection is None
            assert not database.connection_retirement_pending
            assert not getattr(connection, "_thread", connection).is_alive()
        else:
            release.set()
            with pytest.raises(DatabaseInitializationCleanupError) as failure:
                await asyncio.wait_for(task, 5)
            assert isinstance(failure.value.initialization_error, FileNotFoundError)
            assert isinstance(failure.value.cleanup_error, RuntimeError)
        assert displaced.exists()
        assert not list((tmp_path / "agent").glob("*.key.enc"))
    finally:
        release.set()
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, DatabaseInitializationCleanupError, FileNotFoundError):
                await task
        for database, connection, real_close in acquired:
            await real_close()
            assert not getattr(connection, "_thread", connection).is_alive()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("properties_json", ["null", "[]"])
async def test_payer_metadata_refuses_unreadable_root_instead_of_normalizing_it(db_backend, monkeypatch, properties_json):
    from kestrel_sovereign.services.payer_resolver import FoundationPayerResolver
    from kestrel_sovereign.storage.db.interface import TransactionError

    identity = "did:test:round18-unreadable:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    try:
        await storage.add_node(GraphNode(node_id=identity, node_type="agent", label="Unreadable native root", properties={}))
        await storage.db.execute_commit("UPDATE graph_nodes SET properties = ? WHERE node_id = ?", (properties_json, identity))
        resolver = object.__new__(FoundationPayerResolver)
        resolver._db = storage.db
        with pytest.raises(TransactionError, match="readable existing properties"):
            await resolver._persist_openrouter_key_hash(identity, "native-public-handle", require_row=True)
        assert (await storage.db.fetchone("SELECT properties FROM graph_nodes WHERE node_id = ?", (identity,)))[0] == properties_json
    finally:
        await storage.close()
