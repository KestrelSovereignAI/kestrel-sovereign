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
