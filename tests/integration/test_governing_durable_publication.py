"""Governance attestation/publication must use native bytes, never session caches."""

import hashlib
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.anchored_bytes import (
    governance_evidence,
    revalidate_governance_evidence,
)
from kestrel_sovereign.constitution.genesis_audit import GenesisAuditError
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.features.privacy.feature import PrivacyAgent
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files
from tests.integration.test_constitution_refusal_races import _agent


def _privacy(agent, storage, mode):
    wrapper = PrivacyEnforcingStorage(storage, mode)
    agent.storage = wrapper
    agent.privacy_agent = PrivacyAgent(wrapper, mode)
    return wrapper


async def _identity(storage, properties):
    await storage.add_node(
        GraphNode(
            node_id=storage.agent_id,
            node_type="agent",
            label="native governance",
            properties=properties,
        )
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("damage", ["corrupt", "missing-owner", "absent"])
async def test_exit_cannot_use_isolated_cache_as_durable_attestation(
    db_backend, damage
):
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:cached-exit:" + uuid4().hex
    )
    await storage.initialize()
    physical = None
    try:
        agent = await _agent(storage)
        wrapper = _privacy(agent, storage, "isolated")
        content = resolve_governing_constitution_bytes(None)
        digest = await storage.store_file(content, "constitution.md")
        physical = await storage.db.fetchone(
            "SELECT content, metadata FROM files WHERE content_hash=?", (digest,)
        )
        assert await wrapper.store_file(content, "cached.md") == digest
        await _identity(storage, {"constitution_hash": digest})
        await agent._anchor_constitution_governance(digest)
        await agent.enter_safe_mode("native corruption must not be hidden")
        before = await agent._constitution_state_store.load(agent.agent_id)
        if damage == "corrupt":
            await storage.db.execute_commit(
                "UPDATE files SET content=?,metadata=NULL WHERE content_hash=?",
                (b"damaged physical constitution", digest),
            )
        else:
            await storage.db.execute_commit(
                "DELETE FROM file_owners WHERE content_hash=? AND agent_id=?",
                (digest, agent.agent_id),
            )
            if damage == "absent":
                await storage.db.execute_commit(
                    "DELETE FROM files WHERE content_hash=?", (digest,)
                )
        assert await wrapper.retrieve_file(digest) == content
        # The locked authority check and the ordinary diagnostic are distinct
        # paths; neither may use the facade's valid session bytes as proof.
        from kestrel_sovereign.storage.db.interface import TransactionError

        with pytest.raises((RuntimeError, TransactionError)):
            async with storage.transaction():
                await ConstitutionMixin._lock_verified_constitution_exit(agent)
        assert await agent._constitution_state_store.load(agent.agent_id) == before
        valid, _ = await agent._verify_constitution_integrity()
        assert valid is False
        assert "Native governing bytes do not match the anchored hash" in (
            await ConstitutionMixin._get_governing_constitution(agent)
        )
        result = await agent.exit_safe_mode(authorization="fixture sovereign")
        assert result.startswith("Safe Mode remains active:"), result
        assert (
            await agent._constitution_state_store.load(agent.agent_id)
        ).safe_mode is True
    finally:
        if physical is not None:
            await storage.db.execute_commit(
                "INSERT OR IGNORE INTO files (content_hash,original_name,content,metadata) VALUES (?,'constitution.md',?,?)",
                (digest, *physical),
            )
            await storage.db.execute_commit(
                "UPDATE files SET content=?,metadata=? WHERE content_hash=?",
                (*physical, digest),
            )
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_genesis_refuses_ownership_removed_during_actual_auditor_await(
    db_backend,
):
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:genesis-owner:" + uuid4().hex
    )
    await storage.initialize()
    try:
        agent = await _agent(storage)
        _privacy(agent, storage, "isolated")
        for name in (
            "_persist_governance_receipt_node",
            "_persist_genesis_audit_completion",
            "_persist_genesis_audit_pending_attempt",
            "perform_genesis_audit",
        ):
            setattr(agent, name, getattr(ConstitutionMixin, name).__get__(agent))
        digest = await storage.store_file(
            resolve_governing_constitution_bytes(None), "constitution.md"
        )
        await _identity(storage, {"constitution_hash": digest})
        await agent._anchor_constitution_governance(digest)
        calls = []

        async def auditor(prompt):
            calls.append(prompt)
            assert storage.owns_open_transaction is False
            await storage.db.execute_commit(
                "DELETE FROM graph_edge_owners WHERE source_id=? AND target_id=? AND label='governed_by' AND agent_id=?",
                (agent.agent_id, digest, agent.agent_id),
            )
            return {
                "risk_level": 1,
                "reasoning": "Synthetic provider seam; real ownership deletion",
            }

        agent.get_audit_response = auditor
        with pytest.raises(GenesisAuditError, match="custody"):
            await agent.perform_genesis_audit()
        assert len(calls) == 1
        record = (await storage.get_node(agent.agent_id)).properties.get(
            "genesis_audit", {}
        )
        assert record.get("status") != "passed"
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("when", ["before-auditor", "during-auditor"])
async def test_genesis_cannot_attest_cached_bytes_over_corrupt_native_blob(
    db_backend, when
):
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:cached-genesis:" + uuid4().hex
    )
    await storage.initialize()
    physical = None
    try:
        agent = await _agent(storage)
        wrapper = _privacy(agent, storage, "isolated")
        for name in (
            "_persist_governance_receipt_node",
            "_persist_genesis_audit_completion",
            "_persist_genesis_audit_pending_attempt",
            "perform_genesis_audit",
        ):
            setattr(agent, name, getattr(ConstitutionMixin, name).__get__(agent))
        content = resolve_governing_constitution_bytes(None)
        digest = await storage.store_file(content, "constitution.md")
        await wrapper.store_file(content, "cached.md")
        physical = await storage.db.fetchone(
            "SELECT content,metadata FROM files WHERE content_hash=?", (digest,)
        )
        await _identity(storage, {"constitution_hash": digest})
        await agent._anchor_constitution_governance(digest)
        calls = []

        async def corrupt():
            await storage.db.execute_commit(
                "UPDATE files SET content=?,metadata=NULL WHERE content_hash=?",
                (b"corrupt native governance", digest),
            )

        async def auditor(prompt):
            calls.append(prompt)
            assert storage.owns_open_transaction is False
            await corrupt()
            return {
                "risk_level": 1,
                "reasoning": "Synthetic auditor; real native corruption",
            }

        agent.get_audit_response = auditor
        if when == "before-auditor":
            await corrupt()
        with pytest.raises(GenesisAuditError):
            await agent.perform_genesis_audit()
        assert len(calls) == (1 if when == "during-auditor" else 0)
        assert (await storage.get_node(agent.agent_id)).properties.get(
            "genesis_audit", {}
        ).get("status") != "passed"
    finally:
        if physical is not None:
            await storage.db.execute_commit(
                "UPDATE files SET content=?,metadata=? WHERE content_hash=?",
                (*physical, digest),
            )
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("existing", ["absent", "corrupt", "intact"])
async def test_isolated_bootstrap_publishes_exact_native_bytes_before_consuming_authority(
    db_backend, existing
):
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:isolated-bootstrap:" + uuid4().hex
    )
    await storage.initialize()
    physical = None
    try:
        agent = await _agent(storage)
        wrapper = _privacy(agent, storage, "isolated")
        await _identity(storage, {})
        content = resolve_governing_constitution_bytes(None)
        digest = hashlib.sha256(content).hexdigest()
        if existing != "absent":
            await storage.store_file(content, "constitution.md")
            physical = await storage.db.fetchone(
                "SELECT content,metadata FROM files WHERE content_hash=?", (digest,)
            )
            if existing == "corrupt":
                await storage.db.execute_commit(
                    "UPDATE files SET content=?,metadata=NULL WHERE content_hash=?",
                    (b"not the governing content", digest),
                )
        before = await agent._constitution_state_store.load(agent.agent_id)
        result = await ConstitutionMixin._get_governing_constitution(agent)
        wrapper._session_files.clear()
        durable = await agent._constitution_state_store.load(agent.agent_id)
        node = await storage.get_node(agent.agent_id)
        if existing == "corrupt":
            assert result.startswith("Error:"), result
            assert durable == before
            assert "constitution_hash" not in node.properties
        else:
            assert result == content.decode(), result
            assert await storage.retrieve_file(digest) == content
            assert await wrapper.retrieve_file(digest) == content
            assert node.properties["constitution_hash"] == digest
            assert durable.bootstrap_pending is False
    finally:
        if physical is not None:
            await storage.db.execute_commit(
                "UPDATE files SET content=?,metadata=? WHERE content_hash=?",
                (*physical, digest),
            )
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("same_hash", [False, True])
async def test_committed_repair_survives_native_conversation_insert_failure(
    db_backend, tmp_path, same_hash
):
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:repair-notice:" + uuid4().hex
    )
    await storage.initialize()
    trigger = "notice_" + uuid4().hex
    installed = False
    try:
        agent = await _agent(storage)
        _privacy(agent, storage, "normal")
        content = resolve_governing_constitution_bytes(None)
        digest = await storage.store_file(
            content if same_hash else content + b"\nold", "old.md"
        )
        await _identity(storage, {"constitution_hash": digest})
        await agent._anchor_constitution_governance(digest)
        artifact, root = _write_authority_files(tmp_path, content)
        agent._sovereign_trust_root_path = root
        if db_backend.backend_type == "sqlite":
            await storage.db.execute_commit(
                f"CREATE TRIGGER {trigger} BEFORE INSERT ON conversation_history BEGIN SELECT RAISE(ABORT, 'fixture native notice refusal'); END"
            )
        else:
            await storage.db.execute_commit(
                f"CREATE FUNCTION {trigger}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'fixture native notice refusal'; END $$"
            )
            await storage.db.execute_commit(
                f"CREATE TRIGGER {trigger} BEFORE INSERT ON conversation_history FOR EACH ROW EXECUTE FUNCTION {trigger}()"
            )
        installed = True
        result = await ConstitutionMixin.reanchor_constitution(
            agent, amendment_artifact_path=str(artifact)
        )
        assert not result.startswith("Error:"), result
        node = await storage.get_node(agent.agent_id)
        assert (
            node.properties["constitution_hash"] == hashlib.sha256(content).hexdigest()
        )
        assert node.properties["constitution_reanchor"]["signed_artifact_hash"]
    finally:
        if installed:
            await storage.db.execute_commit(
                f"DROP TRIGGER {trigger}"
                + (
                    " ON conversation_history"
                    if db_backend.backend_type == "postgres"
                    else ""
                )
            )
        if db_backend.backend_type == "postgres":
            await storage.db.execute_commit(f"DROP FUNCTION IF EXISTS {trigger}()")
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_governance_revalidation_is_independent_of_native_collation(
    db_backend, monkeypatch
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL actual ICU locale ordering")
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:governance-collation:" + uuid4().hex
    )
    await storage.initialize()
    try:
        await _identity(storage, {})
        stem = uuid4().hex
        targets = [stem + "a0", stem + "B0"]
        for target in targets:
            await storage.add_node(
                GraphNode(
                    node_id=target, node_type="document", label=target, properties={}
                )
            )
            await storage.add_edge(storage.agent_id, target, "governed_by")
        native_fetch = db_backend.fetch_all
        captured = []

        async def locale_order(query, params=()):
            # Execute real PG SQL with locale ordering where the production
            # query relies on a default collation; never fabricate returned rows.
            if query.startswith("SELECT target_id") and "ORDER BY target_id" in query:
                if 'COLLATE "C"' not in query:
                    query = query.replace(
                        "ORDER BY target_id", 'ORDER BY target_id COLLATE "en-US-x-icu"'
                    )
                rows = await native_fetch(query, params)
                captured.append(rows)
                return rows
            return await native_fetch(query, params)

        monkeypatch.setattr(db_backend, "fetch_all", locale_order)
        expected = governance_evidence({}, targets)
        async with storage.transaction():
            await storage.lock_nodes_for_update([storage.agent_id, *targets])
            await revalidate_governance_evidence(storage, storage.agent_id, expected)
        assert captured
    finally:
        await storage.close()
