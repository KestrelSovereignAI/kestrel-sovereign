"""Auditor completion must not overwrite a newer native graph witness."""

from copy import deepcopy
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_refusal_races import _agent


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_changed_hash_genesis_can_complete_in_safe_mode_before_explicit_exit(
    db_backend, tmp_path
):
    from kestrel_sovereign.features.privacy.feature import PrivacyAgent
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    from tests.integration.test_constitution_reanchor_e2e import _write_authority_files

    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:genesis-recovery:" + uuid4().hex
    )
    await storage.initialize()
    try:
        agent = await _agent(storage)
        for name in (
            "_persist_governance_receipt_node",
            "_persist_genesis_audit_completion",
            "_persist_genesis_audit_pending_attempt",
            "_ensure_genesis_audit_ready",
            "perform_genesis_audit",
        ):
            setattr(agent, name, getattr(ConstitutionMixin, name).__get__(agent))
        agent.privacy_agent = PrivacyAgent(
            PrivacyEnforcingStorage(storage, "isolated"), "isolated"
        )
        content = resolve_governing_constitution_bytes(None)
        old = await storage.store_file(content + b"\nold governing revision", "old.md")
        await storage.add_node(
            GraphNode(
                node_id=agent.agent_id,
                node_type="agent",
                label="recovery",
                properties={"constitution_hash": old},
            )
        )
        await agent._anchor_constitution_governance(old)
        await agent.enter_safe_mode("old governing drift")
        artifact, root = _write_authority_files(tmp_path, content)
        agent._sovereign_trust_root_path = root
        result = await ConstitutionMixin.reanchor_constitution(
            agent, amendment_artifact_path=str(artifact)
        )
        assert not result.startswith("Error:"), result
        assert (await storage.get_node(agent.agent_id)).properties["genesis_audit"][
            "status"
        ] == "pending"
        assert (
            await agent.exit_safe_mode(authorization="fixture sovereign")
        ).startswith("Safe Mode remains active:")
        calls = []

        async def auditor(prompt):
            # Only the provider-bearing seam is injected. Native receipt,
            # graph, CAS, privacy notification and exit custody are real.
            calls.append(prompt)
            assert storage.owns_open_transaction is False
            assert agent._safe_mode is True
            return {
                "risk_level": 1,
                "reasoning": "Injected recovery plumbing verdict, not live acceptance",
            }

        agent.get_audit_response = auditor
        assert (
            await ConstitutionMixin._genesis_audit_cognition_block(
                agent, "ordinary request"
            )
            is None
        )
        assert agent._safe_mode is True
        assert (await storage.get_node(agent.agent_id)).properties["genesis_audit"][
            "status"
        ] == "passed"
        assert "deactivated" in await agent.exit_safe_mode(
            authorization="fixture sovereign"
        )
        assert await agent.perform_genesis_audit() is True
        assert len(calls) == 1
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("mode", ["isolated", "ephemeral"])
@pytest.mark.parametrize("failure", ["stale-fence", "runtime-write"])
async def test_refused_genesis_does_not_publish_actual_volatile_conversation(
    db_backend, mode, failure
):
    from kestrel_sovereign.features.privacy.feature import PrivacyAgent
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    from kestrel_sovereign.constitution.genesis_audit import GenesisAuditError
    from dataclasses import replace

    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:volatile-genesis:" + uuid4().hex
    )
    await storage.initialize()
    try:
        agent = await _agent(storage)
        agent._persist_governance_receipt_node = (
            ConstitutionMixin._persist_governance_receipt_node.__get__(agent)
        )
        agent.privacy_agent = PrivacyAgent(PrivacyEnforcingStorage(storage, mode), mode)
        digest = await storage.store_file(
            resolve_governing_constitution_bytes(None), "KESTREL_CONSTITUTION.md"
        )
        await storage.add_node(
            GraphNode(
                node_id=agent.agent_id,
                node_type="agent",
                label="volatile",
                properties={"constitution_hash": digest},
            )
        )
        await agent._anchor_constitution_governance(digest)
        node = await storage.get_node(agent.agent_id)
        expected = await ConstitutionMixin._genesis_publication_witness(agent, node)
        before = dict(node.properties)
        store = agent._constitution_state_store
        if failure == "stale-fence":
            current = await store.load(agent.agent_id)
            await store.write(
                replace(current, updated_at=agent._constitution_now()),
                event_type="test concurrent transition",
            )
        else:

            async def rejected_write(*args, **kwargs):
                raise RuntimeError("injected publication write failure")

            store.write = rejected_write
        record = {
            "status": "passed",
            "risk_level": 1,
            "reasoning": "Native notification plumbing fixture",
            "constitution_hash": digest,
        }
        with pytest.raises(GenesisAuditError):
            await ConstitutionMixin._persist_genesis_audit_completion(
                agent, node, record, expected=expected
            )
        assert (await storage.get_node(agent.agent_id)).properties == before
        assert agent.privacy_agent.isolated_session == []
        if mode == "ephemeral":
            assert agent.privacy_agent.ephemeral_session.messages == []
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_native_edge_deletion_respects_graph_reservations(db_backend):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL independent native graph deletion custody")
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    identity, target = "did:test:delete-graph:" + uuid4().hex, uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    peer_backend = PostgresBackend(db_backend._dsn, min_pool_size=1, max_pool_size=1)
    await peer_backend.connect()
    peer = AsyncGraphStore(AsyncDatabase(peer_backend), agent_id=identity)
    try:
        for node_id in (identity, target):
            await storage.add_node(
                GraphNode(
                    node_id=node_id,
                    node_type="agent" if node_id == identity else "note",
                    label="native deletion",
                    properties={},
                )
            )
        await storage.add_edge(identity, target, "fixture-edge")
        async with storage.transaction():
            await storage.lock_nodes_for_update([identity, target])
            with pytest.raises(Exception, match="lock timeout"):
                async with peer_backend.transaction():
                    await peer_backend.execute("SET LOCAL lock_timeout = '100ms'")
                    await peer.delete_edge(identity, target, "fixture-edge")
        assert len(await storage.get_edges_from(identity)) == 1
        await peer.delete_edge(identity, target, "fixture-edge")
        assert await storage.get_edges_from(identity) == []
    finally:
        await peer_backend.close()
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("changed", ["metadata", "governance", "receipt"])
@pytest.mark.parametrize("entrypoint", ["explicit", "cognition"])
async def test_native_genesis_publication_preserves_or_refuses_newer_node(
    db_backend,
    monkeypatch,
    changed,
    entrypoint,
):
    storage = AsyncStorage(
        backend=db_backend, agent_id="did:test:genesis-custody:" + uuid4().hex
    )
    await storage.initialize()
    try:
        agent = await _agent(storage)
        for name in (
            "_persist_governance_receipt_node",
            "_persist_genesis_audit_completion",
            "_persist_genesis_audit_pending_attempt",
            "_ensure_genesis_audit_ready",
            "perform_genesis_audit",
        ):
            setattr(agent, name, getattr(ConstitutionMixin, name).__get__(agent))
        digest = await storage.store_file(
            resolve_governing_constitution_bytes(None), "KESTREL_CONSTITUTION.md"
        )
        await storage.add_node(
            GraphNode(
                node_id=agent.agent_id,
                node_type="agent",
                label="genesis",
                properties={"constitution_hash": digest, "metadata": "original"},
            )
        )
        await agent._anchor_constitution_governance(digest)
        winner = None

        async def actual_auditor(_prompt):
            nonlocal winner
            node = await storage.get_node(agent.agent_id)
            if changed == "metadata":
                node.properties["metadata"] = "concurrent update"
            elif changed == "governance":
                node.properties["constitution_hash"] = "f" * 64
                node.properties["constitution_reanchor_history"] = [
                    {"preserved": "winner"}
                ]
            else:
                node.properties["genesis_audit"] = {
                    "status": "failed",
                    "constitution_hash": digest,
                    "risk_level": 3,
                    "completed_at": "2026-10-09T21:00:00Z",
                    "audited": True,
                }
            await storage.add_node(node)
            winner = deepcopy(node.properties)
            return {
                "risk_level": 1,
                "reasoning": "Injected native plumbing result only.",
            }

        agent.get_audit_response = actual_auditor

        async def run():
            if entrypoint == "explicit":
                return await ConstitutionMixin.perform_genesis_audit(agent)
            return await ConstitutionMixin._genesis_audit_cognition_block(
                agent, "ordinary cognition"
            )

        if changed == "metadata":
            assert await run() is (True if entrypoint == "explicit" else None)
            assert (await storage.get_node(agent.agent_id)).properties[
                "metadata"
            ] == "concurrent update"
        else:
            if entrypoint == "explicit":
                from kestrel_sovereign.constitution.genesis_audit import (
                    GenesisAuditError,
                )

                with pytest.raises(
                    GenesisAuditError, match="governing evidence changed"
                ) as refusal:
                    await run()
                assert refusal.value.__cause__ is not None
            else:
                result = await run()
                assert "GENESIS AUDIT BLOCKED" in result, result
                assert "No cognition request was sent" in result
            assert (await storage.get_node(agent.agent_id)).properties == winner
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("entrypoint", ["explicit", "cognition"])
async def test_native_edge_deletion_waits_for_genesis_publication(
    db_backend,
    monkeypatch,
    entrypoint,
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL independent native edge producer")
    from kestrel_sovereign.constitution import anchored_bytes
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    identity = "did:test:genesis-edge-custody:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    peer_backend = PostgresBackend(db_backend._dsn, min_pool_size=1, max_pool_size=1)
    await peer_backend.connect()
    peer = AsyncGraphStore(AsyncDatabase(peer_backend), agent_id=identity)
    native_revalidate = anchored_bytes.revalidate_governance_evidence
    observed = []
    try:
        agent = await _agent(storage)
        for name in (
            "_persist_governance_receipt_node",
            "_persist_genesis_audit_completion",
            "_persist_genesis_audit_pending_attempt",
            "_ensure_genesis_audit_ready",
            "perform_genesis_audit",
        ):
            setattr(agent, name, getattr(ConstitutionMixin, name).__get__(agent))
        digest = await storage.store_file(
            resolve_governing_constitution_bytes(None), "KESTREL_CONSTITUTION.md"
        )
        await storage.add_node(
            GraphNode(
                node_id=identity,
                node_type="agent",
                label="genesis",
                properties={"constitution_hash": digest},
            )
        )
        await agent._anchor_constitution_governance(digest)

        async def fixture_auditor(_prompt):
            return {"risk_level": 1, "reasoning": "Deterministic custody test only."}

        agent.get_audit_response = fixture_auditor

        async def checked_then_delete(raw, agent_id, expected):
            fresh = await native_revalidate(raw, agent_id, expected)
            with pytest.raises(Exception, match="lock timeout"):
                async with peer_backend.transaction():
                    await peer_backend.execute("SET LOCAL lock_timeout = '100ms'")
                    await peer.delete_edge(identity, digest, "governed_by")
            observed.append(True)
            return fresh

        monkeypatch.setattr(
            anchored_bytes, "revalidate_governance_evidence", checked_then_delete
        )
        if entrypoint == "explicit":
            assert await ConstitutionMixin.perform_genesis_audit(agent) is True
        else:
            assert (
                await ConstitutionMixin._genesis_audit_cognition_block(
                    agent, "ordinary cognition"
                )
                is None
            )
        assert observed
        assert (await storage.get_node(identity)).properties["genesis_audit"][
            "status"
        ] == "passed"
        # The independent writer succeeds once publication releases custody.
        await peer.delete_edge(identity, digest, "governed_by")
        assert not [
            edge
            for edge in await storage.get_edges_from(identity)
            if edge.label == "governed_by"
        ]
    finally:
        await peer_backend.close()
        await storage.close()
