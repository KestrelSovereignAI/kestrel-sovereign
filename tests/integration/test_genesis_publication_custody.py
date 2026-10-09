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
            await storage.add_node(GraphNode(node_id=node_id, node_type="agent" if node_id == identity else "note", label="native deletion", properties={}))
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
