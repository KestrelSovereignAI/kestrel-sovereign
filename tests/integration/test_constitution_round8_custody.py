"""Native lost-state recovery and receipt-preserving doctrine writers."""
import asyncio
import hashlib
from collections import OrderedDict
from contextlib import suppress
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.agent.doctrine_bundle import (
    DoctrineBundleError, anchor_doctrine_bundle, reanchor_doctrine_bundle,
)
from kestrel_sovereign.constitution.genesis_audit import (
    GenesisAuditRejectedError, pending_genesis_audit,
)
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from tests.integration.test_constitution_refusal_races import _agent


async def _seed(storage):
    agent = await _agent(storage)
    content = resolve_governing_constitution_bytes(None)
    digest = await storage.store_file(content, "constitution.md")
    await storage.add_node(GraphNode(
        node_id=storage.agent_id, node_type="agent", label="receipt custody",
        properties={"constitution_hash": digest,
                    "genesis_audit": pending_genesis_audit(digest, provenance="test:local")},
    ))
    await agent._anchor_constitution_governance(digest)
    return agent


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("new_identity", [False, True])
async def test_missing_runtime_row_with_history_remains_restricted(db_backend, new_identity):
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:lost-state:" + uuid4().hex)
    await storage.initialize()
    try:
        first = await _seed(storage)
        assert await first.enter_safe_mode("restriction that must survive row loss")
        before = await first._constitution_state_store.load(storage.agent_id)
        events = await first._constitution_state_store.list_events(storage.agent_id)
        await storage.db.execute_commit("DELETE FROM constitution_runtime_state WHERE agent_id=?", (storage.agent_id,))
        fresh = await _agent(storage, is_new_identity=new_identity)
        assert fresh._safe_mode is True
        assert fresh._constitution_bootstrap_pending is False
        recovered = await fresh._constitution_state_store.load(storage.agent_id)
        assert recovered.safe_mode is True
        assert recovered.generation != before.generation
        assert recovered.last_successful_audit_at is None
        assert "surviving transition history" in recovered.safe_mode_reason
        assert (await fresh._constitution_state_store.list_events(storage.agent_id))[:-1] == events
        assert await fresh.exit_safe_mode() == "Safe Mode remains active: explicit Sovereign authorization is required."
        await fresh._audit_constitution_on_startup()
        assert fresh._safe_mode is True
        assert (await fresh._constitution_state_store.load(storage.agent_id)).safe_mode is True
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["dispatcher", "anchor", "reanchor"])
@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL])
async def test_doctrine_snapshot_cannot_erase_actual_completed_genesis(db_backend, tmp_path, monkeypatch, writer, mode):
    if writer == "reanchor" and mode == PrivacyMode.EPHEMERAL:
        pytest.skip("Explicit free-text ratification remains denied in volatile mode")
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:doctrine-receipt:" + uuid4().hex)
    await storage.initialize()
    task = None
    reached, release = asyncio.Event(), asyncio.Event()
    try:
        agent = await _seed(storage)
        if writer == "reanchor":
            node = await storage.get_node(storage.agent_id)
            node.properties["doctrine_bundle_hash"] = "a" * 64
            await storage.add_node(node)
        # Explicit ratification introduces a free-text receipt and remains
        # default-denied in volatile mode; do not weaken that privacy rule.
        agent.storage = PrivacyEnforcingStorage(storage, mode)
        peer = await _agent(storage, is_new_identity=False)
        for name in ("perform_genesis_audit", "_genesis_publication_witness",
                     "_persist_genesis_audit_completion", "_persist_governance_receipt_node"):
            setattr(peer, name, getattr(ConstitutionMixin, name).__get__(peer))
        peer.get_audit_response = AsyncMock(return_value={"risk_level": 3, "reasoning": "Synthetic native rejection"})
        native_get = agent.storage.get_node

        async def read_before_concurrent_completion(identity):
            node = await native_get(identity)
            if asyncio.current_task() is task and not reached.is_set():
                reached.set()
                await release.wait()
            return node

        monkeypatch.setattr(agent.storage, "get_node", read_before_concurrent_completion)
        agent.compute_live_doctrine_bundle_hash = AsyncMock(return_value=hashlib.sha256(b"").hexdigest())
        agent._resolve_project_root_for_doctrine = AsyncMock(return_value=None)

        async def write():
            if writer == "dispatcher":
                return await ConstitutionMixin.ensure_doctrine_bundle_anchored(agent)
            if writer == "anchor":
                return await anchor_doctrine_bundle(agent, project_root=tmp_path, bootstrap_files=OrderedDict())
            return await reanchor_doctrine_bundle(
                agent, project_root=tmp_path, bootstrap_files=OrderedDict(),
                expected_hash=hashlib.sha256(b"").hexdigest(), authorization="explicit local owner",
            )

        task = asyncio.create_task(write())
        await asyncio.wait_for(reached.wait(), 10)
        with pytest.raises(GenesisAuditRejectedError):
            await peer.perform_genesis_audit()
        completed = (await storage.get_node(storage.agent_id)).properties["genesis_audit"]
        assert completed["status"] == "failed"
        release.set()
        refusal = None
        try:
            result = await asyncio.wait_for(task, 10)
        except DoctrineBundleError as exc:
            refusal = exc
            result = None
        assert (await storage.get_node(storage.agent_id)).properties["genesis_audit"] == completed
        assert result is None
        if writer != "dispatcher":
            assert refusal is not None and "snapshot changed" in str(refusal)
        peer.get_audit_response.reset_mock()
        with pytest.raises(GenesisAuditRejectedError):
            await peer.perform_genesis_audit()
        peer.get_audit_response.assert_not_awaited()
    finally:
        release.set()
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        await storage.close()
