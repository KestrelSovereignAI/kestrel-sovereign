"""Real direct and streamed entrypoints recheck native genesis after queuing."""

import asyncio
from contextlib import asynccontextmanager, suppress
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.genesis_audit import (
    evaluate_genesis_constitution,
    supersede_genesis_audit,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.unit.test_non_streaming_turn_privacy_span import _turn_agent
from tests.integration.test_constitution_refusal_races import _agent
from kestrel_sovereign.storage.async_graph_store import GraphNode


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("streaming", [False, True])
async def test_queued_cognition_cannot_use_a_new_pending_governing_hash(
    db_backend, streaming
):
    identity = "did:test:queued-genesis:" + uuid4().hex
    storage = AsyncStorage(backend=db_backend, agent_id=identity)
    await storage.initialize()
    task = None
    release = asyncio.Event()
    entered_queue = asyncio.Event()
    try:
        native = await _agent(storage)
        old = ("old governing bytes " + uuid4().hex).encode()
        digest = await storage.store_file(old, "old.md")

        async def safe_auditor(prompt):
            return {"risk_level": 1, "reasoning": "Synthetic native receipt"}

        receipt = await evaluate_genesis_constitution(
            old,
            constitution_hash=digest,
            auditor=safe_auditor,
            provenance="test:queued-admission",
        )
        await storage.add_node(
            GraphNode(
                node_id=identity,
                node_type="agent",
                label="queue",
                properties={"constitution_hash": digest, "genesis_audit": receipt},
            )
        )
        await native._anchor_constitution_governance(digest)

        agent = _turn_agent(identity)
        agent.storage = agent._raw_storage = storage
        agent._genesis_audit_cognition_block = (
            ConstitutionMixin._genesis_audit_cognition_block.__get__(agent)
        )
        agent._ensure_genesis_audit_ready = (
            ConstitutionMixin._ensure_genesis_audit_ready.__get__(agent)
        )
        agent._persist_governance_receipt_node = (
            ConstitutionMixin._persist_governance_receipt_node.__get__(agent)
        )
        agent._persist_genesis_audit_pending_attempt = (
            ConstitutionMixin._persist_genesis_audit_pending_attempt.__get__(agent)
        )
        agent.perform_genesis_audit = ConstitutionMixin.perform_genesis_audit.__get__(
            agent
        )
        agent.get_audit_response = AsyncMock(
            return_value={
                "risk_level": 1,
                "reasoning": "No auditor available",
                "audited": False,
            }
        )
        cognition = AsyncMock(
            side_effect=AssertionError("pending new hash must not reach cognition")
        )
        agent._process_input_traced_locked = cognition

        async def forbidden_stream(*args, **kwargs):
            await cognition(*args, **kwargs)
            yield "unreachable"

        agent._process_input_streaming_traced_locked = forbidden_stream
        lifecycle = agent._turn_lifecycle

        @asynccontextmanager
        async def queued_lifecycle():
            entered_queue.set()
            await release.wait()
            async with lifecycle() as turn:
                yield turn

        agent._turn_lifecycle = queued_lifecycle

        async def request():
            if streaming:
                return "".join(
                    [part async for part in agent.process_input_streaming("hello")]
                )
            return await agent.process_input("hello")

        task = asyncio.create_task(request())
        await asyncio.wait_for(entered_queue.wait(), 5)
        # Commit the native governance generation change while the request is
        # queued. This is the persisted state a changed-hash repair publishes.
        new = ("new governing bytes " + uuid4().hex).encode()
        new_digest = await storage.store_file(new, "new.md")
        node = await storage.get_node(identity)
        node.properties["constitution_hash"] = new_digest
        supersede_genesis_audit(
            node.properties,
            constitution_hash=new_digest,
            provenance="test:committed-governance-change",
            recorded_at=native._get_timestamp(),
        )
        async with storage.transaction():
            await storage.lock_nodes_for_update([identity, digest, new_digest])
            await native._anchor_constitution_governance(new_digest)
            await storage.add_node(node)
        release.set()
        result = await asyncio.wait_for(task, 10)
        assert "GENESIS AUDIT" in result and (
            "PENDING" in result or "BLOCKED" in result
        )
        cognition.assert_not_awaited()
        agent.get_audit_response.assert_awaited_once()
        assert (await storage.get_node(identity)).properties["genesis_audit"][
            "status"
        ] == "pending"
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await storage.close()
