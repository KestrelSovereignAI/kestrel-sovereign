"""Native cross-replica restrictions and hash-bound direct/streamed admission."""

import asyncio
from contextlib import asynccontextmanager, suppress
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.genesis_audit import evaluate_genesis_constitution
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode
from tests.integration.test_constitution_refusal_races import _agent
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files
from tests.unit.test_non_streaming_turn_privacy_span import _turn_agent


async def _ready_turn(storage, *, is_new_identity=True):
    native = await _agent(storage, is_new_identity=is_new_identity)
    content = resolve_governing_constitution_bytes(None)
    digest = await storage.store_file(content, "constitution.md")

    async def auditor(prompt):
        return {"risk_level": 1, "reasoning": "Synthetic native receipt"}

    receipt = await evaluate_genesis_constitution(content, constitution_hash=digest, auditor=auditor, provenance="test:turn-admission")
    await storage.add_node(GraphNode(node_id=storage.agent_id, node_type="agent", label="turn", properties={"constitution_hash": digest, "genesis_audit": receipt}))
    await native._anchor_constitution_governance(digest)
    assert await native._record_successful_constitution_audit(source="native fixture")
    agent = _turn_agent(storage.agent_id)
    agent.extension = None
    agent.storage = agent._raw_storage = storage
    for name, value in vars(native).items():
        if name.startswith(("_constitution", "_safe_mode")) or name in ("_interaction_count", "_last_audit_time"):
            setattr(agent, name, value)
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent)
    agent._genesis_audit_cognition_block = ConstitutionMixin._genesis_audit_cognition_block.__get__(agent)
    agent._ensure_genesis_audit_ready = ConstitutionMixin._ensure_genesis_audit_ready.__get__(agent)
    return agent, content


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("change", ["queued-restriction", "admitted-signed-repair"])
async def test_turn_cannot_outrun_native_restriction_or_governing_receipt(db_backend, tmp_path, monkeypatch, streaming, change):
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:turn-admission:" + uuid4().hex)
    await storage.initialize()
    task = None
    reached, release = asyncio.Event(), asyncio.Event()
    try:
        agent, content = await _ready_turn(storage)
        cognition = AsyncMock(side_effect=AssertionError("restricted turn reached cognition"))

        async def body(user_input, *args, **kwargs):
            if change == "queued-restriction":
                return await cognition(user_input)
            reached.set()
            await release.wait()
            # This is the REAL producer used by both traced turn bodies. The
            # gate has already passed, but no model may consume unaudited J.
            result = await ConstitutionMixin._get_governing_constitution(agent)
            if not result.startswith("Error:"):
                await cognition(user_input)
            return result

        async def streamed_body(*args, **kwargs):
            yield await body(*args, **kwargs)

        agent._process_input_traced_locked = body
        agent._process_input_streaming_traced_locked = streamed_body
        if change == "queued-restriction":
            lifecycle = agent._turn_lifecycle

            @asynccontextmanager
            async def queued():
                reached.set()
                await release.wait()
                async with lifecycle() as turn:
                    yield turn

            agent._turn_lifecycle = queued

        async def request():
            if streaming:
                return "".join([part async for part in agent.process_input_streaming("hello")])
            return await agent.process_input("hello")

        task = asyncio.create_task(request())
        await asyncio.wait_for(reached.wait(), 10)
        replica = await _agent(storage)
        if change == "queued-restriction":
            assert await replica.enter_safe_mode("Another replica committed an integrity restriction")
            assert (await replica._constitution_state_store.load(storage.agent_id)).safe_mode
        else:
            replacement = content + b"\nNew signed governing revision for the admission race.\n"
            canonical = tmp_path / "KESTREL_CONSTITUTION.md"
            canonical.write_bytes(replacement)
            import kestrel_sovereign.config as config

            monkeypatch.setattr(config, "CONSTITUTION_PATH", str(canonical))
            artifact, root = _write_authority_files(tmp_path, replacement)
            replica._sovereign_trust_root_path = root
            repaired = await ConstitutionMixin.reanchor_constitution(replica, amendment_artifact_path=str(artifact))
            assert not repaired.startswith("Error:"), repaired
            assert (await storage.get_node(storage.agent_id)).properties["genesis_audit"]["status"] == "pending"
        release.set()
        result = await asyncio.wait_for(task, 10)
        if change == "queued-restriction":
            assert "SAFE MODE" in result.upper(), result
        else:
            assert result.startswith("Error:") and "changed after" in result, result
        cognition.assert_not_awaited()
    finally:
        release.set()
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_admission_retains_runtime_row_until_graph_snapshot_is_captured(
    db_backend, monkeypatch,
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL independent runtime-row custody")
    import asyncpg

    storage = AsyncStorage(backend=db_backend, agent_id="did:test:turn-custody:" + uuid4().hex)
    await storage.initialize()
    peer = probe = None
    capture_task = restriction_task = None
    release, captured, updating = asyncio.Event(), asyncio.Event(), asyncio.Event()
    try:
        agent, _ = await _ready_turn(storage)
        peer = AsyncStorage(backend="postgres", dsn=db_backend._dsn, agent_id=storage.agent_id)
        await peer.initialize()
        replica = await _agent(peer, is_new_identity=False)
        probe = await asyncpg.connect(db_backend._dsn)
        store = agent._constitution_state_store
        native_load = store.load
        native_fetch = peer._backend.fetch_one
        pid = None

        async def captured_runtime(identity):
            state = await native_load(identity)
            captured.set()
            await release.wait()
            return state

        async def observed_update(query, params=()):
            nonlocal pid
            if "UPDATE constitution_runtime_state SET" in query:
                pid = (await native_fetch("SELECT pg_backend_pid()"))[0]
                updating.set()
            return await native_fetch(query, params)

        monkeypatch.setattr(store, "load", captured_runtime)
        monkeypatch.setattr(peer._backend, "fetch_one", observed_update)
        capture_task = asyncio.create_task(ConstitutionMixin._locked_turn_governance(agent, store))
        await asyncio.wait_for(captured.wait(), 5)
        restriction_task = asyncio.create_task(replica.enter_safe_mode("Independent committed restriction"))
        await asyncio.wait_for(updating.wait(), 5)
        async with asyncio.timeout(5):
            while await probe.fetchval(
                "SELECT wait_event_type FROM pg_stat_activity WHERE pid=$1", pid,
            ) != "Lock":
                await asyncio.sleep(0.02)
        assert not restriction_task.done()
        release.set()
        state, digest, receipt = await asyncio.wait_for(capture_task, 5)
        assert state.safe_mode is False
        assert receipt["constitution_hash"] == digest
        assert receipt["status"] == "passed"
        assert await asyncio.wait_for(restriction_task, 5) is True
        assert (await native_load(storage.agent_id)).safe_mode is True
    finally:
        release.set()
        for task in (capture_task, restriction_task):
            if task is not None:
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        if probe is not None:
            await probe.close()
        if peer is not None:
            await peer.close()
        await storage.close()
