"""Real native SQL races and fail-closed malformed historical receipt repair."""

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin, SafeModeCause
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.setup import constitution_reanchor as offline
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files
from tests.unit.test_constitution_audit import _DurableConstitutionHarness


async def _agent(storage):
    agent = _DurableConstitutionHarness(storage, datetime.now(timezone.utc))
    agent.agent_id = storage.agent_id
    await agent._initialize_constitution_runtime_state(is_new_identity=True)
    agent.extension = None
    for name in (
        "_anchor_constitution_governance", "_agent_signing_dids",
        "_trusted_sovereign_did_document", "verify_constitution_overlay",
        "_verify_spawn_mandate_constraints", "_verify_constitution_integrity",
    ):
        setattr(agent, name, getattr(KestrelAgent, name).__get__(agent))
    return agent


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("boundary", ["blocked_update", "committed_release"])
async def test_postgres_refusal_during_exit_write_cannot_publish_success(
    db_backend, tmp_path, monkeypatch, boundary,
):
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL row-lock and connection-release ordering")
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:refusal-race:" + uuid4().hex)
    await storage.initialize()
    exit_task = None
    proceed = asyncio.Event()
    try:
        agent = await _agent(storage)
        digest = await storage.store_file(resolve_governing_constitution_bytes(None), "KESTREL_CONSTITUTION.md")
        await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="exit", properties={"constitution_hash": digest}))
        await agent._anchor_constitution_governance(digest)
        await agent.enter_safe_mode("old integrity restriction")
        before = await agent._constitution_state_store.load(agent.agent_id)
        events_before = await agent._constitution_state_store.list_events(agent.agent_id)
        reached = asyncio.Event()
        native_fetch = db_backend.fetch_one
        native_transaction = db_backend.transaction
        writer_pid = None
        depth = 0

        async def observed_fetch(sql, params=()):
            nonlocal writer_pid
            if boundary == "blocked_update" and asyncio.current_task() is exit_task and "UPDATE constitution_runtime_state SET" in sql:
                writer_pid = (await native_fetch("SELECT pg_backend_pid()"))[0]
                reached.set()
            return await native_fetch(sql, params)

        @asynccontextmanager
        async def observed_transaction():
            nonlocal depth
            is_exit = asyncio.current_task() is exit_task
            if is_exit:
                depth += 1
            outer = is_exit and depth == 1
            try:
                async with native_transaction():
                    yield
                if outer and boundary == "committed_release":
                    # The native commit really finished. Pause only delivery
                    # of its result, modelling awaited pool/lease release.
                    reached.set()
                    await proceed.wait()
            finally:
                if is_exit:
                    depth -= 1

        monkeypatch.setattr(db_backend, "fetch_one", observed_fetch)
        if boundary == "committed_release":
            monkeypatch.setattr(db_backend, "transaction", observed_transaction)

        if boundary == "blocked_update":
            async with storage.transaction():
                await native_fetch(
                    "SELECT agent_id FROM constitution_runtime_state WHERE agent_id = ? FOR UPDATE",
                    (agent.agent_id,),
                )
                exit_task = asyncio.create_task(agent.exit_safe_mode(authorization="test sovereign"))
                await asyncio.wait_for(reached.wait(), 5)
                # Observe the real blocked UPDATE, not merely its call seam.
                async with asyncio.timeout(5):
                    while True:
                        row = await native_fetch(
                            "SELECT wait_event_type FROM pg_stat_activity WHERE pid = ?", (writer_pid,)
                        )
                        if row is not None and row[0] == "Lock":
                            break
                        await asyncio.sleep(0.02)
                assert await agent.enter_safe_mode(
                    "new feature quarantine", cause=SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value,
                ) is False
        else:
            exit_task = asyncio.create_task(agent.exit_safe_mode(authorization="test sovereign"))
            await asyncio.wait_for(reached.wait(), 5)
            async with storage.transaction():
                assert await agent.enter_safe_mode(
                    "new feature quarantine", cause=SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value,
                ) is False
            proceed.set()
        result = await asyncio.wait_for(exit_task, 5)
        assert result.startswith("Safe Mode remains active:"), result
        assert agent._safe_mode is True
        assert agent._safe_mode_cause == SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value
        assert agent._feature_lifecycle_integrity_uncertain is True
        assert agent._feature_lifecycle_repair_verified is False
        assert agent._constitution_state_persistence_pending is True
        if boundary == "blocked_update":
            # Invalidation before commit rolls back BOTH state and exit event.
            assert await agent._constitution_state_store.load(agent.agent_id) == before
            assert await agent._constitution_state_store.list_events(agent.agent_id) == events_before
        else:
            # A refusal during committed-result delivery is volatile, never a
            # claim that the newly requested restriction reached durable SQL.
            durable = await agent._constitution_state_store.load(agent.agent_id)
            assert durable.safe_mode is False
            assert durable.revision == before.revision + 1
    finally:
        proceed.set()
        if exit_task is not None and not exit_task.done():
            exit_task.cancel()
        if exit_task is not None:
            with suppress(asyncio.CancelledError):
                await exit_task
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("writer", ["runtime", "offline"])
@pytest.mark.parametrize("kind,history,missing_field", [
    ("constitution_reanchor", False, "new_hash"),
    ("constitution_reanchor", True, "new_hash"),
    ("constitution_reanchor", True, "superseded_by_constitution_hash"),
    ("genesis_audit", True, "superseded_by_constitution_hash"),
    ("genesis_audit", False, "constitution_hash"),
])
@pytest.mark.parametrize("null_destination", [False, True])
async def test_public_repair_refuses_receipt_without_required_hash(
    db_backend, tmp_path, monkeypatch, writer, kind, history, missing_field, null_destination,
):
    storage = (
        AsyncStorage(str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id="did:test:bad-receipt:" + uuid4().hex)
        if db_backend.backend_type == "sqlite"
        else AsyncStorage(backend=db_backend, agent_id="did:test:bad-receipt:" + uuid4().hex)
    )
    await storage.initialize()
    closed = False
    try:
        agent = await _agent(storage)
        content = resolve_governing_constitution_bytes(None)
        old_hash = await storage.store_file(content, "superseded-dormant.md")
        receipt = (
            {"old_hash": "none", "new_hash": old_hash}
            if kind == "constitution_reanchor"
            else {"constitution_hash": old_hash}
        )
        entry = {"receipt": receipt, "superseded_by_constitution_hash": old_hash}
        malformed = entry if missing_field == "superseded_by_constitution_hash" else receipt
        malformed.pop(missing_field)
        if null_destination:
            malformed[missing_field] = None
        key = kind + "_history" if history else kind
        props = {key: [entry] if history else receipt}
        await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="malformed receipt", properties=props))
        before = await agent._constitution_state_store.load(agent.agent_id)
        events_before = await agent._constitution_state_store.list_events(agent.agent_id)
        artifact, root = _write_authority_files(tmp_path, content)
        if writer == "runtime":
            agent._sovereign_trust_root_path = root
            result = await ConstitutionMixin.reanchor_constitution(agent, amendment_artifact_path=str(artifact))
            assert result.startswith("Error:") and "unreadable historical governance" in result, result
            restored = storage
        else:
            target = (
                offline.ReanchorTarget(Path(storage._backend.db_path), "sqlite", agent.agent_id)
                if db_backend.backend_type == "sqlite"
                else offline.ReanchorTarget(None, "postgres", agent.agent_id, db_backend._dsn)
            )
            await storage.close()
            closed = True

            async def exact_target(*args, **kwargs):
                return target

            @asynccontextmanager
            async def no_embedding(*args, **kwargs):
                yield None

            monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
            monkeypatch.setattr(offline, "_agent_embedding", no_embedding)
            result = await offline.reanchor_constitution(
                agent_name="malformed receipt", agent_dir=tmp_path if target.anchor_path else None,
                force=True, sovereign_trust_root_path=root, amendment_artifact_path=artifact,
                runtime_backend=target.backend, runtime_dsn=target.dsn,
                hosted_agent_did=agent.agent_id if target.backend == "postgres" else None, environ={},
            )
            assert not result.reanchored and "unreadable historical governance" in (result.error or ""), result.error
            async with target.open_storage() as restored:
                from kestrel_sovereign.constitution.runtime_state import ConstitutionRuntimeStateStore
                store = ConstitutionRuntimeStateStore(restored._backend)
                assert (await restored.graph.get_node(agent.agent_id)).properties == props
                assert await store.load(agent.agent_id) == before
                assert await store.list_events(agent.agent_id) == events_before
            return
        assert (await restored.get_node(agent.agent_id)).properties == props
        assert await agent._constitution_state_store.load(agent.agent_id) == before
        assert await agent._constitution_state_store.list_events(agent.agent_id) == events_before
    finally:
        if not closed:
            await storage.close()
