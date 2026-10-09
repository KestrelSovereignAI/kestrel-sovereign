"""Real file-backed SQLite lock ordering, with event barriers and bounded cleanup."""

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files
from tests.unit.test_constitution_audit import _DurableConstitutionHarness


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor_writer", ["automatic", "signed_same", "signed_new"])
@pytest.mark.parametrize("transition", ["entry", "exit", "audit_begin", "audit_record", "audit_run", "periodic"])
async def test_outer_transaction_refuses_before_constitution_lock_wait(
    tmp_path, monkeypatch, anchor_writer, transition
):
    storage = AsyncStorage(str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id="did:test:lock-order:" + uuid4().hex)
    await storage.initialize()
    anchor_task = None
    allow_anchor = asyncio.Event()
    try:
        agent = _DurableConstitutionHarness(storage, datetime.now(timezone.utc))
        agent.agent_id = storage.agent_id
        await agent._initialize_constitution_runtime_state(is_new_identity=True)
        agent.extension = None
        agent._anchor_constitution_governance = KestrelAgent._anchor_constitution_governance.__get__(agent)
        agent._agent_signing_dids = ConstitutionMixin._agent_signing_dids.__get__(agent)
        agent._trusted_sovereign_did_document = ConstitutionMixin._trusted_sovereign_did_document.__get__(agent)
        content = resolve_governing_constitution_bytes(None)
        props = {}
        if anchor_writer == "signed_same":
            digest = await storage.store_file(content, "KESTREL_CONSTITUTION.md")
            props["constitution_hash"] = digest
        await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="lock-order", properties=props))
        if anchor_writer == "signed_same":
            await agent._anchor_constitution_governance(digest)
        artifact, root = _write_authority_files(tmp_path, content)
        agent._sovereign_trust_root_path = root
        before = await agent._constitution_state_store.load(agent.agent_id)
        if transition == "periodic":
            # Exercise the mutable periodic branch, not its already-pending
            # startup-audit no-op. Durable bootstrap custody is unchanged.
            agent._constitution_audit_pending = False
        prepared = asyncio.Event()
        anchor_waiting_for_database = asyncio.Event()
        original_guard = ConstitutionMixin._constitution_state_guard
        original_transaction = storage.transaction

        @asynccontextmanager
        async def coordinated_guard(owner):
            # Delay only the native anchor's first lock entry, after its real
            # reads/source/signature checks. No lock or SQL behavior is faked.
            if asyncio.current_task() is anchor_task and not prepared.is_set():
                prepared.set()
                await allow_anchor.wait()
            async with original_guard(owner):
                yield

        @asynccontextmanager
        async def observed_transaction(*args, **kwargs):
            if asyncio.current_task() is anchor_task:
                assert agent._constitution_state_lock.locked()
                anchor_waiting_for_database.set()
            async with original_transaction(*args, **kwargs):
                yield

        monkeypatch.setattr(ConstitutionMixin, "_constitution_state_guard", coordinated_guard)
        monkeypatch.setattr(storage, "transaction", observed_transaction)

        async def anchor():
            if anchor_writer == "automatic":
                return await ConstitutionMixin._get_governing_constitution(agent)
            return await ConstitutionMixin.reanchor_constitution(agent, amendment_artifact_path=str(artifact))

        anchor_task = asyncio.create_task(anchor())
        await asyncio.wait_for(prepared.wait(), 5)
        async with storage.transaction():
            assert storage.owns_open_transaction is True
            allow_anchor.set()
            await asyncio.wait_for(anchor_waiting_for_database.wait(), 5)
            try:
                # asyncio.timeout keeps this SAME transaction-owning task.
                # wait_for(coroutine) would create a new task and erase the
                # ownership relation the regression must exercise.
                async with asyncio.timeout(1):
                    if transition == "entry":
                        result = await agent.enter_safe_mode("caller restriction")
                    elif transition == "exit":
                        result = await agent.exit_safe_mode(authorization="test sovereign")
                    elif transition == "audit_begin":
                        result = await agent._begin_explicit_constitution_audit()
                    elif transition == "audit_record":
                        result = await agent._record_successful_constitution_audit(source="caller")
                    elif transition == "audit_run":
                        result = await agent._run_explicit_constitution_audit()
                    else:
                        result = await agent._maybe_audit()
            except TimeoutError:
                pytest.fail("ambient transaction waited on constitution lock: lock inversion deadlock")
            assert agent._safe_mode is True
            assert agent._constitution_state_persistence_pending is True
            assert (agent._constitution_state_revision, agent._constitution_state_generation) == (before.revision, before.generation)
            if transition in {"entry", "audit_begin", "audit_record"}:
                assert result is False
            elif transition == "exit":
                assert result.startswith("Safe Mode remains active:")
            elif transition == "audit_run":
                assert result[0] is None and result[2] is False
            else:
                assert result is None
        outcome = await asyncio.wait_for(anchor_task, 5)
        assert not outcome.startswith("Error:"), outcome
        durable = await agent._constitution_state_store.load(agent.agent_id)
        # No refused audit/entry/exit was silently included in the anchor's
        # later owned commit. Its original restricted state remains exact.
        assert durable.safe_mode == before.safe_mode
        assert durable.last_successful_audit_at == before.last_successful_audit_at
        assert durable.interaction_count == before.interaction_count
    finally:
        allow_anchor.set()
        if anchor_task is not None and not anchor_task.done():
            anchor_task.cancel()
        if anchor_task is not None:
            with suppress(asyncio.CancelledError):
                await anchor_task
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_at", ["verification", "database"])
async def test_refused_lifecycle_entry_invalidates_inflight_native_exit(
    tmp_path, monkeypatch, blocked_at
):
    """A lock-free failure latch must not be erased by the current lock owner."""
    from kestrel_sovereign.agent.constitution import SafeModeCause

    storage = AsyncStorage(str(tmp_path / "kestrel_prime.db"), backend="sqlite", agent_id="did:test:refused-exit:" + uuid4().hex)
    await storage.initialize()
    exit_task = None
    proceed = asyncio.Event()
    try:
        agent = _DurableConstitutionHarness(storage, datetime.now(timezone.utc))
        agent.agent_id = storage.agent_id
        await agent._initialize_constitution_runtime_state(is_new_identity=True)
        agent.extension = None
        agent._anchor_constitution_governance = KestrelAgent._anchor_constitution_governance.__get__(agent)
        content = resolve_governing_constitution_bytes(None)
        digest = await storage.store_file(content, "KESTREL_CONSTITUTION.md")
        await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="exit", properties={"constitution_hash": digest}))
        await agent._anchor_constitution_governance(digest)
        await agent.enter_safe_mode("old integrity restriction")
        before = await agent._constitution_state_store.load(agent.agent_id)
        agent.verify_constitution_overlay = ConstitutionMixin.verify_constitution_overlay.__get__(agent)
        agent._verify_spawn_mandate_constraints = ConstitutionMixin._verify_spawn_mandate_constraints.__get__(agent)
        native_verify = ConstitutionMixin._verify_constitution_integrity.__get__(agent)
        verified = asyncio.Event()
        blocked = asyncio.Event()
        database_waiting = asyncio.Event()
        original_transaction = storage._backend.transaction

        async def observed_verify():
            result = await native_verify()
            assert result[0] is True, result
            verified.set()
            if blocked_at == "verification":
                blocked.set()
                await proceed.wait()
            return result

        @asynccontextmanager
        async def observed_transaction(*args, **kwargs):
            if asyncio.current_task() is exit_task and verified.is_set() and not blocked.is_set():
                blocked.set()
                await proceed.wait()
                database_waiting.set()
            async with original_transaction(*args, **kwargs):
                yield

        agent._verify_constitution_integrity = observed_verify
        if blocked_at == "database":
            monkeypatch.setattr(storage._backend, "transaction", observed_transaction)
        exit_task = asyncio.create_task(agent.exit_safe_mode(authorization="test sovereign"))
        await asyncio.wait_for(blocked.wait(), 5)
        async with storage.transaction():
            if blocked_at == "database":
                proceed.set()
                await asyncio.wait_for(database_waiting.wait(), 5)
            async with asyncio.timeout(1):
                assert await agent.enter_safe_mode(
                    "feature quarantine could not persist",
                    cause=SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value,
                ) is False
            assert agent._safe_mode_cause == SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value
            assert agent._feature_lifecycle_integrity_uncertain is True
            assert agent._feature_lifecycle_repair_verified is False
        proceed.set()
        result = await asyncio.wait_for(exit_task, 5)
        assert result.startswith("Safe Mode remains active:"), result
        assert agent._safe_mode is True
        assert agent._constitution_state_persistence_pending is True
        assert await agent._constitution_state_store.load(agent.agent_id) == before
    finally:
        proceed.set()
        if exit_task is not None and not exit_task.done():
            exit_task.cancel()
        if exit_task is not None:
            with suppress(asyncio.CancelledError):
                await exit_task
        await storage.close()
