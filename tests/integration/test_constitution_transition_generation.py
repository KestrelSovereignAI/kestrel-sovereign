"""A newer native-transaction refusal invalidates the whole older transition."""

import asyncio
from contextlib import suppress
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.boot import BootPhaseState
from kestrel_sovereign.agent.constitution import ConstitutionMixin, SafeModeCause
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.constitution.runtime_state import ConstitutionRuntimeStateStore
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_refusal_races import _agent


async def _ready(storage):
    agent = await _agent(storage)
    digest = await storage.store_file(resolve_governing_constitution_bytes(None), "KESTREL_CONSTITUTION.md")
    await storage.add_node(GraphNode(node_id=agent.agent_id, node_type="agent", label="transition", properties={"constitution_hash": digest}))
    await agent._anchor_constitution_governance(digest)
    assert await agent._record_successful_constitution_audit(source="ready fixture")
    agent._boot_state = BootPhaseState.READY
    return agent, digest


async def _refuse(agent, storage):
    async with storage.transaction():
        assert await agent.enter_safe_mode(
            "later lifecycle quarantine", cause=SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value,
        ) is False


def _assert_latched(agent):
    assert agent._safe_mode is True
    assert agent._safe_mode_reason == "later lifecycle quarantine"
    assert agent._safe_mode_cause == SafeModeCause.FEATURE_LIFECYCLE_UNCERTAIN.value
    assert agent._constitution_state_persistence_pending is True
    assert agent._feature_lifecycle_integrity_uncertain is True
    assert agent._feature_lifecycle_repair_verified is False


async def _join(task, proceed):
    proceed.set()
    if task is not None and not task.done():
        task.cancel()
    if task is not None:
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("boundary", ["initialize", "load"])
@pytest.mark.parametrize("buffered", [False, True])
async def test_restore_cannot_erase_refusal_during_native_read(db_backend, monkeypatch, boundary, buffered):
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:restore-generation:" + uuid4().hex)
    await storage.initialize()
    task = None
    proceed, waiting = asyncio.Event(), asyncio.Event()
    try:
        agent, _ = await _ready(storage)
        before = await agent._constitution_state_store.load(agent.agent_id)
        events = await agent._constitution_state_store.list_events(agent.agent_id)
        if buffered:
            async with storage.transaction():
                assert await agent.enter_safe_mode("earlier buffered integrity restriction") is False
        native = getattr(ConstitutionRuntimeStateStore, boundary)

        async def paused(store, *args, **kwargs):
            result = await native(store, *args, **kwargs)
            if asyncio.current_task() is task:
                waiting.set()
                await proceed.wait()
            return result

        monkeypatch.setattr(ConstitutionRuntimeStateStore, boundary, paused)
        task = asyncio.create_task(agent._initialize_constitution_runtime_state())
        await asyncio.wait_for(waiting.wait(), 5)
        await _refuse(agent, storage)
        proceed.set()
        await asyncio.wait_for(task, 5)
        _assert_latched(agent)
        assert await agent._constitution_state_store.load(agent.agent_id) == before
        assert await agent._constitution_state_store.list_events(agent.agent_id) == events
    finally:
        await _join(task, proceed)
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("transition", ["entry", "restore", "explicit", "startup", "periodic", "successful", "audit_started"])
async def test_waiting_transition_cannot_adopt_later_refusal(db_backend, monkeypatch, transition):
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:queued-generation:" + uuid4().hex)
    await storage.initialize()
    audit_task = queued_task = None
    proceed, auditing, queued = asyncio.Event(), asyncio.Event(), asyncio.Event()
    try:
        agent, _ = await _ready(storage)
        native_verify = agent._verify_constitution_integrity
        native_lock = agent._constitution_state_lock

        class NativeLockObserver:
            async def __aenter__(self):
                if asyncio.current_task() is queued_task:
                    assert native_lock.locked()
                    queued.set()
                await native_lock.acquire()
                return self

            async def __aexit__(self, *args):
                native_lock.release()

        async def paused_verification():
            result = await native_verify()
            if asyncio.current_task() is audit_task:
                auditing.set()
                await proceed.wait()
            return result

        monkeypatch.setattr(agent, "_constitution_state_lock", NativeLockObserver())
        monkeypatch.setattr(agent, "_verify_constitution_integrity", paused_verification)
        audit_task = asyncio.create_task(agent._run_explicit_constitution_audit())
        await asyncio.wait_for(auditing.wait(), 5)
        invocation = {
            "entry": lambda: agent.enter_safe_mode("older waiting integrity entry"),
            "restore": agent._initialize_constitution_runtime_state,
            "explicit": agent._run_explicit_constitution_audit,
            "startup": agent._audit_constitution_on_startup,
            "periodic": agent._maybe_audit,
            "successful": lambda: agent._record_successful_constitution_audit(source="older waiting success"),
            "audit_started": agent._begin_explicit_constitution_audit,
        }[transition]
        queued_task = asyncio.create_task(invocation())
        await asyncio.wait_for(queued.wait(), 5)
        before = await agent._constitution_state_store.load(agent.agent_id)
        events = await agent._constitution_state_store.list_events(agent.agent_id)
        await _refuse(agent, storage)
        proceed.set()
        await asyncio.wait_for(audit_task, 5)
        result = await asyncio.wait_for(queued_task, 5)
        _assert_latched(agent)
        if transition == "entry":
            assert result is False
        assert await agent._constitution_state_store.load(agent.agent_id) == before
        assert await agent._constitution_state_store.list_events(agent.agent_id) == events
    finally:
        await _join(queued_task, proceed)
        await _join(audit_task, proceed)
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("audit", ["explicit", "startup", "periodic"])
@pytest.mark.parametrize("valid", [True, False])
async def test_audit_result_cannot_adopt_a_newer_refusal(db_backend, monkeypatch, audit, valid):
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:audit-generation:" + uuid4().hex)
    await storage.initialize()
    task = None
    proceed, waiting = asyncio.Event(), asyncio.Event()
    try:
        agent, digest = await _ready(storage)
        if not valid:
            await db_backend.execute("DELETE FROM files WHERE content_hash = ?", (digest,))
        agent._interaction_count = agent.AUDIT_INTERVAL
        native = agent._verify_constitution_integrity

        async def paused():
            result = await native()
            assert result[0] is valid, result
            waiting.set()
            await proceed.wait()
            return result

        monkeypatch.setattr(agent, "_verify_constitution_integrity", paused)
        invocation = {
            "explicit": agent._run_explicit_constitution_audit,
            "startup": agent._audit_constitution_on_startup,
            "periodic": agent._maybe_audit,
        }[audit]
        task = asyncio.create_task(invocation())
        await asyncio.wait_for(waiting.wait(), 5)
        before = await agent._constitution_state_store.load(agent.agent_id)
        events = await agent._constitution_state_store.list_events(agent.agent_id)
        await _refuse(agent, storage)
        proceed.set()
        result = await asyncio.wait_for(task, 5)
        _assert_latched(agent)
        assert await agent._constitution_state_store.load(agent.agent_id) == before
        assert await agent._constitution_state_store.list_events(agent.agent_id) == events
        if audit == "explicit":
            assert result[0] is None and result[2] is False, result
    finally:
        await _join(task, proceed)
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_exit_notification_cannot_report_later_restriction_as_recovery(db_backend):
    storage = AsyncStorage(backend=db_backend, agent_id="did:test:notification-generation:" + uuid4().hex)
    await storage.initialize()
    task = None
    proceed, waiting = asyncio.Event(), asyncio.Event()
    try:
        agent, _ = await _ready(storage)
        assert await agent.enter_safe_mode("original restriction")

        class NotificationBarrier:
            async def add_conversation(self, *args, **kwargs):
                waiting.set()
                await proceed.wait()

        # Provider-bearing embedding/notification seam only; real verifier,
        # runtime state, events, native transaction and commit are untouched.
        agent.privacy_agent = NotificationBarrier()
        task = asyncio.create_task(agent.exit_safe_mode(authorization="test sovereign"))
        await asyncio.wait_for(waiting.wait(), 5)
        committed = await agent._constitution_state_store.load(agent.agent_id)
        events = await agent._constitution_state_store.list_events(agent.agent_id)
        assert committed.safe_mode is False
        assert events[-1]["event_type"] == "safe_mode_exited"
        await _refuse(agent, storage)
        proceed.set()
        result = await asyncio.wait_for(task, 5)
        _assert_latched(agent)
        assert "remains active" in result, result
        assert await agent._constitution_state_store.load(agent.agent_id) == committed
        assert await agent._constitution_state_store.list_events(agent.agent_id) == events
    finally:
        await _join(task, proceed)
        await storage.close()
