"""Native queued admission must wake a reader even before its source enters."""
import asyncio

import pytest
import httpx

from kestrel_sovereign._async_ownership import OwnedAsyncIterator, await_owned_task
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock
from kestrel_sovereign.execution_custody import NativeExecutionStreamOwner, execution_cleanup_debts, _explicit_control_chain


@pytest.mark.asyncio
async def test_native_traffic_release_cancel_cannot_replace_original_unknown_receipt():
    from kestrel_sovereign.features.isolated_runtime import _TrafficGate
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError
    gate = _TrafficGate()
    original = ExecutionCommitOutcomeError("unknown")
    held, release = asyncio.Event(), asyncio.Event()
    blockers = []
    async def hold_release():
        async with gate._condition:
            held.set()
            await release.wait()
    async def native_rpc():
        async with gate.admit():
            blockers.append(asyncio.create_task(hold_release()))
            await held.wait()
            raise original
    task = asyncio.create_task(native_rpc())
    try:
        await asyncio.wait_for(held.wait(), 2)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and gate._active == 1
        release.set()
        result = await await_owned_task(task)
        assert result.error is original and gate._active == 0
    finally:
        release.set()
        await await_owned_task(task)
        for blocker in blockers:
            await await_owned_task(blocker)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_pending", [False, True])
async def test_native_scheduler_retains_cancelled_sqlite_publication_receipt(tmp_path, cancel_pending):
    from kestrel_sovereign._async_ownership import create_owned_task
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError, execution_commit_outcome
    from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db import SQLiteBackend
    backend = SQLiteBackend(str(tmp_path / "publisher.db"))
    await backend.connect()
    db = AsyncDatabase(backend)
    notices = []
    async def no_work(*args):
        raise AssertionError("telemetry lifecycle must not execute scheduled work")
    runner = SchedulerRunner(db, "did:test:publication61", no_work, on_protocol_failure=notices.append)
    committed = asyncio.Event()
    original = ExecutionCommitOutcomeError("unknown")
    async def publish():
        await db.execute("CREATE TABLE native_publication(id INTEGER)")
        await db.execute_commit("INSERT INTO native_publication VALUES (1)")
        committed.set()
        if cancel_pending:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise cancellation from original
        else:
            raise asyncio.CancelledError("native commit acknowledgement") from original
    task = create_owned_task(publish())
    runner._runtime_status_publish_task = task
    try:
        await asyncio.wait_for(committed.wait(), 2)
        if not cancel_pending:
            await await_owned_task(task)
        await runner._join_runtime_status_publish_task(deadline=asyncio.get_running_loop().time())
        assert runner._runtime_status_publish_task is None
        assert execution_commit_outcome(runner._readiness_failure) == "unknown"
        assert execution_commit_outcome(runner._execution_control_error) == "unknown"
        runner._harvest_runtime_status_publication(task)
        assert len(notices) == 1
        assert await db.fetchval("SELECT COUNT(*) FROM native_publication") == 1
    finally:
        if not task.done():
            task.cancel()
        await await_owned_task(task)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [None, "unknown", "committed"])
async def test_actual_asgi_peer_envelope_preserves_control_without_private_prose(outcome):
    from fastapi import FastAPI
    from kestrel_sovereign.api_errors import register_api_error_handlers
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError, ExecutionCommitOutcomeError,
        execution_commit_outcome, is_execution_control_error,
    )
    from kestrel_sovereign.features.peers.directory import LocalHostPeerDirectory
    from kestrel_sovereign.features.isolated_runtime import _redacted_host_ingress_error
    app = FastAPI()
    register_api_error_handlers(app)
    original = ExecutionAuthorityError("private provider diagnostic not for transport") if outcome is None else ExecutionCommitOutcomeError(outcome)
    carrier = RuntimeError("private native host error")
    carrier.__cause__ = original

    @app.get("/native-peer")
    async def native_peer():
        raise carrier

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://offline") as client:
        response = await client.get("/native-peer")
    assert response.status_code == 500
    assert "private" not in response.text and "diagnostic" not in response.text
    with pytest.raises(ExecutionAuthorityError) as caught:
        LocalHostPeerDirectory._raise_for_route_status(response, action="native peer dispatch")
    assert is_execution_control_error(caught.value)
    assert execution_commit_outcome(caught.value) == outcome
    redacted = _redacted_host_ingress_error(carrier)
    assert redacted.status == "generic-failure" and redacted.commit_outcome is None
    # A foreign exception graph is no native provenance. The independently
    # decoded canonical ASGI control is an explicit trusted input instead.
    redacted = _redacted_host_ingress_error(caught.value, trusted=True)
    assert redacted.status == "execution-control" and redacted.commit_outcome == outcome


@pytest.mark.parametrize("outcome", [{}, [], 1, "not-a-receipt"])
def test_malformed_peer_control_details_still_cannot_become_retryable_transport(outcome):
    from kestrel_sovereign.features.peers.directory import LocalHostPeerDirectory
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError, execution_commit_outcome
    response = httpx.Response(500, json={"error": {"code": "execution_control", "details": [{"commit_outcome": outcome}]}}, request=httpx.Request("POST", "http://offline/peer"))
    with pytest.raises(ExecutionAuthorityError) as caught:
        LocalHostPeerDirectory._raise_for_route_status(response, action="native peer dispatch")
    assert execution_commit_outcome(caught.value) is None


@pytest.mark.asyncio
async def test_copied_context_sibling_cannot_inherit_native_privacy_lifecycle():
    from kestrel_sovereign.execution_custody import ExecutionStreamLifecycle, execution_stream
    lock = ReentrantTransitionLock()
    native_factory = lock.capture_native_work_factory()
    registered, children, effects = [], [], []
    attempted = asyncio.Event()
    lifecycle = ExecutionStreamLifecycle(
        owner_factory=lambda run, context: native_factory(run, context=context),
        owner_started=lambda iterator, task: registered.append((iterator, task)),
        require_work=lambda: None,
    )
    owner = object()

    async def sibling_body():
        attempted.set()
        async with lock:
            effects.append("independent native lock admission")
            yield b"child"

    async def sibling():
        stream = execution_stream(owner, sibling_body())
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    async def parent_body():
        assert lock.locked()
        # asyncio creates a COPY of the live carrier, not the producer task.
        children.append(asyncio.create_task(sibling()))
        await attempted.wait()
        yield b"parent"

    parent = execution_stream(owner, parent_body(), lifecycle=lifecycle)
    try:
        assert await anext(parent) == b"parent"
        done, _ = await asyncio.wait(children, timeout=.1)
        assert not done and not effects, "copied-context sibling borrowed native producer reentry"
        assert len(registered) == 1, "sibling registered under the parent's feature lifecycle"
        with pytest.raises(StopAsyncIteration):
            await anext(parent)
        result = await asyncio.wait_for(await_owned_task(children[0]), 2)
        assert result.error is None and result.result == b"child"
        assert effects == ["independent native lock admission"]
    finally:
        await parent.aclose()
        for child in children:
            if not child.done():
                child.cancel()
            await await_owned_task(child)


@pytest.mark.asyncio
async def test_original_owner_cancelled_before_privacy_admission_wakes_reader():
    lock = ReentrantTransitionLock()
    entered = False

    async def body():
        nonlocal entered
        entered = True
        yield b"must-not-run"

    # Capture outside the mutex: this unrelated producer must queue normally.
    native_factory = lock.capture_native_work_factory()
    locked, release = asyncio.Event(), asyncio.Event()
    async def unrelated_holder():
        async with lock:
            locked.set()
            await release.wait()
    holder = asyncio.create_task(unrelated_holder())
    await locked.wait()
    try:
        source = OwnedAsyncIterator(
            body, operation="native queued privacy admission",
            owner_factory=lambda run, context: native_factory(run, context=context),
        )
        reader = asyncio.create_task(anext(source))
        try:
            await asyncio.sleep(0)
            await source.aclose()
            done, _ = await asyncio.wait({reader}, timeout=.3)
            assert done, "reader stranded after original queued owner physically retired"
            assert not entered
            assert isinstance((await await_owned_task(reader)).error, (asyncio.CancelledError, StopAsyncIteration))
        finally:
            if not reader.done():
                reader.cancel()
            await await_owned_task(reader)
            await source.aclose()
    finally:
        release.set()
        await await_owned_task(holder)


@pytest.mark.asyncio
async def test_failed_original_httpx_close_is_retained_and_never_retried():
    attempts = 0
    original = OSError("native body retirement failed")
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"owned"
        async def aclose(self):
            nonlocal attempts
            attempts += 1
            raise original
    response = httpx.Response(200, stream=Body())
    owner = NativeExecutionStreamOwner()
    owner.retain(response)
    terminal = None
    for _ in range(2):
        with pytest.raises(BaseException) as caught:
            await owner.close()
        assert original in tuple(_explicit_control_chain(caught.value))
        if terminal is None:
            terminal = caught.value
        assert caught.value is terminal, "repeated close grew a new terminal history"
        assert owner.handles == [response]
        assert caught.value._kestrel_native_retirement_owner is owner
        assert execution_cleanup_debts(caught.value)
        assert all(resource is response and task.done() for resource, task in execution_cleanup_debts(caught.value))
        assert tuple(_explicit_control_chain(original)) == (original,)
        assert not execution_cleanup_debts(original), "private close debt contaminated original source"
    assert response.is_closed and attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_reanchor_embedding_joins_original_native_close_and_preserves_commit(monkeypatch, cancel):
    from kestrel_sovereign import cli_embeddings
    from kestrel_sovereign.setup.constitution_reanchor import _agent_embedding
    from kestrel_sovereign.execution_custody import execution_commit_outcome
    from tests.unit.test_execution_custody_review34 import native_close_owner
    from kestrel_sovereign.llm.lifecycle import UnpublishedLLMOwner
    service, client, entered, release = await native_close_owner(monkeypatch)
    resolution = cli_embeddings.AgentEmbeddingResolution(
        service, None, None, _private_owner=UnpublishedLLMOwner(service, ()),
    )
    async def resolve(_db, _did):
        return resolution
    monkeypatch.setattr(cli_embeddings, "resolve_agent_embedding", resolve)
    committed = []
    async def repair():
        async with _agent_embedding(None, "did:test:review61", needed=True):
            committed.append(True)
    task = asyncio.create_task(repair())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if cancel:
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=.15)
        assert not done and not client.is_closed, "repair returned before its original native closer"
        original_closer = service._llm_close_tasks[("adapter", id(client))]
        release.set()
        result = await await_owned_task(task)
        assert committed == [True] and client.is_closed and original_closer.done()
        if cancel:
            assert execution_commit_outcome(result.error) == "committed"
            assert any(isinstance(error, asyncio.CancelledError) for error in _explicit_control_chain(result.error))
        else:
            assert result.error is None
        # The initial bounded close was incomplete, but the same canonical
        # owner has now physically joined it. Do not fabricate failed native
        # retirement debt after successful settlement or retain observer C on
        # the shared service's original close receipt.
        assert not execution_cleanup_debts(result.error)
        assert resolution._private_owner.retired
        assert resolution._private_owner.control is None
        await resolution.aclose()
        assert service._llm_close_tasks[("adapter", id(client))] is None
    finally:
        release.set()
        await await_owned_task(task)
        await service.wait_for_close_completion()


@pytest.mark.asyncio
@pytest.mark.parametrize("restriction", [None, "safe-mode", "pending-genesis", "missing-runtime"])
async def test_native_realtime_cognition_gate_reads_durable_governance(tmp_path, restriction):
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from tests.integration.test_constitution_turn_admission import _ready_turn
    async with AsyncStorage(str(tmp_path / "cognition.db"), agent_id="did:test:cognition61") as storage:
        agent, _ = await _ready_turn(storage)
        if restriction == "safe-mode":
            await agent.enter_safe_mode("native persisted restriction")
        elif restriction == "pending-genesis":
            node = await storage.get_node(storage.agent_id)
            node.properties["genesis_audit"]["status"] = "pending"
            await storage.add_node(node)
        elif restriction == "missing-runtime":
            await storage.db.execute("DELETE FROM constitution_runtime_state WHERE agent_id = ?", (storage.agent_id,))
        if restriction:
            with pytest.raises(ExecutionAuthorityError):
                await ConstitutionMixin.require_cognition_admission(agent)
        else:
            await ConstitutionMixin.require_cognition_admission(agent)
