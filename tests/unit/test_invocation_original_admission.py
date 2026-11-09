"""Original invocation admission retires accepted children before registration.

Actual Agent constructor, lifecycle bookkeeping and SQLite worker are used;
the decorated SQL component is not process_input, Agent READY or real cognition.
"""
import asyncio
import inspect
import json
import sys

import pytest

from kestrel_sovereign._async_ownership import await_owned_operation, await_owned_task, create_owned_task
from kestrel_sovereign.agent.invocation import bind_async_invocation
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
from tests.unit.test_cleanup_task_admission import _partial_native_agent
from tests.unit.test_host_context_native_cleanup import _actual_operation


def _invocation_request(agent, payload):
    from fastapi import FastAPI
    from starlette.requests import Request
    app = FastAPI()
    app.state.agent = agent
    body = json.dumps(payload).encode()
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}
    request = Request({"type": "http", "http_version": "1.1", "method": "POST",
                           "scheme": "http", "path": "/api/agent/invoke", "raw_path": b"/api/agent/invoke",
                           "query_string": b"", "headers": [(b"content-type", b"application/json")],
                           "client": ("test", 1), "server": ("test", 80), "app": app}, receive)
    return request


def _kite_invocation_request(agent, request_id):
    return _invocation_request(agent, {"request_id": request_id,
        "kite_evidence": {"operation": "diagnostics", "nonce": "a" * 64}})


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 12), reason="native eager Task requires Python 3.12")
@pytest.mark.parametrize("physical_kind", [None, "ordinary", "sdk"])
@pytest.mark.parametrize("publication_kind", [None, "ordinary", "sdk", "audit"])
@pytest.mark.parametrize("stop", [False, True])
async def test_actual_privacy_invocation_observes_every_native_completion(
    tmp_path, monkeypatch, physical_kind, publication_kind, stop,
):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.agent.invocation import InvocationCancelledError
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, _explicit_control_chain
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    await db.execute("CREATE TABLE native_completion_payload_effects (id INTEGER)")
    lock = agent._get_privacy_transition_lock()
    request_id = "original-privacy-native-completion"
    physical = RuntimeError("original physical invocation failure") if physical_kind == "ordinary" else None
    if physical_kind == "sdk":
        physical = asyncio.CancelledError("original physical invocation cancellation")
        physical.__cause__ = ResourceOutcomeUnknown("original physical invocation uncertain")
    publication = RuntimeError("original privacy ownership publication failure") if publication_kind == "ordinary" else None
    if publication_kind in {"sdk", "audit"}:
        publication = asyncio.CancelledError("original privacy ownership publication cancellation")
        publication.__cause__ = (ResourceOutcomeUnknown("original publication outcome uncertain")
            if publication_kind == "sdk" else ToolAuditUnconfirmedError("original publication audit uncertain"))
    effect = tmp_path / "original-privacy-invocation-physical-effect"
    owners, generations, publications = [], [], []
    @bind_async_invocation("request_id", track_request_lifecycle=True)
    async def invoke(self, request_id=None):
        await db.execute("INSERT INTO native_completion_payload_effects VALUES (1)")
        return "payload must stay unentered"
    class OriginalTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "completed":
                publications.append(self)
                if publication is not None:
                    raise publication
            return super().add_done_callback(callback, context=context)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **options):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is None or not code.co_qualname.endswith(".start_owned_native_work.<locals>.run"):
            return previous(loop, coroutine, **options) if previous else asyncio.Task(coroutine, loop=loop, **options)
        async def outer():
            generations.append(agent._active_request_generations[request_id])
            coroutine.close()
            effect.write_text("original native physical effect completed once")
            if stop:
                assert agent.cancel_current_request(request_id)
            if physical is not None:
                raise physical
            return "original physical success"
        options["eager_start"] = True
        task = OriginalTask(outer(), loop=loop, **options)
        owners.append(task)
        return task
    async def drive():
        async with lock:
            return await invoke(agent, request_id=request_id)
    try:
        loop.set_task_factory(factory)
        outcome = await await_owned_task(create_owned_task(drive(), retain_originals=True))
        loop.set_task_factory(previous)
        chain = tuple(_explicit_control_chain(outcome.error))
        if publication is not None:
            assert publication in chain, "Native invocation completion bypassed original publication receipt"
        if physical is not None:
            assert physical in chain
        if publication is physical is None:
            if stop:
                assert isinstance(outcome.error, InvocationCancelledError)
            else:
                assert outcome.error is None and outcome.result == "original physical success"
        terminal = physical_kind == "sdk" or publication_kind in {"sdk", "audit"}
        assert await agent.wait_for_request_completion(request_id, generation=generations[0]) is (
            RequestCompletionDisposition.ABANDONED if terminal else RequestCompletionDisposition.COMPLETED), (
                "Native invocation receipt uncertainty retired as successful completion")
        assert owners == publications and len(owners) == 1 and owners[0].done()
        assert effect.read_text() == "original native physical effect completed once"
        assert await db.fetchval("SELECT COUNT(*) FROM native_completion_payload_effects") == 0
        assert not lock.locked() and not lock._native_work
    finally:
        loop.set_task_factory(previous)
        for task in owners:
            await await_owned_task(task)
        await agent.llm_service.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["allocation", "binding", "callback-before", "callback-after",
                                  "canonical-before", "canonical-after"])
@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("retirement_fails", [False, True])
async def test_kite_endpoint_retires_original_observation_before_registration(
    tmp_path, monkeypatch, mode, prior, retirement_fails,
):
    from fastapi import Response
    from kestrel_sovereign.endpoints.agent import invoke_agent
    # This exercises the actual endpoint and Agent lifecycle with an explicit
    # SQL observation component, not full Kite diagnostics or Agent READY.
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    await db.execute("CREATE TABLE kite_admission_effects (id INTEGER)")
    request_id = "original-kite-admission"
    initial = agent.register_active_request(request_id, nested=False) if prior else None
    release = asyncio.Event()
    accepted, refused, cleanup_observations = [], [], []
    refusal = RuntimeError("actual Kite observation " + mode + " refused")
    retirement = RuntimeError("actual Kite registration retirement refused")
    actual_bind, actual_cleanup = type(agent).bind_request_operation, type(agent)._cleanup_cancelled_request
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    async def native_evidence(*args, **kwargs):
        await db.execute("INSERT INTO kite_admission_effects VALUES (1)")
        await release.wait()
        return "diagnostics", {"completed": True}
    def binding(self, rid, operation):
        if self is agent and rid == request_id and mode == "binding":
            raise refusal
        return actual_bind(self, rid, operation)
    def cleanup(self, rid, *args, **kwargs):
        if self is agent and rid == request_id:
            cleanup_observations.append((tuple(task.done() for task in accepted), kwargs.get("disposition")))
            if retirement_fails:
                raise retirement
        return actual_cleanup(self, rid, *args, **kwargs)
    class RefusedCallback(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            target = (getattr(callback, "__name__", "") == "close_unstarted" if mode.startswith("canonical")
                      else getattr(callback, "__qualname__", "").endswith("capture_request_operation_binding.<locals>.publish.<locals>.release"))
            if target and mode != "binding":
                if mode.endswith("-after"):
                    super().add_done_callback(callback, context=context)
                raise refusal
            return super().add_done_callback(callback, context=context)
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is native_evidence.__code__:
            if mode == "allocation":
                refused.append((coroutine, operation))
                raise refusal
            task = RefusedCallback(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    request = _kite_invocation_request(agent, request_id)
    owner = None
    try:
        monkeypatch.setattr("kestrel_sovereign.endpoints.agent._kite_runtime_observation", native_evidence)
        monkeypatch.setattr(type(agent), "bind_request_operation", binding)
        monkeypatch.setattr(type(agent), "_cleanup_cancelled_request", cleanup)
        loop.set_task_factory(factory)
        endpoint = getattr(invoke_agent, "__wrapped__", invoke_agent)
        owner = create_owned_task(endpoint(request, Response()), retain_originals=True)
        result = await await_owned_task(owner)
        loop.set_task_factory(previous)
        assert result.error is not None
        assert all(task.done() for task in accepted), "Kite endpoint returned before accepted observation physically joined"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair), "Kite allocation refusal leaked original unstarted observation"
        assert cleanup_observations and all(all(done) and disposition is RequestCompletionDisposition.ABANDONED
                                            for done, disposition in cleanup_observations), "Kite registration retired before observation or was falsely COMPLETED"
        assert await db.fetchval("SELECT count(*) FROM kite_admission_effects") == 0
        assert not vars(agent).get("_request_operation_tasks", {})
        if retirement_fails:
            from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
            chain = tuple(_explicit_control_chain(result.error))
            # Ordinary HTTP normalization may already wrap the admission
            # failure; its explicit context must still retain that original.
            assert refusal in chain and retirement in chain
            generation = agent._active_request_generations[request_id]
            assert any(resource == (agent, request_id, generation) and task is None
                       for resource, task in execution_cleanup_debts(result.error))
            assert not execution_cleanup_debts(refusal) and not execution_cleanup_debts(retirement)
            assert agent._active_request_generation_counts[(request_id, generation)] == 1
            if prior:
                assert generation != initial
                assert agent._active_request_generation_counts[(request_id, initial)] == 1
        else:
            assert agent._active_request_generation_counts == ({(request_id, initial): 1} if prior else {})
    finally:
        loop.set_task_factory(previous)
        release.set()
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        if owner is not None:
            await await_owned_task(owner)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        monkeypatch.setattr(type(agent), "_cleanup_cancelled_request", actual_cleanup)
        for (rid, generation), count in tuple(agent._active_request_generation_counts.items()):
            for _ in range(count):
                actual_cleanup(agent, rid, generation=generation,
                               disposition=RequestCompletionDisposition.ABANDONED)
        await agent.llm_service.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["allocation", "binding", "callback-before", "callback-after"])
@pytest.mark.parametrize("prior", [False, True])
async def test_invocation_admission_retires_exact_original_child_and_registration(
    tmp_path, monkeypatch, mode, prior,
):
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    assert not agent.llm_service.providers
    await db.execute("CREATE TABLE invocation_admission_effects (id INTEGER)")
    request_id = "original-refused-invocation"
    initial = agent.register_active_request(request_id, nested=False) if prior else None
    refusal = RuntimeError("actual original invocation " + mode + " refused")
    refused, accepted, cleanup_observations = [], [], []
    release = asyncio.Event()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    actual_bind, actual_cleanup = type(agent).bind_request_operation, type(agent)._cleanup_cancelled_request
    owner = None

    @bind_async_invocation("request_id", track_request_lifecycle=True)
    async def invoke(self, request_id=None):
        await db.execute("INSERT INTO invocation_admission_effects VALUES (1)")
        await release.wait()
        return "unaccepted invocation"

    def binding(self, rid, operation):
        if self is agent and rid == request_id and mode == "binding":
            raise refusal
        return actual_bind(self, rid, operation)

    def cleanup(self, rid, *args, **kwargs):
        if self is agent and rid == request_id:
            cleanup_observations.append(tuple(task.done() for task in accepted))
            assert all(task.done() for task in accepted), "invocation registration retired before accepted child physically joined"
        return actual_cleanup(self, rid, *args, **kwargs)

    class RefusedBindingCallback(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if mode.startswith("callback") and getattr(callback, "__qualname__", "").endswith("capture_request_operation_binding.<locals>.publish.<locals>.release"):
                if mode == "callback-after":
                    super().add_done_callback(callback, context=context)
                raise refusal
            return super().add_done_callback(callback, context=context)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname.endswith(".run_isolated_operation"):
            if mode == "allocation":
                refused.append((coroutine, operation))
                raise refusal
            task = RefusedBindingCallback(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    try:
        monkeypatch.setattr(type(agent), "bind_request_operation", binding)
        monkeypatch.setattr(type(agent), "_cleanup_cancelled_request", cleanup)
        loop.set_task_factory(factory)
        owner = create_owned_task(invoke(agent, request_id=request_id), retain_originals=True)
        outcome = await await_owned_task(owner)
        loop.set_task_factory(previous)
        if mode == "allocation":
            from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
            assert refusal in _explicit_control_chain(outcome.error)
            assert execution_cleanup_debts(outcome.error), "opaque invocation admission was falsely settled"
        else:
            assert outcome.error is refusal, "invocation admission failure replaced original refusal"
        assert all(task.done() for task in accepted), "invocation returned with original accepted child still pending"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair), "refused invocation leaked unstarted operation"
        assert not any(task in members for members in vars(agent).get("_request_operation_tasks", {}).values() for task in accepted), "failed invocation binding left original operation published"
        assert agent._active_request_generation_counts == ({(request_id, initial): 1} if prior else {})
        assert agent._active_request_counts == ({request_id: 1} if prior else {})
        assert cleanup_observations
        if not prior:
            assert await agent.wait_for_request_completion(request_id, generation=agent._next_request_generation) is RequestCompletionDisposition.ABANDONED
        assert await db.fetchval("SELECT count(*) FROM invocation_admission_effects") == 0
    finally:
        loop.set_task_factory(previous)
        release.set()
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        if owner is not None:
            await await_owned_task(owner)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        monkeypatch.setattr(type(agent), "_cleanup_cancelled_request", actual_cleanup)
        for (rid, generation), count in tuple(agent._active_request_generation_counts.items()):
            for _ in range(count):
                actual_cleanup(agent, rid, disposition=RequestCompletionDisposition.ABANDONED, generation=generation)
        await agent.llm_service.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,stop", [(kind, False)
    for kind in (None, "plain", "unknown", "committed", "sdk", "audit")]
    + [("sdk", True), ("audit", True)])
async def test_isolated_invocation_recovers_original_terminal_cancellation_receipt(
    tmp_path, monkeypatch, kind, stop,
):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError, ToolAuditUnconfirmedError, _explicit_control_chain,
    )
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    await db.execute("CREATE TABLE original_invocation_receipt_effects (id INTEGER)")
    original = asyncio.CancelledError("original isolated invocation acknowledgement")
    if kind not in {None, "plain"}:
        original.__cause__ = (ResourceOutcomeUnknown("original SDK acknowledgement unknown") if kind == "sdk"
            else ToolAuditUnconfirmedError("original audit acknowledgement unknown") if kind == "audit"
            else ExecutionCommitOutcomeError(kind))

    generations = []
    @bind_async_invocation("request_id", track_request_lifecycle=True)
    async def invoke(self, request_id=None):
        await db.execute("INSERT INTO original_invocation_receipt_effects VALUES (1)")
        generations.append(agent._active_request_generations[request_id])
        if stop:
            assert agent.cancel_current_request(request_id)
        if kind is not None:
            raise original
        return "original invocation completed"

    try:
        outcome = await await_owned_task(create_owned_task(
            invoke(agent, request_id="original-terminal-receipt"), retain_originals=True))
        if kind is None:
            assert outcome.error is outcome.cancellation is None
            assert outcome.result == "original invocation completed"
        else:
            assert original in tuple(_explicit_control_chain(outcome.error)), (
                "Isolated invocation replaced its original terminal cancellation with typed Stop")
        assert await db.fetchval("SELECT COUNT(*) FROM original_invocation_receipt_effects") == 1
        assert not any(vars(agent).get("_request_operation_tasks", {}).values())
        disposition = await agent.wait_for_request_completion("original-terminal-receipt", generation=generations[0])
        assert disposition is (RequestCompletionDisposition.ABANDONED
            if kind in {"unknown", "committed", "sdk", "audit"} else RequestCompletionDisposition.COMPLETED), (
                "Original invocation uncertainty was retired as successful completion")
    finally:
        await agent.llm_service.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "unknown", "committed", "sdk", "audit"])
@pytest.mark.parametrize("stop", [False, True])
async def test_kite_endpoint_recovers_original_terminal_cancellation_before_stop(
    tmp_path, monkeypatch, kind, stop,
):
    from fastapi import Response
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.endpoints.agent import invoke_agent
    from kestrel_sovereign.execution_custody import (
        ExecutionCommitOutcomeError, ToolAuditUnconfirmedError, _explicit_control_chain,
    )
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    from tests.unit.test_kite_release_evidence import _kite_evidence_env
    evidence_home = tmp_path / "kite-evidence-home"
    evidence_home.mkdir(mode=0o700)
    _kite_evidence_env(monkeypatch, evidence_home)
    await db.execute("CREATE TABLE kite_original_receipt_effects (id INTEGER)")
    request_id = "original-kite-terminal-receipt"
    original = asyncio.CancelledError("original Kite observation acknowledgement")
    if kind is not None:
        original.__cause__ = (ResourceOutcomeUnknown("original Kite effect uncertain") if kind == "sdk"
            else ToolAuditUnconfirmedError("original Kite audit uncertain") if kind == "audit"
            else ExecutionCommitOutcomeError(kind))
    async def native_evidence(*args, **kwargs):
        await db.execute("INSERT INTO kite_original_receipt_effects VALUES (1)")
        if stop:
            assert agent.cancel_current_request(request_id)
        if kind is not None:
            raise original
        return "diagnostics", {"completed": True}
    monkeypatch.setattr("kestrel_sovereign.endpoints.agent._kite_runtime_observation", native_evidence)
    request = _kite_invocation_request(agent, request_id)
    endpoint = getattr(invoke_agent, "__wrapped__", invoke_agent)
    try:
        outcome = await await_owned_task(create_owned_task(endpoint(request, Response()), retain_originals=True))
        if kind is None:
            assert outcome.error is outcome.cancellation is None
        else:
            assert original in tuple(_explicit_control_chain(outcome.error)), (
                "Kite endpoint replaced its original terminal cancellation with Stop or synthetic cancellation")
        assert await db.fetchval("SELECT COUNT(*) FROM kite_original_receipt_effects") == 1
    finally:
        await agent.llm_service.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "sdk", "audit"])
@pytest.mark.parametrize("stop", [False, True])
async def test_ordinary_endpoint_preserves_original_tool_uncertainty_after_stop(
    tmp_path, monkeypatch, kind, stop,
):
    from fastapi import Response
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.endpoints.agent import invoke_agent
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, _explicit_control_chain
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    await db.execute("CREATE TABLE ordinary_original_receipt_effects (id INTEGER)")
    request_id = "ordinary-original-terminal-receipt"
    original = asyncio.CancelledError("original ordinary invocation acknowledgement")
    if kind is not None:
        original.__cause__ = (ResourceOutcomeUnknown("ordinary invocation effect uncertain")
            if kind == "sdk" else ToolAuditUnconfirmedError("ordinary invocation audit uncertain"))

    @bind_async_invocation("invocation_id", track_request_lifecycle=True)
    async def process_input(self, user_input, *, invocation_id=None, **kwargs):
        await db.execute("INSERT INTO ordinary_original_receipt_effects VALUES (1)")
        if stop:
            assert agent.cancel_current_request(invocation_id)
        if kind is not None:
            raise original
        return "ordinary invocation completed"

    monkeypatch.setattr(agent, "process_input", process_input.__get__(agent, type(agent)))
    endpoint = getattr(invoke_agent, "__wrapped__", invoke_agent)
    request = _invocation_request(agent, {"input": "work", "request_id": request_id})
    try:
        outcome = await await_owned_task(create_owned_task(endpoint(request, Response()), retain_originals=True))
        if kind is None:
            assert outcome.error is outcome.cancellation is None
            assert outcome.result["response"] == ("Request stopped during execution."
                if stop else "ordinary invocation completed")
        else:
            assert original in tuple(_explicit_control_chain(outcome.error)), (
                "Ordinary endpoint acknowledged Stop while original tool uncertainty remained")
        assert await db.fetchval("SELECT COUNT(*) FROM ordinary_original_receipt_effects") == 1
    finally:
        await agent.llm_service.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "bridge", "direct"])
@pytest.mark.parametrize("kind", [None, "sdk", "audit"])
async def test_formatted_stream_terminal_retains_actual_abandoned_generation(tmp_path, monkeypatch, transport, kind):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.agent.invocation import bind_async_generator_invocation
    from kestrel_sovereign.endpoints.agent import stream_agent_response
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, _explicit_control_chain
    from kestrel_sovereign.features.bridge.feature import BridgeFeature
    from kestrel_sovereign.features.bridge.protocol import BridgeRequest
    from kestrel_sovereign.features.bridge.router import get_router
    from kestrel_sovereign.streams.tap import AgentStreamTap
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    await db.execute("CREATE TABLE formatted_stream_original_effects (id INTEGER)")
    request_id = "formatted-original-terminal"
    original = (ResourceOutcomeUnknown("private original tool outcome uncertain") if kind == "sdk"
        else ToolAuditUnconfirmedError("private original audit outcome uncertain"))
    generations = []
    async def process_input_streaming(self, user_input, *, request_id=None, **kwargs):
        generations.append(agent._active_request_generations[request_id])
        await db.execute("INSERT INTO formatted_stream_original_effects VALUES (1)")
        yield "original partial text"
        if kind is not None:
            raise original
    # The direct component owns its decorated registration. Endpoint cases
    # exercise their own actual registration without a nested decorator that
    # could independently preserve the same abandoned-generation tombstone.
    if transport == "direct":
        process_input_streaming = bind_async_generator_invocation(
            "request_id", track_request_lifecycle=True)(process_input_streaming)
    monkeypatch.setattr(agent, "process_input_streaming", process_input_streaming.__get__(agent, type(agent)))
    AgentStreamTap.reset()
    tap = AgentStreamTap.get_instance()
    response = None
    try:
        if transport == "direct":
            async def consume():
                return [frame async for frame in agent.process_input_streaming("work", request_id=request_id)]
            outcome = await await_owned_operation(consume(), retain_originals=True)
            if kind is None:
                assert outcome.error is outcome.cancellation is None
                assert outcome.result == ["original partial text"]
            else:
                assert original in tuple(_explicit_control_chain(outcome.error))
            assert await agent.wait_for_request_completion(request_id, generation=generations[0]) is (
                RequestCompletionDisposition.COMPLETED if kind is None else RequestCompletionDisposition.ABANDONED), (
                    "Standalone stream uncertainty retired its actual generation as COMPLETED")
            assert await db.fetchval("SELECT COUNT(*) FROM formatted_stream_original_effects") == 1
            return
        request = _invocation_request(agent, {"input": "work", "request_id": request_id})
        if transport == "bridge":
            bridge = BridgeFeature(agent)
            await bridge.initialize()
            agent.features["BridgeFeature"] = bridge
            route = next(route for route in get_router().routes if getattr(route, "path", None) == "/api/bridge/stream")
            endpoint = getattr(route.endpoint, "__wrapped__", route.endpoint)
            response = await endpoint(request, BridgeRequest(message="work", request_id=request_id))
        else:
            endpoint = getattr(stream_agent_response, "__wrapped__", stream_agent_response)
            response = await endpoint(request)
        frames = [frame async for frame in response.body_iterator]
        wire = "".join(frame.decode() if isinstance(frame, bytes) else frame for frame in frames)
        assert "original partial text" in wire
        if kind is not None:
            assert "uncertain" in wire.lower() and "private original" not in wire
        if transport == "http":
            delivery = response.headers["X-Stream-Delivery-ID"]
            assert tap._requests[delivery].drain_retained is (kind is None), (
                "Formatted terminal uncertainty retained successful speech drain custody")
            tap.unregister(delivery)
        disposition = await agent.wait_for_request_completion(request_id, generation=generations[0])
        assert disposition is (RequestCompletionDisposition.COMPLETED if kind is None
            else RequestCompletionDisposition.ABANDONED), (
                "Formatted terminal uncertainty retired its actual request generation as COMPLETED")
        assert await db.fetchval("SELECT COUNT(*) FROM formatted_stream_original_effects") == 1
    finally:
        if response is not None:
            await await_owned_operation(response.body_iterator.aclose(), retain_originals=True)
        AgentStreamTap.reset()
        await agent.llm_service.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "sdk", "audit"])
async def test_direct_stream_close_uncertainty_preserves_actual_generation(tmp_path, monkeypatch, kind):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.agent.invocation import bind_async_generator_invocation
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, _explicit_control_chain
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    await db.execute("CREATE TABLE direct_stream_close_effects (phase TEXT)")
    request_id = "original-direct-stream-close"
    original = (ResourceOutcomeUnknown("original source close effect uncertain") if kind == "sdk"
        else ToolAuditUnconfirmedError("original source close audit uncertain"))
    generations = []
    @bind_async_generator_invocation("request_id", track_request_lifecycle=True)
    async def source(self, request_id=None):
        generations.append(agent._active_request_generations[request_id])
        try:
            await db.execute("INSERT INTO direct_stream_close_effects VALUES ('body')")
            yield "original body"
        finally:
            await db.execute("INSERT INTO direct_stream_close_effects VALUES ('close')")
            if kind is not None:
                raise original
    iterator = source(agent, request_id=request_id)
    try:
        assert await anext(iterator) == "original body"
        closed = await await_owned_operation(iterator.aclose(), retain_originals=True)
        if kind is None:
            assert closed.error is closed.cancellation is None
        else:
            assert original in tuple(_explicit_control_chain(closed.error))
        assert await agent.wait_for_request_completion(request_id, generation=generations[0]) is (
            RequestCompletionDisposition.COMPLETED if kind is None else RequestCompletionDisposition.ABANDONED), (
                "Original stream close uncertainty was retired as successful completion")
        assert await db.fetchval("SELECT COUNT(*) FROM direct_stream_close_effects") == 2
    finally:
        await await_owned_operation(iterator.aclose(), retain_originals=True)
        await agent.llm_service.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,phase", [("http", "source"), ("bridge", "source"), ("http", "tap")])
@pytest.mark.parametrize("kind", [None, "sdk", "audit"])
async def test_actual_stream_retirement_classifies_late_terminal_outcomes(tmp_path, monkeypatch, transport, phase, kind):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.endpoints.agent import stream_agent_response
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, _explicit_control_chain
    from kestrel_sovereign.features.bridge.feature import BridgeFeature
    from kestrel_sovereign.features.bridge.protocol import BridgeRequest
    from kestrel_sovereign.features.bridge.router import get_router
    from kestrel_sovereign.streams.tap import AgentStreamTap
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    await db.execute("CREATE TABLE late_stream_retirement_effects (phase TEXT)")
    request_id = "original-late-stream-terminal"
    original = (ResourceOutcomeUnknown("original late stream effect uncertain") if kind == "sdk"
        else ToolAuditUnconfirmedError("original late stream audit uncertain"))
    entered, release = asyncio.Event(), asyncio.Event()
    handle = (tmp_path / "actual-late-stream-resource").open("w")
    closes, generations = [], []
    async def source(self, user_input, *, request_id=None, **kwargs):
        generations.append(agent._active_request_generations[request_id])
        try:
            await db.execute("INSERT INTO late_stream_retirement_effects VALUES ('body')")
            yield "original body"
            if phase == "source":
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    if kind is not None:
                        raise original
                    raise
        finally:
            await db.execute("INSERT INTO late_stream_retirement_effects VALUES ('close')")
            closes.append(asyncio.current_task())
            handle.close()
    monkeypatch.setattr(agent, "process_input_streaming", source.__get__(agent, type(agent)))
    AgentStreamTap.reset()
    tap = AgentStreamTap.get_instance()
    actual_finish = tap.finish
    if phase == "tap":
        async def finish(delivery_id, *, completed=False):
            await actual_finish(delivery_id, completed=completed)
            await db.execute("INSERT INTO late_stream_retirement_effects VALUES ('tap')")
            if kind is not None:
                raise original
        monkeypatch.setattr(tap, "finish", finish)
    response, reader = None, None
    try:
        request = _invocation_request(agent, {"input": "work", "request_id": request_id})
        if transport == "bridge":
            bridge = BridgeFeature(agent)
            await bridge.initialize()
            agent.features["BridgeFeature"] = bridge
            endpoint = next(route.endpoint for route in get_router().routes if getattr(route, "path", None) == "/api/bridge/stream")
            endpoint = getattr(endpoint, "__wrapped__", endpoint)
            response = await endpoint(request, BridgeRequest(message="work", request_id=request_id))
        else:
            endpoint = getattr(stream_agent_response, "__wrapped__", stream_agent_response)
            response = await endpoint(request)
        assert "original body" in await anext(response.body_iterator)
        reader = create_owned_task(anext(response.body_iterator), retain_originals=True)
        if phase == "source":
            await asyncio.wait_for(entered.wait(), 1)
            reader.cancel("actual response reader disconnected")
        outcome = await await_owned_task(reader)
        if kind is not None:
            assert original in tuple(_explicit_control_chain(outcome.error))
        elif phase == "source":
            assert isinstance(outcome.error, asyncio.CancelledError)
        else:
            assert isinstance(outcome.error, StopAsyncIteration)
        assert handle.closed and len(closes) == 1
        if transport == "http":
            delivery = response.headers["X-Stream-Delivery-ID"]
            if kind is not None and phase == "tap":
                assert not tap._requests[delivery].is_live(), (
                    "Late terminal tap uncertainty retained successful drain authority")
            tap.unregister(delivery)
        assert await agent.wait_for_request_completion(request_id, generation=generations[0]) is (
            RequestCompletionDisposition.COMPLETED if kind is None else RequestCompletionDisposition.ABANDONED), (
                "Late stream retirement uncertainty was classified from stale completion state")
        assert await db.fetchval("SELECT COUNT(*) FROM late_stream_retirement_effects") == (3 if phase == "tap" else 2)
    finally:
        release.set()
        if reader is not None:
            await await_owned_task(reader)
        if response is not None:
            await await_owned_operation(response.body_iterator.aclose(), retain_originals=True)
        AgentStreamTap.reset()
        handle.close()
        await agent.llm_service.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [
    pytest.param("eager", marks=pytest.mark.skipif(sys.version_info < (3, 12), reason="native eager Task requires Python 3.12")),
    pytest.param("eager-preconsumed", marks=pytest.mark.skipif(sys.version_info < (3, 12), reason="native eager Task requires Python 3.12")),
    "prior-callback",
])
@pytest.mark.parametrize("kind", [None, "sdk", "audit"])
@pytest.mark.parametrize("stop", [False, True])
async def test_actual_invocation_retains_eager_or_preobserved_original_completion(tmp_path, monkeypatch, mode, kind, stop):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.agent.invocation import InvocationCancelledError
    from kestrel_sovereign.execution_custody import (
        ToolAuditUnconfirmedError, _explicit_control_chain, execution_cleanup_debts,
        is_tool_execution_terminal_error,
    )
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    await db.execute("CREATE TABLE eager_original_payload_effects (id INTEGER)")
    request_id = "original-eager-invocation"
    original = asyncio.CancelledError("original eager invocation physical failure")
    if kind is not None:
        original.__cause__ = (ResourceOutcomeUnknown("original eager external effect uncertain") if kind == "sdk"
            else ToolAuditUnconfirmedError("original eager audit uncertain"))
    effect = tmp_path / "actual-original-eager-invocation-effect"
    original_owners, generations, consumed = [], [], []
    @bind_async_invocation("request_id", track_request_lifecycle=True)
    async def invoke(self, request_id=None):
        await db.execute("INSERT INTO eager_original_payload_effects VALUES (1)")
        return "payload must stay unentered"
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def consume(task):
        try:
            task.exception()
        except asyncio.CancelledError as error:
            consumed.append(error)
    def factory(loop, coroutine, **options):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is None or not code.co_qualname.endswith(".run_isolated_operation"):
            return previous(loop, coroutine, **options) if previous else asyncio.Task(coroutine, loop=loop, **options)
        async def outer():
            if mode == "prior-callback":
                await asyncio.sleep(0)
            generations.append(agent._active_request_generations[request_id])
            coroutine.close()
            effect.write_text("original native outer effect completed once")
            if stop:
                assert agent.cancel_current_request(request_id)
            if kind is not None:
                raise original
            return "original outer success"
        if mode.startswith("eager"):
            options["eager_start"] = True
        task = asyncio.Task(outer(), loop=loop, **options)
        original_owners.append(task)
        if mode == "prior-callback":
            task.add_done_callback(consume)
        elif mode == "eager-preconsumed":
            consume(task)
        return task
    try:
        loop.set_task_factory(factory)
        outcome = await await_owned_task(create_owned_task(
            invoke(agent, request_id=request_id), retain_originals=True))
        loop.set_task_factory(previous)
        if kind is None:
            if stop:
                assert isinstance(outcome.error, InvocationCancelledError)
            else:
                assert outcome.error is outcome.cancellation is None and outcome.result == "original outer success"
        else:
            assert is_tool_execution_terminal_error(outcome.error), (
                "Eager or preobserved original uncertainty became typed invocation Stop")
            if mode == "eager":
                assert original in tuple(_explicit_control_chain(outcome.error))
            else:
                assert consumed and consumed[0] is original
                assert any(task is original_owners[0] for _, task in execution_cleanup_debts(outcome.error))
        assert await agent.wait_for_request_completion(request_id, generation=generations[0]) is (
            RequestCompletionDisposition.COMPLETED if kind is None else RequestCompletionDisposition.ABANDONED), (
                "Eager or preobserved original uncertainty retired as completed")
        assert effect.read_text() == "original native outer effect completed once"
        assert len(original_owners) == 1 and original_owners[0].done()
        assert await db.fetchval("SELECT COUNT(*) FROM eager_original_payload_effects") == 0
    finally:
        loop.set_task_factory(previous)
        for task in original_owners:
            await await_owned_task(task)
        await agent.llm_service.close()
        await db.close()
