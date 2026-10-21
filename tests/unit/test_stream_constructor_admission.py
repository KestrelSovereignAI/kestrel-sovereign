"""Original native stream Tasks remain owned at every constructor publication."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission, await_owned_task, create_owned_task, owned_task_error,
)
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError, ExecutionStreamLifecycle, _explicit_control_chain,
    execution_commit_outcome, execution_stream,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock, bind_transition_lock_reentry


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("phase,factory_kind,first_kind", [
    (phase, factory, kind)
    for phase in ("owner-admission", "owner-completion", "lifecycle", "watcher-allocation", "watcher-admission", "watcher-completion", "privacy-completion", "join-publication")
    for factory, kind in [("default", None), *[(name, item) for name in ("custom", "privacy") for item in (None, "ordinary", "unknown", "committed")]]
    if phase != "owner-admission" or kind is None
    if phase != "privacy-completion" or factory == "privacy"
])
async def test_native_constructor_failure_joins_all_original_tasks(tmp_path, factory_kind, first_kind, phase, eager):
    db = await AsyncDatabase.sqlite(str(tmp_path / "stream-effects.db"))
    await db.execute("CREATE TABLE effects(id INTEGER)")
    first = (None if first_kind is None else OSError("original gate A")
        if first_kind == "ordinary" else ExecutionCommitOutcomeError(first_kind))
    second = OSError("constructor publication B: " + phase)
    third = OSError("original join publication C")
    accepted, bodies = [], []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()

    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            name = getattr(callback, "__name__", "")
            if self.role == "run_owned":
                if name == "close_unstarted" and phase == "owner-admission":
                    raise second
                if name == "close_unstarted" and first is not None:
                    raise first
                if name == "_publish_owner_completion" and phase in {"owner-completion", "join-publication"}:
                    raise second
                if name == "completed" and phase == "privacy-completion":
                    raise second
                if name == "mark_complete" and phase == "join-publication":
                    raise third
            if self.role == "_watch_owner_cancellation":
                if (name == "close_unstarted" and phase == "watcher-admission") or (
                    name == "_interrupt_if_watch_never_started" and phase == "watcher-completion"
                ):
                    raise second
            return super().add_done_callback(callback, context=context)

    def native_factory(loop, coroutine, **options):
        operation = coroutine.cr_frame.f_locals.get("operation") if coroutine.cr_frame else None
        role = getattr(getattr(operation, "cr_code", None), "co_name", "")
        if role == "run" and "ReentrantTransitionLock.start_owned_native_work" in operation.__qualname__:
            role = "run_owned"
        if role in {"run_owned", "_watch_owner_cancellation"}:
            if role == "_watch_owner_cancellation" and phase == "watcher-allocation":
                raise second
            task = PublicationTask(coroutine, loop=loop, **options, **({"eager_start": True} if eager else {}))
            task.role = role
            accepted.append(task)
            return task
        return asyncio.Task(coroutine, loop=loop, **options)

    def custom_factory(body, context):
        admission = OwnedTaskAdmission()
        try:
            return create_owned_task(body(), context=context, admission=admission, retain_originals=True)
        except BaseException:
            if admission.task is None:
                raise
            return admission.task

    def started(iterator, task):
        if phase == "lifecycle":
            raise second

    async def source():
        bodies.append(asyncio.current_task())
        await db.execute_commit("INSERT INTO effects VALUES (1)")
        yield b"must never dispatch"

    lock = ReentrantTransitionLock()
    factory = (None if factory_kind == "default" else lock.capture_native_work_factory()
        if factory_kind == "privacy" else custom_factory)
    stream = execution_stream(None, source(), lifecycle=ExecutionStreamLifecycle(factory, started, lambda: None))
    loop.set_task_factory(native_factory)
    reader = asyncio.create_task(anext(stream))
    try:
        done, _ = await asyncio.wait({reader}, timeout=1)
        assert reader in done, "constructor failure left its original reader blocked"
        with pytest.raises(BaseException) as caught:
            await reader
        assert accepted and all(task.done() for task in accepted), "constructor returned with an unjoined original Task"
        chain = tuple(_explicit_control_chain(caught.value))
        if factory_kind == "privacy" and phase == "privacy-completion" and first is not None:
            # The earlier failed canonical gate never reaches privacy's later
            # publication. Fail-fast returns that exact failed Task instead.
            assert second not in chain
        else:
            assert second in chain
        if phase == "join-publication":
            assert third in chain and third.__cause__ is None
        if first is not None:
            assert first in chain
            owner, = [task for task in accepted if task.role == "run_owned"]
            assert first in tuple(_explicit_control_chain(owned_task_error(owner)))
        assert execution_commit_outcome(caught.value) == (first_kind if first_kind in {"unknown", "committed"} else None)
        assert second.__cause__ is None and (first is None or first.__cause__ is None)
        assert not lock._native_work and not lock._foreign_native_dependencies
        assert not bodies and await db.fetchval("SELECT COUNT(*) FROM effects") == 0
    finally:
        loop.set_task_factory(previous)
        try:
            for task in accepted:
                if not task.done():
                    task.cancel()
                try:
                    await await_owned_task(task)
                except asyncio.CancelledError:
                    assert task.done()  # Even a broken error cache cannot skip DB cleanup.
            if not reader.done():
                reader.cancel()
            await await_owned_task(reader)
            closing = create_owned_task(stream.aclose(), retain_originals=True)
            await await_owned_task(closing)
        finally:
            await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["agent", "bridge"])
@pytest.mark.parametrize("join_failure", [False, True])
async def test_public_http_constructor_failure_physically_joins_original(tmp_path, surface, join_failure):
    """Actual handlers/Starlette requests; this is not an auth/cognition proof."""
    from fastapi import FastAPI
    from starlette.requests import Request
    from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
    from kestrel_sovereign.endpoints.agent import stream_agent_response
    from kestrel_sovereign.features.bridge.protocol import BridgeRequest
    from kestrel_sovereign.features.bridge.router import get_router

    db = await AsyncDatabase.sqlite(str(tmp_path / "http-effects.db"))
    await db.execute("CREATE TABLE effects(id INTEGER)")
    opening, joining = OSError("actual initial publication"), OSError("actual join publication")
    accepted, bodies = [], []
    async def session(*args, **kwargs):
        return SimpleNamespace(id="constructor-session", gateway_session_id=None)
    async def log(*args, **kwargs):
        pass
    async def resolve(session_id):
        return session_id
    class Agent(RequestLifecycleMixin):
        did = "did:test:constructor"
        def __init__(self):
            self._current_request_id = None
            self._active_request_ids, self._cancelled_requests = set(), set()
            self._active_request_counts, self._active_request_started_at = {}, {}
            self._request_completion_events = {}
            self.features = {"BridgeFeature": SimpleNamespace(get_or_create_session=session, log_invocation=log)}
            self.storage = SimpleNamespace(resolve_session_id=resolve)
        async def process_input_streaming(self, *args, **kwargs):
            bodies.append(asyncio.current_task())
            await db.execute_commit("INSERT INTO effects VALUES (1)")
            yield "must never dispatch"
    app = FastAPI()
    app.state.agent, app.state.agent_manager = Agent(), None
    payload = json.dumps({"input": "must never dispatch", "request_id": "http-constructor"}).encode()
    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}
    path = "/api/agent/stream" if surface == "agent" else "/api/bridge/stream"
    request = Request({"type": "http", "method": "POST", "path": path,
        "headers": [], "app": app, "query_string": b"", "scheme": "http",
        "server": ("test", 80), "client": ("test", 1)}, receive)
    endpoint = getattr(stream_agent_response, "__wrapped__", stream_agent_response)
    if surface == "bridge":
        route, = [route for route in get_router().routes if getattr(route, "path", None) == path]
        endpoint = getattr(route.endpoint, "__wrapped__", route.endpoint)
        response = await endpoint(request, BridgeRequest(message="must never dispatch", request_id="http-constructor"))
    else:
        response = await endpoint(request)

    class PublicationTask(asyncio.Task):
        joins = 0
        def add_done_callback(self, callback, *, context=None):
            name = getattr(callback, "__name__", "")
            if name == "_publish_owner_completion":
                raise opening
            if name == "mark_complete":
                self.joins += 1
                if join_failure:
                    raise joining
            return super().add_done_callback(callback, context=context)
    def factory(loop, coroutine, **options):
        operation = coroutine.cr_frame.f_locals.get("operation") if coroutine.cr_frame else None
        if getattr(getattr(operation, "cr_code", None), "co_name", "") == "run_owned":
            task = PublicationTask(coroutine, loop=loop, **options)
            accepted.append(task)
            return task
        return asyncio.Task(coroutine, loop=loop, **options)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    loop.set_task_factory(factory)
    returned = []
    try:
        try:
            async for item in response.body_iterator:
                returned.append(item)
        except BaseException as error:
            chain = tuple(_explicit_control_chain(error))
            assert opening in chain
            if join_failure:
                assert joining in chain
        owner, = accepted
        assert owner.done() and owner.joins > 0, "public HTTP failure did not join its original accepted producer"
        assert opening.__cause__ is joining.__cause__ is None
        assert not bodies and await db.fetchval("SELECT COUNT(*) FROM effects") == 0
    finally:
        loop.set_task_factory(previous)
        try:
            for task in accepted:
                if not task.done():
                    task.cancel()
                await await_owned_task(task)
            closing = create_owned_task(response.body_iterator.aclose(), retain_originals=True)
            await await_owned_task(closing)
        finally:
            await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("publication", ["close_unstarted", "completed"])
@pytest.mark.parametrize("receipt", [None, "unknown", "committed"])
async def test_actual_foreign_privacy_frame_retains_failed_native_handoff(tmp_path, publication, receipt):
    db = await AsyncDatabase.sqlite(str(tmp_path / "foreign-effects.db"))
    await db.execute("CREATE TABLE effects(id INTEGER)")
    lock, accepted = ReentrantTransitionLock(), []
    original = ExecutionCommitOutcomeError(receipt) if receipt else OSError("native foreign publication")
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == publication:
                if publication == "completed":
                    frame = asyncio.current_task()
                    assert lock._native_foreign_frames[self] is frame
                    assert self in lock._foreign_native_dependencies[frame]
                raise original
            return super().add_done_callback(callback, context=context)
    def factory(loop, coroutine, **options):
        operation = coroutine.cr_frame.f_locals.get("operation") if coroutine.cr_frame else None
        if "ReentrantTransitionLock.start_owned_native_work" in getattr(operation, "__qualname__", ""):
            task = PublicationTask(coroutine, loop=loop, **options)
            accepted.append(task)
            return task
        return asyncio.Task(coroutine, loop=loop, **options)
    async def effect():
        await db.execute_commit("INSERT INTO effects VALUES (1)")
    async def foreign(token):
        with bind_transition_lock_reentry(token):
            async with lock:
                child = lock.start_owned_native_work(effect)
                outcome = await await_owned_task(child)
                assert child.done() and original in tuple(_explicit_control_chain(outcome.error))
                assert execution_commit_outcome(outcome.error) == receipt
                assert not lock._native_work and not lock._foreign_native_dependencies
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    reader = None
    try:
        async with lock:
            loop.set_task_factory(factory)
            reader = create_owned_task(foreign(lock.current_reentry_token()), retain_originals=True)
            try:
                done, _ = await asyncio.wait({reader}, timeout=1)
                assert reader in done, "failed native publication stranded its actual foreign frame"
                assert (await await_owned_task(reader)).error is None
                assert accepted and all(task.done() for task in accepted)
                assert original.__cause__ is None
                assert await db.fetchval("SELECT COUNT(*) FROM effects") == 0
            finally:
                loop.set_task_factory(previous)
                for task in accepted:
                    if not task.done():
                        task.cancel()
                    await await_owned_task(task)
                if not reader.done():
                    reader.cancel()
                await await_owned_task(reader)
        assert not lock.locked()
    finally:
        loop.set_task_factory(previous)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign", [False, True], ids=["original-consumer", "foreign-consumer"])
async def test_lazy_genai_stream_retains_original_creation_authority(tmp_path, foreign):
    from kestrel_sovereign.execution_custody import bind_execution_custody, execution_denial_scopes, require_execution_work
    from kestrel_sovereign.llm.genai_custody import genai_stream
    from tests.unit.test_execution_custody import Authority

    db = await AsyncDatabase.sqlite(str(tmp_path / "creation-effects.db"))
    await db.execute("CREATE TABLE effects(id INTEGER)")
    class Provider:
        @genai_stream
        async def source(self, client):
            require_execution_work()
            await db.execute_commit("INSERT INTO effects VALUES (1)")
            yield b"must not borrow the later consumer's context"
    stream, reader = None, None
    try:
        with bind_execution_custody(Authority()) as original:
            stream = Provider().source(object())
        if foreign:
            reader = create_owned_task(anext(stream), retain_originals=True)
            error = (await await_owned_task(reader)).error
        else:
            error = None
            try:
                await anext(stream)
            except BaseException as caught:
                error = caught
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 0
        assert original in execution_denial_scopes(error)
    finally:
        try:
            if reader is not None:
                await await_owned_task(reader)
            if stream is not None:
                await await_owned_task(create_owned_task(stream.aclose(), retain_originals=True))
        finally:
            await db.close()
