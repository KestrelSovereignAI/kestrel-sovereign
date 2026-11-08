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
@pytest.mark.parametrize("reserved", [False, True], ids=["lazy", "reserved"])
@pytest.mark.parametrize("eager", [False, True] if hasattr(asyncio, "eager_task_factory") else [False])
@pytest.mark.parametrize("phase,factory_kind,first_kind", [
    (phase, factory, kind)
    for phase in ("owner-admission", "owner-completion", "lifecycle", "resource-lifecycle", "watcher-allocation", "watcher-admission", "watcher-completion", "privacy-completion", "join-publication")
    for factory, kind in [("default", None), *[(name, item) for name in ("custom", "privacy") for item in (None, "ordinary", "unknown", "committed")]]
    if phase != "owner-admission" or kind is None
    if phase != "privacy-completion" or factory == "privacy"
])
async def test_native_constructor_failure_joins_all_original_tasks(tmp_path, factory_kind, first_kind, phase, eager, reserved):
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

    def resource_started(iterator, task, resource):
        assert resource is stream and resource.retirement_task is not task
        assert resource._iterator is iterator and iterator.producer_task is task
        if phase == "resource-lifecycle":
            raise second

    async def source():
        bodies.append(asyncio.current_task())
        await db.execute_commit("INSERT INTO effects VALUES (1)")
        yield b"must never dispatch"

    lock = ReentrantTransitionLock()
    factory = (None if factory_kind == "default" else lock.capture_native_work_factory()
        if factory_kind == "privacy" else custom_factory)
    stream = execution_stream(None, source(), lifecycle=ExecutionStreamLifecycle(
        factory, started, lambda: None, resource_started=resource_started))
    loop.set_task_factory(native_factory)
    async def drive():
        if reserved:
            await stream.reserve()
        return await anext(stream)
    reader = asyncio.create_task(drive())
    try:
        done, _ = await asyncio.wait({reader}, timeout=1)
        assert reader in done, "constructor failure left its original reader blocked"
        with pytest.raises(BaseException) as caught:
            await reader
        assert accepted and all(task.done() for task in accepted), "constructor returned with an unjoined original Task"
        chain = tuple(_explicit_control_chain(caught.value))
        # Even a failed start gate retains its actual native ownership.
        # A later failed publication is another real original, not permission
        # to discard either failure or enter the body behind the failed gate.
        assert second in chain, "original native publication error was discarded"
        if phase == "join-publication":
            assert third in chain and third.__cause__ is None
        if first is not None:
            assert first in chain
            owner, = [task for task in accepted if task.role == "run_owned"]
            assert first in tuple(_explicit_control_chain(owned_task_error(owner)))
        assert execution_commit_outcome(caught.value) == (first_kind if first_kind in {"unknown", "committed"} else None)
        assert second.__cause__ is None and (first is None or first.__cause__ is None)
        assert not lock._native_work
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
                    assert lock._members[self] is frame
                    assert lock._admissions[self].task is self
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
            async with lock.policy_span():
                child = lock.start_owned_native_work(effect)
                outcome = await await_owned_task(child)
                assert child.done() and original in tuple(_explicit_control_chain(outcome.error))
                assert execution_commit_outcome(outcome.error) == receipt
                assert not lock._native_work
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    reader = None
    try:
        async with lock.policy_span():
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


@pytest.mark.asyncio
@pytest.mark.parametrize('pull', [False, True], ids=['unpulled', 'pulled'])
@pytest.mark.parametrize('deny', [False, True], ids=['live', 'retired'])
async def test_public_reservation_defers_payload_and_joins_its_exact_original_owner(tmp_path, pull, deny):
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody, ExecutionAuthorityError, ProcessRuntimeExecutionFence,
        bind_execution_custody_snapshot, reserve_execution_stream,
    )

    path = tmp_path / 'reserved-native-source'
    handle = path.open('w')
    owners, bodies = [], []
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            bodies.append(asyncio.current_task())
            handle.write('payload\n')
            handle.flush()
            return b'payload'
        async def aclose(self):
            handle.write('closed\n')
            handle.close()

    fence = ProcessRuntimeExecutionFence('owned-native-stream-reservation')
    custody = ExecutionCustody(fence)
    lifecycle = ExecutionStreamLifecycle(None, lambda iterator, task: owners.append((iterator, task)), lambda: None)
    with bind_execution_custody_snapshot((custody,)):
        stream = await reserve_execution_stream(None, Source(), lifecycle=lifecycle)
    await asyncio.sleep(0)
    assert not bodies and path.read_text() == ''
    assert len(owners) == 1, 'Reservation did not retain exactly one original native producer'
    original, = owners
    iterator, task = original
    if deny:
        fence.retire()
    if pull:
        if deny:
            with pytest.raises(ExecutionAuthorityError):
                await anext(stream)
        else:
            assert await anext(stream) == b'payload'
            assert bodies == [task]
    await stream.aclose()
    await stream.aclose()
    assert task.done() and iterator.producer_task is task and iterator.owner_task.done()
    assert path.read_text().splitlines() == (['payload', 'closed'] if pull and not deny else ['closed'])
    with pytest.raises(ExecutionAuthorityError, match='closed execution stream'):
        await stream.reserve()


@pytest.mark.asyncio
@pytest.mark.parametrize('pull_first', [False, True], ids=['close-first', 'pull-first'])
@pytest.mark.parametrize('concurrent', [None, False, True], ids=['single', 'repeated', 'concurrent'])
async def test_unstarted_reserved_source_keeps_one_original_failed_close(tmp_path, concurrent, pull_first):
    from kestrel_sovereign.execution_custody import reserve_execution_stream
    from kestrel_sovereign._async_ownership import await_owned_operation

    opening = OSError('original native producer start publication refused')
    close_error = ExecutionCommitOutcomeError('unknown')
    calls, entered, release = [], asyncio.Event(), asyncio.Event()
    handle = (tmp_path / 'native-source-resource').open('w')
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise AssertionError('A refused original worker dispatched payload')
        async def aclose(self):
            calls.append(asyncio.current_task())
            entered.set()
            await release.wait()
            if not handle.closed:
                handle.write('original physical close\n')
                handle.close()
            raise close_error

    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refused = False
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, '__name__', '') == 'close_unstarted':
                raise opening
            return super().add_done_callback(callback, context=context)
    def task_factory(loop, coroutine, **options):
        nonlocal refused
        operation = coroutine.cr_frame.f_locals.get('operation') if coroutine.cr_frame else None
        if not refused and getattr(getattr(operation, 'cr_code', None), 'co_name', '') == 'run_owned':
            refused = True
            return PublicationTask(coroutine, loop=loop, **options)
        return asyncio.Task(coroutine, loop=loop, **options)
    def original_factory(body, context):
        admission = OwnedTaskAdmission()
        try:
            return create_owned_task(body(), context=context, admission=admission, retain_originals=True)
        except BaseException:
            assert admission.task is not None
            return admission.task

    loop.set_task_factory(task_factory)
    try:
        stream = await reserve_execution_stream(None, Source(),
            lifecycle=ExecutionStreamLifecycle(original_factory, lambda *_: None, lambda: None))
    finally:
        loop.set_task_factory(previous)
    if pull_first:
        release.set()
        initial = await await_owned_operation(anext(stream), retain_originals=True)
        assert opening in tuple(_explicit_control_chain(initial.error))
    if concurrent is None:
        release.set()
    first = create_owned_task(await_owned_operation(stream.aclose(), retain_originals=True))
    second = None
    try:
        if concurrent is not None:
            await asyncio.wait_for(entered.wait(), 2)
        if concurrent:
            second = create_owned_task(await_owned_operation(stream.aclose(), retain_originals=True))
        release.set()
        observations = [(await await_owned_task(first)).result]
        if concurrent is not None:
            observations.append((await await_owned_task(second)).result if second is not None else
                await await_owned_operation(stream.aclose(), retain_originals=True))
    finally:
        release.set()
        await await_owned_task(first)
        if second is not None:
            await await_owned_task(second)
    assert len(calls) == 1, 'Repeated observers executed the original failed physical source close again'
    assert all(opening in tuple(_explicit_control_chain(item.error))
        and close_error in tuple(_explicit_control_chain(item.error)) for item in observations)
    assert (tmp_path / 'native-source-resource').read_text() == 'original physical close\n'


@pytest.mark.asyncio
async def test_reserved_native_producer_derives_descendants_while_original_parent_closes(tmp_path):
    from kestrel_sovereign.execution_custody import reserve_execution_stream

    lock, ready = ReentrantTransitionLock(), asyncio.Event()
    streams, scopes, owners, descendants = [], [], [], []
    path = tmp_path / 'original-native-descendant'

    def lifecycle(factory):
        def descendant():
            descendants.append(asyncio.current_task())
            assert lock.current_native_scope() is scopes[0]
            return lifecycle(lock.capture_native_work_factory())
        return ExecutionStreamLifecycle(
            lambda body, context: factory(body, context=context),
            lambda iterator, task: owners.append((iterator, task)), lambda: None,
            descendant_lifecycle=descendant)

    async def leaf():
        assert lock.current_native_scope() is scopes[0]
        path.write_text('original descendant physically completed')
        yield b'descendant'

    async def source():
        child = execution_stream(None, leaf())
        try:
            yield await anext(child)
        finally:
            await child.aclose()

    async def parent():
        async with lock.policy_span():
            scopes.append(lock.current_native_scope())
            streams.append(await reserve_execution_stream(None, source(),
                lifecycle=lifecycle(lock.capture_native_work_factory())))
            ready.set()

    original = create_owned_task(parent(), retain_originals=True)
    try:
        await asyncio.wait_for(ready.wait(), 2)
        assert scopes[0].closing and not original.done()
        assert not path.exists() and len(owners) == 1
        stream, = streams
        assert await anext(stream) == b'descendant'
        await stream.aclose()
        outcome = await await_owned_task(original)
        assert outcome.error is None
        assert len(owners) == 2 and descendants == [task for _, task in owners]
        assert path.read_text() == 'original descendant physically completed'
        assert not lock.locked() and not lock._native_scopes
        assert all(task.done() and iterator.owner_task.done() for iterator, task in owners)
    finally:
        for stream in streams:
            await stream.aclose()
        await await_owned_task(original)


@pytest.mark.asyncio
@pytest.mark.parametrize('returned', [None, object()], ids=['none', 'foreign-object'])
async def test_reserved_native_descendant_lifecycle_refuses_invalid_result_before_payload(tmp_path, returned):
    from kestrel_sovereign.execution_custody import reserve_execution_stream
    from kestrel_sovereign._async_ownership import await_owned_operation

    handle = (tmp_path / 'descendant-resource').open('w')
    calls, owners = [], []
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            calls.append('payload')
            handle.write('unauthorized payload')
            return b'payload'
        async def aclose(self):
            calls.append('close')
            handle.close()
    stream = await reserve_execution_stream(None, Source(), lifecycle=ExecutionStreamLifecycle(
        None, lambda iterator, task: owners.append((iterator, task)), lambda: None,
        descendant_lifecycle=lambda: returned))
    outcome = await await_owned_operation(anext(stream), retain_originals=True)
    await await_owned_operation(stream.aclose(), retain_originals=True)
    assert any(isinstance(error, TypeError) and 'descendant stream lifecycle' in str(error)
        for error in _explicit_control_chain(outcome.error)), 'Invalid descendant lifecycle dispatched source payload'
    assert calls == ['close'] and handle.closed
    assert all(task.done() and iterator.owner_task.done() for iterator, task in owners)


@pytest.mark.asyncio
async def test_every_close_observer_parks_its_original_native_privacy_writer(tmp_path, monkeypatch):
    from contextvars import Context
    from contextlib import asynccontextmanager
    from kestrel_sovereign._async_ownership import OwnedOperation, owned_native_wait
    from kestrel_sovereign.execution_custody import reserve_execution_stream

    lock, entered = ReentrantTransitionLock(), asyncio.Event()
    writer, source_owners = asyncio.current_task(), []
    receipt = tmp_path / 'closed-original-native-source'
    original_park = lock.park_native_wait
    stream = foreign = None

    async def source():
        try:
            yield b'original'
        finally:
            receipt.write_text('original source physically closed')

    async def foreign_close():
        entered.set()
        await stream.aclose()

    @asynccontextmanager
    async def park(targets):
        async with original_park(targets):
            if (stream is not None and stream.retirement_task in targets
                    and asyncio.current_task() is writer):
                assert lock._writer is not writer, (
                    'Original writer observer joined a foreign closer without lending its native producer writer')
            yield

    async with lock.policy_span():
        async with lock:
            factory = lock.capture_native_work_factory()
            stream = await reserve_execution_stream(None, source(), lifecycle=ExecutionStreamLifecycle(
                lambda body, context: factory(body, context=context),
                lambda iterator, task: source_owners.append((iterator, task)), lambda: None))
            assert await anext(stream) == b'original'
            assert lock._writer is writer
            monkeypatch.setattr(lock, 'park_native_wait', park)
            foreign = asyncio.create_task(foreign_close(), context=Context())
            try:
                await entered.wait()
                assert stream._close_operation is not None
                await stream.aclose()
                assert receipt.read_text() == 'original source physically closed'
                assert lock._writer is writer
            finally:
                monkeypatch.setattr(lock, 'park_native_wait', original_park)
                # Even a failed observation explicitly joins the SAME admitted
                # source while lending its original writer, so RED cannot hang.
                async with owned_native_wait(tuple(task for _, task in source_owners)):
                    await stream.aclose()
                await await_owned_task(foreign)
    assert all(task.done() and iterator.owner_task.done() for iterator, task in source_owners)
    assert not lock.locked() and not lock._native_scopes


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['work', 'captured-backend'])
@pytest.mark.parametrize('reserved', [False, True], ids=['lazy', 'reserved'])
@pytest.mark.parametrize('revoke', [False, True], ids=['live', 'revoked'])
async def test_later_stream_custody_reaches_native_child_queued_before_that_consumer(tmp_path, reserved, revoke, boundary):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    import threading
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody, ProcessRuntimeExecutionFence, bind_execution_custody_snapshot,
        reserve_execution_stream, run_execution_blocking_operation, is_execution_control_error,
        current_execution_custody, require_execution_backend,
    )

    loop, queued = asyncio.get_running_loop(), asyncio.Event()
    entered, release = threading.Event(), threading.Event()
    witness = tmp_path / 'previously-queued-native-effect'
    children, native_operations = [], []
    class NativeQueue(ThreadPoolExecutor):
        def submit(self, function, /, *args, **kwargs):
            future = super().submit(function, *args, **kwargs)
            operation = function.args[0] if isinstance(function, partial) and function.args else None
            if operation in native_operations or getattr(operation, '__qualname__', '').endswith('run_execution_blocking_operation.<locals>.run_original_work'):
                loop.call_soon_threadsafe(queued.set)
            return future
    def occupy():
        entered.set()
        assert release.wait(10), 'Owned native queue was not released'
    async def source():
        captured = current_execution_custody()
        def dispatch():
            require_execution_backend('postgres', captured)
            return witness.write_text('original queued native effect')
        native_operations.append(dispatch)
        if boundary == 'captured-backend':
            from kestrel_sovereign._async_ownership import run_blocking_operation
            operation = run_blocking_operation(dispatch)
        else:
            operation = run_execution_blocking_operation(None,
                witness.write_text, 'original queued native effect')
        child = create_owned_task(operation, retain_originals=True)
        children.append(child)
        try:
            await queued.wait()
            yield b'first'
            yield b'second'
        finally:
            await await_owned_task(child)

    prior_executor = loop._default_executor
    executor, stream = NativeQueue(max_workers=1), None
    blocker = executor.submit(occupy)
    fence = ProcessRuntimeExecutionFence('later-consumer-of-previously-queued-child')
    later = ExecutionCustody(fence)
    try:
        assert entered.wait(2), 'Owned executor did not enter its actual worker'
        loop._default_executor = executor
        stream = (await reserve_execution_stream(None, source()) if reserved else execution_stream(None, source()))
        assert await anext(stream) == b'first'
        with bind_execution_custody_snapshot((later,)):
            assert await anext(stream) == b'second'
        if revoke:
            fence.retire()
        release.set()
        outcome = await await_owned_task(children[0])
        if revoke:
            assert is_execution_control_error(outcome.error), (
                'Previously queued native child discarded a later consumer restriction')
            assert not witness.exists(), 'Previously queued native child made an effect after later consumer retirement'
        else:
            assert outcome.error is None and witness.read_text() == 'original queued native effect'
    finally:
        release.set()
        for child in children:
            await await_owned_task(child)
        if stream is not None:
            from kestrel_sovereign._async_ownership import await_owned_operation
            await await_owned_operation(stream.aclose(), retain_originals=True)
        blocker.result()
        loop._default_executor = prior_executor
        executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('revoke', [False, True], ids=['live', 'revoked'])
async def test_native_validation_retains_consumer_scopes_added_during_original_lock_wait(tmp_path, monkeypatch, revoke):
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody, ProcessRuntimeExecutionFence, bind_execution_custody_snapshot,
        lock_execution_authority, is_execution_control_error,
    )
    from kestrel_sovereign._async_ownership import await_owned_operation

    first, later = [ProcessRuntimeExecutionFence(name) for name in ['original-validator', 'later-validator']]
    first_scope, later_scope = ExecutionCustody(first), ExecutionCustody(later)
    entered, release = asyncio.Event(), asyncio.Event()
    observed, children = [], []
    connection = object()  # A process fence requires no database rows or lease.
    witness = tmp_path / 'native-validation-completed'
    original_first, original_later = first.lock_and_validate, later.lock_and_validate
    async def first_validation(actual):
        assert actual is connection
        await original_first(actual)
        observed.append(first)
        entered.set()
        await release.wait()
    async def later_validation(actual):
        assert actual is connection
        await original_later(actual)
        observed.append(later)
    monkeypatch.setattr(first, 'lock_and_validate', first_validation)
    monkeypatch.setattr(later, 'lock_and_validate', later_validation)
    async def child():
        await lock_execution_authority(connection, 'postgres', (first_scope,))
        witness.write_text('all original and later native validators completed')
    async def source():
        task = create_owned_task(child(), retain_originals=True)
        children.append(task)
        try:
            await entered.wait()
            yield b'first'
            yield b'second'
        finally:
            await await_owned_task(task)
    with bind_execution_custody_snapshot((first_scope,)):
        stream = execution_stream(None, source())
    try:
        assert await anext(stream) == b'first'
        with bind_execution_custody_snapshot((later_scope,)):
            assert await anext(stream) == b'second'
        if revoke:
            later.retire()
        release.set()
        outcome = await await_owned_task(children[0])
        if revoke:
            assert is_execution_control_error(outcome.error) and not witness.exists()
        else:
            assert outcome.error is None
            assert observed == [first, later], 'Native lock validation omitted a later consumer scope'
            assert witness.read_text() == 'all original and later native validators completed'
    finally:
        release.set()
        for task in children:
            await await_owned_task(task)
        await await_owned_operation(stream.aclose(), retain_originals=True)


@pytest.mark.asyncio
async def test_failed_reserved_owner_keeps_a_valid_writer_for_remaining_source_cleanup(tmp_path):
    from kestrel_sovereign._async_ownership import await_owned_operation
    from kestrel_sovereign.execution_custody import reserve_execution_stream

    lock, owners = ReentrantTransitionLock(), []
    opening = OSError('original producer gate refused before native source entry')
    path = tmp_path / 'native-close-under-original-writer'
    handle = path.open('w')
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise AssertionError('Refused native source dispatched payload')
        async def aclose(self):
            # Observe the actual wait eligibility before a real lock acquire;
            # the RED exits at this boundary instead of hanging its owner.
            assert lock._writer_eligible(asyncio.current_task()), (
                'Remaining physical source cleanup had no eligible original privacy writer')
            async with lock:
                handle.write('original native close')
                handle.close()
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, '__name__', '') == 'close_unstarted':
                raise opening
            return super().add_done_callback(callback, context=context)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refused = False
    def task_factory(loop, coroutine, **options):
        nonlocal refused
        operation = coroutine.cr_frame.f_locals.get('operation') if coroutine.cr_frame else None
        native_body = (operation.cr_frame.f_locals.get('operation')
            if getattr(operation, 'cr_frame', None) is not None else None)
        if (not refused and 'ReentrantTransitionLock.start_owned_native_work' in getattr(operation, '__qualname__', '')
                and getattr(native_body, '__name__', '') == 'run_owned'):
            refused = True
            return PublicationTask(coroutine, loop=loop, **options)
        return asyncio.Task(coroutine, loop=loop, **options)
    try:
        async with lock.policy_span():
            async with lock:
                native = lock.capture_native_work_factory()
                loop.set_task_factory(task_factory)
                try:
                    stream = await reserve_execution_stream(None, Source(), lifecycle=ExecutionStreamLifecycle(
                        lambda body, context: native(body, context=context),
                        lambda iterator, task: owners.append((iterator, task)), lambda: None))
                finally:
                    loop.set_task_factory(previous)
                assert refused, 'Original native producer gate was not refused'
                settled = await await_owned_task(owners[0][1])
                assert opening in tuple(_explicit_control_chain(settled.error))
                terminal = None
                try:
                    await stream.aclose()
                except BaseException as error:
                    terminal = error
                assert opening in tuple(_explicit_control_chain(terminal))
                assert handle.closed, 'Remaining physical source cleanup could not reacquire its original writer'
                assert path.read_text() == 'original native close'
    finally:
        loop.set_task_factory(previous)
        if not handle.closed:
            handle.close()
    assert all(task.done() and iterator.owner_task.done() for iterator, task in owners)


@pytest.mark.asyncio
async def test_private_close_interrupt_of_unentered_native_source_is_not_caller_withdrawal(tmp_path):
    from kestrel_sovereign._async_ownership import await_owned_operation
    from kestrel_sovereign.execution_custody import reserve_execution_stream

    lock, owners = ReentrantTransitionLock(), []
    path = tmp_path / 'private-native-interrupt'
    handle = path.open('w')
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise AssertionError('Unpulled native source dispatched payload')
        async def aclose(self):
            handle.write('original close completed')
            handle.close()
    try:
        async with lock.policy_span():
            async with lock:
                native = lock.capture_native_work_factory()
                stream = await reserve_execution_stream(None, Source(), lifecycle=ExecutionStreamLifecycle(
                    lambda body, context: native(body, context=context),
                    lambda iterator, task: owners.append((iterator, task)), lambda: None))
                await asyncio.sleep(0)
                assert not stream._producer_started and not owners[0][1].done()
                first = await await_owned_operation(stream.aclose(), retain_originals=True)
                second = await await_owned_operation(stream.aclose(), retain_originals=True)
                assert first.error is first.cancellation is second.error is second.cancellation is None, (
                    'Private source close interruption was replayed as caller withdrawal')
                assert path.read_text() == 'original close completed'
    finally:
        if not handle.closed:
            handle.close()
    assert all(task.done() and iterator.owner_task.done() for iterator, task in owners)


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_observer', [False, True], ids=['live-observer', 'withdrawn-observer'])
async def test_close_observer_retains_native_outcome_through_original_writer_restoration(tmp_path, monkeypatch, cancel_observer):
    from contextvars import Context
    from kestrel_sovereign._async_ownership import await_owned_operation
    from kestrel_sovereign.execution_custody import reserve_execution_stream

    lock = ReentrantTransitionLock()
    writer = asyncio.current_task()
    entered, release, restoring = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = ExecutionCommitOutcomeError('unknown')
    owners, children, withdrawals = [], [], []
    witness = tmp_path / 'native-closed-before-writer-restoration'
    original_wait = lock._wait_change
    async def wait_change():
        if lock._parked and lock._parked[-1].task is writer and lock._parked[-1].restoring:
            restoring.set()
        try:
            return await original_wait()
        except asyncio.CancelledError as withdrawal:
            if asyncio.current_task() is writer:
                withdrawals.append(withdrawal)
            raise
    monkeypatch.setattr(lock, '_wait_change', wait_change)
    async def descendant():
        entered.set()
        await release.wait()
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            return b'original'
        async def aclose(self):
            child = lock.capture_native_work_factory()(descendant)
            children.append(child)
            await entered.wait()
            witness.write_text('native physical close completed')
            raise original
    async def restore_observer():
        await restoring.wait()
        if cancel_observer:
            writer.cancel('original close observer withdrew during writer restoration')
        release.set()
    observer = None
    try:
        async with lock.policy_span():
            async with lock:
                native = lock.capture_native_work_factory(acquire_span=False)
                stream = await reserve_execution_stream(None, Source(), lifecycle=ExecutionStreamLifecycle(
                    lambda body, context: native(body, context=context),
                    lambda iterator, task: owners.append((iterator, task)), lambda: None))
                assert await anext(stream) == b'original'
                observer = asyncio.create_task(restore_observer(), context=Context())
                terminal = None
                try:
                    await stream.aclose()
                except BaseException as error:
                    terminal = error
                assert original in tuple(_explicit_control_chain(terminal)), (
                    'Writer restoration discarded the collected original native close outcome')
                assert witness.read_text() == 'native physical close completed'
                if cancel_observer:
                    assert withdrawals and all(error in tuple(_explicit_control_chain(terminal)) for error in withdrawals)
                else:
                    assert not withdrawals
    finally:
        release.set()
        if observer is not None:
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
        for child in children:
            await await_owned_task(child)
    assert all(task.done() and iterator.owner_task.done() for iterator, task in owners)


@pytest.mark.asyncio
@pytest.mark.parametrize('publication', ['allocate', 'accepted-start'])
async def test_native_retirement_owner_admission_refusal_retains_original_unclosed_resource(tmp_path, publication):
    from kestrel_sovereign.execution_custody import reserve_execution_stream, execution_cleanup_debts
    from kestrel_sovereign._async_ownership import await_owned_operation

    lock, owners, bodies, accepted = ReentrantTransitionLock(), [], [], []
    refusal = OSError('original native stream retirement admission refused')
    handle = (tmp_path / 'original-unclosed-native-resource').open('w')
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            bodies.append('payload')
            return b'payload'
        async def aclose(self):
            handle.close()
    source = Source()
    class RefusedTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, '__name__', '') == 'close_unstarted':
                raise refusal
            return super().add_done_callback(callback, context=context)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **options):
        operation = coroutine.cr_frame.f_locals.get('operation') if coroutine.cr_frame else None
        native_body = (operation.cr_frame.f_locals.get('operation')
            if getattr(operation, 'cr_frame', None) is not None else None)
        if getattr(native_body, '__name__', '') == '_close_when_requested':
            if publication == 'allocate':
                raise refusal
            task = RefusedTask(coroutine, loop=loop, **options)
            accepted.append(task)
            return task
        return asyncio.Task(coroutine, loop=loop, **options)
    try:
        async with lock.policy_span():
            native = lock.capture_native_work_factory()
            def cleanup(body, context, admission, resource):
                return native(body, context=context, admission=admission,
                    retirement_resource=resource, acquire_span=False)
            lifecycle = ExecutionStreamLifecycle(lambda body, context: native(body, context=context),
                lambda iterator, task: owners.append(task), lambda: None, cleanup_factory=cleanup)
            loop.set_task_factory(factory)
            try:
                outcome = await await_owned_operation(reserve_execution_stream(None, source, lifecycle=lifecycle), retain_originals=True)
            finally:
                loop.set_task_factory(previous)
            assert refusal in tuple(_explicit_control_chain(outcome.error))
            debts = execution_cleanup_debts(outcome.error)
            assert any(getattr(resource, '_source', None) is source for resource, task in debts), (
                'Failed retirement admission discarded its original acquired source')
            assert not handle.closed and not owners and not bodies
            assert len(accepted) == (publication == 'accepted-start')
            assert all(task.done() for task in accepted)
    finally:
        loop.set_task_factory(previous)
        handle.close()
