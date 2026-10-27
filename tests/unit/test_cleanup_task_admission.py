"""Original native resources survive refusal of a new cleanup observer Task."""

import asyncio
import inspect

import httpx
import pytest
from tests.utils.original_failure import OriginalFailure

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ExecutionStreamLifecycle,
    NativeExecutionStreamOwner,
    ProcessRuntimeExecutionFence,
    _explicit_control_chain,
    bind_execution_custody_snapshot,
    execution_cleanup_debts,
    execution_stream,
    is_execution_control_error,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock
from kestrel_sovereign.storage.sqla import make_session_factory


def _operation_name(coroutine):
    frame = getattr(coroutine, "cr_frame", None)
    operation = (
        frame.f_locals.get("operation", coroutine) if frame is not None else coroutine
    )
    code = getattr(operation, "cr_code", None)
    return code.co_qualname if code is not None else ""


@pytest.mark.asyncio
async def test_refused_native_http_close_keeps_actual_handle_and_debt():
    peers, peer_closed = set(), {"/held": asyncio.Event(), "/other": asyncio.Event()}

    async def peer(reader, writer):
        task = asyncio.current_task()
        peers.add(task)
        path = None
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            path = header.split(b" ")[1].decode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            if path in peer_closed:
                peer_closed[path].set()
            peers.discard(task)

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    client = httpx.AsyncClient(trust_env=False)
    inventory = NativeExecutionStreamOwner()
    loop, refused = asyncio.get_running_loop(), []
    factory = loop.get_task_factory()
    refusal = RuntimeError("original native close Task refused")
    source = OriginalFailure.capture(refusal)

    def refuse_once(loop, coroutine, **kwargs):
        if not refused and _operation_name(coroutine).endswith(".owned_close"):
            refused.append(coroutine)
            raise refusal
        if factory is not None:
            return factory(loop, coroutine, **kwargs)
        return asyncio.Task(coroutine, loop=loop, **kwargs)

    try:
        responses = []
        for path in ("/other", "/held"):
            response = await client.send(
                client.build_request(
                    "GET",
                    f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}{path}",
                ),
                stream=True,
            )
            responses.append(response)
            inventory.retain(response)
        other, held = responses
        loop.set_task_factory(refuse_once)
        with pytest.raises(BaseException) as caught:
            await inventory.close()
        loop.set_task_factory(factory)
        source.assert_retained(caught.value)
        assert refused and any(resource is held and task is None
                               for resource, task in execution_cleanup_debts(caught.value))
        assert inventory.handles == [held], (
            "unclosed actual response disappeared from native inventory"
        )
        assert is_execution_control_error(caught.value)
        assert (held, None) in execution_cleanup_debts(caught.value)
        assert not held.is_closed and not peer_closed["/held"].is_set()
        await asyncio.wait_for(peer_closed["/other"].wait(), 2)
        assert other.is_closed, "independent native handle was not physically retired"
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused
        )
        with pytest.raises(BaseException) as repeated:
            await inventory.close()
        source.assert_retained(repeated.value)
        assert repeated.value is caught.value and len(refused) == 1
    finally:
        loop.set_task_factory(factory)
        for coroutine in refused:
            coroutine.close()
        await client.aclose()
        server.close()
        await server.wait_closed()
        for task in tuple(peers):
            await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["registration", "watcher"])
async def test_failed_stream_setup_joins_dispatch_gated_original_producer(
    tmp_path, failure
):
    lock, producers, refused = ReentrantTransitionLock(), [], []
    path = tmp_path / "must-not-dispatch.db"
    refusal = RuntimeError("original stream " + failure + " refused")
    loop, factory = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )

    class Host:
        def _get_privacy_transition_lock(self):
            return lock

    async def source():
        # Native file creation would prove a wrongly admitted body. No fake
        # successful source or replacement generator is supplied after failure.
        db = await AsyncDatabase.sqlite(str(path))
        try:
            yield b"forbidden"
        finally:
            await db.close()

    def started(iterator, owner):
        producers.append(owner)
        if failure == "registration":
            raise refusal

    def refuse_watcher(loop, coroutine, **kwargs):
        if _operation_name(coroutine).endswith("._watch_owner_cancellation"):
            refused.append(coroutine)
            raise refusal
        if factory is not None:
            return factory(loop, coroutine, **kwargs)
        return asyncio.Task(coroutine, loop=loop, **kwargs)

    lifecycle = ExecutionStreamLifecycle(
        lock.capture_native_work_factory(), started, lambda: None
    )
    stream = execution_stream(Host(), source(), lifecycle=lifecycle)
    try:
        if failure == "watcher":
            loop.set_task_factory(refuse_watcher)
        with pytest.raises(RuntimeError) as caught:
            await anext(stream)
        loop.set_task_factory(factory)
        if failure == "watcher":
            assert refusal in _explicit_control_chain(caught.value)
            assert execution_cleanup_debts(caught.value), (
                "opaque watcher factory acceptance cannot be certified absent"
            )
        else:
            assert caught.value is refusal
        assert producers
        _, pending = await asyncio.wait(producers, timeout=0.1)
        assert not pending, (
            "original producer stranded at the constructor dispatch barrier"
        )
        assert not path.exists() and not lock.locked()
        if failure != "watcher":
            assert not execution_cleanup_debts(caught.value), (
                "joined constructor reservation was not reconciled"
            )
        else:
            assert all(task not in producers for _, task in execution_cleanup_debts(caught.value)), (
                "joined producer was incorrectly relabelled as watcher debt"
            )
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused
        )
    finally:
        loop.set_task_factory(factory)
        for owner in producers:
            if not owner.done():
                owner.cancel()
            await await_owned_task(owner)
        for coroutine in refused:
            coroutine.close()
        await stream.aclose()


@pytest.mark.asyncio
async def test_refused_optional_disposal_still_physically_closes_primary_sqlite(
    tmp_path,
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "primary-close.db"))
    connection = db._backend._connection
    worker = getattr(connection, "_thread", connection)
    loop, factory, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    refusal = RuntimeError("optional engine disposal Task refused")
    source = OriginalFailure.capture(refusal)

    def refuse_disposal(loop, coroutine, **kwargs):
        if _operation_name(coroutine) == "AsyncDatabase.dispose_cached_sqla_factory":
            refused.append(coroutine)
            raise refusal
        if factory is not None:
            return factory(loop, coroutine, **kwargs)
        return asyncio.Task(coroutine, loop=loop, **kwargs)

    try:
        loop.set_task_factory(refuse_disposal)
        with pytest.raises(BaseException) as caught:
            await db.close()
        loop.set_task_factory(factory)
        source.assert_retained(caught.value)
        assert refused and any(task is None for _, task in execution_cleanup_debts(caught.value))
        assert db._backend._connection is None, (
            "optional disposal allocation skipped the actual primary close"
        )
        assert not worker.is_alive(), (
            "original aiosqlite worker still live after database close"
        )
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused
        )
    finally:
        loop.set_task_factory(factory)
        for coroutine in refused:
            coroutine.close()
        await db.close()


@pytest.mark.asyncio
async def test_concurrent_feature_shutdown_rechecks_original_join_graph(
    tmp_path, monkeypatch
):
    from kestrel_sovereign import _async_ownership as ownership

    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    feature = _native_owned_feature(agent)
    await db.execute("CREATE TABLE concurrent_feature_effects (id INTEGER)")
    ready, release, settling = asyncio.Event(), asyncio.Event(), asyncio.Event()
    entered, originals, attempts = [], [], []
    original_join = ownership.await_owned_task

    async def guard_actual_closing_join(target, *args, **kwargs):
        if target in ownership.owned_task_join_ancestors():
            attempts.append((asyncio.current_task(), target))
            raise RuntimeError(
                "Core feature shutdown closed an actual concurrent join cycle"
            )
        return await original_join(target, *args, **kwargs)

    async def original(index):
        await db.execute("INSERT INTO concurrent_feature_effects VALUES (?)", (index,))
        entered.append(index)
        if len(entered) == 3:
            ready.set()
        if index == 0:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                settling.set()
                await original_join(create_owned_task(release.wait()))
            return
        try:
            await ready.wait()
        except asyncio.CancelledError:
            pass
        await feature.shutdown()

    monkeypatch.setattr(ownership, "await_owned_task", guard_actual_closing_join)
    try:
        for index in range(3):
            originals.append(
                feature._track_owned_background_task(
                    original(index), name=f"native-concurrent-owner:{index}"
                )
            )
        await asyncio.wait_for(ready.wait(), 5)
        await asyncio.wait_for(settling.wait(), 5)
        async with asyncio.timeout(5):
            while len(getattr(originals[0], "_kestrel_owned_join_waiters", ())) != 2:
                await asyncio.sleep(0.01)
        release.set()
        _, pending = await asyncio.wait(originals, timeout=5)
        assert not pending, (
            "original feature Tasks formed a permanent concurrent drain cycle"
        )
        assert not attempts, (
            "an entry-time ancestor snapshot became stale after a native join"
        )
        assert await db.fetchval("SELECT COUNT(*) FROM concurrent_feature_effects") == 3
        assert not feature._owned_background_tasks
        assert not agent._runtime_publication_ready.is_set()
    finally:
        release.set()
        for task in originals:
            if not task.done():
                task.cancel()
            await original_join(task)
        await agent.shutdown()
        await db.close()


def _explicit_errors(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.append(current.__cause__)
        pending.extend(getattr(current, "exceptions", ()))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,cleanup_kind", [("write", "rollback"), ("write", "close"), ("read", "close")]
)
async def test_native_sqla_cleanup_cannot_replace_actual_custody_denial(
    tmp_path, monkeypatch, mode, cleanup_kind
):
    from sqlalchemy import text

    db = await AsyncDatabase.sqlite(str(tmp_path / "session-cleanup.db"))
    await db._backend.execute("CREATE TABLE cleanup_effects (id INTEGER PRIMARY KEY)")
    factory = make_session_factory(db)
    scope = ExecutionCustody(ProcessRuntimeExecutionFence("original session admission"))
    cleanup_error = OSError(
        "original native " + cleanup_kind + " acknowledgement failed"
    )
    denials, cleanup_calls = [], []
    try:
        with pytest.raises(BaseException) as caught:
            context = (
                factory.write_session() if mode == "write" else factory.read_session()
            )
            async with context as session:
                await session.execute(text("INSERT INTO cleanup_effects VALUES (1)"))
                original = getattr(session, cleanup_kind)

                async def native_cleanup_then_failure():
                    await original()
                    cleanup_calls.append(True)
                    raise cleanup_error

                monkeypatch.setattr(session, cleanup_kind, native_cleanup_then_failure)
                with bind_execution_custody_snapshot((scope,)):
                    try:
                        await session.execute(
                            text("INSERT INTO cleanup_effects VALUES (2)")
                        )
                    except BaseException as error:
                        denials.append(error)
                        raise
        assert len(denials) == 1 and is_execution_control_error(denials[0])
        assert is_execution_control_error(caught.value), (
            "native cleanup erased the actual SQLAlchemy custody refusal"
        )
        assert denials[0] in _explicit_errors(caught.value)
        assert cleanup_error in _explicit_errors(caught.value) and cleanup_calls == [
            True
        ]
        assert await db._backend.fetch_val("SELECT COUNT(*) FROM cleanup_effects") == 0
        assert not factory._active_sessions
    finally:
        # Session fault injection has finished its actual native rollback/close;
        # independent engine + primary retirement is still physically joined.
        await db.close()


@pytest.mark.asyncio
async def test_cancelled_batch_joins_original_unpublished_resources_without_collector_task(
    tmp_path, monkeypatch
):
    await assert_cancelled_batch_native_cleanup(
        tmp_path, monkeypatch, cancel_again=False
    )


async def _partial_native_agent(tmp_path, monkeypatch):
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm import service as service_module
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    db = await AsyncDatabase.sqlite(str(tmp_path / "unpublished-native.db"))
    # Real provider service with no configured routes and the original native
    # usage database. No provider calls, fake canary, genesis pass, or readiness
    # bypass: this is private PARTIAL initialization resource cleanup only.
    monkeypatch.setattr(service_module, "load_section", lambda _name: {})
    service = service_module.LLMService(usage_db=db, agent_data_dir=tmp_path)
    agent = KestrelAgent(
        did="did:test:unpublished-cleanup",
        storage_path=str(tmp_path / "agent.db"),
        llm_service=service,
        allowed_features=[],
    )
    storage = AsyncStorage(backend=db.backend)
    storage.db = db
    agent.storage = agent._raw_storage = storage
    assert not agent._runtime_publication_ready.is_set()
    return agent, db


async def assert_cancelled_batch_native_cleanup(tmp_path, monkeypatch, *, cancel_again):
    from kestrel_sovereign.multi_agent import agent_manager as manager_module
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig

    actual_join, withdrawals = manager_module.await_owned_task, []
    async def observed_join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(manager_module, "await_owned_task", observed_join)

    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    connection = db._backend._connection
    worker = getattr(connection, "_thread", connection)
    close_entered, close_release = asyncio.Event(), asyncio.Event()
    if cancel_again:
        native_close = connection.close

        async def held_original_close_ack():
            await native_close()
            close_entered.set()
            await close_release.wait()

        monkeypatch.setattr(connection, "close", held_original_close_ack)
    manager = manager_module.AgentManager(base_data_dir=tmp_path)
    beta_entered, alpha_done = asyncio.Event(), asyncio.Event()
    loop, factory, refused = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
        [],
    )
    refusal = RuntimeError("new batch recovery collector Task refused")

    async def original_partial_initialize(name, *_args, **_kwargs):
        if name == "Alpha":
            manager._initializing_agents[name] = agent
        await manager_module._BATCH_STAGE_CALLBACK.get()(
            name, agent if name == "Alpha" else None
        )
        if name == "Alpha":
            return agent
        beta_entered.set()
        await asyncio.get_running_loop().create_future()

    def refuse_collector(loop, coroutine, **kwargs):
        name = _operation_name(coroutine)
        if name.endswith(".collect_initializer_results"):
            refused.append(coroutine)
            raise refusal
        task = (
            factory(loop, coroutine, **kwargs)
            if factory is not None
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )
        # Actual completion of Alpha's original initializer, not a fabricated
        # successful aggregate or a replacement batch observer.
        frame = getattr(coroutine, "cr_frame", None)
        operation = frame.f_locals.get("operation", coroutine) if frame is not None else coroutine
        frame = getattr(operation, "cr_frame", None)
        if (
            frame is not None
            and frame.f_locals.get("name") == "Alpha"
            and name.endswith("._bounded_initialize")
        ):
            task.add_done_callback(lambda _: alpha_done.set())
        return task

    monkeypatch.setattr(manager, "_initialize_agent", original_partial_initialize)
    config = MultiAgentConfig(
        agents={
            name: LocalAgentConfig(data_dir=name.lower(), port=18901 + index)
            for index, name in enumerate(("Alpha", "Beta"))
        }
    )
    batch = None
    try:
        loop.set_task_factory(refuse_collector)
        batch = create_owned_task(manager.load_from_config(config), retain_originals=True)
        await asyncio.wait_for(beta_entered.wait(), 3)
        await asyncio.wait_for(alpha_done.wait(), 3)
        batch.cancel()
        if cancel_again:
            await asyncio.wait_for(close_entered.wait(), 5)
            batch.cancel()
            await asyncio.sleep(0)
            assert not batch.done(), (
                "second cancellation abandoned the accepted native close ACK"
            )
            close_release.set()
        outcome = await await_owned_task(batch)
        loop.set_task_factory(factory)
        assert db.backend._connection is None, (
            "recovery collector refusal abandoned the original unpublished database"
        )
        assert not worker.is_alive()
        assert not manager._initializing_agents and not manager.list_agents()
        assert not manager._agent_operations
        assert withdrawals and all(error in tuple(_explicit_control_chain(outcome.error))
                                   for error in withdrawals)
        assert not agent._runtime_publication_ready.is_set()
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused
        )
    finally:
        loop.set_task_factory(factory)
        close_release.set()
        if batch is not None:
            await await_owned_task(batch)
        for coroutine in refused:
            coroutine.close()
        await agent.shutdown()
        await db.close()


def _native_owned_feature(agent):
    from kestrel_sovereign.features.base import Feature

    class NativeFeature(Feature):
        async def initialize(self):
            return None

        @property
        def tool_description(self):
            return "native reentrant resource ownership"

    return NativeFeature(agent)


@pytest.mark.asyncio
async def test_reentrant_feature_task_keeps_its_actual_root_until_completion(
    tmp_path, monkeypatch
):
    from kestrel_sovereign import _async_ownership as ownership

    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    feature = _native_owned_feature(agent)
    original_join, self_joins = ownership.await_owned_task, []

    async def reject_actual_self_join(task, *args, **kwargs):
        if task is asyncio.current_task():
            self_joins.append(task)
            raise RuntimeError("actual Core feature owner attempted to join itself")
        return await original_join(task, *args, **kwargs)

    async def original_worker():
        await db.execute("CREATE TABLE original_feature_effects (id INTEGER)")
        await db.execute("INSERT INTO original_feature_effects VALUES (1)")
        await feature.shutdown()
        assert asyncio.current_task() in feature._owned_background_tasks
        return await db.fetchval("SELECT COUNT(*) FROM original_feature_effects")

    monkeypatch.setattr(ownership, "await_owned_task", reject_actual_self_join)
    task = feature._track_owned_background_task(
        original_worker(), name="actual-feature-self-shutdown"
    )
    try:
        outcome = await original_join(task)
        assert not self_joins, (
            "the original feature Task reached an impossible self-join"
        )
        assert outcome.error is None and outcome.result == 1
        assert (
            not feature._owned_background_tasks and task not in agent._background_tasks
        )
        assert not agent._runtime_publication_ready.is_set()
    finally:
        await original_join(task)
        await agent.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_actual_resident_publisher_closes_both_coroutines_on_task_refusal(
    tmp_path, monkeypatch
):
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    refused, effects = [], []
    failure = RuntimeError("actual resident publication Task allocation refused")

    async def original_resident():
        effects.append(True)
        await db.execute("CREATE TABLE resident_must_not_enter (id INTEGER)")

    operation = original_resident()

    def factory(loop, coroutine, **kwargs):
        frame = getattr(coroutine, "cr_frame", None)
        inner = frame.f_locals.get("operation") if frame is not None else None
        candidate = inner if inspect.iscoroutine(inner) else coroutine
        code = getattr(candidate, "cr_code", None)
        if (
            code is not None
            and code.co_qualname == "KestrelAgent._track_runtime_task.<locals>.resident"
        ):
            refused.append((coroutine, candidate))
            raise failure
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        loop.set_task_factory(factory)
        with pytest.raises(RuntimeError) as propagated:
            agent._track_runtime_task(
                operation, name="native-original-resident-refused"
            )
        assert failure in _explicit_control_chain(propagated.value) and len(refused) == 1
        assert execution_cleanup_debts(propagated.value), (
            "opaque resident factory acceptance must remain explicit debt"
        )
        assert inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED, (
            "original unadmitted resident coroutine leaked"
        )
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        ), "original publication wrapper leaked"
        assert not effects and not agent._background_tasks
        assert not agent._runtime_publication_ready.is_set(), (
            "allocation proof must not forge READY"
        )
        assert not await db.fetchone(
            "SELECT name FROM sqlite_master WHERE name='resident_must_not_enter'"
        )
    finally:
        loop.set_task_factory(previous)
        operation.close()
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await agent.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_feature_child_shutdown_keeps_actual_waiting_parent_root(
    tmp_path, monkeypatch
):
    from kestrel_sovereign import _async_ownership as ownership

    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    feature = _native_owned_feature(agent)
    original_join, attempts = ownership.await_owned_task, []
    parent, child = None, None

    async def guard_parent_join(target, *args, **kwargs):
        if target is parent and asyncio.current_task() is child:
            attempts.append(target)
            raise RuntimeError(
                "actual feature descendant tried to join its waiting parent"
            )
        return await original_join(target, *args, **kwargs)

    async def original_child():
        await db.execute("CREATE TABLE descendant_native_effects (id INTEGER)")
        await db.execute("INSERT INTO descendant_native_effects VALUES (1)")
        await feature.shutdown()
        assert parent in feature._owned_background_tasks
        return await db.fetchval("SELECT COUNT(*) FROM descendant_native_effects")

    async def original_parent():
        nonlocal child
        child = create_owned_task(original_child())
        outcome = await original_join(child)
        if outcome.error is not None:
            raise outcome.error
        return outcome.result

    monkeypatch.setattr(ownership, "await_owned_task", guard_parent_join)
    parent = feature._track_owned_background_task(
        original_parent(), name="actual-native-feature-parent"
    )
    try:
        outcome = await original_join(parent)
        assert not attempts
        assert outcome.error is None and outcome.result == 1
        assert parent.done() and child.done()
        assert (
            not feature._owned_background_tasks
            and parent not in agent._background_tasks
        )
        assert not agent._runtime_publication_ready.is_set()
    finally:
        await original_join(parent)
        if child is not None:
            await original_join(child)
        await agent.shutdown()
        await db.close()
