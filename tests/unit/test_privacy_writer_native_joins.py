"""Exclusive privacy writers across original native task joins."""
import asyncio

import pytest
from tests.unit.test_tool_effect_bound_operation import owner, bound_operation

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, gather_owned, owned_task_error
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock, bind_transition_lock_reentry


@pytest.mark.asyncio
async def test_arbitrary_event_wait_does_not_lend_writer_to_accepted_child(tmp_path):
    lock = ReentrantTransitionLock()
    attempted, release = asyncio.Event(), asyncio.Event()
    path = tmp_path / 'effects'

    async def child():
        attempted.set()
        async with lock:
            path.write_text('child')

    async with lock:
        task = create_owned_task(child())
        await attempted.wait()
        await asyncio.sleep(0)
        assert not path.exists()
        release.set()
    assert task.done() and path.read_text() == 'child'


@pytest.mark.asyncio
async def test_native_join_restores_parent_before_unrelated_token_writer(tmp_path):
    lock = ReentrantTransitionLock()
    attempted = asyncio.Event()
    order = []

    async def unrelated(token):
        with bind_transition_lock_reentry(token):
            attempted.set()
            async with lock:
                order.append('unrelated')

    async def child():
        async with lock:
            order.append('child')
            await asyncio.sleep(0)

    async with lock.policy_span():
        async with lock:
            stranger = asyncio.create_task(unrelated(lock.current_reentry_token()))
            await attempted.wait()
            task = create_owned_task(child())
            outcome = await await_owned_task(task)
            assert outcome.error is None
            order.append('parent')
            assert order == ['child', 'parent']
        await stranger
    assert order == ['child', 'parent', 'unrelated']


@pytest.mark.asyncio
async def test_nested_join_and_gather_serialize_original_writers(tmp_path):
    lock = ReentrantTransitionLock()
    path = tmp_path / 'effects'
    inside = maximum = 0

    async def leaf(index):
        nonlocal inside, maximum
        async with lock:
            inside += 1
            maximum = max(maximum, inside)
            await asyncio.sleep(0)
            with path.open('a') as stream:
                stream.write(str(index))
            inside -= 1

    async def branch(index):
        async with lock:
            child = create_owned_task(leaf(index))
            outcome = await await_owned_task(child)
            assert outcome.error is None

    async with lock:
        await gather_owned([branch(1), branch(2)])
    assert maximum == 1 and sorted(path.read_text()) == ['1', '2']


@pytest.mark.asyncio
async def test_policy_span_drains_accepted_writer_after_parent_returns():
    lock = ReentrantTransitionLock()
    entered, release, returned, transition = (asyncio.Event() for _ in range(4))

    async def child():
        async with lock:
            entered.set()
            await release.wait()

    async def turn():
        async with lock.policy_span():
            factory = lock.capture_native_work_factory(acquire_span=False)
            factory(child)
            returned.set()

    owner = asyncio.create_task(turn())
    await returned.wait()
    await entered.wait()

    async def change():
        async with lock:
            transition.set()

    changer = asyncio.create_task(change())
    await asyncio.sleep(0)
    assert not owner.done() and not transition.is_set()
    release.set()
    await owner
    await changer
    assert transition.is_set()


@pytest.mark.asyncio
async def test_closed_span_refuses_captured_factory_and_token():
    lock = ReentrantTransitionLock()
    async with lock.policy_span():
        token = lock.current_reentry_token()
        factory = lock.capture_native_work_factory()
    with pytest.raises(RuntimeError, match='closed privacy span'):
        factory(lambda: asyncio.sleep(0))
    with bind_transition_lock_reentry(token):
        with pytest.raises(RuntimeError, match='closed or stale'):
            async with lock:
                pytest.fail('stale span entered')
    assert not lock.locked()


@pytest.mark.asyncio
async def test_stream_handshake_restores_parent_and_producer_before_continuation(tmp_path):
    from kestrel_sovereign._async_ownership import OwnedAsyncIterator
    lock = ReentrantTransitionLock()
    path = tmp_path / 'stream-effects'

    def append(value):
        assert lock._writer is asyncio.current_task()
        with path.open('a') as output:
            output.write(value)

    async def source():
        async with lock:
            append('A')
            yield 'first'
            append('B')

    async with lock:
        stream = OwnedAsyncIterator(source, operation='privacy-handshake')
        assert await anext(stream) == 'first'
        append('P')
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        await stream.aclose()
    assert path.read_text() == 'APB'


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_writer_restoration():
    lock = ReentrantTransitionLock()
    entered, release = asyncio.Event(), asyncio.Event()
    order = []

    async def child():
        async with lock:
            entered.set()
            await release.wait()
            order.append('child-finished')

    async def parent():
        async with lock:
            task = create_owned_task(child())
            try:
                await await_owned_task(task, defer_cancellation=False)
            finally:
                assert lock._writer is asyncio.current_task()
                order.append('parent-finally')

    owner = asyncio.create_task(parent())
    await entered.wait()
    owner.cancel('first')
    await asyncio.sleep(0)
    owner.cancel('second')
    await asyncio.sleep(0)
    assert order == [] and not owner.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert order == ['child-finished', 'parent-finally']
    assert not lock.locked()


@pytest.mark.asyncio
async def test_foreign_reader_accepts_only_original_callback_scope():
    import contextvars
    from kestrel_sovereign._async_ownership import OwnedCallbackScope
    lock = ReentrantTransitionLock()
    accepted, callback_entered = asyncio.Event(), asyncio.Event()
    tasks = []

    async def effect():
        async with lock:
            callback_entered.set()
            await asyncio.sleep(0)

    async def reader(scope):
        tasks.append(scope.start(effect))
        accepted.set()

    async with lock:
        scope = OwnedCallbackScope()
        reader_task = asyncio.create_task(reader(scope), context=contextvars.Context())
        async with scope.wait():
            await accepted.wait()
            await callback_entered.wait()
            outcome = await await_owned_task(tasks[0])
            assert outcome.error is None
        assert lock._writer is asyncio.current_task()
        scope.close()
        with pytest.raises(RuntimeError, match='admission is closed'):
            scope.start(effect)
        await reader_task
    assert not lock.locked()


@pytest.mark.asyncio
async def test_first_completed_gather_cancels_sibling_before_parent_restoration():
    lock = ReentrantTransitionLock()
    entered, release, retired = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow():
        async with lock:
            entered.set()
            try:
                await release.wait()
            finally:
                retired.set()

    async def quick():
        await entered.wait()
        return 'first completion'

    async def parent():
        async with lock:
            return await gather_owned([slow(), quick()], return_when=asyncio.FIRST_COMPLETED)

    calling = asyncio.create_task(parent())
    try:
        await entered.wait()
        done, _ = await asyncio.wait({calling}, timeout=.2)
        assert calling in done, 'gather restored its parent before cancelling the protected sibling'
        assert retired.is_set()
        assert await calling == [None, 'first completion']
    finally:
        release.set()
        await asyncio.gather(calling, return_exceptions=True)


@pytest.mark.asyncio
async def test_actual_codex_reader_dispatch_uses_native_callback_receipt(bound_operation, tmp_path):
    import contextvars
    import json
    import sys
    from kestrel_sovereign._async_ownership import OwnedCallbackScope
    from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient

    owner, binding, witness = bound_operation
    lock = owner._get_privacy_transition_lock()
    response_path = tmp_path / 'native-jsonrpc-response'
    script = (
        'import json,pathlib,sys\n'
        'print(json.dumps({"id":41,"method":"item/tool/call","params":{"threadId":"owned-reader"}}),flush=True)\n'
        'pathlib.Path(sys.argv[1]).write_text(sys.stdin.readline())\n'
    )
    process = await asyncio.create_subprocess_exec(sys.executable, '-u', '-c', script,
        str(response_path), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    client = CodexAppServerClient(binary='unused-owned-stdio-fixture')
    client._proc = process
    reader = None
    try:
        async with lock:
            scope = OwnedCallbackScope()

            async def handler(params):
                return await binding.invoke({'marker': 'native-reader'},
                    session_id='owned-reader', operation_id='native-reader-intent')

            handler._codex_native_callback_scope = scope
            client.register_server_request_handler('item/tool/call', handler, thread_id='owned-reader')

            async def foreign_reader():
                message = json.loads(await process.stdout.readline())
                client._dispatch(message)
                for task in tuple(client._server_request_tasks):
                    assert task in scope.tasks
                    outcome = await await_owned_task(task)
                    assert outcome.error is None

            reader = asyncio.create_task(foreign_reader(), context=contextvars.Context())
            async with scope.wait():
                await reader
            assert lock._writer is asyncio.current_task()
            scope.close()
        assert await process.wait() == 0
        assert json.loads(response_path.read_text())['result']['success'] is True
        assert witness.read_text() == 'native-reader\n'
        assert await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()
        if reader is not None and not reader.done():
            reader.cancel()
        if reader is not None:
            await asyncio.gather(reader, return_exceptions=True)
        for task in tuple(client._server_request_tasks):
            if not task.done():
                task.cancel()
            await await_owned_task(task)


@pytest.mark.asyncio
async def test_closing_span_retains_accepted_continuation_but_closes_root_factory():
    lock = ReentrantTransitionLock()
    returned, advance = asyncio.Event(), asyncio.Event()
    executed = []
    factories = []

    async def leaf():
        async with lock:
            executed.append('original-child')

    async def accepted():
        await advance.wait()
        continuation = lock.capture_native_work_factory()
        child = continuation(leaf)
        outcome = await await_owned_task(child)
        assert outcome.error is None

    async def root():
        async with lock.policy_span():
            factories.append(lock.capture_native_work_factory())
            child = create_owned_task(accepted())
            returned.set()
        assert child.done() and owned_task_error(child) is None

    calling = asyncio.create_task(root())
    try:
        await returned.wait()
        with pytest.raises(RuntimeError, match='closed privacy span'):
            factories[0](leaf)
        advance.set()
        await calling
        assert executed == ['original-child']
    finally:
        advance.set()
        await asyncio.gather(calling, return_exceptions=True)


@pytest.mark.asyncio
async def test_copied_callback_inventory_cannot_invent_native_admission(tmp_path):
    from kestrel_sovereign._async_ownership import OwnedCallbackScope
    lock = ReentrantTransitionLock()
    attempted = asyncio.Event()
    witness = tmp_path / 'unadmitted-writer'

    async def stranger(token):
        with bind_transition_lock_reentry(token):
            attempted.set()
            async with lock:
                witness.write_text('queued writer')

    async with lock.policy_span():
        async with lock:
            scope = OwnedCallbackScope()
            task = asyncio.create_task(stranger(lock.current_reentry_token()))
            await attempted.wait()
            scope.tasks.add(task)
            async with scope.wait():
                await asyncio.sleep(0)
            assert not witness.exists(), 'callback inventory lent a writer without original native admission'
            scope.close()
        await task
    assert witness.read_text() == 'queued writer'


@pytest.mark.asyncio
async def test_resident_health_interval_handoff_does_not_extend_invocation_scope(owner):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.features.health.feature import HealthFeature
    from tests.unit.test_tool_effect_invocation import governed
    from tests.utils.native_a2a import manager_for
    from kestrel_sdk.native_a2a.types import TaskState

    feature = HealthFeature(owner)
    await feature.initialize()
    owner.features[feature.name] = feature
    original_resident = feature._background_task
    assert not owner._runtime_publication_ready.is_set()
    manager = manager_for(owner.tool_effect_store.backend,
        NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()
    manager.register_provider(feature.name, feature.get_agent_skills(), feature)

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        return await manager.execute_skill(feature.name, 'health_interval',
            {'seconds': 90}, sync=True)

    calling = asyncio.create_task(invoke(owner, 'Change resident interval',
        invocation_id='native-resident-interval'))
    try:
        done, _ = await asyncio.wait({calling}, timeout=2)
        assert calling in done, 'invocation adopted the replacement resident as an effect child'
        result = await calling
        assert result.status.state == TaskState.TASK_STATE_COMPLETED
        replacement = feature._background_task
        assert replacement is not original_resident and not replacement.done()
        assert replacement in owner._background_tasks and replacement._kestrel_runtime_service
        assert not owner._runtime_publication_ready.is_set()
        assert not owner._get_privacy_transition_lock().locked()
        assert await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
    finally:
        await feature._stop_background_loop()
        await asyncio.gather(calling, return_exceptions=True)
        await feature.shutdown()
        await manager.close()


@pytest.mark.asyncio
async def test_callback_admission_precedes_first_policy_activation(bound_operation):
    import contextvars
    from kestrel_sovereign._async_ownership import OwnedCallbackScope
    from tests.unit.test_tool_effect_invocation import governed

    owner, binding, witness = bound_operation
    lock = owner._get_privacy_transition_lock()

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        assert lock.current_native_scope() is not None and not lock.locked()
        scope = OwnedCallbackScope()

        async def foreign_reader():
            task = scope.start(lambda: binding.invoke({'marker': 'lazy-policy-callback'},
                session_id='lazy-policy-callback', operation_id='lazy-policy-callback'))
            outcome = await await_owned_task(task)
            assert outcome.error is None
            return outcome.result

        reader = asyncio.create_task(foreign_reader(), context=contextvars.Context())
        try:
            async with scope.wait():
                return await reader
        finally:
            scope.close()
            for task in tuple(scope.tasks):
                await await_owned_task(task)

    result = await invoke(owner, 'First effect from the original reader',
        invocation_id='lazy-policy-callback-invocation')
    assert result['success'] is True
    assert witness.read_text() == 'lazy-policy-callback\n'
    assert not lock.locked()
    assert await owner.tool_effect_store.database.fetchall(
        'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]


@pytest.mark.asyncio
async def test_queued_token_writer_cannot_run_after_its_policy_span_retires(tmp_path):
    lock = ReentrantTransitionLock()
    attempted = asyncio.Event()
    witness = tmp_path / 'retired-policy-effect'

    async def queued(token):
        with bind_transition_lock_reentry(token):
            attempted.set()
            async with lock:
                witness.write_text('invalid retired-span write')

    async with lock:
        task = asyncio.create_task(queued(lock.current_reentry_token()))
        await attempted.wait()
    with pytest.raises(RuntimeError, match='retained span|closed or stale'):
        await task
    assert not witness.exists() and not lock.locked()
