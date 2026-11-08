"""Exclusive privacy writers across original native task joins."""
import asyncio
import sys

import pytest
from tests.unit.test_tool_effect_bound_operation import owner as owner, bound_operation as bound_operation

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, gather_owned, owned_task_error
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock, bind_transition_lock_reentry


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 12), reason="native eager Task requires Python 3.12")
@pytest.mark.parametrize("withdraw_cache", [False, True])
@pytest.mark.parametrize("physical_kind", [None, "ordinary", "sdk", "audit"])
@pytest.mark.parametrize("publication_kind", [None, "ordinary", "sdk", "audit"])
async def test_eager_native_completion_keeps_later_privacy_publication_failure(
    tmp_path, withdraw_cache, physical_kind, publication_kind,
):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign._async_ownership import OwnedTaskAdmission
    from kestrel_sovereign.execution_custody import (
        ToolAuditUnconfirmedError, _explicit_control_chain,
    )
    lock = ReentrantTransitionLock()
    receipt = OwnedTaskAdmission()
    effect = tmp_path / "original-eager-privacy-physical-effect"
    bodies, owners, reads, publications = [], [], [], []
    def failure(kind, phase):
        if kind is None:
            return None
        error = RuntimeError("original " + phase + " failure")
        if kind in {"sdk", "audit"}:
            error = asyncio.CancelledError("original " + phase + " cancellation")
            error.__cause__ = (ResourceOutcomeUnknown("original " + phase + " outcome uncertain")
                if kind == "sdk" else ToolAuditUnconfirmedError("original " + phase + " audit uncertain"))
        return error
    physical = failure(physical_kind, "physical")
    publication = failure(publication_kind, "publication")
    async def operation():
        bodies.append(asyncio.current_task())
    class OriginalTask(asyncio.Task):
        def exception(self):
            reads.append(self)
            return super().exception()
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "completed":
                publications.append(self)
                if publication is not None:
                    if withdraw_cache:
                        self._kestrel_owned_terminal_error = None
                    raise publication
            return super().add_done_callback(callback, context=context)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **options):
        async def outer():
            coroutine.close()
            effect.write_text("original native outer physically completed once")
            if physical is not None:
                raise physical
            return "original physical success"
        options["eager_start"] = True
        task = OriginalTask(outer(), loop=loop, **options)
        owners.append(task)
        return task
    try:
        loop.set_task_factory(factory)
        task = lock.start_owned_native_work(operation, admission=receipt)
        loop.set_task_factory(previous)
        outcome = await await_owned_task(task)
        chain = tuple(_explicit_control_chain(outcome.error))
        if publication is not None:
            assert publication in chain, "Physical completion snapshot erased later privacy publication failure"
        if physical is not None:
            assert physical in chain, "Late publication erased the single captured original physical failure"
        if physical is publication is None:
            assert outcome.error is None and outcome.result == "original physical success"
        assert owned_task_error(task) is outcome.error
        assert (await await_owned_task(task)).error is outcome.error
        await asyncio.sleep(0)
        assert owned_task_error(task) is outcome.error
        assert reads == [task], "Late admission failure re-harvested consumed original native completion"
        assert owners == publications == [task] and receipt.task is task and task.done()
        assert not bodies and not lock._native_work and not lock.locked()
        assert effect.read_text() == "original native outer physically completed once"
    finally:
        loop.set_task_factory(previous)
        for task in owners:
            await await_owned_task(task)


@pytest.mark.asyncio
async def test_foreign_lifetime_callback_inventory_cannot_lend_this_writer(tmp_path):
    import contextvars
    from kestrel_sovereign._async_ownership import OwnedCallbackScope, bind_owned_coordinator
    lock = ReentrantTransitionLock()
    registered, retire = asyncio.Event(), asyncio.Event()
    scopes = []
    witness = tmp_path / 'different-lifetime-callback'

    async def other_lifetime():
        lifetime, owned = lock.begin_native_scope(None)
        assert owned
        try:
            with bind_owned_coordinator(lock):
                scopes.append(OwnedCallbackScope())
                registered.set()
                await retire.wait()
        finally:
            await lock.close_native_scope(lifetime)

    async def effect():
        witness.write_text('original admitted child')

    other = asyncio.create_task(other_lifetime(), context=contextvars.Context())
    try:
        async with lock:
            await registered.wait()
            child = lock.start_owned_native_work(effect)
            scopes[0].tasks.add(child)  # Corrupt data cannot change its original lifetime.
            async with scopes[0].wait():
                await asyncio.sleep(0)
            assert not witness.exists(), 'callback scope from another native lifetime lent the current writer'
        assert witness.read_text() == 'original admitted child'
        assert (await await_owned_task(child)).error is None
    finally:
        retire.set()
        await other
    assert not lock.locked() and not lock._native_scopes


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


@pytest.mark.asyncio
@pytest.mark.parametrize('dispatch', ['same-task', 'child'])
async def test_token_writer_does_not_gain_original_effect_admission(owner, tmp_path, dispatch):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.tools.effect_journal import bind_tool_effect_turn, execute_tool_effect
    from tests.unit.test_tool_effect_execution import admit

    lock = owner._get_privacy_transition_lock()
    turn = await admit(owner, key='token-cannot-admit-effect')
    witness = tmp_path / 'token-forged-effect'

    async def effect():
        witness.write_text('effect without original native acceptance')

    async def copied(token):
        with bind_transition_lock_reentry(token):
            async with lock:
                pass
        operation = execute_tool_effect(owner, registration='unadmitted-token-effect',
            arguments={}, execute=effect)
        if dispatch == 'child':
            return await create_owned_task(operation)
        return await operation

    try:
        with bind_tool_effect_turn((turn, None)):
            async with lock.policy_span():
                task = asyncio.create_task(copied(lock.current_reentry_token()))
                with pytest.raises(ExecutionAuthorityError, match='original native privacy lifetime'):
                    await task
        assert not witness.exists(), 'ordinary token writer acquired original effect admission'
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    finally:
        await turn.close()


@pytest.mark.asyncio
async def test_concurrent_first_activation_joins_one_original_policy_acquisition(tmp_path):
    from kestrel_sovereign._async_ownership import bind_owned_coordinator
    lock = ReentrantTransitionLock()
    held, release = asyncio.Event(), asyncio.Event()
    attempts = [asyncio.Event(), asyncio.Event()]
    witness = tmp_path / 'first-activation'

    async def other_scope():
        async with lock:
            held.set()
            await release.wait()

    other = asyncio.create_task(other_scope())
    await held.wait()
    scope, owned = lock.begin_native_scope(None)
    assert owned

    async def effect(index):
        attempts[index].set()
        async with lock:
            with witness.open('a') as output:
                output.write(str(index))

    with bind_owned_coordinator(lock):
        children = [create_owned_task(effect(index)) for index in range(2)]
    try:
        await asyncio.gather(*(event.wait() for event in attempts))
        await asyncio.sleep(0)
        release.set()
        await other
        done, _ = await asyncio.wait(children, timeout=.3)
        assert len(done) == 2, 'two original members queued separate acquisitions of one retained policy lock'
        assert all(owned_task_error(task) is None for task in children)
        assert sorted(witness.read_text()) == ['0', '1']
    finally:
        release.set()
        for task in children:
            if not task.done():
                task.cancel()
        await asyncio.gather(other, *children, return_exceptions=True)
        try:
            await lock.close_native_scope(scope, None)
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize('journaled', [False, True])
async def test_nested_native_privacy_transition_does_not_join_its_waiting_ancestor(owner, tmp_path, journaled):
    from tests.unit.test_tool_effect_invocation import governed
    lock = owner._get_privacy_transition_lock()
    witness = tmp_path / 'nested-native-transition'
    children = []

    async def child():
        async with owner.privacy_transition():
            witness.write_text('transition completed')

    async def parent():
        from kestrel_sovereign._async_ownership import owned_native_wait
        from kestrel_sovereign.tools.effect_journal import execute_tool_effect

        async def operation():
            task = lock.capture_native_work_factory()(child)
            children.append(task)
            # A bounded original native wait also provides deterministic cleanup
            # on the broken implementation: the ancestor eventually returns.
            async with owned_native_wait((task,)):
                done, _ = await asyncio.wait({task}, timeout=2)
            if task in done:
                outcome = await await_owned_task(task)
                if journaled:
                    assert isinstance(outcome.error, RuntimeError)
                    assert 'wait on itself' in str(outcome.error)
                else:
                    assert outcome.error is None

        if journaled:
            return await execute_tool_effect(owner, registration='native-transition-parent',
                arguments={}, execute=operation)
        return await operation()

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        task = lock.capture_native_work_factory(acquire_span=False)(parent)
        outcome = await await_owned_task(task)
        assert outcome.error is None

    calling = asyncio.create_task(invoke(owner, 'Native nested privacy transition',
        invocation_id='nested-native-privacy-transition'))
    try:
        done, _ = await asyncio.wait({calling}, timeout=.3)
        assert calling in done, 'privacy transition joined its native ancestor while that ancestor awaited the caller'
        await calling
        if journaled:
            assert not witness.exists()
        else:
            assert witness.read_text() == 'transition completed'
        assert not lock.locked()
    finally:
        for task in children:
            if not task.done():
                task.cancel()
        await asyncio.gather(calling, *children, return_exceptions=True)


@pytest.mark.asyncio
async def test_ordinary_token_writer_cannot_promote_itself_to_an_original_root(owner):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.tools.effect_journal import bind_tool_effect_turn
    from tests.unit.test_tool_effect_execution import admit

    lock = owner._get_privacy_transition_lock()
    turn = await admit(owner, key='ordinary-token-cannot-open-native-root')

    async def copied(token):
        with bind_transition_lock_reentry(token):
            async with lock:
                pass
        scope, owned = lock.begin_native_scope('unaccepted-token-caller')
        try:
            return scope
        finally:
            if owned:
                await lock.close_native_scope(scope, None)

    try:
        with bind_tool_effect_turn((turn, None)):
            async with lock.policy_span():
                task = asyncio.create_task(copied(lock.current_reentry_token()))
                with pytest.raises(ExecutionAuthorityError, match='original root or accepted child'):
                    await task
                assert lock._native_scopes == {turn.privacy_scope}
        assert await owner.tool_effect_store.database.fetchone(
            'SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    finally:
        await turn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('producer', ['gather', 'callback'])
async def test_shared_gate_upgrade_is_refused_during_native_wait(owner, tmp_path, producer):
    import contextvars
    from kestrel_sovereign._async_ownership import OwnedCallbackScope
    from kestrel_sovereign.tools.effect_journal import execute_tool_effect
    from tests.unit.test_tool_effect_invocation import governed

    lock = owner._get_privacy_transition_lock()
    witness = tmp_path / 'invalid-native-shared-upgrade'
    children = []

    async def child():
        children.append(asyncio.current_task())
        with pytest.raises(RuntimeError, match='wait on itself'):
            async with owner.privacy_transition():
                witness.write_text('invalid lease upgrade')
        return 'original upgrade refused'

    async def operation():
        if producer == 'gather':
            return await gather_owned([child],
                work_factory=lock.capture_native_work_factory(acquire_span=False),
                retain_originals=True)
        scope = OwnedCallbackScope()

        async def reader():
            task = scope.start(child)
            outcome = await await_owned_task(task)
            if outcome.error is not None:
                raise outcome.error
            return [outcome.result]

        reading = asyncio.create_task(reader(), context=contextvars.Context())
        try:
            async with scope.wait():
                return await reading
        finally:
            scope.close()
            await asyncio.gather(reading, return_exceptions=True)
            for task in tuple(scope.tasks):
                await await_owned_task(task)

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        return await execute_tool_effect(owner, registration='native-wait-upgrade',
            arguments={'producer': producer}, execute=operation)

    calling = asyncio.create_task(invoke(owner, 'Original native wait',
        invocation_id='native-wait-upgrade-' + producer))
    try:
        done, _ = await asyncio.wait({calling}, timeout=.3)
        assert calling in done, 'native wait failed to record the actual parent dependency before its child acquired the writer'
        assert await calling == ['original upgrade refused']
        assert not witness.exists()
        assert not lock.locked() and not owner._get_durable_persistence_gate().locked()
        assert await owner.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
    finally:
        for task in children:
            if not task.done():
                task.cancel()
        await asyncio.gather(calling, *children, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'failed-acquisition'])
async def test_finished_privacy_writer_is_not_retained_by_empty_binding_stack(outcome):
    import gc
    import weakref
    lock = ReentrantTransitionLock()
    attempted = asyncio.Event()

    async def writer(token=None):
        try:
            with bind_transition_lock_reentry(token):
                attempted.set()
                async with lock:
                    pass
        except RuntimeError as error:
            assert outcome == 'failed-acquisition' and 'retained span' in str(error)
        return 'physical writer finished'

    if outcome == 'failed-acquisition':
        async with lock:
            task = asyncio.create_task(writer(lock.current_reentry_token()))
            await attempted.wait()
    else:
        task = asyncio.create_task(writer())
    assert await task == 'physical writer finished'
    reference = weakref.ref(task)
    task = None
    await asyncio.sleep(0)
    gc.collect()
    assert reference() is None, 'empty privacy binding stack retained a completed writer Task'
    assert not lock.locked()


@pytest.mark.asyncio
async def test_partial_gather_admission_keeps_all_original_cleanup_writers(tmp_path, monkeypatch):
    import kestrel_sovereign._async_ownership as ownership

    lock = ReentrantTransitionLock()
    started = [asyncio.Event(), asyncio.Event()]
    stall = asyncio.Event()
    second_retired = asyncio.Event()
    witness = tmp_path / 'original-sibling-cleanup'
    accepted = []
    publication_error = RuntimeError('original postaccept native publication refusal')
    create_original = ownership.create_owned_task

    def accept_then_refuse(*args, **kwargs):
        task = create_original(*args, **kwargs)
        accepted.append(task)
        if len(accepted) == 3:
            raise publication_error
        return task

    monkeypatch.setattr(ownership, 'create_owned_task', accept_then_refuse)

    async def first():
        started[0].set()
        try:
            await stall.wait()
        finally:
            # This business acknowledgement is distinct from a direct Task
            # join. The gather already owns BOTH original cleanup writers.
            await second_retired.wait()

    async def second():
        started[1].set()
        try:
            await stall.wait()
        finally:
            async with lock:
                witness.write_text('original sibling physically retired')
            second_retired.set()

    async def third():
        await asyncio.gather(*(event.wait() for event in started))
        return 'original accepted third child'

    async def parent():
        async with lock:
            return await gather_owned([first(), second(), third()], retain_originals=True)

    calling = asyncio.create_task(parent())
    try:
        done, _ = await asyncio.wait({calling}, timeout=.3)
        assert calling in done, 'partial gather admission excluded an original sibling needed by cancellation cleanup'
        result = await await_owned_task(calling)
        assert result.error is publication_error
        assert len(accepted) == 3 and all(task.done() for task in accepted)
        assert witness.read_text() == 'original sibling physically retired'
        assert not lock.locked()
    finally:
        for task in accepted:
            if not task.done():
                task.cancel()
        await asyncio.gather(calling, *accepted, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('live_producer', [False, True])
async def test_native_factory_binds_its_actual_admission_and_preserves_other_context(
    live_producer, tmp_path,
):
    from kestrel_sovereign.storage.privacy_wrapper import current_bound_reentry_token
    from kestrel_sovereign.telemetry import bind_current_turn_id, current_turn_id

    lock = ReentrantTransitionLock()
    ready, finish = asyncio.Event(), asyncio.Event()
    factories, scopes, observed = [], [], []
    receipt = tmp_path / 'native-factory-policy'

    async def produce():
        async with lock.policy_span():
            scopes.append(lock.current_native_scope())
            if live_producer:
                factories.append(lock.capture_native_work_factory())
            ready.set()
            await finish.wait()

    async def effect():
        actual = lock.current_native_scope()
        observed.append((actual, current_bound_reentry_token(), current_turn_id()))
        assert actual is not None
        if live_producer:
            assert actual is scopes[0]
            assert current_bound_reentry_token() is actual, (
                'native worker did not bind its actual original admission')
        else:
            assert actual is not scopes[0]
            assert current_bound_reentry_token() is None, (
                'fresh native worker borrowed a retired caller span')
        assert current_turn_id() == 'native-work-consumer'
        receipt.write_text('native policy decision acknowledged')

    producer = create_owned_task(produce())
    work = None
    try:
        await asyncio.wait_for(ready.wait(), timeout=1)
        if not live_producer:
            finish.set()
            assert (await await_owned_task(producer)).error is None
            factories.append(lock.capture_native_work_factory())
        consumer_context = object() if live_producer else scopes[0]
        with bind_transition_lock_reentry(consumer_context), bind_current_turn_id('native-work-consumer'):
            work = factories[0](effect)
            outcome = await await_owned_task(work)
            assert current_bound_reentry_token() is consumer_context
        assert outcome.error is None, 'native work rejected its actual admission or fresh policy span'
        assert receipt.read_text() == 'native policy decision acknowledged'
        assert len(observed) == 1
    finally:
        finish.set()
        if work is not None:
            await await_owned_task(work)
        await await_owned_task(producer)
    assert not lock.locked() and not lock._native_scopes
