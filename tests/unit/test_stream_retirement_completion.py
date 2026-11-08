"""Original producer epilogues and admission gates remain physically owned."""
import asyncio

import pytest
from kestrel_sdk.resources import ResourceOutcomeUnknown

from kestrel_sovereign._async_ownership import (
    OwnedCallbackScope, await_owned_operation, await_owned_task, create_owned_task, owned_task_error,
)
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ExecutionStreamLifecycle,
    ToolAuditUnconfirmedError, _explicit_control_chain, execution_observation_error, execution_stream,
    reserve_execution_stream,
)
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_epilogue', [False, True])
@pytest.mark.parametrize('body_kind', ['ordinary', 'refusal', 'committed', 'unknown', 'cancel', 'wrapped-cancel', 'group-cancel', 'sdk-unknown', 'refusal-unknown'])
async def test_delivered_body_failure_does_not_hide_an_original_native_epilogue(tmp_path, failed_epilogue, body_kind):
    body_error = (ExecutionCommitOutcomeError(body_kind) if body_kind in {'committed', 'unknown'}
                  else ExecutionAuthorityError('original source refused') if body_kind == 'refusal'
                  else OSError('original source body'))
    if 'cancel' in body_kind:
        cancellation = asyncio.CancelledError('original source withdrawal')
        if body_kind == 'cancel':
            body_error = cancellation
        elif body_kind == 'wrapped-cancel':
            body_error.__cause__ = cancellation
        else:
            body_error = BaseExceptionGroup('original source failure', [body_error, cancellation])
    epilogue_error = OSError('original native epilogue')
    if body_kind in {'sdk-unknown', 'refusal-unknown'}:
        unknown = ResourceOutcomeUnknown('original external effect uncertain')
        body_error = unknown if body_kind == 'sdk-unknown' else ExecutionAuthorityError('original refusal carries uncertainty')
        if body_kind == 'refusal-unknown':
            body_error.__cause__ = unknown
    handle = (tmp_path / 'native-epilogue').open('w')
    attempts = []
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise body_error
        async def aclose(self):
            attempts.append(asyncio.current_task())
    def factory(body, context):
        async def original():
            try:
                await body()
            except BaseException as error:
                handle.close()
                if failed_epilogue:
                    raise execution_observation_error(error, epilogue_error)
                raise
        return create_owned_task(original(), context=context, retain_originals=True)
    stream = execution_stream(None, Source(), lifecycle=ExecutionStreamLifecycle(factory, None, lambda: None))
    try:
        with pytest.raises(BaseException) as terminal:
            await anext(stream)
        assert body_error in tuple(_explicit_control_chain(terminal.value))
        outcome = await await_owned_operation(stream.aclose(), retain_originals=True)
        if failed_epilogue:
            assert epilogue_error in tuple(_explicit_control_chain(outcome.error))
        elif body_kind in {'committed', 'unknown', 'cancel', 'wrapped-cancel', 'group-cancel', 'sdk-unknown', 'refusal-unknown'}:
            assert body_error in tuple(_explicit_control_chain(outcome.error))
        else:
            assert outcome.error is outcome.cancellation is None
        assert handle.closed and len(attempts) == 1
    finally:
        await await_owned_operation(stream.aclose(), retain_originals=True)
        handle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_close', [False, True], ids=['acknowledged-source', 'failed-source-close'])
async def test_terminal_retirement_joins_original_native_child_before_settling(tmp_path, failed_close):
    lock = ReentrantTransitionLock()
    release, child_entered = asyncio.Event(), asyncio.Event()
    handle = (tmp_path / 'original-source').open('w')
    child_handle = (tmp_path / 'original-native-child').open('w')
    original = OSError('original source physical close failed before native drain')
    children, owners, attempts = [], [], []
    reader = closer = stream = None
    async def child():
        child_entered.set()
        await release.wait()
        child_handle.close()
    class Source:
        pulls = 0
        def __aiter__(self):
            return self
        async def __anext__(self):
            self.pulls += 1
            if self.pulls == 1:
                children.append(lock.start_owned_native_work(child, acquire_span=False))
                await child_entered.wait()
                return b'original'
            raise StopAsyncIteration
        async def aclose(self):
            attempts.append(asyncio.current_task())
            handle.close()
            if failed_close:
                raise original
    try:
        native = lock.capture_native_work_factory()
        stream = execution_stream(None, Source(), lifecycle=ExecutionStreamLifecycle(
            lambda body, context: native(body, context=context),
            lambda iterator, task: owners.append((iterator, task)), lambda: None))
        try:
            assert await anext(stream) == b'original'
            reader = create_owned_task(anext(stream), retain_originals=True)
            async def terminal_delivered():
                while not stream._iterator._closed:
                    await asyncio.sleep(0)
            await asyncio.wait_for(terminal_delivered(), 1)
            assert not owners[0][1].done() and not children[0].done()
            assert handle.closed and not child_handle.closed
            closer = create_owned_task(stream.aclose(), retain_originals=True)
            done, _ = await asyncio.wait({closer}, timeout=.05)
            assert closer not in done, 'Retirement acknowledged a producer whose native child was still physically open'
            release.set()
            outcome = await await_owned_task(closer)
            terminal = await await_owned_task(reader)
            if failed_close:
                assert original in tuple(_explicit_control_chain(outcome.error))
                assert original in tuple(_explicit_control_chain(terminal.error))
            else:
                assert outcome.error is outcome.cancellation is None
                assert isinstance(terminal.error, StopAsyncIteration)
            assert child_handle.closed and len(attempts) == 1
            assert all(task.done() and iterator.owner_task.done() for iterator, task in owners)
        finally:
            release.set()
            for task in [reader, closer, *children]:
                if task is not None:
                    await await_owned_task(task)
            await await_owned_operation(stream.aclose(), retain_originals=True)
    finally:
        release.set()
        handle.close()
        child_handle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['privacy', 'cohort'])
@pytest.mark.parametrize('metadata_name,cancel_before_start', [
    ('_kestrel_owned_admission_error', False),
    ('_kestrel_owned_terminal_error', True),
], ids=['removed-admission-marker', 'diagnostic-cache-withdrawal'])
async def test_admission_metadata_refusal_cannot_strand_an_original_start_gate(
    tmp_path, boundary, metadata_name, cancel_before_start,
):
    lock, accepted, effects = ReentrantTransitionLock(), [], []
    opening = OSError('original admission inventory publication refused')
    metadata = AttributeError('task refuses diagnostic error metadata')
    path = tmp_path / 'must-not-dispatch'
    class PublicationTask(asyncio.Task):
        def __setattr__(self, name, value):
            if name == metadata_name:
                raise metadata
            return super().__setattr__(name, value)
        def add_done_callback(self, callback, *, context=None):
            if boundary == 'privacy' and 'ReentrantTransitionLock.start_owned_native_work.<locals>.completed' in getattr(callback, '__qualname__', ''):
                raise opening
            return super().add_done_callback(callback, context=context)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **options):
        task = PublicationTask(coroutine, loop=loop, **options)
        accepted.append(task)
        return task
    async def operation():
        effects.append(asyncio.current_task())
        with path.open('w') as handle:
            handle.write('unadmitted payload')
    scope = None
    try:
        loop.set_task_factory(factory)
        try:
            if boundary == 'privacy':
                lock.start_owned_native_work(operation)
            else:
                scope = OwnedCallbackScope()
                def refuse(task):
                    raise opening
                scope._join_observers.add(refuse)
                scope.start(operation)
        except BaseException as error:
            assert error is metadata or opening in tuple(_explicit_control_chain(error))
        finally:
            loop.set_task_factory(previous)
        assert len(accepted) == 1
        if cancel_before_start:
            accepted[0].cancel()
        _, pending = await asyncio.wait(set(accepted), timeout=.1)
        assert not pending, 'Metadata publication failure stranded the accepted original task behind its start gate'
        outcome = await await_owned_task(accepted[0])
        assert opening in tuple(_explicit_control_chain(outcome.error))
        if metadata_name == '_kestrel_owned_terminal_error':
            assert metadata in tuple(_explicit_control_chain(outcome.error))
            assert owned_task_error(accepted[0]) is outcome.error
        assert not effects and not path.exists()
    finally:
        loop.set_task_factory(previous)
        if scope is not None:
            scope.close()
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)


@pytest.mark.asyncio
async def test_original_admission_retains_both_native_and_cohort_publication_failures(tmp_path):
    lock = ReentrantTransitionLock()
    first, second = ExecutionCommitOutcomeError('unknown'), OSError('original cohort publication failed')
    metadata = AttributeError('diagnostic task cache refused')
    effect = tmp_path / 'unadmitted-native-effect'
    class PublicationTask(asyncio.Task):
        def __setattr__(self, name, value):
            if name == '_kestrel_owned_terminal_error':
                raise metadata
            super().__setattr__(name, value)
        def add_done_callback(self, callback, *, context=None):
            if 'ReentrantTransitionLock.start_owned_native_work.<locals>.completed' in getattr(callback, '__qualname__', ''):
                raise first
            return super().add_done_callback(callback, context=context)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    async def operation():
        effect.write_text('unadmitted native effect')
    def refuse(task):
        raise second
    task = None
    async with lock.policy_span():
        scope = OwnedCallbackScope()
        scope._join_observers.add(refuse)
        try:
            loop.set_task_factory(lambda loop, coroutine, **options: PublicationTask(coroutine, loop=loop, **options))
            task = scope.start(operation)
            task.cancel('withdraw before first original step')
            loop.set_task_factory(previous)
            outcome = await await_owned_task(task)
            graph = tuple(_explicit_control_chain(outcome.error))
            assert first in graph, 'Second admission publication erased the original native failure'
            assert second in graph and metadata in graph
            assert not effect.exists()
        finally:
            loop.set_task_factory(previous)
            scope.close()
            if task is not None:
                await await_owned_task(task)


@pytest.mark.asyncio
@pytest.mark.parametrize('completion_kind', ['ordinary', 'unknown', 'committed', 'cancel-unknown'])
async def test_original_terminal_receipt_reconciles_stale_diagnostic_cache(tmp_path, completion_kind):
    body = OSError('original body failure')
    completion = (ExecutionCommitOutcomeError(completion_kind) if completion_kind in {'unknown', 'committed'}
                  else asyncio.CancelledError('original completion cancelled') if completion_kind == 'cancel-unknown'
                  else OSError('original completion publication failed'))
    if completion_kind == 'cancel-unknown':
        completion.__cause__ = ExecutionCommitOutcomeError('unknown')
    metadata = AttributeError('terminal diagnostic update refused')
    completed = []
    handle = (tmp_path / 'original-completion-resource').open('w')
    class PublicationTask(asyncio.Task):
        def __setattr__(self, name, value):
            if name == '_kestrel_owned_terminal_error' and getattr(self, '_first_terminal_cached', False):
                raise metadata
            super().__setattr__(name, value)
            if name == '_kestrel_owned_terminal_error':
                self._first_terminal_cached = True
    async def operation():
        handle.close()
        raise body
    def publish(task, error):
        completed.append((task, error))
        raise completion
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    task = None
    try:
        loop.set_task_factory(lambda loop, coroutine, **options: PublicationTask(coroutine, loop=loop, **options))
        task = create_owned_task(operation(), on_completion=publish, retain_originals=True)
        loop.set_task_factory(previous)
        outcome = await await_owned_task(task)
        graph = tuple(_explicit_control_chain(outcome.error))
        assert completion in graph, 'Stale diagnostic cache hid the original physical terminal failure'
        assert body in graph and metadata in graph
        assert handle.closed and completed == [(task, body)]
        assert owned_task_error(task) is outcome.error
    finally:
        loop.set_task_factory(previous)
        if task is not None:
            await await_owned_task(task)
        handle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('phase,kind', [
    ('unwind', None), ('completion', None),
    ('admission', None), ('admission', 'sdk'), ('admission', 'audit'),
    ('unwind', 'sdk'), ('unwind', 'audit'),
    ('completion', 'sdk'), ('completion', 'audit'), ('completion', 'plain'),
])
async def test_private_close_cannot_acknowledge_an_independent_original_failure(tmp_path, phase, kind):
    entered, release = asyncio.Event(), asyncio.Event()
    handle = (tmp_path / 'original-private-close-source').open('w')
    independent = asyncio.CancelledError('distinct original completion or unwind failure')
    if kind in {'sdk', 'audit'}:
        independent.__cause__ = (ResourceOutcomeUnknown('original external effect uncertain')
            if kind == 'sdk' else ToolAuditUnconfirmedError('original audit receipt uncertain'))
    attempts, completions = [], []
    class Source:
        pulls = 0
        def __aiter__(self):
            return self
        async def __anext__(self):
            self.pulls += 1
            if self.pulls == 1:
                return b'original'
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                if phase == 'unwind' and kind is not None:
                    raise independent
                raise
            raise StopAsyncIteration
        async def aclose(self):
            attempts.append(asyncio.current_task())
            handle.close()
    def factory(body, context):
        async def original():
            if phase == 'admission':
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    if kind is not None:
                        raise independent
                    raise
            await body()
        def publish(task, error):
            completions.append(task)
            if phase == 'completion' and kind is not None:
                raise independent
        return create_owned_task(original(), context=context, retain_originals=True, on_completion=publish)
    lifecycle = ExecutionStreamLifecycle(factory, None, lambda: None)
    stream = (await reserve_execution_stream(None, Source(), lifecycle=lifecycle) if phase == 'admission'
              else execution_stream(None, Source(), lifecycle=lifecycle))
    reader = None
    try:
        if phase != 'admission':
            assert await anext(stream) == b'original'
            reader = create_owned_task(anext(stream), retain_originals=True)
        await asyncio.wait_for(entered.wait(), 1)
        for _ in range(2):
            outcome = await await_owned_operation(stream.aclose(), retain_originals=True)
            if kind is None:
                assert outcome.error is outcome.cancellation is None
            else:
                assert independent in tuple(_explicit_control_chain(outcome.error)), (
                    'Requested close erased an independent original failure')
                assert independent in tuple(_explicit_control_chain(stream._iterator.terminal_error))
                assert independent in tuple(_explicit_control_chain(stream.terminal_error))
        assert handle.closed and len(attempts) == len(completions) == 1
    finally:
        release.set()
        await await_owned_operation(stream.aclose(), retain_originals=True)
        if reader is not None:
            await await_owned_task(reader)
        handle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', [None, 'sdk', 'audit'])
async def test_original_watcher_failure_is_not_a_private_stop_acknowledgement(tmp_path, kind):
    from kestrel_sovereign._async_ownership import OwnedAsyncIterator
    handle = (tmp_path / 'original-watcher-owned-source').open('w')
    independent = asyncio.CancelledError('independent original watcher completion')
    if kind is not None:
        independent.__cause__ = (ResourceOutcomeUnknown('watcher retains original effect uncertainty')
            if kind == 'sdk' else ToolAuditUnconfirmedError('watcher retains original audit uncertainty'))
    attempts, watchers = [], []
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration
        async def aclose(self):
            attempts.append(asyncio.current_task())
            handle.close()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **options):
        operation = coroutine.cr_frame.f_locals.get('operation')
        if getattr(operation, 'cr_code', None) is OwnedAsyncIterator._watch_owner_cancellation.__code__:
            async def observe():
                await coroutine
                if kind is not None:
                    raise independent
            task = asyncio.Task(observe(), loop=loop, **options)
            watchers.append(task)
            return task
        return asyncio.Task(coroutine, loop=loop, **options)
    stream = execution_stream(None, Source())
    try:
        loop.set_task_factory(factory)
        terminal = await await_owned_operation(anext(stream), retain_originals=True)
        loop.set_task_factory(previous)
        if kind is None:
            assert isinstance(terminal.error, StopAsyncIteration)
        else:
            assert independent in tuple(_explicit_control_chain(terminal.error)), (
                'Original watcher uncertainty was classified as private Stop')
        for _ in range(2):
            closed = await await_owned_operation(stream.aclose(), retain_originals=True)
            if kind is None:
                assert closed.error is closed.cancellation is None
            else:
                assert independent in tuple(_explicit_control_chain(closed.error))
        assert handle.closed and len(attempts) == len(watchers) == 1 and watchers[0].done()
    finally:
        loop.set_task_factory(previous)
        await await_owned_operation(stream.aclose(), retain_originals=True)
        handle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['sdk', 'audit'])
async def test_original_watcher_uncertainty_does_not_issue_a_private_source_interrupt(tmp_path, kind):
    refused = asyncio.Event()
    independent = asyncio.CancelledError('original watcher observation publication failed')
    independent.__cause__ = (ResourceOutcomeUnknown('original observer effect uncertain')
        if kind == 'sdk' else ToolAuditUnconfirmedError('original observer audit uncertain'))
    handle = (tmp_path / 'original-unconsumed-observer-source').open('w')
    attempts = []
    class Source:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration
        async def aclose(self):
            attempts.append(asyncio.current_task())
            handle.close()
    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            caller = asyncio.current_task()
            if (not refused.is_set() and caller is not None
                    and caller.get_name().startswith('owned_async_iterator_cancel:')
                    and 'await_owned_task.<locals>.mark_complete' in getattr(callback, '__qualname__', '')):
                refused.set()
                raise independent
            return super().add_done_callback(callback, context=context)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    stream = None
    try:
        loop.set_task_factory(lambda loop, coroutine, **options: PublicationTask(coroutine, loop=loop, **options))
        stream = await reserve_execution_stream(None, Source())
        await asyncio.wait_for(refused.wait(), 1)
        watcher = await await_owned_task(stream._iterator.owner_task)
        assert independent in tuple(_explicit_control_chain(watcher.error))
        assert not stream._iterator.producer_task.cancelling() and handle.closed is False, (
            'Watcher uncertainty was translated into a private source cancellation')
        loop.set_task_factory(previous)
        closed = await await_owned_operation(stream.aclose(), retain_originals=True)
        assert independent in tuple(_explicit_control_chain(closed.error))
        assert handle.closed and len(attempts) == 1
    finally:
        loop.set_task_factory(previous)
        if stream is not None:
            await await_owned_operation(stream.aclose(), retain_originals=True)
        handle.close()
