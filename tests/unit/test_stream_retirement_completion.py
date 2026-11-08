"""Original producer epilogues and admission gates remain physically owned."""
import asyncio

import pytest

from kestrel_sovereign._async_ownership import (
    OwnedCallbackScope, await_owned_operation, await_owned_task, create_owned_task, owned_task_error,
)
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ExecutionStreamLifecycle,
    _explicit_control_chain, execution_observation_error, execution_stream,
)
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_epilogue', [False, True])
@pytest.mark.parametrize('body_kind', ['ordinary', 'refusal', 'committed', 'unknown'])
async def test_delivered_body_failure_does_not_hide_an_original_native_epilogue(tmp_path, failed_epilogue, body_kind):
    body_error = (ExecutionCommitOutcomeError(body_kind) if body_kind in {'committed', 'unknown'}
                  else ExecutionAuthorityError('original source refused') if body_kind == 'refusal'
                  else OSError('original source body'))
    epilogue_error = OSError('original native epilogue')
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
        elif body_kind in {'committed', 'unknown'}:
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
