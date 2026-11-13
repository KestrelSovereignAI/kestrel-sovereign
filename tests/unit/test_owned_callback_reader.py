"""Actual reader callbacks settle their waiting caller on native terminal exit."""

import asyncio

import pytest

from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign.execution_custody import execution_evidence_carriers
from tests.utils.codex_inline import OwnedCallbackReader, PreTurnReader


@pytest.mark.asyncio
@pytest.mark.parametrize('reader_type', [OwnedCallbackReader, PreTurnReader])
@pytest.mark.parametrize('kind', ['cancel', 'base-group', 'unknown'])
async def test_owned_reader_delivers_the_original_terminal_without_waiting_forever(reader_type, kind):
    cancel = asyncio.CancelledError('original owned callback cancellation')
    original = (cancel if kind == 'cancel' else
        BaseExceptionGroup('original callback terminal group', [cancel, ValueError('private detail')])
        if kind == 'base-group' else ResourceOutcomeUnknown('original callback uncertainty'))
    entered = asyncio.Event()
    reader = reader_type()
    await reader.start()

    async def execute(name, arguments):
        entered.set()
        raise original

    calling = asyncio.create_task(reader.dispatch(execute, 'owned', {}))
    try:
        _done, pending = await asyncio.wait({calling}, timeout=0.5)
        assert entered.is_set(), 'The owned reader never entered its callback'
        assert not pending, 'The original callback terminal did not reach its waiting caller'
        with pytest.raises(BaseException) as caught:
            await calling
        assert any(error is original for error in execution_evidence_carriers(caught.value))
        assert not reader._handlers
    finally:
        if not calling.done():
            calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)
        await reader.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('reader_type', [OwnedCallbackReader, PreTurnReader])
async def test_owned_reader_preserves_its_reply_contract(reader_type):
    reader = reader_type()
    await reader.start()
    try:
        async def execute(name, arguments):
            return arguments, {'success': True, 'name': name}

        result = await reader.dispatch(execute, 'owned', {'marker': 'one'})
        expected = {'success': True, 'name': 'owned'}
        assert result == (({'marker': 'one'}, expected) if reader_type is OwnedCallbackReader else expected)
    finally:
        await reader.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('reader_type', [OwnedCallbackReader, PreTurnReader])
@pytest.mark.parametrize('kind', ['success', 'unknown', 'base-group'])
async def test_owned_reader_retains_caller_withdrawal_through_callback_settlement(
    reader_type, kind, tmp_path,
):
    entered, finish = asyncio.Event(), asyncio.Event()
    original = (ResourceOutcomeUnknown('original callback uncertainty after withdrawal')
        if kind == 'unknown' else BaseExceptionGroup('original callback terminal',
            [asyncio.CancelledError('source cancellation'), ValueError('source failure')])
        if kind == 'base-group' else None)
    receipt = tmp_path / 'callback-settled'
    reader = reader_type()
    await reader.start()

    async def execute(name, arguments):
        entered.set()
        await finish.wait()
        receipt.write_text('original callback physically settled')
        if original is not None:
            raise original
        return arguments, {'success': True, 'name': name}

    loop = asyncio.get_running_loop()
    previous_handler, callback_errors = loop.get_exception_handler(), []

    def record_callback_error(loop, context):
        callback_errors.append(context.get('exception'))
        if previous_handler is None:
            loop.default_exception_handler(context)
        else:
            previous_handler(loop, context)

    loop.set_exception_handler(record_callback_error)
    calling = asyncio.create_task(reader.dispatch(execute, 'owned', {}))
    try:
        await asyncio.wait_for(entered.wait(), timeout=0.5)
        calling.cancel('first reader caller withdrawal')
        await asyncio.sleep(0)
        assert not calling.done(), 'Reader abandoned its accepted callback before settlement'
        finish.set()
        completed, pending = await asyncio.wait({calling}, timeout=0.5)
        assert not pending, 'Reader did not join its original callback settlement'
        with pytest.raises(BaseException) as caught:
            await calling
        evidence = execution_evidence_carriers(caught.value)
        assert any(isinstance(error, asyncio.CancelledError)
            and error.args == ('first reader caller withdrawal',) for error in evidence), (
                'Reader lost the original caller withdrawal at callback settlement')
        assert not any(isinstance(error, asyncio.InvalidStateError) for error in evidence), (
            'Reader manufactured a reply-publication error after caller withdrawal')
        if original is not None:
            assert any(error is original for error in evidence), (
                'Reader lost the original callback terminal after caller withdrawal')
        else:
            assert isinstance(caught.value, asyncio.CancelledError)
        assert receipt.read_text() == 'original callback physically settled'
        assert not reader._handlers
        assert not callback_errors, 'Reader published an orphaned callback error after withdrawal'
    finally:
        finish.set()
        if not calling.done():
            calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)
        try:
            await reader.stop()
        finally:
            loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize('reader_type', [OwnedCallbackReader, PreTurnReader])
async def test_owned_reader_refuses_a_withdrawn_callback_before_admission(reader_type):
    release_reader = asyncio.Event()

    class DelayedReader(reader_type):
        async def _read(self):
            await release_reader.wait()
            await super()._read()

    reader = DelayedReader()
    await reader.start()
    executed = []

    async def execute(name, arguments):
        executed.append(name)
        return arguments, {'success': True}

    calling = asyncio.create_task(reader.dispatch(execute, 'withdrawn', {}))
    try:
        async with asyncio.timeout(0.5):
            while reader._calls.empty():
                await asyncio.sleep(0)
        calling.cancel('withdrawn before reader admission')
        completed, pending = await asyncio.wait({calling}, timeout=0.5)
        assert not pending, 'Unaccepted callback retained its withdrawn caller'
        with pytest.raises(asyncio.CancelledError) as caught:
            await calling
        assert caught.value.args == ('withdrawn before reader admission',)
        release_reader.set()
    finally:
        release_reader.set()
        if not calling.done():
            calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)
        await reader.stop()
    assert executed == [], 'Reader admitted a callback after its caller withdrew'
    assert not reader._handlers


@pytest.mark.asyncio
@pytest.mark.parametrize('reader_type', [OwnedCallbackReader, PreTurnReader])
@pytest.mark.parametrize('accepted', [False, True])
async def test_owned_reader_settles_native_admission_failure_before_callback_entry(
    reader_type, accepted, tmp_path,
):
    loop, release_reader = asyncio.get_running_loop(), asyncio.Event()
    previous = loop.get_task_factory()
    original = RuntimeError('original reader native task publication failed')
    allocated, entered = [], []
    receipt = tmp_path / 'unadmitted-reader-effect'

    class DelayedReader(reader_type):
        async def _read(self):
            await release_reader.wait()
            await super()._read()

    def refuse_or_accept_then_raise(loop, coroutine, **kwargs):
        if accepted:
            task = (previous(loop, coroutine, **kwargs) if previous else
                asyncio.Task(coroutine, loop=loop, **kwargs))
            allocated.append(task)
        raise original

    async def execute(name, arguments):
        entered.append(name)
        receipt.write_text('unexpected callback effect')
        return arguments, {'success': True}

    reader = DelayedReader()
    await reader.start()
    calling = asyncio.create_task(reader.dispatch(execute, 'unadmitted', {}))
    try:
        async with asyncio.timeout(0.5):
            while reader._calls.empty():
                await asyncio.sleep(0)
        loop.set_task_factory(refuse_or_accept_then_raise)
        release_reader.set()
        completed, pending = await asyncio.wait({calling}, timeout=0.5)
        loop.set_task_factory(previous)
        assert not pending, 'Reader admission failure never settled its reply waiter'
        with pytest.raises(BaseException) as caught:
            await calling
        assert any(error is original for error in execution_evidence_carriers(caught.value))
        assert len(allocated) == int(accepted) and all(task.done() for task in allocated)
        assert not entered and not receipt.exists(), 'Failed admission entered its callback body'
        assert not reader._handlers
        # A failed request does not kill the resident reader or authorize replay.
        async def following(name, arguments):
            return arguments, {'success': True, 'name': name}
        result = await reader.dispatch(following, 'independent-following-call', {})
        expected = {'success': True, 'name': 'independent-following-call'}
        assert result == (({}, expected) if reader_type is OwnedCallbackReader else expected)
    finally:
        loop.set_task_factory(previous)
        release_reader.set()
        if not calling.done():
            calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)
        await reader.stop()
        for task in allocated:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('reader_type', [OwnedCallbackReader, PreTurnReader])
@pytest.mark.parametrize('published', [False, True])
@pytest.mark.parametrize('withdrawal', [False, True])
async def test_owned_reader_joins_callback_when_completion_publication_fails(
    reader_type, published, withdrawal, tmp_path,
):
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    entered, finish = asyncio.Event(), asyncio.Event()
    original = RuntimeError('original reader completion registration failed')
    refusals = []
    receipt = tmp_path / 'reader-publication-effect'

    class PublicationFailingTask(asyncio.Task):
        def add_done_callback(self, callback, **kwargs):
            if getattr(callback, '__qualname__', '').endswith(
                    'OwnedCallbackReader._read.<locals>.<lambda>'):
                refusals.append(self)
                if published:
                    super().add_done_callback(callback, **kwargs)
                raise original
            return super().add_done_callback(callback, **kwargs)

    def task_factory(loop, coroutine, **kwargs):
        return PublicationFailingTask(coroutine, loop=loop, **kwargs)

    async def execute(name, arguments):
        entered.set()
        await finish.wait()
        receipt.write_text('accepted reader effect settled')
        return arguments, {'success': True}

    reader = reader_type()
    await reader.start()
    calling = asyncio.create_task(reader.dispatch(execute, 'accepted', {}))
    try:
        loop.set_task_factory(task_factory)
        await asyncio.wait_for(entered.wait(), timeout=0.5)
        loop.set_task_factory(previous)
        assert len(refusals) == 1
        assert not calling.done(), 'Reader returned its publication error before physical settlement'
        if withdrawal:
            calling.cancel('withdrawn while joining failed reader publication')
            await asyncio.sleep(0)
            assert not calling.done(), 'Reader abandoned its accepted callback while joining'
        finish.set()
        completed, pending = await asyncio.wait({calling}, timeout=0.5)
        assert not pending, 'Reader did not retire its original accepted callback'
        with pytest.raises(BaseException) as caught:
            await calling
        evidence = execution_evidence_carriers(caught.value)
        assert any(error is original for error in evidence)
        if withdrawal:
            assert any(isinstance(error, asyncio.CancelledError)
                and error.args == ('withdrawn while joining failed reader publication',)
                for error in evidence), 'Reader lost withdrawal while collecting failed publication'
        assert receipt.read_text() == 'accepted reader effect settled'
        assert all(task.done() for task in refusals) and not reader._handlers
    finally:
        loop.set_task_factory(previous)
        finish.set()
        if not calling.done():
            calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)
        await reader.stop()
