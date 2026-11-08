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
    finally:
        finish.set()
        if not calling.done():
            calling.cancel()
        await asyncio.gather(calling, return_exceptions=True)
        await reader.stop()


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
