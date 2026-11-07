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
