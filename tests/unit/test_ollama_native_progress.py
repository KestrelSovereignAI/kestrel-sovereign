"""Actual Ollama SDK/native HTTP/watchdog; synthetic protocol, never inference."""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time

import ollama
import pytest
import pytest_asyncio
from kestrel_sdk.llm import RequestOptions

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent.orchestrator_engine import LLMCallInactivityTimeout, LLMCallWatchdog
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ProcessRuntimeExecutionFence, _explicit_control_chain,
    bind_execution_custody, execution_cleanup_debts, owned_native_http_client,
)
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.call_progress import call_progress_listener
from kestrel_sovereign.llm.generation_gate import NoCompletionEvidenceError, judge_generation
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter

MODEL = 'owned-native-protocol-only'
MESSAGES = [{'role': 'user', 'content': 'Protocol fixture only'}]
TOOLS = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {}}}]
BOUND = .3  # Existing real watchdog test scale; no production timeout change.


class Peer:
    def __init__(self):
        self.requests = []
        self.writes = []
        self.handlers = []
        self.writers = []
        self.sockets = []
        self.shutdown = asyncio.Event()
        self.cadence = .06
        self.parts = 10
        self.errors = []
        self.clients = []
        self.mode = 'normal'
        self.values = None
        self.release = asyncio.Event()
        self.body_sent = asyncio.Event()

    async def start(self):
        def accept(reader, writer):
            self.writers.append(writer)
            self.sockets.append(writer.get_extra_info('socket'))
            self.handlers.append(create_owned_task(self.serve(reader, writer)))
        self.server = await asyncio.start_server(accept, '127.0.0.1', 0)
        self.sockets.extend(self.server.sockets)
        self.port = self.server.sockets[0].getsockname()[1]

    def response(self, *, tools):
        message = {'role': 'assistant', 'content': 'Finished protocol response',
                   'thinking': 'PRIVATE-native-thought'}
        if tools:
            message['tool_calls'] = [{'function': {'name': 'lookup', 'arguments': {'key': 'value'}}}]
        return {'model': MODEL, 'message': message, 'done': True, 'done_reason': 'stop',
                'prompt_eval_count': 3, 'eval_count': 5}

    async def serve(self, reader, writer):
        try:
            head = await reader.readuntil(b'\r\n\r\n')
            first, *headers = head.split(b'\r\n')
            assert first == b'POST /api/chat HTTP/1.1'
            length = next(int(h.split(b':', 1)[1]) for h in headers
                          if h.lower().startswith(b'content-length:'))
            raw = await reader.readexactly(length)
            body = json.loads(raw)
            self.requests.append({'body': body, 'raw_hex': raw.hex(), 'at': time.monotonic()})
            assert body['model'] == MODEL and body['messages'] == MESSAGES
            streamed = body.get('stream', True)
            assert type(streamed) is bool
            final = self.response(tools=bool(body.get('tools')))
            if streamed:
                values = [{'model': MODEL, 'message': {'role': 'assistant', 'thinking': str(i)},
                           'done': False} for i in range(self.parts - 1)]
                values.append(final)
                if self.values is not None:
                    values = self.values
                if self.mode == 'retry-once' and len(self.requests) == 1:
                    values = [{'model': MODEL, 'message': {'role': 'assistant', 'content': 'DISCARD'},
                               'done': False}, {'error': 'overloaded'}]
                pieces = [(json.dumps(value) + '\n').encode() for value in values]
                mime = b'application/x-ndjson'
            else:
                raw = json.dumps(final).encode()
                cut = max(1, len(raw) // self.parts)
                pieces = [raw[i:i+cut] for i in range(0, len(raw), cut)]
                mime = b'application/json'
            if self.mode == 'silent':
                pieces = []
            elif self.mode == 'partial-record':
                pieces = [b'{"message":', b'{"content":', b'"unfinished', b' private', b' text']
            elif self.mode == 'malformed':
                pieces = [b'{not valid JSON}\n']
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: ' + mime +
                         b'\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n')
            await writer.drain()
            for index, piece in enumerate(pieces):
                if self.cadence:
                    try:
                        await asyncio.wait_for(self.shutdown.wait(), self.cadence)
                        break
                    except TimeoutError:
                        pass
                if self.shutdown.is_set():
                    break
                writer.write(f'{len(piece):x}\r\n'.encode() + piece + b'\r\n')
                await writer.drain()
                self.writes.append({'request': len(self.requests), 'part': index,
                                    'stream': streamed, 'at': time.monotonic(), 'hex': piece.hex()})
            self.body_sent.set()
            if self.mode in {'silent', 'partial-record', 'hold-terminal'}:
                await self.release.wait()
            if not self.shutdown.is_set():
                writer.write(b'0\r\n\r\n')
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # The actual watchdog/Stop closes this owned peer connection.
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    @asynccontextmanager
    async def client(self):
        client = ollama.AsyncClient(host=f'http://127.0.0.1:{self.port}', trust_env=False)
        self.clients.append(client)
        async with owned_native_http_client(client._client):
            yield client
        assert client._client.is_closed

    async def close(self):
        self.shutdown.set()
        self.release.set()
        self.server.close()
        await self.server.wait_closed()
        outcomes = [await await_owned_task(task) for task in self.handlers]
        assert all(o.error is None and o.cancellation is None for o in outcomes)
        assert all(w.is_closing() for w in self.writers)
        assert all(s.fileno() == -1 for s in self.sockets)
        assert all(c._client.is_closed for c in self.clients)
        return {'handlers_joined': len(outcomes), 'all_handler_outcomes_clean': True,
                'sockets': len(self.sockets), 'all_socket_filenos_minus_one': True,
                'all_original_clients_closed': True}


@pytest_asyncio.fixture
async def peer(request, tmp_path):
    peer = Peer()
    await peer.start()
    try:
        yield peer
    finally:
        cleanup = await peer.close()
        root = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', str(tmp_path))) / 'protocol-facts'
        root.mkdir(exist_ok=True)
        (root / (request.node.name + '.json')).write_text(json.dumps({
            'requests': peer.requests, 'writes': peer.writes, 'cleanup': cleanup,
            'synthetic_native_protocol_only': True, 'actual_model_call': False,
        }, indent=2) + '\n')


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['direct', 'tools-facade'])
async def test_original_logical_completion_survives_continued_native_payload(peer, entry):
    async with peer.client() as client:
        adapter = OllamaAdapter()
        watchdog = LLMCallWatchdog(BOUND)
        try:
            if entry == 'direct':
                response = await watchdog.call(adapter.get_response(
                    client, MODEL, MESSAGES, tools=TOOLS))
            else:
                seen = [item async for item in watchdog.watch(
                    adapter.get_streaming_response_with_tools(client, MODEL, MESSAGES, tools=TOOLS))]
                response, = [item for item in seen if isinstance(item, LLMResponse)]
        except LLMCallInactivityTimeout as original:
            assert len(peer.requests) == 1 and peer.requests[0]['body']['stream'] is False
            assert len(peer.writes) >= 3
            raise AssertionError('Native HTTP body progressed but logical completion went falsely idle') from original
        assert response.content == 'Finished protocol response'
        assert response.tool_calls[0].name == 'lookup'
        assert response.input_tokens == 3 and response.output_tokens == 5
        assert len(peer.requests) == 1


@pytest.mark.asyncio
async def test_existing_native_stream_control_survives_beyond_idle_window(peer):
    async with peer.client() as client:
        seen = [item async for item in LLMCallWatchdog(BOUND).watch(
            OllamaAdapter().get_streaming_response_with_tools(client, MODEL, MESSAGES))]
        response, = [item for item in seen if isinstance(item, LLMResponse)]
        assert response.content == 'Finished protocol response'
        assert peer.requests[0]['body']['stream'] is True
        assert peer.writes[-1]['at'] - peer.writes[0]['at'] > BOUND
        assert judge_generation(response, provider='ollama', model=MODEL) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', ['default', False, True], ids=['default', 'false', 'true'])
async def test_logical_raw_stream_options_keep_final_shape(peer, stream):
    peer.cadence = 0
    async with peer.client() as client:
        kwargs = {} if stream == 'default' else {'request_options': RequestOptions(raw={'stream': stream})}
        if stream is True:
            with pytest.raises(ollama.RequestError, match='raw stream=True'):
                await OllamaAdapter().get_response(client, MODEL, MESSAGES, tools=TOOLS, **kwargs)
            assert peer.requests == []
        else:
            response = await OllamaAdapter().get_response(client, MODEL, MESSAGES, tools=TOOLS, **kwargs)
            assert peer.requests[0]['body']['stream'] is True
            assert judge_generation(response) is None and response.tool_calls


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['silent', 'partial-record'])
async def test_unparsed_or_absent_wire_bytes_do_not_report_progress(peer, mode):
    peer.mode = mode
    reports = []
    async with peer.client() as client:
        with call_progress_listener(lambda: reports.append(())):
            with pytest.raises(LLMCallInactivityTimeout):
                await LLMCallWatchdog(BOUND).call(
                    OllamaAdapter().get_response(client, MODEL, MESSAGES, tools=TOOLS))
        assert reports == []
        assert len(peer.requests) == 1
        if mode == 'partial-record':
            assert len(peer.writes) >= 3


@pytest.mark.asyncio
async def test_no_logical_result_before_original_native_body_closes(peer):
    peer.mode, peer.cadence = 'hold-terminal', 0
    async with peer.client() as client:
        original = create_owned_task(OllamaAdapter().get_response(client, MODEL, MESSAGES, tools=TOOLS))
        try:
            await asyncio.wait_for(peer.body_sent.wait(), 2)
            # The complete terminal event is on the wire, but native body
            # ownership is still open. The collector must finish its drain.
            await asyncio.sleep(.02)
            assert not original.done()
        finally:
            peer.release.set()
            outcome = await await_owned_task(original)
        assert outcome.error is None and outcome.cancellation is None
        assert outcome.result.content == 'Finished protocol response'


@pytest.mark.asyncio
async def test_aggregate_preserves_complete_tool_order_usage_and_private_progress(peer):
    peer.cadence = 0
    peer.values = [
        {'model': MODEL, 'message': {'role': 'assistant', 'content': 'A', 'thinking': 'PRIVATE-A',
         'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'key': 1}}}]},
         'done': False, 'prompt_eval_count': 7},
        {'model': MODEL, 'message': {'role': 'assistant', 'content': 'B', 'thinking': 'PRIVATE-B',
         'tool_calls': [{'function': {'name': 'lookup', 'arguments': {'key': 2}}}]},
         'done': False},
        {'model': MODEL, 'message': {'role': 'assistant', 'content': ''},
         'done': True, 'done_reason': 'stop', 'eval_count': 9},
    ]
    reports = []
    async with peer.client() as client:
        with call_progress_listener(lambda *args: reports.append(args)):
            response = await OllamaAdapter().get_response(
                client, MODEL, MESSAGES, tools=TOOLS,
                request_options=RequestOptions(raw={'options': {'num_ctx': 32768}, 'think': False}))
        assert response.content == 'AB'
        assert [call.id for call in response.tool_calls] == ['ollama_call_0', 'ollama_call_1']
        assert [call.arguments for call in response.tool_calls] == [{'key': 1}, {'key': 2}]
        assert response.raw.message.thinking == 'PRIVATE-APRIVATE-B'
        assert (response.input_tokens, response.output_tokens, response.total_tokens) == (7, 9, 16)
        assert reports == [(), (), ()]  # No private provider text in progress.
        request = peer.requests[0]['body']
        assert request['think'] is False and request['options']['num_ctx'] == 32768
        assert request['tools'] == [ollama.Tool(**tool).model_dump(exclude_none=True) for tool in TOOLS]
        assert judge_generation(response) is None


@pytest.mark.asyncio
async def test_retry_covers_lazy_drain_and_discards_failed_attempt_payload(peer):
    peer.mode, peer.cadence = 'retry-once', 0
    async with peer.client() as client:
        response = await OllamaAdapter().get_response(client, MODEL, MESSAGES, tools=TOOLS)
        assert len(peer.requests) == 2
        assert response.content == 'Finished protocol response'
        assert 'DISCARD' not in response.content
        assert response.raw.message.thinking == ''.join(str(i) for i in range(peer.parts - 1)) + 'PRIVATE-native-thought'
        assert [call.id for call in response.tool_calls] == ['ollama_call_0']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['no-terminal', 'malformed', 'after-terminal'])
async def test_invalid_native_completion_is_not_delivered_or_retried(peer, mode):
    peer.cadence = 0
    if mode == 'no-terminal':
        peer.values = [{'model': MODEL, 'message': {'role': 'assistant', 'content': 'partial'}, 'done': False}]
        error = NoCompletionEvidenceError
    elif mode == 'malformed':
        peer.mode = mode
        error = ValueError
    else:
        peer.values = [peer.response(tools=True), peer.response(tools=True)]
        error = ValueError
    async with peer.client() as client:
        with pytest.raises(error):
            await OllamaAdapter().get_response(client, MODEL, MESSAGES, tools=TOOLS)
        assert len(peer.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('withdrawal', ['token', 'caller', 'repeated-caller'])
async def test_original_withdrawal_joins_native_read_without_retry(peer, withdrawal):
    peer.mode = 'silent'
    stopped = False
    async with peer.client() as client:
        original = create_owned_task(OllamaAdapter().get_response(
            client, MODEL, MESSAGES, tools=TOOLS, cancel_token=lambda: stopped), retain_originals=True)
        await asyncio.wait_for(peer.body_sent.wait(), 2)
        if withdrawal == 'token':
            stopped = True
        else:
            original.cancel('original caller withdrew')
            if withdrawal == 'repeated-caller':
                original.cancel('same original caller withdrew again')
        outcome = await await_owned_task(original)
        assert isinstance(outcome.error, asyncio.CancelledError)
        assert outcome.cancellation is None  # This observer did not withdraw.
        assert len(peer.requests) == 1 and original.done()


@pytest.mark.asyncio
async def test_retired_original_execution_cannot_deliver_or_retry_native_body(peer):
    peer.mode, peer.cadence = 'hold-terminal', 0
    fence = ProcessRuntimeExecutionFence('owned-native-protocol-only')
    with bind_execution_custody(fence):
        async with peer.client() as client:
            original = create_owned_task(OllamaAdapter().get_response(
                client, MODEL, MESSAGES, tools=TOOLS), retain_originals=True)
            await asyncio.wait_for(peer.body_sent.wait(), 2)
            fence.retire()  # Genuine retained execution lifetime, no task cancel.
            peer.release.set()
            outcome = await await_owned_task(original)
            chain = tuple(_explicit_control_chain(outcome.error))
            assert any(isinstance(error, ExecutionAuthorityError) for error in chain)
            assert outcome.result is None and outcome.cancellation is None
            assert len(peer.requests) == 1


@pytest.mark.asyncio
async def test_failed_original_native_close_retains_debt_and_never_retries(peer, monkeypatch):
    peer.cadence = 0
    failure = OSError('owned actual native response close ACK lost')
    closes = []

    async def observed_response(response):
        # The canonical response hook already retained its native closer.
        # Fail the ACK only AFTER that actual stream physically closes.
        original = response.stream.aclose

        async def close_then_fail():
            closes.append(response)
            await original()
            raise failure

        monkeypatch.setattr(response.stream, 'aclose', close_then_fail)

    async with peer.client() as client:
        client._client.event_hooks['response'].append(observed_response)
        with pytest.raises(BaseException) as raised:
            await OllamaAdapter().get_response(client, MODEL, MESSAGES, tools=TOOLS)
        assert failure in tuple(_explicit_control_chain(raised.value))
        assert execution_cleanup_debts(raised.value)
        assert len(closes) == 1 and len(peer.requests) == 1
