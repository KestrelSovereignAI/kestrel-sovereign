"""Contract grammar at Core's real adapter boundary, independent of policy."""

import asyncio
import importlib

import pytest

from kestrel_sdk.llm import LLMProtocolError, LLMResponse, ToolCall, ToolCallStarted
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.llm.completion_protocol import GenerationProtocolError, adapter_call, adapter_stream, decode_tool_arguments
from kestrel_sovereign.llm.generation_gate import judge_generation
from kestrel_sovereign.llm.output_ceiling import incomplete_generation, reported_usage
from kestrel_sovereign.llm.retry import is_retryable_error


@pytest.mark.parametrize("arguments", [None, [], 17, True, "", "{", "[]", "null", '"text"'])
def test_tool_argument_decoder_refuses_invalid_or_nonobject_data(arguments):
    with pytest.raises(LLMProtocolError):
        decode_tool_arguments(arguments)


@pytest.mark.parametrize("arguments,expected", [({}, {}), ("{}", {}),
    ({"_raw": "native data"}, {"_raw": "native data"}), ('{"_raw":"native data"}', {"_raw": "native data"})])
def test_tool_argument_decoder_preserves_real_object_data(arguments, expected):
    assert decode_tool_arguments(arguments) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{", "[]", "null", [], 17, True])
async def test_codex_malformed_inline_arguments_cannot_reach_execution(arguments):
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    executions = []

    async def execute(name, args):
        executions.append((name, args))
        return {"success": True, "result": "must not execute"}

    handler = CodexAdapter()._make_tool_call_handler(execute, "owned-thread", frozenset({"write"}))
    with pytest.raises(LLMProtocolError):
        await handler({"threadId": "owned-thread", "tool": "write", "callId": "owned-call", "arguments": arguments})
    assert executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("calls", [
    "invalid", [{}], [ToolCall(id="", name="write", arguments={})],
    [ToolCall(id="owned", name="", arguments={})],
    [ToolCall(id="owned", name="write", arguments=[])],
    [ToolCall(id="owned", name="write", arguments=None)],
])
async def test_adapter_boundary_refuses_invalid_tool_call_shapes(streaming, calls):
    response = LLMResponse(content="private", raw={"private": "data"}, tool_calls=calls,
        stop_reason="tool_calls", input_tokens=2)

    async def call():
        return response

    async def source():
        yield response

    with pytest.raises(GenerationProtocolError) as raised:
        if streaming:
            _ = [item async for item in adapter_stream(source())]
        else:
            await adapter_call(call())
    assert raised.value.usage.input_tokens == 2
    assert raised.value.usage.content is None and raised.value.usage.raw is None and raised.value.usage.tool_calls is None


@pytest.mark.asyncio
async def test_terminal_waits_for_eof_and_actual_source_cleanup():
    cleanup = []
    terminal = LLMResponse(content="owned", stop_reason="stop")

    async def source():
        try:
            yield "owned"
            yield terminal
        finally:
            cleanup.append("closed")

    observed = []
    async for item in adapter_stream(source()):
        if isinstance(item, LLMResponse):
            assert cleanup == ["closed"]
        observed.append(item)
    assert observed == ["owned", terminal]


@pytest.mark.asyncio
@pytest.mark.parametrize("trailing", ["", "trailing", LLMResponse(stop_reason="stop"), ToolCallStarted(index=0)])
async def test_every_post_terminal_item_is_a_failed_generation_with_usage(trailing):
    terminal = LLMResponse(content="never retained", raw={"private": "never retained"},
        input_tokens=2, output_tokens=3, total_tokens=5, cache_read_input_tokens=1, stop_reason="stop")

    async def source():
        yield terminal
        yield trailing

    observed = []
    with pytest.raises(GenerationProtocolError) as raised:
        async for item in adapter_stream(source()):
            observed.append(item)
    assert observed == []
    assert incomplete_generation(raised.value) is raised.value
    assert not is_retryable_error(raised.value)
    usage = reported_usage(raised.value)
    assert usage is not None and usage.content is None and usage.raw is None and usage.tool_calls is None
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens, usage.cache_read_input_tokens) == (2, 3, 5, 1)
    assert usage.stop_reason == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["", 17, True])
async def test_invalid_reason_is_protocol_failure(reason):
    async def source():
        yield LLMResponse(stop_reason=reason)

    with pytest.raises(GenerationProtocolError):
        _ = [item async for item in adapter_stream(source())]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, "future_reason", "length", "stop"])
async def test_valid_shape_preserves_reason_and_separate_generation_policy(reason):
    terminal = LLMResponse(stop_reason=reason)

    async def source():
        yield terminal

    assert [item async for item in adapter_stream(source())] == [terminal]
    assert (judge_generation(terminal) is None) == (reason == "stop")


@pytest.mark.asyncio
async def test_normal_eof_without_terminal_is_protocol_failure():
    async def source():
        yield "partial"

    observed = []
    with pytest.raises(GenerationProtocolError):
        async for item in adapter_stream(source()):
            observed.append(item)
    assert observed == ["partial"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_sdk_protocol_error_is_classified_before_any_route_wrapper(streaming):
    failure = LLMProtocolError("429 timeout private vendor body")

    async def call():
        raise failure

    async def source():
        await call()
        yield  # pragma: no cover - generator shape for the failed call

    with pytest.raises(GenerationProtocolError) as raised:
        if streaming:
            _ = [item async for item in adapter_stream(source())]
        else:
            await adapter_call(call())
    assert raised.value.__cause__ is failure
    assert "private" not in str(raised.value)
    assert incomplete_generation(raised.value) is raised.value
    assert not is_retryable_error(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_protocol_error_with_explicit_authority_cause_stays_control(streaming):
    failure = LLMProtocolError("owned")
    failure.__cause__ = ExecutionAuthorityError("owned control")

    async def call():
        raise failure

    async def source():
        await call()
        yield

    with pytest.raises(LLMProtocolError) as raised:
        if streaming:
            _ = [item async for item in adapter_stream(source())]
        else:
            await adapter_call(call())
    assert raised.value is failure


@pytest.mark.asyncio
async def test_source_cleanup_failure_cannot_expose_terminal():
    failure = RuntimeError("owned close failure")

    async def source():
        try:
            yield LLMResponse(stop_reason="stop")
        finally:
            raise failure

    observed, retained = [], []
    with pytest.raises(RuntimeError) as raised:
        async for item in adapter_stream(source(), on_terminal=retained.append):
            observed.append(item)
    assert raised.value is failure and observed == []
    assert len(retained) == 1 and retained[0].stop_reason == "stop"


@pytest.mark.asyncio
async def test_cancel_joins_source_before_propagating():
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def source():
        try:
            yield "partial"
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    stream = adapter_stream(source())
    assert await anext(stream) == "partial"
    task = asyncio.create_task(anext(stream))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled() and closed.is_set()
    await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("module,class_name", [
    ("openai", "OpenAIAdapter"), ("anthropic", "AnthropicAdapter"),
    ("google", "GoogleAdapter"), ("vertex", "VertexAIAdapter"),
    ("ollama", "OllamaAdapter"), ("openrouter", "OpenRouterAdapter"),
    ("codex", "CodexAdapter"), ("mock", "MockAdapter"),
])
async def test_every_plain_adapter_refuses_requested_tools_before_client_use(module, class_name):
    adapter = getattr(importlib.import_module("kestrel_sovereign.llm." + module + "_adapter"), class_name)()
    with pytest.raises(LLMProtocolError, match="does not accept tools"):
        await anext(adapter.get_streaming_response(client=object(), model="owned", messages=[], tools=[{}]))


@pytest.mark.asyncio
async def test_demo_adapter_has_an_unknown_terminal_and_never_fabricates_completion():
    from kestrel_sovereign.llm.mock_adapter import MockAdapter

    adapter = MockAdapter()
    messages = [{"role": "user", "content": "Hello"}]
    response = await adapter.get_response(None, "demo", messages)
    assert isinstance(response, LLMResponse) and response.stop_reason is None
    assert judge_generation(response) is not None
    items = [item async for item in adapter.get_streaming_response(None, "demo", messages)]
    assert len(items) == 2 and items[0] == response.content
    assert isinstance(items[1], LLMResponse) and items[1].stop_reason is None


@pytest.mark.asyncio
async def test_protocol_accepts_the_sdk_async_iterator_without_an_optional_close():
    terminal = LLMResponse(stop_reason="stop")

    class Items:
        def __init__(self):
            self.items = iter([terminal])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration

    assert [item async for item in adapter_stream(Items())] == [terminal]
