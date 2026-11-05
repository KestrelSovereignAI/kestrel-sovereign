"""The adapter boundary: one terminal response after EOF and owned cleanup."""

from contextlib import AsyncExitStack
import json

from typing import Any, AsyncIterator, Awaitable, Callable, Optional, TypeVar

from kestrel_sdk.llm import LLMProtocolError, LLMResponse, ToolCall
from kestrel_sovereign.execution_custody import (
    is_execution_control_error,
)
from .output_ceiling import IncompleteGenerationError

_T = TypeVar("_T")


def decode_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Decode native object arguments without fabricating an executable payload."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise LLMProtocolError("Native tool arguments must contain valid JSON.") from error
    if not isinstance(arguments, dict):
        raise LLMProtocolError("Native tool arguments must contain a JSON object.")
    return arguments


def response_usage_only(response: LLMResponse) -> LLMResponse:
    """Retain native counts and completion metadata without text or tool effects."""
    return LLMResponse(
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        total_tokens=response.total_tokens,
        cache_creation_input_tokens=response.cache_creation_input_tokens,
        cache_read_input_tokens=response.cache_read_input_tokens,
        stop_reason=response.stop_reason,
    )


class GenerationProtocolError(IncompleteGenerationError, LLMProtocolError):
    """An adapter violated the generation contract; none of its text is evidence."""

    summary = "returned an invalid generation protocol"

    def __init__(self, response: Optional[LLMResponse] = None) -> None:
        super().__init__("The provider returned an invalid generation protocol; the attempt failed.")
        if response is not None:
            self.usage = response_usage_only(response)


def _validate_response_shape(response: LLMResponse) -> None:
    if response.stop_reason is not None and (type(response.stop_reason) is not str or not response.stop_reason):
        raise GenerationProtocolError(response)
    calls = response.tool_calls
    if calls is None:
        return
    if not isinstance(calls, list):
        raise GenerationProtocolError(response)
    for call in calls:
        if not isinstance(call, ToolCall) or any(type(value) is not str or not value for value in (call.id, call.name)):
            raise GenerationProtocolError(response)
        if not isinstance(call.arguments, dict):
            raise GenerationProtocolError(response)


async def adapter_call(pending: Awaitable[_T]) -> _T:
    """Translate an SDK contract failure before route wrapping and retry decisions."""
    try:
        result = await pending
        if isinstance(result, LLMResponse):
            _validate_response_shape(result)
        return result
    except LLMProtocolError as error:
        if isinstance(error, GenerationProtocolError) or is_execution_control_error(error):
            raise
        raise GenerationProtocolError() from error


async def adapter_stream(stream: AsyncIterator[Any], *,
                         on_terminal: Optional[Callable[[LLMResponse], None]] = None) -> AsyncIterator[Any]:
    """Forward deltas, retaining the terminal until the actual iterator closes.

    Any item after a terminal, including a duplicate terminal, is a failure.
    Completion policy remains in generation_gate; a missing native stop reason
    is valid protocol evidence of an unknown outcome, never a fabricated stop.
    """
    terminal = None
    try:
        async with AsyncExitStack() as cleanup:
            close = getattr(stream, "aclose", None)
            if callable(close):
                cleanup.push_async_callback(close)
            async for item in stream:
                if terminal is not None:
                    raise GenerationProtocolError(terminal)
                if isinstance(item, LLMResponse):
                    terminal = item
                    if on_terminal is not None:
                        on_terminal(item)
                    _validate_response_shape(item)
                else:
                    yield item
    except LLMProtocolError as error:
        if isinstance(error, GenerationProtocolError) or is_execution_control_error(error):
            raise
        raise GenerationProtocolError(terminal) from error
    if terminal is None:
        raise GenerationProtocolError()
    yield terminal
