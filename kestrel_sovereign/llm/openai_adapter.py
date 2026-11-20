"""
OpenAI LLM Adapter

Adapter for OpenAI's chat completion API with full support for:
- Tool/function calling
- Streaming responses
- Vision (image inputs)
- API-based model discovery

A native OpenAI model whose chat completions endpoint refuses function tools
while it reasons has its tool-bearing requests sent to the Responses API
instead (#4198); see :data:`_RESPONSES_FUNCTION_TOOL_MODEL_PATTERN`.
"""

from contextlib import aclosing

import json
import os
import re
import openai
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, AsyncIterator, Type, TYPE_CHECKING, Union

import httpx

if TYPE_CHECKING:
    from .embedding_discovery import EmbeddingModelInfo
from pydantic import BaseModel

from kestrel_sdk.llm import LLMProtocolError, ToolCallStarted

from .completion_protocol import GenerationProtocolError, decode_tool_arguments
from .adapter import (
    LLMAdapter,
    LLMResponse,
    ReportedUsage,
    ThinkingContentSplitter,
    ThinkingDelta,
    ToolCall,
    ToolCallWithExtraContent,
    provider_usage_cost,
    split_thinking_from_content,
)
from kestrel_sdk.llm import (
    ProviderCapabilities,
    StructuredOutputMode,
    ToolStreamingMode,
    VisionInputMode,
)
try:  # SDK v5 optional surface.
    from kestrel_sdk.llm import (
        BatchHandle,
        BatchMode,
        BatchRequest,
        BatchResult,
        BatchStatus,
        CodeExecOptions,
        FileRef,
        FilesMode,
        PromptCacheMode,
        RawResponse,
        ReasoningControlMode,
        RequestOptions,
        ServerToolMode,
        TokenCount,
        TokenCountMode,
        WebSearchOptions,
    )
except ImportError:  # pragma: no cover - compatibility with SDK v4 checkouts.
    BatchHandle = BatchMode = BatchRequest = BatchResult = BatchStatus = None
    CodeExecOptions = FileRef = FilesMode = PromptCacheMode = None
    RawResponse = ReasoningControlMode = RequestOptions = ServerToolMode = None
    TokenCount = TokenCountMode = WebSearchOptions = None
from .call_progress import report_call_progress
from .gpt5_overlay import prepend_gpt5_overlay
from .model_metadata import ModelInfo, ModelCategory
from .provider_served import ProviderServed, attach_provider_served, provider_served
from .output_ceiling import (
    ProviderReportedFailure,
    attach_stop_reason,
    join_output_ceiling_notice,
    output_ceiling_notice_chunk,
    output_limit_cut_notice,
)
from .retry import with_retry
from kestrel_sovereign.execution_custody import (
    owned_native_http_client,
    native_execution_response_owner, native_sdk_operation,
    close_native_execution_stream, retain_native_execution_http_client_close, is_execution_control_error,
)

logger = logging.getLogger(__name__)

#: The chat-completions ``finish_reason`` for a response cut at the output
#: limit (OpenRouter forwards every upstream's cut as this value too).
_LENGTH_FINISH_REASON = "length"

#: The provider APIs the adapter calls, as recorded in provider-served
#: evidence: the OpenAI SDK's names for them, whatever a route's base URL.
_CHAT_COMPLETIONS_API = "chat.completions"
_RESPONSES_API = "responses"

#: The Responses API's ``incomplete_details.reason`` for a response cut at
#: its output limit.
_RESPONSES_OUTPUT_LIMIT_REASON = "max_output_tokens"


def _field(source: Any, name: str) -> Any:
    """``name`` of an SDK object or of the plain dict a test or batch holds."""
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _structured_response_format(model: Type[BaseModel]) -> Dict[str, Any]:
    # Reuse the SDK's public strict Pydantic formatter. Root-only closure leaves
    # defaulted and nested properties optional, which strict output rejects.
    function = openai.pydantic_function_tool(model)["function"]
    return {"type": "json_schema", "json_schema": {
        "name": function["name"], "schema": function["parameters"], "strict": True,
    }}


def _single_choice(choices: Any, *, allow_empty: bool = False) -> Any:
    """One selected generation; usage-only stream trailers can have none."""
    if not isinstance(choices, list):
        raise LLMProtocolError("OpenAI completion choices must be a list.")
    if not choices and allow_empty:
        return None
    if len(choices) != 1:
        raise LLMProtocolError("OpenAI completion must contain one selected choice.")
    choice = choices[0]
    index = choice.get("index") if isinstance(choice, dict) else getattr(choice, "index", None)
    if type(index) is not int or index != 0:
        raise LLMProtocolError("OpenAI completion choice must have index zero.")
    return choice


def _finish_reason(choice: Any) -> Optional[str]:
    """A chat-completions choice's ``finish_reason``, if the provider gave one."""
    reason = choice.get("finish_reason") if isinstance(choice, dict) else getattr(choice, "finish_reason", None)
    return reason if isinstance(reason, str) and reason else None


def _caller_output_budget(request_kwargs: Dict[str, Any]) -> bool:
    """Whether the request carries an output budget a caller chose.

    The adapter never sends one of its own (#3355): without one the provider
    applies the model's own limit, so a ``length`` stop is a cut nobody chose.
    A ``max_tokens`` kwarg, a raw request option, or an ``extra_body`` field
    all count, since each reaches the wire.
    """
    sources = [request_kwargs]
    extra_body = request_kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        sources.append(extra_body)
    return any(
        source.get(key) is not None
        for source in sources
        for key in ("max_completion_tokens", "max_tokens")
    )

_split_thinking_from_content = split_thinking_from_content
_ThinkingContentSplitter = ThinkingContentSplitter

# Native OpenAI currently rejects function tools combined with reasoning on
# gpt-5.6 through /v1/chat/completions. Keep the model matcher beside the
# endpoint adapter: compatible providers and the Responses API do not inherit
# this transport-specific restriction. The boundary accepts qualified ids and
# snapshots without accidentally matching a future gpt-5.60 family.
_GPT56_CHAT_FUNCTION_REASONING_MODEL_PATTERN = re.compile(
    r"(?:^|[/:])gpt-5\.6(?:[.-]|$)",
    re.IGNORECASE,
)

# Native OpenAI refuses function tools through /v1/chat/completions while
# gpt-6.2-sol reasons, and the model does not accept reasoning_effort 'none'
# (#4198), so the gpt-5.6 normalization above cannot apply: the endpoint
# cannot serve its tool-bearing turns at all. Those requests go to the
# Responses API, which serves function tools with reasoning. Requests without
# function tools keep chat completions. The boundary accepts qualified ids,
# dated snapshots and fine-tunes, and no other model: neither gpt-6.2-solar
# nor a sibling such as gpt-6.2-sol-mini, whose endpoint support is unknown.
_RESPONSES_FUNCTION_TOOL_MODEL_PATTERN = re.compile(
    r"(?:^|[/:])gpt-6\.2-sol(?:-\d{4}-\d{2}-\d{2})?(?::|$)",
    re.IGNORECASE,
)

#: Chat-completions request fields the Responses API takes under the same
#: name with the same meaning. Every other field is translated explicitly or
#: refused, never dropped.
_RESPONSES_SAME_FIELDS = frozenset({
    "metadata",
    "parallel_tool_calls",
    "safety_identifier",
    "service_tier",
    "temperature",
    "top_p",
    "user",
})


def _effective_chat_request(request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The payload a prepared chat request actually sends.

    The OpenAI SDK merges ``extra_body`` over its typed fields, so a nested
    value wins over the same top-level field. Every decision about the request
    (which endpoint serves it, what the Responses API is sent) reads this
    merged payload, never the top level alone (#4198).
    """
    effective = {key: value for key, value in request_kwargs.items() if key != "extra_body"}
    extra_body = request_kwargs.get("extra_body")
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise ValueError("A chat request's extra_body must be an object.")
        effective.update(extra_body)
    return effective


def _has_function_tools(request_kwargs: Dict[str, Any]) -> bool:
    """Whether the payload ``request_kwargs`` sends carries a function tool,
    whether it is named at the top level or in ``extra_body``."""
    tools = _effective_chat_request(request_kwargs).get("tools")
    return isinstance(tools, list) and any(
        isinstance(tool, dict) and tool.get("type") == "function" for tool in tools
    )


def _capability_kwargs(**kwargs: Any) -> Dict[str, Any]:
    fields = getattr(ProviderCapabilities, "__dataclass_fields__", {})
    if not fields:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in fields}


def _enum_value(enum_type: Any, name: str, fallback: str) -> Any:
    if enum_type is None:
        return fallback
    return getattr(enum_type, name)


def _object_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return {k: v for k, v in asdict(obj).items() if v is not None}
    data: Dict[str, Any] = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        try:
            value = getattr(obj, key)
        except Exception:
            continue
        if callable(value) or value is None:
            continue
        data[key] = value
    return data


def _normalized_openai_usage(
    usage: Any,
) -> tuple[
    Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]
]:
    """Return SDK-semantic usage with cached prompt tokens split out.

    OpenAI-compatible routes are not perfectly uniform. Current OpenAI SDK
    objects put both counters under ``prompt_tokens_details``; some compatible
    providers return ``cached_tokens`` (and occasionally ``cache_write_tokens``)
    at the top level. Preserve either reported shape, preferring the canonical
    nested value when both exist.
    """
    field = _field

    def token_count(value: Any) -> Optional[int]:
        return (
            value
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            else None
        )

    # The Responses API names the same counters ``input_tokens`` /
    # ``output_tokens`` / ``input_tokens_details`` with the same meaning
    # (cached tokens included in the input count), #4198.
    input_tokens = field(usage, "prompt_tokens")
    if input_tokens is None:
        input_tokens = field(usage, "input_tokens")
    output_tokens = field(usage, "completion_tokens")
    if output_tokens is None:
        output_tokens = field(usage, "output_tokens")
    total_tokens = field(usage, "total_tokens")
    details = field(usage, "prompt_tokens_details")
    if details is None:
        details = field(usage, "input_tokens_details")
    cache_creation_input_tokens = token_count(
        field(details, "cache_write_tokens")
    )
    if cache_creation_input_tokens is None:
        cache_creation_input_tokens = token_count(
            field(usage, "cache_write_tokens")
        )
    cache_read_input_tokens = token_count(field(details, "cached_tokens"))
    if cache_read_input_tokens is None:
        cache_read_input_tokens = token_count(field(usage, "cached_tokens"))

    separated_cache_tokens = sum(
        value for value in (
            cache_creation_input_tokens, cache_read_input_tokens
        ) if value is not None
    )
    if separated_cache_tokens:
        if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
            input_tokens = max(0, input_tokens - separated_cache_tokens)
        if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
            total_tokens = max(0, total_tokens - separated_cache_tokens)
    return (
        input_tokens,
        output_tokens,
        total_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
    )


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _protocol_failure_usage(usage: Any, stop_reason: Optional[str]) -> Optional[LLMResponse]:
    """Retain only reported token counts when native response parsing fails."""
    if usage is None:
        return None
    inputs, outputs, total, writes, reads = _normalized_openai_usage(usage)
    return LLMResponse(input_tokens=inputs, output_tokens=outputs, total_tokens=total,
        cache_creation_input_tokens=writes, cache_read_input_tokens=reads,
        stop_reason=stop_reason)


def _usage_fields(usage: Any) -> Dict[str, Optional[int]]:
    """``LLMResponse`` token fields for a reported usage object, if any."""
    if not usage:
        return {}
    inputs, outputs, total, writes, reads = _normalized_openai_usage(usage)
    return {
        "input_tokens": inputs,
        "output_tokens": outputs,
        "total_tokens": total,
        "cache_creation_input_tokens": writes,
        "cache_read_input_tokens": reads,
    }


# ---- Responses API (#4198) ------------------------------------------------
#
# The agent hands every adapter OpenAI chat-format messages. A Responses API
# call takes ``input`` items instead, and a reasoning model's tool calls must
# be sent back with the reasoning that produced them. Nothing is stored by the
# provider (``store=False``): the Kestrel session is the only owner of the
# conversation, and the reasoning travels between the calls of a turn as the
# encrypted items the provider returns, on the replayed tool call's
# ``extra_content`` (the carrier Gemini's thought signatures use, #3619).
# Replayed items omit their provider ids: with nothing stored there is no
# provider-side item for an id to name.
#
# The provider keeps a reasoning item only when the history before it is the
# history it was produced after, so a response's output is sent back as it
# came: every reasoning item, message and function call in its original
# order, each message with its own phase. The replay record holds that order
# and the encrypted reasoning, but no text and no call: a message is named by
# its length and a call by its id, and both are read back from the Kestrel
# session's own assistant message, so nothing is sent that the session does
# not hold. A session message that no longer fits its record is refused.

#: The ``extra_content`` namespace of an OpenAI-format tool call holding what
#: a Responses API call must send back with it, and the key within it.
OPENAI_EXTRA_CONTENT = "openai"
_RESPONSES_REPLAY = "responses"


class ResponsesAPIFailure(openai.OpenAIError, ProviderReportedFailure):
    """The Responses API reported that a response it accepted failed.

    The request itself was accepted, so there is no HTTP status to report.
    ``code`` and ``provider_message`` are the provider's own words, kept so
    the failure reaches the caller as the provider stated it. ``usage`` is
    what the provider reported for the attempt (its token counts, its status
    and what served it, never text), so the failed attempt is recorded with
    the tokens it consumed and the model and response id that answered.
    """

    def __init__(
        self,
        *,
        event: str,
        code: Optional[str] = None,
        message: Optional[str] = None,
        response_id: Optional[str] = None,
    ) -> None:
        self.event = event
        self.code = code
        self.provider_message = message
        self.response_id = response_id
        detail = f"OpenAI Responses API {event}"
        if code:
            detail += f" ({code})"
        if message:
            detail += f": {message}"
        super().__init__(detail)


def _failed_attempt_evidence(
    usage: Any, stop_reason: Optional[str], served: Optional[ProviderServed],
) -> Optional[LLMResponse]:
    """What the provider reported for an attempt that failed: its token
    counts, its stop reason and what served it, never text. ``None`` when it
    reported neither counts nor a model or response id."""
    evidence = _protocol_failure_usage(usage, stop_reason)
    if evidence is None:
        if served is None:
            return None
        evidence = LLMResponse(stop_reason=stop_reason)
    return attach_provider_served(evidence, served)


def _responses_failure(
    event: str,
    *,
    served: Optional[ProviderServed],
    response: Any = None,
    error: Any = None,
) -> ResponsesAPIFailure:
    """The provider's failure of an accepted response, carrying what it
    reported for the attempt. ``served`` is what the failed response, or for
    a stream the response it began, named as serving it."""
    if error is None:
        error = _field(response, "error")
    code = _field(error, "code")
    message = _field(error, "message")
    failure = ResponsesAPIFailure(
        event=event,
        code=code if isinstance(code, str) else None,
        message=message if isinstance(message, str) else None,
        response_id=served.response_id if served is not None else None,
    )
    failure.usage = _failed_attempt_evidence(
        _field(response, "usage") if response is not None else None,
        _responses_stop_reason(response) if response is not None else None,
        served,
    )
    return failure


def _responses_tool(tool: Any) -> Dict[str, Any]:
    """A chat-format function tool as a Responses API function tool."""
    if not isinstance(tool, dict) or tool.get("type") != "function":
        kind = tool.get("type") if isinstance(tool, dict) else type(tool).__name__
        raise ValueError(
            f"The OpenAI Responses API path sends function tools only, not {kind!r}."
        )
    function = tool.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError("A function tool must name its function.")
    converted: Dict[str, Any] = {
        "type": "function",
        "name": name,
        "parameters": function.get("parameters"),
        # The Responses API requires ``strict``. Chat completions leaves it
        # off when unset, so an unset value keeps that meaning.
        "strict": bool(function.get("strict", False)),
    }
    if function.get("description") is not None:
        converted["description"] = function["description"]
    return converted


def _responses_tool_choice(choice: Any) -> Any:
    if isinstance(choice, str):
        return choice
    function = choice.get("function") if isinstance(choice, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    if (
        isinstance(choice, dict)
        and choice.get("type") == "function"
        and isinstance(name, str)
        and name
    ):
        return {"type": "function", "name": name}
    raise ValueError("This tool_choice has no Responses API equivalent.")


def _responses_text_format(response_format: Any) -> Dict[str, Any]:
    kind = response_format.get("type") if isinstance(response_format, dict) else None
    if kind == "json_schema":
        schema = response_format.get("json_schema")
        if not isinstance(schema, dict):
            raise ValueError("A json_schema response_format must carry its schema.")
        converted = {
            "type": "json_schema",
            "name": schema.get("name"),
            "schema": schema.get("schema"),
        }
        for key in ("strict", "description"):
            if schema.get(key) is not None:
                converted[key] = schema[key]
        return converted
    if kind in ("json_object", "text"):
        return {"type": kind}
    raise ValueError(f"A response_format of type {kind!r} has no Responses API equivalent.")


def _responses_input_part(part: Any) -> Dict[str, Any]:
    """A chat content part as a Responses API input content part."""
    kind = part.get("type") if isinstance(part, dict) else None
    if kind in ("text", "input_text"):
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("A text content part must carry text.")
        return {"type": "input_text", "text": text}
    if kind == "image_url":
        image = part.get("image_url")
        url = image.get("url") if isinstance(image, dict) else image
        if not isinstance(url, str) or not url:
            raise ValueError("An image_url content part must carry a URL.")
        detail = image.get("detail") if isinstance(image, dict) else None
        return {"type": "input_image", "image_url": url, "detail": detail or "auto"}
    if kind == "input_image":
        return dict(part)
    raise ValueError(f"A content part of type {kind!r} has no Responses API equivalent.")


def _responses_content(content: Any, *, role: str) -> Any:
    """A chat message's content as Responses API message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"A {role} message's content must be text or content parts.")
    if role == "assistant":
        texts = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            text = part.get("text") if isinstance(part, dict) else None
            if kind not in ("text", "output_text") or not isinstance(text, str):
                raise ValueError("An assistant message's content parts must be text.")
            texts.append(text)
        return "".join(texts)
    return [_responses_input_part(part) for part in content]


def _responses_tool_output(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_responses_input_part(part) for part in content]
    raise ValueError("A tool result's content must be text or content parts.")


#: The kinds of entry a replay record holds, in the response's own order.
_REPLAY_REASONING = "reasoning"
_REPLAY_MESSAGE = "message"
_REPLAY_FUNCTION_CALL = "function_call"


def _replay_entry(entry: Any) -> Dict[str, Any]:
    """One validated entry of a replay record, as :func:`_parse_responses_output`
    wrote it: an encrypted reasoning item, a message's phase and text length,
    or the id of a function call."""
    kind = entry.get("type") if isinstance(entry, dict) else None
    if kind == _REPLAY_REASONING:
        encrypted = entry.get("encrypted_content")
        summary = entry.get("summary", [])
        if not isinstance(encrypted, str) or not encrypted:
            raise ValueError("Responses API replay reasoning must be encrypted reasoning items.")
        if not isinstance(summary, list) or not all(
            isinstance(part, dict)
            and part.get("type") == "summary_text"
            and isinstance(part.get("text"), str)
            for part in summary
        ):
            raise ValueError("A replayed reasoning item's summary must be summary text.")
        return {
            "type": "reasoning",
            "summary": [dict(part) for part in summary],
            "encrypted_content": encrypted,
        }
    if kind == _REPLAY_MESSAGE:
        phase = entry.get("phase")
        length = entry.get("text_length")
        if phase is not None and not isinstance(phase, str):
            raise ValueError("A Responses API replay phase must be a string.")
        if type(length) is not int or length < 0:
            raise ValueError("A replayed message must record its text length.")
        return {"type": kind, "phase": phase, "text_length": length}
    if kind == _REPLAY_FUNCTION_CALL:
        call_id = entry.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("A replayed function call must record its call id.")
        return {"type": kind, "call_id": call_id}
    raise ValueError(f"A Responses API replay entry of type {kind!r} is not one Kestrel wrote.")


def _responses_replay(tool_calls: List[Any]) -> Optional[List[Dict[str, Any]]]:
    """The replay record an assistant turn's tool calls carry for the
    Responses API, as :func:`_responses_tool_calls` put it, validated; or
    ``None`` when no call carries one."""
    found = None
    for call in tool_calls:
        extra = call.get("extra_content") if isinstance(call, dict) else None
        if extra is None:
            continue
        if not isinstance(extra, dict):
            raise ValueError("A tool call's extra_content must be an object.")
        namespace = extra.get(OPENAI_EXTRA_CONTENT)
        if namespace is None:
            continue
        if not isinstance(namespace, dict):
            raise ValueError("A tool call's OpenAI extra_content must be an object.")
        replay = namespace.get(_RESPONSES_REPLAY)
        if replay is None:
            continue
        if not isinstance(replay, dict):
            raise ValueError("A tool call's Responses API replay data must be an object.")
        if found is not None:
            raise ValueError(
                "Two tool calls of one assistant turn carry Responses API replay data."
            )
        found = replay
    if found is None:
        return None
    output = found.get("output")
    if not isinstance(output, list) or set(found) != {"output"}:
        raise ValueError("Responses API replay data must hold only the response's output order.")
    return [_replay_entry(entry) for entry in output]


def _responses_function_call(call: Any) -> Dict[str, Any]:
    """A chat-format tool call of the session as a Responses API function call."""
    function = call.get("function") if isinstance(call, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    call_id = call.get("id") if isinstance(call, dict) else None
    if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
        raise ValueError("A replayed tool call must carry its id and name.")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {})
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _replayed_output(
    replay: List[Dict[str, Any]], text: str, calls: List[Any],
) -> List[Dict[str, Any]]:
    """The session's assistant tool-call message, sent back in the order the
    response produced it (see the section comment above).

    ``text`` is the session message's content and ``calls`` its tool calls.
    Each recorded message takes its length of ``text`` in turn, and each
    recorded call is the session's call with that id. A session message whose
    text is not exactly as long as its recorded messages, or whose calls are
    not exactly the recorded ones, no longer describes the response: it is
    refused rather than reordered or merged.
    """
    pending: Dict[str, Dict[str, Any]] = {}
    for call in calls:
        item = _responses_function_call(call)
        if item["call_id"] in pending:
            raise ValueError("An assistant message names one tool call id twice.")
        pending[item["call_id"]] = item
    recorded = sum(
        entry["text_length"] for entry in replay if entry["type"] == _REPLAY_MESSAGE
    )
    if recorded != len(text):
        raise ValueError(
            "An assistant message's text no longer matches the Responses output "
            "it replays; its reasoning cannot be sent back in place."
        )
    items: List[Dict[str, Any]] = []
    offset = 0
    for entry in replay:
        if entry["type"] == _REPLAY_REASONING:
            items.append(entry)
        elif entry["type"] == _REPLAY_MESSAGE:
            end = offset + entry["text_length"]
            message: Dict[str, Any] = {
                "type": "message", "role": "assistant", "content": text[offset:end],
            }
            offset = end
            if entry["phase"] is not None:
                message["phase"] = entry["phase"]
            items.append(message)
        else:
            call = pending.pop(entry["call_id"], None)
            if call is None:
                raise ValueError(
                    "A Responses API replay names a tool call its assistant message "
                    "does not carry."
                )
            items.append(call)
    if pending:
        raise ValueError(
            "An assistant message carries a tool call its Responses API replay "
            "does not place."
        )
    return items


def _responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chat-format ``messages`` as Responses API ``input`` items."""
    items: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer", "user"):
            items.append({"role": role, "content": _responses_content(content, role=role)})
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("An assistant message's tool_calls must be a list.")
            text = _responses_content(content, role="assistant")
            replay = _responses_replay(calls)
            if replay is not None:
                items.extend(_replayed_output(replay, text, calls))
                continue
            # A message no Responses API call produced (an earlier route's, or
            # a chat completion's) has no native order to keep.
            if text:
                items.append({"role": "assistant", "content": text})
            items.extend(_responses_function_call(call) for call in calls)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("A tool result must name the call it answers.")
            items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": _responses_tool_output(content),
            })
        else:
            raise ValueError(f"A message with role {role!r} has no Responses API equivalent.")
    return items


def _responses_request(
    *, model: str, messages: List[Dict[str, Any]], chat_request: Dict[str, Any],
) -> Dict[str, Any]:
    """The Responses API request equivalent to the prepared chat request.

    ``chat_request`` is what :meth:`OpenAIAdapter._prepare_chat_completion_request`
    produced, request options and raw fields included, so both endpoints
    start from the same caller intent. What is translated is the merged
    payload :func:`_effective_chat_request` gives, the one endpoint selection
    also reads: a nested value wins over the same top-level field, exactly as
    on chat completions. Every field is translated through one table; a field
    the Responses API has no equivalent for is refused rather than dropped,
    and nothing is forwarded untranslated.
    """
    effective = _effective_chat_request(chat_request)

    request: Dict[str, Any] = {
        "model": model,
        "input": _responses_input(messages),
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    reasoning: Dict[str, Any] = {}
    text: Dict[str, Any] = {}
    effort = None
    text_overrides: Dict[str, Any] = {}
    output_budgets = set()
    refused = []
    for key, value in effective.items():
        if key == "tools":
            request["tools"] = [_responses_tool(tool) for tool in value]
        elif key == "tool_choice":
            request["tool_choice"] = _responses_tool_choice(value)
        elif key == "response_format":
            text_overrides["format"] = _responses_text_format(value)
        elif key == "verbosity":
            text_overrides["verbosity"] = value
        elif key in ("max_completion_tokens", "max_tokens"):
            if value is not None:
                output_budgets.add(value)
                request["max_output_tokens"] = value
        elif key == "reasoning_effort":
            effort = value
        elif key in ("reasoning", "text"):
            # Responses API objects a raw option may carry; they merge with,
            # never replace, what the chat fields translate to.
            if not isinstance(value, dict):
                raise ValueError(f"A raw {key} field must be an object.")
            (reasoning if key == "reasoning" else text).update(value)
        elif key in _RESPONSES_SAME_FIELDS or key == "prompt_cache_key":
            request[key] = value
        else:
            refused.append(key)
    if refused:
        raise ValueError(
            f"OpenAI {model} tool requests use the Responses API, which has no "
            f"equivalent of {', '.join(sorted(refused))}; refusing to drop them."
        )
    # Two spellings of one field (a chat field and a raw Responses object, or
    # both chat output budgets) are refused rather than one chosen.
    conflicts = []
    if len(output_budgets) > 1:
        conflicts.append("max_output_tokens")
    if effort is not None and "effort" in reasoning:
        conflicts.append("reasoning.effort")
    conflicts.extend(f"text.{name}" for name in sorted(text_overrides) if name in text)
    if conflicts:
        raise ValueError(
            f"OpenAI {model} tool request sets {', '.join(conflicts)} twice; "
            "refusing to choose one."
        )
    if effort is not None:
        reasoning["effort"] = effort
    text.update(text_overrides)
    if reasoning:
        request["reasoning"] = reasoning
    if text:
        request["text"] = text
    return request


def _responses_stop_reason(response: Any) -> Optional[str]:
    """The Responses API's own verdict: its ``status``, or for an incomplete
    response the ``incomplete_details.reason`` it gives."""
    status = _field(response, "status")
    if status == "incomplete":
        reason = _field(_field(response, "incomplete_details"), "reason")
        return reason if isinstance(reason, str) and reason else status
    return status if isinstance(status, str) and status else None


def _responses_reasoning_replay_item(item: Any) -> Optional[Dict[str, Any]]:
    """A returned reasoning item as it is sent back, or ``None`` when the
    provider gave no encrypted content, which is all that can be sent back."""
    encrypted = _field(item, "encrypted_content")
    if not isinstance(encrypted, str) or not encrypted:
        return None
    summary = []
    for part in _field(item, "summary") or []:
        text = _field(part, "text")
        if isinstance(text, str):
            summary.append({"type": "summary_text", "text": text})
    return {"type": "reasoning", "summary": summary, "encrypted_content": encrypted}


def _parse_responses_output(response: Any) -> tuple:
    """``(content, function_calls, replay)`` of a final response.

    ``content`` is every message's text, in order; ``replay`` the response's
    output order as its replay record holds it (see the section comment): each
    sendable reasoning item, each message's phase and text length, and each
    function call's id, exactly where the response produced them.
    """
    output = _field(response, "output")
    if not isinstance(output, list):
        raise LLMProtocolError("OpenAI Responses output must be a list.")
    texts: List[str] = []
    function_calls: List[tuple] = []
    replay: List[Dict[str, Any]] = []
    for item in output:
        kind = _field(item, "type")
        if kind == "reasoning":
            replay_item = _responses_reasoning_replay_item(item)
            if replay_item is not None:
                replay.append(replay_item)
        elif kind == "message":
            item_phase = _field(item, "phase")
            message_texts = []
            for part in _field(item, "content") or []:
                part_kind = _field(part, "type")
                text = _field(part, "text" if part_kind == "output_text" else "refusal")
                if part_kind not in ("output_text", "refusal") or not isinstance(text, str):
                    raise LLMProtocolError("OpenAI Responses message content must be text.")
                message_texts.append(text)
            message_text = "".join(message_texts)
            texts.append(message_text)
            replay.append({
                "type": _REPLAY_MESSAGE,
                "phase": item_phase if isinstance(item_phase, str) else None,
                "text_length": len(message_text),
            })
        elif kind == "function_call":
            call_id = _field(item, "call_id")
            name = _field(item, "name")
            if (
                type(call_id) is not str or not call_id
                or type(name) is not str or not name
            ):
                raise LLMProtocolError("Native tool identity must be a non-empty string.")
            function_calls.append(
                (call_id, name, decode_tool_arguments(_field(item, "arguments")))
            )
            replay.append({"type": _REPLAY_FUNCTION_CALL, "call_id": call_id})
        else:
            raise LLMProtocolError(
                "OpenAI Responses returned an output item Kestrel did not request."
            )
    return ("".join(texts) or None), function_calls, replay


def _responses_tool_calls(
    function_calls: List[tuple], replay: List[Dict[str, Any]],
) -> Optional[List[ToolCall]]:
    """The response's tool calls. The first carries the turn's replay record,
    so its follow-up sends the response back as it came; a response of
    function calls alone has no order the calls themselves do not keep."""
    carries_replay = any(entry["type"] != _REPLAY_FUNCTION_CALL for entry in replay)
    calls: List[ToolCall] = []
    for index, (call_id, name, arguments) in enumerate(function_calls):
        if index == 0 and carries_replay:
            calls.append(ToolCallWithExtraContent(
                id=call_id,
                name=name,
                arguments=arguments,
                extra_content={OPENAI_EXTRA_CONTENT: {_RESPONSES_REPLAY: {
                    "output": replay,
                }}},
            ))
        else:
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return calls or None


def _responses_served(response: Any, started: Any = None) -> Optional[ProviderServed]:
    """What ``response`` names as serving it. A stream's ``started`` response
    (the one its ``response.created`` event carried) supplies what a failed or
    unfinished stream's own response does not name."""
    model = _field(response, "model") if response is not None else None
    response_id = _field(response, "id") if response is not None else None
    if started is not None:
        model = model or _field(started, "model")
        response_id = response_id or _field(started, "id")
    return provider_served(_RESPONSES_API, model=model, response_id=response_id)


class OpenAIAdapter(LLMAdapter):
    """
    Adapter for interacting with the OpenAI API.

    Supports:
    - Chat completions with tool calling
    - Streaming responses
    - JSON mode
    """

    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    DEFAULT_EMBEDDING_DIM = 1536

    def __init__(
        self,
        name: str = "openai",
        *,
        supports_embeddings: Optional[bool] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        native_openai: bool = False,
    ):
        self.name = name
        if supports_embeddings is None:
            supports_embeddings = name == "openai"
        if supports_embeddings and embedding_model is None:
            embedding_model = self.DEFAULT_EMBEDDING_MODEL
        if supports_embeddings and embedding_dim is None:
            embedding_dim = self.DEFAULT_EMBEDDING_DIM
        self._supports_embeddings = supports_embeddings
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        # Canonical OpenAI (real openai vendor on the official base_url) is the
        # only deployment that exposes /batches, /files, /responses. This is set
        # explicitly by the registry — NOT inferred from embedding support,
        # since an OpenAI-compatible route can enable embeddings too. Defaults
        # False so any other construction path is treated as compatible-only.
        self._native_openai = bool(native_openai)

    def provider_capabilities(self) -> ProviderCapabilities:
        # OpenAI-native endpoints — /batches, /files, /responses and the
        # prompt_cache_key param — are exposed only by canonical OpenAI, not by
        # the OpenAI-*compatible* routes this adapter also serves via a custom
        # base_url (Kimi, DeepSeek, OpenRouter, …). Gate those on the same
        # native-OpenAI signal used for embeddings so the framework never routes
        # batch/file/raw operations to a compatible endpoint that would 404.
        # Token counting (local tiktoken estimate) and reasoning effort (a
        # request param) are endpoint-agnostic and stay ungated.
        native = self._native_openai
        return ProviderCapabilities(**_capability_kwargs(
            supports_tools=True,
            supports_streaming=True,
            supports_vision=True,
            supports_structured_output=True,
            supports_embeddings=self._supports_embeddings,
            supports_token_counting=True,
            supports_reasoning_control=True,
            supports_batch=native,
            supports_files=native,
            supports_prompt_cache=native,
            supports_raw_passthrough=native,
            # Native OpenAI accepts a ``system``/``developer`` message in any
            # position, so operator signals can be delivered as a
            # mid-conversation ``system`` turn (which ``_normalize_messages``
            # passes through untouched) rather than a user-visible
            # ``<operator_notice>`` bubble. Gated on ``native`` — the
            # OpenAI-*compatible* base_urls this adapter also serves
            # (Kimi/DeepSeek/OpenRouter/…) are not guaranteed to honor a
            # trailing system turn, so we don't over-claim for them.
            supports_inline_system=native,
            structured_output_mode=StructuredOutputMode.JSON_SCHEMA,
            tool_streaming_mode=ToolStreamingMode.NATIVE_DELTA,
            vision_input_mode=VisionInputMode.OPENAI_IMAGE_URL,
            embedding_model=self._embedding_model,
            embedding_dim=self._embedding_dim,
            reasoning_control_mode=_enum_value(
                ReasoningControlMode, "EFFORT", "effort"
            ),
            prompt_cache_mode=_enum_value(
                PromptCacheMode, "AUTOMATIC" if native else "NONE",
                "automatic" if native else "none",
            ),
            batch_mode=_enum_value(
                BatchMode, "FILE_BASED" if native else "NONE",
                "file_based" if native else "none",
            ),
            files_mode=_enum_value(
                FilesMode, "UPLOAD" if native else "NONE",
                "upload" if native else "none",
            ),
            token_count_mode=_enum_value(TokenCountMode, "ESTIMATE", "estimate"),
            reasoning_effort_levels=("minimal", "low", "medium", "high"),
            raw_operations=(
                (
                    "chat.completions.create",
                    "responses.create",
                    "responses.retrieve",
                    "files.create",
                    "files.list",
                    "files.retrieve",
                    "files.delete",
                    "files.content",
                    "batches.create",
                    "batches.retrieve",
                    "batches.cancel",
                )
                if native
                else ()
            ),
            model_dependent=("vision",),
            notes=(
                "Structured output uses response_format=json_schema for Pydantic models.",
                "OpenAI prompt caching is implicit for stable prefixes.",
                "Token counting is estimated locally with tiktoken.",
            ),
        ))

    def contract_features(self) -> frozenset[str]:
        features = {"token_counting", "reasoning_control"}
        if self._native_openai:
            features |= {"batch", "files", "prompt_cache", "raw_passthrough"}
        return frozenset(features)

    async def probe_reachable(
        self,
        client: Any,
        *,
        base_url: Optional[str] = None,
        timeout: float = 1.5,
    ) -> Optional[bool]:
        """Probe an OpenAI-compatible local route's models endpoint."""
        root = base_url
        if not root:
            root = str(getattr(client, "base_url", "") or "") or None
        if not root:
            return None

        root = str(root).rstrip("/")
        url = f"{root}/models" if root.endswith("/v1") else f"{root}/v1/models"
        try:
            async with owned_native_http_client(httpx.AsyncClient(timeout=timeout)) as http:
                retain_native_execution_http_client_close(http)
                response = await http.get(url)
            return response.status_code < 500
        except httpx.HTTPError as error:
            if is_execution_control_error(error):
                raise
            return False

    @staticmethod
    def _report_embedding_usage(
        usage_sink: Optional[ReportedUsage], response: Any
    ) -> None:
        """Copy an embeddings response's usage into ``usage_sink`` (#3426).

        ``usage.prompt_tokens`` is the billed input (embeddings have no
        output); OpenRouter adds ``usage.cost``.
        """
        if usage_sink is None:
            return
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "prompt_tokens", None)
        if tokens is None:
            tokens = getattr(usage, "total_tokens", None)
        usage_sink.add(
            input_tokens=tokens,
            cost=provider_usage_cost(usage),
            model=getattr(response, "model", None),
        )

    @native_sdk_operation
    async def aembed(
        self,
        client: openai.AsyncOpenAI,
        text: str,
        *,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        usage_sink: Optional[ReportedUsage] = None,
        **kwargs: Any,
    ) -> Optional[List[float]]:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        create_kwargs: Dict[str, Any] = {
            "model": model or self.DEFAULT_EMBEDDING_MODEL,
            "input": text,
        }
        # Matryoshka-capable models (text-embedding-3-*, qwen3-embedding-*)
        # accept a ``dimensions`` param to truncate the output vector. Only
        # forward it when set so providers that don't support it are unaffected.
        if dimensions is not None:
            create_kwargs["dimensions"] = dimensions
        response = await with_retry(
            client.embeddings.create,
            **create_kwargs,
        )
        self._report_embedding_usage(usage_sink, response)
        data = getattr(response, "data", None) or []
        if not data:
            return None
        embedding = getattr(data[0], "embedding", None)
        return list(embedding) if embedding is not None else None

    @native_sdk_operation
    async def aembed_batch(
        self,
        client: openai.AsyncOpenAI,
        texts: List[str],
        *,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        usage_sink: Optional[ReportedUsage] = None,
        **kwargs: Any,
    ) -> List[Optional[List[float]]]:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        if not texts:
            return []
        create_kwargs: Dict[str, Any] = {
            "model": model or self.DEFAULT_EMBEDDING_MODEL,
            "input": texts,
        }
        if dimensions is not None:
            create_kwargs["dimensions"] = dimensions
        response = await with_retry(
            client.embeddings.create,
            **create_kwargs,
        )
        self._report_embedding_usage(usage_sink, response)
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        for item in getattr(response, "data", None) or []:
            index = getattr(item, "index", None)
            embedding = getattr(item, "embedding", None)
            if isinstance(index, int) and 0 <= index < len(embeddings) and embedding is not None:
                embeddings[index] = list(embedding)
        return embeddings

    def contribute_system_prompt(
        self, model_id: str, base: Optional[str]
    ) -> Optional[str]:
        """Inject the GPT-5 behavior contract for gpt-5 family models.

        See #807 / #806. Same overlay as ``CodexAdapter`` because GPT-5 needs
        the discipline contracts regardless of which OAuth/API path the model
        is reached through.
        """
        return prepend_gpt5_overlay(base, model_id)

    def _output_limit_notice(
        self,
        model: str,
        finish_reason: Optional[str],
        caller_budget: bool,
        *,
        cut_reason: str = _LENGTH_FINISH_REASON,
    ) -> Optional[str]:
        """The notice for a ``length`` cut at a limit nobody chose, else None.

        No output budget is sent unless a caller chose one, so the limit a
        ``length`` stop hit is the provider's own and its size is not reported
        (#3355). A request with a caller's budget keeps its text untouched.
        ``cut_reason`` is the API's own name for that stop: ``length`` on chat
        completions, ``max_output_tokens`` on the Responses API.
        """
        return output_limit_cut_notice(
            provider=self.name,
            model=model,
            stop_reason=finish_reason,
            cut_reason=cut_reason,
            caller_budget=caller_budget,
        )

    @native_sdk_operation
    async def get_response(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Gets a response from the OpenAI model.

        Args:
            client: OpenAI async client
            model: Model name to use
            messages: Chat messages
            format: Response format (e.g., "json") - DEPRECATED, use response_format
            tools: Optional list of tools in OpenAI function calling format
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            LLMResponse with content and/or tool calls
        """
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        reported_native_usage = None
        finish_reason = None
        try:
            # Apply provider/model-specific system-prompt contributions (e.g. the
            # GPT-5 behavior contract). #807 / #806.
            messages = self._apply_system_prompt_contribution(messages, model)
            # Normalize messages - OpenAI requires tool_calls arguments to be JSON strings
            normalized_messages = self._normalize_messages(messages)

            extra_kwargs = {}

            # Handle structured output via Pydantic model
            if response_format is not None and issubclass(response_format, BaseModel):
                extra_kwargs["response_format"] = _structured_response_format(response_format)
            elif format == "json":
                extra_kwargs["response_format"] = {"type": "json_object"}

            if tools:
                extra_kwargs["tools"] = tools
                # Let the model decide whether to use tools or respond directly
                extra_kwargs["tool_choice"] = "auto"

            # Pass through additional kwargs (max_tokens, temperature, etc.)
            # Note: newer models (gpt-5.x, o1, o3, etc.) use max_completion_tokens instead of max_tokens
            if "max_tokens" in kwargs:
                extra_kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            for key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
                if key in kwargs:
                    extra_kwargs[key] = kwargs[key]

            # Provider-specific body extensions (e.g. llama.cpp's `cache_prompt`,
            # which tells llama-server to be aggressive about retaining the
            # slot's KV cache for prefix matching).  The service layer decides
            # when to set this and passes it through via `extra_body` kwarg.
            # See issue #704.
            if "extra_body" in kwargs and kwargs["extra_body"]:
                extra_kwargs["extra_body"] = kwargs["extra_body"]

            extra_kwargs = self._prepare_chat_completion_request(
                extra_kwargs,
                model=model,
                options=kwargs.get("request_options"),
            )

            if self._uses_responses_api(extra_kwargs, model=model):
                # The Responses path translates the messages itself, replay
                # data included; ``normalized_messages`` is chat-only.
                return await self._get_responses_api_response(
                    client, model, messages, extra_kwargs,
                )

            response = await with_retry(
                client.chat.completions.create,
                model=model,
                messages=normalized_messages,
                **extra_kwargs
            )

            reported_native_usage = getattr(response, "usage", None)
            choice = _single_choice(response.choices)
            message = choice.message
            # #3355: the provider's own verdict on whether the response is
            # finished; ``length`` means it was cut at an output limit.
            finish_reason = _finish_reason(choice)

            # Parse tool calls if present
            # Use getattr to safely access tool_calls (may not exist on mock objects)
            parsed_tool_calls = None
            message_tool_calls = getattr(message, 'tool_calls', None)
            if message_tool_calls:
                parsed_tool_calls = []
                for tc in message_tool_calls:
                    args = decode_tool_arguments(tc.function.arguments)

                    parsed_tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args
                    ))

            # Extract token usage from response
            input_tokens = None
            output_tokens = None
            total_tokens = None
            cache_creation_input_tokens = None
            cache_read_input_tokens = None
            if hasattr(response, 'usage') and response.usage:
                (
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_creation_input_tokens,
                    cache_read_input_tokens,
                ) = _normalized_openai_usage(response.usage)

            _, content = _split_thinking_from_content(
                message.content,
                getattr(message, 'reasoning_content', None),
            )

            notice = self._output_limit_notice(
                model, finish_reason, _caller_output_budget(extra_kwargs),
            )
            if notice is not None:
                content = join_output_ceiling_notice(content, notice)

            return attach_provider_served(attach_stop_reason(LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                raw=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ), finish_reason), provider_served(
                _CHAT_COMPLETIONS_API,
                model=_field(response, "model"),
                response_id=_field(response, "id"),
            ))

        except LLMProtocolError as error:
            if is_execution_control_error(error) or isinstance(error, GenerationProtocolError):
                raise
            raise GenerationProtocolError(
                _protocol_failure_usage(reported_native_usage, finish_reason)
            ) from error
        except openai.RateLimitError as e:
            logger.error("OpenAI rate limit exceeded: %s", type(e).__name__)
            raise
        except openai.APIConnectionError as e:
            logger.error("OpenAI connection error: %s", type(e).__name__)
            raise
        except openai.APIError as e:
            logger.error("OpenAI API error: %s", type(e).__name__)
            raise
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.error("OpenAI adapter failed: %s", type(e).__name__)
            raise

    async def get_streaming_response(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, LLMResponse]]:
        """Yield plain text and one terminal response from the canonical stream."""
        async with aclosing(self._plain_stream(
            client=client, model=model, messages=messages, tools=tools,
            response_format=response_format, **kwargs,
        )) as stream:
            async for item in stream:
                yield item

    async def get_streaming_response_with_tools(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, ThinkingDelta, LLMResponse]]:
        """
        Stream response with tool call detection.

        This method streams text content as it arrives AND detects tool calls
        from the streaming chunks. Tool calls arrive as deltas with indices
        that need to be accumulated and assembled.

        Args:
            client: OpenAI async client
            model: Model name to use
            messages: Chat messages
            tools: Optional tools in OpenAI function calling format
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Yields:
            str: Text content chunks as they arrive
            LLMResponse: Terminal response at end-of-stream carrying token usage (and tool_calls when present)

        Example:
            async for item in adapter.get_streaming_response_with_tools(...):
                if isinstance(item, str):
                    print(item, end='', flush=True)  # Stream text to user
                elif isinstance(item, LLMResponse):
                    if item.has_tool_calls:
                        for tc in item.tool_calls:
                            result = execute_tool(tc)
                            # Continue conversation with tool results
        """
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        reported_native_usage = None
        finish_reason = None
        try:
            # Apply provider/model-specific system-prompt contributions (e.g. the
            # GPT-5 behavior contract). #807 / #806.
            messages = self._apply_system_prompt_contribution(messages, model)
            # Normalize messages for OpenAI compatibility
            normalized_messages = self._normalize_messages(messages)

            extra_kwargs = {}
            if tools:
                extra_kwargs["tools"] = tools
                extra_kwargs["tool_choice"] = "auto"

            # Handle structured output in streaming mode
            if response_format is not None and issubclass(response_format, BaseModel):
                extra_kwargs["response_format"] = _structured_response_format(response_format)

            # Pass through additional kwargs
            if "max_tokens" in kwargs:
                extra_kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            for key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
                if key in kwargs:
                    extra_kwargs[key] = kwargs[key]

            # Provider-specific body extensions (issue #704).  See get_response().
            if "extra_body" in kwargs and kwargs["extra_body"]:
                extra_kwargs["extra_body"] = kwargs["extra_body"]

            extra_kwargs = self._prepare_chat_completion_request(
                extra_kwargs,
                model=model,
                options=kwargs.get("request_options"),
            )

            if self._uses_responses_api(extra_kwargs, model=model):
                async with aclosing(self._stream_responses_api(
                    client, model, messages, extra_kwargs,
                )) as responses_stream:
                    async for item in responses_stream:
                        yield item
                return

            # Request streaming with usage stats
            extra_kwargs["stream_options"] = {"include_usage": True}

            logger.info(f"Starting OpenAI stream with tools for model: {model}")
            async def acquire_stream(**request):
                async with native_execution_response_owner(getattr(client, "_client", None)) as responses:
                    acquired = await client.chat.completions.create(**request)
                    responses.handles.clear()
                    return acquired

            stream = await with_retry(
                acquire_stream,
                model=model,
                messages=normalized_messages,
                stream=True,
                acquired_result_cleanup=close_native_execution_stream,
                **extra_kwargs
            )

            from kestrel_sovereign.execution_custody import retain_native_execution_stream_close
            retain_native_execution_stream_close(stream, self)

            # Accumulator for tool calls - keyed by index
            # Each tool call arrives in chunks with the same index
            tool_calls_accumulator: Dict[int, Dict[str, Any]] = {}
            splitter = _ThinkingContentSplitter(provider=self.name)
            text_content = ""
            reasoning_content = ""
            chunk_count = 0
            input_tokens = None
            output_tokens = None
            total_tokens = None
            cache_creation_input_tokens = None
            cache_read_input_tokens = None
            call_cost = None
            finish_reason = None
            served_model = None
            response_id = None

            try:
                async for chunk in stream:
                    report_call_progress()
                    # #4198: every chunk names the response and the model the
                    # provider served; keep the first it reports.
                    served_model = served_model or _field(chunk, "model")
                    response_id = response_id or _field(chunk, "id")
                    # Extract usage from final chunk (OpenAI sends it with stream_options)
                    if hasattr(chunk, 'usage') and chunk.usage:
                        reported_native_usage = chunk.usage
                        (
                            input_tokens,
                            output_tokens,
                            total_tokens,
                            cache_creation_input_tokens,
                            cache_read_input_tokens,
                        ) = _normalized_openai_usage(chunk.usage)
                        # OpenRouter reports exact per-call cost on the usage-only
                        # final chunk when ``usage: {include: true}`` was sent
                        # (kestrel #1806). Stash it for the terminal response.
                        _cost = getattr(chunk.usage, 'cost', None)
                        if _cost is None:
                            _extra = getattr(chunk.usage, 'model_extra', None)
                            if isinstance(_extra, dict):
                                _cost = _extra.get('cost')
                        if _cost is not None:
                            try:
                                call_cost = float(_cost)
                            except (TypeError, ValueError):
                                call_cost = None

                    # Skip chunks with no choices (like the final usage-only chunk)
                    choice = _single_choice(chunk.choices, allow_empty=True)
                    if choice is None:
                        continue

                    # #3355: the finish reason rides on the last choice chunk.
                    finish_reason = _finish_reason(choice) or finish_reason
                    delta = choice.delta

                    # Handle text content - yield immediately for real-time streaming
                    delta_reasoning_content = getattr(delta, "reasoning_content", None)
                    if isinstance(delta_reasoning_content, str) and delta_reasoning_content:
                        reasoning_content += delta_reasoning_content
                        yield ThinkingDelta(delta_reasoning_content, provider=self.name)

                    content_delta = getattr(delta, "content", None)
                    if isinstance(content_delta, str) and content_delta:
                        for item in splitter.feed(content_delta):
                            if isinstance(item, str):
                                chunk_count += 1
                                text_content += item
                            yield item

                    # Handle tool call deltas - accumulate by index
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index

                            # Initialize accumulator for this tool call if
                            # needed AND emit the SDK 0.7.0 ToolCallStarted
                            # marker — exactly once per distinct index, on
                            # the first delta for that index. OpenAI's
                            # first delta typically carries id and name (in
                            # the same fragment that introduces the index),
                            # but the contract permits None for either when
                            # the provider stream hasn't surfaced them yet.
                            # We capture whatever's on this delta and emit
                            # ``None`` for any field that arrives empty —
                            # the final LLMResponse is the source of truth
                            # for the assembled call.
                            is_new_call = idx not in tool_calls_accumulator
                            if is_new_call:
                                tool_calls_accumulator[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": ""
                                }

                            # Identity fields repeat whole values; only arguments fragment.
                            current = tool_calls_accumulator[idx]
                            for key, value in (("id", tc_delta.id), ("name", getattr(tc_delta.function, "name", None))):
                                if value is None:
                                    continue
                                if type(value) is not str or not value:
                                    raise LLMProtocolError("Native tool identity must be a non-empty string.")
                                if current[key] and current[key] != value:
                                    raise LLMProtocolError("Native tool delta returned conflicting identity.")
                                current[key] = value

                            if tc_delta.function:
                                # Accumulate arguments (comes in multiple chunks)
                                arguments = tc_delta.function.arguments
                                if arguments is not None:
                                    if type(arguments) is not str:
                                        raise LLMProtocolError("Native tool argument fragments must be strings.")
                                    current["arguments"] += arguments

                            if is_new_call:
                                # Emit ToolCallStarted only after we've
                                # absorbed the first delta's id/name so the
                                # marker carries them when present rather
                                # than always being ``(None, None)``.
                                current = tool_calls_accumulator[idx]
                                yield ToolCallStarted(
                                    index=idx,
                                    id=current["id"] or None,
                                    name=current["name"] or None,
                                )
            finally:
                await close_native_execution_stream(stream, self)

            logger.info(f"Stream completed. Text chunks: {chunk_count}, Tool calls: {len(tool_calls_accumulator)}")

            for item in splitter.flush():
                if isinstance(item, str):
                    chunk_count += 1
                    text_content += item
                yield item

            # #3355: the text has already streamed, so a cut at the provider's
            # output limit is marked by a final chunk — and carried on the
            # terminal response's content, which mirrors what was shown.
            notice = self._output_limit_notice(
                model, finish_reason, _caller_output_budget(extra_kwargs),
            )
            if notice is not None:
                notice_chunk = output_ceiling_notice_chunk(
                    notice, follows_text=bool(text_content),
                )
                text_content += notice_chunk
                yield notice_chunk

            # Assemble any tool calls collected during the stream.
            parsed_tool_calls = None
            if tool_calls_accumulator:
                parsed_tool_calls = []
                for tc_data in tool_calls_accumulator.values():
                    args = decode_tool_arguments(tc_data["arguments"] or "{}")

                    parsed_tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args
                    ))
                parsed_tool_calls = parsed_tool_calls or None

            # Always emit a terminal LLMResponse carrying token usage, even for
            # text-only turns. Previously this was nested under `if
            # tool_calls_accumulator`, so text-only streams — the common case —
            # dropped their usage entirely, a silent billing undercount. The
            # service layer meters streamed turns from this terminal response;
            # consumers read it only for tool_calls / usage (visible content was
            # already streamed as chunks). Mirrors the anthropic adapter fix
            # (#1686/#1684). OpenRouter inherits this via super() delegation.
            _raw = {}
            if reasoning_content:
                _raw["reasoning_content"] = reasoning_content
            if call_cost is not None:
                _raw["cost"] = call_cost
            yield attach_provider_served(attach_stop_reason(LLMResponse(
                content=text_content if text_content else None,
                tool_calls=parsed_tool_calls,
                raw=_raw or None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ), finish_reason), provider_served(
                _CHAT_COMPLETIONS_API, model=served_model, response_id=response_id,
            ))

        except LLMProtocolError as error:
            if is_execution_control_error(error) or isinstance(error, GenerationProtocolError):
                raise
            raise GenerationProtocolError(
                _protocol_failure_usage(reported_native_usage, finish_reason)
            ) from error
        except openai.RateLimitError as e:
            logger.error("OpenAI rate limit exceeded during streaming with tools: %s", type(e).__name__)
            raise
        except openai.APIConnectionError as e:
            logger.error("OpenAI connection error during streaming with tools: %s", type(e).__name__)
            raise
        except openai.APIError as e:
            logger.error("OpenAI API error during streaming with tools: %s", type(e).__name__)
            raise
        except Exception as e:
            logger.error("OpenAI streaming with tools failed: %s", type(e).__name__)
            raise

    async def continue_with_tool_results(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Continue a conversation after executing tool calls.

        Args:
            client: OpenAI async client
            model: Model name
            messages: Original messages (including assistant's tool call message)
            tool_results: List of tool results in format:
                [{"tool_call_id": str, "content": str}, ...]
            tools: Optional tools to include for multi-turn

        Returns:
            LLMResponse with the model's follow-up
        """
        # Append tool results as tool messages
        extended_messages = messages.copy()
        for result in tool_results:
            extended_messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["content"]
            })

        return await self.get_response(
            client=client,
            model=model,
            messages=extended_messages,
            tools=tools
        )

    async def count_tokens(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        total = self._estimate_message_tokens(model, messages)
        return TokenCount(input_tokens=total, total_tokens=total) if TokenCount else {
            "input_tokens": total,
            "total_tokens": total,
        }

    @native_sdk_operation
    async def batch_submit(
        self,
        client: Any,
        requests: List[Any],
        **kwargs: Any,
    ) -> Any:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        lines = []
        for index, request in enumerate(requests):
            body = self._batch_request_body(request)
            # OpenAI Batch requires a unique, non-empty custom_id per line.
            # BatchRequest.custom_id defaults to "" in the SDK, so fall back to a
            # stable per-index id rather than emit empty ids that get rejected.
            custom_id = getattr(request, "custom_id", "") or f"request-{index}"
            lines.append(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    },
                    separators=(",", ":"),
                )
            )
        filename = kwargs.get("filename", "kestrel-openai-batch.jsonl")
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        uploaded = await self.file_upload(
            client,
            (filename, payload),
            purpose="batch",
        )
        batch_kwargs = {
            "input_file_id": uploaded.id,
            "endpoint": "/v1/chat/completions",
            "completion_window": kwargs.get("completion_window", "24h"),
        }
        if kwargs.get("metadata") is not None:
            batch_kwargs["metadata"] = kwargs["metadata"]
        batch = await with_retry(client.batches.create, **batch_kwargs)
        return self._batch_handle(batch)

    @native_sdk_operation
    async def batch_poll(self, client: Any, handle: Any, **kwargs: Any) -> Any:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        batch = await with_retry(client.batches.retrieve, getattr(handle, "id", handle))
        return self._batch_handle(batch)

    @native_sdk_operation
    async def batch_results(self, client: Any, handle: Any, **kwargs: Any) -> List[Any]:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        raw_handle = getattr(handle, "raw", None) or handle
        output_file_id = (
            getattr(raw_handle, "output_file_id", None)
            or getattr(handle, "output_file_id", None)
            or kwargs.get("output_file_id")
        )
        if not output_file_id:
            refreshed = await self.batch_poll(client, handle)
            raw_handle = getattr(refreshed, "raw", None) or refreshed
            output_file_id = getattr(raw_handle, "output_file_id", None)
        if not output_file_id:
            return []
        content_response = await with_retry(client.files.content, output_file_id)
        content = await self._read_file_content(content_response)
        results = []
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id", "")
            error = item.get("error")
            response_body = ((item.get("response") or {}).get("body") or None)
            response = None
            error_text = json.dumps(error) if error is not None else None
            if response_body:
                try:
                    response = self._llm_response_from_chat_completion(response_body)
                except LLMProtocolError:
                    # Each native batch item owns its outcome. A refused item
                    # cannot hide valid neighbors or expose an executable call.
                    error_text = "Invalid generation protocol in batch result."
            if BatchResult:
                results.append(
                    BatchResult(
                        custom_id=custom_id,
                        response=response,
                        error=error_text,
                        raw=item,
                    )
                )
            else:
                results.append(
                    {
                        "custom_id": custom_id,
                        "response": response,
                        "error": error_text,
                        "raw": item,
                    }
                )
        return results

    @native_sdk_operation
    async def batch_cancel(self, client: Any, handle: Any, **kwargs: Any) -> Any:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        batch = await with_retry(client.batches.cancel, getattr(handle, "id", handle))
        return self._batch_handle(batch)

    @native_sdk_operation
    async def file_upload(
        self,
        client: Any,
        file: Any,
        *,
        purpose: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        uploaded = await with_retry(
            client.files.create,
            file=file,
            purpose=purpose or kwargs.get("purpose") or "assistants",
        )
        return self._file_ref(uploaded)

    @native_sdk_operation
    async def file_list(self, client: Any, **kwargs: Any) -> List[Any]:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        response = await with_retry(client.files.list, **kwargs)
        return [self._file_ref(item) for item in getattr(response, "data", [])]

    @native_sdk_operation
    async def file_get(self, client: Any, file_id: str, **kwargs: Any) -> Any:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        response = await with_retry(client.files.retrieve, file_id)
        return self._file_ref(response)

    @native_sdk_operation
    async def file_delete(self, client: Any, file_id: str, **kwargs: Any) -> bool:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        response = await with_retry(client.files.delete, file_id)
        return bool(getattr(response, "deleted", False) or getattr(response, "id", None))

    def file_reference(self, file_ref: Any) -> Dict[str, Any]:
        file_id = getattr(file_ref, "id", None)
        if not file_id:
            raise ValueError("file_ref.id is required")
        return {"file_id": file_id}

    def _prepare_chat_completion_request(
        self,
        request_kwargs: Dict[str, Any],
        *,
        model: str,
        options: Any = None,
    ) -> Dict[str, Any]:
        """Apply options and enforce chat-completions compatibility once.

        All non-streaming, text-streaming, tool-streaming, and batch request
        builders finish here. Keeping the endpoint policy after raw option
        merging means neither a caller-supplied top-level value nor
        ``extra_body`` can restore a combination the native endpoint rejects.
        """
        out = request_kwargs
        if options is not None:
            out = self.apply_request_options(out, options, model=model)

        if not self._requires_none_reasoning_for_function_tools(out, model=model):
            return out

        extra_body = out.get("extra_body")
        nested_effort = (
            extra_body.get("reasoning_effort") if isinstance(extra_body, dict) else None
        )
        requested_effort = out.get("reasoning_effort")
        if requested_effort in (None, "none") and nested_effort not in (
            None,
            "none",
        ):
            requested_effort = nested_effort
        if requested_effort not in (None, "none"):
            logger.warning(
                "OpenAI %s function-tool request uses /v1/chat/completions, "
                "which requires reasoning_effort='none'; normalized %r (#2501)",
                model,
                requested_effort,
            )

        out["reasoning_effort"] = "none"
        if isinstance(extra_body, dict) and "reasoning_effort" in extra_body:
            # The OpenAI SDK merges extra_body into the final payload after its
            # typed arguments, so a nested value could otherwise override the
            # normalized top-level field. Copy before removing it to preserve
            # the caller's dictionary and every unrelated cache/custom field.
            normalized_extra_body = dict(extra_body)
            normalized_extra_body.pop("reasoning_effort")
            out["extra_body"] = normalized_extra_body
        return out

    def _requires_none_reasoning_for_function_tools(
        self,
        request_kwargs: Dict[str, Any],
        *,
        model: str,
    ) -> bool:
        """Whether native chat-completions requires reasoning to be disabled."""
        if (
            not self._native_openai
            or not _GPT56_CHAT_FUNCTION_REASONING_MODEL_PATTERN.search(model)
        ):
            return False
        return _has_function_tools(request_kwargs)

    def _uses_responses_api(
        self,
        request_kwargs: Dict[str, Any],
        *,
        model: str,
    ) -> bool:
        """Whether this request must go to the Responses API (#4198).

        Read from the payload the request actually sends
        (:func:`_has_function_tools`), so a function tool named only in
        ``extra_body`` selects the Responses API exactly as a top-level one
        does: the payload translated is the payload selected on.
        """
        return (
            self._native_openai
            and bool(_RESPONSES_FUNCTION_TOOL_MODEL_PATTERN.search(model))
            and _has_function_tools(request_kwargs)
        )

    async def _get_responses_api_response(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        chat_request: Dict[str, Any],
    ) -> LLMResponse:
        """One non-streaming Responses API call for a prepared chat request.

        Runs inside ``get_response``'s native SDK operation, so the call has
        the same custody, retry and error reporting as a chat completion.
        """
        reported_usage = None
        stop_reason = None
        served = None
        try:
            request = _responses_request(
                model=model, messages=messages, chat_request=chat_request,
            )
            response = await with_retry(client.responses.create, **request)
            reported_usage = _field(response, "usage")
            served = _responses_served(response)
            stop_reason = _responses_stop_reason(response)
            if _field(response, "status") == "failed":
                raise _responses_failure(
                    "response failed", response=response, served=served,
                )
            content, function_calls, replay = _parse_responses_output(response)
            notice = self._output_limit_notice(
                model, stop_reason, _caller_output_budget(chat_request),
                cut_reason=_RESPONSES_OUTPUT_LIMIT_REASON,
            )
            if notice is not None:
                content = join_output_ceiling_notice(content, notice)
            return attach_provider_served(attach_stop_reason(LLMResponse(
                content=content,
                tool_calls=_responses_tool_calls(function_calls, replay),
                raw=response,
                **_usage_fields(reported_usage),
            ), stop_reason), served)
        except LLMProtocolError as error:
            if is_execution_control_error(error) or isinstance(error, GenerationProtocolError):
                raise
            raise GenerationProtocolError(
                _failed_attempt_evidence(reported_usage, stop_reason, served)
            ) from error

    async def _stream_responses_api(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        chat_request: Dict[str, Any],
    ) -> AsyncIterator[Union[str, ThinkingDelta, ToolCallStarted, LLMResponse]]:
        """Stream one Responses API call for a prepared chat request.

        Text and reasoning summaries stream as they arrive, and each function
        call is announced as it starts. The terminal response is built from
        the provider's final ``response.completed`` (or ``response.incomplete``)
        event: its tool calls, usage, stop reason and served evidence. The
        stream is acquired, owned and closed exactly as the chat-completions
        stream is, so Stop or an early close retires it the same way. A
        ``response.failed`` or ``error`` event raises the provider's failure,
        carrying the usage it reported and what served the response (from the
        stream's ``response.created`` event when the failure names neither).
        The streamed text is the turn's text: a final response holding other
        text is a protocol failure, since its replay record describes only the
        final response's output.
        """
        reported_usage = None
        stop_reason = None
        served = None
        try:
            request = _responses_request(
                model=model, messages=messages, chat_request=chat_request,
            )
            logger.info("Starting OpenAI Responses stream with tools for model: %s", model)

            async def acquire_stream(**acquire_request):
                async with native_execution_response_owner(getattr(client, "_client", None)) as responses:
                    acquired = await client.responses.create(**acquire_request)
                    responses.handles.clear()
                    return acquired

            stream = await with_retry(
                acquire_stream,
                stream=True,
                acquired_result_cleanup=close_native_execution_stream,
                **request,
            )

            from kestrel_sovereign.execution_custody import retain_native_execution_stream_close
            retain_native_execution_stream_close(stream, self)

            text_content = ""
            started: Dict[Any, int] = {}
            final = None
            # The response ``response.created`` announced: what serves the
            # stream before its terminal event names it.
            opened = None
            try:
                async for event in stream:
                    report_call_progress()
                    kind = _field(event, "type")
                    if kind == "response.created" and opened is None:
                        opened = _field(event, "response")
                        served = _responses_served(None, opened)
                    elif kind in ("response.output_text.delta", "response.refusal.delta"):
                        delta = _field(event, "delta")
                        if isinstance(delta, str) and delta:
                            text_content += delta
                            yield delta
                    elif kind in (
                        "response.reasoning_summary_text.delta",
                        "response.reasoning_text.delta",
                    ):
                        delta = _field(event, "delta")
                        if isinstance(delta, str) and delta:
                            yield ThinkingDelta(delta, provider=self.name)
                    elif kind == "response.output_item.added":
                        item = _field(event, "item")
                        key = _field(event, "output_index")
                        if _field(item, "type") == "function_call" and key not in started:
                            started[key] = len(started)
                            call_id = _field(item, "call_id")
                            name = _field(item, "name")
                            yield ToolCallStarted(
                                index=started[key],
                                id=call_id if isinstance(call_id, str) and call_id else None,
                                name=name if isinstance(name, str) and name else None,
                            )
                    elif kind in ("response.completed", "response.incomplete"):
                        final = _field(event, "response")
                    elif kind == "response.failed":
                        failed = _field(event, "response")
                        served = _responses_served(failed, opened)
                        raise _responses_failure(
                            "response failed", response=failed, served=served,
                        )
                    elif kind == "error":
                        raise _responses_failure(
                            "stream error", error=event, served=served,
                        )
            finally:
                await close_native_execution_stream(stream, self)

            if final is None:
                raise LLMProtocolError(
                    "OpenAI Responses stream ended without a terminal response."
                )
            reported_usage = _field(final, "usage")
            served = _responses_served(final, opened)
            stop_reason = _responses_stop_reason(final)
            final_text, function_calls, replay = _parse_responses_output(final)
            if not text_content and final_text:
                text_content = final_text
                yield final_text
            elif text_content != (final_text or ""):
                raise LLMProtocolError(
                    "OpenAI Responses streamed text its final response does not hold."
                )
            logger.info(
                "Responses stream completed. Tool calls: %d", len(function_calls),
            )

            notice = self._output_limit_notice(
                model, stop_reason, _caller_output_budget(chat_request),
                cut_reason=_RESPONSES_OUTPUT_LIMIT_REASON,
            )
            if notice is not None:
                notice_chunk = output_ceiling_notice_chunk(
                    notice, follows_text=bool(text_content),
                )
                text_content += notice_chunk
                yield notice_chunk

            yield attach_provider_served(attach_stop_reason(LLMResponse(
                content=text_content or None,
                tool_calls=_responses_tool_calls(function_calls, replay),
                **_usage_fields(reported_usage),
            ), stop_reason), served)
        except LLMProtocolError as error:
            if is_execution_control_error(error) or isinstance(error, GenerationProtocolError):
                raise
            raise GenerationProtocolError(
                _failed_attempt_evidence(reported_usage, stop_reason, served)
            ) from error

    def apply_request_options(
        self,
        request_kwargs: Dict[str, Any],
        options: Any,
        *,
        model: str,
    ) -> Dict[str, Any]:
        out = request_kwargs
        # OpenAI chat-completions accepts ``reasoning_effort`` directly; there
        # is no top-level ``reasoning`` argument (that is a Responses-API field).
        if getattr(options, "reasoning_effort", None):
            out["reasoning_effort"] = options.reasoning_effort

        if getattr(options, "cache_markers", None):
            # chat.completions supports prompt_cache_key (a stable string that
            # pins the cache prefix); it has no `cache_markers` field, so derive
            # the key from the markers and send only that.
            extra_body = dict(out.get("extra_body") or {})
            extra_body["prompt_cache_key"] = self._cache_key_for_options(
                model, options.cache_markers
            )
            out["extra_body"] = extra_body

        # NOTE: web_search / code_execution are intentionally NOT translated
        # here. They are Responses/Assistants-API server tools — the chat
        # completions endpoint rejects ``web_search_preview`` / ``code_interpreter``
        # tool entries. This adapter therefore does not advertise those
        # capabilities (see provider_capabilities). They can be added once this
        # adapter routes through the Responses API.

        raw = getattr(options, "raw", None)
        if isinstance(raw, dict):
            for key, value in raw.items():
                if key == "extra_body" and isinstance(value, dict):
                    out.setdefault("extra_body", {}).update(value)
                else:
                    out[key] = value
        return out

    @native_sdk_operation
    async def raw_request(
        self,
        client: Any,
        operation: str,
        payload: Optional[Any] = None,
        *,
        http_method: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        target: Any = client
        for part in operation.split("."):
            target = getattr(target, part)
        call_kwargs = dict(payload or {}) if isinstance(payload, dict) else {}
        call_kwargs.update(kwargs)
        if payload is not None and not isinstance(payload, dict):
            data = await _maybe_await(target(payload, **kwargs))
        else:
            data = await _maybe_await(target(**call_kwargs))
        if RawResponse:
            return RawResponse(operation=operation, data=data, raw=data)
        return {"operation": operation, "data": data, "raw": data}

    @native_sdk_operation
    async def list_models(self, client: Any = None) -> List[ModelInfo]:
        """List available models from OpenAI API (or any OpenAI-compatible
        endpoint the route was initialized against).

        Uses the framework-initialized ``client`` so the call goes to the
        same ``base_url`` and authenticates with the same key the route's
        ``get_response`` calls do. Routes pointed at custom endpoints
        (Azure, Kimi, DeepSeek, OpenRouter-via-OpenAI-compat) get their
        own catalog rather than silently falling through to api.openai.com.

        Args:
            client: The route's ``openai.AsyncOpenAI`` client. The
                framework always passes this; the env-var fallback below
                is only for legacy callers (existing tests, scripts that
                build a bare ``OpenAIAdapter()`` and call ``list_models()``
                directly) that pass ``None`` or omit the argument.
                Liskov-widened to optional from the SDK 0.5.0 abstract
                signature so those callers keep working.

        Returns:
            List of ModelInfo objects.
        """
        temporary_client = None
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        try:
            if client is None:
                # Legacy fallback. Modern callers always pass the route
                # client; warn and let env vars rebuild a canonical-OpenAI
                # client so we don't lose discovery for callers that
                # haven't migrated yet.
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logger.warning(
                        "OpenAIAdapter.list_models called with client=None "
                        "and no OPENAI_API_KEY; returning empty model list"
                    )
                    return []
                logger.warning(
                    "OpenAIAdapter.list_models called with client=None — "
                    "rebuilding from OPENAI_API_KEY (canonical OpenAI only)"
                )
                client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
                temporary_client = client
                retain_native_execution_http_client_close(getattr(client, "_client", None))

            response = await client.models.list()

            models = []
            for model in response.data:
                # Generate a display name from the model ID
                display_name = model.id.replace("-", " ").title()

                models.append(ModelInfo(
                    id=model.id,
                    provider="openai",
                    display_name=display_name,
                    category=ModelCategory.CHAT,  # Will be enriched by catalog service
                    supports_tools=True,  # OpenAI models support tools
                    supports_streaming=True,  # OpenAI streams every chat model
                    created_at=str(model.created) if hasattr(model, 'created') else None,
                ))

            logger.info(f"OpenAI returned {len(models)} models")
            return models

        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded while listing models: {e}")
            if is_execution_control_error(e):
                raise
            return []
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error while listing models: {e}")
            if is_execution_control_error(e):
                raise
            return []
        except openai.APIError as e:
            logger.error(f"OpenAI API error while listing models: {e}")
            if is_execution_control_error(e):
                raise
            return []
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}", exc_info=True)
            if is_execution_control_error(e):
                raise
            return []
        finally:
            if temporary_client is not None:
                await close_native_execution_stream(temporary_client)

    #: Substrings that mark an OpenAI ``/v1/models`` id as an embedding model.
    #: OpenAI's model list carries no category, so discovery filters by id.
    _EMBEDDING_ID_TOKENS = ("text-embedding", "embedding", "embed")

    #: Generic OpenAI-compatible routes expose embeddings in the SAME
    #: ``/v1/models`` listing chat discovery already fetched, so the embedding
    #: facet id-filters that listing rather than issuing a second request
    #: (#2433). OpenRouter overrides this to ``False`` — its generic listing
    #: omits embedding models, so it needs a dedicated call.
    derives_embeddings_from_chat_listing: bool = True

    @native_sdk_operation
    async def list_embedding_models(
        self, client: Any = None, chat_models: Any = None
    ) -> List["EmbeddingModelInfo"]:
        """Discover embedding models by id-prefix filtering ``/v1/models`` (#2338).

        OpenAI's model list reports no category, so the embedding facet filters
        the same ``/v1/models`` payload chat discovery uses down to
        ``text-embedding-*`` ids. OpenAI-compatible routes (Azure, RunPod vLLM,
        etc.) that pass their own client get their own embedding catalog.

        Preferred path (#2433): ``chat_models`` carries the chat model ids
        already discovered for this route, so we filter them WITHOUT a second
        network request — a chat-only route (e.g. a RunPod vLLM endpoint whose
        model-list 404s) is never probed for embeddings a second time. An empty
        ``chat_models`` list means "chat discovery found nothing", which
        correctly yields no embedding models. ``chat_models is None`` (a direct
        caller without the listing on hand) falls back to fetching ``/v1/models``
        once; failures propagate so the discovery layer can decide whether they
        are loud (route claims embedding support) or expected-unsupported.
        """
        retain_native_execution_http_client_close(getattr(client, "_client", None))
        provider = getattr(self, "name", "openai") or "openai"

        if chat_models is not None:
            return self._embedding_models_from_ids(
                (getattr(m, "id", m) for m in chat_models), provider
            )

        temporary_client = None
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "OpenAIAdapter.list_embedding_models called with client=None "
                    "and no OPENAI_API_KEY; returning empty list"
                )
                return []
            client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
            temporary_client = client
            retain_native_execution_http_client_close(getattr(client, "_client", None))

        try:
            response = await client.models.list()
            return self._embedding_models_from_ids(
                (model.id for model in response.data), provider
            )
        finally:
            if temporary_client is not None:
                await close_native_execution_stream(temporary_client)

    def _embedding_models_from_ids(
        self, ids: Any, provider: str
    ) -> List["EmbeddingModelInfo"]:
        """Filter model ids down to embedding models by id-token match (#2433)."""
        from .embedding_discovery import EmbeddingModelInfo

        results: List[EmbeddingModelInfo] = []
        for model_id in ids:
            if not model_id:
                continue
            low = str(model_id).lower()
            if not any(tok in low for tok in self._EMBEDDING_ID_TOKENS):
                continue
            results.append(EmbeddingModelInfo(
                id=model_id,
                provider=provider,
                display_name=model_id,
            ))
        logger.info(f"OpenAI: discovered {len(results)} embedding models")
        return results

    @staticmethod
    def _cache_key_for_options(model: str, markers: List[Any]) -> str:
        payload = json.dumps(
            [getattr(marker, "label", None) or getattr(marker, "index", None) for marker in markers],
            sort_keys=True,
            default=str,
        )
        return f"kestrel:{model}:{payload}"

    @staticmethod
    def _estimate_message_tokens(model: str, messages: List[Dict[str, Any]]) -> int:
        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("o200k_base")
            total = 0
            for message in messages:
                total += 3
                for key, value in message.items():
                    total += len(encoding.encode(OpenAIAdapter._token_text(value)))
                    if key == "name":
                        total += 1
            return total + 3
        except Exception:
            text = "\n".join(OpenAIAdapter._token_text(message) for message in messages)
            return max(1, len(text) // 4)

    @staticmethod
    def _token_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(OpenAIAdapter._token_text(item) for item in value)
        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                return value["text"]
            return json.dumps(value, sort_keys=True, default=str)
        return str(value)

    def _batch_request_body(self, request: Any) -> Dict[str, Any]:
        model = getattr(request, "model", None) or "gpt-5-mini"
        # Mirror get_response: apply the model's system-prompt contribution
        # (e.g. the GPT-5 behavior overlay) before normalizing, so batched
        # calls don't silently lose it.
        messages = self._apply_system_prompt_contribution(
            getattr(request, "messages", []) or [], model
        )
        body = {
            "model": model,
            "messages": self._normalize_messages(messages),
        }
        if getattr(request, "tools", None):
            body["tools"] = getattr(request, "tools")
            body["tool_choice"] = "auto"
        if getattr(request, "format", None) == "json":
            body["response_format"] = {"type": "json_object"}
        body.update(getattr(request, "kwargs", None) or {})
        return self._prepare_chat_completion_request(
            body,
            model=body["model"],
            options=getattr(request, "request_options", None),
        )

    @staticmethod
    def _batch_handle(batch: Any) -> Any:
        status_map = {
            "validating": "running",
            "in_progress": "running",
            "finalizing": "running",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
            "cancelling": "cancelled",
            "expired": "expired",
        }
        status_value = status_map.get(str(getattr(batch, "status", "") or ""), "unknown")
        status = (
            getattr(BatchStatus, status_value.upper(), status_value)
            if BatchStatus is not None
            else status_value
        )
        if BatchHandle:
            return BatchHandle(
                id=getattr(batch, "id", ""),
                status=status,
                created_at=getattr(batch, "created_at", None),
                expires_at=getattr(batch, "expires_at", None),
                raw=batch,
            )
        return {"id": getattr(batch, "id", ""), "status": status, "raw": batch}

    @staticmethod
    def _file_ref(file_obj: Any) -> Any:
        if FileRef:
            return FileRef(
                id=getattr(file_obj, "id", ""),
                filename=getattr(file_obj, "filename", None),
                purpose=getattr(file_obj, "purpose", None),
                size_bytes=getattr(file_obj, "bytes", None),
                created_at=getattr(file_obj, "created_at", None),
                raw=file_obj,
            )
        return {
            "id": getattr(file_obj, "id", ""),
            "filename": getattr(file_obj, "filename", None),
            "purpose": getattr(file_obj, "purpose", None),
            "raw": file_obj,
        }

    @staticmethod
    async def _read_file_content(content_response: Any) -> bytes:
        if isinstance(content_response, bytes):
            return content_response
        if isinstance(content_response, str):
            return content_response.encode("utf-8")
        if hasattr(content_response, "read"):
            return await _maybe_await(content_response.read())
        if hasattr(content_response, "content"):
            content = content_response.content
            return content if isinstance(content, bytes) else str(content).encode("utf-8")
        return str(content_response).encode("utf-8")

    def _llm_response_from_chat_completion(self, response_body: Any) -> LLMResponse:
        body = response_body if isinstance(response_body, dict) else _object_to_dict(response_body)
        choice = _single_choice(body.get("choices"))
        message = choice.get("message") or {}
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                fn = tc.get("function") or {}
                parsed = decode_tool_arguments(fn.get("arguments"))
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=parsed,
                    )
                )
        usage = body.get("usage") or {}
        (
            input_tokens,
            output_tokens,
            total_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
        ) = _normalized_openai_usage(usage)
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            raw=response_body,
            stop_reason=_finish_reason(choice),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize messages for OpenAI API compatibility.
        
        OpenAI requires tool_calls arguments to be JSON strings, not dicts.
        This method converts any dict arguments to JSON strings.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Normalized messages with string arguments in tool_calls
        """
        normalized = []
        for msg in messages:
            if "tool_calls" in msg and msg["tool_calls"]:
                # Deep copy the message to avoid modifying original
                new_msg = {**msg}
                new_tool_calls = []
                for tc in msg["tool_calls"]:
                    new_tc = {**tc}
                    extra_content = tc.get("extra_content")
                    if isinstance(extra_content, dict) and OPENAI_EXTRA_CONTENT in extra_content:
                        # Responses API replay data (#4198) has no meaning on
                        # chat completions; other namespaces stay as they are.
                        kept = {
                            name: value for name, value in extra_content.items()
                            if name != OPENAI_EXTRA_CONTENT
                        }
                        if kept:
                            new_tc["extra_content"] = kept
                        else:
                            new_tc.pop("extra_content")
                    if "function" in tc and "arguments" in tc["function"]:
                        args = tc["function"]["arguments"]
                        if isinstance(args, dict):
                            new_tc["function"] = {
                                **tc["function"],
                                "arguments": json.dumps(args)
                            }
                    new_tool_calls.append(new_tc)
                new_msg["tool_calls"] = new_tool_calls
                normalized.append(new_msg)
            else:
                normalized.append(msg)
        return normalized

    # ---- Provider metadata (SDK 0.6.0) -------------------------------------

    def substrate_type(self) -> Optional[str]:
        return "gpt"

    def display_name(self) -> Optional[str]:
        return "OpenAI"

    def key_env_var(self) -> Optional[str]:
        return "OPENAI_API_KEY"

    # cost_per_1m_tokens: omitted — OpenAI pricing varies dramatically by
    # model (gpt-5-mini vs o3 vs gpt-3.5-turbo), so per-model pricing
    # belongs on ModelInfo rather than the adapter. Council costing falls
    # back to the framework's conservative default when this returns None.

    # deliberation_style: omitted — OpenAI spans the speed/cost spectrum
    # (gpt-5-mini fast, o3 careful), so a single-adapter hint isn't
    # meaningful. Council leaves this routing decision to model-level
    # heuristics.
