"""The OpenAI Responses API path for models that need it for tools (#4198).

``gpt-6.2-sol`` refuses function tools on ``/v1/chat/completions`` while
reasoning is on, and it does not accept reasoning effort ``none``: the
provider directs tool-bearing calls to ``/v1/responses``. A native OpenAI
route therefore sends a function-tool-bearing request for such a model to the
Responses API, statelessly (``store=false``) with encrypted reasoning
included, through :class:`~kestrel_sovereign.llm.openai_adapter.OpenAIAdapter`'s
ordinary ``get_response`` and ``get_streaming_response_with_tools``.

Both entry points consume the same event stream through one
:class:`ResponsesEventState`, so a non-streaming call keeps the model and
response id ``response.created`` named even when the transport fails later.

A tool continuation is served from the turn's
:class:`~kestrel_sovereign.llm.responses_continuation.ResponsesSnapshot`,
never rebuilt from the agent's message list (see that module). The request
that continues a snapshot is the snapshot's items, unchanged, followed by
what the message list gained after the reply the snapshot ends with:

- ``function_call_output`` items for that reply's tool calls;
- system or user messages (an operator turn, a repair prompt);
- one developer ``additional_tools`` item declaring tools exposed since the
  snapshot's request, which are not added to ``tools``. OpenAI documents this
  as the way to add tools to a conversation without changing its prefix, and
  that the item is replayed in the position it was first sent
  (https://developers.openai.com/api/docs/guides/keep-prefix-unchanged,
  https://developers.openai.com/api/reference/resources/responses/methods/create).

Tools the registry stopped offering stay declared: a declaration is never
removed. A continuation that offers fewer tools than its chain declared
restricts the model to the ones it offers with ``tool_choice`` (``none``, or
``allowed_tools``), the restriction the same guide documents in place of
changing ``tools``. Which calls run is still decided where it always was, by
the tool dispatcher's allowlist and permission checks.

A request goes to the Responses API when it carries function tools, when its
messages carry tool history, or when the turn holds a snapshot on its route,
live or retired. A turn that has a snapshot on a route therefore never leaves
the Responses API there: a tool-free continuation keeps its chain, and one
that cannot continue it fails closed on the Responses API, never on chat
completions.

Tools have one source: the typed ``tools`` parameter. On this route a
request whose merged body (every raw-option channel applied) carries
``extra_body["tools"]`` is refused before anything is sent, even when it
repeats the typed declaration exactly; declarations are never merged or
de-duplicated (:func:`refuse_raw_tool_declarations`). Other routes pass raw
options through unchanged.

A snapshot is bound to the configured route, the model requested, the reply's
response id and served model, and :func:`request_digest`: a digest of the
merged provider-visible request body, ``extra_body`` included, without its
input items, its transport-only ``stream`` flag, and the per-request tool
offer (``tools`` the caller offers now and the ``tool_choice`` that narrows
the chain's declarations to them). A continuation whose digest differs ends
the chain.

Pruning wins. A continuation must still carry, unchanged, every item its
chain's requests built from the caller's messages, and every reply the chain
absorbed as the caller received it: a reply's tool call ids find the
assistant message replaying it, and that message must still hold the
reply's text and each call's name and arguments, compared by value. When the
caller dropped or rewrote any of it, text it excluded from a reply whose
calls it kept included, the tool loop pruned or excluded it and replaying
the snapshot would restore it, so the chain ends at boundary
``prefix_pruned``; one that only inserted items ends at ``prefix_changed``.
A continuation whose replay would exceed the context budget the tool loop
pruned its own messages to ends at ``budget_exceeded``.

The one exception is the turn notices that close the chain's first request,
system or developer messages directly after its last user message (an
operator notice): the tool loop never carries those back, so the snapshot
replays them in place. The exception rests on role and position alone, so a
caller cannot mark such a message removed: one it leaves out is replayed,
and one it still carries but changed ends the chain. The core turn never
removes one; its inline notice is never stored as a conversation row, so no
exclusion can name it.

A request that carries tool history no snapshot of the turn accounts for (no
turn, another route or model, a provider that served a different model) is
also a continuation boundary (``no_snapshot`` or ``outside_turn``); one that
continues a chain the turn retired is ``retired``, with why. A boundary
replays no encrypted reasoning and no tool call or result: it is sent as an
ordinary conversation, the caller's current text messages only, and its
``provider_served`` evidence records the boundary and why.

A reply served by a model other than the one its chain was served by is
returned and acted on: it is the provider's genuine answer to the request,
and Kestrel cannot know why the provider routed it. But nothing it carries is
replayed: its chain ends at that reply and its ``provider_served`` evidence
names both models. A reply whose provider reports a ``prefix_mismatch`` (it
dropped replayed input) is handled the same way: it is used, and its degraded
chain is never replayed again. That chain ends the moment the terminal
response reporting the mismatch arrives, so a reply that then fails, or whose
connection drops before the stream ends cleanly, still ends it, and its error
carries the mismatch.

``provider_served.continuation`` describes the request as sent;
``provider_served.snapshot`` says whether its reply's snapshot was kept, and
if not, why.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kestrel_sdk.llm import LLMProtocolError, ToolCallStarted

from .adapter import LLMResponse, ThinkingDelta, ToolCall
from .completion_protocol import decode_tool_arguments
from .output_ceiling import IncompleteGenerationError, attach_stop_reason
from .provider_served import ProviderAttemptReceipt, ProviderServed, attach_provider_served
from .responses_continuation import (
    BUDGET_EXCEEDED,
    DUPLICATE_TOOL_OUTPUT,
    PARENT_RETIRED,
    PREFIX_CHANGED,
    PREFIX_MISMATCH,
    PREFIX_PRUNED,
    REQUEST_SETTINGS_CHANGED,
    SERVED_MODEL_MISMATCH,
    UNANCHORED_ASSISTANT_MESSAGE,
    UNIDENTIFIED_RESPONSE,
    UNKNOWN_TOOL_OUTPUT,
    UNSUPPORTED_MESSAGE_ROLE,
    ResponsesAnchor,
    ResponsesContinuations,
    ResponsesSnapshot,
    RetiredSnapshot,
    current_responses_continuations,
)

logger = logging.getLogger(__name__)

#: Models whose function-tool requests a native OpenAI route sends to the
#: Responses API. Matches qualified ids, dated snapshots, fine-tunes and
#: hyphenated variants of the family (``gpt-6.2-sol-mini``), but not a longer
#: family name that merely starts the same way (``gpt-6.2-solar``). A variant
#: is matched on purpose: the Responses API serves every model chat
#: completions does, while chat completions refuses this family's function
#: tools with reasoning on, so a variant sent to the wrong one fails.
RESPONSES_FUNCTION_TOOL_MODEL_PATTERN = re.compile(
    r"(?:^|[/:])gpt-6\.2-sol(?:[.:-]|$)",
    re.IGNORECASE,
)

#: The value ``provider_served.api`` takes for this path.
RESPONSES_API = "responses"

#: What the request asks to be included, so reasoning items carry the
#: encrypted content a stateless replay needs.
_INCLUDE = ("reasoning.encrypted_content",)

#: Chat-completions request keys copied unchanged onto a Responses request.
_PASSTHROUGH_PARAMS = frozenset({
    "temperature",
    "top_p",
    "parallel_tool_calls",
    "metadata",
    "service_tier",
    "user",
    "safety_identifier",
    "prompt_cache_key",
})

#: Keys only this module decides; a caller setting one is refused.
_RESERVED_PARAMS = frozenset({
    "input",
    "instructions",
    "store",
    "include",
    "previous_response_id",
    "conversation",
    "stream",
    "model",
})

#: Output items whose text is shown to the user.
_TEXT_PART_TYPES = frozenset({"output_text", "refusal"})


class ResponsesAPIFailure(IncompleteGenerationError):
    """The Responses API reported that a response failed or did not finish.

    Carries the provider's own code and message, the usage it reported
    (:attr:`usage`, read by :func:`~kestrel_sovereign.llm.output_ceiling.reported_usage`)
    and the provider's identification of the response, both attached before
    the error is raised.
    """

    summary = "reported a failed response"

    def __init__(
        self,
        *,
        model: str,
        status: str,
        code: Optional[str],
        message: Optional[str],
        usage: Optional[LLMResponse],
        served: ProviderServed,
    ) -> None:
        self.status = status
        self.code = code
        self.provider_message = message
        detail = ": ".join(part for part in (code, message) if part) or "no detail"
        super().__init__(
            f"OpenAI Responses API reported status {status!r} for model "
            f"{model!r}: {detail}"
        )
        self.usage = attach_provider_served(usage, served)
        attach_provider_served(self, served)


class ResponsesStreamInterrupted(IncompleteGenerationError):
    """The stream broke after the terminal response arrived, before its end.

    The provider finished and reported its usage, but the transport failed
    before the stream closed cleanly, so the attempt did not finish. The usage
    the terminal response reported is kept on :attr:`usage` (``None`` when it
    reported none, which is recorded as unknown, never as zero) along with
    the provider's identification of the response. Raised from the transport
    error.
    """

    summary = "lost its stream after the provider's final response"

    def __init__(self, *, model: str, usage: Optional[LLMResponse], served: ProviderServed) -> None:
        super().__init__(
            f"The OpenAI Responses stream for model {model!r} failed after its "
            "terminal response, before the stream ended"
        )
        self.usage = attach_provider_served(usage, served)
        attach_provider_served(self, served)


@dataclass(frozen=True)
class ResponsesRoute:
    """The route a request is sent on, as a snapshot is bound to it.

    ``label`` is the configured route's name (``openai:primary``), or the
    adapter's name and base URL for an adapter built outside the provider
    registry. ``adapter`` and ``client`` are the route's own: the provider
    registry builds one of each per configured route, so two routes that
    share a vendor and base URL (and possibly a different account or
    organization) are still two routes, even if they were given one name. A
    snapshot holds both, so a route rebuilt mid-turn is a new route, never a
    reused id.
    """

    label: str
    client: object
    adapter: object = None

    def serves(self, snapshot: Any) -> bool:
        """Whether ``snapshot`` (live or retired) was captured on this route."""
        return (
            snapshot.route == self.label
            and snapshot.client is self.client
            and snapshot.adapter is self.adapter
        )


#: The ``extra_body`` key this route refuses (see :func:`refuse_raw_tool_declarations`).
_RAW_TOOLS_KEY = "tools"


def refuse_raw_tool_declarations(request: Mapping[str, Any]) -> None:
    """Refuse tools declared through ``extra_body`` on this route.

    ``request`` is the merged request, every raw-option channel applied. The
    typed ``tools`` parameter is the one source of declarations: a second one
    would make the request's tools whichever merge the SDK applied last, and
    the snapshot would replay declarations nobody chose. A declaration
    identical to a typed one is refused too; declarations are never merged or
    de-duplicated. Only the ``tools`` key is refused.
    """
    extra_body = request.get("extra_body")
    if isinstance(extra_body, Mapping) and _RAW_TOOLS_KEY in extra_body:
        raise ValueError(
            "The OpenAI Responses route does not accept tools through "
            "extra_body['tools']: declare every tool through the typed `tools` "
            "parameter, which is the only tool source on this route (#4198)."
        )


def requires_responses_api(
    *,
    native_openai: bool,
    model: str,
    request: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    route: ResponsesRoute,
) -> bool:
    """Whether this request must go to the Responses API.

    Decided from the request as it will be sent (``request`` is the merged
    request) and from its messages: function tools, tool history, or a turn
    that holds a snapshot on this route, live or retired. A continuation
    offered no tools is still a continuation, and tool history without a
    snapshot must meet the recorded boundary rather than be replayed on chat
    completions.

    Raises :class:`ValueError` for tools declared through ``extra_body``
    (:func:`refuse_raw_tool_declarations`), before anything is sent.
    """
    if not native_openai or not RESPONSES_FUNCTION_TOOL_MODEL_PATTERN.search(model):
        return False
    refuse_raw_tool_declarations(request)
    if any(_is_function_tool(tool) for tool in _chat_tools(request)):
        return True
    if any(_has_tool_history(message) for message in messages):
        return True
    registry = current_responses_continuations()
    return registry is not None and any(route.serves(key) for key in registry.routes())


def _chat_tools(request: Mapping[str, Any]) -> List[Any]:
    tools = request.get("tools")
    if tools is None:
        return []
    if isinstance(tools, (list, tuple)):
        return list(tools)
    raise ValueError(f"tools must be a list of tool declarations, not {type(tools).__name__}.")


def _is_function_tool(tool: Any) -> bool:
    return isinstance(tool, Mapping) and tool.get("type") == "function"


def _json_text(value: Any) -> str:
    """The canonical text a captured item or setting is stored as."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def _responses_tool(tool: Any) -> Dict[str, Any]:
    if not _is_function_tool(tool):
        raise ValueError(
            "The OpenAI Responses path accepts function tools only; got "
            f"{tool.get('type') if isinstance(tool, Mapping) else type(tool).__name__!r}."
        )
    function = tool.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise ValueError("A function tool must name its function.")
    translated: Dict[str, Any] = {"type": "function", "name": function["name"]}
    if function.get("description") is not None:
        translated["description"] = function["description"]
    translated["parameters"] = function.get("parameters")
    # The Responses API makes function tools strict unless told otherwise;
    # chat-completions tools are not, and Kestrel's schemas are not written
    # for strict mode.
    translated["strict"] = bool(function.get("strict", False))
    return translated


def _tool_choice(choice: Any) -> Any:
    if isinstance(choice, Mapping) and choice.get("type") == "function":
        function = choice.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            return {"type": "function", "name": function["name"]}
    return choice


def _text_format(response_format: Any) -> Dict[str, Any]:
    if not isinstance(response_format, Mapping):
        raise ValueError("response_format must be an object.")
    kind = response_format.get("type")
    if kind == "json_schema":
        schema = response_format.get("json_schema") or {}
        return {"format": {
            "type": "json_schema",
            "name": schema.get("name"),
            "schema": schema.get("schema"),
            "strict": bool(schema.get("strict", False)),
        }}
    if kind in ("json_object", "text"):
        return {"format": {"type": kind}}
    raise ValueError(f"Unsupported response_format type {kind!r} on the Responses path.")


@dataclass(frozen=True)
class _Translated:
    settings: Dict[str, Any]
    params: Dict[str, Any]


#: The request's output budget, under every name a caller may give it.
_OUTPUT_BUDGET_KEYS = ("max_completion_tokens", "max_tokens", "max_output_tokens")


class _OneValue:
    """Values that may be given both as a request key and in ``extra_body``.

    The SDK merges ``extra_body`` over the request body, so a key given in
    both places would be sent as whichever the merge left. Two different
    values are refused instead: only one of them can be what was meant.
    """

    def __init__(self) -> None:
        self._values: Dict[str, Tuple[str, Any]] = {}

    def give(self, name: str, value: Any, *, source: str) -> None:
        if value is None:
            return
        given = self._values.get(name)
        if given is not None and given[1] != value:
            raise ValueError(
                f"The request gives {name!r} twice, as {given[0]} and as "
                f"{source}, with different values."
            )
        self._values[name] = (source, value)

    def get(self, name: str) -> Any:
        given = self._values.get(name)
        return None if given is None else given[1]


def _merged_object(name: str, base: Optional[Mapping[str, Any]], raw: Any) -> Dict[str, Any]:
    """``base`` with the fields of an ``extra_body`` object of the same name."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"extra_body {name!r} must be an object.")
    merged = dict(base or {})
    for key, value in raw.items():
        if key in merged and merged[key] != value:
            raise ValueError(
                f"extra_body {name}.{key} conflicts with the request's own {name}.{key}."
            )
        merged[key] = value
    return merged


def _translate_request(request: Mapping[str, Any]) -> _Translated:
    """Split chat-completions request kwargs into Responses settings.

    ``settings`` are the parts of the request a changed value would change the
    reasoning prefix for; ``params`` are the rest. Anything this path cannot
    express is refused rather than dropped.

    ``extra_body`` is read as part of the request, not merged over it later:
    each key in it is translated exactly like the request key of that name, so
    ``extra_body`` ``text`` and ``reasoning`` are captured settings like the
    request's own response format and reasoning effort. Keys this path does
    not know stay in a captured ``extra_body`` setting, part of the prefix, and
    none of them can name a field the request body already sets. Tools come
    from the typed ``tools`` only (:func:`refuse_raw_tool_declarations`).
    """
    refuse_raw_tool_declarations(request)
    extra_body = dict(request.get("extra_body") or {})
    for key in extra_body:
        if key in _RESERVED_PARAMS:
            raise ValueError(f"The OpenAI Responses path sets {key!r} itself.")
    tools = [_responses_tool(tool) for tool in _chat_tools(request)]

    values = _OneValue()
    values.give("reasoning_effort", request.get("reasoning_effort"), source="a request key")
    values.give("reasoning_effort", extra_body.pop("reasoning_effort", None), source="extra_body")
    values.give("response_format", request.get("response_format"), source="a request key")
    values.give("response_format", extra_body.pop("response_format", None), source="extra_body")
    values.give("tool_choice", request.get("tool_choice"), source="a request key")
    values.give("tool_choice", extra_body.pop("tool_choice", None), source="extra_body")
    for budget_key in _OUTPUT_BUDGET_KEYS:
        values.give("max_output_tokens", request.get(budget_key), source=budget_key)
        values.give("max_output_tokens", extra_body.pop(budget_key, None), source=f"extra_body {budget_key}")
    for key in _PASSTHROUGH_PARAMS:
        values.give(key, request.get(key), source="a request key")
        values.give(key, extra_body.pop(key, None), source="extra_body")

    settings: Dict[str, Any] = {
        "tools": tools,
        "include": list(_INCLUDE),
        "store": False,
    }
    effort = values.get("reasoning_effort")
    reasoning = {"effort": effort} if effort is not None else None
    if "reasoning" in extra_body:
        reasoning = _merged_object("reasoning", reasoning, extra_body.pop("reasoning"))
    if reasoning is not None:
        settings["reasoning"] = reasoning
    response_format = values.get("response_format")
    text = _text_format(response_format) if response_format is not None else None
    if "text" in extra_body:
        text = _merged_object("text", text, extra_body.pop("text"))
    if text is not None:
        settings["text"] = text
    if extra_body:
        settings["extra_body"] = extra_body

    params: Dict[str, Any] = {}
    if values.get("tool_choice") is not None:
        params["tool_choice"] = _tool_choice(values.get("tool_choice"))
    if values.get("max_output_tokens") is not None:
        params["max_output_tokens"] = values.get("max_output_tokens")
    for key in sorted(_PASSTHROUGH_PARAMS):
        if values.get(key) is not None:
            params[key] = values.get(key)

    handled = {
        "tools", "tool_choice", "response_format", "reasoning_effort",
        "extra_body", *_OUTPUT_BUDGET_KEYS, *_PASSTHROUGH_PARAMS,
    }
    for key in request:
        if key in handled:
            continue
        if key in _RESERVED_PARAMS:
            raise ValueError(f"The OpenAI Responses path sets {key!r} itself.")
        raise ValueError(
            f"Request parameter {key!r} has no equivalent on the OpenAI "
            "Responses API."
        )
    return _Translated(settings=settings, params=params)


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    raise ValueError(f"Unsupported message content of type {type(content).__name__}.")


def _input_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"Unsupported message content of type {type(content).__name__}.")
    parts: List[Dict[str, Any]] = []
    for part in content:
        kind = part.get("type") if isinstance(part, Mapping) else None
        if kind in ("text", "input_text") and isinstance(part.get("text"), str):
            parts.append({"type": "input_text", "text": part["text"]})
        elif kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, Mapping) else image
            detail = image.get("detail") if isinstance(image, Mapping) else None
            parts.append({
                "type": "input_image", "image_url": url, "detail": detail or "auto",
            })
        else:
            raise ValueError(f"Unsupported message content part {kind!r}.")
    return parts


def _message_item(message: Mapping[str, Any]) -> Dict[str, Any]:
    role = message.get("role")
    if role == "assistant":
        return {"type": "message", "role": "assistant", "content": _text_of(message.get("content"))}
    return {"type": "message", "role": role, "content": _input_content(message.get("content"))}


def _tool_output(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _text_of(content)
    return _json_text(content)


def _has_tool_history(message: Mapping[str, Any]) -> bool:
    return message.get("role") == "tool" or (
        message.get("role") == "assistant" and bool(message.get("tool_calls"))
    )


def _ordinary_items(messages: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """``messages`` as an ordinary conversation: text messages only.

    Returns the items and how many tool calls and tool results were left out.
    """
    items: List[Dict[str, Any]] = []
    omitted = 0
    for message in messages:
        role = message.get("role")
        if role == "tool":
            omitted += 1
            continue
        if role == "assistant":
            omitted += len(message.get("tool_calls") or ())
            if not _text_of(message.get("content")):
                continue
        elif role not in ("system", "developer", "user"):
            raise ValueError(f"Unsupported message role {role!r} on the Responses path.")
        items.append(_message_item(message))
    return items, omitted


def _turn_request(messages: Sequence[Mapping[str, Any]]) -> Optional[str]:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _json_text(message.get("content"))
    return None


def _notices_from(items: Sequence[Mapping[str, Any]]) -> int:
    """Where the turn notices start in a chain's first request's ``items``.

    The turn notices are the system or developer messages after the request's
    last user message (an operator notice the turn appended). The tool loop
    rebuilds its continuations without them, so they are the one part of a
    captured prefix a continuation need not carry. ``len(items)`` when there
    are none.
    """
    start = len(items)
    while start > 0 and items[start - 1].get("type") == "message" and (
        items[start - 1].get("role") in ("system", "developer")
    ):
        start -= 1
    # Only notices that directly follow the user message are the turn's own;
    # one after a later assistant message was appended by the tool loop.
    if start == len(items) or start == 0 or items[start - 1].get("role") != "user":
        return len(items)
    return start


class _Boundary(Exception):
    """A matched snapshot whose continuation would not be append-only."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _call_ids(message: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(
        call.get("id") for call in (message.get("tool_calls") or ())
        if isinstance(call, Mapping)
    )


def _calls_digest(calls: Sequence[Tuple[Any, Any, Any]]) -> Optional[str]:
    """The digest of ``(id, name, arguments)`` tool calls, compared by value.

    Arguments are compared as the JSON object they encode, so the tool loop's
    re-encoding of the provider's argument text (a dict, keys in any order)
    is the same call and a changed value is not. ``None`` when an argument
    encodes no JSON object: such a call carries nothing a reply holds. That
    includes missing and falsy arguments (``None``, ``""``, ``[]``, ``0``,
    ``False``), which are not read as ``{}``: a reply whose call took no
    arguments was replayed with ``{}``, and anything else in its place is a
    rewrite.
    """
    canonical: List[Dict[str, Any]] = []
    for call_id, name, arguments in calls:
        try:
            decoded = decode_tool_arguments(arguments)
        except LLMProtocolError:
            return None
        canonical.append({"id": call_id, "name": name, "arguments": decoded})
    text = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _message_calls(message: Mapping[str, Any]) -> List[Tuple[Any, Any, Any]]:
    calls: List[Tuple[Any, Any, Any]] = []
    for call in message.get("tool_calls") or ():
        if not isinstance(call, Mapping):
            calls.append((None, None, None))
            continue
        function = call.get("function")
        function = function if isinstance(function, Mapping) else {}
        calls.append((call.get("id"), function.get("name"), function.get("arguments")))
    return calls


def _matches(message: Mapping[str, Any], anchor: ResponsesAnchor) -> bool:
    if message.get("role") != "assistant":
        return False
    if anchor.call_ids:
        return _call_ids(message) == anchor.call_ids
    return not message.get("tool_calls") and _text_of(message.get("content")) == anchor.content


def _carries_reply(message: Mapping[str, Any], anchor: ResponsesAnchor) -> bool:
    """Whether ``message``, found by ``anchor``, still holds the reply as the
    caller received it: its text and every tool call, compared by value.

    A tool call id only finds the message. A caller that kept a reply's calls
    may still have excluded or rewritten the text beside them, or a call's
    arguments, and replaying the reply would restore what it removed.
    """
    digest = _calls_digest(_message_calls(message))
    return (
        digest is not None and digest == anchor.calls_digest
        and _text_of(message.get("content")) == anchor.content
    )


def _locate(
    chain: "ResponsesSnapshot | RetiredSnapshot", messages: Sequence[Mapping[str, Any]],
) -> Optional[List[int]]:
    """The indexes of the messages replaying each of the chain's replies, in order.

    Every reply the chain absorbed must still be present, in order. A chain
    whose replies carried no tool call is confirmed by its first user message.
    """
    before = len(messages)
    positions: List[int] = []
    for anchor in reversed(chain.anchors):
        found = next(
            (index for index in range(before - 1, -1, -1) if _matches(messages[index], anchor)),
            None,
        )
        if found is None:
            return None
        positions.append(found)
        before = found
    if not positions:
        return None
    positions.reverse()
    if not any(anchor.call_ids for anchor in chain.anchors):
        if chain.turn_request is None or not any(
            message.get("role") == "user"
            and _json_text(message.get("content")) == chain.turn_request
            for message in messages[:positions[0]]
        ):
            return None
    return positions


def _appended_items(
    tail: Sequence[Mapping[str, Any]], call_ids: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    """What the message list gained after one of the chain's replies."""
    items: List[Dict[str, Any]] = []
    answered: set[str] = set()
    for message in tail:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in call_ids:
                raise _Boundary(UNKNOWN_TOOL_OUTPUT)
            if call_id in answered:
                raise _Boundary(DUPLICATE_TOOL_OUTPUT)
            answered.add(call_id)
            items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": _tool_output(message.get("content")),
            })
        elif role in ("system", "developer", "user"):
            items.append(_message_item(message))
        elif role == "assistant":
            raise _Boundary(UNANCHORED_ASSISTANT_MESSAGE)
        else:
            raise _Boundary(UNSUPPORTED_MESSAGE_ROLE)
    return items


def _texts(items: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    return tuple(_json_text(item) for item in items)


def _carried_texts(items: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    """``items`` as compared for what a continuation still carries.

    By content, not shape: a message whose content is text parts only is the
    same message as one whose content is their text (a subagent sends its
    task as parts and rebuilds it as a string). Anything else, an image
    included, is compared exactly.
    """
    keys: List[str] = []
    for item in items:
        content = item.get("content")
        if item.get("type") == "message" and isinstance(content, list) and all(
            isinstance(part, Mapping) and part.get("type") == "input_text"
            and isinstance(part.get("text"), str)
            for part in content
        ):
            item = {**item, "content": "".join(part["text"] for part in content)}
        keys.append(_json_text(item))
    return tuple(keys)


def _carried_now(
    snapshot: ResponsesSnapshot,
    messages: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
) -> List[Tuple[str, ...]]:
    """What ``messages`` carries before each of the chain's replies, as items.

    Built the way the chain's requests built them: the first from the whole
    list before the first reply, as an ordinary conversation; each later one
    from what follows the previous reply.
    """
    first, _ = _ordinary_items(messages[:positions[0]])
    carried = [_carried_texts(first)]
    for index in range(1, len(positions)):
        try:
            between = _appended_items(
                messages[positions[index - 1] + 1:positions[index]],
                snapshot.anchors[index - 1].call_ids,
            )
        except _Boundary:
            # A message the chain never sent sits inside its prefix.
            raise _Boundary(PREFIX_CHANGED) from None
        carried.append(_carried_texts(between))
    return carried


def _pruned_chain(
    registry: ResponsesContinuations,
    route: ResponsesRoute,
    model: str,
    messages: Sequence[Mapping[str, Any]],
) -> Optional[ResponsesSnapshot]:
    """The newest live snapshot whose latest reply ``messages`` still carry,
    though not every earlier one: the caller pruned part of its prefix.

    Only replies with tool calls count; their call ids are the provider's
    own, so a match is never a coincidence of text.
    """
    for snapshot in registry.snapshots():
        if not route.serves(snapshot) or snapshot.requested_model != model:
            continue
        latest = snapshot.anchors[-1] if snapshot.anchors else None
        if latest is not None and latest.call_ids and any(
            _matches(message, latest) for message in messages
        ):
            return snapshot
    return None


def _redefined(snapshot: ResponsesSnapshot, live_tools: Sequence[Mapping[str, Any]]) -> bool:
    """Whether a tool offered now redefines one the chain declared."""
    declared = dict(zip(snapshot.declared_tools, snapshot.tool_definitions))
    return any(
        tool["name"] in declared and _json_text(tool) != declared[tool["name"]]
        for tool in live_tools
    )


def _is_subsequence(required: Sequence[str], present: Sequence[str]) -> bool:
    remaining = iter(present)
    return all(any(item == candidate for candidate in remaining) for item in required)


def _prefix_change(snapshot: ResponsesSnapshot, carried: Sequence[Tuple[str, ...]]) -> Optional[str]:
    """Why ``carried`` cannot continue ``snapshot``, or ``None`` if it can.

    ``prefix_pruned`` when an item the snapshot holds is no longer carried
    unchanged (the tool loop pruned or stubbed it, or the caller excluded
    it): replaying the snapshot would restore it. ``prefix_changed`` when
    every one is carried but the caller inserted others among them.
    """
    if len(snapshot.carried) != len(carried):
        return PREFIX_CHANGED
    for index, (captured, present) in enumerate(zip(snapshot.carried, carried)):
        required = captured[:snapshot.notices_from] if index == 0 else captured
        if not _is_subsequence(required, present):
            return PREFIX_PRUNED
        if present != captured and present != required:
            return PREFIX_CHANGED
    return None


def _continued(
    registry: ResponsesContinuations,
    route: ResponsesRoute,
    model: str,
    messages: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[ResponsesSnapshot, List[int]]]:
    """The newest snapshot on ``route`` for ``model`` that ``messages`` continue,
    with the indexes of the messages replaying its replies."""
    for snapshot in registry.snapshots():
        if not route.serves(snapshot) or snapshot.requested_model != model:
            continue
        positions = _locate(snapshot, messages)
        if positions is not None:
            return snapshot, positions
    return None


def _retired_chain(
    registry: ResponsesContinuations,
    route: ResponsesRoute,
    model: str,
    messages: Sequence[Mapping[str, Any]],
) -> Optional[RetiredSnapshot]:
    """The newest retired chain on ``route`` for ``model`` that ``messages`` continue."""
    for record in registry.retired():
        if route.serves(record) and record.requested_model == model and _locate(record, messages) is not None:
            return record
    return None


@dataclass(frozen=True)
class ResponsesRequest:
    """One serialized Responses request and what its reply will extend."""

    body: Dict[str, Any]
    route: ResponsesRoute
    model: str
    items: Tuple[str, ...]
    settings: str
    request_digest: str
    declared_tools: Tuple[str, ...]
    tool_definitions: Tuple[str, ...]
    anchors: Tuple[ResponsesAnchor, ...]
    carried: Tuple[Tuple[str, ...], ...]
    notices_from: int
    turn_request: Optional[str]
    continuation: Mapping[str, Any]
    parent: Optional[ResponsesSnapshot]
    registry: Optional[ResponsesContinuations]

    @property
    def caller_output_budget(self) -> bool:
        return self.body.get("max_output_tokens") is not None


def _digest(model: str, translated: _Translated) -> str:
    body: Dict[str, Any] = {"model": model}
    body.update({key: value for key, value in translated.settings.items() if key not in ("tools", "extra_body")})
    body.update({key: value for key, value in translated.params.items() if key != "tool_choice"})
    # The SDK merges ``extra_body`` over the body; translation keeps it
    # disjoint from the body's own keys, so the merge is a union.
    body.update(translated.settings.get("extra_body") or {})
    canonical = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def request_digest(*, model: str, request: Mapping[str, Any]) -> str:
    """The digest a chain's requests are bound to.

    The SHA-256 of the merged provider-visible request body (``extra_body``
    merged in, as the SDK sends it) without the input items, which are
    checked item by item, the transport-only ``stream`` flag, and the
    per-request tool offer: the tools ``request`` offers now and the
    ``tool_choice`` narrowing the chain's declarations to them, which a
    continuation may change without changing its prefix. Any other field
    that differs between two requests of a chain ends the chain.
    """
    return _digest(model, _translate_request(request))


def _narrowed_tool_choice(choice: Any, *, offered: Sequence[str], declared: Sequence[str]) -> Any:
    """``choice`` restricted to the tools this request offers.

    A continuation keeps every tool its chain declared, and the caller may
    offer fewer now: none at all, or a registry that withdrew one. OpenAI
    documents ``tool_choice`` with ``allowed_tools`` as the way to restrict a
    turn's tools instead of changing ``tools``
    (https://developers.openai.com/api/docs/guides/keep-prefix-unchanged).
    A choice that names a tool the request does not offer is refused.
    """
    offered_set = set(offered)
    if offered_set == set(declared) or choice == "none":
        return choice
    if isinstance(choice, Mapping) and choice.get("type") == "function":
        if choice.get("name") in offered_set:
            return choice
        raise ValueError(
            f"tool_choice names {choice.get('name')!r}, which this request does not offer."
        )
    if isinstance(choice, Mapping) and choice.get("type") == "allowed_tools":
        names = {tool.get("name") for tool in choice.get("tools") or () if isinstance(tool, Mapping)}
        if not names <= offered_set:
            raise ValueError("tool_choice allows tools this request does not offer.")
        return choice
    if choice not in (None, "auto", "required"):
        raise ValueError(f"Unsupported tool_choice {choice!r} on the Responses path.")
    if not offered:
        if choice == "required":
            raise ValueError("tool_choice 'required' on a request that offers no tools.")
        return "none"
    return {
        "type": "allowed_tools",
        "mode": "required" if choice == "required" else "auto",
        "tools": [{"type": "function", "name": name} for name in offered],
    }


def request_text_chars(body: Mapping[str, Any]) -> int:
    """The characters of text one Responses request carries.

    The unit of the tool loop's context budget
    (``OrchestratorEngineMixin._prune_orchestrator_messages``): message text,
    tool-call arguments, tool results and tool declarations. Encrypted
    reasoning is opaque and counted by no caller's budget, so it is not
    counted here either.
    """
    total = sum(len(_json_text(tool)) for tool in body.get("tools") or ())
    for item in body.get("input") or ():
        kind = item.get("type")
        if kind == "message":
            total += len(_text_of(item.get("content")))
        elif kind == "function_call":
            total += len(item.get("arguments") or "")
        elif kind == "function_call_output":
            total += len(_tool_output(item.get("output")))
        elif kind == "additional_tools":
            total += sum(len(_json_text(tool)) for tool in item.get("tools") or ())
    return total


#: ``continuation.reason`` for a request that continues a chain the turn retired.
RETIRED = "retired"

#: ``continuation.status`` for a request that continues no snapshot and
#: crosses no boundary: it sends the caller's messages as they are.
FRESH = "fresh"


def prepare_responses_request(
    *,
    route: ResponsesRoute,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
) -> ResponsesRequest:
    """Serialize one request: a snapshot continuation, or an ordinary one."""
    translated = _translate_request(request)
    digest = _digest(model, translated)
    registry = current_responses_continuations()
    live_tools = translated.settings["tools"]
    offered = [tool["name"] for tool in live_tools]

    boundary: Optional[Dict[str, Any]] = None
    found = _continued(registry, route, model, messages) if registry is not None else None
    if found is not None:
        parent, positions = found
        try:
            if not all(
                _carries_reply(messages[index], anchor)
                for anchor, index in zip(parent.anchors, positions)
            ):
                # The caller kept a reply's tool calls but not all of the
                # reply: replaying the snapshot would restore what it left out.
                raise _Boundary(PREFIX_PRUNED)
            change = _prefix_change(parent, _carried_now(parent, messages, positions))
            if change is not None:
                raise _Boundary(change)
            appended = _appended_items(messages[positions[-1] + 1:], parent.anchors[-1].call_ids)
            if digest != parent.request_digest or _redefined(parent, live_tools):
                raise _Boundary(REQUEST_SETTINGS_CHANGED)
        except _Boundary as refused:
            boundary = {"reason": refused.reason}
        else:
            settings = json.loads(parent.settings)
            carried = parent.carried + (_carried_texts(appended),)
            declared = list(parent.declared_tools)
            definitions = list(parent.tool_definitions)
            exposed = [tool for tool in live_tools if tool["name"] not in declared]
            if exposed:
                appended.append({"type": "additional_tools", "role": "developer", "tools": exposed})
                declared.extend(tool["name"] for tool in exposed)
                definitions.extend(_json_text(tool) for tool in exposed)
            params = dict(translated.params)
            choice = _narrowed_tool_choice(params.get("tool_choice"), offered=offered, declared=declared)
            if choice is not None:
                params["tool_choice"] = choice
            continuation: Dict[str, Any] = {
                "status": "continued",
                "previous_response_id": parent.response_id,
                "appended_items": len(appended),
            }
            if exposed:
                continuation["additional_tools"] = [tool["name"] for tool in exposed]
            prepared = _finish_request(
                settings=settings, params=params,
                items=parent.items + _texts(appended),
                route=route, model=model, request_digest=parent.request_digest,
                declared=tuple(declared), tool_definitions=tuple(definitions),
                anchors=parent.anchors,
                carried=carried, notices_from=parent.notices_from,
                turn_request=parent.turn_request, continuation=continuation,
                parent=parent, registry=registry,
            )
            budget = registry.context_budget
            chars = request_text_chars(prepared.body)
            if budget is None or chars <= budget:
                return prepared
            # The tool loop pruned its messages to fit ``budget``; replaying
            # the snapshot would send more than that, and a continuation may
            # only append. So the chain ends here rather than overrunning it.
            boundary = {"reason": BUDGET_EXCEEDED, "request_chars": chars, "budget_chars": budget}
        registry.discard(parent, boundary["reason"])
    elif registry is not None:
        pruned = _pruned_chain(registry, route, model, messages)
        retired = None if pruned is not None else _retired_chain(registry, route, model, messages)
        if pruned is not None:
            boundary = {"reason": PREFIX_PRUNED}
            registry.discard(pruned, PREFIX_PRUNED)
        elif retired is not None:
            boundary = {
                "reason": RETIRED,
                "retired_reason": retired.reason,
                "retired_response_id": retired.response_id,
            }

    ordinary, omitted = _ordinary_items(messages)
    if boundary is None and any(_has_tool_history(message) for message in messages):
        boundary = {"reason": "no_snapshot" if registry is not None else "outside_turn"}
    continuation = {"status": FRESH}
    if boundary is not None:
        continuation = {"status": "boundary", **boundary, "omitted_tool_items": omitted}
        logger.warning(
            "OpenAI Responses continuation boundary for %s: %s; sending an "
            "ordinary conversation without replayed reasoning, tool calls "
            "or tool results (%d left out)",
            model, boundary["reason"], omitted,
        )
    items = _texts(ordinary)
    return _finish_request(
        settings=translated.settings, params=translated.params,
        items=items, route=route, model=model, request_digest=digest,
        declared=tuple(offered), tool_definitions=tuple(_json_text(tool) for tool in live_tools),
        anchors=(),
        carried=(_carried_texts(ordinary),), notices_from=_notices_from(ordinary),
        turn_request=_turn_request(messages), continuation=continuation,
        parent=None, registry=registry,
    )


def _finish_request(
    *,
    settings: Dict[str, Any],
    params: Mapping[str, Any],
    items: Tuple[str, ...],
    route: ResponsesRoute,
    model: str,
    request_digest: str,
    declared: Tuple[str, ...],
    tool_definitions: Tuple[str, ...],
    anchors: Tuple[ResponsesAnchor, ...],
    carried: Tuple[Tuple[str, ...], ...],
    notices_from: int,
    turn_request: Optional[str],
    continuation: Mapping[str, Any],
    parent: Optional[ResponsesSnapshot],
    registry: Optional[ResponsesContinuations],
) -> ResponsesRequest:
    body: Dict[str, Any] = {"model": model, "input": [json.loads(item) for item in items]}
    body.update({key: value for key, value in settings.items() if key != "extra_body"})
    if not body.get("tools"):
        body.pop("tools", None)
    body.update(params)
    extra_body = settings.get("extra_body")
    if extra_body:
        # The SDK merges ``extra_body`` over the body; by construction it
        # holds only keys the body does not set, so it overwrites nothing.
        overlap = sorted(set(extra_body) & set(body))
        if overlap:
            raise ValueError(f"extra_body would overwrite {overlap} on the Responses path.")
        body["extra_body"] = dict(extra_body)
    return ResponsesRequest(
        body=body, route=route, model=model, items=items, settings=_json_text(settings),
        request_digest=request_digest, declared_tools=declared,
        tool_definitions=tool_definitions, anchors=anchors,
        carried=carried, notices_from=notices_from, turn_request=turn_request,
        continuation=dict(continuation), parent=parent, registry=registry,
    )


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _count(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def responses_usage(usage: Any) -> Dict[str, Optional[int]]:
    """Usage in the service's terms, with cached input split out.

    Matches the chat-completions normalization: tokens read from or written
    to the prompt cache are reported separately and taken out of the input
    and total counts.
    """
    inputs = _count(_field(usage, "input_tokens"))
    outputs = _count(_field(usage, "output_tokens"))
    total = _count(_field(usage, "total_tokens"))
    details = _field(usage, "input_tokens_details")
    reads = _count(_field(details, "cached_tokens"))
    writes = _count(_field(details, "cache_write_tokens"))
    separated = (reads or 0) + (writes or 0)
    if separated:
        inputs = None if inputs is None else max(0, inputs - separated)
        total = None if total is None else max(0, total - separated)
    return {
        "input_tokens": inputs,
        "output_tokens": outputs,
        "total_tokens": total,
        "cache_creation_input_tokens": writes,
        "cache_read_input_tokens": reads,
    }


def _usage_response(usage: Any, stop_reason: Optional[str]) -> Optional[LLMResponse]:
    if usage is None:
        return None
    counts = responses_usage(usage)
    if all(value is None for value in counts.values()):
        return None
    return attach_stop_reason(LLMResponse(**counts), stop_reason)


def _item_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    to_dict = getattr(item, "to_dict", None)
    if not callable(to_dict):
        raise LLMProtocolError("A Responses output item has no serializable form.")
    return to_dict(mode="json")


def _prefix_mismatch(response: Any) -> Optional[Tuple[str, ...]]:
    """Ids of replayed items the provider removed for a changed prefix.

    ``None`` when the response reports no ``prefix_mismatch`` among its
    ``input_transformations``; otherwise the ids it named (empty when the
    transformation named none).
    """
    try:
        transformations = _item_dict(response).get("input_transformations")
    except LLMProtocolError:
        return None
    if not isinstance(transformations, list):
        return None
    reported = False
    removed: List[str] = []
    for entry in transformations:
        if not isinstance(entry, Mapping):
            continue
        values = list(entry.values()) + [
            nested for value in entry.values() if isinstance(value, Mapping)
            for nested in value.values()
        ]
        if "prefix_mismatch" not in values:
            continue
        reported = True
        for key in ("item_id", "id"):
            if isinstance(entry.get(key), str):
                removed.append(entry[key])
                break
    return tuple(removed) if reported else None


def _stop_reason(status: str, response: Any, has_calls: bool) -> str:
    if status == "completed":
        return "tool_calls" if has_calls else "stop"
    reason = _field(_field(response, "incomplete_details"), "reason")
    if reason == "max_output_tokens":
        return "length"
    return reason if isinstance(reason, str) and reason else "incomplete"


def _message_text(item: Mapping[str, Any]) -> str:
    return "".join(
        part.get("text") or part.get("refusal") or ""
        for part in item.get("content") or ()
        if isinstance(part, Mapping) and part.get("type") in _TEXT_PART_TYPES
    )


def _parse_output(output: Sequence[Mapping[str, Any]]) -> Tuple[str, List[ToolCall]]:
    texts: List[str] = []
    calls: List[ToolCall] = []
    for item in output:
        kind = item.get("type")
        if kind == "reasoning":
            continue
        if kind == "message":
            text = _message_text(item)
            if text:
                texts.append(text)
            continue
        if kind == "function_call":
            call_id, name, arguments = (item.get(key) for key in ("call_id", "name", "arguments"))
            if any(type(value) is not str or not value for value in (call_id, name)):
                raise LLMProtocolError("A Responses function call must name its call and function.")
            if type(arguments) is not str:
                raise LLMProtocolError("Responses function call arguments must be a JSON string.")
            calls.append(ToolCall(id=call_id, name=name, arguments=decode_tool_arguments(arguments or "{}")))
            continue
        raise LLMProtocolError(f"Unsupported Responses output item {kind!r}.")
    return "\n\n".join(texts), calls


class ResponsesEventState:
    """One Responses call's events, reduced to what both entry points return.

    :meth:`observe` turns each event into the items a streaming consumer is
    shown, and ends the chain on the terminal response if the provider
    reports a ``prefix_mismatch``; :meth:`finish` checks the terminal
    response and builds the
    :class:`LLMResponse`; :meth:`settle` extends or retires the turn's
    snapshot. :attr:`served` is the provider's identification so far, kept
    for an error raised at any point after ``response.created``.

    ``receipt`` is the attempt's :class:`~.provider_served.ProviderAttemptReceipt`.
    :meth:`observe` records on it what each event reports as the event
    arrives (#4207): the identification once the provider has named its
    response, a ``prefix_mismatch`` the terminal response reports, and the
    usage the terminal response reported. Whatever ends the attempt after
    that, the receipt already holds it.
    """

    def __init__(
        self, request: ResponsesRequest, *, provider: str, receipt: ProviderAttemptReceipt,
    ) -> None:
        self.request = request
        self.provider = provider
        self.receipt = receipt
        self.served = ProviderServed(api=RESPONSES_API, continuation=request.continuation)
        self._text = ""
        self._text_item: Optional[str] = None
        self._done: Dict[int, Any] = {}
        self._final: Any = None
        self._output: Tuple[Dict[str, Any], ...] = ()

    @property
    def observed(self) -> bool:
        return self.served.model is not None or self.served.response_id is not None

    @property
    def terminal_arrived(self) -> bool:
        """Whether the terminal response (and any usage it reported) arrived."""
        return self._final is not None

    @property
    def text(self) -> str:
        return self._text

    def _identify(self, response: Any) -> None:
        self.served = self.served.with_response(
            model=_field(response, "model"), response_id=_field(response, "id"),
        )
        self._receive()

    def _receive(self) -> None:
        """Record on the receipt the identification as it is now, once the
        provider has named its response."""
        if self.observed:
            self.receipt.identify(self.served)

    def _visible(self, delta: Any, item_id: Any) -> List[str]:
        if not isinstance(delta, str) or not delta:
            return []
        chunks: List[str] = []
        if self._text and item_id != self._text_item:
            chunks.append("\n\n")
        self._text_item = item_id
        chunks.append(delta)
        self._text += "".join(chunks)
        return chunks

    def observe(self, event: Any) -> List[Any]:
        kind = _field(event, "type")
        if kind in ("response.created", "response.queued", "response.in_progress"):
            self._identify(_field(event, "response"))
        elif kind == "response.output_item.added":
            item = _field(event, "item")
            if _field(item, "type") == "function_call":
                return [ToolCallStarted(
                    index=_field(event, "output_index"),
                    id=_field(item, "call_id") or None,
                    name=_field(item, "name") or None,
                )]
        elif kind in ("response.output_text.delta", "response.refusal.delta"):
            return self._visible(_field(event, "delta"), _field(event, "item_id"))
        elif kind == "response.reasoning_summary_text.delta":
            delta = _field(event, "delta")
            if isinstance(delta, str) and delta:
                return [ThinkingDelta(delta, provider=self.provider)]
        elif kind == "response.output_item.done":
            index = _field(event, "output_index")
            if isinstance(index, int):
                self._done[index] = _field(event, "item")
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            self._final = _field(event, "response")
            self._identify(self._final)
            self._end_degraded_chain(self._final)
            self.receipt.report_usage(self.reported_usage())
        elif kind == "error":
            raise ResponsesAPIFailure(
                model=self.request.model, status="error", code=_field(event, "code"),
                message=_field(event, "message"), usage=None, served=self.served,
            )
        return []

    def reported_usage(self) -> Optional[LLMResponse]:
        """The counts the terminal response reported, if one arrived."""
        if self._final is None:
            return None
        status = _field(self._final, "status")
        return _usage_response(_field(self._final, "usage"), status if isinstance(status, str) else None)

    def finish(self, *, notice: Any = None) -> Tuple[List[str], LLMResponse]:
        """The chunks still to show and the terminal response.

        ``notice(stop_reason, caller_budget)`` returns the output-limit notice
        for a response cut at a limit nobody chose, or ``None``.
        """
        response = self._final
        if response is None:
            raise LLMProtocolError("The Responses stream ended without a terminal response.")
        status = _field(response, "status")
        usage = _field(response, "usage")
        if status == "failed" or status not in ("completed", "incomplete"):
            error = _field(response, "error")
            raise ResponsesAPIFailure(
                model=self.request.model, status=str(status), code=_field(error, "code"),
                message=_field(error, "message"),
                usage=_usage_response(usage, str(status)), served=self.served,
            )
        output = [_item_dict(item) for item in (_field(response, "output") or ())]
        if not output:
            output = [_item_dict(self._done[index]) for index in sorted(self._done)]
        self._output = tuple(output)
        text, calls = _parse_output(output)
        stop_reason = _stop_reason(status, response, bool(calls))

        chunks: List[str] = []
        if not self._text and text:
            chunks.append(text)
            self._text = text
        limit_notice = notice(stop_reason, self.request.caller_output_budget) if notice else None
        if limit_notice is not None:
            chunks.append(limit_notice)
            self._text += limit_notice

        result = attach_stop_reason(
            LLMResponse(
                content=self._text or None,
                tool_calls=calls or None,
                **responses_usage(usage),
            ),
            stop_reason,
        )
        return chunks, result

    def _end_degraded_chain(self, response: Any) -> None:
        """End the chain as soon as a terminal response reports a prefix mismatch.

        Runs when the terminal event arrives, before the stream's clean end:
        whatever becomes of this reply (it failed, is malformed, or the
        connection drops before EOF), its parent is never replayed again,
        and every error raised from here on carries the mismatch and the
        chain's end in :attr:`served`.
        """
        removed = _prefix_mismatch(response)
        if removed is None:
            return
        self.served = replace(
            self.served,
            prefix_mismatch=removed,
            continuation={**self.served.continuation, "chain_ended": PREFIX_MISMATCH},
        )
        self._receive()
        self._retire_parent(PREFIX_MISMATCH)

    def _retire_parent(self, reason: str) -> None:
        if self.request.parent is not None and self.request.registry is not None:
            self.request.registry.discard(self.request.parent, reason)

    def settle(self, result: LLMResponse) -> ProviderServed:
        """Extend or retire the turn's snapshot for this reply.

        The returned evidence's ``continuation`` still describes the request
        as it was sent; its ``snapshot`` says whether this reply's snapshot
        was kept and, if not, why. A reply served by another model than its
        chain, or one whose provider reports a ``prefix_mismatch``, is still
        returned to the caller and acted on, but its chain ends: neither the
        chain it extended nor its own snapshot is replayed again.
        """
        request = self.request
        parent = request.parent
        registry = request.registry
        served_model = self.served.model
        response_id = self.served.response_id
        snapshot: Optional[ResponsesSnapshot] = None
        if served_model and response_id:
            snapshot = ResponsesSnapshot(
                route=request.route.label,
                client=request.route.client,
                adapter=request.route.adapter,
                requested_model=request.model,
                served_model=served_model,
                response_id=response_id,
                settings=request.settings,
                request_digest=request.request_digest,
                items=request.items + tuple(_json_text(item) for item in self._output),
                declared_tools=request.declared_tools,
                tool_definitions=request.tool_definitions,
                anchors=request.anchors + (ResponsesAnchor(
                    content=result.content or "",
                    call_ids=tuple(call.id for call in result.tool_calls or ()),
                    # Calls that cannot be digested match no message, so a
                    # reply holding them is never continued.
                    calls_digest=_calls_digest([
                        (call.id, call.name, call.arguments) for call in result.tool_calls or ()
                    ]) or "",
                ),),
                carried=request.carried,
                notices_from=request.notices_from,
                turn_request=request.turn_request,
            )

        ended: Optional[str] = None
        continuation: Dict[str, Any] = dict(request.continuation)
        if self.served.prefix_mismatch is not None:
            # First: the terminal event already retired the parent for it
            # (:meth:`_end_degraded_chain`), so the evidence and the
            # retirement record name the same reason.
            ended = PREFIX_MISMATCH
            continuation.update(chain_ended=ended)
            logger.warning(
                "OpenAI Responses reported a prefix mismatch for %s; the reply "
                "is used but its chain is never replayed again", request.model,
            )
        elif parent is not None and snapshot is None:
            # The reply named no model or response id: nothing says what
            # served it, so nothing of the chain is replayed after it.
            ended = UNIDENTIFIED_RESPONSE
            continuation.update(chain_ended=ended)
        elif parent is not None and served_model != parent.served_model:
            ended = SERVED_MODEL_MISMATCH
            continuation.update(chain_ended=ended)
            logger.warning(
                "OpenAI Responses continuation of %s was served by %r, not %r; "
                "the reply is used but its chain ends here",
                request.model, served_model, parent.served_model,
            )
        if parent is not None and served_model and served_model != parent.served_model:
            continuation["expected_model"] = parent.served_model
        if ended is not None:
            if registry is not None:
                if parent is not None:
                    self._retire_parent(ended)
                if snapshot is not None:
                    registry.retire(snapshot, ended)
            return self._settled(continuation, {"stored": False, "reason": ended})
        if registry is None:
            return self._settled(continuation, {"stored": False, "reason": OUTSIDE_TURN})
        if snapshot is None:
            return self._settled(continuation, {"stored": False, "reason": UNIDENTIFIED_RESPONSE})
        if registry.record(snapshot, replaces=parent):
            return self._settled(continuation, {"stored": True})
        if registry.expired:
            return self._settled(continuation, {"stored": False, "reason": TURN_ENDED})
        # Another request of the chain extended or retired ``parent`` first.
        registry.retire(snapshot, PARENT_RETIRED)
        outcome: Dict[str, Any] = {"stored": False, "reason": PARENT_RETIRED}
        parent_reason = registry.retirement_of(parent) if parent is not None else None
        if parent_reason is not None:
            outcome["parent_reason"] = parent_reason
        return self._settled(continuation, outcome)

    def _settled(self, continuation: Mapping[str, Any], snapshot: Mapping[str, Any]) -> ProviderServed:
        self.served = replace(self.served, continuation=dict(continuation), snapshot=dict(snapshot))
        return self.served


#: ``snapshot.reason`` values for a reply whose snapshot was not kept that are
#: not retirements (see :data:`~.responses_continuation.RETIREMENT_REASONS`).
OUTSIDE_TURN = "outside_turn"  # no turn registry: nothing is kept outside a turn
TURN_ENDED = "turn_ended"  # the turn's registry expired before the reply


__all__ = [
    "FRESH",
    "OUTSIDE_TURN",
    "RESPONSES_API",
    "RESPONSES_FUNCTION_TOOL_MODEL_PATTERN",
    "RETIRED",
    "ResponsesAPIFailure",
    "ResponsesEventState",
    "ResponsesRequest",
    "ResponsesRoute",
    "ResponsesStreamInterrupted",
    "TURN_ENDED",
    "UNIDENTIFIED_RESPONSE",
    "prepare_responses_request",
    "refuse_raw_tool_declarations",
    "request_digest",
    "request_text_chars",
    "requires_responses_api",
    "responses_usage",
]
