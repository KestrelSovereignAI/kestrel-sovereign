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
messages carry tool history, or when it continues one of the turn's
snapshots. A tool-free continuation therefore keeps its chain, and tool
history no snapshot accounts for is never sent to chat completions.

A request that carries tool history no snapshot of the turn accounts for (no
turn, an expired or retired snapshot, another route or model, a provider that
served a different model) is a continuation boundary. So is a continuation
whose replay would exceed the context budget the turn's tool loop pruned its
own messages to: replaying the snapshot would restore what the loop removed.
A boundary replays no encrypted reasoning and no tool call or result: it is
sent as an ordinary conversation, its text messages only, and its
``provider_served`` evidence records the boundary and why.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kestrel_sdk.llm import LLMProtocolError, ToolCallStarted

from .adapter import LLMResponse, ThinkingDelta, ToolCall
from .completion_protocol import decode_tool_arguments
from .output_ceiling import IncompleteGenerationError, attach_stop_reason
from .provider_served import ProviderServed, attach_provider_served
from .responses_continuation import (
    ResponsesAnchor,
    ResponsesContinuations,
    ResponsesSnapshot,
    current_responses_continuations,
)

logger = logging.getLogger(__name__)

#: Models whose function-tool requests a native OpenAI route sends to the
#: Responses API. Matches qualified ids, dated snapshots and fine-tunes, but
#: not a longer family name that merely starts the same way.
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


@dataclass(frozen=True)
class ResponsesRoute:
    """The route a request is sent on, as a snapshot is bound to it.

    ``label`` names the adapter and its base URL, for logs. ``client`` is the
    route's own SDK client: the provider registry builds one per configured
    route, so two routes that share a vendor and base URL (and possibly a
    different account or organization) are still two routes. A snapshot holds
    its client, so a route rebuilt mid-turn is a new route, never a reused id.
    """

    label: str
    client: object

    def serves(self, snapshot: ResponsesSnapshot) -> bool:
        return snapshot.route == self.label and snapshot.client is self.client


def requires_responses_api(
    *,
    native_openai: bool,
    model: str,
    request: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    route: ResponsesRoute,
) -> bool:
    """Whether this request must go to the Responses API.

    Decided from the request as it will be sent, top-level ``tools`` and any
    ``tools`` a caller put in ``extra_body`` both counting, and from its
    messages: tool history, or a reply of one of the turn's snapshots on this
    route. A continuation offered no tools is still a continuation, and tool
    history without a snapshot must meet the recorded boundary rather than
    be replayed on chat completions.
    """
    if not native_openai or not RESPONSES_FUNCTION_TOOL_MODEL_PATTERN.search(model):
        return False
    if any(_is_function_tool(tool) for tool in _effective_chat_tools(request)):
        return True
    if any(_has_tool_history(message) for message in messages):
        return True
    registry = current_responses_continuations()
    return registry is not None and _continued(registry, route, model, messages) is not None


def _effective_chat_tools(request: Mapping[str, Any]) -> List[Any]:
    tools: List[Any] = []
    for source in (request, request.get("extra_body")):
        if isinstance(source, Mapping) and isinstance(source.get("tools"), list):
            tools.extend(source["tools"])
    return tools


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
    none of them can name a field the request body already sets.
    """
    extra_body = dict(request.get("extra_body") or {})
    for key in extra_body:
        if key in _RESERVED_PARAMS:
            raise ValueError(f"The OpenAI Responses path sets {key!r} itself.")
    tools = [_responses_tool(tool) for tool in _effective_chat_tools(request)]
    extra_body.pop("tools", None)

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


def _matches(message: Mapping[str, Any], anchor: ResponsesAnchor) -> bool:
    if message.get("role") != "assistant":
        return False
    if anchor.call_ids:
        return _call_ids(message) == anchor.call_ids
    return not message.get("tool_calls") and _text_of(message.get("content")) == anchor.content


def _locate(snapshot: ResponsesSnapshot, messages: Sequence[Mapping[str, Any]]) -> Optional[int]:
    """The index of the message replaying the snapshot's latest reply.

    Every reply the chain absorbed must still be present, in order. A chain
    whose replies carried no tool call is confirmed by its first user message.
    """
    before = len(messages)
    positions: List[int] = []
    for anchor in reversed(snapshot.anchors):
        found = next(
            (index for index in range(before - 1, -1, -1) if _matches(messages[index], anchor)),
            None,
        )
        if found is None:
            return None
        positions.append(found)
        before = found
    if not any(anchor.call_ids for anchor in snapshot.anchors):
        first = positions[-1]
        if snapshot.turn_request is None or not any(
            message.get("role") == "user"
            and _json_text(message.get("content")) == snapshot.turn_request
            for message in messages[:first]
        ):
            return None
    return positions[0]


def _appended_items(
    tail: Sequence[Mapping[str, Any]], call_ids: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    """What the message list gained after the snapshot's latest reply."""
    items: List[Dict[str, Any]] = []
    answered: set[str] = set()
    for message in tail:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in call_ids:
                raise _Boundary("unknown_tool_output")
            if call_id in answered:
                raise _Boundary("duplicate_tool_output")
            answered.add(call_id)
            items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": _tool_output(message.get("content")),
            })
        elif role in ("system", "developer", "user"):
            items.append(_message_item(message))
        elif role == "assistant":
            raise _Boundary("unanchored_assistant_message")
        else:
            raise _Boundary("unsupported_message_role")
    return items


def _continued(
    registry: ResponsesContinuations,
    route: ResponsesRoute,
    model: str,
    messages: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[ResponsesSnapshot, int]]:
    """The newest snapshot on ``route`` for ``model`` that ``messages`` continue,
    with the index of the message replaying its latest reply."""
    for snapshot in registry.snapshots():
        if not route.serves(snapshot) or snapshot.requested_model != model:
            continue
        index = _locate(snapshot, messages)
        if index is not None:
            return snapshot, index
    return None


@dataclass(frozen=True)
class ResponsesRequest:
    """One serialized Responses request and what its reply will extend."""

    body: Dict[str, Any]
    route: ResponsesRoute
    model: str
    items: Tuple[str, ...]
    settings: str
    declared_tools: Tuple[str, ...]
    anchors: Tuple[ResponsesAnchor, ...]
    turn_request: Optional[str]
    continuation: Mapping[str, Any]
    parent: Optional[ResponsesSnapshot]
    registry: Optional[ResponsesContinuations]

    @property
    def caller_output_budget(self) -> bool:
        return self.body.get("max_output_tokens") is not None


def _settings_without_tools(settings: Mapping[str, Any]) -> str:
    return _json_text({key: value for key, value in settings.items() if key != "tools"})


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


def prepare_responses_request(
    *,
    route: ResponsesRoute,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
) -> ResponsesRequest:
    """Serialize one request: a snapshot continuation, or an ordinary one."""
    translated = _translate_request(request)
    registry = current_responses_continuations()
    live_tools = translated.settings["tools"]
    offered = [tool["name"] for tool in live_tools]

    boundary: Optional[Dict[str, Any]] = None
    found = _continued(registry, route, model, messages) if registry is not None else None
    if found is not None:
        parent, index = found
        try:
            appended = _appended_items(messages[index + 1:], parent.anchors[-1].call_ids)
            if _settings_without_tools(json.loads(parent.settings)) != (
                _settings_without_tools(translated.settings)
            ):
                raise _Boundary("request_settings_changed")
        except _Boundary as refused:
            boundary = {"reason": refused.reason}
        else:
            settings = json.loads(parent.settings)
            declared = list(parent.declared_tools)
            exposed = [tool for tool in live_tools if tool["name"] not in declared]
            if exposed:
                appended.append({"type": "additional_tools", "role": "developer", "tools": exposed})
                declared.extend(tool["name"] for tool in exposed)
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
                items=parent.items + tuple(_json_text(item) for item in appended),
                route=route, model=model, declared=tuple(declared), anchors=parent.anchors,
                turn_request=parent.turn_request, continuation=continuation,
                parent=parent, registry=registry,
            )
            budget = registry.context_budget
            chars = request_text_chars(prepared.body)
            if budget is None or chars <= budget:
                return prepared
            # The tool loop pruned its messages to fit ``budget``; the
            # snapshot still holds what it pruned, and a continuation may only
            # append. So the chain ends here rather than restoring it.
            boundary = {"reason": "context_budget", "request_chars": chars, "budget_chars": budget}
        registry.discard(parent)

    ordinary, omitted = _ordinary_items(messages)
    if boundary is None and any(_has_tool_history(message) for message in messages):
        boundary = {"reason": "no_snapshot" if registry is not None else "outside_turn"}
    continuation = {"status": "fresh"}
    if boundary is not None:
        continuation = {"status": "boundary", **boundary, "omitted_tool_items": omitted}
        logger.warning(
            "OpenAI Responses continuation boundary for %s: %s; sending an "
            "ordinary conversation without replayed reasoning, tool calls "
            "or tool results (%d left out)",
            model, boundary["reason"], omitted,
        )
    return _finish_request(
        settings=translated.settings, params=translated.params,
        items=tuple(_json_text(item) for item in ordinary), route=route, model=model,
        declared=tuple(offered), anchors=(),
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
    declared: Tuple[str, ...],
    anchors: Tuple[ResponsesAnchor, ...],
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
        declared_tools=declared, anchors=anchors, turn_request=turn_request,
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
    """Ids of reasoning items the provider removed for a changed prefix.

    ``None`` when the response reports no ``input_transformations``.
    """
    try:
        transformations = _item_dict(response).get("input_transformations")
    except LLMProtocolError:
        return None
    if not isinstance(transformations, list):
        return None
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
        for key in ("item_id", "id"):
            if isinstance(entry.get(key), str):
                removed.append(entry[key])
                break
    return tuple(removed)


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
    shown; :meth:`finish` checks the terminal response and builds the
    :class:`LLMResponse`; :meth:`settle` extends or retires the turn's
    snapshot. :attr:`served` is the provider's identification so far, kept
    for an error raised at any point after ``response.created``.
    """

    def __init__(self, request: ResponsesRequest, *, provider: str) -> None:
        self.request = request
        self.provider = provider
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
    def text(self) -> str:
        return self._text

    def _identify(self, response: Any) -> None:
        self.served = self.served.with_response(
            model=_field(response, "model"), response_id=_field(response, "id"),
        )

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
        self.served = replace(self.served, prefix_mismatch=_prefix_mismatch(response))
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

    def settle(self, result: LLMResponse) -> ProviderServed:
        """Extend or retire the turn's snapshot for this reply."""
        request = self.request
        parent = request.parent
        registry = request.registry
        served_model = self.served.model
        response_id = self.served.response_id
        if parent is not None and served_model != parent.served_model:
            if registry is not None:
                registry.discard(parent)
            self.served = replace(self.served, continuation={
                **request.continuation,
                "chain_ended": "served_model_mismatch",
                "expected_model": parent.served_model,
            })
            logger.warning(
                "OpenAI Responses continuation of %s was served by %r, not %r; "
                "the chain ends here and its next request is an ordinary conversation",
                request.model, served_model, parent.served_model,
            )
            return self.served
        if registry is None or not served_model or not response_id:
            return self.served
        anchor = ResponsesAnchor(
            content=result.content or "",
            call_ids=tuple(call.id for call in result.tool_calls or ()),
        )
        registry.record(
            ResponsesSnapshot(
                route=request.route.label,
                client=request.route.client,
                requested_model=request.model,
                served_model=served_model,
                response_id=response_id,
                settings=request.settings,
                items=request.items + tuple(_json_text(item) for item in self._output),
                declared_tools=request.declared_tools,
                anchors=request.anchors + (anchor,),
                turn_request=request.turn_request,
            ),
            replaces=parent,
        )
        return self.served


__all__ = [
    "RESPONSES_API",
    "RESPONSES_FUNCTION_TOOL_MODEL_PATTERN",
    "ResponsesAPIFailure",
    "ResponsesEventState",
    "ResponsesRequest",
    "ResponsesRoute",
    "prepare_responses_request",
    "request_text_chars",
    "requires_responses_api",
    "responses_usage",
]
