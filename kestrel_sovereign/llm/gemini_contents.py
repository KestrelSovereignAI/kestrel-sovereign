"""Chat messages as Gemini ``contents``, shared by the Google and Vertex adapters.

The agent hands every adapter OpenAI-format messages (the SDK contract:
``{"role": "user", "content": ...}``, an ``assistant`` turn carrying
``tool_calls``, ``{"role": "tool", "tool_call_id": ...}`` results), but
google-genai validates ``contents`` as Gemini ``Content`` and refuses any other
shape before a request is sent. Both adapters convert at their boundary
through :func:`to_gemini_contents` (#3608):

* ``system`` turns, and ``VertexAIAdapter``'s ``_system`` marker, become the
  ``system_instruction``;
* ``user``/``assistant`` ``content`` (a string or a typed-part list, including
  ``image_url``) becomes ``parts``, with ``assistant`` sent as ``model``;
* ``assistant.tool_calls`` become ``function_call`` parts, their arguments
  JSON-decoded;
* ``tool`` results become ``function_response`` parts. Gemini matches a
  response to its call by function name, so the name is recovered from the
  ``tool_calls`` of the model turn the result answers, by ``tool_call_id``.

A message already in Gemini shape (it carries ``parts``) is sent unchanged.
"""

import json
import mimetypes
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

#: The role ``VertexAIAdapter.create_messages`` gives its system prompt.
SYSTEM_MARKER_ROLE = "_system"


def to_gemini_contents(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Split ``messages`` into Gemini's ``system_instruction`` and ``contents``.

    Raises :class:`ValueError` for a message Gemini cannot be sent: an unknown
    role, a content part with no Gemini equivalent, tool-call arguments that
    are not a JSON object, or a tool result that answers no call of the model
    turn before it.
    """
    system_texts: List[str] = []
    contents: List[Dict[str, Any]] = []
    # The function names of the latest model turn's calls, by tool_call_id.
    # Calls are numbered per response, so ids repeat across turns; only the
    # turn a result follows can answer it.
    call_names: Dict[str, str] = {}
    # The function-response turn consecutive tool results are gathered into.
    response_turn: Optional[Dict[str, Any]] = None

    for message in messages:
        role = message.get("role")
        if role in ("system", SYSTEM_MARKER_ROLE):
            text = _system_text(message)
            if text:
                system_texts.append(text)
            continue
        if role == "tool" and "parts" not in message:
            part = {"function_response": {
                "name": _answered_call_name(message, call_names),
                "response": {"result": _text_of(message.get("content"))},
            }}
            if response_turn is None:
                response_turn = {"role": "user", "parts": []}
                contents.append(response_turn)
            response_turn["parts"].append(part)
            continue

        response_turn = None
        call_names = {}
        if "parts" in message:
            contents.append(message)
        elif role == "user":
            parts = _content_parts(message.get("content"))
            if parts:
                contents.append({"role": "user", "parts": parts})
        elif role == "assistant":
            parts = _content_parts(message.get("content"))
            for call in message.get("tool_calls") or ():
                name, args = _function_call(call)
                call_id = call.get("id")
                if isinstance(call_id, str) and call_id:
                    call_names[call_id] = name
                parts.append({"function_call": {"name": name, "args": args}})
            if parts:
                contents.append({"role": "model", "parts": parts})
        else:
            raise ValueError(f"a message with role {role!r} cannot be sent to Gemini")

    return ("\n\n".join(system_texts) or None), contents


def tool_result_messages(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``continue_with_tool_results``' results as OpenAI ``tool`` messages.

    Each result is ``{"tool_call_id": str, "content": str}``. The messages they
    follow must end with the assistant turn whose ``tool_calls`` they answer:
    :func:`to_gemini_contents` names each response from that turn.
    """
    return [
        {
            "role": "tool",
            "tool_call_id": result["tool_call_id"],
            "content": result["content"],
        }
        for result in tool_results
    ]


def _system_text(message: Dict[str, Any]) -> str:
    if "parts" in message:
        return "\n\n".join(
            part["text"] for part in message["parts"]
            if isinstance(part, dict) and isinstance(part.get("text"), str)
            and part["text"]
        )
    return _text_of(message.get("content"))


def _text_of(content: Any) -> str:
    """The text of a ``system`` or ``tool`` message, which carries nothing else."""
    parts = _content_parts(content)
    if any("text" not in part for part in parts):
        raise ValueError("a system or tool message can only carry text to Gemini")
    return "\n".join(part["text"] for part in parts)


def _content_parts(content: Any) -> List[Dict[str, Any]]:
    """OpenAI message ``content`` as Gemini parts; empty text is dropped
    (Gemini refuses an empty text part)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        raise ValueError(
            f"message content of type {type(content).__name__} cannot be sent to Gemini"
        )
    parts: List[Dict[str, Any]] = []
    for item in content:
        part = _content_part(item)
        if part is not None:
            parts.append(part)
    return parts


def _content_part(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        return {"text": item} if item else None
    kind = item.get("type") if isinstance(item, dict) else None
    if kind == "text":
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError("a text content part carries no text")
        return {"text": text} if text else None
    if kind == "image_url":
        return _image_part(item.get("image_url"))
    raise ValueError(f"a content part of type {kind!r} has no Gemini equivalent")


def _image_part(image_url: Any) -> Dict[str, Any]:
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        raise ValueError("an image_url content part carries no url")
    # RFC 2397: the scheme and the base64 token are case-insensitive.
    if url[:len("data:")].lower() == "data:":
        header, separator, data = url.partition(",")
        mime_type, *parameters = header[len("data:"):].split(";")
        if (
            not separator or not mime_type
            or "base64" not in (parameter.lower() for parameter in parameters)
        ):
            raise ValueError("an image data URL must be base64 and name its MIME type")
        return {"inline_data": {"mime_type": mime_type, "data": data}}
    # Gemini requires the MIME type of a file it fetches by URI.
    mime_type, _ = mimetypes.guess_type(urlsplit(url).path)
    if mime_type is None:
        raise ValueError(
            "the MIME type Gemini requires for an image URL cannot be told "
            "from its path"
        )
    return {"file_data": {"file_uri": url, "mime_type": mime_type}}


def _function_call(call: Any) -> Tuple[str, Dict[str, Any]]:
    """An OpenAI ``tool_calls`` entry's function name and decoded arguments."""
    function = call.get("function") if isinstance(call, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError("a tool call carries no function name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else None
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"tool call {name!r} has arguments that are not JSON: {exc}"
            ) from exc
    if arguments is None:
        return name, {}
    if not isinstance(arguments, dict):
        raise ValueError(f"tool call {name!r} has arguments that are not a JSON object")
    return name, arguments


def _answered_call_name(message: Dict[str, Any], call_names: Dict[str, str]) -> str:
    call_id = message.get("tool_call_id")
    name = call_names.get(call_id) if isinstance(call_id, str) else None
    if name is None:
        raise ValueError(
            f"tool result {call_id!r} answers no tool call of the model turn "
            "before it; Gemini names a function response after its call"
        )
    return name
