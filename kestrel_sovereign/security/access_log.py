"""Redact credentials from HTTP request targets before access logging.

Uvicorn builds its access-log request target from the mutable ASGI ``scope``
when the application sends ``http.response.start``.  Query authentication is
unavoidable for browser ``EventSource`` clients, so keep the original query
available to authentication and endpoint code, then replace sensitive values
at that final send seam.  Uvicorn consequently logs the redacted scope while
request semantics remain unchanged.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode


_REDACTED_VALUE = "redacted"


def _url_path_literal(value: str) -> str:
    # ASGI supplies decoded paths. A double-escaped malformed target can leave
    # encoded static segments after that decode. Recognize both representations
    # for diagnostics only, including separators on unrouted failures.
    return "".join(f"(?:{re.escape(char)}|%{ord(char):02x})" for char in value)


_LEGACY_WORKFLOW_ARTIFACT_MARKER = "/api/workflows/artifact-metadata"
_LEGACY_WORKFLOW_ARTIFACT_TARGET = (
    # Commit to the first run prefix. Restarting a suffix scan at every run
    # prefix makes malformed repeated-prefix paths quadratic before auth.
    # Any later artifact suffix is already covered by that first prefix.
    "(?i:" + r"\A(?>[\s\S]*?" + _url_path_literal("/api/workflows/runs/") + ")"
    + r"[\s\S]*" + _url_path_literal("/artifacts") + r"(?:/|%(?:2f|3f|23)|[?#]|$)|"
    # Escaped errors reach the outer error handler with this marker already
    # installed. It must still classify as private to suppress exception text.
    + re.escape(_LEGACY_WORKFLOW_ARTIFACT_MARKER) + r"(?:/|$))"
)
_PRIVATE_LEGACY_WORKFLOW_TARGET = re.compile(_LEGACY_WORKFLOW_ARTIFACT_TARGET)
# Also used before optional auto-instrumentation creates a URL-bearing span.
PRIVATE_CAPABILITY_URL_PATTERN = (
    r"/api/(?:artifacts|visual-computer)(?:[/?#]|$)|"
    + _LEGACY_WORKFLOW_ARTIFACT_TARGET
)
_PRIVATE_TARGET = re.compile(r"/api/(artifacts|visual-computer)(?:/|$)")
_PRIVATE_TOOL = re.compile(r"(?:^|__|:)(?:artifact_|visual_)")
_SENSITIVE_QUERY_NAMES = frozenset(
    {
        "apikey",
        "authorization",
        "clientsecret",
        "password",
        "secret",
    }
)


def _is_sensitive_query_name(name: str) -> bool:
    normalized = name.casefold().replace("-", "").replace("_", "")
    return normalized in _SENSITIVE_QUERY_NAMES or normalized.endswith("token")


def private_capability_path(path: str) -> str | None:
    """Return a content-free service path, including on unrouted failures."""
    # Both ASGI and OTel's pre-span URL builder supply the decoded path. An
    # escaped '?', '#' or newline in a malformed run ID is still path data;
    # do not parse it as a query/fragment separator or normalize handler input.
    # Only the legacy artifact metadata family is private here, not all runs.
    if _PRIVATE_LEGACY_WORKFLOW_TARGET.search(path):
        return _LEGACY_WORKFLOW_ARTIFACT_MARKER
    match = _PRIVATE_TARGET.search(path)
    return f"/api/{match[1]}" if match else None


def private_capability_tool_name(name: str | None) -> bool:
    """Classify private tool diagnostics, including registered aliases.

    These tool families carry artifact source, paths, screen prompts and input
    text. Their operational audit rows remain, but payloads never belong in
    plaintext diagnostic stores, even in NORMAL or PUBLIC privacy mode. This
    is only a redaction rule and grants no execution or content authority.
    """
    return isinstance(name, str) and _PRIVATE_TOOL.search(name) is not None


def redact_private_capability_tool_calls(value: Any) -> Any:
    """Copy recognized private calls/results for diagnostics, never execution.

    Flat canonical, OpenAI function and Anthropic tool_use calls share the
    existing name classifier. Only structural call/history fields are walked;
    ordinary tool arguments and prose remain untouched. Original call IDs may
    identify later result messages within THIS value; they grant no authority
    and are never registered or retained outside this pure transformation.
    """
    private_ids: set[str] = set()

    def private_call(item: dict) -> bool:
        function = item.get("function")
        return "role" not in item and (
            private_capability_tool_name(item.get("name"))
            or isinstance(function, dict)
            and private_capability_tool_name(function.get("name"))
        )

    def collect(item: Any, depth: int = 0, call_position: bool = True) -> None:
        if depth > 16:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                collect(child, depth + 1, call_position)
        elif isinstance(item, dict):
            if item.get("role") in ("tool", "function") or item.get("type") == "tool_result":
                return
            if not call_position and item.get("type") != "tool_use":
                return
            if private_call(item) and isinstance(item.get("id"), str):
                private_ids.add(item["id"])
            for key in ("tool_calls", "content"):
                if isinstance(item.get(key), (dict, list, tuple)):
                    collect(item[key], depth + 1, key == "tool_calls")

    def elide(item: dict, fields: tuple[str, ...]) -> dict:
        safe = {key: item[key] for key in fields
                if isinstance(item.get(key), str)}
        safe["_private_capability"] = True
        for key in ("arguments", "input", "args", "parameters", "content"):
            if key in item:
                safe[key] = {"_private_capability": True}
        return safe

    def redact(item: Any, depth: int = 0, call_position: bool = True) -> Any:
        if depth > 16:
            return "<max-depth>"
        if isinstance(item, (list, tuple)):
            copied = [redact(child, depth + 1, call_position) for child in item]
            return tuple(copied) if isinstance(item, tuple) else copied
        if not isinstance(item, dict):
            return item
        if item.get("role") in ("tool", "function"):
            if (private_capability_tool_name(item.get("name"))
                    or isinstance(item.get("tool_call_id"), str)
                    and item["tool_call_id"] in private_ids):
                return elide(item, ("role", "name", "tool_call_id"))
            return item
        if item.get("type") == "tool_result":
            if (isinstance(item.get("tool_use_id"), str)
                    and item["tool_use_id"] in private_ids):
                safe = elide(item, ("type", "tool_use_id"))
                if isinstance(item.get("is_error"), bool):
                    safe["is_error"] = item["is_error"]
                return safe
            return item
        # Provider content blocks and ordinary result payloads are data. A
        # name inside text/image data does not make that data a tool call.
        if not call_position and item.get("type") != "tool_use":
            return item
        if private_call(item):
            safe = elide(item, ("id", "type", "name"))
            if "function" in item:
                function = item["function"]
                safe["function"] = (
                    elide(function, ("name",)) if isinstance(function, dict)
                    else {"_private_capability": True}
                )
            return safe
        safe = dict(item)
        for key in ("tool_calls", "content"):
            if isinstance(item.get(key), (dict, list, tuple)):
                safe[key] = redact(item[key], depth + 1, key == "tool_calls")
        return safe

    collect(value)
    return redact(value)


def redact_sensitive_query_string(query_string: bytes) -> bytes:
    """Return a log-safe query string with credential values replaced.

    The returned bytes are used only after application handling is complete,
    so normalizing URL encoding cannot affect routing or request parameters.
    Malformed input fails closed: if parsing itself fails, the whole query is
    replaced rather than risking credential disclosure.
    """
    if not query_string:
        return query_string

    try:
        pairs = parse_qsl(
            query_string.decode("latin-1"),
            keep_blank_values=True,
            errors="replace",
        )
    except (UnicodeError, ValueError):
        return b"redacted"

    changed = False
    redacted_pairs: list[tuple[str, str]] = []
    for name, value in pairs:
        if _is_sensitive_query_name(name):
            value = _REDACTED_VALUE
            changed = True
        redacted_pairs.append((name, value))

    if not changed:
        return query_string
    return urlencode(redacted_pairs, doseq=True).encode("ascii")


class SensitiveQueryStringRedactionMiddleware:
    """ASGI middleware that redacts the scope at Uvicorn's logging seam."""

    def __init__(self, app: Callable[..., Awaitable[Any]]) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> Any:
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        # Artifact and computer IDs, keys and file paths are private metadata,
        # including on failed/unauthenticated requests and agent-prefixed URLs.
        private_path = private_capability_path(scope.get("path", ""))
        query = scope.get("query_string", b"")
        redacted_query = b"" if private_path else redact_sensitive_query_string(query)
        if not private_path and redacted_query == query:
            return await self.app(scope, receive, send)

        def redact_scope() -> None:
            scope["query_string"] = redacted_query
            if private_path:
                scope["path"] = private_path
                scope["raw_path"] = private_path.encode("ascii")

        async def send_with_redacted_scope(message: dict) -> None:
            if message.get("type") == "http.response.start":
                # Uvicorn's RequestResponseCycle.send reads this exact scope
                # immediately after the wrapped send is entered.
                original_query = scope.get("query_string", b"")
                original_path = scope.get("path", "")
                original_raw_path = scope.get("raw_path")
                redact_scope()
                try:
                    await send(message)
                finally:
                    # A streaming response may continue running application
                    # code after response-start. Keep its request semantics
                    # unchanged after the synchronous access-log seam.
                    scope["query_string"] = original_query
                    if private_path:
                        scope["path"] = original_path
                        if original_raw_path is None:
                            scope.pop("raw_path", None)
                        else:
                            scope["raw_path"] = original_raw_path
                return
            await send(message)

        try:
            return await self.app(scope, receive, send_with_redacted_scope)
        except BaseException:
            # ServerErrorMiddleware sits outside user middleware. Its fallback
            # response and error handler see this scope after an escaped error,
            # including an outer delivery refusal before response-start. Keep
            # that terminal path redacted too; successful streams still restore
            # their request semantics after each start-send as above.
            redact_scope()
            raise
