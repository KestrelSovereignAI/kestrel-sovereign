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
from typing import Any
from urllib.parse import parse_qsl, urlencode


_REDACTED_VALUE = "redacted"
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
        if scope.get("type") != "http" or not scope.get("query_string"):
            return await self.app(scope, receive, send)

        redacted_query = redact_sensitive_query_string(scope["query_string"])
        if redacted_query == scope["query_string"]:
            return await self.app(scope, receive, send)

        async def send_with_redacted_scope(message: dict) -> None:
            if message.get("type") == "http.response.start":
                # Uvicorn's RequestResponseCycle.send reads this exact scope
                # immediately after the wrapped send is entered.
                original_query = scope["query_string"]
                scope["query_string"] = redacted_query
                try:
                    await send(message)
                finally:
                    # A streaming response may continue running application
                    # code after response-start. Keep its request semantics
                    # unchanged after the synchronous access-log seam.
                    scope["query_string"] = original_query
                return
            await send(message)

        return await self.app(scope, receive, send_with_redacted_scope)
