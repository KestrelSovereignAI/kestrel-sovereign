"""Credential redaction at Uvicorn's ASGI access-log seam (#2429)."""

from __future__ import annotations

from urllib.parse import parse_qs

import pytest
from uvicorn.protocols.utils import get_path_with_query_string

from kestrel_sovereign.security.access_log import (
    SensitiveQueryStringRedactionMiddleware,
    redact_sensitive_query_string,
)


def test_sensitive_query_values_are_redacted_without_dropping_safe_fields():
    original = (
        b"session_id=s-1&api_key=top-secret&access_token=bearer-secret&"
        b"client-secret=client-secret-value&page=2"
    )

    redacted = redact_sensitive_query_string(original)
    parsed = parse_qs(redacted.decode("ascii"))

    assert parsed == {
        "session_id": ["s-1"],
        "api_key": ["redacted"],
        "access_token": ["redacted"],
        "client-secret": ["redacted"],
        "page": ["2"],
    }
    assert b"top-secret" not in redacted
    assert b"bearer-secret" not in redacted
    assert b"client-secret-value" not in redacted


def test_percent_encoded_sensitive_name_is_redacted():
    redacted = redact_sensitive_query_string(b"api%5Fkey=encoded-secret&safe=yes")

    assert b"encoded-secret" not in redacted
    assert parse_qs(redacted.decode("ascii"))["api_key"] == ["redacted"]


def test_safe_query_is_returned_byte_for_byte():
    query = b"session_id=a%2Fb&cursor=one%20two"
    assert redact_sensitive_query_string(query) is query


@pytest.mark.asyncio
async def test_multi_agent_request_is_redacted_only_during_response_start_send():
    secret = b"live-sse-key-2429"
    query = b"api_key=" + secret + b"&session_id=chat-7"
    scope = {
        "type": "http",
        "path": "/api/agents/Kite/api/agent/notifications/sse",
        "raw_path": b"/api/agents/Kite/api/agent/notifications/sse",
        "query_string": query,
    }
    observed: dict[str, object] = {}

    async def inner_app(inner_scope, receive, send):
        observed["before_start"] = inner_scope["query_string"]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        observed["after_start"] = inner_scope["query_string"]
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def receive():
        return {"type": "http.disconnect"}

    async def uvicorn_send(message):
        if message["type"] == "http.response.start":
            # Uvicorn 0.48 calls this helper against the same scope before
            # emitting its access log record.
            observed["access_target"] = get_path_with_query_string(scope)

    middleware = SensitiveQueryStringRedactionMiddleware(inner_app)
    await middleware(scope, receive, uvicorn_send)

    assert observed["before_start"] == query
    assert observed["after_start"] == query
    assert scope["query_string"] == query
    target = str(observed["access_target"])
    assert secret.decode() not in target
    assert "api_key=redacted" in target
    assert "session_id=chat-7" in target
