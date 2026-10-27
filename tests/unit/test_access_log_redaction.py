"""Credential redaction at Uvicorn's ASGI access-log seam (#2429)."""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from urllib.parse import parse_qs, unquote

import pytest
from uvicorn.protocols.utils import get_path_with_query_string

from kestrel_sovereign.security.access_log import (
    SensitiveQueryStringRedactionMiddleware,
    PRIVATE_CAPABILITY_URL_PATTERN,
    private_capability_path,
    redact_sensitive_query_string,
)


LEGACY_PRIVATE_PATHS = (
    "/api/workflows/runs/private-run/artifacts/private-artifact",
    "/api/workflows/runs/private-run/artifacts",
    "/api/workflows/runs//artifacts/private-artifact",
    "/api/workflows/runs/private-run/extra/artifacts/private-artifact/invalid",
    "/api/workflows/runs/private%3Frun/artifacts/private-artifact",
    "/api/workflows/runs/private%23run/artifacts/private-artifact",
    "/api/workflows/runs/private%0Arun/artifacts/private-artifact",
    "/api/workflows/runs/private%00run/artifacts/private-artifact",
    "/api/workflows/runs/private%2Frun/artifacts/private-artifact",
    "/api/workflows/runs/private%not-hex/artifacts/private-artifact",
    "/%61pi/workflows/runs/private-run/%61rtifacts/private-artifact",
    "%2fapi%2fworkflows%2fruns%2fprivate-run%2fartifacts%2fprivate-artifact",
    "/API/WORKFLOWS/RUNS/private-run/ARTIFACTS/private-artifact",
    "/api/workflows/runs/private-run/artifacts%3Fprivate-artifact",
    "/api/workflows/runs/private-run/artifacts%23private-artifact",
)


@pytest.mark.parametrize("path", LEGACY_PRIVATE_PATHS)
@pytest.mark.parametrize("prefix", ["", "/api/agents/private-agent", "/api/agent-routes/cHJpdmF0ZQ"])
def test_legacy_artifact_family_is_private_in_raw_and_decoded_paths(path, prefix):
    for representation in (prefix + path, unquote(prefix + path)):
        assert private_capability_path(representation) == "/api/workflows/artifact-metadata"
        # The real OTel pre-span URL builder appends decoded scope.path to the
        # authority, without a query. Neither '?' nor '#' here is a separator.
        assert re.search(PRIVATE_CAPABILITY_URL_PATTERN, "http://localhost" + representation)


@pytest.mark.parametrize("path", [
    "/health", "/api/workflows/runs", "/api/workflows/runs/run-1",
    "/api/workflows/runs/run-1/stream", "/api/workflows/runs/run-1/controls/pause",
    "/api/workflows/runs/run-1/artifacts-extra", "/api/workflows/runs/run-1/artifactsXYZ/file",
    "/api/workflows/run/run-1/artifacts/file", "/api/workflows/artifacts/file",
    "/api/agents/Kite/api/workflows/runs/run-1/stream",
])
def test_legacy_classifier_does_not_claim_all_workflow_urls(path):
    assert private_capability_path(path) is None
    assert re.search(PRIVATE_CAPABILITY_URL_PATTERN, "http://localhost" + path) is None


def test_repeated_run_prefixes_remain_bounded_before_authentication():
    """A malformed target cannot monopolize the ASGI loop while classifying.

    The amplification makes a quadratic scan reliably exceed the generous
    child deadline. The owned process also bounds a regressed regex itself,
    which cannot be interrupted by an event-loop or thread timeout.
    """
    probe = r'''
import importlib.util
import re
import sys

spec = importlib.util.spec_from_file_location("private_path_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
path = "/api/workflows/runs/x" * 65536
for suffix, private in [("", False), ("/artifacts-extra", False),
                        ("/artifacts/private-id", True)]:
    target = path + suffix
    assert bool(module.private_capability_path(target)) is private, "path classification"
    assert bool(re.search(module.PRIVATE_CAPABILITY_URL_PATTERN,
                          "http://localhost" + target)) is private, "pre-span exclusion"
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe, inspect.getfile(private_capability_path)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("path", LEGACY_PRIVATE_PATHS)
async def test_legacy_artifact_access_send_preserves_decoded_handler_path(path):
    decoded = unquote("/api/agents/private-agent" + path)
    query = b"path=private-file&signature=private-signature"
    scope = {"type": "http", "path": decoded, "raw_path": path.encode(), "query_string": query}
    observed = []

    async def inner(inner_scope, receive, send):
        assert inner_scope["path"] == decoded and inner_scope["query_string"] == query
        await send({"type": "http.response.start", "status": 404, "headers": []})
        assert inner_scope["path"] == decoded and inner_scope["query_string"] == query
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        assert inner_scope["raw_path"] == path.encode()

    async def send(message):
        if message["type"] == "http.response.start":
            observed.append(get_path_with_query_string(scope))

    await SensitiveQueryStringRedactionMiddleware(inner)(scope, None, send)
    assert observed == ["/api/workflows/artifact-metadata"]
    assert scope["path"] == decoded and scope["query_string"] == query


@pytest.mark.asyncio
@pytest.mark.parametrize("during_send", [False, True])
async def test_legacy_artifact_escaped_error_keeps_outer_scope_private(during_send):
    path = "/api/agents/private-agent/api/workflows/runs/private?run\n/artifacts/private-id"
    scope = {"type": "http", "path": path, "raw_path": path.encode(), "query_string": b"signature=private-value"}
    failure = RuntimeError("private-exception")

    async def inner(scope, receive, send):
        if during_send:
            await send({"type": "http.response.start", "status": 500, "headers": []})
        raise failure

    async def send(message):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        await SensitiveQueryStringRedactionMiddleware(inner)(scope, None, send)
    assert caught.value is failure
    assert get_path_with_query_string(scope) == "/api/workflows/artifact-metadata"


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


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", ["artifacts", "visual-computer"])
@pytest.mark.parametrize("prefix", ["", "/agent", "/api/agents/Kite", "/api/agent-routes/a2l0ZQ"])
@pytest.mark.parametrize("query", [b"", b"path=private%2Ffile.md&cursor=private-cursor"])
async def test_artifact_identifiers_are_redacted_without_changing_handler_semantics(capability, prefix, query):
    path = prefix + f"/api/{capability}/publications/private-publication-key"
    scope = {"type": "http", "path": path, "raw_path": path.encode(), "query_string": query}
    observed = []
    async def inner(inner_scope, receive, send):
        assert inner_scope["path"] == path and inner_scope["query_string"] == query
        await send({"type": "http.response.start", "status": 403, "headers": []})
        assert inner_scope["path"] == path and inner_scope["query_string"] == query
    async def receive():
        return {"type": "http.disconnect"}
    async def send(message):
        observed.append(get_path_with_query_string(scope))
    await SensitiveQueryStringRedactionMiddleware(inner)(scope, receive, send)
    assert observed == [f"/api/{capability}"]
    assert scope["path"] == path and scope["raw_path"] == path.encode() and scope["query_string"] == query


@pytest.mark.asyncio
@pytest.mark.parametrize("during_send", [False, True])
@pytest.mark.parametrize("capability", [None, "artifacts", "visual-computer"])
async def test_escaped_error_keeps_scope_redacted_for_outer_error_handler(during_send, capability):
    path = f"/api/{capability}/private-id/file" if capability else "/events"
    query = b"path=private-file" if capability else b"api_key=private-key"
    scope = {"type": "http", "path": path, "raw_path": path.encode(), "query_string": query}
    failure = RuntimeError("refused before response start")
    async def inner(scope, receive, send):
        if during_send:
            await send({"type": "http.response.start", "status": 200, "headers": []})
        raise failure
    async def send(message):
        raise failure
    with pytest.raises(RuntimeError) as error:
        await SensitiveQueryStringRedactionMiddleware(inner)(scope, None, send)
    assert error.value is failure
    assert get_path_with_query_string(scope) == (f"/api/{capability}" if capability else "/events?api_key=redacted")
