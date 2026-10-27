"""Private capability leases cover real ASGI delivery, not a response queue."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kestrel_sovereign.security.private_delivery import (
    AuthorizedCapabilityResponse,
    PrivateDeliveryMiddleware,
)
from tests.unit.test_host_capability_authority import Action, Admission


def scope():
    return {"type": "http", "method": "GET", "path": "/api/visual-computer/frame",
            "headers": [], "query_string": b"", "scheme": "http",
            "server": ("test", 80), "client": ("test", 1234), "root_path": ""}


async def receive():
    return {"type": "http.request", "body": b"", "more_body": False}


def private_response(a, request, *, build=None, prepare=None, resolved=None):
    @asynccontextmanager
    async def resolve(_request):
        if resolved is not None:
            resolved.set()
        with a.bind():
            yield a.context

    async def default_build(_context):
        return Response(b"private-screen", media_type="image/png")

    return AuthorizedCapabilityResponse(
        request, authority=a.authority, resolve_context=resolve,
        operation=lambda context, value: a.authority.operation(
            context, Action.OBSERVE, "visual-computer", target=value),
        build=build or default_build, prepare=prepare,
        error_response=lambda error, prepared: JSONResponse(
            {"error": "visual_access_denied"}, status_code=403),
    )


@pytest.mark.asyncio
async def test_missing_outer_boundary_refuses_before_resolving_private_context():
    a = Admission()
    resolved = asyncio.Event()
    messages = []

    async def send(message):
        messages.append(message)

    s = scope()
    await private_response(a, Request(s, receive), resolved=resolved)(s, receive, send)
    assert not resolved.is_set()
    assert messages[0]["status"] == 403
    assert b"private-screen" not in messages[-1]["body"]


@pytest.mark.asyncio
async def test_private_response_retains_authority_until_actual_body_acknowledgement():
    a = Admission()
    body_started = asyncio.Event()
    body_ack = asyncio.Event()
    transitioned = asyncio.Event()
    messages = []

    async def endpoint(s, recv, send):
        await private_response(a, Request(s, recv))(s, recv, send)

    async def send(message):
        if message["type"] == "http.response.body":
            body_started.set()
            await body_ack.wait()
        messages.append(message)

    async def transition():
        async with a.gate:
            a.current = False
            transitioned.set()

    task = asyncio.create_task(PrivateDeliveryMiddleware(endpoint)(scope(), receive, send))
    await asyncio.wait_for(body_started.wait(), timeout=1)
    writer = asyncio.create_task(transition())
    await asyncio.sleep(0)
    assert not transitioned.is_set()
    assert a.gate.locked()
    body_ack.set()
    await asyncio.wait_for(task, timeout=1)
    await asyncio.wait_for(writer, timeout=1)
    assert transitioned.is_set()
    assert messages[-1]["body"] == b"private-screen"
    headers = dict(messages[0]["headers"])
    assert b"no-store" in headers[b"cache-control"]
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert headers[b"content-security-policy"] == b"sandbox; default-src 'none'"


@pytest.mark.asyncio
async def test_body_preparation_does_not_block_revocation_and_build_rechecks_authority():
    a = Admission()
    preparing = asyncio.Event()
    continue_parse = asyncio.Event()
    built = False
    messages = []

    async def prepare():
        preparing.set()
        await continue_parse.wait()
        return "untrusted input"

    async def build(context, value):
        nonlocal built
        built = True
        return Response(b"private-screen")

    async def endpoint(s, recv, send):
        await private_response(a, Request(s, recv), build=build, prepare=prepare)(s, recv, send)

    async def send(message):
        messages.append(message)

    task = asyncio.create_task(PrivateDeliveryMiddleware(endpoint)(scope(), receive, send))
    await asyncio.wait_for(preparing.wait(), timeout=1)
    await asyncio.wait_for(a.gate.acquire(), timeout=1)
    a.current = False
    a.gate.release()
    continue_parse.set()
    await asyncio.wait_for(task, timeout=1)
    assert not built
    assert messages[0]["status"] == 403


@pytest.mark.asyncio
async def test_transport_failure_releases_operation_without_a_second_private_body():
    a = Admission()
    bodies = 0

    async def endpoint(s, recv, send):
        await private_response(a, Request(s, recv))(s, recv, send)

    async def send(message):
        nonlocal bodies
        if message["type"] == "http.response.body":
            bodies += 1
            raise BrokenPipeError("closed transport")

    with pytest.raises(BrokenPipeError):
        await PrivateDeliveryMiddleware(endpoint)(scope(), receive, send)
    assert bodies == 1
    assert not a.gate.locked()


@pytest.mark.asyncio
async def test_public_and_non_http_requests_do_not_require_private_admission():
    messages = []

    async def public(s, recv, send):
        await send({"type": "custom.public", "scope_type": s["type"]})

    async def send(message):
        messages.append(message)

    middleware = PrivateDeliveryMiddleware(public)
    await middleware(scope(), receive, send)
    await middleware({"type": "lifespan"}, receive, send)
    assert [m["scope_type"] for m in messages] == ["http", "lifespan"]
