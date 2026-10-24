"""Real private ASGI publication/view/edit flow against both database engines."""

import asyncio
import base64
from contextlib import asynccontextmanager

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from kestrel_sdk.operator import OperatorAuthorizationError

from kestrel_sovereign.artifacts.http import ArtifactDeliveryMiddleware, create_artifact_router
from tests.integration.test_artifact_repository import repository as _repository_fixture
from tests.integration.test_artifact_service import access as _access_fixture

repository = _repository_fixture
access = _access_fixture


@pytest_asyncio.fixture
async def api(access):
    @asynccontextmanager
    async def resolve(request):
        # This fixture is the trusted host resolver, not production auth proof.
        if request.headers.get("authorization") != "Bearer test-only":
            raise OperatorAuthorizationError("caller is not authenticated")
        context = access.context()
        with access.bind(context):
            yield context
    app = FastAPI()
    app.add_middleware(ArtifactDeliveryMiddleware)
    app.include_router(create_artifact_router(access.service, resolve_context=resolve))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client, app


AUTH = {"Authorization": "Bearer test-only"}


async def publish(client, *, key="publish", ref=None, content=b"# Hello", media_type="text/markdown"):
    staged = await client.post("/api/artifacts/staging", headers=AUTH, json={
        "publication_key": key, "path": "report.md", "media_type": media_type,
        "content_base64": base64.b64encode(content).decode(),
    })
    assert staged.status_code == 200, staged.text
    commit = {"idempotency_key": key, "title": "Saved report",
              "manifest": {"files": [staged.json()], "entrypoint": "report.md"}}
    if ref:
        commit.update(artifact_id=ref["artifact_id"], expected_revision_id=ref["revision_id"])
    response = await client.post("/api/artifacts/publications", headers=AUTH, json=commit)
    return response, commit


def revision_url(ref):
    return f'/api/artifacts/{ref["artifact_id"]}/revisions/{ref["revision_id"]}'


@pytest.mark.asyncio
async def test_private_publish_edit_history_and_delete(api):
    client, _ = api
    initial, request = await publish(client)
    assert initial.status_code == 200, initial.text
    ref = initial.json()["ref"]
    metadata = await client.get(revision_url(ref), headers=AUTH)
    assert metadata.json()["title"] == "Saved report"
    assert (await client.get(revision_url(ref) + "/file", params={"path": "report.md"}, headers=AUTH)).content == b"# Hello"
    second, _ = await publish(client, key="edit", ref=ref, content=b"# Edited")
    assert second.status_code == 200
    assert second.json()["ref"]["artifact_id"] == ref["artifact_id"]
    history = await client.get(f'/api/artifacts/{ref["artifact_id"]}/history', headers=AUTH)
    assert len(history.json()["items"]) == 2
    assert (await client.get("/api/artifacts/publications/edit", headers=AUTH)).json() == second.json()
    conflict, _ = await publish(client, key="conflict", ref=ref)
    assert conflict.status_code == 409
    # A lost acknowledgement is resolved by the original exact request/key.
    retry = await client.post("/api/artifacts/publications", headers=AUTH, json=request)
    assert retry.json() == initial.json()
    gone = await client.delete(f'/api/artifacts/{ref["artifact_id"]}', headers=AUTH,
                              params={"expected_revision_id": second.json()["ref"]["revision_id"]})
    assert gone.status_code == 204
    assert (await client.get(revision_url(ref), headers=AUTH)).status_code == 404
    assert (await client.post("/api/artifacts/publications", headers=AUTH, json=request)).status_code == 404


@pytest.mark.asyncio
async def test_authentication_scope_input_and_private_response_headers(api):
    client, _ = api
    denied = await client.get("/api/artifacts")
    assert denied.status_code == 403
    assert "no-store" in denied.headers["cache-control"]
    initial, request = await publish(client)
    ref = initial.json()["ref"]
    assert (await client.get(revision_url(ref))).status_code == 403
    forged = await client.post("/api/artifacts/publications", headers=AUTH, json={
        **request, "scope": {"principal_id": "victim"},
    })
    assert forged.status_code == 400
    assert "victim" not in forged.text
    for response in (initial, await client.get(revision_url(ref), headers=AUTH), denied, forged):
        assert response.headers["cache-control"] == "private, no-store, max-age=0"
        assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.asyncio
async def test_untrusted_mime_cannot_execute_on_application_origin(api):
    client, _ = api
    response, _ = await publish(client, content=b"<script>parent.alert(1)</script>", media_type="text/html")
    body = await client.get(revision_url(response.json()["ref"]) + "/file", headers=AUTH, params={"path": "report.md"})
    assert body.headers["content-type"] == "application/octet-stream"
    assert body.headers["content-disposition"].startswith("attachment;")
    assert "sandbox" in body.headers["content-security-policy"]


@pytest.mark.asyncio
async def test_bounded_body_and_validation_do_not_echo_private_content(api, access):
    from dataclasses import replace
    client, _ = api
    access.service.repository.limits = replace(access.service.repository.limits, max_file_bytes=1)
    response = await client.post("/api/artifacts/staging", headers=AUTH, json={
        "publication_key": "large", "path": "report.md", "media_type": "text/plain",
        "content_base64": "private-content" * 5000,
    })
    assert response.status_code == 413
    assert "private-content" not in response.text
    malformed = await client.post("/api/artifacts/staging", headers=AUTH, json={"private-content": "secret"})
    assert malformed.status_code == 400
    assert "secret" not in malformed.text


@pytest.mark.asyncio
async def test_authority_gate_retained_until_body_send(api, access):
    client, app = api
    initial, _ = await publish(client)
    ref = initial.json()["ref"]
    path = revision_url(ref) + "/file"
    entered, release, revoked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            entered.set()
            await release.wait()
        messages.append(message)

    async def revoke():
        async with access.authority_gate:
            access.granted = False
            revoked.set()

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(),
             "query_string": b"path=report.md", "root_path": "", "server": ("test", 80),
             "client": ("test", 1), "headers": [(b"authorization", b"Bearer test-only")]}
    serving = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(entered.wait(), 3)
    revoking = asyncio.create_task(revoke())
    try:
        await asyncio.sleep(0)
        assert not revoked.is_set()
        release.set()
        await asyncio.wait_for(serving, 3)
        await asyncio.wait_for(revoking, 3)
        assert b"".join(item.get("body", b"") for item in messages) == b"# Hello"
        assert (await client.get(revision_url(ref), headers=AUTH)).status_code == 403
    finally:
        release.set()
        for task in (serving, revoking):
            if not task.done():
                task.cancel()
        await asyncio.gather(serving, revoking, return_exceptions=True)


@pytest.mark.asyncio
async def test_slow_upload_has_deadline_and_releases_authority(access):
    @asynccontextmanager
    async def resolve(request):
        with access.bind(access.context()) as context:
            yield context
    app = FastAPI()
    app.add_middleware(ArtifactDeliveryMiddleware)
    app.include_router(create_artifact_router(access.service, resolve_context=resolve, timeout_seconds=0.05))
    entered = asyncio.Event()

    async def slow_body():
        entered.set()
        await asyncio.Event().wait()
        yield b"{}"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await asyncio.wait_for(client.post("/api/artifacts/staging", content=slow_body(),
                                     headers={"Content-Type": "application/json"}), 1)
    assert entered.is_set()
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "artifact_outcome_unknown"
    assert not access.authority_gate.locked()
    assert not access.authority._bindings
    assert not access.authority._operations


@pytest.mark.asyncio
async def test_idle_upload_does_not_hold_privacy_or_revocation_gates(api, access):
    client, _ = api
    entered, release, revoked = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def upload():
        entered.set()
        await release.wait()
        yield b'{"publication_key":"queued","path":"a.txt","media_type":"text/plain","content_base64":"YQ=="}'

    async def revoke():
        async with access.authority_gate:
            access.granted = False
            revoked.set()

    sending = asyncio.create_task(client.post("/api/artifacts/staging", headers={
        **AUTH, "Content-Type": "application/json"}, content=upload()))
    await asyncio.wait_for(entered.wait(), 3)
    revoking = asyncio.create_task(revoke())
    try:
        await asyncio.wait_for(revoked.wait(), 0.3)
        assert not access.holder.gate.locked()
        release.set()
        response = await asyncio.wait_for(sending, 3)
        assert response.status_code == 403
        await asyncio.wait_for(revoking, 3)
    finally:
        release.set()
        for task in (sending, revoking):
            if not task.done():
                task.cancel()
        await asyncio.gather(sending, revoking, return_exceptions=True)


@pytest.mark.asyncio
async def test_http_post_commit_lookup_failure_never_reports_plain_failure(api, access, monkeypatch):
    client, _ = api
    response = await client.post("/api/artifacts/staging", headers=AUTH, json={
        "publication_key": "lookup", "path": "a.txt", "media_type": "text/plain", "content_base64": "YQ==",
    })
    file = response.json()
    check = access.check
    failed = False
    def lookup(*args):
        if failed:
            raise PermissionError("private lookup failure")
        return check(*args)
    access.check = lookup
    transaction = access.service.repository.db.transaction
    @asynccontextmanager
    async def fail_after_commit(*args, **kwargs):
        nonlocal failed
        async with transaction(*args, **kwargs):
            yield
        failed = True
    monkeypatch.setattr(access.service.repository.db, "transaction", fail_after_commit)
    response = await client.post("/api/artifacts/publications", headers=AUTH, json={
        "idempotency_key": "lookup", "title": "private",
        "manifest": {"files": [file], "entrypoint": "a.txt"},
    })
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "artifact_outcome_unknown"
    monkeypatch.setattr(access.service.repository.db, "transaction", transaction)
    failed = False
    assert (await client.get("/api/artifacts/publications/lookup", headers=AUTH)).json()["ref"]


@pytest.mark.asyncio
async def test_http_failure_after_receipt_encoding_is_unknown(api, access, monkeypatch):
    from kestrel_sdk.artifacts import PublicationReceipt

    from kestrel_sovereign.artifacts import http
    client, _ = api
    original_json = http._json
    check = access.check
    failed = False
    def encode(*args, **kwargs):
        nonlocal failed
        response = original_json(*args, **kwargs)
        if isinstance(args[0], PublicationReceipt):
            failed = True
        return response
    def lookup(*args):
        if failed:
            raise RuntimeError("private infrastructure diagnostic")
        return check(*args)
    access.check = lookup
    monkeypatch.setattr(http, "_json", encode)
    response, request = await publish(client)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "artifact_outcome_unknown"
    monkeypatch.setattr(http, "_json", original_json)
    failed = False
    receipt = await client.get("/api/artifacts/publications/publish", headers=AUTH)
    assert receipt.status_code == 200
    assert receipt.json()["idempotency_key"] == request["idempotency_key"]


@pytest.mark.asyncio
async def test_link_resolver_always_receives_real_target(api, access):
    client, _ = api
    receipt, _ = await publish(client)
    check = access.check
    def require_link_target(context, action, link):
        from kestrel_sovereign.artifacts.service import ArtifactAction
        if action == ArtifactAction.LINK and link is None:
            raise OperatorAuthorizationError("target required")
        return check(context, action, link)
    access.check = require_link_target
    response = await client.post("/api/artifacts/links", headers=AUTH, json={
        "ref": receipt.json()["ref"], "context_kind": "conversation", "context_id": "conversation",
    })
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_current_revision_route_is_authorized_private_and_tombstone_aware(api, access):
    client, _ = api
    first, _ = await publish(client)
    ref = first.json()["ref"]
    before = access.service.repository.clock()
    from datetime import timedelta
    access.service.repository.clock = lambda: before - timedelta(days=1)
    second, _ = await publish(client, key="head-edit", ref=ref, content=b"# Current")
    current = second.json()["ref"]
    url = f'/api/artifacts/{ref["artifact_id"]}/head'
    assert (await client.get(url)).status_code == 403
    result = await client.get(url, headers=AUTH)
    assert result.status_code == 200
    assert result.json()["ref"] == current
    assert result.headers["cache-control"] == "private, no-store, max-age=0"
    access.granted = False
    assert (await client.get(url, headers=AUTH)).status_code == 403
    access.granted = True
    await client.delete(f'/api/artifacts/{ref["artifact_id"]}', headers=AUTH,
                        params={"expected_revision_id": current["revision_id"]})
    deleted = await client.get(url, headers=AUTH)
    assert deleted.status_code == 404 and deleted.json()["error"]["code"] == "artifact_not_found"


@pytest.mark.asyncio
async def test_corrupt_head_is_storage_failure_not_a_deletion_signal(api, access):
    client, _ = api
    first, _ = await publish(client)
    ref = first.json()["ref"]
    await access.service.repository.db.execute(
        "UPDATE artifact_heads SET revision_id = ? WHERE artifact_id = ?",
        ("missing-revision", ref["artifact_id"]),
    )
    response = await client.get(f'/api/artifacts/{ref["artifact_id"]}/head', headers=AUTH)
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "artifact_integrity_failure"}}


@pytest.mark.asyncio
async def test_missing_outer_delivery_boundary_refuses_before_repository_access(access):
    resolved = False
    @asynccontextmanager
    async def resolve(request):
        nonlocal resolved
        resolved = True
        with access.bind(access.context()) as context:
            yield context
    app = FastAPI()
    app.include_router(create_artifact_router(access.service, resolve_context=resolve))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/artifacts")
    assert response.status_code == 403
    assert not resolved


@pytest.mark.asyncio
async def test_blocked_outer_send_obeys_deadline_and_releases_original_leases(access):
    @asynccontextmanager
    async def resolve(request):
        with access.bind(access.context()) as context:
            yield context
    app = FastAPI()
    @app.middleware("http")
    async def buffering(request, call_next):
        return await call_next(request)
    app.add_middleware(ArtifactDeliveryMiddleware)
    app.include_router(create_artifact_router(access.service, resolve_context=resolve, timeout_seconds=0.05))
    entered, cancelled = asyncio.Event(), asyncio.Event()
    bodies = []
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    async def send(message):
        if message["type"] == "http.response.start" and message["status"] == 200:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        if message["type"] == "http.response.body":
            bodies.append(message.get("body", b""))
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
             "http_version": "1.1", "method": "GET", "scheme": "http", "path": "/api/artifacts",
             "raw_path": b"/api/artifacts", "query_string": b"", "root_path": "",
             "server": ("test", 80), "client": ("test", 1), "headers": []}
    serving = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        # Observe cancellation by the artifact deadline, not by our test timeout.
        await asyncio.wait_for(cancelled.wait(), 0.5)
        outcome = await asyncio.wait_for(asyncio.gather(serving, return_exceptions=True), 2)
        assert isinstance(outcome[0], BaseException)
        assert b'"items"' not in b"".join(bodies)
        assert not access.authority_gate.locked()
        assert not access.holder.gate.locked()
        assert not access.authority._bindings
        assert not access.authority._operations
    finally:
        if not serving.done():
            serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
