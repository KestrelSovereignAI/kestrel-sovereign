"""Private revision routes constructed only with a host authentication resolver.

The resolver owns an ArtifactAuthority.bind_authenticated lifetime through the
entire response. It derives tenant/principal/agent from host authentication,
never request JSON or URL selectors. No router is installed at module import.
Host middleware remains responsible for credential verification and CSRF.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import math
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import APIRouter, Request
from kestrel_sdk.artifacts import ArtifactCommit, ArtifactLink, ArtifactRef
from kestrel_sdk.operator import OperatorAuthorizationError, OperatorContext
from kestrel_sdk.resources import (
    ResourceConflict,
    ResourceNotFound,
    ResourceOutcomeUnknown,
)
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.responses import JSONResponse, Response

from .repository import ArtifactIntegrityError, ArtifactQuotaExceeded
from .service import ArtifactAccessDenied, ArtifactAction, AuthorizedArtifactService

ContextResolver = Callable[[Request], AbstractAsyncContextManager[OperatorContext]]
_HEADERS = {
    "Cache-Control": "private, no-store, max-age=0",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "sandbox; default-src 'none'",
}

_DELIVERY_KEY = "kestrel.artifact_delivery"


class _ArtifactDelivery:
    """One response's acknowledgement from the host's outer ASGI boundary."""

    def __init__(self):
        self.check_current = None
        self.deadline = None
        self.active = False
        self.failed = False
        self.finished = asyncio.Event()

    def begin(self, check, deadline):
        if self.check_current is not None:
            raise ArtifactAccessDenied("artifact delivery already bound")
        self.check_current, self.deadline, self.active = check, deadline, True

    def check(self):
        if not self.active or self.failed:
            raise ArtifactAccessDenied("artifact delivery is no longer active")
        self.check_current()


class ArtifactDeliveryMiddleware:
    """Install outside every buffering middleware on an artifact-serving host.

    BaseHTTPMiddleware forwards messages through task queues. Its inner send
    is not a delivery acknowledgement. Keep the endpoint's original leases
    alive until this boundary sends the final body, and run its current-policy
    checker here before each actual transport handoff. No second lease is
    acquired on this task, which would deadlock behind a queued policy writer.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        delivery = _ArtifactDelivery()
        scope[_DELIVERY_KEY] = delivery

        async def guarded_send(message):
            if delivery.check_current is None:
                await send(message)
                return
            try:
                async with asyncio.timeout_at(delivery.deadline):
                    delivery.check()
                    await send(message)
            except BaseException:
                delivery.failed = True
                delivery.finished.set()
                raise
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                delivery.finished.set()

        try:
            await self.app(scope, receive, guarded_send)
        finally:
            delivery.active = False
            delivery.finished.set()


class _InvalidRequest(ValueError):
    pass


class _Stage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    publication_key: str
    path: str
    media_type: str
    content_base64: str


async def _body(request: Request, limit: int, model):
    if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
        raise _InvalidRequest("JSON request required")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > limit:
            raise ArtifactQuotaExceeded("artifact request exceeds byte limit")
        content.extend(chunk)
    try:
        return model.model_validate_json(bytes(content))
    except ValidationError:
        # Pydantic errors can echo the complete private file in their input.
        raise _InvalidRequest("invalid artifact request") from None


def _json(value: Any, status: int = 200) -> JSONResponse:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return JSONResponse(value, status_code=status, headers=_HEADERS)


def _error(error: Exception) -> Response:
    if isinstance(error, OperatorAuthorizationError):
        status, code = 403, "artifact_access_denied"
    elif isinstance(error, ResourceNotFound):
        status, code = 404, "artifact_not_found"
    elif isinstance(error, ResourceConflict):
        status, code = 409, "artifact_conflict"
    elif isinstance(error, ResourceOutcomeUnknown):
        status, code = 503, "artifact_outcome_unknown"
    elif isinstance(error, ArtifactQuotaExceeded):
        status, code = 413, "artifact_quota_exceeded"
    elif isinstance(error, ArtifactIntegrityError):
        status, code = 503, "artifact_integrity_failure"
    elif isinstance(error, (ValueError, ValidationError)):
        status, code = 400, "invalid_artifact_request"
    else:
        status, code = 500, "artifact_service_error"
    return _json({"error": {"code": code}}, status)


class _AuthorizedResponse(Response):
    """Keep caller/privacy/grant custody until the final ASGI body is sent."""

    def __init__(self, request, service, resolve_context, action, build, timeout_seconds, prepare=None):
        super().__init__()
        self.request, self.service = request, service
        self.resolve_context, self.action, self.build = resolve_context, action, build
        self.timeout_seconds = timeout_seconds
        self.prepare = prepare

    async def __call__(self, scope, receive, send):
        started = False
        prepared = False
        delivery = scope.get(_DELIVERY_KEY)
        async def guarded_send(message):
            nonlocal started
            check()
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            if not isinstance(delivery, _ArtifactDelivery):
                raise ArtifactAccessDenied("artifact transport custody is unavailable")
            async with asyncio.timeout(self.timeout_seconds) as request_timeout:
                async with self.resolve_context(self.request) as context:
                    remaining = (context.expires_at - self.service.authority.clock()).total_seconds()
                    context_deadline = asyncio.get_running_loop().time() + max(0, remaining)
                    async with asyncio.timeout(max(0, remaining)):
                        # Authentication owns the context, but untrusted input
                        # must not retain a privacy/revocation lease while it
                        # trickles in. Recheck current policy AFTER parsing.
                        value = await self.prepare() if self.prepare is not None else None
                        async with self.service.authority.operation(
                            context, self.action, self.service.repository.storage_id,
                            link=value if isinstance(value, ArtifactLink) else None,
                        ) as check:
                            response = await self.build(context, value) if self.prepare is not None else await self.build(context)
                            prepared = True
                            check()
                            delivery.begin(check, min(request_timeout.when(), context_deadline))
                            try:
                                await response(scope, receive, guarded_send)
                                await delivery.finished.wait()
                                delivery.check()
                            finally:
                                delivery.active = False
        except Exception as error:
            # Headers already sent cannot be replaced by a second response.
            # Abort a partial transfer; never send private bytes after denial.
            if started:
                raise
            if prepared and self.request.method in {"POST", "DELETE"}:
                error = ResourceOutcomeUnknown("artifact result requires reconciliation")
            if isinstance(error, TimeoutError):
                # A timed-out mutation is never asserted to have rolled back.
                # Body upload and blocked response sends also have a deadline,
                # so a slow client cannot retain a revocation gate indefinitely.
                code = ("artifact_outcome_unknown" if self.request.method in {"POST", "DELETE"}
                        else "artifact_timeout")
                await _json({"error": {"code": code}}, 503)(scope, receive, send)
                return
            await _error(error)(scope, receive, send)


def create_artifact_router(service: AuthorizedArtifactService, *, resolve_context: ContextResolver,
                           timeout_seconds: float = 30) -> APIRouter:
    """Build routes for a lifecycle-owned host service; no anonymous fallback."""
    if not callable(resolve_context):
        raise TypeError("host artifact context resolver required")
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("artifact HTTP timeout must be positive and finite")
    router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])

    def respond(request, action, build, prepare=None):
        return _AuthorizedResponse(request, service, resolve_context, action, build, timeout_seconds, prepare)

    def page_query(request):
        try:
            limit = int(request.query_params.get("limit", "50"))
        except ValueError:
            raise _InvalidRequest("invalid page limit") from None
        return {"cursor": request.query_params.get("cursor"), "limit": limit}

    @router.get("")
    async def list_artifacts(request: Request) -> Response:
        async def build(context):
            return _json(await service.list(context, **page_query(request)))
        return respond(request, ArtifactAction.READ, build)

    @router.post("/staging")
    async def stage(request: Request) -> Response:
        async def prepare():
            maximum = service.repository.limits.max_file_bytes
            # Base64 expansion plus bounded path/media/key JSON overhead.
            value = await _body(request, 4 * ((maximum + 2) // 3) + 16384, _Stage)
            try:
                content = base64.b64decode(value.content_base64, validate=True)
            except (ValueError, binascii.Error):
                raise _InvalidRequest("invalid artifact encoding") from None
            return value, content
        async def build(context, parsed):
            value, content = parsed
            file = await service.stage(context, content, publication_key=value.publication_key,
                                       path=value.path, media_type=value.media_type)
            return _json(file)
        return respond(request, ArtifactAction.WRITE, build, prepare)

    @router.post("/publications")
    async def commit(request: Request) -> Response:
        async def build(context, value):
            return _json(await service.commit(context, value))
        return respond(request, ArtifactAction.WRITE, build,
                       lambda: _body(request, service.repository.limits.max_metadata_bytes, ArtifactCommit))

    @router.get("/publications/{key}")
    async def publication(key: str, request: Request) -> Response:
        async def build(context):
            # Null is unresolved, not a failed publication or permission to
            # replace the original request/key. The viewer retains its edit.
            return _json(await service.publication_status(context, key))
        return respond(request, ArtifactAction.WRITE, build)

    @router.get("/{artifact_id}/history")
    async def history(artifact_id: str, request: Request) -> Response:
        async def build(context):
            return _json(await service.history(context, artifact_id, **page_query(request)))
        return respond(request, ArtifactAction.READ, build)

    @router.get("/{artifact_id}/head")
    async def head(artifact_id: str, request: Request) -> Response:
        async def build(context):
            return _json(await service.head(context, artifact_id))
        return respond(request, ArtifactAction.READ, build)

    @router.get("/{artifact_id}/revisions/{revision_id}")
    async def revision(artifact_id: str, revision_id: str, request: Request) -> Response:
        async def build(context):
            ref = ArtifactRef(artifact_id=artifact_id, revision_id=revision_id)
            return _json(await service.get(context, ref))
        return respond(request, ArtifactAction.READ, build)

    @router.get("/{artifact_id}/revisions/{revision_id}/file")
    async def read(artifact_id: str, revision_id: str, request: Request) -> Response:
        async def build(context):
            ref = ArtifactRef(artifact_id=artifact_id, revision_id=revision_id)
            path = request.query_params.get("path")
            if path is None:
                raise _InvalidRequest("artifact path required")
            content = await service.read(context, ref, path)
            # Never let a supplied MIME type turn this origin into an HTML/SVG
            # application. The viewer fetches bytes and chooses safe renderers.
            return Response(content, media_type="application/octet-stream", headers={
                **_HEADERS, "Content-Disposition": 'attachment; filename="artifact.bin"',
            })
        return respond(request, ArtifactAction.READ, build)

    @router.post("/links")
    async def link(request: Request) -> Response:
        async def build(context, value):
            await service.link(context, value)
            return Response(status_code=204, headers=_HEADERS)
        return respond(request, ArtifactAction.LINK, build, lambda: _body(request, 16384, ArtifactLink))

    @router.delete("/links")
    async def unlink(request: Request) -> Response:
        async def build(context, value):
            await service.unlink(context, value)
            return Response(status_code=204, headers=_HEADERS)
        return respond(request, ArtifactAction.LINK, build, lambda: _body(request, 16384, ArtifactLink))

    @router.delete("/{artifact_id}")
    async def delete(artifact_id: str, request: Request) -> Response:
        async def build(context):
            ref = ArtifactRef(artifact_id=artifact_id,
                              revision_id=request.query_params.get("expected_revision_id"))
            await service.delete(context, ref)
            return Response(status_code=204, headers=_HEADERS)
        return respond(request, ArtifactAction.DELETE, build)

    return router
