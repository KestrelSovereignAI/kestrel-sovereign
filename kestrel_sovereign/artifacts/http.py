"""Private revision routes constructed only with a host authentication resolver.

The resolver owns an ArtifactAuthority.bind_authenticated lifetime through the
entire response. It derives tenant/principal/agent from host authentication,
never request JSON or URL selectors. No router is installed at module import.
Host middleware remains responsible for credential verification and CSRF.
"""

from __future__ import annotations

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

from kestrel_sovereign.security.private_delivery import (
    AuthorizedCapabilityResponse, PRIVATE_RESPONSE_HEADERS,
)

from .preview import ArtifactPreviewRefused, ArtifactPreviewUnavailable
from .repository import ArtifactIntegrityError, ArtifactQuotaExceeded
from .service import ArtifactAccessDenied, ArtifactAction, AuthorizedArtifactService

ContextResolver = Callable[[Request], AbstractAsyncContextManager[OperatorContext]]


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
    return JSONResponse(value, status_code=status, headers=PRIVATE_RESPONSE_HEADERS)


def _error(error: Exception) -> Response:
    if isinstance(error, OperatorAuthorizationError):
        status, code = 403, "artifact_access_denied"
    elif isinstance(error, ResourceNotFound):
        status, code = 404, "artifact_not_found"
    elif isinstance(error, ResourceConflict):
        status, code = 409, "artifact_conflict"
    elif isinstance(error, ResourceOutcomeUnknown):
        status, code = 503, "artifact_outcome_unknown"
    elif isinstance(error, ArtifactPreviewUnavailable):
        status, code = 503, "artifact_preview_unavailable"
    elif isinstance(error, ArtifactPreviewRefused):
        status, code = 422, "artifact_preview_refused"
    elif isinstance(error, ArtifactQuotaExceeded):
        status, code = 413, "artifact_quota_exceeded"
    elif isinstance(error, ArtifactIntegrityError):
        status, code = 503, "artifact_integrity_failure"
    elif isinstance(error, (ValueError, ValidationError)):
        status, code = 400, "invalid_artifact_request"
    else:
        status, code = 500, "artifact_service_error"
    return _json({"error": {"code": code}}, status)


def _delivery_error(error: Exception, prepared: bool, request: Request) -> Response:
    if prepared and request.method in {"POST", "DELETE"}:
        error = ResourceOutcomeUnknown("artifact result requires reconciliation")
    if isinstance(error, TimeoutError):
        code = ("artifact_outcome_unknown" if request.method in {"POST", "DELETE"}
                else "artifact_timeout")
        return _json({"error": {"code": code}}, 503)
    return _error(error)


def create_artifact_router(service: AuthorizedArtifactService, *, resolve_context: ContextResolver,
                           timeout_seconds: float = 30) -> APIRouter:
    """Build routes for a lifecycle-owned host service; no anonymous fallback."""
    if not callable(resolve_context):
        raise TypeError("host artifact context resolver required")
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("artifact HTTP timeout must be positive and finite")
    router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])

    def respond(request, action, build, prepare=None, preview_ref=None):
        def operation(context, value):
            if preview_ref is not None:
                return service.preview_delivery(context, preview_ref())
            return service.authority.operation(
                context, action, service.repository.storage_id,
                link=value if isinstance(value, ArtifactLink) else None,
            )
        return AuthorizedCapabilityResponse(
            request, authority=service.authority, resolve_context=resolve_context,
            operation=operation, build=build, timeout_seconds=timeout_seconds, prepare=prepare,
            error_response=lambda error, prepared: _delivery_error(error, prepared, request),
            access_denied=ArtifactAccessDenied,
        )

    def page_query(request):
        try:
            limit = int(request.query_params.get("limit", "50"))
        except ValueError:
            raise _InvalidRequest("invalid page limit") from None
        return {"cursor": request.query_params.get("cursor"), "limit": limit}

    @router.get("")
    async def list_artifacts(request: Request) -> Response:
        async def build(context):
            return _json(await service.list(context, query=request.query_params.get("q", ""), **page_query(request)))
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
                **PRIVATE_RESPONSE_HEADERS, "Content-Disposition": 'attachment; filename="artifact.bin"',
            })
        return respond(request, ArtifactAction.READ, build)

    @router.get("/{artifact_id}/revisions/{revision_id}/preview")
    async def preview(artifact_id: str, revision_id: str, request: Request) -> Response:
        def reference():
            return ArtifactRef(artifact_id=artifact_id, revision_id=revision_id)
        async def build(context):
            ref = reference()
            result = await service.preview(context, ref)
            return _json(result.payload(ref))
        return respond(request, ArtifactAction.READ, build, preview_ref=reference)

    @router.post("/links")
    async def link(request: Request) -> Response:
        async def build(context, value):
            await service.link(context, value)
            return Response(status_code=204, headers=PRIVATE_RESPONSE_HEADERS)
        return respond(request, ArtifactAction.LINK, build, lambda: _body(request, 16384, ArtifactLink))

    @router.delete("/links")
    async def unlink(request: Request) -> Response:
        async def build(context, value):
            await service.unlink(context, value)
            return Response(status_code=204, headers=PRIVATE_RESPONSE_HEADERS)
        return respond(request, ArtifactAction.LINK, build, lambda: _body(request, 16384, ArtifactLink))

    @router.delete("/{artifact_id}")
    async def delete(artifact_id: str, request: Request) -> Response:
        async def build(context):
            ref = ArtifactRef(artifact_id=artifact_id,
                              revision_id=request.query_params.get("expected_revision_id"))
            await service.delete(context, ref)
            return Response(status_code=204, headers=PRIVATE_RESPONSE_HEADERS)
        return respond(request, ArtifactAction.DELETE, build)

    return router
