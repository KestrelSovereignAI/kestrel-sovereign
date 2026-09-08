"""Canonical HTTP API error envelopes and FastAPI exception handlers."""

from __future__ import annotations

import logging
from datetime import datetime
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.utils import is_body_allowed_for_status_code
from starlette.exceptions import HTTPException as StarletteHTTPException

if TYPE_CHECKING:
    from kestrel_sovereign.llm.retry import AdvisedWaitExceedsRetryBudget

from kestrel_sovereign.logging_config import (
    correlation_id_var,
    resolve_correlation_id,
)

logger = logging.getLogger(__name__)

_LEGACY_DETAIL_UNSET = object()


class ApiHTTPException(HTTPException):
    """HTTP exception carrying a stable public error code and safe details."""

    def __init__(
        self,
        status_code: int,
        *,
        code: str,
        message: str,
        details: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code
        self.message = message
        self.details = details


def rate_limited_until(
    declined: AdvisedWaitExceedsRetryBudget, *, now: datetime | None = None
) -> ApiHTTPException:
    """The response an invocation returns when its model route declined to wait.

    The retry loop stopped because the provider's advised cool-down exceeds
    what the call could wait (#3127). A throttle answers ``429 rate_limited``;
    an overload that advised a cool-down answers ``503 route_unavailable``,
    so an outage is never reported as a quota problem. The reset time is
    derived from the provider's number, clamped to the advice horizon so a
    wrong header cannot name a year decades out; ``Retry-After`` is measured
    when this response is built so it agrees with the time in the message.
    Provider prose and the route's name stay behind the boundary.
    """
    if declined.throttled:
        status, code, condition = 429, "rate_limited", "rate limited"
    else:
        status, code, condition = 503, "route_unavailable", "unavailable"
    return ApiHTTPException(
        status_code=status,
        code=code,
        message=(
            f"The model route is {condition} {declined.reset_phrase()}; "
            "retry after that time."
        ),
        headers={"Retry-After": str(declined.retry_after_header_seconds(now))},
    )


def _default_message(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return f"HTTP {status_code}"


def _message_from_detail(detail: Any, status_code: int) -> str:
    if isinstance(detail, str) and detail.strip():
        return detail
    if isinstance(detail, dict):
        for key in ("message", "msg", "reason", "error"):
            value = detail.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return _default_message(status_code)


def api_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
    legacy_detail: Any = _LEGACY_DETAIL_UNSET,
    headers: dict[str, str] | None = None,
    correlation_id: str | None = None,
) -> Response:
    """Build the canonical envelope while retaining ``detail`` compatibility."""
    correlation_id = resolve_correlation_id(correlation_id)
    response_headers = {
        key: value
        for key, value in (headers or {}).items()
        if key.lower() != "x-correlation-id"
    }
    response_headers["X-Correlation-ID"] = correlation_id

    # Match FastAPI's built-in HTTPException behavior for statuses whose wire
    # contract forbids a body (for example 204 and 304). The correlation header
    # still gives operators a support handle for the response.
    if not is_body_allowed_for_status_code(status_code):
        return Response(status_code=status_code, headers=response_headers)

    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "correlation_id": correlation_id,
    }
    if details:
        error["details"] = details

    content = {
        "error": error,
        # Existing clients and endpoint tests still consume FastAPI's
        # historical top-level detail field during the migration window.
        "detail": (
            message
            if legacy_detail is _LEGACY_DETAIL_UNSET
            else legacy_detail
        ),
    }
    return JSONResponse(
        status_code=status_code,
        # Match FastAPI's built-in HTTPException handler for JSON-compatible
        # values such as datetimes, enums, dataclasses, and Pydantic models.
        # Passing those values straight to JSONResponse would turn a formerly
        # valid HTTPException into a secondary serialization failure.
        content=jsonable_encoder(content),
        headers=response_headers,
    )


async def api_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> Response:
    """Translate framework and typed HTTP exceptions into one envelope."""
    message = getattr(exc, "message", None) or _message_from_detail(
        exc.detail, exc.status_code
    )
    return api_error_response(
        status_code=exc.status_code,
        code=getattr(exc, "code", f"http_{exc.status_code}"),
        message=message,
        details=getattr(exc, "details", None),
        legacy_detail=exc.detail,
        headers=exc.headers,
        correlation_id=getattr(request.state, "correlation_id", None),
    )


async def api_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    """Return validation locations/messages without echoing rejected inputs."""
    details = [
        {
            "location": list(error.get("loc", ())),
            "message": str(error.get("msg", "Invalid value")),
            "code": str(error.get("type", "validation_error")),
        }
        for error in exc.errors()
    ]
    return api_error_response(
        status_code=422,
        code="validation_error",
        message="Request validation failed.",
        details=details,
        # Preserve the useful FastAPI shape, but deliberately omit ``input``
        # and ``ctx`` because either may contain submitted credentials/secrets.
        legacy_detail=details,
        correlation_id=getattr(request.state, "correlation_id", None),
    )


async def api_unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> Response:
    """Log unexpected failures and return a non-sensitive canonical 500."""
    correlation_id = resolve_correlation_id(
        getattr(request.state, "correlation_id", None)
    )
    token = correlation_id_var.set(correlation_id)
    try:
        logger.error(
            "Unhandled API error for %s %s",
            request.method,
            request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return api_error_response(
            status_code=500,
            code="internal_error",
            message="An internal error occurred.",
            correlation_id=correlation_id,
        )
    finally:
        correlation_id_var.reset(token)


def register_api_error_handlers(app: FastAPI) -> None:
    """Install the canonical global exception-handler path on ``app``."""
    app.add_exception_handler(StarletteHTTPException, api_http_exception_handler)
    app.add_exception_handler(RequestValidationError, api_validation_exception_handler)
    app.add_exception_handler(Exception, api_unhandled_exception_handler)
