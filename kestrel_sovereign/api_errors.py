"""Canonical HTTP API error envelopes and FastAPI exception handlers."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from kestrel_sovereign.logging_config import correlation_id_var, get_correlation_id

logger = logging.getLogger(__name__)


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
    legacy_detail: Any | None = None,
    headers: dict[str, str] | None = None,
    correlation_id: str | None = None,
) -> JSONResponse:
    """Build the canonical envelope while retaining ``detail`` compatibility."""
    correlation_id = correlation_id or get_correlation_id()
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "correlation_id": correlation_id,
    }
    if details:
        error["details"] = details

    response_headers = dict(headers or {})
    response_headers["X-Correlation-ID"] = correlation_id
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            # Existing clients and endpoint tests still consume FastAPI's
            # historical top-level detail field during the migration window.
            "detail": message if legacy_detail is None else legacy_detail,
        },
        headers=response_headers,
    )


async def api_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
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
) -> JSONResponse:
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
) -> JSONResponse:
    """Log unexpected failures and return a non-sensitive canonical 500."""
    correlation_id = (
        getattr(request.state, "correlation_id", None) or get_correlation_id()
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
