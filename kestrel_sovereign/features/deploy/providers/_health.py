"""Shared HTTP health probing for deployment providers."""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import NotRequired, TypedDict


class HealthCheckResult(TypedDict):
    """Stable result returned by deployment health checks."""

    healthy: bool
    status_code: int | None
    response_time: float | None
    error: NotRequired[str]


def build_health_url(service_url: str, path: str = "/health") -> str:
    """Join a service URL and health path without duplicate separators."""
    base_url = service_url.rstrip("/")
    if path == "":
        return base_url
    return f"{base_url}/{path.lstrip('/')}"


async def probe_http_health(
    service_url: str,
    *,
    path: str = "/health",
    timeout: float = 10.0,
) -> HealthCheckResult:
    """Perform one HTTP health probe.

    ``2xx`` and ``3xx`` responses mean the service is reachable. Cancellation
    is intentionally not caught: callers own the lifecycle of the probe.
    """
    import httpx

    health_url = build_health_url(service_url, path)
    started_at = monotonic()

    try:
        # HTTPX's scalar timeout is per network operation/read-idle period,
        # not an end-to-end deadline.  Keep it for phase-specific errors, but
        # also bound the complete client lifecycle so a response that drips
        # bytes forever cannot outlive the caller's timeout.
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(health_url)
    except Exception as exc:
        error = str(exc)
        if not error.strip():
            error = type(exc).__name__
        return {
            "healthy": False,
            "status_code": None,
            "response_time": None,
            "error": error,
        }

    return {
        "healthy": 200 <= response.status_code < 400,
        "status_code": response.status_code,
        "response_time": max(0.0, monotonic() - started_at),
    }
