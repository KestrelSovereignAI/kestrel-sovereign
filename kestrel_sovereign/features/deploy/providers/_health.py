"""Shared HTTP health probing for deployment providers."""

from __future__ import annotations

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
    return f"{service_url.rstrip('/')}/{path.lstrip('/')}"


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
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(health_url)
    except Exception as exc:
        return {
            "healthy": False,
            "status_code": None,
            "response_time": None,
            "error": str(exc),
        }

    return {
        "healthy": 200 <= response.status_code < 400,
        "status_code": response.status_code,
        "response_time": max(0.0, monotonic() - started_at),
    }
