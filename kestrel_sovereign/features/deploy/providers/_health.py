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
    # Class name of the transport exception; a stable identifier that is
    # safe to surface where the exception message is not.
    error_type: NotRequired[str]
    # Present only when the body is a JSON object carrying the Kestrel
    # ``/health`` ``agent_initialized`` flag.
    agent_initialized: NotRequired[bool]


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

    ``2xx`` and ``3xx`` responses mean the service is reachable, unless the
    body reports ``agent_initialized: false`` — a host with no agent is not
    ready whatever its status code. Cancellation is intentionally not
    caught: callers own the lifecycle of the probe.
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
            "error_type": type(exc).__name__,
        }

    result: HealthCheckResult = {
        "healthy": 200 <= response.status_code < 400,
        "status_code": response.status_code,
        "response_time": max(0.0, monotonic() - started_at),
    }
    agent_initialized = _reported_agent_initialized(response)
    if agent_initialized is not None:
        result["agent_initialized"] = agent_initialized
        if not agent_initialized:
            result["healthy"] = False
    return result


def _reported_agent_initialized(response) -> bool | None:
    """Read ``agent_initialized`` from a JSON health body, if it has one."""
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict) and isinstance(body.get("agent_initialized"), bool):
        return body["agent_initialized"]
    return None


def classify_health_failure(result: HealthCheckResult) -> tuple[str, str]:
    """Map an unhealthy probe to a fixed category and a sanitized detail.

    The exception message is never echoed: transport errors can carry
    URLs, headers, or proxy detail. Only the status code, the exception
    class name, and the body's ``agent_initialized`` flag are surfaced.
    """
    status_code = result.get("status_code")
    if result.get("agent_initialized") is False:
        return (
            "agent_not_initialized",
            f"HTTP {status_code}: the health endpoint reports no initialized agent",
        )
    if status_code in (401, 403):
        return (
            "auth_rejected",
            f"HTTP {status_code}: the service rejected the unauthenticated "
            "readiness probe (is the invoker IAM binding in place?)",
        )
    if status_code is not None:
        return "http_status", f"HTTP {status_code} from the health endpoint"
    error_type = result.get("error_type") or "unknown error"
    if "timeout" in error_type.lower():
        return "probe_timeout", f"no response before the probe timeout ({error_type})"
    return "unreachable", f"the health endpoint could not be reached ({error_type})"
