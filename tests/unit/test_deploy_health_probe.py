"""Contract tests for the shared deployment health probe."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.features.deploy.core import DeployManagerCore
from kestrel_sovereign.features.deploy.providers._health import (
    build_health_url,
    probe_http_health,
)
from kestrel_sovereign.features.deploy.providers.azure_container import (
    AzureContainerProvider,
)
from kestrel_sovereign.features.deploy.providers.base import DeployProvider
from kestrel_sovereign.features.deploy.providers.cloudrun import CloudRunProvider


@pytest.mark.parametrize(
    ("service_url", "path", "expected"),
    [
        ("https://agent.example", "/health", "https://agent.example/health"),
        ("https://agent.example/", "health", "https://agent.example/health"),
        ("https://agent.example///", "//ready", "https://agent.example/ready"),
    ],
)
def test_build_health_url_normalizes_boundary_slashes(
    service_url: str, path: str, expected: str
) -> None:
    assert build_health_url(service_url, path) == expected


@pytest.mark.parametrize(
    ("status_code", "healthy"),
    [(200, True), (301, True), (399, True), (400, False), (503, False)],
)
async def test_probe_classifies_status_and_uses_monotonic_elapsed_time(
    status_code: int, healthy: bool
) -> None:
    response = MagicMock(status_code=status_code)

    with (
        patch("httpx.AsyncClient") as client_class,
        patch(
            "kestrel_sovereign.features.deploy.providers._health.monotonic",
            side_effect=[50.0, 50.25],
        ),
    ):
        client_class.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=response
        )

        result = await probe_http_health(
            "https://agent.example/", path="/ready", timeout=3.0
        )

    assert result == {
        "healthy": healthy,
        "status_code": status_code,
        "response_time": 0.25,
    }
    client_class.assert_called_once_with(timeout=3.0)
    client_class.return_value.__aenter__.return_value.get.assert_awaited_once_with(
        "https://agent.example/ready"
    )


async def test_probe_translates_request_error_without_fabricating_timing() -> None:
    with patch("httpx.AsyncClient") as client_class:
        client_class.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=ValueError("Invalid URL")
        )

        result = await probe_http_health("not a URL")

    assert result == {
        "healthy": False,
        "status_code": None,
        "response_time": None,
        "error": "Invalid URL",
    }


async def test_probe_never_consumes_caller_cancellation() -> None:
    with patch("httpx.AsyncClient") as client_class:
        client_class.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=asyncio.CancelledError
        )

        with pytest.raises(asyncio.CancelledError):
            await probe_http_health("https://agent.example")


def test_cloud_providers_share_the_default_health_contract() -> None:
    assert CloudRunProvider.health_check is DeployProvider.health_check
    assert AzureContainerProvider.health_check is DeployProvider.health_check


async def test_readiness_polling_delegates_custom_path_to_shared_probe() -> None:
    manager = DeployManagerCore.__new__(DeployManagerCore)
    manager.health_check_timeout = 120
    manager.health_check_path = "readyz"
    probe = AsyncMock(
        return_value={
            "healthy": True,
            "status_code": 204,
            "response_time": 0.01,
        }
    )

    with patch("kestrel_sovereign.features.deploy.core.probe_http_health", probe):
        assert await manager._verify_health("https://agent.example/", timeout=5)

    probe.assert_awaited_once()
    assert probe.await_args.args == ("https://agent.example/",)
    assert probe.await_args.kwargs["path"] == "readyz"
    assert 0 < probe.await_args.kwargs["timeout"] <= 5


async def test_readiness_backoff_cannot_sleep_past_deadline() -> None:
    manager = DeployManagerCore.__new__(DeployManagerCore)
    manager.health_check_timeout = 120
    manager.health_check_path = "/health"
    now = 100.0
    sleeps: list[float] = []

    async def advance_clock(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    probe = AsyncMock(
        return_value={
            "healthy": False,
            "status_code": None,
            "response_time": None,
            "error": "not ready",
        }
    )

    with (
        patch(
            "kestrel_sovereign.features.deploy.core.monotonic",
            side_effect=lambda: now,
        ),
        patch("kestrel_sovereign.features.deploy.core.asyncio.sleep", advance_clock),
        patch("kestrel_sovereign.features.deploy.core.probe_http_health", probe),
    ):
        assert not await manager._verify_health(
            "https://agent.example", timeout=2, poll_interval=30
        )

    assert sleeps == [2.0]
    probe.assert_awaited_once_with("https://agent.example", path="/health", timeout=2.0)


async def test_readiness_polling_never_consumes_caller_cancellation() -> None:
    manager = DeployManagerCore.__new__(DeployManagerCore)
    manager.health_check_timeout = 120
    manager.health_check_path = "/health"

    with patch(
        "kestrel_sovereign.features.deploy.core.probe_http_health",
        AsyncMock(side_effect=asyncio.CancelledError),
    ):
        with pytest.raises(asyncio.CancelledError):
            await manager._verify_health("https://agent.example", timeout=5)
