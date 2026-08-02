"""Host-only, fail-closed routing tests for private inference leases."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from kestrel_sdk.llm import (
    InferenceLease,
    InferenceLeaseState,
    InferencePrivacy,
    InferenceRoute,
)
from pydantic import SecretStr

from kestrel_sovereign.llm.remote_backend import BackendType, RemoteBackendMixin
from kestrel_sovereign.llm.service import LLMServiceError


class _Client:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    async def close(self):
        self.closed = True


class _Host(RemoteBackendMixin):
    def __init__(self):
        self._backend = BackendType.CLOUD
        self._default_backend = BackendType.CLOUD
        self._remote_lease = None
        self._remote_client = None
        self._remote_adapter = object()
        self._remote_route_condition = asyncio.Condition()
        self._remote_inflight = 0
        self._remote_accepting = False
        self._remote_capabilities = frozenset()
        self._last_remote_error = None
        self._mandate_preference = {"model": None, "vendor": None, "route": None}

    def _remote_first_allowed(self, model_override):
        return model_override is None or "/" not in model_override


def _lease(
    *,
    lease_id: str = "lease-1",
    expires_at: datetime | None = None,
    headers: dict[str, SecretStr] | None = None,
    api_key: str | None = "route-secret",
) -> InferenceLease:
    now = datetime.now(UTC)
    created_at = (
        min(now, expires_at - timedelta(minutes=10)) if expires_at is not None else now
    )
    return InferenceLease(
        lease_id=lease_id,
        quote_id="quote-1",
        request_id="request-1",
        owner_id="agent-1",
        provider_name="runpod",
        state=InferenceLeaseState.READY,
        model="qwen3:8b",
        runtime="ollama",
        privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
        created_at=created_at,
        updated_at=created_at,
        expires_at=expires_at or now + timedelta(minutes=10),
        region="us-tx-1",
        hourly_cost_usd=Decimal("0.50"),
        estimated_total_cost_usd=Decimal("0.20"),
        route=InferenceRoute(
            endpoint=SecretStr("https://private.example.test/v1"),
            model="qwen3:8b",
            api_key=SecretStr(api_key) if api_key is not None else None,
            secret_headers=headers or {},
        ),
    )


@pytest.mark.asyncio
async def test_activation_keeps_route_host_only_and_disables_sdk_retries(monkeypatch):
    clients = []

    def create_client(**kwargs):
        client = _Client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        create_client,
    )
    host = _Host()
    lease = _lease()

    await host.activate_inference_lease(lease, capabilities=("chat", "streaming"))

    assert clients[0].kwargs["max_retries"] == 0
    assert clients[0].kwargs["api_key"] == "route-secret"
    status = host.get_backend_status()
    assert status["remote_active"] is True
    assert "private.example" not in repr(status)
    assert "route-secret" not in repr(status)


@pytest.mark.asyncio
async def test_custom_authorization_header_overrides_client_sentinel(monkeypatch):
    captured = {}

    def create_client(**kwargs):
        captured.update(kwargs)
        return _Client(**kwargs)

    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        create_client,
    )
    host = _Host()
    lease = _lease(
        api_key=None,
        headers={"Authorization": SecretStr("Bearer provider-token")},
    )

    await host.activate_inference_lease(lease, capabilities=("chat",))

    assert captured["default_headers"] == {"Authorization": "Bearer provider-token"}
    assert captured["api_key"] == "kestrel-private-route"


@pytest.mark.asyncio
async def test_ready_reconciliation_rotates_the_host_only_client(monkeypatch):
    clients = []

    def create_client(**kwargs):
        client = _Client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        create_client,
    )
    host = _Host()
    await host.activate_inference_lease(_lease(), capabilities=("chat",))

    await host.activate_inference_lease(
        _lease(api_key="rotated-route-secret"),
        capabilities=("chat",),
    )

    assert len(clients) == 2
    assert clients[0].closed is True
    assert clients[1].closed is False
    assert clients[1].kwargs["api_key"] == "rotated-route-secret"
    assert host._remote_client is clients[1]


@pytest.mark.asyncio
async def test_unchanged_ready_poll_does_not_rebuild_or_drain_client(monkeypatch):
    clients = []

    def create_client(**kwargs):
        client = _Client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        create_client,
    )
    host = _Host()
    original = _lease()
    await host.activate_inference_lease(original, capabilities=("chat",))

    async with host._remote_route_attempt(
        force_local_only=False,
        model_override=None,
        required_capabilities=("chat",),
    ) as snapshot:
        assert snapshot is not None
        refreshed = replace(
            original,
            updated_at=datetime.now(UTC),
            expires_at=original.expires_at + timedelta(minutes=5),
        )
        await host.activate_inference_lease(refreshed, capabilities=("chat",))

        assert len(clients) == 1
        assert clients[0].closed is False
        assert host._remote_accepting is True
        assert host._remote_inflight == 1
        assert host._remote_lease is refreshed


@pytest.mark.asyncio
async def test_route_attempt_requires_capabilities_and_never_falls_back(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **kwargs: _Client(**kwargs),
    )
    host = _Host()
    await host.activate_inference_lease(_lease(), capabilities=("chat",))

    with pytest.raises(LLMServiceError, match="lacks required capabilities"):
        async with host._remote_route_attempt(
            force_local_only=False,
            model_override=None,
            required_capabilities=("chat", "vision"),
        ):
            pytest.fail("unsupported route must not be yielded")


@pytest.mark.asyncio
async def test_release_drains_an_inflight_route_before_forgetting_secrets(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **kwargs: _Client(**kwargs),
    )
    host = _Host()
    await host.activate_inference_lease(_lease(), capabilities=("chat",))

    async with host._remote_route_attempt(
        force_local_only=False,
        model_override=None,
        required_capabilities=("chat",),
    ) as snapshot:
        assert snapshot is not None
        release_task = asyncio.create_task(host.deactivate_inference_lease("lease-1"))
        await asyncio.sleep(0)
        assert release_task.done() is False
        assert host._remote_accepting is False

    await release_task
    assert host._remote_lease is None
    assert host._remote_client is None


@pytest.mark.asyncio
async def test_expired_managed_route_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **kwargs: _Client(**kwargs),
    )
    host = _Host()
    await host.activate_inference_lease(
        _lease(expires_at=datetime.now(UTC) - timedelta(seconds=1)),
        capabilities=("chat",),
    )

    with pytest.raises(LLMServiceError, match="expired"):
        async with host._remote_route_attempt(
            force_local_only=False,
            model_override=None,
            required_capabilities=("chat",),
        ):
            pytest.fail("expired route must not be yielded")

    assert host._remote_accepting is False


def test_direct_remote_configuration_is_retired():
    host = _Host()

    with pytest.raises(LLMServiceError, match="Direct remote GPU configuration"):
        host.switch_backend(
            BackendType.REMOTE_GPU,
            config=SimpleNamespace(base_url="https://unsafe.example"),
        )


def test_managed_failure_exposes_only_a_safe_category():
    host = _Host()

    with pytest.raises(LLMServiceError, match="no cloud fallback") as caught:
        host._raise_managed_remote_failure(
            RuntimeError("secret endpoint https://private.example/token")
        )

    assert "private.example" not in str(caught.value)
    assert host._last_remote_error == "RuntimeError"
