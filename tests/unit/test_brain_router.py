"""Integration coverage for the managed private route in ``LLMService``."""

from __future__ import annotations

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

from kestrel_sovereign.llm.remote_backend import BackendType
from kestrel_sovereign.llm.service import LLMService, LLMServiceError


class FakeRemoteClient:
    def __init__(
        self,
        *,
        should_fail: bool = False,
        response_text: str = "remote-response",
        **_kwargs,
    ):
        self.should_fail = should_fail
        self.response_text = response_text
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **_kwargs):
        if self.should_fail:
            raise RuntimeError("private transport details")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.response_text))
            ]
        )

    async def close(self):
        self.closed = True


def _lease() -> InferenceLease:
    now = datetime.now(UTC)
    return InferenceLease(
        lease_id="lease-1",
        quote_id="quote-1",
        request_id="request-1",
        owner_id="agent-1",
        provider_name="runpod",
        state=InferenceLeaseState.READY,
        model="llama-3",
        runtime="ollama",
        privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(minutes=10),
        hourly_cost_usd=Decimal("0.50"),
        estimated_total_cost_usd=Decimal("0.20"),
        route=InferenceRoute(
            endpoint=SecretStr("https://private.example.test/v1"),
            model="llama-3",
            api_key=SecretStr("route-secret"),
        ),
    )


def _touch_current(service: LLMService, calls: list[str] | None = None):
    async def touch(lease_id: str) -> InferenceLease:
        if calls is not None:
            calls.append("touch")
        lease = service._remote_lease
        assert lease is not None and lease.lease_id == lease_id
        return lease

    return touch


@pytest.mark.asyncio
async def test_validated_lease_activates_and_releases(monkeypatch):
    client = FakeRemoteClient(response_text="gpu-ok")
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **_kwargs: client,
    )
    service = LLMService()
    try:
        await service.activate_inference_lease(
            _lease(),
            capabilities=("chat", "streaming", "tools"),
            touch_lease=_touch_current(service),
        )

        status = service.get_backend_status()
        assert status["current_backend"] == BackendType.REMOTE_GPU.value
        assert status["remote_active"] is True

        await service.deactivate_inference_lease("lease-1")
        assert service.get_backend_status()["remote_active"] is False
        assert client.closed is True
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_generate_uses_validated_private_route(monkeypatch):
    client = FakeRemoteClient(response_text="gpu-ok")
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **_kwargs: client,
    )
    service = LLMService()
    calls: list[str] = []
    try:
        await service.activate_inference_lease(
            _lease(),
            capabilities=("chat",),
            touch_lease=_touch_current(service, calls),
        )

        result = await service.generate(system_prompt="sys", user_prompt="hi")

        assert result == "gpu-ok"
        assert calls == ["touch"]
        assert service.get_backend_status()["remote_active"] is True
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_private_route_failure_does_not_fall_back(monkeypatch):
    client = FakeRemoteClient(should_fail=True)
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **_kwargs: client,
    )
    service = LLMService()
    try:
        await service.activate_inference_lease(
            _lease(),
            capabilities=("chat",),
            touch_lease=_touch_current(service),
        )

        with pytest.raises(LLMServiceError, match="no cloud fallback") as caught:
            await service.generate(system_prompt="sys", user_prompt="hi")

        assert "private transport details" not in str(caught.value)
        assert service.get_backend_status()["remote_active"] is True
    finally:
        await service.close()
