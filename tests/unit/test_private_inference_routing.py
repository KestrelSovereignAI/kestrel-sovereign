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
        self._remote_touch_lease = None
        self._last_remote_error = None
        self._remote_route_epoch = 0
        self._remote_route_closed = False
        self._mandate_preference = {"model": None, "vendor": None, "route": None}

    def _remote_first_allowed(self, model_override):
        return model_override is None or "/" not in model_override


def _touch_current(host: _Host, events: list[str] | None = None):
    async def touch(lease_id: str) -> InferenceLease:
        if events is not None:
            events.append("touch")
        lease = host._remote_lease
        assert lease is not None
        assert lease.lease_id == lease_id
        return lease

    return touch


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

    await host.activate_inference_lease(
        lease,
        capabilities=("chat", "streaming"),
        touch_lease=_touch_current(host),
    )

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

    await host.activate_inference_lease(
        lease,
        capabilities=("chat",),
        touch_lease=_touch_current(host),
    )

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
    await host.activate_inference_lease(
        _lease(),
        capabilities=("chat",),
        touch_lease=_touch_current(host),
    )

    await host.activate_inference_lease(
        _lease(api_key="rotated-route-secret"),
        capabilities=("chat",),
        touch_lease=_touch_current(host),
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
    await host.activate_inference_lease(
        original,
        capabilities=("chat",),
        touch_lease=_touch_current(host),
    )

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
        await host.activate_inference_lease(
            refreshed,
            capabilities=("chat",),
            touch_lease=_touch_current(host),
        )

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
    await host.activate_inference_lease(
        _lease(),
        capabilities=("chat",),
        touch_lease=_touch_current(host),
    )

    with pytest.raises(LLMServiceError, match="lacks required capabilities"):
        async with host._remote_route_attempt(
            force_local_only=False,
            model_override=None,
            required_capabilities=("chat", "vision"),
        ):
            pytest.fail("unsupported route must not be yielded")


@pytest.mark.asyncio
async def test_route_attempt_touches_idle_deadline_before_pinning_traffic(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **kwargs: _Client(**kwargs),
    )
    events: list[str] = []
    host = _Host()
    await host.activate_inference_lease(
        _lease(),
        capabilities=("chat",),
        touch_lease=_touch_current(host, events),
    )

    async with host._remote_route_attempt(
        force_local_only=False,
        model_override=None,
        required_capabilities=("chat",),
    ) as snapshot:
        events.append("inference")
        assert snapshot is not None
        assert host._remote_inflight == 1

    assert events == ["touch", "inference"]
    assert host._remote_inflight == 0


@pytest.mark.asyncio
async def test_touch_failure_blocks_inference_without_cloud_fallback(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **kwargs: _Client(**kwargs),
    )
    host = _Host()

    async def fail_touch(_lease_id: str) -> InferenceLease:
        raise ConnectionError("private control-plane details")

    await host.activate_inference_lease(
        _lease(),
        capabilities=("chat",),
        touch_lease=fail_touch,
    )

    with pytest.raises(LLMServiceError, match="no cloud fallback") as caught:
        async with host._remote_route_attempt(
            force_local_only=False,
            model_override=None,
            required_capabilities=("chat",),
        ):
            pytest.fail("traffic must not run when renewal fails")

    assert "control-plane" not in str(caught.value)
    assert host._last_remote_error == "ConnectionError"
    assert host._remote_inflight == 0


@pytest.mark.asyncio
async def test_release_winning_touch_to_pin_race_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **kwargs: _Client(**kwargs),
    )
    host = _Host()
    touch_started = asyncio.Event()
    finish_touch = asyncio.Event()

    async def blocked_touch(lease_id: str) -> InferenceLease:
        lease = host._remote_lease
        assert lease is not None and lease.lease_id == lease_id
        touch_started.set()
        await finish_touch.wait()
        return lease

    await host.activate_inference_lease(
        _lease(),
        capabilities=("chat",),
        touch_lease=blocked_touch,
    )

    async def attempt() -> None:
        async with host._remote_route_attempt(
            force_local_only=False,
            model_override=None,
            required_capabilities=("chat",),
        ):
            pytest.fail("a released route must not be pinned")

    attempt_task = asyncio.create_task(attempt())
    await touch_started.wait()
    await host.deactivate_inference_lease("lease-1")
    finish_touch.set()

    with pytest.raises(LLMServiceError, match="no cloud fallback"):
        await attempt_task
    assert host._remote_lease is None
    assert host._remote_inflight == 0


@pytest.mark.asyncio
async def test_release_then_faithful_touch_reactivation_still_fails_closed(monkeypatch):
    """The release/touch race, with a touch double that behaves like the real one.

    ``coordinator.touch`` ends in ``_apply_provider_lease``, which calls
    ``activate_inference_lease`` unconditionally for a READY lease. A double
    that merely returns the lease cannot exercise the race at all: it leaves
    the deactivated state in place, so the re-check trivially sees a missing
    route. The dangerous case is the opposite one - the renewal REBUILDS a
    route that compares equal to the one just torn down, and comparing
    observable state cannot tell the difference. Only the route epoch can.
    """
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **kwargs: _Client(**kwargs),
    )
    host = _Host()
    touch_started = asyncio.Event()
    finish_touch = asyncio.Event()

    async def reactivating_touch(lease_id: str) -> InferenceLease:
        touch_started.set()
        await finish_touch.wait()
        # This is what the real coordinator does on the way back.
        lease = _lease(lease_id=lease_id)
        await host.activate_inference_lease(
            lease,
            capabilities=("chat",),
            touch_lease=reactivating_touch,
        )
        return lease

    await host.activate_inference_lease(
        _lease(),
        capabilities=("chat",),
        touch_lease=reactivating_touch,
    )

    async def attempt() -> None:
        async with host._remote_route_attempt(
            force_local_only=False,
            model_override=None,
            required_capabilities=("chat",),
        ):
            pytest.fail("a released route must not be pinned, even if rebuilt")

    attempt_task = asyncio.create_task(attempt())
    await touch_started.wait()
    await host.deactivate_inference_lease("lease-1")
    finish_touch.set()

    with pytest.raises(LLMServiceError, match="released during renewal"):
        await attempt_task
    # The renewal did rebuild the route - which is exactly why comparing
    # observable state would have passed the call through.
    assert host._remote_lease is not None
    assert host._remote_inflight == 0


@pytest.mark.asyncio
async def test_touch_cannot_reactivate_a_route_after_shutdown(monkeypatch):
    """Renewal must not resurrect a route after ``close()`` has drained it.

    ``LLMService.close()`` calls ``deactivate_inference_lease`` without the
    coordinator lock and then returns. A renewal still in its provider
    round-trip would otherwise call ``activate_inference_lease`` afterwards,
    building an ``AsyncOpenAI`` client that nothing will ever close and
    restoring ``_backend = REMOTE_GPU`` on a service that has finished shutting
    down.
    """
    clients: list[_Client] = []

    def _make_client(**kwargs):
        client = _Client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI", _make_client
    )
    host = _Host()
    await host.activate_inference_lease(
        _lease(),
        capabilities=("chat",),
        touch_lease=_touch_current(host),
    )
    assert len(clients) == 1

    # Shutdown latches the flag, then drains and forgets the route.
    async with host._remote_route_condition:
        host._remote_route_closed = True
    await host.deactivate_inference_lease("lease-1", require_active=False)
    assert clients[0].closed is True

    with pytest.raises(LLMServiceError, match="shutting down"):
        await host.activate_inference_lease(
            _lease(),
            capabilities=("chat",),
            touch_lease=_touch_current(host),
        )

    # No second client was built, so none was leaked.
    assert len(clients) == 1
    assert host._remote_lease is None
    assert host._backend is BackendType.CLOUD


@pytest.mark.asyncio
async def test_release_drains_an_inflight_route_before_forgetting_secrets(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **kwargs: _Client(**kwargs),
    )
    host = _Host()
    await host.activate_inference_lease(
        _lease(),
        capabilities=("chat",),
        touch_lease=_touch_current(host),
    )

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
        touch_lease=_touch_current(host),
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


@pytest.mark.asyncio
async def test_shutdown_during_client_construction_closes_the_built_client(
    monkeypatch,
):
    """The second latch guard exists to close a client built during the race.

    ``activate_inference_lease`` checks the latch, releases the route
    condition, builds the ``AsyncOpenAI`` client, then re-acquires. ``close()``
    can land in exactly that window. The first guard cannot cover it - that one
    runs before the client exists - so without the second guard the built
    client is assigned to a service that has finished shutting down, and
    nothing ever closes it.

    The client factory flips the latch to model ``close()`` completing while
    construction is in flight; that is the only way to land inside a window
    that contains no await point.
    """
    host = _Host()
    clients: list[_Client] = []

    def _make_client(**kwargs):
        client = _Client(**kwargs)
        clients.append(client)
        # close() wins the race, between guard one and guard two.
        host._remote_route_closed = True
        return client

    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI", _make_client
    )

    with pytest.raises(LLMServiceError, match="shutting down"):
        await host.activate_inference_lease(
            _lease(),
            capabilities=("chat",),
            touch_lease=_touch_current(host),
        )

    # The client WAS built, and it was closed rather than leaked.
    assert len(clients) == 1
    assert clients[0].closed is True
    assert host._remote_lease is None
    assert host._remote_client is None
    assert host._backend is BackendType.CLOUD


@pytest.mark.asyncio
async def test_release_draining_behind_a_pinned_call_still_fails_closed(monkeypatch):
    """The post-touch route re-check, isolated from the epoch guard.

    ``deactivate_inference_lease`` bumps the epoch AFTER its drain loop, so
    while it is blocked draining an already-pinned call the epoch is unchanged
    and the epoch guard passes. This re-check is then the only thing standing
    between a renewing call and a route the owner has already released.

    Without it the call pins behind the drain, sends agent data over a released
    route, and ``deactivate`` blocks a further HTTP_TIMEOUT_MEDIUM or raises
    "timed out draining".
    """
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **kwargs: _Client(**kwargs),
    )
    host = _Host()
    touch_started = asyncio.Event()
    finish_touch = asyncio.Event()
    touches = {"count": 0}

    async def touch(lease_id: str) -> InferenceLease:
        touches["count"] += 1
        if touches["count"] == 1:
            return host._remote_lease  # the first call proceeds normally
        touch_started.set()
        await finish_touch.wait()
        return host._remote_lease

    await host.activate_inference_lease(
        _lease(), capabilities=("chat",), touch_lease=touch
    )

    async with host._remote_route_attempt(
        force_local_only=False,
        model_override=None,
        required_capabilities=("chat",),
    ) as pinned:
        assert pinned is not None
        assert host._remote_inflight == 1

        async def renewing_call() -> None:
            async with host._remote_route_attempt(
                force_local_only=False,
                model_override=None,
                required_capabilities=("chat",),
            ):
                pytest.fail("must not pin behind a draining release")

        renewal = asyncio.create_task(renewing_call())
        await touch_started.wait()

        epoch_before = host._remote_route_epoch
        release = asyncio.create_task(host.deactivate_inference_lease("lease-1"))
        while host._remote_accepting:
            await asyncio.sleep(0)
        # The release is blocked in its drain loop, so the epoch has NOT moved.
        assert host._remote_route_epoch == epoch_before

        finish_touch.set()
        with pytest.raises(LLMServiceError, match="draining or unavailable"):
            await renewal
        assert host._remote_inflight == 1

    await release
    assert host._remote_lease is None


@pytest.mark.asyncio
async def test_lease_expiring_during_renewal_fails_closed(monkeypatch):
    """The post-touch expiry re-check — the consequence of adopting a shortening.

    ``touch`` deliberately ADOPTS a nearer expiry, because the provider no
    longer honours the longer one. That makes this re-check load-bearing rather
    than incidental: the pre-touch check passed against the OLD deadline, and
    only this one sees the renewed lease. Without it, traffic goes to leased
    capacity past its expiry, which the provider is entitled to have reclaimed.
    """
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **kwargs: _Client(**kwargs),
    )
    host = _Host()

    async def touch(lease_id: str) -> InferenceLease:
        current = host._remote_lease
        assert current is not None
        # A provider that models its idle deadline as min(now + ttl, cap) can
        # return an expiry that has already passed by the time we see it.
        shortened = replace(
            current, expires_at=datetime.now(UTC) - timedelta(seconds=1)
        )
        await host.activate_inference_lease(
            shortened, capabilities=("chat",), touch_lease=touch
        )
        return shortened

    # created_at is min(now, expires_at - 10min), so a 5-minute expiry puts it
    # 5 minutes in the past and leaves room for a shortened-but-valid deadline.
    await host.activate_inference_lease(
        _lease(expires_at=datetime.now(UTC) + timedelta(minutes=5)),
        capabilities=("chat",),
        touch_lease=touch,
    )

    with pytest.raises(LLMServiceError, match="expired during renewal"):
        async with host._remote_route_attempt(
            force_local_only=False,
            model_override=None,
            required_capabilities=("chat",),
        ):
            pytest.fail("expired capacity must not receive traffic")

    assert host._remote_inflight == 0
    assert host._remote_accepting is False
