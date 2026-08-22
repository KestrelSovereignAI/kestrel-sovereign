"""Provider-neutral orchestration and crash-recovery tests for #2844."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from kestrel_sdk.llm import (
    InferenceLease,
    InferenceLeaseConstraintError,
    InferenceLeaseFailure,
    InferenceLeaseNotFoundError,
    InferenceLeaseProviderUnavailableError,
    InferenceLeaseProvisioningError,
    InferenceLeaseQuote,
    InferenceLeaseRequest,
    InferenceLeaseState,
    InferencePrivacy,
    InferenceProviderCapability,
    InferenceRoute,
)
from pydantic import SecretStr

from kestrel_sovereign.llm.inference_leases import (
    InferenceLeaseCoordinator,
    InferenceLeaseProviderDiscoveryError,
    discover_inference_lease_providers,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _request(
    *,
    request_id: str = "request-1",
    owner_id: str = "agent-1",
    max_total: str = "1.00",
) -> InferenceLeaseRequest:
    return InferenceLeaseRequest(
        request_id=request_id,
        owner_id=owner_id,
        model="qwen3:8b",
        runtime="ollama",
        max_hourly_cost_usd=Decimal("1.00"),
        max_total_cost_usd=Decimal(max_total),
        privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
        capabilities=("chat", "streaming", "tools"),
        allowed_regions=("us-tx-1",),
        expected_session_seconds=900,
        idle_ttl_seconds=300,
        ready_deadline_seconds=600,
    )


def _quote(
    request: InferenceLeaseRequest,
    *,
    provider: str = "runpod",
    total: str = "0.20",
    hourly: str = "0.50",
    ready_seconds: int = 30,
) -> InferenceLeaseQuote:
    return InferenceLeaseQuote(
        quote_id=f"quote-{provider}-{request.request_id}",
        request_id=request.request_id,
        provider_name=provider,
        runtime=request.runtime,
        region="us-tx-1",
        privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
        hourly_cost_usd=Decimal(hourly),
        estimated_total_cost_usd=Decimal(total),
        estimated_ready_seconds=ready_seconds,
        expires_at=_now() + timedelta(minutes=5),
    )


def _lease(
    request: InferenceLeaseRequest,
    quote: InferenceLeaseQuote,
    *,
    state: InferenceLeaseState,
    updated_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> InferenceLease:
    created = request.requested_at
    updated = updated_at or created
    route = None
    if state is InferenceLeaseState.READY:
        route = InferenceRoute(
            endpoint=SecretStr("https://private.example.test/v1"),
            model=request.model,
            api_key=SecretStr("route-secret"),
        )
    failure = (
        InferenceLeaseFailure(code="provider_failed", message="capacity failed")
        if state is InferenceLeaseState.FAILED
        else None
    )
    if expires_at is None:
        expires_at = (
            created + timedelta(microseconds=1)
            if state is InferenceLeaseState.EXPIRED
            else updated + timedelta(seconds=request.idle_ttl_seconds)
        )
    return InferenceLease(
        lease_id=f"lease-{request.request_id}",
        quote_id=quote.quote_id,
        request_id=request.request_id,
        owner_id=request.owner_id,
        provider_name=quote.provider_name,
        state=state,
        model=request.model,
        runtime=request.runtime,
        privacy=quote.privacy,
        created_at=created,
        updated_at=updated,
        expires_at=expires_at,
        region=quote.region,
        hourly_cost_usd=quote.hourly_cost_usd,
        estimated_total_cost_usd=quote.estimated_total_cost_usd,
        route=route,
        failure=failure,
    )


class _Provider:
    def __init__(
        self,
        name: str,
        *,
        total: str,
        ready_seconds: int,
        events: list[str] | None = None,
    ) -> None:
        self.provider_name = name
        self.total = total
        self.ready_seconds = ready_seconds
        self.events = events if events is not None else []
        self.acquire_state = InferenceLeaseState.PENDING
        self.status_state = InferenceLeaseState.READY
        self.acquire_calls = 0
        self.status_calls = 0
        self.touch_calls = 0
        self.release_calls = 0
        self.last_request: InferenceLeaseRequest | None = None
        self.last_quote: InferenceLeaseQuote | None = None

    def capabilities(self):
        return (
            InferenceProviderCapability(
                runtime="ollama",
                privacy=(InferencePrivacy.AUTHENTICATED_ENDPOINT,),
                capabilities=("chat", "streaming", "tools"),
                regions=("us-tx-1",),
            ),
        )

    def is_available(self) -> bool:
        return True

    async def quote(self, request: InferenceLeaseRequest) -> InferenceLeaseQuote:
        return _quote(
            request,
            provider=self.provider_name,
            total=self.total,
            ready_seconds=self.ready_seconds,
        )

    async def acquire(
        self,
        request: InferenceLeaseRequest,
        quote: InferenceLeaseQuote,
    ) -> InferenceLease:
        self.events.append("provider.acquire")
        self.acquire_calls += 1
        self.last_request = request
        self.last_quote = quote
        return _lease(request, quote, state=self.acquire_state)

    async def status(self, owner_id: str, lease_id: str) -> InferenceLease:
        self.events.append("provider.status")
        self.status_calls += 1
        assert self.last_request is not None
        assert self.last_quote is not None
        lease = _lease(
            self.last_request,
            self.last_quote,
            state=self.status_state,
            updated_at=_now(),
        )
        lease.assert_owner(owner_id)
        assert lease.lease_id == lease_id
        return lease

    async def touch(self, owner_id: str, lease_id: str) -> InferenceLease:
        self.events.append("provider.touch")
        self.touch_calls += 1
        assert self.last_request is not None
        assert self.last_quote is not None
        lease = _lease(
            self.last_request,
            self.last_quote,
            state=InferenceLeaseState.READY,
            updated_at=_now(),
        )
        lease.assert_owner(owner_id)
        assert lease.lease_id == lease_id
        return lease

    async def release(self, owner_id: str, lease_id: str) -> InferenceLease:
        self.events.append("provider.release")
        self.release_calls += 1
        assert self.last_request is not None
        assert self.last_quote is not None
        lease = _lease(
            self.last_request,
            self.last_quote,
            state=InferenceLeaseState.RELEASED,
            updated_at=_now(),
        )
        lease.assert_owner(owner_id)
        assert lease.lease_id == lease_id
        return lease


class _LLMService:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.activated: InferenceLease | None = None
        self.capabilities: tuple[str, ...] = ()
        self.touch_lease = None

    async def activate_inference_lease(self, lease, *, capabilities=(), touch_lease):
        self.events.append("llm.activate")
        self.activated = lease
        self.capabilities = tuple(capabilities)
        self.touch_lease = touch_lease

    async def deactivate_inference_lease(self, lease_id, *, require_active=True):
        self.events.append("llm.deactivate")
        if self.activated is not None and self.activated.lease_id == lease_id:
            self.activated = None
            self.touch_lease = None


def _coordinator(
    providers,
    *,
    llm_service=None,
    events: list[str] | None = None,
):
    persisted: list[dict | None] = []

    async def persist(payload):
        if events is not None:
            events.append("persist")
        persisted.append(payload)

    return (
        InferenceLeaseCoordinator(
            llm_service=llm_service or _LLMService(events),
            owner_id="agent-1",
            providers=providers,
            persist_state=persist,
        ),
        persisted,
    )


@pytest.mark.asyncio
async def test_selects_cheapest_quote_then_fastest_without_mutating_others():
    cheap_slow = _Provider("cheap", total="0.10", ready_seconds=50)
    cheap_fast = _Provider("fast", total="0.10", ready_seconds=20)
    expensive = _Provider("expensive", total="0.20", ready_seconds=1)
    coordinator, persisted = _coordinator(
        {item.provider_name: item for item in (cheap_slow, cheap_fast, expensive)}
    )

    lease = await coordinator.acquire(_request())

    assert lease.provider_name == "fast"
    assert cheap_fast.acquire_calls == 1
    assert cheap_slow.acquire_calls == expensive.acquire_calls == 0
    assert persisted[0]["lease"] is None
    assert persisted[-1]["lease"]["state"] == "pending"


@pytest.mark.asyncio
async def test_persists_idempotency_plan_before_billable_acquire():
    events: list[str] = []
    provider = _Provider("runpod", total="0.20", ready_seconds=10, events=events)
    coordinator, _persisted = _coordinator(
        {"runpod": provider},
        events=events,
    )

    await coordinator.acquire(_request(request_id="durable-attempt"))

    assert events.index("persist") < events.index("provider.acquire")


@pytest.mark.asyncio
async def test_pending_status_activates_ready_route_without_public_secrets():
    provider = _Provider("runpod", total="0.20", ready_seconds=10)
    llm = _LLMService()
    coordinator, _persisted = _coordinator({"runpod": provider}, llm_service=llm)
    pending = await coordinator.acquire(_request())

    ready = await coordinator.status(pending.lease_id)

    assert ready.state is InferenceLeaseState.READY
    assert llm.activated is ready
    assert llm.capabilities == ("chat", "streaming", "tools")
    assert llm.touch_lease == coordinator.touch
    public = ready.to_public_dict()
    assert public["route"] == {
        "model": "qwen3:8b",
        "protocol": "openai",
        "context_window": None,
        "authenticated": True,
    }
    assert "private.example" not in repr(public)
    assert "route-secret" not in repr(public)


@pytest.mark.asyncio
async def test_touch_renews_and_persists_before_next_inference_call():
    events: list[str] = []
    provider = _Provider("runpod", total="0.20", ready_seconds=10, events=events)
    provider.acquire_state = InferenceLeaseState.READY
    llm = _LLMService(events)
    coordinator, persisted = _coordinator(
        {"runpod": provider},
        llm_service=llm,
        events=events,
    )
    ready = await coordinator.acquire(_request())
    events.clear()

    touched = await coordinator.touch(ready.lease_id)

    assert touched.state is InferenceLeaseState.READY
    assert touched.expires_at >= ready.expires_at
    assert provider.touch_calls == 1
    assert events[:3] == ["provider.touch", "persist", "llm.activate"]
    assert persisted[-1]["lease"]["expires_at"] == touched.expires_at.isoformat()


@pytest.mark.asyncio
async def test_touch_and_release_are_serialized_without_lease_resurrection():
    events: list[str] = []
    touch_started = asyncio.Event()
    finish_touch = asyncio.Event()

    class _BlockingTouchProvider(_Provider):
        async def touch(self, owner_id: str, lease_id: str) -> InferenceLease:
            events.append("provider.touch.start")
            touch_started.set()
            await finish_touch.wait()
            events.append("provider.touch.finish")
            return await super().touch(owner_id, lease_id)

    provider = _BlockingTouchProvider(
        "runpod", total="0.20", ready_seconds=10, events=events
    )
    provider.acquire_state = InferenceLeaseState.READY
    coordinator, _persisted = _coordinator(
        {"runpod": provider},
        events=events,
    )
    ready = await coordinator.acquire(_request())
    events.clear()

    touch_task = asyncio.create_task(coordinator.touch(ready.lease_id))
    await touch_started.wait()
    release_task = asyncio.create_task(coordinator.release(ready.lease_id))
    await asyncio.sleep(0)
    assert release_task.done() is False
    assert provider.release_calls == 0

    finish_touch.set()
    await touch_task
    released = await release_task

    assert released.state is InferenceLeaseState.RELEASED
    assert events.index("provider.touch.finish") < events.index("provider.release")
    with pytest.raises(InferenceLeaseProvisioningError, match="only a ready"):
        await coordinator.touch(ready.lease_id)


@pytest.mark.asyncio
async def test_touch_adopts_a_shortened_ready_deadline():
    """A provider that renews to a NEARER expiry is authoritative, not wrong.

    Nothing in the SDK contract makes expires_at monotonic across touch, and a
    provider modelling its idle deadline as min(now + idle_ttl, authorized_cap)
    returns a nearer expiry on its first renewal. Shortening tightens the
    authorization envelope, so it must be adopted rather than rejected -
    rejecting it would leave the coordinator advertising an expiry the provider
    no longer honours. Extension, the direction that actually threatens the
    envelope, is covered by
    test_touch_rejects_expiry_beyond_request_deadline_without_side_effects.
    """

    class _ShorteningProvider(_Provider):
        async def touch(self, owner_id: str, lease_id: str) -> InferenceLease:
            assert self.last_request is not None
            assert self.last_quote is not None
            current = _lease(
                self.last_request,
                self.last_quote,
                state=InferenceLeaseState.READY,
            )
            shortened = _lease(
                self.last_request,
                self.last_quote,
                state=InferenceLeaseState.READY,
                updated_at=current.updated_at + timedelta(seconds=1),
                expires_at=current.expires_at - timedelta(seconds=1),
            )
            shortened.assert_owner(owner_id)
            assert shortened.lease_id == lease_id
            return shortened

    provider = _ShorteningProvider("runpod", total="0.20", ready_seconds=10)
    provider.acquire_state = InferenceLeaseState.READY
    coordinator, persisted = _coordinator({"runpod": provider})
    ready = await coordinator.acquire(_request())

    renewed = await coordinator.touch(ready.lease_id)

    assert renewed.state is InferenceLeaseState.READY
    assert renewed.expires_at < ready.expires_at
    # The shorter expiry is what gets persisted, so a restart cannot resurrect
    # the longer one the provider has stopped honouring.
    assert persisted[-1] is not None
    assert persisted[-1]["lease"]["expires_at"] == renewed.expires_at.isoformat()


@pytest.mark.asyncio
async def test_touch_rejects_expiry_beyond_request_deadline_without_side_effects():
    class _OverextendingProvider(_Provider):
        async def touch(self, owner_id: str, lease_id: str) -> InferenceLease:
            self.events.append("provider.touch")
            self.touch_calls += 1
            assert self.last_request is not None
            assert self.last_quote is not None
            latest_expiry = self.last_request.requested_at + timedelta(
                seconds=(
                    self.last_request.ready_deadline_seconds
                    + self.last_request.expected_session_seconds
                )
            )
            lease = _lease(
                self.last_request,
                self.last_quote,
                state=InferenceLeaseState.READY,
                updated_at=latest_expiry - timedelta(seconds=1),
                expires_at=latest_expiry + timedelta(seconds=1),
            )
            lease.assert_owner(owner_id)
            assert lease.lease_id == lease_id
            return lease

    events: list[str] = []
    provider = _OverextendingProvider(
        "runpod", total="0.20", ready_seconds=10, events=events
    )
    provider.acquire_state = InferenceLeaseState.READY
    llm = _LLMService(events)
    coordinator, persisted = _coordinator(
        {"runpod": provider},
        llm_service=llm,
        events=events,
    )
    ready = await coordinator.acquire(_request())
    persisted_before_touch = list(persisted)
    events.clear()

    with pytest.raises(InferenceLeaseConstraintError, match="session deadline"):
        await coordinator.touch(ready.lease_id)

    assert provider.touch_calls == 1
    assert events == ["provider.touch"]
    assert persisted == persisted_before_touch
    assert coordinator.current_record is not None
    assert coordinator.current_record.lease is ready
    assert llm.activated is ready


@pytest.mark.asyncio
async def test_release_deactivates_and_persists_before_provider_mutation():
    events: list[str] = []
    provider = _Provider("runpod", total="0.20", ready_seconds=10, events=events)
    provider.acquire_state = InferenceLeaseState.READY
    llm = _LLMService(events)
    coordinator, _persisted = _coordinator(
        {"runpod": provider},
        llm_service=llm,
        events=events,
    )
    ready = await coordinator.acquire(_request())
    events.clear()

    released = await coordinator.release(ready.lease_id)

    assert released.state is InferenceLeaseState.RELEASED
    assert events[:3] == ["llm.deactivate", "persist", "provider.release"]


@pytest.mark.asyncio
async def test_owner_scope_rejects_unknown_lease_before_provider_call():
    provider = _Provider("runpod", total="0.20", ready_seconds=10)
    coordinator, _persisted = _coordinator({"runpod": provider})
    await coordinator.acquire(_request())

    with pytest.raises(InferenceLeaseNotFoundError):
        await coordinator.status("lease-another-agent")

    assert provider.status_calls == 0


@pytest.mark.asyncio
async def test_restore_plan_resumes_same_request_after_pre_return_crash():
    provider = _Provider("runpod", total="0.20", ready_seconds=10)
    request = _request(request_id="survives-crash")
    quote = await provider.quote(request)
    plan = {
        "schema_version": 1,
        "provider_name": "runpod",
        "request": request.to_public_dict(),
        "quote": quote.to_public_dict(),
        "lease": None,
    }
    second, _second_persisted = _coordinator({"runpod": provider})

    restored = await second.restore(plan)

    assert restored is not None
    assert restored.request_id == "survives-crash"
    assert provider.acquire_calls == 1


@pytest.mark.asyncio
async def test_restore_ready_reference_refetches_host_only_route():
    provider = _Provider("runpod", total="0.20", ready_seconds=10)
    provider.acquire_state = InferenceLeaseState.READY
    first, persisted = _coordinator({"runpod": provider})
    await first.acquire(_request())
    payload = persisted[-1]
    assert payload is not None
    assert "private.example" not in repr(payload)
    llm = _LLMService()
    second, _second_persisted = _coordinator(
        {"runpod": provider},
        llm_service=llm,
    )

    restored = await second.restore(payload)

    assert restored is not None
    assert restored.state is InferenceLeaseState.READY
    assert provider.status_calls == 1
    assert llm.activated is restored
    assert llm.touch_lease == second.touch

    touched = await llm.touch_lease(restored.lease_id)
    assert touched.state is InferenceLeaseState.READY
    assert provider.touch_calls == 1


@pytest.mark.asyncio
async def test_restore_terminal_reference_does_not_repoll_provider():
    provider = _Provider("runpod", total="0.20", ready_seconds=10)
    provider.acquire_state = InferenceLeaseState.READY
    first, persisted = _coordinator({"runpod": provider})
    ready = await first.acquire(_request())
    released = await first.release(ready.lease_id)
    payload = persisted[-1]
    assert payload is not None
    provider.status_calls = 0
    second, _second_persisted = _coordinator({"runpod": provider})

    restored = await second.restore(payload)

    assert restored is not None
    assert restored.state is InferenceLeaseState.RELEASED
    assert restored.lease_id == released.lease_id
    assert provider.status_calls == 0


@pytest.mark.asyncio
async def test_cost_refusal_happens_before_acquire():
    provider = _Provider("runpod", total="0.20", ready_seconds=10)
    coordinator, _persisted = _coordinator({"runpod": provider})

    with pytest.raises(InferenceLeaseProviderUnavailableError):
        await coordinator.acquire(_request(max_total="0.01"))

    assert provider.acquire_calls == 0


@pytest.mark.asyncio
async def test_state_regression_is_rejected():
    provider = _Provider("runpod", total="0.20", ready_seconds=10)
    provider.acquire_state = InferenceLeaseState.READY
    coordinator, _persisted = _coordinator({"runpod": provider})
    ready = await coordinator.acquire(_request())
    provider.status_state = InferenceLeaseState.PENDING

    with pytest.raises(InferenceLeaseConstraintError, match="invalid.*transition"):
        await coordinator.status(ready.lease_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_state",
    (InferenceLeaseState.FAILED, InferenceLeaseState.EXPIRED),
)
async def test_terminal_status_removes_active_route(terminal_state):
    events: list[str] = []
    provider = _Provider("runpod", total="0.20", ready_seconds=10, events=events)
    provider.acquire_state = InferenceLeaseState.READY
    llm = _LLMService(events)
    coordinator, _persisted = _coordinator(
        {"runpod": provider},
        llm_service=llm,
        events=events,
    )
    ready = await coordinator.acquire(_request())
    events.clear()
    provider.status_state = terminal_state

    terminal = await coordinator.status(ready.lease_id)

    assert terminal.state is terminal_state
    assert llm.activated is None
    assert events[-1] == "llm.deactivate"


@pytest.mark.asyncio
async def test_durable_write_failure_prevents_billable_acquire():
    provider = _Provider("runpod", total="0.20", ready_seconds=10)

    async def fail_persist(_payload):
        raise OSError("database path is unavailable")

    coordinator = InferenceLeaseCoordinator(
        llm_service=_LLMService(),
        owner_id="agent-1",
        providers={"runpod": provider},
        persist_state=fail_persist,
    )

    with pytest.raises(
        InferenceLeaseProvisioningError,
        match="durable inference lease state could not be saved",
    ):
        await coordinator.acquire(_request())

    assert provider.acquire_calls == 0


@pytest.mark.asyncio
async def test_acquire_failure_retries_only_with_same_request_id():
    class _FailOnceProvider(_Provider):
        async def acquire(self, request, quote):
            self.acquire_calls += 1
            self.last_request = request
            self.last_quote = quote
            if self.acquire_calls == 1:
                raise ConnectionError("provider control plane unavailable")
            return _lease(request, quote, state=InferenceLeaseState.PENDING)

    provider = _FailOnceProvider("runpod", total="0.20", ready_seconds=10)
    coordinator, _persisted = _coordinator({"runpod": provider})
    request = _request(request_id="durable-request")

    with pytest.raises(
        InferenceLeaseProvisioningError,
        match="retry with the same request_id",
    ):
        await coordinator.acquire(request)
    with pytest.raises(InferenceLeaseProvisioningError, match="already in progress"):
        await coordinator.acquire(_request(request_id="different-request"))

    lease = await coordinator.acquire(request)
    assert lease.request_id == "durable-request"
    assert provider.acquire_calls == 2


class _EntryPoint:
    def __init__(self, name, value, provider, distribution):
        self.name = name
        self.value = value
        self._provider = provider
        self.dist = SimpleNamespace(name=distribution)

    def load(self):
        return self._provider


class _EntryPoints(tuple):
    def select(self, *, group):
        assert group == "kestrel_sovereign.inference_lease_providers"
        return self


def test_discovery_rejects_duplicate_provider_claims(monkeypatch):
    first = _Provider("runpod", total="0.20", ready_seconds=10)
    second = _Provider("runpod", total="0.20", ready_seconds=10)
    entry_points = _EntryPoints(
        (
            _EntryPoint("runpod", "a:Provider", first, "package-a"),
            _EntryPoint("runpod", "b:Provider", second, "package-b"),
        )
    )
    monkeypatch.setattr(
        "kestrel_sovereign.llm.inference_leases.importlib_metadata.entry_points",
        lambda: entry_points,
    )

    with pytest.raises(InferenceLeaseProviderDiscoveryError, match="duplicate"):
        discover_inference_lease_providers()


def test_discovery_rejects_duplicate_claims_from_one_distribution(monkeypatch):
    first = _Provider("runpod", total="0.20", ready_seconds=10)
    second = _Provider("runpod", total="0.20", ready_seconds=10)
    entry_points = _EntryPoints(
        (
            _EntryPoint("runpod", "a:Provider", first, "package-a"),
            _EntryPoint("runpod", "b:Provider", second, "package-a"),
        )
    )
    monkeypatch.setattr(
        "kestrel_sovereign.llm.inference_leases.importlib_metadata.entry_points",
        lambda: entry_points,
    )

    with pytest.raises(InferenceLeaseProviderDiscoveryError, match="duplicate"):
        discover_inference_lease_providers()


def test_discovery_rejects_entry_point_name_mismatch(monkeypatch):
    provider = _Provider("runpod", total="0.20", ready_seconds=10)
    entry_points = _EntryPoints(
        (_EntryPoint("vast", "a:Provider", provider, "package-a"),)
    )
    monkeypatch.setattr(
        "kestrel_sovereign.llm.inference_leases.importlib_metadata.entry_points",
        lambda: entry_points,
    )

    with pytest.raises(InferenceLeaseProviderDiscoveryError, match="different"):
        discover_inference_lease_providers()


def test_invalid_provider_protocol_is_rejected(monkeypatch):
    entry_points = _EntryPoints(
        (_EntryPoint("broken", "a:Provider", object(), "package-a"),)
    )
    monkeypatch.setattr(
        "kestrel_sovereign.llm.inference_leases.importlib_metadata.entry_points",
        lambda: entry_points,
    )

    with pytest.raises(InferenceLeaseProviderDiscoveryError, match="SDK contract"):
        discover_inference_lease_providers()


@pytest.mark.parametrize("declare_touch_as_none", [False, True])
def test_provider_without_idle_renewal_is_rejected_at_discovery(
    monkeypatch, declare_touch_as_none
):
    """Discovery rejects a provider that cannot renew a lease from traffic.

    Both shapes are the SDK protocol's job, not ours: contract 6 made ``touch``
    a required member of the runtime-checkable ``InferenceLeaseProvider``, so a
    0.34-era provider that omits it entirely and one that declares it as
    ``None`` both fail ``isinstance``. Discovery must not re-implement that
    check, and this test exists to prove the SDK-owned one is load-bearing.
    """

    class _LegacyProvider:
        provider_name = "legacy"

        def capabilities(self):
            return ()

        def is_available(self):
            return True

        async def quote(self, request):
            raise NotImplementedError

        async def acquire(self, request, quote):
            raise NotImplementedError

        async def status(self, owner_id, lease_id):
            raise NotImplementedError

        async def release(self, owner_id, lease_id):
            raise NotImplementedError

    if declare_touch_as_none:
        _LegacyProvider.touch = None

    entry_points = _EntryPoints(
        (_EntryPoint("legacy", "legacy:Provider", _LegacyProvider(), "legacy"),)
    )
    monkeypatch.setattr(
        "kestrel_sovereign.llm.inference_leases.importlib_metadata.entry_points",
        lambda: entry_points,
    )

    with pytest.raises(InferenceLeaseProviderDiscoveryError, match="SDK contract"):
        discover_inference_lease_providers()
