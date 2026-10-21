"""Terminal routing state is not permanent evidence of provider absence."""

import asyncio
from dataclasses import replace
from decimal import Decimal

import pytest
from kestrel_sdk.llm import (
    InferenceLeaseConstraintError,
    InferenceLeaseNotFoundError,
    InferenceLeaseOwnershipError,
    InferenceLeaseProviderUnavailableError,
    InferenceLeaseProvisioningError,
    InferenceLeaseState,
)

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    ProcessRuntimeExecutionFence,
    bind_execution_custody,
    require_execution_work,
)
from kestrel_sovereign.llm.inference_leases import InferenceLeaseCoordinator
from tests.unit.test_inference_lease_coordinator import (
    _coordinator,
    _LLMService,
    _Provider,
    _request,
)

TERMINAL_STATES = (
    InferenceLeaseState.FAILED,
    InferenceLeaseState.RELEASED,
    InferenceLeaseState.EXPIRED,
)
ENTRY_POINTS = ("restore", "status", "acquire", "release")


class ReconciliationProvider(_Provider):
    """Exercise the real coordinator with explicit original-provider responses."""

    status_error: BaseException | None = None
    response_change: str | None = None

    async def status(self, owner_id, lease_id):
        if self.status_error is not None:
            self.status_calls += 1
            raise self.status_error
        lease = await super().status(owner_id, lease_id)
        if self.response_change == "foreign":
            return replace(lease, lease_id="another-lease")
        if self.response_change == "stale":
            return replace(lease, updated_at=lease.created_at)
        if self.response_change == "owner":
            return replace(lease, owner_id="another-owner")
        if self.response_change == "request":
            return replace(lease, request_id="another-request")
        if self.response_change == "quote":
            return replace(lease, quote_id="another-quote")
        if self.response_change == "provider":
            return replace(lease, provider_name="another-provider")
        if self.response_change == "region":
            return replace(lease, region="us-ca-1")
        if self.response_change == "hourly":
            return replace(lease, hourly_cost_usd=Decimal("0.51"))
        if self.response_change == "total":
            return replace(lease, estimated_total_cost_usd=Decimal("0.21"))
        return lease

    async def release(self, owner_id, lease_id):
        lease = await super().release(owner_id, lease_id)
        # Unresolved ownership remains nonterminal even after exact cleanup is
        # requested; a request/ACK is not evidence of physical absence.
        return replace(lease, state=InferenceLeaseState.RELEASING)


async def original_terminal_record(state, entry_point):
    provider = ReconciliationProvider("runpod", total="0.20", ready_seconds=10)
    provider.acquire_state = state
    llm = _LLMService()
    coordinator, persisted = _coordinator({"runpod": provider}, llm_service=llm)
    cached = await coordinator.acquire(_request())
    original_payload = persisted[-1]
    if entry_point == "restore":
        coordinator, persisted = _coordinator({"runpod": provider}, llm_service=llm)
    return provider, coordinator, persisted, cached, original_payload, llm


async def enter(coordinator, entry_point, cached, payload):
    if entry_point == "restore":
        return await coordinator.restore(payload)
    if entry_point == "status":
        return await coordinator.status(cached.lease_id)
    if entry_point == "release":
        return await coordinator.release(cached.lease_id)
    return await coordinator.acquire(_request(request_id="replacement"))


@pytest.mark.asyncio
@pytest.mark.parametrize("state", TERMINAL_STATES)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
async def test_original_provider_debt_invalidates_terminal_cache(state, entry_point):
    (
        provider,
        coordinator,
        persisted,
        cached,
        payload,
        llm,
    ) = await original_terminal_record(state, entry_point)
    provider.status_state = InferenceLeaseState.RELEASING

    if entry_point == "acquire":
        with pytest.raises(
            InferenceLeaseProvisioningError, match="release the current"
        ):
            await enter(coordinator, entry_point, cached, payload)
    else:
        current = await enter(coordinator, entry_point, cached, payload)
        assert current.state is InferenceLeaseState.RELEASING
        assert current.lease_id == cached.lease_id
        assert current.request_id == cached.request_id
        assert current.hourly_cost_usd == cached.hourly_cost_usd
        assert current.estimated_total_cost_usd == cached.estimated_total_cost_usd
        assert current.route is None

    assert provider.status_calls == 1
    assert provider.acquire_calls == 1
    assert provider.release_calls == int(entry_point == "release")
    assert llm.activated is None
    assert persisted[-1]["lease"]["state"] == "releasing"
    assert persisted[-1]["request"]["request_id"] == cached.request_id

    # The invalidated record remains nonterminal on another process restore.
    restarted, _ = _coordinator({"runpod": provider})
    restored = await restarted.restore(persisted[-1])
    assert restored.state is InferenceLeaseState.RELEASING
    with pytest.raises(InferenceLeaseProvisioningError, match="release the current"):
        await restarted.acquire(_request(request_id="third-request"))
    assert provider.acquire_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("state", TERMINAL_STATES)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
@pytest.mark.parametrize(
    "failure", ("unavailable", "missing", "denied", "unknown", "cancelled")
)
async def test_failed_refresh_never_permits_new_capacity(state, entry_point, failure):
    (
        provider,
        coordinator,
        persisted,
        cached,
        payload,
        llm,
    ) = await original_terminal_record(state, entry_point)
    error = {
        "unavailable": ConnectionError("original provider unavailable"),
        "missing": InferenceLeaseNotFoundError("original evidence missing"),
        "denied": ExecutionAuthorityError("original admission lost"),
        "unknown": ExecutionCommitOutcomeError("unknown"),
        "cancelled": asyncio.CancelledError("original caller cancelled"),
    }[failure]
    provider.status_error = error
    before = list(persisted)

    expected = (
        InferenceLeaseProvisioningError
        if failure in {"unavailable", "missing"}
        else type(error)
    )
    with pytest.raises(expected) as caught:
        await enter(coordinator, entry_point, cached, payload)
    if failure in {"unavailable", "missing"}:
        assert caught.value.__cause__ is error
    else:
        assert caught.value is error

    assert provider.status_calls == 1
    assert provider.acquire_calls == 1
    assert provider.release_calls == 0
    assert persisted == before
    assert coordinator.current_record.lease.lease_id == cached.lease_id
    assert coordinator.current_record.lease.state is state
    assert llm.activated is None


@pytest.mark.asyncio
@pytest.mark.parametrize("state", TERMINAL_STATES)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
@pytest.mark.parametrize(
    "change",
    (
        "foreign",
        "stale",
        "pending",
        "ready",
        "owner",
        "request",
        "quote",
        "provider",
        "region",
        "hourly",
        "total",
    ),
)
async def test_refresh_rejects_identity_change_staleness_and_work_revival(
    state, entry_point, change
):
    (
        provider,
        coordinator,
        persisted,
        cached,
        payload,
        llm,
    ) = await original_terminal_record(state, entry_point)
    provider.status_state = {
        "pending": InferenceLeaseState.PENDING,
        "ready": InferenceLeaseState.READY,
    }.get(change, InferenceLeaseState.RELEASING)
    provider.response_change = change
    before = list(persisted)

    expected = (
        InferenceLeaseOwnershipError
        if change == "owner"
        else InferenceLeaseConstraintError
    )
    with pytest.raises(expected):
        await enter(coordinator, entry_point, cached, payload)

    assert provider.status_calls == 1
    assert provider.acquire_calls == 1
    assert provider.release_calls == 0
    assert persisted == before
    assert coordinator.current_record.lease.lease_id == cached.lease_id
    assert coordinator.current_record.lease.state is state
    assert llm.activated is None


@pytest.mark.asyncio
@pytest.mark.parametrize("state", TERMINAL_STATES)
async def test_confirmed_terminal_original_allows_replacement(state):
    (
        provider,
        coordinator,
        persisted,
        cached,
        _payload,
        llm,
    ) = await original_terminal_record(state, "acquire")
    provider.status_state = state
    provider.acquire_state = InferenceLeaseState.PENDING

    replacement = await coordinator.acquire(_request(request_id="replacement"))

    assert provider.status_calls == 1
    assert provider.acquire_calls == 2
    assert replacement.request_id == "replacement"
    assert replacement.lease_id != cached.lease_id
    assert persisted[-1]["request"]["request_id"] == "replacement"
    assert llm.activated is None


@pytest.mark.asyncio
@pytest.mark.parametrize("state", TERMINAL_STATES)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
async def test_removed_original_provider_cannot_fall_through_to_another(
    state, entry_point
):
    (
        original,
        _coordinator_before_restart,
        _persisted,
        cached,
        payload,
        llm,
    ) = await original_terminal_record(state, "restore")
    other = _Provider("other", total="0.10", ready_seconds=5)
    coordinator, persisted = _coordinator({"other": other}, llm_service=llm)
    # A failed restore must retain the original record, not clear its identity.
    with pytest.raises(InferenceLeaseProviderUnavailableError):
        await coordinator.restore(payload)
    with pytest.raises(InferenceLeaseProviderUnavailableError):
        await enter(coordinator, entry_point, cached, payload)
    assert original.status_calls == 0
    assert original.acquire_calls == 1
    assert other.acquire_calls == 0
    assert persisted == []
    assert coordinator.current_record.lease.lease_id == cached.lease_id
    assert llm.activated is None


@pytest.mark.asyncio
@pytest.mark.parametrize("state", TERMINAL_STATES)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
async def test_failed_debt_persistence_does_not_authorize_replacement(
    state, entry_point
):
    provider = ReconciliationProvider("runpod", total="0.20", ready_seconds=10)
    provider.acquire_state = state
    persisted = []
    failure = None

    async def persist(payload):
        if failure is not None:
            raise failure
        persisted.append(payload)

    coordinator = InferenceLeaseCoordinator(
        llm_service=_LLMService(),
        owner_id="agent-1",
        providers={"runpod": provider},
        persist_state=persist,
    )
    cached = await coordinator.acquire(_request())
    payload = persisted[-1]
    before = list(persisted)
    failure = OSError("original durable store unavailable")
    provider.status_state = InferenceLeaseState.RELEASING
    with pytest.raises(InferenceLeaseProvisioningError) as caught:
        await enter(coordinator, entry_point, cached, payload)
    assert caught.value.__cause__ is failure
    assert provider.acquire_calls == 1
    assert provider.release_calls == 0
    assert persisted == before
    assert coordinator.current_record.lease.state is InferenceLeaseState.RELEASING
    with pytest.raises(InferenceLeaseProvisioningError, match="release the current"):
        await coordinator.acquire(_request(request_id="not-a-retry"))
    assert provider.acquire_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("state", TERMINAL_STATES)
async def test_exact_cleanup_does_not_reopen_original_denied_work(state):
    (
        provider,
        coordinator,
        _persisted,
        cached,
        _payload,
        _llm,
    ) = await original_terminal_record(state, "release")
    provider.status_state = InferenceLeaseState.RELEASING
    fence = ProcessRuntimeExecutionFence("original-cleanup-owner")
    with bind_execution_custody(fence):
        fence.retire()
        with pytest.raises(ExecutionAuthorityError):
            require_execution_work()
        result = await coordinator.release(cached.lease_id)
        assert result.state is InferenceLeaseState.RELEASING
        with pytest.raises(ExecutionAuthorityError):
            require_execution_work()
    assert provider.status_calls == 1
    assert provider.acquire_calls == 1
    assert provider.release_calls == 1
