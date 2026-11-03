"""Actual runtime custody negotiation and canonical owned native TLS create.

Only the downstream session driver/readiness are bounded doubles. These tests
never provision a Pod or establish a remote provider's worker conformance.
"""
from datetime import UTC, datetime
import copy
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import SecretBytes
from kestrel_sdk.computer_use import ComputerSession, DriverConnection, EnvironmentLease
from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionPreparation, EnvironmentAcquisitionStorage
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceUnsupported
from kestrel_sovereign.execution_sync_http import await_native_sync_http, bind_native_sync_http_execution_admission
from kestrel_sovereign.execution_custody import ExecutionAuthorityError, is_execution_control_error
from kestrel_sovereign.visual_computer.runtime import VisualComputerRuntime, VisualPolicy
from kestrel_sovereign.visual_computer.runpod_acquisition import prepare_runpod_acquisition_producer
from tests.integration.test_visual_runpod_acquisition import (
    custody as custody, vendor as vendor, peer as peer, SCOPE, original_host, original_quote,
)


class BoundedSessionDriver:
    driver_id = 'bounded-downstream-driver'

    def __init__(self):
        self.opens = 0

    async def open(self, request, connection):
        self.opens += 1
        return ComputerSession(session_id=request.session_id, scope=request.scope,
            lease_id=request.lease_id, driver_id=self.driver_id, surface=request.surface,
            capabilities=request.required_capabilities, state='ready', controller='agent',
            control_epoch=0, expires_at=request.expires_at)


def runtime_for(host, provider, driver):
    def destination(location):
        host.check()
        if location != 'remote':
            raise ResourceUnsupported('fixture destination was not explicitly granted')
    return VisualComputerRuntime(host.store, host.authority,
        providers={host.request.provider_id: provider}, drivers={driver.driver_id: driver},
        policy=VisualPolicy(permitted_locations=frozenset({'remote'}), max_lifetime_seconds=3600,
            max_hourly_cost_usd=Decimal('0.20'), max_total_cost_usd=Decimal('0.11')),
        check_destination=destination)


class NativeFixtureProvider:
    """SDK protocol fixture over the genuine closed canonical producer."""

    def __init__(self, host, *, mode):
        self.host, self.mode = host, mode
        self.connects = self.modern_calls = self.legacy_calls = 0
        self.original = None
        self.stages = []

    async def capabilities(self):
        return original_quote(self.host.plan).capabilities

    async def quote(self, request):
        assert request == self.host.request and request.digest() == self.host.request.digest()
        return original_quote(self.host.plan)

    async def acquire(self, *args):
        self.legacy_calls += 1
        pytest.fail('legacy create must never run after modern negotiation')

    async def inspect(self, scope, lease):
        return lease.model_copy(update={'state': 'ready'})  # Bounded downstream double.

    async def connect(self, scope, lease):
        self.connects += 1
        return DriverConnection(lease_id=lease.lease_id, driver_id=BoundedSessionDriver.driver_id,
            endpoint='bounded-owned-fixture', credential='private-bounded-double', expires_at=lease.expires_at)

    async def release(self, *args):
        raise ResourceUnsupported('bounded fixture has no physical resource release')

    async def acquire_with_custody(self, request, quote, custody):
        self.modern_calls += 1
        host = self.host
        current = await host.store.get(SCOPE, request_id=request.request_id, guard=host.check)
        self.stages.append(current.phase)
        assert current.lease is None
        async with custody.operation(request, quote) as attempt:
            accepted = await self._run_original_attempt(attempt, request, quote)
        if type(accepted) is EnvironmentLease:
            return accepted
        if self.mode == 'stale-row':
            current = await host.store.get(SCOPE, request_id=request.request_id, guard=host.check)
            current.last_activity_at = datetime.now(UTC)
            await host.store.save(SCOPE, current, host.check)
        elif self.mode == 'wrong-dto':
            return accepted.lease.model_copy(update={'provider_resource_id': 'forged-provider-dto'})
        return accepted.lease

    async def _run_original_attempt(self, attempt, request, quote):
        host = self.host
        if self.mode == 'prepared-only':
            assert attempt.checkpoint.state == 'unprepared'
            return EnvironmentLease(lease_id='unproven-sdk-dto', request_id=request.request_id,
                quote_id=quote.quote_id, scope=request.scope, provider_id=request.provider_id,
                location=request.location, expires_at=request.expires_at, state='pending',
                provider_resource_id='unproven-sdk-dto')
        dto = host.plan.pod_request(request, host.plan.profile, host.plan.noncompute, at=host.plan.planned_at)
        prepared = host.control.prepare_cpu_pod(dto, plan=host.plan)
        preparation = EnvironmentAcquisitionPreparation(create_payload=SecretBytes(prepared.exact_body()),
            worker_identity=SecretBytes(b'PRIVATE-ORIGINAL-WORKER-TLS-IDENTITY'))
        producer = await prepare_runpod_acquisition_producer(attempt, host.control, dto,
            prepared=prepared, plan=host.plan, preparation=preparation)
        self.original = original = SimpleNamespace(attempt=attempt, producer=producer,
            handle=producer._original_acquisition)
        assert await attempt.begin_dispatch() is True
        if self.mode == 'native-no-id':
            del self.host.peer.reply['id']
        try:
            phase = await original.handle.run_original()
        except BaseException:
            await original.handle.persist_minimal()
            raise
        checkpoint = await original.handle.persist_minimal()
        producer.finish_optional_cost(phase)
        if self.mode == 'missing-accepted':
            return checkpoint.lease.model_copy(update={'state': 'pending'})
        accepted = await attempt.record_accepted(checkpoint.lease.model_copy(update={'state': 'pending'}))
        return accepted


async def source_acquisition(host, provider):
    """Explicit source-only conformance; never a runtime admission bypass."""
    from kestrel_sovereign.visual_computer.repository import CustodyRecord
    try:
        current = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
    except ResourceNotFound:
        current = await host.store.admit(SCOPE,
            CustodyRecord(request=host.request, session_id='source-only-session'), host.check)
        current.quote, current.phase = original_quote(host.plan), 'quoted'
        await host.store.save(SCOPE, current, host.check)
    provider.modern_calls += 1
    async with host.store.original_acquisition_result(SCOPE, current, host.request,
            current.quote, guard=host.check) as attempt:
        return await provider._run_original_attempt(attempt, host.request, current.quote)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['genuine', 'missing-accepted', 'wrong-dto', 'stale-row'])
async def test_runtime_refuses_native_worker_before_any_create(custody, vendor, peer, monkeypatch, mode):
    async with original_host(custody, vendor, peer, monkeypatch) as host:
        provider, driver = NativeFixtureProvider(host, mode=mode), BoundedSessionDriver()
        runtime = runtime_for(host, provider, driver)
        with pytest.raises(ResourceUnsupported, match='worker'):
            await runtime.admit(host.context, host.request)
        assert driver.opens == provider.connects == provider.legacy_calls == 0
        current = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        saved = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage)
        assert saved.checkpoint.state == 'unprepared' and saved.preparation is None
        assert current.lease is None and current.acquisition.binding is None
        assert current.phase == 'quoted' and provider.stages == ['quoted']
        assert provider.modern_calls == 1 and not peer.requests and not peer.snis


@pytest.mark.asyncio
async def test_sdk_preparation_scope_without_native_acceptance_cannot_open_and_can_retire(custody, vendor, peer, monkeypatch):
    async with original_host(custody, vendor, peer, monkeypatch) as host:
        provider, driver = NativeFixtureProvider(host, mode='prepared-only'), BoundedSessionDriver()
        runtime = runtime_for(host, provider, driver)
        with pytest.raises(ResourceOutcomeUnknown, match='accepted native ACK'):
            await runtime.admit(host.context, host.request)
        assert provider.connects == driver.opens == provider.legacy_calls == 0
        assert not peer.requests and not peer.snis
        current = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        checkpoint = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage).checkpoint
        assert current.phase == 'quoted' and checkpoint.state == 'unprepared' and current.lease is None
        outcome = await runtime.reconcile_request(host.context, host.request.request_id)
        assert outcome['retired'] is True and outcome['acquired'] is outcome['released'] is False
        retired = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        assert retired.phase == 'retired' and retired.acquisition == current.acquisition


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['withdrawn-protocol', 'missing-capability'])
async def test_advertised_custody_cannot_fall_back_to_uncustodied_native_post(custody, vendor, peer, monkeypatch, fault):
    async with original_host(custody, vendor, peer, monkeypatch) as host:
        from kestrel_sovereign import execution_sync_http as native_module
        provider, driver = NativeFixtureProvider(host, mode='genuine'), BoundedSessionDriver()
        # A deliberate native API2 effect trap, not a remote provider adapter.
        # If cutover falls through, the SAME actual client sends to our owned
        # TLS peer. No fake handle or second HTTP engine can hide the failure.
        dto = host.plan.pod_request(host.request, host.plan.profile, host.plan.noncompute, at=host.plan.planned_at)
        body = host.control.prepare_cpu_pod(dto, plan=host.plan).exact_body()
        async def forbidden_uncustodied(request, quote):
            provider.legacy_calls += 1
            def original_io(client):
                return client.request('POST', 'pods', content=body, headers={'Content-Type': 'application/json'})
            with bind_native_sync_http_execution_admission(host.native, host.check):
                await await_native_sync_http(host.native, original_io)
            raise RuntimeError('UNCUSTODIED_ACTUAL_POSTS='+str(len(peer.requests)))
        monkeypatch.setattr(provider, 'acquire', forbidden_uncustodied)
        actual_quote = provider.quote
        async def quote_then_withdraw(request):
            quote = await actual_quote(request)
            if fault == 'withdrawn-protocol':
                monkeypatch.delattr(NativeFixtureProvider, 'acquire_with_custody')
            else:
                monkeypatch.setattr(native_module, 'NATIVE_SYNC_HTTP_ACQUISITION_API_VERSION', None)
            return quote
        monkeypatch.setattr(provider, 'quote', quote_then_withdraw)
        runtime = runtime_for(host, provider, driver)
        with pytest.raises(ResourceUnsupported):
            await runtime.admit(host.context, host.request)
        assert provider.legacy_calls == provider.connects == driver.opens == 0
        assert not peer.requests and not peer.snis


@pytest.mark.asyncio
async def test_copied_custody_cannot_replace_same_original_completion(custody, vendor, peer, monkeypatch):
    async with original_host(custody, vendor, peer, monkeypatch) as host:
        provider, driver = NativeFixtureProvider(host, mode='genuine'), BoundedSessionDriver()
        actual = provider.acquire_with_custody
        async def copied(request, quote, custody):
            return await actual(request, quote, copy.copy(custody))
        monkeypatch.setattr(provider, 'acquire_with_custody', copied)
        runtime = runtime_for(host, provider, driver)
        with pytest.raises(ExecutionAuthorityError, match='Original runtime acquisition consumer'):
            await runtime.admit(host.context, host.request)
        current = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        saved = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage)
        assert saved.checkpoint.state == 'unprepared' and saved.preparation is None
        assert current.acquisition.binding is None and current.lease is None
        assert driver.opens == provider.connects == 0 and not peer.requests


@pytest.mark.asyncio
async def test_real_native_unknown_without_id_never_recreates_after_fresh_host_admission(custody, vendor, peer, monkeypatch):
    from kestrel_cloud_runpod.models import RunPodAmbiguousResultError
    async with original_host(custody, vendor, peer, monkeypatch) as host:
        host.peer = peer
        provider, driver = NativeFixtureProvider(host, mode='native-no-id'), BoundedSessionDriver()
        runtime = runtime_for(host, provider, driver)
        with pytest.raises(RunPodAmbiguousResultError):
            await source_acquisition(host, provider)
        current = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        saved = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage)
        assert saved.checkpoint.state == 'unknown' and saved.checkpoint.lease is None
        assert current.lease is None and current.phase == 'acquiring'
        assert len(peer.requests) == 1 and host.native.is_closed
        host.turn.revoke('global Stop after joined original unknown/no-ID fact')
        with pytest.raises(Exception) as denied:
            await host.store.db.fetchone('SELECT 1')
        assert is_execution_control_error(denied.value)
    async with original_host(custody, vendor, peer, monkeypatch) as fresh:
        provider, driver = NativeFixtureProvider(fresh, mode='genuine'), BoundedSessionDriver()
        async def forbidden_uncustodied(request, quote):
            provider.legacy_calls += 1
            def io(client):
                return client.request('POST', 'pods', content=saved.preparation.create_payload.get_secret_value(),
                    headers={'Content-Type': 'application/json'})
            with bind_native_sync_http_execution_admission(fresh.native, fresh.check):
                await await_native_sync_http(fresh.native, io)
            raise RuntimeError('REPLAY_ACTUAL_POSTS='+str(len(peer.requests)))
        monkeypatch.setattr(provider, 'acquire', forbidden_uncustodied)
        runtime = runtime_for(fresh, provider, driver)
        with pytest.raises(ResourceOutcomeUnknown, match='create replay is forbidden'):
            await runtime.reconcile_request(fresh.context, current.request.request_id)
        unchanged = await fresh.store.get(SCOPE, request_id=current.request.request_id, guard=fresh.check)
        assert unchanged.revision == current.revision and unchanged.acquisition == current.acquisition
        assert provider.modern_calls == provider.legacy_calls == provider.connects == driver.opens == 0
        assert len(peer.requests) == 1 and len(peer.snis) == 1
