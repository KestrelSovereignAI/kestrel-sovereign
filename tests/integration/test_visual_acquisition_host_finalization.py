"""Original host lifetime must retain joined native facts without provider help.

Only owned TLS peers are contacted; no Pod, remote worker or billing proof.
"""
import asyncio
import copy
import ssl
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from httpcore._backends.sync import SyncStream
from pydantic import SecretBytes
from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionPreparation, EnvironmentAcquisitionStorage
from kestrel_sovereign.execution_custody import _explicit_control_chain, is_execution_control_error
from kestrel_sovereign.execution_sync_http import await_native_sync_http, bind_native_sync_http_execution_admission
from kestrel_sovereign.visual_computer.runpod_acquisition import prepare_runpod_acquisition_producer
from tests.integration.test_visual_custodied_runtime import NativeFixtureProvider, BoundedSessionDriver, runtime_for
from tests.integration.test_visual_runpod_acquisition import (
    custody as custody, vendor as vendor, peer as peer, SCOPE,
    original_host, original_acquisition,
)


class ProviderAfterNativeError(RuntimeError):
    pass


class FailingAfterNativeProvider(NativeFixtureProvider):
    """A genuine SDK provider which never calls persist_minimal."""
    def __init__(self, host, error, *, after_native=None, explicit=None):
        super().__init__(host, mode='raise-after-native')
        self.error, self.after_native, self.explicit = error, after_native, explicit

    async def acquire_with_custody(self, request, quote, custody):
        self.modern_calls += 1
        host = self.host
        async with custody.operation(request, quote) as attempt:
            dto = host.plan.pod_request(request, host.plan.profile, host.plan.noncompute, at=host.plan.planned_at)
            prepared = host.control.prepare_cpu_pod(dto, plan=host.plan)
            preparation = EnvironmentAcquisitionPreparation(create_payload=SecretBytes(prepared.exact_body()),
                worker_identity=SecretBytes(b'PRIVATE-ORIGINAL-WORKER-TLS-IDENTITY'))
            producer = await prepare_runpod_acquisition_producer(attempt, host.control, dto,
                prepared=prepared, plan=host.plan, preparation=preparation)
            self.original = SimpleNamespace(attempt=attempt, producer=producer, handle=producer._original_acquisition)
            assert await attempt.begin_dispatch() is True
            phase = await self.original.handle.run_original()
            if self.explicit is not None:
                checkpoint = await self.original.handle.persist_minimal()
                if self.explicit == 'accepted':
                    producer.finish_optional_cost(phase)
                    await attempt.record_accepted(checkpoint.lease.model_copy(update={'state': 'pending'}))
            if self.after_native is not None:
                await self.after_native(self.original)
            # The HOST sealed the original phase; no caller supplied Pod DTO,
            # optional-cost continuation or minimal-persistence call occurs.
            assert self.original.handle._receipt.lease.provider_resource_id == 'owned-native-pod'
            raise self.error


@pytest.mark.asyncio
async def test_host_does_not_adopt_a_native_receipt_whose_original_joined_seal_was_refused(custody, vendor, peer, monkeypatch):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        root = original.attempt._result
        assert await original.attempt.begin_dispatch() is True
        start_tls, backing = SyncStream.start_tls, root._native_backing
        def changed_after_tls(stream, *args, **kwargs):
            result = start_tls(stream, *args, **kwargs)
            root._native_backing = copy.copy(backing)
            return result
        monkeypatch.setattr(SyncStream, 'start_tls', changed_after_tls)
        try:
            with pytest.raises(BaseException) as refused:
                await original.handle.run_original()
            assert is_execution_control_error(refused.value)
        finally:
            root._native_backing = backing
        receipt = original.handle._receipt
        assert receipt is not None and receipt.phase is None and receipt.lease is None
        assert root._registration.native_receipt is None
        assert original.attempt.checkpoint.state == 'dispatching'
        assert not original.handle._minimal_attempted and not peer.requests
    assert original.handle._state == 'closed' and original.native.is_closed
    assert original.handle._worker_exited and original.handle._admission.task.done()
    assert not original.handle._minimal_attempted and not original.handle._minimal_acknowledged
    assert not original.store._original_results and not original.store.db.backend._original_result_holds


@pytest.mark.asyncio
async def test_host_retains_original_pod_fact_when_provider_raises_after_native_run(custody, vendor, peer, monkeypatch):
    async with original_host(custody, vendor, peer, monkeypatch) as host:
        error = ProviderAfterNativeError('provider failed after original native return')
        provider = FailingAfterNativeProvider(host, error)
        driver = BoundedSessionDriver()
        runtime = runtime_for(host, provider, driver)
        with pytest.raises(ProviderAfterNativeError) as raised:
            await runtime.admit(host.context, host.request)
        assert raised.value is error
        assert provider.modern_calls == 1 and provider.legacy_calls == provider.connects == driver.opens == 0
        assert len(peer.requests) == 1 and peer.requests[0][0] == '/v2/pods'
        assert host.native.is_closed and len(host.closes) == 1
        handle = provider.original.handle
        assert handle._admission.task.done() and handle._worker_exited and not handle._thread.sockets
        assert handle._state == 'closed' and not host.store._original_results
        assert not host.store.db.backend._original_result_holds
        current = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        checkpoint = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage).checkpoint
        assert current.lease is not None, 'host closed without retaining its original sealed Pod fact'
        assert current.lease.provider_resource_id == checkpoint.lease.provider_resource_id == 'owned-native-pod'
        assert checkpoint.state == 'unknown' and current.lease.state == 'unknown'
        assert current.lease.actual_cost_usd is None and current.acquisition.native_result is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['closed', 'direct-without-spending'])
async def test_original_native_entry_always_rechecks_spending_between_tls_writes(custody, vendor, peer, monkeypatch, entry):
    async with original_acquisition(custody, vendor, peer, monkeypatch, quote_seconds=2) as original:
        assert await original.attempt.begin_dispatch() is True
        native_send, written, clients = ssl.SSLSocket.send, [], []
        def partial(sock, data, flags=0):
            if bytes(data[:5]) == b'POST ' and not clients:
                clients.append(sock)
                count = native_send(sock, data[:17], flags)
                written.append(bytes(data[:count]))
                remaining = (original.plan.expires_at-datetime.now(UTC)).total_seconds()
                if remaining > 0:
                    time.sleep(remaining+0.05)
                return count
            count = native_send(sock, data, flags)
            if clients and sock is clients[0]:
                written.append(bytes(data[:count]))
            return count
        monkeypatch.setattr(ssl.SSLSocket, 'send', partial)
        failure = None
        try:
            if entry == 'closed':
                await original.handle.run_original()
            else:
                # Genuine original Host checker, deliberately NO spending hook.
                with bind_native_sync_http_execution_admission(original.native, original.check):
                    await await_native_sync_http(original.native, original.producer.operation,
                        _original_acquisition=original.handle)
        except BaseException as error:
            failure = error
        assert original.native.is_closed and original.handle._worker_exited
        assert original.handle._admission.task.done() and not original.handle._thread.sockets
        assert len(written) == 1 and len(written[0]) == 17, 'original native entry allowed bytes after spending expiry'
        assert failure is not None and original.handle._thread.sent
        assert original.handle._receipt.phase is None and not peer.requests


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit', ['minimal', 'accepted'])
async def test_host_finalization_preserves_explicit_original_ack_without_another_write(custody, vendor, peer, monkeypatch, explicit):
    async with original_host(custody, vendor, peer, monkeypatch) as host:
        calls, actual = [], host.store.db.backend._settle_original_acquisition_result
        async def counted(owner):
            calls.append(owner._entry.kind)
            return await actual(owner)
        monkeypatch.setattr(host.store.db.backend, '_settle_original_acquisition_result', counted)
        error = ProviderAfterNativeError('after an explicit original ACK')
        provider = FailingAfterNativeProvider(host, error, explicit=explicit)
        driver = BoundedSessionDriver()
        with pytest.raises(ProviderAfterNativeError) as raised:
            await runtime_for(host, provider, driver).admit(host.context, host.request)
        assert raised.value is error and calls.count('minimal') == 1
        assert calls.count('accepted') == (1 if explicit == 'accepted' else 0)
        current = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        checkpoint = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage).checkpoint
        assert checkpoint.state == ('accepted' if explicit == 'accepted' else 'unknown')
        assert current.lease.provider_resource_id == 'owned-native-pod' and current.lease.actual_cost_usd is None
        assert not host.store._original_results and not host.store.db.backend._original_result_holds
        assert len(peer.requests) == 1 and host.native.is_closed and len(host.closes) == 1
        assert provider.connects == driver.opens == 0


@pytest.mark.asyncio
async def test_host_finalization_retains_fact_after_original_grant_is_withdrawn(custody, vendor, peer, monkeypatch):
    async with original_host(custody, vendor, peer, monkeypatch) as host:
        async def withdraw(original):
            host.current['live'] = False
        error = ProviderAfterNativeError('original grant withdrawn after native join')
        provider = FailingAfterNativeProvider(host, error, after_native=withdraw)
        with pytest.raises(ProviderAfterNativeError) as raised:
            await runtime_for(host, provider, BoundedSessionDriver()).admit(host.context, host.request)
        assert raised.value is error and host.current['live'] is False
        # The fixture observes its own encrypted database, without supplying a
        # substitute current checker. No operation/parse permission is renewed.
        current = await host.store.get(SCOPE, request_id=host.request.request_id)
        checkpoint = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage).checkpoint
        assert checkpoint.state == 'unknown' and current.lease.provider_resource_id == 'owned-native-pod'
        assert current.lease.actual_cost_usd is None and current.acquisition.native_result is not None
        assert not host.store._original_results and not host.store.db.backend._original_result_holds
        assert len(peer.requests) == 1 and host.native.is_closed and len(host.closes) == 1


@pytest.mark.asyncio
async def test_host_finalization_retains_fact_on_actual_original_caller_withdrawal(custody, vendor, peer, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    observed = {}
    async def owner():
        async with original_host(custody, vendor, peer, monkeypatch) as host:
            async def after_native(original):
                observed['handle'] = original.handle
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError as error:
                    observed['original_cancel'] = error
                    raise
            provider = FailingAfterNativeProvider(host, ProviderAfterNativeError('unreached'), after_native=after_native)
            driver = BoundedSessionDriver()
            try:
                await runtime_for(host, provider, driver).admit(host.context, host.request)
            except BaseException as error:
                observed['returned_error'] = error
                observed['record'] = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
            observed['holds'] = (len(host.store._original_results), len(host.store.db.backend._original_result_holds))
            observed['opens'] = (provider.connects, driver.opens)
    original_task = asyncio.create_task(owner(), name='original-acquisition-caller')
    try:
        await asyncio.wait_for(entered.wait(), 5)
        original_task.cancel('original caller withdrew')
        await original_task
    finally:
        release.set()
        await asyncio.gather(original_task, return_exceptions=True)
    assert observed['returned_error'] is observed['original_cancel']
    current = observed['record']
    checkpoint = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage).checkpoint
    assert current.lease is not None and current.lease.provider_resource_id == 'owned-native-pod'
    assert checkpoint.state == 'unknown' and current.lease.actual_cost_usd is None
    assert observed['holds'] == observed['opens'] == (0, 0)
    handle = observed['handle']
    assert handle._state == 'closed' and handle._worker_exited and handle._admission.task.done()
    assert not handle._thread.sockets and len(peer.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['before-native-write', 'lost-native-ack', 'after-native-close'])
@pytest.mark.parametrize('control', [False, True])
async def test_host_finalization_keeps_original_cause_and_uncertainty_without_retry(custody, vendor, peer, monkeypatch, fault, control):
    async with original_host(custody, vendor, peer, monkeypatch) as host:
        provider_error = (asyncio.CancelledError('original provider withdrawal') if control
            else ProviderAfterNativeError('original provider error'))
        finalization_error = OSError('original finalization fault: '+fault)
        writes, actual = [], host.store.db.backend._settle_original_acquisition_result
        async def faulted(owner):
            if owner._entry.kind == 'minimal':
                writes.append(owner)
                if fault == 'before-native-write':
                    raise finalization_error
                await actual(owner)
                if fault == 'lost-native-ack':
                    raise finalization_error
                return
            return await actual(owner)
        monkeypatch.setattr(host.store.db.backend, '_settle_original_acquisition_result', faulted)
        async def close_fault(original):
            if fault == 'after-native-close':
                close = original.handle._close_original
                async def closed_then_failed():
                    await close()
                    raise finalization_error
                monkeypatch.setattr(original.handle, '_close_original', closed_then_failed)
        provider = FailingAfterNativeProvider(host, provider_error, after_native=close_fault)
        driver = BoundedSessionDriver()
        with pytest.raises(BaseException) as raised:
            await runtime_for(host, provider, driver).admit(host.context, host.request)
        chain = tuple(_explicit_control_chain(raised.value))
        assert any(error is provider_error for error in chain)
        assert any(error is finalization_error for error in chain)
        assert len(writes) == 1 and provider.modern_calls == len(peer.requests) == 1
        handle = provider.original.handle
        assert handle._state == 'closed' and handle._worker_exited and handle._admission.task.done()
        assert host.native.is_closed and len(host.closes) == 1 and not handle._thread.sockets
        assert not host.store._original_results and not host.store.db.backend._original_result_holds
        assert provider.connects == driver.opens == 0
        current = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        checkpoint = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage).checkpoint
        if fault == 'before-native-write':
            assert checkpoint.state == 'dispatching' and current.lease is None
            assert current.acquisition.native_result is None
        else:
            # A read observes the actual row; a lost outward ACK is NOT promoted
            # to a host-acknowledged transition, nor retried or replayed.
            assert checkpoint.state == 'unknown' and current.lease.provider_resource_id == 'owned-native-pod'
            assert current.lease.actual_cost_usd is None and current.acquisition.native_result is not None
        assert handle._minimal_acknowledged is (fault == 'after-native-close')
