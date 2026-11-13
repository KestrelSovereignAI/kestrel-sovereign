"""Actual optional canonical producer, native TLS and original encrypted row.

These synthetic peers never contact Runpod. A mandatory local proof refuses
an absent source-candidate vendor; ordinary Core has no vendor dependency.
"""
from datetime import UTC, datetime, timedelta
from contextlib import asynccontextmanager
from decimal import Decimal
from http.server import BaseHTTPRequestHandler
import asyncio
import copy
import json
import os
import queue
import ssl
import sys
import threading
import time
from types import SimpleNamespace

import httpx
from httpcore._backends.sync import SyncStream
import pytest
from pydantic import SecretBytes
from kestrel_sdk.computer_use import EnvironmentCapabilities, EnvironmentQuote, EnvironmentRequest
from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionPreparation
from kestrel_sdk.operator import OperatorContext
from kestrel_sdk.resources import ResourceUnsupported
from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, bind_execution_custody_snapshot,
    execution_evidence_carriers, execution_terminal_error, is_execution_control_error,
)
from kestrel_sovereign.execution_sync_http import (
    await_native_sync_http, bind_native_sync_http_execution_admission,
)
from kestrel_sovereign.privacy import PrivacyConfig, PrivacyMode
from kestrel_sovereign.security.capability_authority import HostCapabilityAuthority
from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate, _SharedGateHold, _SharedGateLease
from kestrel_sovereign.visual_computer.repository import CustodyRecord
from kestrel_sovereign.visual_computer.runpod_acquisition import prepare_runpod_acquisition_producer
from kestrel_sovereign.visual_computer.runtime import VisualAccessDenied, VisualAction
from tests.integration.test_visual_custody_backends import custody as custody, SCOPE
from tests.integration.test_visual_original_result_backends import OriginalTurn
from tests.utils.tls import make_tls_contexts, tls_http_server


@pytest.fixture
def vendor():
    try:
        import kestrel_cloud_runpod.clients as clients
        import kestrel_cloud_runpod.cpu_environment as environment
        from kestrel_cloud_runpod.cpu_catalog import CpuPodCatalogMatch
        assert callable(getattr(clients.RunpodControlPlaneClient, 'prepare_cpu_acquisition_producer', None))
    except (ImportError, AssertionError):
        if os.environ.get('KESTREL_REQUIRE_RUNPOD_ACQUISITION_TESTS') == '1':
            pytest.fail('mandatory canonical Runpod acquisition producer is unavailable')
        pytest.skip('optional canonical Runpod acquisition producer is not installed')
    return SimpleNamespace(clients=clients, environment=environment, match=CpuPodCatalogMatch)


def original_plan(vendor, *, quote_seconds=30):
    now = datetime.now(UTC)
    profile = vendor.environment.CpuEnvironmentProfile(provider_id='runpod-cpu', profile_id='cpu-browser',
        image='registry.example/browser@sha256:'+'a'*64, cpu_id='cpu5c', vcpu_count=4,
        data_center_id='US-TX-3', disk_gb=20, max_lifetime_seconds=3600, cleanup_grace_seconds=60)
    request = EnvironmentRequest(request_id='original', scope=SCOPE, provider_id=profile.provider_id,
        profile_id=profile.profile_id, image=profile.image, surface='browser', location='remote',
        region=profile.data_center_id, required_capabilities=frozenset({'screenshot', 'click'}),
        requested_at=now, expires_at=now+timedelta(minutes=30), idle_ttl_seconds=60,
        ready_timeout_seconds=60, max_hourly_cost_usd=Decimal('0.20'), max_total_cost_usd=Decimal('0.11'))
    policy = vendor.environment.CpuNonComputeCostPolicy(policy_id='explicit-estimate', provider_id=profile.provider_id,
        profile_id=profile.profile_id, profile_sha256=profile.digest(), disk_gb=20, data_center_id=profile.data_center_id,
        covered_components=frozenset({'container_disk', 'transfer', 'cleanup'}),
        observed_at=now-timedelta(minutes=1), expires_at=now+timedelta(hours=2),
        disk_transfer_cleanup_hourly_usd=Decimal('0.02'), disk_transfer_cleanup_fixed_usd=Decimal('0.01'))
    match = vendor.match('cpu5c', 4, ('US-TX-3',), Decimal('10'), Decimal('0.16'), now)
    return vendor.environment.CpuEnvironmentAcquisitionPlan(request, profile, policy, match, now, now+timedelta(seconds=quote_seconds))


@pytest.fixture
def peer(tmp_path):
    server_context, _, certificate = make_tls_contexts(tmp_path)
    state = SimpleNamespace(requests=[], snis=[], reply=None, before_reply=None, resets=[], raw=None, encoding=[], chunked=False, declared_length=None, raw_lines=[])
    server_context.set_servername_callback(lambda sock, name, context: state.snis.append(name))
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError) as error:
                state.resets.append(type(error).__name__)
        def parse_request(self):
            state.raw_lines.append(self.raw_requestline)
            return super().parse_request()
        def do_POST(self):
            raw = self.rfile.read(int(self.headers['Content-Length']))
            state.requests.append((self.path, raw, dict(self.headers), self.connection.version()))
            if state.before_reply is not None:
                state.before_reply()
            content = json.dumps(state.reply).encode() if state.raw is None else state.raw
            self.send_response(201)
            self.send_header('Content-Type', 'application/json')
            for encoding in state.encoding:
                self.send_header('Content-Encoding', encoding)
            if state.chunked:
                self.send_header('Transfer-Encoding', 'chunked')
            else:
                self.send_header('Content-Length', str(len(content) if state.declared_length is None else state.declared_length))
            self.end_headers()
            if state.chunked:
                for offset in range(0, len(content), 8192):
                    chunk = content[offset:offset+8192]
                    self.wfile.write(f'{len(chunk):x}\r\n'.encode()+chunk+b'\r\n')
                self.wfile.write(b'0\r\n\r\n')
            else:
                self.wfile.write(content)
        def log_message(self, *args):
            pass
    with tls_http_server(server_context, Handler) as server:
        state.url = f'https://127.0.0.1:{server.server_port}/v2'
        state.ca = certificate.read_bytes()
        yield state


@asynccontextmanager
async def original_host(custody, vendor, peer, monkeypatch, *, quote_seconds=30, operation_seconds=120):
    store, _ = custody
    plan = original_plan(vendor, quote_seconds=quote_seconds)
    request = plan.request
    peer.reply = dict(id='owned-native-pod', name='kestrel-cpu-'+request.digest()[:32], status='RUNNING',
        image=plan.profile.image, dataCenterId='US-TX-3', disk=20,
        cpu=dict(id='cpu5c', vcpuCount=4, memory=10), cost=0.16)
    authority = HostCapabilityAuthority(capability='visual-computer', label='visual computer', access_denied=VisualAccessDenied)
    holder = SimpleNamespace(privacy_mode=PrivacyMode.NORMAL, privacy_config=PrivacyConfig(),
        _get_durable_persistence_gate=lambda: privacy)
    privacy = DurablePersistenceGate()
    now = datetime.now(UTC)
    context = OperatorContext(tenant_id=SCOPE.tenant_id, principal_id=SCOPE.principal_id, acting_agent_id=SCOPE.agent_id,
        correlation_id='owned-native-proof', issued_at=now, expires_at=now+timedelta(seconds=operation_seconds),
        granted_capabilities={'visual-computer'}, permitted_boundary_ids={store.storage_id}, granted_actions={VisualAction.ADMIT.value})
    current = {'live': True, 'checks': 0}
    def host_current(ctx, action, target):
        assert ctx is context and action == VisualAction.ADMIT
        current['checks'] += 1
        if not current['live']:
            raise PermissionError('original host grant withdrawn')
    control = vendor.clients.RunpodControlPlaneClient(api_key='owned-native-api-key', base_url=peer.url,
        prepared_cpu_account='owned-native-account', prepared_cpu_ca_data=peer.ca,
        prepared_cpu_acquisition=True, connect_timeout=2, read_timeout=4)
    native = control.transport._client
    closes = []
    original_close = httpx.Client.close
    def close(client):
        if client is native:
            closes.append(client)
        return original_close(client)
    monkeypatch.setattr(httpx.Client, 'close', close)
    turn = ExecutionCustody(OriginalTurn(store.db.backend_type))
    try:
        # Retain the original actual backend; never synthesize PG admission
        # for the standalone SQLite half of this native contract fixture.
        with bind_execution_custody_snapshot((turn,)):
            with authority.bind_authenticated(context, privacy_holder=holder, check_current=host_current,
                    authority_gate=AsyncReaderWriterLock()):
                async with authority.operation(context, VisualAction.ADMIT, store.storage_id) as check:
                    yield SimpleNamespace(store=store, plan=plan, request=request, native=native,
                        control=control, check=check, turn=turn, current=current, closes=closes,
                        context=context, authority=authority)
    finally:
        if not native.is_closed:
            control.close()
    assert len(closes) == 1


def original_quote(plan):
    request = plan.request
    return EnvironmentQuote(quote_id='original-quote', request_id=request.request_id,
        request_sha256=request.digest(), scope=SCOPE, provider_id=request.provider_id, location='remote',
        region=request.region, expires_at=plan.expires_at, hourly_cost_usd=plan.all_in_hourly_usd,
        estimated_total_cost_usd=plan.estimated_total_usd,
        capabilities=EnvironmentCapabilities(provider_id=request.provider_id, location='remote',
            surfaces=frozenset({'browser'}), operations=request.required_capabilities))


@asynccontextmanager
async def original_acquisition(custody, vendor, peer, monkeypatch, *, quote_seconds=30):
    async with original_host(custody, vendor, peer, monkeypatch, quote_seconds=quote_seconds) as host:
        store, request, plan, control, check = host.store, host.request, host.plan, host.control, host.check
        value = await store.admit(SCOPE, CustodyRecord(request=request, session_id='original-session'), check)
        value.quote, value.phase = original_quote(plan), 'quoted'
        await store.save(SCOPE, value, check)
        dto = plan.pod_request(request, plan.profile, plan.noncompute, at=plan.planned_at)
        prepared = control.prepare_cpu_pod(dto, plan=plan)
        preparation = EnvironmentAcquisitionPreparation(create_payload=SecretBytes(prepared.exact_body()),
            worker_identity=SecretBytes(b'PRIVATE-ORIGINAL-WORKER-TLS-IDENTITY'))
        async with store.original_acquisition_result(SCOPE, value, request, value.quote, guard=check) as attempt:
            producer = await prepare_runpod_acquisition_producer(attempt, control, dto,
                prepared=prepared, plan=plan, preparation=preparation)
            def spending_current():
                control._prepared_body(dto, prepared, plan)
            yield SimpleNamespace(**vars(host), attempt=attempt, producer=producer,
                handle=producer._original_acquisition, prepared=prepared, spending=spending_current)


async def invoke(original, *, check=None):
    if check is None:
        return await original.handle.run_original()
    with bind_native_sync_http_execution_admission(original.native,
            original.check if check is None else check, check_before_request_write=original.spending):
        return await await_native_sync_http(original.native, original.producer.operation,
            _original_acquisition=original.handle)


@pytest.mark.asyncio
@pytest.mark.parametrize('cost', ['valid', 'missing', 'malformed', 'huge'])
async def test_native_original_producer_minimal_ack_precedes_optional_cost(custody, vendor, peer, monkeypatch, cost):
    # The context manager runs on THIS exact test task. An async fixture yield
    # would move issuance and use across tasks and could not supply authority.
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        if cost in {'malformed', 'huge'}:
            peer.reply['cost'] = 'PRIVATE-INVALID-COST'+('9'*200000 if cost == 'huge' else '')
        elif cost == 'missing':
            del peer.reply['cost']
        with pytest.raises(ExecutionAuthorityError):
            original.producer.operation(original.native)
        assert not peer.requests and not peer.snis
        assert await original.attempt.begin_dispatch() is True
        assert await original.attempt.begin_dispatch() is False
        assert original.store.db.owns_open_transaction is False
        phase = await invoke(original)
        assert original.native.is_closed and len(original.closes) == 1
        with pytest.raises(ExecutionAuthorityError):
            original.producer.finish_optional_cost(phase)
        checkpoint = await original.handle.persist_minimal()
        assert checkpoint.state == 'unknown' and checkpoint.lease.provider_resource_id == 'owned-native-pod'
        assert checkpoint.lease.actual_cost_usd is None
        durable = await original.store.get(SCOPE, request_id=original.request.request_id, guard=original.check)
        assert durable.lease == checkpoint.lease and durable.acquisition.native_result is not None
        if cost in {'malformed', 'huge'}:
            from kestrel_cloud_runpod.models import RunPodAmbiguousResultError
            with pytest.raises(RunPodAmbiguousResultError):
                original.producer.finish_optional_cost(phase)
        else:
            public = original.producer.finish_optional_cost(phase)
            assert public.id == 'owned-native-pod'
            if cost == 'missing':
                assert public.cost_per_hr is None
            else:
                assert type(public.cost_per_hr) is Decimal
                assert public.cost_per_hr == Decimal('0.16')
            accepted = await original.attempt.record_accepted(checkpoint.lease.model_copy(update={'state': 'pending'}))
            assert accepted.state == 'accepted' and accepted.lease.actual_cost_usd is None
        assert original.current['checks'] > 1
        saved = original.prepared.exact_body()
    assert len(peer.requests) == 1 and peer.requests[0][0] == '/v2/pods'
    assert peer.requests[0][1] == saved
    assert peer.requests[0][2]['Accept-Encoding'] == 'identity' and peer.requests[0][3] == 'TLSv1.3'


def _withdraw_original_hold_marker(root, restore):
    """Retire only the original frame marker while retaining its read lease."""
    descriptor = root._native_backing
    hold, gate, lease = descriptor.privacy_holds[0]
    assert type(hold) is _SharedGateHold and type(gate) is DurablePersistenceGate
    assert type(lease) is _SharedGateLease
    assert hold.gate is gate and hold.lease is lease
    assert hold.task is descriptor.owner_task is root._owner_task
    frames, holds = frozenset(gate._shared_frames), lease.holds
    assert hold in frames and hold.live is True and type(holds) is int and holds > 0

    def reset():
        assert hold.gate is gate and hold.lease is lease
        assert hold.task is descriptor.owner_task is root._owner_task
        assert lease.holds == holds and hold not in gate._shared_frames
        gate._shared_frames.add(hold)
        assert hold.live is True and gate._shared_frames == frames

    restore.append(reset)
    gate._shared_frames.remove(hold)
    assert hold.live is False and lease.holds == holds
    assert gate._shared_frames == frames-{hold}


def _inject_after_tls(start_tls, inject):
    """Own the exact TLS return until the controlled injection hands it back."""
    def after_tls(stream, *args, **kwargs):
        actual = start_tls(stream, *args, **kwargs)
        try:
            inject(actual)
            return actual
        except BaseException as error:
            try:
                actual.close()
            except BaseException as cleanup:
                raise execution_terminal_error(error, cleanup, retain_originals=True)
            raise
    return after_tls


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['copy', 'generation', 'backing', 'hold-marker', 'global-stop'])
async def test_original_worker_refuses_changed_captured_lifetime_after_tls(custody, vendor, peer, monkeypatch, fault):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        root = original.attempt._result
        await original.attempt.begin_dispatch()
        start_tls = SyncStream.start_tls
        restore = []
        entered = []
        def inject(actual):
            entered.append(threading.current_thread())
            if fault == 'copy':
                value = root._native_backing
                restore.append(lambda: setattr(root, '_native_backing', value))
                root._native_backing = copy.copy(value)
            elif fault == 'generation':
                target = root._backend if root._generation is not None else root._native_backing.postgres_pool.pool
                name = '_connection_generation' if root._generation is not None else '_generation'
                value = getattr(target, name)
                restore.append(lambda: setattr(target, name, value))
                setattr(target, name, 'changed' if type(value) is str else value+1)
            elif fault == 'backing':
                target = root._backend
                name = '_connection' if root._generation is not None else '_pool'
                value = getattr(target, name)
                restore.append(lambda: setattr(target, name, value))
                setattr(target, name, object())
            elif fault == 'hold-marker':
                _withdraw_original_hold_marker(root, restore)
            else:
                original.turn.revoke('global Stop after genuine TLS and before application write')
        monkeypatch.setattr(SyncStream, 'start_tls', _inject_after_tls(start_tls, inject))
        try:
            with pytest.raises(BaseException) as denied:
                await invoke(original, check=lambda: None)
            assert is_execution_control_error(denied.value)
            assert len(entered) == 1 and not peer.requests
        finally:
            for reset in reversed(restore):
                reset()
        assert original.native.is_closed and original.handle._receipt.phase is None
        if fault != 'global-stop':
            # A changed original physical descriptor/hold cannot issue a
            # joined seal. Resetting a controlled fault is cleanup, not proof.
            with pytest.raises(ExecutionAuthorityError):
                await original.handle.persist_minimal()
            assert original.attempt.checkpoint.state == 'dispatching'
            return
        checkpoint = await original.handle.persist_minimal()
        assert checkpoint.state == 'unknown' and checkpoint.lease is None
        if fault == 'global-stop':
            with pytest.raises(Exception) as ordinary:
                await original.store.db.fetchone('SELECT 1')
            assert is_execution_control_error(ordinary.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('lose_close_ack', [False, True])
async def test_failed_tls_injection_retires_exact_return_and_retains_errors(
    custody, vendor, peer, monkeypatch, lose_close_ack,
):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        await original.attempt.begin_dispatch()
        start_tls, close = SyncStream.start_tls, SyncStream.close
        primary = RuntimeError('owned after-TLS injection failure')
        cleanup = OSError('owned after-TLS close acknowledgement loss')
        returns, closes = [], []

        def inject(actual):
            returns.append(actual)
            raise primary

        def close_original(stream):
            close(stream)
            if returns and stream is returns[0]:
                closes.append(stream)
                if lose_close_ack:
                    raise cleanup

        monkeypatch.setattr(SyncStream, 'close', close_original)
        monkeypatch.setattr(SyncStream, 'start_tls', _inject_after_tls(start_tls, inject))
        try:
            with pytest.raises(BaseException) as failure:
                await invoke(original)
            pending, found = [failure.value], set()
            while pending:
                error = pending.pop()
                if id(error) in found:
                    continue
                found.add(id(error))
                pending.extend(value for value in (error.__cause__, error.__context__) if value is not None)
                pending.extend(getattr(error, 'exceptions', ()))
            assert id(primary) in found
            assert (id(cleanup) in found) is lose_close_ack
            if lose_close_ack:
                assert failure.value is not cleanup, 'TLS close failure replaced the injection failure'
            actual, = returns
            assert closes == [actual] and actual._sock.fileno() == -1
            assert original.native.is_closed and len(original.closes) == 1
            assert not peer.requests and original.handle._receipt.phase is None
            checkpoint = await original.handle.persist_minimal()
            assert checkpoint.state == 'unknown' and checkpoint.lease is None
        finally:
            # Guard mutations must fail before this cleanup, while the same
            # real TLS peer still gets its original socket/thread retirement.
            for actual in returns:
                close(actual)


@pytest.mark.asyncio
@pytest.mark.parametrize('inject_failure', [False, True], ids=['foreign-thread', 'injection-failure'])
async def test_running_original_worker_lifetime_cannot_be_borrowed_by_foreign_thread_or_stage(
    custody, vendor, peer, monkeypatch, inject_failure,
):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        root, handle = original.attempt._result, original.handle
        with pytest.raises(ExecutionAuthorityError):
            root._require_native_producer_lifetime(handle)
        await original.attempt.begin_dispatch()
        start_tls, close, refused = SyncStream.start_tls, SyncStream.close, queue.Queue()
        primary = RuntimeError('owned foreign-thread test injection failure')
        returns = []
        def inject(actual):
            returns.append(actual)
            if inject_failure:
                raise primary
            root._require_native_producer_lifetime(handle)  # THIS genuine worker.
            def other():
                try:
                    root._require_native_producer_lifetime(handle)
                except BaseException as error:
                    refused.put(error)
            foreign = threading.Thread(target=other, name='owned-foreign-negative', daemon=False)
            foreign.start()
            foreign.join(2)
            assert not foreign.is_alive()
        monkeypatch.setattr(SyncStream, 'start_tls', _inject_after_tls(start_tls, inject))
        try:
            if inject_failure:
                with pytest.raises(BaseException) as failure:
                    await invoke(original)
                assert any(error is primary for error in execution_evidence_carriers(failure.value))
                actual, = returns
                assert original.native.is_closed and len(original.closes) == 1
                assert not peer.requests and original.handle._receipt.phase is None
                assert actual._sock.fileno() == -1, 'after-TLS injection lost its exact original stream'
                checkpoint = await handle.persist_minimal()
                assert checkpoint.state == 'unknown' and checkpoint.lease is None
                return
            await invoke(original)
            assert type(refused.get_nowait()) is ExecutionAuthorityError
            with pytest.raises(ExecutionAuthorityError):
                root._require_native_producer_lifetime(handle)
            assert (await handle.persist_minimal()).lease.provider_resource_id == 'owned-native-pod'
        finally:
            # A wiring inverse must fail before this cleanup; the exact real
            # peer socket still retires before its original fixture joins.
            for actual in returns:
                close(actual)
    assert len(peer.requests) == 1


@pytest.mark.asyncio
async def test_changed_decoder_after_genuine_native_return_keeps_unknown_no_id(custody, vendor, peer, monkeypatch):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        await original.attempt.begin_dispatch()
        original_json, start_tls = httpx.Response.json, SyncStream.start_tls
        request_code = httpx.Client.request.__code__
        substituted = []
        def ghost(response, **kwargs):
            value = original_json(response, **kwargs)
            value['id'] = 'FORGED-GHOST-ID'
            return value
        def after_tls(stream, *args, **kwargs):
            actual = start_tls(stream, *args, **kwargs)
            previous = sys.getprofile()
            def returned(frame, event, value):
                if event == 'return' and frame.f_code is request_code and type(value) is httpx.Response:
                    sys.setprofile(previous)
                    substituted.append(value)
                    monkeypatch.setattr(httpx.Response, 'json', ghost)
            sys.setprofile(returned)
            return actual
        monkeypatch.setattr(SyncStream, 'start_tls', after_tls)
        with pytest.raises(BaseException) as failure:
            await invoke(original)
        assert is_execution_control_error(failure.value)
        assert len(substituted) == len(peer.requests) == 1
        assert original.native.is_closed and original.handle._receipt.phase is None
        before = await original.store.get(SCOPE, request_id=original.request.request_id, guard=original.check)
        assert before.acquisition is not None and before.lease is None
        assert original.attempt.checkpoint.state == 'dispatching'
        # Leave the substitution in place through the genuine joined fact
        # write and root closure. Restoring it here would hide a custody gap.
        checkpoint = await original.handle.persist_minimal()
        assert checkpoint.state == 'unknown' and checkpoint.lease is None
        after = await original.store.get(SCOPE, request_id=original.request.request_id, guard=original.check)
        assert after.acquisition.native_result is None and after.lease is None
        assert after.revision == before.revision+1


@pytest.mark.asyncio
async def test_original_complete_post_may_capture_reply_after_spending_quote_expiry(custody, vendor, peer, monkeypatch):
    async with original_acquisition(custody, vendor, peer, monkeypatch, quote_seconds=2) as original:
        replied = []
        def after_quote():
            remaining = (original.plan.expires_at-datetime.now(UTC)).total_seconds()
            if remaining > 0:
                time.sleep(remaining+0.05)
            replied.append(datetime.now(UTC))
        peer.before_reply = after_quote
        await original.attempt.begin_dispatch()
        phase = await invoke(original)
        assert replied[0] > original.plan.expires_at
        assert phase.id == 'owned-native-pod'
        minimal = await original.handle.persist_minimal()
        assert minimal.lease.provider_resource_id == 'owned-native-pod'
        public = original.producer.finish_optional_cost(phase)
        assert public.id == phase.id
    assert len(peer.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('decoder', ['response', 'loads'])
async def test_substituted_canonical_decoder_before_issuer_refuses_without_network(custody, vendor, peer, monkeypatch, decoder):
    if decoder == 'response':
        original = httpx.Response.json
        def ghost(response, **kwargs):
            value = original(response, **kwargs)
            value['id'] = 'FORGED-GHOST-ID'
            return value
        monkeypatch.setattr(httpx.Response, 'json', ghost)
    else:
        original = json.loads
        def substitute(*args, **kwargs):
            return original(*args, **kwargs)
        monkeypatch.setattr(json, 'loads', substitute)
    with pytest.raises((ExecutionAuthorityError, ResourceUnsupported)):
        async with original_acquisition(custody, vendor, peer, monkeypatch) as admitted:
            await admitted.attempt.begin_dispatch()
            await invoke(admitted)
            checkpoint = await admitted.handle.persist_minimal()
            # A removed decoder guard must expose the actual forged ID after
            # genuine TLS and same-row ACK, rather than merely miss a raise.
            assert checkpoint.lease.provider_resource_id == 'owned-native-pod'
            pytest.fail('substituted decoder reached original producer')
    assert not peer.requests and not peer.snis


@pytest.mark.asyncio
@pytest.mark.parametrize('when', ['before-capture', 'after-capture'])
@pytest.mark.parametrize('cancel', [False, True], ids=['no-cancel', 'cancel'])
async def test_original_turn_stop_retains_only_already_captured_native_fact(custody, vendor, peer, monkeypatch, when, cancel):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        await original.attempt.begin_dispatch()
        owner, loop = asyncio.current_task(), asyncio.get_running_loop()
        events = []
        def revoke():
            original.turn.revoke('original turn Stop at '+when)
            events.append(when)
            if cancel:
                loop.call_soon_threadsafe(owner.cancel)
        if when == 'before-capture':
            start_tls, request_code = SyncStream.start_tls, httpx.Client.request.__code__
            def after_tls(stream, *args, **kwargs):
                actual = start_tls(stream, *args, **kwargs)
                previous = sys.getprofile()
                def returned(frame, event, value):
                    if event == 'return' and frame.f_code is request_code and type(value) is httpx.Response:
                        sys.setprofile(previous)
                        revoke()
                sys.setprofile(returned)
                return actual
            monkeypatch.setattr(SyncStream, 'start_tls', after_tls)
        else:
            actual_close = httpx.Client.close
            def close(client):
                if client is original.native:
                    assert original.handle._phase is original.producer._phase
                    assert original.handle._phase is not None
                    revoke()
                return actual_close(client)
            monkeypatch.setattr(httpx.Client, 'close', close)
        with pytest.raises(BaseException) as denied:
            await invoke(original)
        assert isinstance(denied.value, asyncio.CancelledError) or is_execution_control_error(denied.value)
        assert events == [when] and len(peer.requests) == 1
        assert original.native.is_closed and len(original.closes) == 1
        with pytest.raises(Exception) as ordinary:
            await original.store.db.fetchone('SELECT 1')
        assert is_execution_control_error(ordinary.value)
        checkpoint = await original.handle.persist_minimal()
        assert checkpoint.state == 'unknown'
        if when == 'before-capture':
            assert checkpoint.lease is None and original.handle._receipt.phase is None
        else:
            assert checkpoint.lease.provider_resource_id == 'owned-native-pod'
            assert checkpoint.lease.actual_cost_usd is None
            assert original.handle._receipt.native_result is not None
        with pytest.raises(BaseException) as parse:
            original.producer.finish_optional_cost(original.handle._phase)
        assert is_execution_control_error(parse.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['receipt', 'row', 'key', 'hold-marker', 'generation'])
async def test_joined_native_fact_refuses_changed_original_custody(custody, vendor, peer, monkeypatch, fault):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        await original.attempt.begin_dispatch()
        await invoke(original)
        root, handle = original.attempt._result, original.handle
        restore = []
        if fault == 'receipt':
            target, name, changed = handle, '_receipt', copy.copy(handle._receipt)
        elif fault == 'row':
            target, name, changed = root, '_identity', ('foreign',)+root._identity[1:]
        elif fault == 'key':
            target, name, changed = original.store, '_key', b'X'*32
        elif fault == 'hold-marker':
            _withdraw_original_hold_marker(root, restore)
        else:
            target = root._backend if root._generation is not None else root._native_backing.postgres_pool.pool
            name = '_connection_generation' if root._generation is not None else '_generation'
            value = getattr(target, name)
            changed = 'changed' if type(value) is str else value+1
        if fault != 'hold-marker':
            value = getattr(target, name)
            restore.append(lambda: setattr(target, name, value))
            setattr(target, name, changed)
        try:
            with pytest.raises(ExecutionAuthorityError):
                await handle.persist_minimal()
        finally:
            for reset in reversed(restore):
                reset()  # Exact owned fixture cleanup only.
        assert original.attempt.checkpoint.state == 'dispatching'
        assert len(peer.requests) == 1


@pytest.mark.asyncio
async def test_valid_original_phase_survives_decoder_retirement_without_new_parse(custody, vendor, peer, monkeypatch):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        await original.attempt.begin_dispatch()
        phase = await invoke(original)
        def forbidden(*args, **kwargs):
            pytest.fail('retired decoder was invoked after original capture')
        monkeypatch.setattr(httpx.Response, 'json', forbidden)
        checkpoint = await original.handle.persist_minimal()
        assert checkpoint.lease.provider_resource_id == 'owned-native-pod'
        with pytest.raises(ExecutionAuthorityError):
            original.producer.finish_optional_cost(phase)
        assert len(peer.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['gzip', 'multiple', 'length', 'chunked', 'exact', 'identity'])
async def test_native_acquisition_bounds_raw_response_before_decoder(custody, vendor, peer, monkeypatch, boundary):
    from kestrel_sovereign.visual_computer.acquisition_native import MAX_ACQUISITION_RESPONSE_BYTES as maximum
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        raw = json.dumps(peer.reply).encode()
        peer.raw = raw+b' '*(maximum-len(raw)+(boundary in {'length', 'chunked'}))
        if boundary == 'gzip':
            peer.encoding = ['gzip']
        elif boundary == 'identity':
            peer.encoding = ['identity']
        elif boundary == 'multiple':
            peer.encoding = ['identity', 'identity']
        elif boundary == 'chunked':
            peer.chunked = True
        decoder_calls = []
        start_tls, decoder_code = SyncStream.start_tls, httpx.Response._get_content_decoder.__code__
        def after_tls(stream, *args, **kwargs):
            actual = start_tls(stream, *args, **kwargs)
            previous = sys.getprofile()
            def observe(frame, event, value):
                if frame.f_code is decoder_code and event == 'call':
                    decoder_calls.append(frame.f_locals['self'])
                if frame.f_code is httpx.Client.request.__code__ and event == 'return':
                    sys.setprofile(previous)
            sys.setprofile(observe)
            return actual
        monkeypatch.setattr(SyncStream, 'start_tls', after_tls)
        await original.attempt.begin_dispatch()
        if boundary in {'exact', 'identity'}:
            phase = await invoke(original)
            assert phase.id == 'owned-native-pod'
            assert original.handle._receipt.raw_bytes == maximum
            checkpoint = await original.handle.persist_minimal()
            assert checkpoint.lease.provider_resource_id == phase.id
        else:
            with pytest.raises(BaseException) as denied:
                await invoke(original)
            assert is_execution_control_error(denied.value)
            assert original.handle._receipt.phase is None
            checkpoint = await original.handle.persist_minimal()
            assert checkpoint.lease is None
            if boundary != 'chunked':
                assert not decoder_calls
                assert original.handle._receipt.raw_bytes == 0
            else:
                assert decoder_calls and original.handle._receipt.raw_bytes <= maximum
        assert original.native.is_closed and len(peer.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('retirement', ['turn', 'host'])
async def test_seventeen_actual_tls_request_bytes_remain_unknown_no_id(custody, vendor, peer, monkeypatch, retirement):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        await original.attempt.begin_dispatch()
        native_send, written, clients = ssl.SSLSocket.send, [], []
        def partial(sock, data, flags=0):
            if bytes(data[:5]) == b'POST ' and not clients:
                clients.append(sock)
                count = native_send(sock, data[:17], flags)
                written.append(bytes(data[:count]))
                if retirement == 'turn':
                    original.turn.revoke('original global Stop after 17 actual TLS application bytes')
                else:
                    original.current['live'] = False
                return count
            count = native_send(sock, data, flags)
            if clients and sock is clients[0]:
                written.append(bytes(data[:count]))
            return count
        monkeypatch.setattr(ssl.SSLSocket, 'send', partial)
        with pytest.raises(BaseException) as denied:
            await original.handle.run_original()
        if retirement == 'turn':
            assert is_execution_control_error(denied.value)
        assert len(written) == 1 and len(written[0]) == 17
        assert original.handle._thread.sent and original.handle._receipt.phase is None
        checkpoint = await original.handle.persist_minimal()
        assert checkpoint.state == 'unknown' and checkpoint.lease is None
        assert not peer.requests and original.native.is_closed
        # A successful TLS send can remain unobserved by a peer after abrupt
        # close. It is still possible effect, never proof of remote absence.


@pytest.mark.asyncio
async def test_lost_original_dispatch_ack_never_arms_or_retries_native_create(custody, vendor, peer, monkeypatch):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
    from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionStorage
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        backend = original.store.db.backend
        actual = backend._settle_original_acquisition_result
        async def lost_ack(result):
            dispatch = result._entry.kind == 'dispatch'
            await actual(result)
            if dispatch:
                raise RuntimeError('owned lost dispatch COMMIT ACK publication')
        with monkeypatch.context() as patch:
            patch.setattr(backend, '_settle_original_acquisition_result', lost_ack)
            with pytest.raises(BaseException):
                await original.attempt.begin_dispatch()
        assert original.handle._dispatch_ack is None and original.handle._state == 'prepared'
        with pytest.raises(ExecutionAuthorityError):
            await invoke(original)
        current = await original.store.get(SCOPE, request_id=original.request.request_id, guard=original.check)
        saved = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage)
        assert current.phase == 'acquiring' and saved.checkpoint.state == 'dispatching'
        assert saved.checkpoint.lease is None
        _, reopen = custody
        db = await reopen()
        try:
            restarted = VisualCustodyRepository(db)
            current = await restarted.get(SCOPE, request_id=original.request.request_id, guard=original.check)
            async with restarted.original_acquisition_result(SCOPE, current, current.request, current.quote,
                    guard=original.check) as attempt:
                assert attempt.checkpoint.attempt_id == saved.checkpoint.attempt_id
                with pytest.raises(ResourceOutcomeUnknown, match='reconciliation only'):
                    await prepare_runpod_acquisition_producer(attempt, original.control,
                        original.producer._request, prepared=original.prepared, plan=original.plan,
                        preparation=saved.preparation)
        finally:
            await db.close()
        assert not peer.requests and not peer.snis


@pytest.mark.asyncio
async def test_expired_spending_quote_never_gets_dispatch_ack_or_native_io(custody, vendor, peer, monkeypatch):
    from kestrel_sdk.resources import ResourceConflict
    async with original_acquisition(custody, vendor, peer, monkeypatch, quote_seconds=2) as original:
        remaining = (original.plan.expires_at-datetime.now(UTC)).total_seconds()
        if remaining > 0:
            await asyncio.sleep(remaining+0.05)
        with pytest.raises(ResourceConflict, match='spending quote expired'):
            await original.attempt.begin_dispatch()
        assert original.handle._dispatch_ack is None and original.handle._state == 'prepared'
        assert original.attempt.checkpoint.state == 'prepared'
        assert not peer.requests and not peer.snis


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['unarmed', 'copy', 'foreign-task', 'preparation', 'host-retired', 'turn-retired', 'async', 'retained-async', 'dispatch-ack-copy'])
async def test_closed_original_invocation_refuses_without_borrowing_another_checker(custody, vendor, peer, monkeypatch, fault):
    from contextlib import nullcontext
    from kestrel_sovereign.execution_custody import (
        bind_native_execution_admission, bind_native_execution_admission_snapshot,
        capture_native_execution_admission,
    )
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        if fault != 'unarmed':
            await original.attempt.begin_dispatch()
        target = original.handle
        checkpoint_before = original.attempt.checkpoint
        ciphertext_before, revision_before = target._root._expected, target._root._revision
        if fault == 'copy':
            target = copy.copy(target)
        elif fault == 'dispatch-ack-copy':
            target._dispatch_ack = copy.copy(target._dispatch_ack)
        elif fault == 'preparation':
            object.__setattr__(original.prepared, 'body', SecretBytes(b'{"original-body":"changed"}'))
        elif fault == 'host-retired':
            original.current['live'] = False
        elif fault == 'turn-retired':
            original.turn.revoke('original turn retired before closed native invocation')
        async def forbidden_validator():
            pytest.fail('Original async validator must not be sampled or bridged')
        gate = nullcontext()
        if fault == 'async':
            gate = bind_native_execution_admission(forbidden_validator)
        elif fault == 'retained-async':
            with bind_native_execution_admission(forbidden_validator):
                captured = capture_native_execution_admission()
            gate = bind_native_execution_admission_snapshot(captured)
        with gate:
            with pytest.raises((ExecutionAuthorityError, ResourceUnsupported, PermissionError)):
                if fault == 'foreign-task':
                    task = asyncio.create_task(target.run_original())
                    try:
                        await task
                    finally:
                        assert task.done()
                else:
                    await target.run_original()
        assert getattr(original.native, '_kestrel_sync_admission', None) is None
        assert not peer.requests and not peer.snis
        assert original.handle._worker is None and original.handle._request is None
        assert checkpoint_before.state == ('prepared' if fault == 'unarmed' else 'dispatching')
        assert original.handle._root._expected is ciphertext_before
        assert original.handle._root._revision == revision_before


@pytest.mark.asyncio
async def test_closed_original_invocation_cannot_reenter_after_original_scope_closes(custody, vendor, peer, monkeypatch):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        await original.attempt.begin_dispatch()
    with pytest.raises(ExecutionAuthorityError):
        await original.handle.run_original()
    assert original.native.is_closed and original.handle._state == 'closed'
    assert not peer.requests and not peer.snis
    assert original.handle._worker is None


@pytest.mark.asyncio
async def test_closed_original_host_retires_after_tls_before_first_application_byte(custody, vendor, peer, monkeypatch):
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        await original.attempt.begin_dispatch()
        start_tls = SyncStream.start_tls
        def after_tls(stream, *args, **kwargs):
            actual = start_tls(stream, *args, **kwargs)
            original.current['live'] = False
            return actual
        monkeypatch.setattr(SyncStream, 'start_tls', after_tls)
        with pytest.raises(BaseException):
            await original.handle.run_original()
        assert len(peer.snis) == 1 and not peer.requests
        assert not original.handle._thread.sent
        assert original.native.is_closed and original.handle._receipt.phase is None
        checkpoint = await original.handle.persist_minimal()
        assert checkpoint.state == 'unknown' and checkpoint.lease is None


@pytest.mark.asyncio
async def test_closed_original_spending_deadline_rechecks_between_actual_partial_writes(custody, vendor, peer, monkeypatch):
    async with original_acquisition(custody, vendor, peer, monkeypatch, quote_seconds=2) as original:
        await original.attempt.begin_dispatch()
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
        with pytest.raises(BaseException):
            await original.handle.run_original()
        assert len(written) == 1 and len(written[0]) == 17
        assert original.handle._thread.sent and original.handle._receipt.phase is None
        checkpoint = await original.handle.persist_minimal()
        assert checkpoint.state == 'unknown' and checkpoint.lease is None
        assert not peer.requests and original.native.is_closed
