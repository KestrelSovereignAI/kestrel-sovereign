"""Synthetic TLS credential-retention regression; no Runpod/cloud resource."""
import pytest

from tests.integration.test_visual_runpod_acquisition import (
    SCOPE,
    custody as custody,
    original_acquisition,
    peer as peer,
    vendor as vendor,
)


def credential_present(raw, credential):
    # Detect the original raw value and the reversible hex storage used by
    # both Request.headers and prepared native configuration. Never print it.
    return credential in raw or credential.hex().encode() in raw.lower()


@pytest.mark.asyncio
async def test_original_acquisition_binding_never_persists_reversible_fixture_credential(custody, vendor, peer, monkeypatch):
    snapshots = {}
    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        credential = b'owned-native-api-key'
        assert original.native.headers['authorization'] == 'Bearer '+credential.decode()
        async def capture(stage):
            row = await original.store.get(SCOPE, request_id=original.request.request_id, guard=original.check)
            assert row.acquisition is not None and row.acquisition.binding is not None
            snapshots[stage] = row.acquisition.binding.get_secret_value()
        await capture('prepared')
        assert await original.attempt.begin_dispatch() is True
        await capture('dispatching')
        phase = await original.handle.run_original()
        minimal = await original.handle.persist_minimal()
        await capture('minimal')
        original.producer.finish_optional_cost(phase)
        await original.attempt.record_accepted(minimal.lease.model_copy(update={'state': 'pending'}))
        await capture('accepted')
        assert len(peer.requests) == 1 and original.native.is_closed and len(original.closes) == 1
        assert original.handle._worker_exited and original.handle._admission.task.done()
        assert not original.handle._thread.sockets
    assert not original.store._original_results and not original.store.db.backend._original_result_holds
    assert original.handle._state == 'closed'
    assert len(snapshots) == 4 and len(set(snapshots.values())) == 1
    leaking = [stage for stage, raw in snapshots.items() if credential_present(raw, credential)]
    assert not leaking, 'reversible fixture credential persisted in original acquisition binding'


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['same', 'credential', 'headers', 'account', 'endpoint', 'tls', 'timeout', 'worker'])
async def test_original_prepared_rebind_requires_the_same_complete_native_fingerprint(custody, vendor, peer, monkeypatch, change):
    import json
    from pydantic import SecretBytes
    from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionPreparation
    from kestrel_sdk.resources import ResourceError
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.visual_computer.repository import CustodyRecord
    from kestrel_sovereign.visual_computer.runpod_acquisition import prepare_runpod_acquisition_producer
    from tests.integration.test_visual_runpod_acquisition import original_host, original_quote

    async with original_host(custody, vendor, peer, monkeypatch) as host:
        row = await host.store.admit(SCOPE, CustodyRecord(request=host.request, session_id='credential-binding-session'), host.check)
        row.quote, row.phase = original_quote(host.plan), 'quoted'
        await host.store.save(SCOPE, row, host.check)
        dto = host.plan.pod_request(host.request, host.plan.profile, host.plan.noncompute, at=host.plan.planned_at)
        prepared = host.control.prepare_cpu_pod(dto, plan=host.plan)
        preparation = EnvironmentAcquisitionPreparation(create_payload=SecretBytes(prepared.exact_body()),
            worker_identity=SecretBytes(b'PRIVATE-ORIGINAL-WORKER-TLS-IDENTITY'))
        async with host.store.original_acquisition_result(SCOPE, row, host.request, row.quote, guard=host.check) as attempt:
            first = await prepare_runpod_acquisition_producer(attempt, host.control, dto,
                prepared=prepared, plan=host.plan, preparation=preparation)
            attempt_id = attempt.checkpoint.attempt_id
        original = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        binding_bytes = original.acquisition.binding.get_secret_value()
        envelope = json.loads(binding_bytes)
        assert set(envelope) == {'schema_version', 'fingerprint'} and envelope['schema_version'] == 2
        assert len(envelope['fingerprint']) == 64 and set(envelope['fingerprint']) <= set('0123456789abcdef')
        assert first._original_acquisition._state == 'closed' and host.native.is_closed
        assert not credential_present(binding_bytes, b'owned-native-api-key')
        kwargs = dict(api_key='owned-native-api-key', base_url=peer.url,
            prepared_cpu_account='owned-native-account', prepared_cpu_ca_data=peer.ca,
            prepared_cpu_acquisition=True, connect_timeout=2, read_timeout=4)
        if change == 'credential': kwargs['api_key'] = 'owned-replacement-fixture-key'
        if change == 'headers': kwargs['user_agent'] = 'owned-different-user-agent'
        if change == 'account': kwargs['prepared_cpu_account'] = 'owned-other-account'
        if change == 'endpoint': kwargs['base_url'] = peer.url.replace('127.0.0.1', 'localhost', 1)
        if change == 'tls': kwargs['prepared_cpu_ca_data'] = peer.ca+b'\n'
        if change == 'timeout': kwargs['connect_timeout'] = 3
        control = vendor.clients.RunpodControlPlaneClient(**kwargs)
        try:
            next_prepared = control.prepare_cpu_pod(dto, plan=host.plan)
            next_preparation = preparation if change != 'worker' else EnvironmentAcquisitionPreparation(
                create_payload=preparation.create_payload, worker_identity=SecretBytes(b'PRIVATE-DIFFERENT-WORKER-IDENTITY'))
            async with host.store.original_acquisition_result(SCOPE, original, host.request, original.quote, guard=host.check) as next_attempt:
                assert next_attempt.checkpoint.attempt_id == attempt_id
                if change == 'same':
                    second = await prepare_runpod_acquisition_producer(next_attempt, control, dto,
                        prepared=next_prepared, plan=host.plan, preparation=next_preparation)
                    assert second._original_acquisition._binding._binding_bytes == binding_bytes
                    assert next_attempt.checkpoint.state == 'prepared' and next_attempt.checkpoint.sequence == 1
                else:
                    with pytest.raises((ExecutionAuthorityError, ResourceError)):
                        await prepare_runpod_acquisition_producer(next_attempt, control, dto,
                            prepared=next_prepared, plan=host.plan, preparation=next_preparation)
                    proposed = next_attempt._result._native._binding._binding_bytes
                    assert proposed is not None and proposed != binding_bytes
                    assert not credential_present(proposed, b'owned-native-api-key')
        finally:
            if not control.transport._client.is_closed:
                control.close()
        current = await host.store.get(SCOPE, request_id=host.request.request_id, guard=host.check)
        assert current.acquisition.binding.get_secret_value() == binding_bytes
        assert current.acquisition.storage == original.acquisition.storage
        # A genuine rebind ACK advances only the row revision, not the
        # original attempt/sequence or private preparation. Rejections do not.
        assert current.revision == original.revision+(1 if change == 'same' else 0)
        assert not peer.requests and not peer.snis and control.transport._client.is_closed
        assert not host.store._original_results and not host.store.db.backend._original_result_holds


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['credential', 'headers', 'configuration', 'body', 'source'])
async def test_original_native_write_refuses_changed_live_binding_before_any_tls(custody, vendor, peer, monkeypatch, change):
    import httpx
    from pydantic import SecretBytes
    from kestrel_sdk.resources import ResourceError
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError

    async with original_acquisition(custody, vendor, peer, monkeypatch) as original:
        assert await original.attempt.begin_dispatch() is True
        binding = original.handle._binding
        with monkeypatch.context() as patch:
            if change in {'credential', 'headers'}:
                name = 'authorization' if change == 'credential' else 'user-agent'
                old = original.native.headers[name]
                original.native.headers[name] = 'Bearer owned-replacement-fixture-key' if change == 'credential' else 'owned-changed-agent'
            elif change == 'configuration':
                old = original.native.timeout
                original.native.timeout = httpx.Timeout(4, connect=3)
            elif change == 'body':
                old = original.prepared.body
                object.__setattr__(original.prepared, 'body', SecretBytes(b'{"changed":"original-body"}'))
            else:
                module = next(module for module, _ in binding._canonical.modules if module.__name__.endswith('.models'))
                patch.setattr(module, '_pod_optional_cost', lambda payload: None)
            try:
                with pytest.raises((ExecutionAuthorityError, ResourceError)):
                    await original.handle.run_original()
                assert not peer.requests and not peer.snis
                assert original.handle._worker is None and original.handle._request is None
            finally:
                if change in {'credential', 'headers'}: original.native.headers[name] = old
                elif change == 'configuration': original.native.timeout = old
                elif change == 'body': object.__setattr__(original.prepared, 'body', old)
        assert original.attempt.checkpoint.state == 'dispatching'
    assert original.native.is_closed and len(original.closes) == 1
    assert not original.store._original_results and not original.store.db.backend._original_result_holds
