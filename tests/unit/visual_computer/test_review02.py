"""Review02 regressions: original native preflight facts and unused requests."""
import asyncio
from pathlib import Path
import sys

import pytest
from kestrel_sdk.resources import ResourceUnsupported, ResourceOutcomeUnknown
from kestrel_sovereign.visual_computer.docker import DockerComputerDriver, DockerEnvironmentProvider, DockerCommandError, PROVIDER
from tests.unit.visual_computer.test_runtime import opened


@pytest.mark.asyncio
async def test_native_inspect_failure_before_rpc_does_not_wedge_original_action(harness, monkeypatch, tmp_path):
    h = harness
    ctx = h.context()
    h.adapter.driver_id = PROVIDER
    with h.bind(ctx):
        session, request = await opened(h, ctx)
        driver = DockerComputerDriver(docker_socket='/tmp/owned-review02.sock', egress_guard=h.runtime.egress.operation)
        driver._connections[session.session_id] = object()
        executable = tmp_path/'owned-daemon'
        executable.write_text(f'#!{sys.executable}\nimport sys\nsys.stderr.write("owned transient inspect failure")\nsys.exit(1)\n')
        executable.chmod(0o700)
        driver.boundary.executable = str(executable)
        async def bound(connection, **kwargs):
            return await driver.boundary.invoke(['container', 'inspect', 'c'*64])
        monkeypatch.setattr(driver, '_bound', bound)
        h.runtime.drivers = {PROVIDER: driver}
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, request)
        assert not await h.repository.has_pending_actions(h.runtime.scope(ctx), session.session_id)
        receipt = await h.runtime.action_status(ctx, session.session_id, request.action_id)
        assert receipt.outcome == 'denied' and receipt.reason_code == 'host_not_dispatched'
        assert await h.runtime.act(ctx, request) == receipt
        assert (await h.runtime.release(ctx, session.session_id))['released']
        assert await h.repository.records(h.runtime.scope(ctx)) == []


@pytest.mark.asyncio
async def test_real_provider_quote_refusals_do_not_burn_live_request_slots(harness):
    h = harness
    ctx = h.context()
    h.repository.max_requests = 1
    provider = DockerEnvironmentProvider(docker_socket='/tmp/owned-never-contacted.sock',
        image='local-worker@sha256:'+'a'*64, credential_key=b'k'*32)
    h.runtime.providers[PROVIDER] = provider
    with h.bind(ctx):
        for index in range(3):
            with pytest.raises(ResourceUnsupported, match='unsupported_offline_browser_request'):
                await h.runtime.admit(ctx, h.request(ctx, request_id='bad-'+str(index), provider_id=PROVIDER,
                    profile_id='offline-v1', image='other@sha256:'+'b'*64))
        good = await h.runtime.admit(ctx, h.request(ctx, request_id='usable'))
        assert good.session_id and h.adapter.acquires == 1


@pytest.mark.asyncio
async def test_quote_cancellation_retires_original_without_resurrection(harness, monkeypatch):
    from kestrel_sdk.resources import ResourceConflict
    h=harness;ctx=h.context();entered=asyncio.Event()
    original=h.adapter.quote
    async def blocked(request):
        entered.set();await asyncio.Event().wait()
    monkeypatch.setattr(h.adapter,'quote',blocked)
    request=h.request(ctx)
    with h.bind(ctx):
        active=asyncio.create_task(h.runtime.admit(ctx,request))
        await entered.wait();active.cancel()
        with pytest.raises(asyncio.CancelledError):await active
        row=await h.repository.get(h.runtime.scope(ctx),request_id=request.request_id)
        assert row.phase=='retired' and row.lease is None and h.adapter.acquires==0
        assert not (await h.runtime.status(ctx,row.session_id))['requires_reconciliation']
        expected={'session_id':row.session_id,'request_id':request.request_id,'retired':True,'acquired':False,'released':False}
        assert await h.runtime.release(ctx,row.session_id)==expected
        assert await h.runtime.reconcile_request(ctx,request.request_id)==expected
        with pytest.raises(ResourceConflict,match='retired without acquisition'):await h.runtime.admit(ctx,request)
        monkeypatch.setattr(h.adapter,'quote',original)
        h.repository.max_requests=1
        assert (await h.runtime.admit(ctx,h.request(ctx,request_id='fresh'))).session_id != row.session_id


@pytest.mark.asyncio
async def test_native_proof_requires_original_live_invocation_not_an_exception_name(harness,monkeypatch):
    from kestrel_sovereign.visual_computer.dispatch import NativeActionNotDispatched
    h=harness;ctx=h.context()
    with h.bind(ctx):
        session,request=await opened(h,ctx)
        async def forged(*args):raise NativeActionNotDispatched(object(),RuntimeError('forged'))
        monkeypatch.setattr(h.adapter,'act',forged)
        with pytest.raises(ResourceOutcomeUnknown,match='not_original'):
            await h.runtime.act(ctx,request)
        assert await h.repository.has_pending_actions(h.runtime.scope(ctx),session.session_id)


@pytest.mark.asyncio
async def test_expired_preacquisition_maintenance_retires_only_original_safe_phases(harness):
    from datetime import UTC, datetime, timedelta
    from kestrel_sovereign.visual_computer.repository import CustodyRecord
    h=harness;ctx=h.context();now=datetime.now(UTC);scope=h.runtime.scope(ctx)
    with h.bind(ctx):
        request=h.request(ctx,requested_at=now-timedelta(seconds=80),expires_at=now-timedelta(seconds=1))
        safe=await h.repository.admit(scope,CustodyRecord(request=request,session_id='safe'),lambda:None)
        unknown=await h.repository.admit(scope,CustodyRecord(request=request.model_copy(update={'request_id':'unknown'}),session_id='unknown'),lambda:None)
        unknown.phase='acquiring';await h.repository.save(scope,unknown)
        assert await h.runtime.maintain_owned_sessions(scope)==['unknown']
        assert (await h.repository.get(scope,session_id='safe')).phase=='retired'
        assert (await h.repository.get(scope,session_id='unknown')).phase=='acquiring'
        assert h.adapter.acquires==0
