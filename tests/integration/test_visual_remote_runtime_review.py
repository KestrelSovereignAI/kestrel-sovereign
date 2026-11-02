"""Review regressions using original issued custody and owned native TLS.

There is no remote worker/session double here. A connection or resident release
is a trap after an actual original Pod POST; TLS peers are never Runpod.
"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from kestrel_sdk.resources import ResourceUnsupported
from kestrel_sdk.operator import OperatorContext

from kestrel_sovereign._async_ownership import await_owned_task, raise_owned_outcome
from kestrel_sovereign.execution_custody import execution_evidence_carriers
from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_custody_snapshot
from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.privacy import PrivacyConfig, PrivacyMode
from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
from kestrel_sovereign.visual_computer.repository import CustodyRecord
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.integration.test_visual_custodied_runtime import NativeFixtureProvider, runtime_for
from tests.integration.test_visual_custody_backends import SCOPE
from tests.integration.test_visual_custody_backends import custody as custody
from tests.integration.test_visual_runpod_acquisition import original_host, original_quote
from tests.integration.test_visual_runpod_acquisition import peer as peer, vendor as vendor
from tests.integration.test_visual_runpod_lifecycle import lifecycle
from tests.integration.test_visual_runpod_quote import quoted_operation
from tests.integration.test_visual_runpod_quote import quote_peer as quote_peer
from tests.integration.test_visual_original_result_backends import OriginalTurn


class NoRemoteWorker:
    driver_id = "unimplemented-remote-worker"


@pytest.mark.asyncio
async def test_acquisition_only_native_pod_cannot_reach_connect(custody, vendor, peer, monkeypatch):
    caught = None
    async with original_host(custody, vendor, peer, monkeypatch) as h:
        provider = NativeFixtureProvider(h, mode="genuine")
        calls = []
        async def connect(*args):
            calls.append("connect")
            raise RuntimeError("unsupported native Pod reached connect")
        monkeypatch.setattr(provider, "connect", connect)
        runtime = runtime_for(h, provider, NoRemoteWorker())
        try:
            await runtime.admit(h.context, h.request)
        except BaseException as error:
            caught = error
        row = await h.store.get(SCOPE, request_id=h.request.request_id, guard=h.check)
        assert row.remote_environment is None and row.acquisition.binding is not None
        assert row.lease.provider_resource_id == "owned-native-pod"
    assert len(peer.requests) == 1 and h.native.is_closed
    assert isinstance(caught, ResourceUnsupported) and "bootstrap" in str(caught)
    assert calls == []


@pytest.mark.asyncio
async def test_acquisition_only_native_pod_never_borrows_resident_release(custody, vendor, peer, monkeypatch):
    async with original_host(custody, vendor, peer, monkeypatch) as h:
        provider = NativeFixtureProvider(h, mode="genuine")
        runtime = runtime_for(h, provider, NoRemoteWorker())
        row = await h.store.admit(SCOPE, CustodyRecord(request=h.request, session_id="unopened"), h.check)
        row.quote, row.phase = original_quote(h.plan), "quoted"
        await h.store.save(SCOPE, row, h.check)
        row = await runtime._acquire_custodied(h.context, row, h.check, provider)
        effects = []
        async def release(scope, lease):
            effects.append(lease.provider_resource_id)
            return lease
        monkeypatch.setattr(provider, "release", release)
        monkeypatch.setattr(runtime, "_now", lambda: row.lease.expires_at + timedelta(seconds=1))
        before = row.model_copy(deep=True)
        debt = await runtime.maintain_owned_sessions(SCOPE)
        after = await h.store.get(SCOPE, session_id=row.session_id, guard=h.check)
    assert len(peer.requests) == 1 and h.native.is_closed
    assert effects == [], "resident maintenance invoked unadmitted native release"
    assert after == before and debt == [row.session_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("release,expired", [(False, False), (True, False), (True, True)], ids=["inspect", "release", "expired-release"])
async def test_live_foreground_lifecycle_survives_maintenance(custody, vendor, quote_peer, monkeypatch, release, expired):
    failure = None
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=release) as h:
        ready = asyncio.Event()
        loop = asyncio.get_running_loop()
        if expired:
            monkeypatch.setattr(h.runtime, "_now", lambda: h.row.lease.expires_at + timedelta(seconds=1))
        quote_peer.before_reply = lambda: loop.call_soon_threadsafe(ready.set)
        async def maintenance():
            await asyncio.wait_for(ready.wait(), 5)
            return await h.runtime.maintain_owned_sessions(SCOPE)
        task = asyncio.create_task(maintenance())
        try:
            try:
                result = await h.lifecycle.run_original()
            except BaseException as error:
                failure = error
        finally:
            outcome = await await_owned_task(task)
            raise_owned_outcome(outcome, operation="original maintenance regression")
        stage = h.lifecycle._stage
        assert stage._worker_entered and stage._worker_exited and stage._client_closed
        assert not stage._thread.sockets
    assert failure is None, "periodic maintenance revoked a fresh foreground lifecycle call"
    assert result.state == ("unknown" if release else "pending")
    assert len(quote_peer.requests) == (2 if release else 1)


@pytest.mark.asyncio
async def test_expired_never_dispatched_remote_quote_retires(custody, vendor, quote_peer, monkeypatch):
    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        quote = await h.call.run_original()
        row = await h.store._accepted_runpod_quote(SCOPE, h.call, quote)
        monkeypatch.setattr(h.runtime, "_now", lambda: row.request.expires_at + timedelta(seconds=1))
        debt = await h.runtime.maintain_owned_sessions(SCOPE)
        after = await h.store.get(SCOPE, session_id=row.session_id, guard=h.check)
    assert len(quote_peer.requests) == 1
    assert after.phase == "retired" and after.retired_from == "quoted"
    assert after.remote_environment == row.remote_environment and after.lease is None
    assert debt == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["entry-refusal", "entry-and-close"])
async def test_quote_preentry_refusal_preserves_original_and_no_worker_exit(custody, vendor, quote_peer, monkeypatch, cleanup_fails):
    from kestrel_sovereign import execution_sync_http
    original, closing = RuntimeError("original native entry refusal"), RuntimeError("original close failure")
    caught = None
    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        async def refused(*args, **kwargs):
            control = h.call._control
            close = control.close
            def close_then_refuse():
                close()
                raise closing
            if cleanup_fails:
                monkeypatch.setattr(control, "close", close_then_refuse)
            raise original
        monkeypatch.setattr(execution_sync_http, "await_native_sync_http", refused)
        try:
            await h.call.run_original()
        except BaseException as error:
            caught = error
        assert h.call._client.is_closed and not h.call._worker_entered
    carriers = execution_evidence_carriers(caught)
    assert original in carriers, "pre-entry physical close replaced original native refusal"
    if cleanup_fails:
        assert closing in carriers
    assert h.call._worker_exited is False, "unentered worker was reported physically exited"
    assert not quote_peer.requests


@pytest.mark.asyncio
async def test_revoke_joins_original_snapshot_after_natural_map_retirement(custody, vendor, quote_peer, monkeypatch):
    manager = lifecycle(custody, vendor, quote_peer, monkeypatch)
    h = await manager.__aenter__()
    original = h.runtime.egress._close_environment_call
    joined_first, continue_drain = asyncio.Event(), asyncio.Event()
    stopper = None

    def pause_after_original_join(call):
        pending = original(call)
        async def join():
            value = await pending
            if call is h.call and asyncio.current_task() is stopper:
                # Passive scheduling barrier AFTER the genuine original join.
                # No replacement Future/receipt, close result or native effect.
                joined_first.set()
                await continue_drain.wait()
            return value
        return join()

    monkeypatch.setattr(h.runtime.egress, "_close_environment_call", pause_after_original_join)
    stopper = asyncio.create_task(h.runtime.egress.revoke_session(SCOPE, h.row.session_id))
    try:
        await asyncio.wait_for(joined_first.wait(), 5)
    finally:
        try:
            # The issued but unused Inspect retires through its actual original
            # context while an external closer is between captured owners.
            await manager.__aexit__(None, None, None)
        finally:
            continue_drain.set()
            outcome = await await_owned_task(stopper)
    assert not h.runtime.egress._calls
    assert not h.store._original_results and not h.store.db.backend._original_result_holds
    assert outcome.error is None, "drain re-authenticated a naturally retired same-map owner after awaiting"


@pytest.mark.asyncio
async def test_native_pod_without_quote_refuses_fresh_foreground_erase(custody, vendor, peer, monkeypatch):
    async with original_host(custody, vendor, peer, monkeypatch) as h:
        provider = NativeFixtureProvider(h, mode="genuine")
        runtime = runtime_for(h, provider, NoRemoteWorker())
        row = await h.store.admit(SCOPE, CustodyRecord(request=h.request, session_id="unopened"), h.check)
        row.quote, row.phase = original_quote(h.plan), "quoted"
        await h.store.save(SCOPE, row, h.check)
        row = await runtime._acquire_custodied(h.context, row, h.check, provider)
        before = row.model_copy(deep=True)
    # The original ADMIT has unwound. A real new foreground RELEASE admission
    # does not reconstruct an absent original native quote or erase contract.
    now = datetime.now(UTC)
    context = OperatorContext(tenant_id=SCOPE.tenant_id, principal_id=SCOPE.principal_id,
        acting_agent_id=SCOPE.agent_id, correlation_id="release-without-private-quote",
        issued_at=now, expires_at=now+timedelta(seconds=30),
        granted_actions={VisualAction.RELEASE.value}, granted_capabilities={"visual-computer"},
        permitted_boundary_ids={h.store.storage_id})
    privacy = DurablePersistenceGate()
    holder = SimpleNamespace(privacy_mode=PrivacyMode.NORMAL, privacy_config=PrivacyConfig(),
        _get_durable_persistence_gate=lambda: privacy)
    def current(ctx, action, target):
        assert ctx is context and action == VisualAction.RELEASE and target is None
    turn = ExecutionCustody(OriginalTurn(h.store.db.backend_type))
    effects, caught = [], None
    async def release(scope, lease):
        effects.append(lease.provider_resource_id)
        return lease
    monkeypatch.setattr(provider, "release", release)
    with (bind_execution_custody_snapshot((turn,)),
          h.authority.bind_authenticated(context, privacy_holder=holder,
              check_current=current, authority_gate=AsyncReaderWriterLock())):
        try:
            await runtime.release(context, row.session_id)
        except BaseException as error:
            caught = error
        after = await h.store.get(SCOPE, session_id=row.session_id)
    assert isinstance(caught, ResourceUnsupported) and "private native quote" in str(caught)
    assert effects == [] and after == before
    assert len(peer.requests) == 1 and h.native.is_closed


@pytest.mark.asyncio
async def test_native_profile_rejects_local_before_preparation(custody, vendor, peer, monkeypatch):
    async with original_host(custody, vendor, peer, monkeypatch) as h:
        # This is an unissued request candidate, never a stored/acquired row.
        # The exact canonical plan constructor revalidates its actual profile;
        # changing the descriptive location cannot issue a native preparation.
        local = h.request.model_copy(update={"location": "local"})
        with pytest.raises(ResourceUnsupported, match="explicit remote browser"):
            replace(h.plan, request=local)
        assert h.plan.request is h.request
    assert h.native.is_closed and not peer.requests
