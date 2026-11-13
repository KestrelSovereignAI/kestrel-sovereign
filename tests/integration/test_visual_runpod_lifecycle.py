"""Actual owned TLS + existing current SQLite/PG custody; no remote readiness."""

import asyncio
import ssl
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionPreparation
from kestrel_sdk.operator import OperatorAuthorizationError, OperatorContext
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from pydantic import SecretBytes

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    bind_execution_custody_snapshot,
)
from kestrel_sovereign.privacy import PrivacyConfig, PrivacyMode
from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
from kestrel_sovereign.visual_computer.runpod_acquisition import (
    prepare_runpod_acquisition_producer,
)
from kestrel_sovereign.visual_computer.runpod_environment import RunpodEnvironmentHost
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.integration.test_visual_custody_backends import SCOPE
from tests.integration.test_visual_custody_backends import (
    custody as custody,  # noqa: PLC0414 - pytest fixture re-export.
)
from tests.integration.test_visual_original_result_backends import OriginalTurn
from tests.integration.test_visual_runpod_acquisition import (
    vendor as vendor,  # noqa: PLC0414 - pytest fixture re-export.
)
from tests.integration.test_visual_runpod_quote import (
    _BoundedQuoteProvider,
    quoted_operation,
)
from tests.integration.test_visual_runpod_quote import (
    quote_peer as quote_peer,  # noqa: PLC0414 - pytest fixture re-export.
)


async def accepted_owned_pod(h, row, peer, *, known_id=True):
    """Genuine existing A issuer/once-POST/minimal native ACK, never a lease DTO shim."""
    plan = h.call._native_plan
    dto = plan.pod_request(h.request, plan.profile, plan.noncompute, at=plan.planned_at)
    prepared = h.control.prepare_cpu_pod(dto, plan=plan)
    peer.reply = {"id": "owned-native-pod", "name": "owned-native", "status": "RUNNING", "cost": 0.16}
    preparation = EnvironmentAcquisitionPreparation(create_payload=SecretBytes(prepared.exact_body()),
        worker_identity=SecretBytes(b"PRIVATE-OWNED-LIFECYCLE-FIXTURE-WORKER"))
    async with h.store.original_acquisition_result(SCOPE, row, h.request, row.quote, guard=h.check) as attempt:
        producer = await prepare_runpod_acquisition_producer(attempt, h.control, dto,
            prepared=prepared, plan=plan, preparation=preparation)
        assert await attempt.begin_dispatch() is True
        if not known_id:
            return await h.store.get(SCOPE, session_id=row.session_id, guard=h.check)
        phase = await producer._original_acquisition.run_original()
        checkpoint = await producer._original_acquisition.persist_minimal()
        assert producer.finish_optional_cost(phase).id == "owned-native-pod"
        await attempt.record_accepted(checkpoint.lease.model_copy(update={"state": "pending"}))
    return await h.store.get(SCOPE, session_id=row.session_id, guard=h.check)


@asynccontextmanager
async def lifecycle(custody, vendor, quote_peer, monkeypatch, *, release=False, activate=True, known_id=True):
    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        quote = await h.call.run_original()
        record = await h.store._accepted_runpod_quote(SCOPE, h.call, quote)
        record = await accepted_owned_pod(h, record, quote_peer, known_id=known_id)
        now = datetime.now(UTC)
        quote_peer.reply = {"id": "owned-native-pod", "name": "kestrel-cpu-"+h.request.digest()[:32],
            "image": h.plan.profile.image, "cloud": "SECURE", "disk": 20, "dataCenterId": "US-TX-3",
            "status": "RUNNING", "ports": [], "env": {}, "mounts": {}, "args": "", "registry": None,
            "locked": False, "template": None, "globalNetworking": {"enabled":False},
            "cpu": {"id":"cpu5c","vcpuCount":4,"memory":10}, "cost": 0.16,
            "startedAt": now.isoformat()}
        quote_peer.requests.clear()
        if not release:
            if activate:
                async with h.host._activate_inspect(h.context, record) as call:
                    yield SimpleNamespace(**vars(h), row=record, lifecycle=call)
            else:
                yield SimpleNamespace(**vars(h), row=record)
        else:
            saved = SimpleNamespace(**vars(h), row=record)
    if release:
        # The ADMIT operation and binding have unwound. This is a NEW original
        # foreground host admission and backend work, not a copied ADMIT context.
        now = datetime.now(UTC)
        context = OperatorContext(tenant_id=SCOPE.tenant_id, principal_id=SCOPE.principal_id,
            acting_agent_id=SCOPE.agent_id, correlation_id="fresh-release", issued_at=now,
            expires_at=now+timedelta(seconds=90), granted_actions={VisualAction.RELEASE.value},
            granted_capabilities={"visual-computer"}, permitted_boundary_ids={saved.store.storage_id})
        privacy = DurablePersistenceGate()
        holder = SimpleNamespace(privacy_mode=PrivacyMode.NORMAL, privacy_config=PrivacyConfig(),
            _get_durable_persistence_gate=lambda: privacy)
        def release_current(ctx, action, target):
            assert ctx is context and action == VisualAction.RELEASE
            if not saved.current["live"]:
                raise PermissionError("original foreground release withdrawn")
        turn = ExecutionCustody(OriginalTurn(saved.store.db.backend_type))
        with (
            bind_execution_custody_snapshot((turn,)),
            saved.authority.bind_authenticated(context, privacy_holder=holder,
                check_current=release_current, authority_gate=AsyncReaderWriterLock()),
        ):
            if activate:
                async with saved.host._activate_release(context, saved.row) as call:
                    fields = dict(vars(saved), lifecycle=call, release_context=context, check=call._check)
                    yield SimpleNamespace(**fields)
            else:
                async with saved.runtime.operation(context, VisualAction.RELEASE) as check:
                    yield SimpleNamespace(**dict(vars(saved), release_context=context, check=check))


@pytest.mark.asyncio
async def test_native_inspect_current_ack_never_admits_worker(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch) as h:
        lease = await h.lifecycle.run_original()
        assert lease.state == "pending" and lease.expires_at == h.row.lease.expires_at
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.remote_environment.inspection.state == "running"
        assert row.remote_environment.inspection.provider_resource_id == lease.provider_resource_id
        assert row.lease == lease and row.connection is None and row.session is None
        assert len(quote_peer.requests) == 1
        assert quote_peer.requests[0][0] == "/v2/pods/owned-native-pod"
        assert quote_peer.requests[0][1]["Accept-Encoding"] == "identity"
        stage = h.lifecycle._stage
        assert stage._client_closed and stage._worker_exited and not stage._thread.sockets
        with pytest.raises(ExecutionAuthorityError):
            await h.lifecycle.run_original()


@pytest.mark.asyncio
async def test_native_release_retains_204_as_unknown_same_row(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True) as h:
        lease = await h.lifecycle.run_original()
        assert lease.state == "unknown" and lease.released_at is None and lease.actual_cost_usd is None
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        fact = row.remote_environment.release
        assert row.phase == "fenced" and fact.state == "acknowledged" and fact.native_status == 204
        assert fact.acknowledged_at >= fact.requested_at
        assert row.lease == lease
        assert [r[0] for r in quote_peer.requests] == ["/v2/pods/owned-native-pod", "/v2/pods/owned-native-pod"]
        assert (await h.store.db.fetchone("SELECT released FROM visual_computer_requests"))[0] == 0


@pytest.mark.asyncio
async def test_admit_descriptor_cannot_enter_release(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch) as h:
        with pytest.raises(OperatorAuthorizationError):
            async with h.host._activate_release(h.context, h.row):
                pytest.fail("ADMIT borrowed RELEASE")
        assert not quote_peer.requests


@pytest.mark.asyncio
async def test_native_lifecycle_direct_and_foreign_task_refuse(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch) as h:
        with pytest.raises(ExecutionAuthorityError):
            async with h.host.release_operation(SCOPE, h.row.lease):
                pytest.fail("cross-purpose borrow")
        with pytest.raises(ExecutionAuthorityError):
            await asyncio.create_task(h.lifecycle.run_original())
        assert not quote_peer.requests


@pytest.mark.asyncio
async def test_native_lifecycle_revocation_after_get_retains_intent_no_delete(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True) as h:
        quote_peer.before_reply = lambda: h.current.update(live=False)
        with pytest.raises((ExecutionAuthorityError, OperatorAuthorizationError, ResourceOutcomeUnknown,
                            BaseExceptionGroup, OSError)):
            await h.lifecycle.run_original()
        assert len(quote_peer.requests) == 1
        h.current["live"] = True  # New current read only, not a replay of the old operation.
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.remote_environment.release.state == "intent" and row.lease.state == "unknown"


@pytest.mark.asyncio
async def test_remote_maintenance_never_borrows_foreground_release(custody, vendor, quote_peer, monkeypatch):
    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        quote = await h.call.run_original()
        row = await h.store._accepted_runpod_quote(SCOPE, h.call, quote)
        row = await accepted_owned_pod(h, row, quote_peer)
        before = await h.store.get(SCOPE, session_id=row.session_id, guard=h.check)
        effects = []
        class UnadmittedRemote:
            async def release(self, scope, lease):
                effects.append((scope, lease))
                return lease
        monkeypatch.setattr(h.runtime, "_provider", lambda record: UnadmittedRemote())
        monkeypatch.setattr(h.runtime, "_now", lambda: row.lease.expires_at+timedelta(seconds=1))
        debt = await h.runtime.maintain_owned_sessions(SCOPE)
        assert effects == [], "maintenance called remote release without a current RELEASE admission"
        after = await h.store.get(SCOPE, session_id=row.session_id, guard=h.check)
        assert after == before and debt == [row.session_id]


@pytest.mark.asyncio
async def test_native_inspect_and_release_keep_original_lease_after_wrong_id(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True) as h:
        quote_peer.reply["id"] = "foreign-pod"
        with pytest.raises(ResourceConflict):
            await h.lifecycle.run_original()
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.lease.provider_resource_id == "owned-native-pod"
        assert row.remote_environment.release.state == "intent" and len(quote_peer.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["gzip", "duplicate", "oversize"])
async def test_native_lifecycle_rejects_encoded_or_oversize_before_decode(custody, vendor, quote_peer, monkeypatch, wire):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch) as h:
        import httpx
        decoded = []
        decoder = httpx.Response._get_content_decoder
        def observe(response):
            decoded.append(True)
            return decoder(response)
        monkeypatch.setattr(httpx.Response, "_get_content_decoder", observe)
        if wire == "oversize":
            quote_peer.raw = b" " * (1024*1024+1)
        else:
            quote_peer.encoding = ["gzip"] if wire == "gzip" else ["identity", "identity"]
        with pytest.raises((ExecutionAuthorityError, BaseExceptionGroup, OSError)):
            await h.lifecycle.run_original()
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.remote_environment.inspection is None
        assert decoded == [], "refused headers reached HTTPX decoder"
        assert h.lifecycle._stage._client_closed


@pytest.mark.asyncio
async def test_fresh_release_after_use_expiry_never_rechecks_spending(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True) as h:
        original_expiry = h.row.lease.expires_at
        monkeypatch.setattr(h.runtime, "_now", lambda: original_expiry+timedelta(seconds=1))
        value = await h.lifecycle.run_original()
        assert value.state == "unknown" and value.expires_at == original_expiry
        assert len(quote_peer.requests) == 2


@pytest.mark.asyncio
async def test_seventeen_actual_tls_bytes_revoked_no_inspection_commit(custody, vendor, quote_peer, monkeypatch):
    sent = []
    native_send = ssl.SSLSocket.send
    async with lifecycle(custody, vendor, quote_peer, monkeypatch) as h:
        def partial(sock, data, flags=0):
            if data.startswith(b"GET /v2/pods/"):
                result = native_send(sock, data[:17], flags)
                sent.append(result)
                h.current["live"] = False
                return result
            return native_send(sock, data, flags)
        monkeypatch.setattr(ssl.SSLSocket, "send", partial)
        with pytest.raises((ExecutionAuthorityError, OperatorAuthorizationError, BaseExceptionGroup, OSError)):
            await h.lifecycle.run_original()
        assert sent == [17]
        assert not quote_peer.requests
        h.current["live"] = True
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.remote_environment.inspection is None


@pytest.mark.asyncio
async def test_release_lost_intent_ack_forbids_all_native_requests(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True) as h:
        original = h.store.db.backend._commit_current_runpod_quote
        lost = ConnectionError("owned lost original intent ACK")
        async def lose_ack(owner):
            await original(owner)
            raise lost
        monkeypatch.setattr(h.store.db.backend, "_commit_current_runpod_quote", lose_ack)
        with pytest.raises(BaseException) as failure:
            await h.lifecycle.run_original()
        pending, seen = [failure.value], set()
        while pending:
            item = pending.pop()
            if id(item) in seen:
                continue
            seen.add(id(item))
            pending.extend(value for value in (item.__cause__, item.__context__) if value is not None)
            pending.extend(getattr(item, "exceptions", ()))
        assert id(lost) in seen, "original lost intent ACK error was replaced"
        assert not quote_peer.requests and h.lifecycle._stage is None
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.remote_environment.release.state == "intent"


@pytest.mark.asyncio
async def test_release_final_commit_rechecks_original_current(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True) as h:
        reached = []
        if h.store.db.backend_type == "postgres":
            from kestrel_sovereign.storage.db import postgres
            prove = postgres._prove_committable
            async def revoke_after_probe(connection):
                await prove(connection)
                if h.lifecycle._kind == "release-acknowledgement" and not reached:
                    reached.append("postgres-post-proof")
                    h.current["live"] = False
            monkeypatch.setattr(postgres, "_prove_committable", revoke_after_probe)
        else:
            conn = h.store.db.backend._connection
            execute = conn._execute
            async def revoke_at_native_commit(fn, *args, **kwargs):
                if (fn == conn._conn.commit and h.lifecycle._kind == "release-acknowledgement"
                        and conn._conn._current_row_owner is h.lifecycle._mutation):
                    def physical():
                        reached.append("sqlite-queued-commit")
                        h.current["live"] = False
                        return fn(*args, **kwargs)
                    return await execute(physical)
                return await execute(fn, *args, **kwargs)
            monkeypatch.setattr(conn, "_execute", revoke_at_native_commit)
        with pytest.raises((ExecutionAuthorityError, OperatorAuthorizationError, ResourceOutcomeUnknown, BaseExceptionGroup, OSError, asyncio.CancelledError)):
            await h.lifecycle.run_original()
        assert reached and len(quote_peer.requests) == 2
        h.current["live"] = True
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.remote_environment.release.state == "intent"
        assert row.lease.released_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_priority_revoke_physically_joins_owned_native_job(custody, vendor, quote_peer, monkeypatch, repeat_cancel):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True) as h:
        loop = asyncio.get_running_loop()
        seen, permit_reply = asyncio.Event(), threading.Event()
        def block_reply():
            loop.call_soon_threadsafe(seen.set)
            assert permit_reply.wait(5), "owned reply barrier did not retire"
        quote_peer.before_reply = block_reply
        async def stop():
            await seen.wait()
            await h.runtime.egress.revoke_session(SCOPE, h.row.session_id)
        stopper = asyncio.create_task(stop())
        if repeat_cancel:
            async def withdraw():
                await seen.wait()
                await asyncio.sleep(0)
                stopper.cancel()
                stopper.cancel()
            cancellation = asyncio.create_task(withdraw())
        try:
            with pytest.raises((ExecutionAuthorityError, OperatorAuthorizationError, ResourceOutcomeUnknown, BaseExceptionGroup, OSError, asyncio.CancelledError)):
                await h.lifecycle.run_original()
            try:
                await stopper
            except asyncio.CancelledError:
                assert repeat_cancel
            assert h.lifecycle._finished.done() and h.lifecycle._stage._worker_exited
            assert not h.lifecycle._stage._thread.sockets
            assert not h.store._original_results and not h.store.db.backend._original_result_holds
        finally:
            permit_reply.set()
            if repeat_cancel:
                await cancellation
            if not stopper.done():
                await stopper


class _CurrentLifecycleProvider(_BoundedQuoteProvider):
    native_environment_lifecycle_api_version = 1

    async def inspect(self, scope, lease):
        async with self.environment_host.inspect_operation(scope, lease) as call:
            return await call.run_original()

    async def release(self, scope, lease):
        async with self.environment_host.release_operation(scope, lease) as call:
            return await call.run_original()


@pytest.mark.asyncio
async def test_runtime_consumes_same_original_release_ack_without_generic_save(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True, activate=False) as h:
        original = h.host
        host = RunpodEnvironmentHost(h.runtime, profile=original._profile, noncompute=original._noncompute,
            capabilities=original._capabilities, account=original._account, api_key=original._api_key,
            base_url=original._base_url, ca_data=original._ca_data, credential_selector=original._credential_selector)
        provider = _CurrentLifecycleProvider(host)
        host._bind_provider(provider)
        h.runtime.providers[h.request.provider_id] = provider
        saves = []
        original_save = h.store.save
        async def observe_save(*args, **kwargs):
            saves.append(True)
            return await original_save(*args, **kwargs)
        monkeypatch.setattr(h.store, "save", observe_save)
        value = await h.runtime.release(h.release_context, h.row.session_id)
        assert value == {"session_id":h.row.session_id,"released":False}
        assert saves == [] and len(quote_peer.requests) == 2
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.remote_environment.release.native_status == 204 and row.lease.state == "unknown"
        with pytest.raises(ResourceOutcomeUnknown, match="physical release unverified"):
            await h.runtime.release(h.release_context, h.row.session_id)
        assert len(quote_peer.requests) == 2, "retained acknowledgement was blindly replayed"


@pytest.mark.asyncio
async def test_native_preentry_refusal_preserves_original_and_cleanup_errors(custody, vendor, quote_peer, monkeypatch):
    import httpx

    from kestrel_sovereign.visual_computer.runpod_lifecycle import (
        _OriginalRunpodNativeStage,
    )
    async with lifecycle(custody, vendor, quote_peer, monkeypatch) as h:
        original, cleanup = PermissionError("owned native preentry refusal"), OSError("owned close acknowledgement loss")
        close = httpx.Client.close
        def refuse(stage, client, operation):
            assert stage is h.lifecycle._stage
            raise original
        def close_then_lose_ack(client):
            close(client)
            if client is h.lifecycle._stage._client:
                raise cleanup
        monkeypatch.setattr(_OriginalRunpodNativeStage, "_require_native_invocation", refuse)
        monkeypatch.setattr(httpx.Client, "close", close_then_lose_ack)
        with pytest.raises(BaseException) as failure:
            await h.lifecycle.run_original()
        pending, found = [failure.value], set()
        while pending:
            error = pending.pop()
            if id(error) in found:
                continue
            found.add(id(error))
            pending.extend(value for value in (error.__cause__, error.__context__) if value is not None)
            pending.extend(getattr(error, "exceptions", ()))
        assert id(original) in found and id(cleanup) in found
        assert failure.value is not cleanup, "cleanup replaced the original refusal"
        assert not quote_peer.requests and not h.lifecycle._stage._worker_entered
        assert not h.lifecycle._stage._worker_exited
        assert h.lifecycle._stage._client.is_closed


@pytest.mark.asyncio
async def test_dispatching_no_id_cannot_get_delete_or_retire(custody, vendor, quote_peer, monkeypatch):
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True, activate=False, known_id=False) as h:
        assert h.row.lease is None
        before = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        with pytest.raises(ResourceOutcomeUnknown, match="cleanup custody"):
            await h.runtime.release(h.release_context, h.row.session_id)
        assert not quote_peer.requests
        after = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert after == before and after.phase not in {"retired", "released"}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 409, 429, 500])
async def test_readback_error_never_means_gone_or_delete_retry(custody, vendor, quote_peer, monkeypatch, status):
    from kestrel_cloud_runpod.models import RunPodAPIError
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True) as h:
        quote_peer.get_status = status
        with pytest.raises(RunPodAPIError):
            await h.lifecycle.run_original()
        assert len(quote_peer.requests) == 1
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.remote_environment.release.state == "intent" and row.lease.state == "unknown"
        assert row.lease.actual_cost_usd is None


@pytest.mark.asyncio
async def test_malformed_optional_cost_retains_known_identity_and_release_ack(custody, vendor, quote_peer, monkeypatch):
    from kestrel_cloud_runpod.models import PodResource, ResourceFieldState

    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True) as h:
        quote_peer.reply["cost"] = "PRIVATE-MALFORMED-COST"
        readbacks = []

        def retain_original_readback():
            if len(quote_peer.requests) == 1:
                readbacks.append(h.lifecycle._stage)

        quote_peer.before_reply = retain_original_readback
        lease = await h.lifecycle.run_original()
        stage, = readbacks
        pod = stage._native_result
        assert type(pod) is PodResource
        assert pod.id == "owned-native-pod" and pod.cost_per_hr is None
        assert pod.field_evidence.invalid_fields == ("cost",)
        assert pod.field_evidence.state("cost") is ResourceFieldState.INVALID
        assert stage._client_closed and stage._worker_exited and not stage._thread.sockets
        assert h.lifecycle._stage._client_closed and h.lifecycle._stage._worker_exited
        assert not h.lifecycle._stage._thread.sockets
        assert len(quote_peer.requests) == 2
        assert all(request[0] == "/v2/pods/owned-native-pod" for request in quote_peer.requests)
        assert lease.state == "unknown" and lease.actual_cost_usd is None and lease.released_at is None
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.lease == lease and row.lease.provider_resource_id == "owned-native-pod"
        assert row.phase == "fenced" and row.remote_environment.release.state == "acknowledged"
        fact = row.remote_environment.release
        assert fact.provider_resource_id == "owned-native-pod" and fact.native_status == 204
        assert fact.acknowledged_at >= fact.requested_at
        assert (await h.store.db.fetchone("SELECT released FROM visual_computer_requests"))[0] == 0


def install_provider(h):
    original = h.host
    host = RunpodEnvironmentHost(h.runtime, profile=original._profile, noncompute=original._noncompute,
        capabilities=original._capabilities, account=original._account, api_key=original._api_key,
        base_url=original._base_url, ca_data=original._ca_data, credential_selector=original._credential_selector)
    provider = _CurrentLifecycleProvider(host)
    host._bind_provider(provider)
    h.runtime.providers[h.request.provider_id] = provider
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("negotiated", [False, True])
async def test_runtime_inspection_precedes_and_refuses_unverified_connect(custody, vendor, quote_peer, monkeypatch, negotiated):
    from kestrel_sdk.resources import ResourceUnsupported
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, activate=False) as h:
        provider = install_provider(h)
        if not negotiated:
            provider.native_environment_lifecycle_api_version = None
        with pytest.raises(ResourceUnsupported, match="unverified" if negotiated else "negotiation"):
            await h.runtime._acquire_open(h.context, h.row, h.check)
        assert len(quote_peer.requests) == (1 if negotiated else 0)
        row = await h.store.get(SCOPE, session_id=h.row.session_id, guard=h.check)
        assert row.connection is None and row.session is None
        assert (row.remote_environment.inspection is not None) is negotiated


@pytest.mark.asyncio
async def test_release_native_wire_is_preceded_by_exact_intent_commit_ack(custody, vendor, quote_peer, monkeypatch):
    import json
    async with lifecycle(custody, vendor, quote_peer, monkeypatch, release=True) as h:
        observed = []
        def at_wire():
            owner = h.lifecycle
            ack, current = owner._ack, owner._mutation
            stored = json.loads(current._plain)
            observed.append((stored["remote_environment"].get("release", {}).get("state"),
                ack is not None and ack.owner._phase == "closed"
                and ack.entry.values[0] == current._expected
                and ack.writer.task.done() and not ack.writer.task.cancelled()))
        quote_peer.before_reply = at_wire
        await h.lifecycle.run_original()
        assert observed == [("intent", True), ("intent", True)]


@pytest.mark.asyncio
async def test_copied_current_lifecycle_ack_is_not_original_native_receipt(custody, vendor, quote_peer, monkeypatch):
    import copy
    async with lifecycle(custody, vendor, quote_peer, monkeypatch) as h:
        lease = await h.lifecycle.run_original()
        actual = h.lifecycle._ack
        h.lifecycle._ack = copy.copy(actual)
        try:
            with pytest.raises(ResourceConflict, match="native ACK"):
                await h.store._accepted_runpod_lifecycle(SCOPE, h.lifecycle, lease)
        finally:
            h.lifecycle._ack = actual


@pytest.mark.asyncio
async def test_quote_reserves_complete_lifecycle_budget_before_create(custody, vendor, quote_peer, monkeypatch):
    from kestrel_sdk.resources import ResourceUnsupported

    from kestrel_sovereign.visual_computer.runpod_quote_storage import (
        _RunpodEnvironmentEnvelope,
    )
    async with quoted_operation(custody, vendor, quote_peer, monkeypatch) as h:
        called = []
        def refuse(projection):
            called.append(projection)
            raise ResourceUnsupported("owned fixed lifecycle reservation refused")
        monkeypatch.setattr(_RunpodEnvironmentEnvelope, "require_lifecycle_reservation", refuse)
        with pytest.raises(ResourceUnsupported, match="reservation refused"):
            await h.call.run_original()
        assert len(called) == 1
        assert [entry[0] for entry in quote_peer.requests] == ["/v2/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=4"]
        row = await h.store.get(SCOPE, session_id=h.record.session_id, guard=h.check)
        assert row.phase == "admitted" and row.remote_environment is None

@pytest.mark.asyncio
async def test_copied_current_root_cannot_close_original(custody, vendor, quote_peer, monkeypatch):
    import copy
    async with (
        lifecycle(custody, vendor, quote_peer, monkeypatch) as h,
        h.store._current_runpod_quote(h.lifecycle) as root,
    ):
        h.lifecycle._mutation = root
        with pytest.raises(ExecutionAuthorityError):
            copy.copy(root).close()
        assert not root._closed.done()
        assert h.store._original_results[root._acceptance] is root
        root.check_current()


@pytest.mark.asyncio
async def test_current_root_cannot_close_before_actual_writer_join(custody, vendor, quote_peer, monkeypatch):
    from kestrel_sovereign.storage import current_runpod_quote as current
    launch = current.create_owned_task
    roots = []
    async with lifecycle(custody, vendor, quote_peer, monkeypatch) as h:
        def before_join(*args, **kwargs):
            task = launch(*args, **kwargs)
            root = h.lifecycle._mutation
            assert kwargs["admission"].task is task
            with pytest.raises(ExecutionAuthorityError):
                root.close()
            assert not root._closed.done()
            roots.append(root)
            return task
        monkeypatch.setattr(current, "create_owned_task", before_join)
        await h.lifecycle.run_original()
        assert len(roots) == 1
        root = roots[0]
        assert root._writer_join.owner is root
        assert root._writer_join.admission is root._writer_admission
        assert root._writer_join.task is root._writer_admission.task
        assert root._closed.done()
