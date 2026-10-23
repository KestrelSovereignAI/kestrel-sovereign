"""Ready runtime publication, signed capability, and fail-closed host startup."""

import asyncio
from dataclasses import replace

import httpx
import pytest
import pytest_asyncio
from kestrel_sdk.security.exceptions import MasterKeyNotConfiguredError

from kestrel_sovereign.a2a.native_card import verify_native_agent_card
from kestrel_sovereign.a2a.native_publication import prepare_hosted_native_publication
from kestrel_sovereign.a2a.native_routes import native_card_path
from kestrel_sovereign.a2a.push_policy import NativePushPolicy
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.storage.db.interface import DatabaseError
from tests.unit.test_a2a_native_host import (
    hosted as hosted,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_host import (
    hosted_recipient_signing as hosted_recipient_signing,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_host import (
    publication as publication,  # noqa: PLC0414 - fixture re-export
)
from tests.unit.test_a2a_native_resident_database import _wait_for_original_recovery
from tests.unit.test_durable_callback_runtime_context import resident_dispatcher
from tests.utils.database_query_faults import prepare_transient_database_query_fault
from tests.utils.native_push import configure_push_test_key
from tests.utils.outbound_tls import (
    outbound_tls as outbound_tls,  # noqa: PLC0414 - actual HTTPS fixture
)


@pytest_asyncio.fixture
async def push_host(publication, monkeypatch):
    p = publication
    configure_push_test_key(monkeypatch)
    runtime, _ = resident_dispatcher()
    runtime._execution_custody = (
        None  # actual standalone runtime, no synthetic DB fence
    )
    runtime.did = p.recipient.did
    logs = SignalLogStore(p.manager.task_store.backend)
    await logs.initialize()
    dispatcher = SignalDispatcher(
        agent=runtime,
        registry=SourceRegistry(),
        lock_manager=OrderedLockManager(),
        store=logs,
    )
    await dispatcher.initialize_durable_delivery()
    p.recipient.dispatcher = dispatcher
    p.recipient._track_runtime_task = runtime._track_runtime_task
    p.runtime = runtime
    try:
        yield p
    finally:
        if p.manager.push_service is not None:
            await p.manager.push_service.close()
        await dispatcher.shutdown_durable_delivery()


async def prepare(p):
    return await prepare_hosted_native_publication(
        p.app,
        p.host,
        "registered-agent",
        p.recipient,
        {
            "KESTREL_BASE_URL": "https://native.test",
            "KESTREL_A2A_PUSH_TIMEOUT_SECONDS": "9",
        },
    )


async def test_host_binds_original_ready_service_before_signed_capability(push_host):
    p = push_host
    native = await prepare(p)
    service = p.manager.push_service
    assert service is not None and service.require_ready() is service.worker
    assert service.task in p.runtime._background_tasks
    assert service.worker.outbox.signals is p.recipient.dispatcher.durable_store
    assert service.worker.transport.timeout == 9
    assert p.app.routes == []
    p.recipient._native_a2a_publication = native
    async with p.host.a2a_lifecycle_lease():
        p.host._register_agent("registered-agent", p.recipient)
    try:
        card = await native.get_card()
        assert card.capabilities.push_notifications
        verify_native_agent_card(
            card,
            signing_did=p.recipient.identity.signing_did,
            trusted_document={
                "id": p.recipient.identity.signing_did,
                "verificationMethod": p.recipient.identity.new_verification_methods,
            },
        )
        await service.close()
        with pytest.raises(RuntimeError):
            await native.get_card()
    finally:
        async with p.host.a2a_lifecycle_lease():
            native.withdraw()


@pytest_asyncio.fixture
async def callback_host(push_host, monkeypatch):
    from kestrel_sovereign.a2a.push_worker import NativePushWorker

    p = push_host
    entered, release = asyncio.Event(), asyncio.Event()
    original_iteration = NativePushWorker.run_once

    async def paused_iteration(self):
        entered.set()
        await release.wait()
        return await original_iteration(self)

    monkeypatch.setattr(NativePushWorker, "run_once", paused_iteration)
    native = await prepare_hosted_native_publication(
        p.app, p.host, "registered-agent", p.recipient,
        {"KESTREL_BASE_URL": "https://native.test", "KESTREL_A2A_PUSH_RENEWAL_SECONDS": "1",
         "KESTREL_A2A_PUSH_LEASE_SECONDS": "4", "KESTREL_A2A_PUSH_RETRY_SECONDS": "1"},
    )
    service = p.manager.push_service
    await entered.wait()
    p.recipient._native_a2a_publication = native
    async with p.host.a2a_lifecycle_lease():
        p.host._register_agent("registered-agent", p.recipient)
    p.callback_publication = native
    p.push_worker = service.worker
    p.push_iteration = lambda: original_iteration(service.worker)
    p.release_push_iteration = release
    try:
        yield p
    finally:
        async with p.host.a2a_lifecycle_lease():
            native.withdraw()


async def register_callback(p, url):
    from kestrel_sovereign._vendor.a2a.types import TaskPushNotificationConfig
    from tests.utils.native_a2a_security import PRINCIPAL, send

    task = await p.manager.create_task(send(p), creator_agent_id=PRINCIPAL)
    await p.push_worker.outbox.register(
        TaskPushNotificationConfig(task_id=task.id, url=url, tenant=p.scope.tenant),
        creator_id=PRINCIPAL,
    )
    return task


async def revoke_callback_creator(p):
    from tests.utils.native_a2a_security import PRINCIPAL

    async with p.host.a2a_lifecycle_lease():
        await p.backend.execute(
            "DELETE FROM native_test_principals WHERE principal = ?", (PRINCIPAL,)
        )


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 3}], indirect=True)
async def test_hosted_callback_rechecks_retained_creator_policy(callback_host, outbound_tls, monkeypatch):
    p = callback_host
    task = await register_callback(p, outbound_tls.url + "/callback")
    assert await p.push_iteration() == 1
    assert len(outbound_tls.posts) == 1
    await revoke_callback_creator(p)
    await p.manager.complete_task(
        task.id, "Private result after policy revocation",
        recipient_agent_id=p.recipient.did,
    )
    reads = []
    original = p.manager.task_store.get_for_creator

    async def read(*args, **kwargs):
        reads.append(args)
        return await original(*args, **kwargs)

    monkeypatch.setattr(p.manager.task_store, "get_for_creator", read)
    assert await p.push_iteration() == 1
    assert not reads, "Revoked creator caused a new task-payload read"
    assert len(outbound_tls.posts) == 1, "Revoked creator received a later task snapshot"
    assert len(outbound_tls.dns) == 2
    assert await p.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(1,)]
    assert not p.manager.push_service.task.done()


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 1}], indirect=True)
async def test_callback_rechecks_policy_after_task_read_before_dns(
    callback_host, outbound_tls, monkeypatch
):
    p = callback_host
    await register_callback(p, outbound_tls.url + "/callback")
    original = p.manager.task_store.get_for_creator

    async def read(*args, **kwargs):
        native = await original(*args, **kwargs)
        await revoke_callback_creator(p)
        return native

    monkeypatch.setattr(p.manager.task_store, "get_for_creator", read)
    assert await p.push_iteration() == 1
    assert not outbound_tls.posts and len(outbound_tls.dns) == 1
    assert await p.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == []


@pytest.mark.parametrize("fault", ["policy", "identity", "registration", "publication"])
@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 1}], indirect=True)
async def test_callbacks_refuse_a_retired_publication_before_dns(
    callback_host, outbound_tls, fault
):
    from types import SimpleNamespace

    p = callback_host
    await register_callback(p, outbound_tls.url + "/callback")
    async with p.host.a2a_lifecycle_lease():
        if fault == "policy":
            p.host.install_a2a_hosted_policy(
                p.recipient, resolver=p.policy.resolver, authorizer=p.policy.authorizer,
                router=p.policy.router, requester=p.policy.requester,
            )
        elif fault == "identity":
            p.recipient.identity = replace(p.recipient.identity, new_verification_methods=[])
        elif fault == "registration":
            p.host._agents["registered-agent"] = SimpleNamespace(**vars(p.recipient))
        else:
            p.callback_publication.withdraw()
    assert await p.push_iteration() == 1
    assert not outbound_tls.posts and len(outbound_tls.dns) == 1
    assert await p.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == []


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_callback_rechecks_policy_after_real_dns_before_http(
    callback_host, outbound_tls, monkeypatch
):
    from kestrel_sovereign.a2a import push_transport

    p = callback_host
    await register_callback(p, outbound_tls.url + "/callback")
    original = push_transport.assert_safe_url
    original_renew = p.push_worker._renew
    original_send = p.push_worker.transport.send
    monitor_renewed, http_retired = asyncio.Event(), asyncio.Event()
    dns_returning = False

    async def renew(delivery):
        await original_renew(delivery)
        if not dns_returning:
            # Hold the monitor after its actual first lease renewal. The
            # final HTTP admission must independently observe the revocation.
            monitor_renewed.set()
            await http_retired.wait()

    async def send(*args, **kwargs):
        try:
            return await original_send(*args, **kwargs)
        finally:
            http_retired.set()

    async def resolve(*args, **kwargs):
        nonlocal dns_returning
        result = await original(*args, **kwargs)
        await asyncio.wait_for(monitor_renewed.wait(), 5)
        await revoke_callback_creator(p)
        dns_returning = True
        return result

    monkeypatch.setattr(p.push_worker, "_renew", renew)
    monkeypatch.setattr(p.push_worker.transport, "send", send)
    monkeypatch.setattr(push_transport, "assert_safe_url", resolve)
    try:
        assert await p.push_iteration() == 1
    finally:
        http_retired.set()
    assert not outbound_tls.posts and len(outbound_tls.dns) == 2
    assert await p.backend.fetch_all("SELECT retired FROM a2a_push_delivery_attempts") == [(1,)]


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_policy_revocation_joins_an_already_admitted_http_owner(callback_host, outbound_tls):
    p = callback_host
    await register_callback(p, outbound_tls.url + "/callback")
    outbound_tls.reply.clear()
    iteration = asyncio.create_task(p.push_iteration())
    try:
        async with asyncio.timeout(5):
            while not outbound_tls.posts:
                await asyncio.sleep(0.01)
        await revoke_callback_creator(p)
        assert await asyncio.wait_for(iteration, 5) == 1
        assert len(outbound_tls.posts) == 1
        assert await p.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == [(1,)]
    finally:
        outbound_tls.reply.set()
        if not iteration.done():
            iteration.cancel()
        await asyncio.gather(iteration, return_exceptions=True)


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_native_policy_outage_retries_on_the_original_resident(
    callback_host, outbound_tls, monkeypatch
):
    p = callback_host
    fail_query = await prepare_transient_database_query_fault(p.backend)
    with pytest.raises(DatabaseError) as failed:
        await fail_query()
    original_error = failed.value
    original = p.policy.router.authorize_inbound_sender
    recovered = asyncio.Event()
    errors = []

    async def authorize(*args):
        if not errors:
            errors.append(original_error)
            raise original_error
        result = await original(*args)
        recovered.set()
        return result

    await register_callback(p, outbound_tls.url + "/callback")
    monkeypatch.setattr(p.policy.router, "authorize_inbound_sender", authorize)
    owner = p.manager.push_service.task
    p.release_push_iteration.set()
    await _wait_for_original_recovery(owner, recovered)
    async with asyncio.timeout(10):
        while not outbound_tls.posts:
            assert not owner.done()
            await asyncio.sleep(0.01)
    assert len(errors) == 1 and len(outbound_tls.posts) == 1
    assert p.manager.push_service.task is owner and not owner.done()


async def test_callback_policy_refusal_retains_original_execution_control(callback_host):
    from starlette.exceptions import HTTPException

    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from tests.utils.native_a2a_security import PRINCIPAL

    p = callback_host
    original = HTTPException(status_code=403, detail="Original policy operation retired")
    original.__cause__ = ExecutionAuthorityError("Original authority retired")

    async def authorize(principal):
        assert principal == PRINCIPAL
        raise original

    native = p.callback_publication
    guard = native._guard
    authority = replace(guard.authority_factory(), authorize_retained_principal=authorize)
    native._guard = replace(guard, authority_factory=lambda: authority)
    with pytest.raises(HTTPException) as failed:
        await native._authorize_push_creator(PRINCIPAL)
    assert failed.value is original


async def test_host_preparation_waits_for_actual_runtime_ready_boundary(
    push_host, monkeypatch
):
    p = push_host
    p.runtime._runtime_publication_ready.clear()
    accepted = asyncio.Event()
    original = p.recipient._track_runtime_task

    def publish(operation, *, name, admission):
        task = original(operation, name=name, admission=admission)
        accepted.set()
        return task

    monkeypatch.setattr(p.recipient, "_track_runtime_task", publish)
    preparing = asyncio.create_task(prepare(p))
    try:
        await accepted.wait()
        assert not preparing.done() and p.app.routes == []
        assert p.manager.push_service is None
        p.runtime._runtime_publication_ready.set()
        await preparing
        assert p.manager.push_service.require_ready()
    finally:
        p.runtime._runtime_publication_ready.set()
        await asyncio.gather(preparing, return_exceptions=True)


async def test_transient_journal_read_preserves_original_signed_publication(
    push_host, monkeypatch
):
    p = push_host
    backend = p.manager.task_store.backend
    fail_query = await prepare_transient_database_query_fault(backend)
    with pytest.raises(DatabaseError) as failed:
        await fail_query()
    original_error = failed.value
    native = await prepare_hosted_native_publication(
        p.app,
        p.host,
        "registered-agent",
        p.recipient,
        {
            "KESTREL_BASE_URL": "https://native.test",
            "KESTREL_A2A_PUSH_RETRY_SECONDS": "1",
        },
    )
    p.recipient._native_a2a_publication = native
    async with p.host.a2a_lifecycle_lease():
        p.host._register_agent("registered-agent", p.recipient)
    service = p.manager.push_service
    owner = service.task
    original_pending = service.worker.outbox.changes.pending
    failures = []
    recovered = asyncio.Event()

    async def pending(*, limit):
        if not failures:
            failures.append(original_error)
            raise original_error
        result = await original_pending(limit=limit)
        recovered.set()
        return result

    monkeypatch.setattr(service.worker.outbox.changes, "pending", pending)
    try:
        await _wait_for_original_recovery(owner, recovered)
        assert service.task is owner and len(failures) == 1
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=p.app), base_url="https://native.test"
        ) as http:
            response = await http.get(native_card_path("registered-agent"))
        assert response.status_code == 200
        from kestrel_sovereign._vendor.a2a.types import AgentCard
        from google.protobuf.json_format import ParseDict

        card = ParseDict(response.json(), AgentCard())
        assert card.capabilities.push_notifications
        verify_native_agent_card(
            card,
            signing_did=p.recipient.identity.signing_did,
            trusted_document={
                "id": p.recipient.identity.signing_did,
                "verificationMethod": p.recipient.identity.new_verification_methods,
            },
        )
    finally:
        async with p.host.a2a_lifecycle_lease():
            native.withdraw()


async def test_missing_encryption_refuses_host_publication_before_push_schema(
    push_host, monkeypatch
):
    from kestrel_sovereign.a2a.stores.unified import push_config_store

    p = push_host
    error = MasterKeyNotConfiguredError("original host key unavailable")

    def refused_key(*args):
        raise error

    monkeypatch.setattr(push_config_store, "get_agent_key", refused_key)
    with pytest.raises(MasterKeyNotConfiguredError) as caught:
        await prepare(p)
    assert caught.value is error
    assert p.manager.push_service is None and p.app.routes == []
    assert not await p.manager.task_store.backend.table_exists("a2a_push_configs")
    assert not p.runtime._background_tasks


async def test_publication_cannot_advertise_an_unprepared_push_policy(push_host):
    p = push_host
    p.publication.configuration = replace(
        p.publication.configuration, push=NativePushPolicy()
    )
    with pytest.raises(RuntimeError, match="differs from its bound service"):
        p.publication._build_card(p.publication._profile)
    assert p.manager.push_service is None and p.app.routes == []
