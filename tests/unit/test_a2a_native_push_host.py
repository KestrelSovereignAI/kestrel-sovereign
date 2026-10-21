"""Ready runtime publication, signed capability, and fail-closed host startup."""

import asyncio
from dataclasses import replace

import pytest
import pytest_asyncio
from kestrel_sdk.security.exceptions import MasterKeyNotConfiguredError

from kestrel_sovereign.a2a.native_card import verify_native_agent_card
from kestrel_sovereign.a2a.native_publication import prepare_hosted_native_publication
from kestrel_sovereign.a2a.push_policy import NativePushPolicy
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from tests.unit.test_a2a_native_host import (
    hosted as hosted,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_host import (
    hosted_recipient_signing as hosted_recipient_signing,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_host import (
    publication as publication,  # noqa: PLC0414 - fixture re-export
)
from tests.unit.test_durable_callback_runtime_context import resident_dispatcher
from tests.utils.native_push import configure_push_test_key


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
