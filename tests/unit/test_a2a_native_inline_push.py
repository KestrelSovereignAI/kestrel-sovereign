"""Inline native callbacks commit with task admission before cognition wakes."""

import asyncio

import pytest
from a2a.types import SendMessageRequest
from a2a.utils.errors import PushNotificationNotSupportedError

from kestrel_sovereign._async_ownership import create_owned_task
from kestrel_sovereign.a2a.native_input import native_request_digest
from kestrel_sovereign.a2a.push_service import NativePushService, PushServiceUnavailable
from kestrel_sovereign.a2a.stores.unified.task_store import TaskInputConflictError
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.unit.test_a2a_native_input import request as inline_request
from tests.unit.test_a2a_native_push_outbox import refuse_original_database_write
from tests.unit.test_a2a_native_push_worker import worker
from tests.unit.test_a2a_native_shutdown import STORE_NAMES
from tests.unit.test_a2a_native_task_manager import (
    manager as manager,  # noqa: PLC0414 - pytest fixture re-export
)
from tests.unit.test_a2a_native_task_manager import request
from tests.utils.native_a2a import manager_for
from tests.utils.native_push import managed_push_service
from tests.utils.outbound_tls import (
    outbound_close_barrier as outbound_close_barrier,  # noqa: PLC0414 - pytest fixture re-export
)
from tests.utils.outbound_tls import (
    outbound_tls as outbound_tls,  # noqa: PLC0414 - pytest fixture re-export
)


def inline(task=None, *, config_id="callback"):
    value = inline_request()
    value.configuration.task_push_notification_config.url = (
        "https://1.1.1.1/callback?secret=do-not-persist"
    )
    value.configuration.task_push_notification_config.id = config_id
    if task is not None:
        value.message.task_id = task.id
        value.message.context_id = task.context_id
    return value


async def test_inline_callback_requires_an_explicit_ready_managed_service(manager):
    with pytest.raises(PushNotificationNotSupportedError):
        await manager.create_task(inline(), creator_agent_id="did:creator")
    assert await manager.task_store.backend.fetch_one(
        "SELECT COUNT(*) FROM a2a_tasks"
    ) == (0,)
    assert await manager.task_store.backend.fetch_one(
        "SELECT COUNT(*) FROM a2a_contexts"
    ) == (0,)


async def test_initial_inline_config_is_committed_and_encrypted_before_the_wake(
    manager, monkeypatch, native_a2a_backend_factory
):
    second = native_a2a_backend_factory()
    await second.connect()
    observations, observers, wakes = [], [], []

    async def observe(snapshot):
        observations.append(
            await second.fetch_one(
                "SELECT COUNT(*) FROM a2a_push_configs AS c JOIN a2a_push_subscriptions AS s "
                "ON c.task_id = s.task_id AND c.generation = s.generation "
                "WHERE c.task_id = ? AND c.state = 'active'",
                (snapshot.task.id,),
            )
        )

    def wake(snapshot):
        wakes.append(snapshot)
        observers.append(asyncio.create_task(observe(snapshot)))

    manager._on_task_submitted = wake
    try:
        async with managed_push_service(
            manager, monkeypatch, pause_delivery=True
        ) as service:
            original = inline()
            before = original.SerializeToString()
            task = await manager.create_task(original, creator_agent_id="did:creator")
            await asyncio.gather(*observers)
            assert original.SerializeToString() == before
            assert observations == [(1,)] and len(wakes) == 1
            assert wakes[0].input_wire_request_digest == native_request_digest(
                original, "SendMessage"
            )
            assert not wakes[0].input_request.configuration.HasField(
                "task_push_notification_config"
            )
            config = await service.worker.outbox.configurations.get(
                task.id, "callback", creator_id="did:creator"
            )
            assert (
                config.config.authentication.credentials
                == "callback-authorization-secret"
            )
            stored = await manager.task_store.backend.fetch_one(
                "SELECT request_json FROM a2a_task_inputs WHERE task_id = ?", (task.id,)
            )
            assert (
                "callback-authorization-secret" not in stored[0]
                and "1.1.1.1" not in stored[0]
            )
    finally:
        await asyncio.gather(*observers, return_exceptions=True)
        await second.close()


async def test_real_registration_refusal_rolls_back_task_context_input_and_wake(
    manager, monkeypatch
):
    wakes = []
    manager._on_task_submitted = wakes.append
    async with managed_push_service(manager, monkeypatch, pause_delivery=True):
        backend = manager.task_store.backend
        await refuse_original_database_write(
            backend, table="durable_signal_consumers", event="INSERT"
        )
        with pytest.raises(TransactionError):
            await manager.create_task(inline(), creator_agent_id="did:creator")
        for table in (
            "a2a_tasks",
            "a2a_contexts",
            "a2a_task_inputs",
            "a2a_task_changes",
            "a2a_push_configs",
            "a2a_push_subscriptions",
            "durable_signal_consumers",
        ):
            assert await backend.fetch_one(f"SELECT COUNT(*) FROM {table}") == (0,)
        assert wakes == []


async def test_inline_continuation_registers_once_and_binds_the_complete_wire_replay(
    manager, monkeypatch
):
    wakes = []
    manager._on_task_submitted = wakes.append
    async with managed_push_service(
        manager, monkeypatch, pause_delivery=True
    ) as service:
        task = await manager.create_task(request(), creator_agent_id="did:creator")
        wakes.clear()
        original = inline(task)
        await manager.continue_task(original, creator_agent_id="did:creator")
        replay = SendMessageRequest.FromString(original.SerializeToString())
        await manager.continue_task(replay, creator_agent_id="did:creator")
        assert len(wakes) == 1
        assert wakes[0].input_revision == 1
        assert wakes[0].input_wire_request_digest == native_request_digest(
            original, "SendMessage"
        )
        assert not wakes[0].input_request.configuration.HasField(
            "task_push_notification_config"
        )
        assert len(await service.worker.outbox.subscriptions()) == 1
        replay.configuration.task_push_notification_config.authentication.credentials = "changed-secret"
        with pytest.raises(TaskInputConflictError, match="Message ID"):
            await manager.continue_task(replay, creator_agent_id="did:creator")
        assert len(wakes) == 1 and len(await service.worker.outbox.subscriptions()) == 1


async def test_continuation_registration_refusal_rolls_back_input_revision_and_wake(
    manager, monkeypatch
):
    wakes = []
    manager._on_task_submitted = wakes.append
    async with managed_push_service(manager, monkeypatch, pause_delivery=True):
        task = await manager.create_task(request(), creator_agent_id="did:creator")
        wakes.clear()
        backend = manager.task_store.backend
        before = await manager.get_task_snapshot(
            task.id, recipient_agent_id=manager.did
        )
        changes = await manager.task_store.changes.pending(limit=10)
        await refuse_original_database_write(
            backend, table="durable_signal_consumers", event="INSERT"
        )
        with pytest.raises(TransactionError):
            await manager.continue_task(inline(task), creator_agent_id="did:creator")
        after = await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
        assert after.input_revision == before.input_revision == 0
        assert after.task == before.task and after.input_request == before.input_request
        assert await manager.task_store.changes.pending(limit=10) == changes
        assert await backend.fetch_one("SELECT COUNT(*) FROM a2a_task_inputs") == (1,)
        assert await backend.fetch_one("SELECT COUNT(*) FROM a2a_push_configs") == (0,)
        assert wakes == []


async def test_cancelled_inline_validation_has_no_task_admission_side_effects(
    manager, monkeypatch
):
    from kestrel_sovereign.a2a import push_transport

    entered, release = asyncio.Event(), asyncio.Event()
    validate = push_transport.assert_safe_url

    async def validation(*args, **kwargs):
        entered.set()
        await release.wait()
        return await validate(*args, **kwargs)

    async with managed_push_service(manager, monkeypatch, pause_delivery=True):
        monkeypatch.setattr(push_transport, "assert_safe_url", validation)
        creating = asyncio.create_task(
            manager.create_task(inline(), creator_agent_id="did:creator")
        )
        try:
            await entered.wait()
            creating.cancel("original inline validation withdrawal")
            with pytest.raises(asyncio.CancelledError) as caught:
                await creating
            assert caught.value.args == ("original inline validation withdrawal",)
            assert await manager.task_store.backend.fetch_one(
                "SELECT COUNT(*) FROM a2a_tasks"
            ) == (0,)
            assert await manager.task_store.backend.fetch_one(
                "SELECT COUNT(*) FROM a2a_push_configs"
            ) == (0,)
        finally:
            release.set()
            await asyncio.gather(creating, return_exceptions=True)


async def test_service_binding_requires_the_exact_canonical_task_store(
    manager, monkeypatch
):
    other = manager_for(
        manager.task_store.backend, manager.task_store.scope, owns_backend=False
    )
    await other.initialize()
    try:
        with pytest.raises(TypeError):
            manager.bind_push_service(object())
        async with managed_push_service(
            other, monkeypatch, pause_delivery=True
        ) as service:
            with pytest.raises(ValueError, match="another canonical task store"):
                manager.bind_push_service(service)
            assert manager.push_service is None
    finally:
        await other.close()


async def test_binding_cannot_replace_the_published_managed_service(
    manager, monkeypatch
):
    async with managed_push_service(
        manager, monkeypatch, pause_delivery=True
    ) as service:
        manager.bind_push_service(service)
        replacement_worker = worker(service.worker.outbox)
        # The replacement is a real ready resident, but has no subscriptions
        # or effects to execute. It must not replace the published generation.
        replacement = NativePushService(
            replacement_worker, task_factory=create_owned_task
        )
        try:
            await replacement.start()
            with pytest.raises(ValueError, match="cannot replace"):
                manager.bind_push_service(replacement)
            assert (
                manager.push_service is service
                and service.require_ready() is service.worker
            )
        finally:
            await replacement.close()


async def test_stopped_resident_is_service_unavailability_not_caller_withdrawal(
    manager, monkeypatch
):
    async with managed_push_service(
        manager, monkeypatch, pause_delivery=True
    ) as service:
        await service.close()
        with pytest.raises(PushServiceUnavailable) as caught:
            await manager.create_task(inline(), creator_agent_id="did:creator")
        assert isinstance(caught.value.__cause__, asyncio.CancelledError)
        assert await manager.task_store.backend.fetch_one(
            "SELECT COUNT(*) FROM a2a_tasks"
        ) == (0,)


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_manager_close_retires_actual_https_before_any_primary_store_close(
    manager,
    monkeypatch,
    native_a2a_backend_factory,
    outbound_tls,
    outbound_close_barrier,
):
    second = native_a2a_backend_factory()
    await second.connect()
    calls = []
    closing = None
    try:
        async with managed_push_service(manager, monkeypatch, pause_delivery=False):
            original = inline()
            original.configuration.task_push_notification_config.url = (
                outbound_tls.url + "/callback"
            )
            task = await manager.create_task(original, creator_agent_id="did:creator")
            await outbound_close_barrier.entered.wait()
            for name in STORE_NAMES:
                store = getattr(manager, name)
                if store is None:
                    continue
                native_close = store.close

                async def observed_close(close=native_close, label=name):
                    assert outbound_close_barrier.release.is_set(), (
                        "Primary store close preceded original HTTP retirement"
                    )
                    calls.append(label)
                    return await close()

                monkeypatch.setattr(store, "close", observed_close)
            closing = asyncio.create_task(manager.close())
            await asyncio.sleep(0)
            assert not closing.done() and calls == []
            assert await manager.task_store.backend.fetch_one("SELECT 1") == (1,)
            outbound_close_barrier.release.set()
            await closing
            assert calls
            assert await second.fetch_one(
                "SELECT retired FROM a2a_push_delivery_attempts WHERE task_id = ?",
                (task.id,),
            ) == (1,)
    finally:
        outbound_close_barrier.release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await second.close()
