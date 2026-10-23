"""Official repeated inline callbacks and immutable generation replacement."""

import asyncio

import pytest
from kestrel_sovereign._vendor.a2a.client import ClientConfig, ClientFactory
from kestrel_sovereign._vendor.a2a.types import GetTaskPushNotificationConfigRequest

from kestrel_sovereign.a2a.push_outbox import NativePushOutbox
from kestrel_sovereign.a2a.stores.unified.push_config_store import (
    PushConfigConflict,
    PushConfigStore,
)
from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.unit.test_a2a_native_http import (
    wire as wire,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_inline_push import inline
from tests.unit.test_a2a_native_push_config_store import admit, config
from tests.unit.test_a2a_native_push_config_store import (
    push_store as push_store,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_push_http import callback as public_callback
from tests.unit.test_a2a_native_push_http import (
    native_push_runtime as native_push_runtime,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_push_outbox import (
    callback,
    refuse_original_database_write,
)
from tests.unit.test_a2a_native_push_outbox import (
    outbox as outbox,  # noqa: PLC0414 - fixture dependency
)
from tests.unit.test_a2a_native_push_worker import worker
from tests.unit.test_a2a_native_task_manager import (
    manager as manager,  # noqa: PLC0414 - fixture dependency
)
from tests.utils.native_a2a_security import PRINCIPAL, send
from tests.utils.native_push import managed_push_service
from tests.utils.outbound_tls import (
    outbound_close_barrier as outbound_close_barrier,  # noqa: PLC0414 - fixture dependency
)
from tests.utils.outbound_tls import (
    outbound_tls as outbound_tls,  # noqa: PLC0414 - fixture dependency
)


@pytest.mark.parametrize("config_id", ["", "callback"])
async def test_official_client_can_repeat_its_inline_callback_on_each_new_input(wire, config_id):
    client = ClientFactory(
        ClientConfig(
            httpx_client=wire.http,
            streaming=False,
            polling=True,
            supported_protocol_bindings=[wire.protocol],
            push_notification_config=public_callback(wire, "", config_id),
        )
    ).create(wire.card, interceptors=[wire.interceptor])
    task = None
    for index in range(3):
        params = send(wire.admission)
        params.message.parts[0].text = f"Input {index}"
        if task is not None:
            params.message.task_id = task.id
            params.message.context_id = task.context_id
        events = [event async for event in client.send_message(params)]
        assert len(events) == 1 and events[0].HasField("task")
        task = events[0].task
    snapshot = await wire.manager.get_task_snapshot(
        task.id, recipient_agent_id=wire.admission.scope.recipient_id
    )
    assert snapshot.input_revision == 2
    assert len(wire.submissions) == 3
    outbox = wire.manager.push_service.worker.outbox
    configs = await outbox.configurations.list(
        task.id, creator_id=PRINCIPAL, limit=10,
    )
    assert len(configs) == 1
    assert configs[0].config.id == (config_id or task.id)
    assert len(await outbox.subscriptions()) == 1
    assert await outbox.backend.fetch_val("SELECT COUNT(*) FROM durable_signal_consumers") == 1


async def test_identical_registration_has_one_snapshot_and_one_original_consumer(outbox):
    value, task = outbox
    first = await value.register(callback(task), creator_id="did:creator")
    before = await value.changes.pending(limit=10)
    repeated = await value.register(callback(task), creator_id="did:creator")
    assert repeated == first
    assert await value.changes.pending(limit=10) == before
    assert len(await value.subscriptions()) == 1
    assert await value.backend.fetch_val("SELECT COUNT(*) FROM durable_signal_consumers") == 1


async def test_repeated_registration_refuses_a_foreign_original_subscriber(outbox):
    value, task = outbox
    original = await value.register(callback(task), creator_id="did:creator")
    await value.backend.execute("UPDATE a2a_push_subscriptions SET creator_id = 'did:impostor'")
    with pytest.raises(PushConfigConflict, match="original subscription"):
        await value.register(callback(task), creator_id="did:creator")
    assert await value.configurations.get(task.id, "callback", creator_id="did:creator") == original
    assert await value.backend.fetch_val("SELECT creator_id FROM a2a_push_subscriptions") == "did:impostor"


async def test_public_create_replaces_a_logical_id_and_keeps_only_its_current_value(wire):
    from tests.unit.test_a2a_native_http import submit

    task = await submit(wire)
    client = wire.client()
    value = public_callback(wire, task.id, "callback")
    first = await client.create_task_push_notification_config(value)
    assert await client.create_task_push_notification_config(value) == first
    value.token = "replacement-token"
    assert await client.create_task_push_notification_config(value) == value
    assert await client.get_task_push_notification_config(
        GetTaskPushNotificationConfigRequest(
            tenant=wire.admission.scope.tenant, task_id=task.id, id=value.id
        )
    ) == value
    outbox = wire.manager.push_service.worker.outbox
    assert len(await outbox.subscriptions()) == 1
    assert await outbox.backend.fetch_val("SELECT COUNT(*) FROM a2a_push_configs") == 2
    active = await outbox.backend.fetch_all("SELECT active FROM durable_signal_consumers")
    assert sorted(bool(row[0]) for row in active) == [False, True]


async def test_two_independent_identical_setters_share_one_generation_and_snapshot(
    outbox, native_a2a_backend_factory
):
    value, task = outbox
    other = native_a2a_backend_factory()
    await other.connect()
    try:
        independent = NativePushOutbox(
            PushConfigStore(TaskStore(other, scope=value.configurations.scope, owns_backend=False)),
            DurableSignalStore(other), retention_days=30, lease_seconds=60,
        )
        await independent.initialize()
        first, second = await asyncio.gather(
            value.register(callback(task), creator_id="did:creator"),
            independent.register(callback(task), creator_id="did:creator"),
        )
        assert first == second
        assert len(await value.subscriptions()) == 1
        assert len(await value.changes.pending(limit=10)) == 1
        assert await value.backend.fetch_val("SELECT COUNT(*) FROM durable_signal_consumers") == 1
    finally:
        await other.close()


async def test_postgres_direct_setter_waits_for_the_original_task_boundary(
    push_store, native_a2a_backend_factory, monkeypatch
):
    store, task = push_store
    if not store.is_postgres:
        pytest.skip("PostgreSQL statement visibility under a real row-lock wait")
    other = native_a2a_backend_factory()
    await other.connect()
    independent = PushConfigStore(TaskStore(other, scope=store.scope, owns_backend=False))
    staged, release, starting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_get = store.get
    statement_pid = None
    first = second = None

    async def staged_get(*args, **kwargs):
        result = await original_get(*args, **kwargs)
        staged.set()
        await release.wait()
        return result

    async def set_other():
        nonlocal statement_pid
        async with other.transaction():
            statement_pid = await other.fetch_val("SELECT pg_backend_pid()")
            starting.set()
            return await independent.create(config(task), creator_id="did:creator")

    monkeypatch.setattr(store, "get", staged_get)
    try:
        first = asyncio.create_task(store.create(config(task), creator_id="did:creator"))
        await staged.wait()
        second = asyncio.create_task(set_other())
        await starting.wait()
        async with asyncio.timeout(5):
            while True:
                waiting = await store.backend.fetch_val(
                    "SELECT wait_event_type FROM pg_stat_activity WHERE pid = ?", (statement_pid,)
                )
                if waiting == "Lock":
                    break
                assert not second.done(), "direct setter bypassed the original task lock"
                await asyncio.sleep(0.01)
        release.set()
        one, two = await asyncio.gather(first, second)
        assert one == two
        assert await store.backend.fetch_val("SELECT COUNT(*) FROM a2a_push_configs") == 1
    finally:
        release.set()
        await asyncio.gather(*(operation for operation in (first, second) if operation), return_exceptions=True)
        await other.close()


async def test_failed_continuation_replacement_restores_the_original_input_and_lease(
    manager, monkeypatch
):
    wakes = []
    manager._on_task_submitted = wakes.append
    async with managed_push_service(manager, monkeypatch, pause_delivery=True) as service:
        task = await manager.create_task(inline(), creator_agent_id="did:creator")
        outbox = service.worker.outbox
        await outbox.forward_pending(limit=10)
        original = await outbox.configurations.get(task.id, "callback", creator_id="did:creator")
        subscription, = await outbox.subscriptions()
        delivery = await outbox.signals.claim_delivery(
            agent_id=outbox.agent_id, consumer_id=subscription.consumer_id, executor_id="original-runtime"
        )
        snapshot = await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
        changes = await outbox.changes.pending(limit=10)
        lease_query = "SELECT status, lease_owner, lease_token FROM durable_signal_deliveries"
        lease = await outbox.backend.fetch_all(lease_query)
        assert delivery.lease_token == lease[0][2]
        await refuse_original_database_write(outbox.backend, table="durable_signal_consumers", event="INSERT")
        params = inline(task)
        params.configuration.task_push_notification_config.token = "replacement-token"
        with pytest.raises(TransactionError):
            await manager.continue_task(params, creator_agent_id="did:creator")
        after = await manager.get_task_snapshot(task.id, recipient_agent_id=manager.did)
        assert after.input_revision == snapshot.input_revision == 0
        assert after.task == snapshot.task
        assert await outbox.configurations.get(task.id, "callback", creator_id="did:creator") == original
        assert await outbox.backend.fetch_all(lease_query) == lease
        assert await outbox.signals.has_active_consumer(outbox.agent_id, subscription.consumer_id)
        assert await outbox.changes.pending(limit=10) == changes
        assert await outbox.backend.fetch_val("SELECT COUNT(*) FROM a2a_push_configs") == 1
        assert len(wakes) == 1


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 10}], indirect=True)
async def test_delete_joins_replaced_http_generation_and_preserves_a_later_setter(
    outbox, outbound_tls, outbound_close_barrier, monkeypatch
):
    value, task = outbox
    native = callback(task)
    native.url = outbound_tls.url + "/original"
    first = await value.register(native, creator_id="did:creator")
    runner = worker(value)
    entered, release = outbound_close_barrier.entered, outbound_close_barrier.release
    sending = asyncio.create_task(runner.run_once())
    witness = asyncio.create_task(entered.wait())
    deleting = None
    revoked = asyncio.Event()
    original_revoke = value.revoke

    async def revoke(*args, **kwargs):
        inventory = await original_revoke(*args, **kwargs)
        revoked.set()
        return inventory

    monkeypatch.setattr(value, "revoke", revoke)
    try:
        await asyncio.wait({sending, witness}, return_when=asyncio.FIRST_COMPLETED)
        assert entered.is_set() and not sending.done(), "original HTTP close was not retained"
        native.url = outbound_tls.url + "/second"
        second = await value.register(native, creator_id="did:creator")
        assert second.generation != first.generation
        assert await value.configurations.retire_ready(limit=10) == 0
        assert await value.backend.fetch_one(
            "SELECT state, ciphertext IS NOT NULL FROM a2a_push_configs WHERE generation = ?",
            (first.generation,),
        ) == ("deleting", True)
        deleting = asyncio.create_task(runner.delete(task.id, native.id, creator_id="did:creator"))
        witness.cancel()
        await asyncio.gather(witness, return_exceptions=True)
        witness = asyncio.create_task(revoked.wait())
        await asyncio.wait({deleting, witness}, return_when=asyncio.FIRST_COMPLETED)
        assert revoked.is_set() and not deleting.done(), "Delete did not join its replaced HTTP owner"
        native.url = outbound_tls.url + "/third"
        third = await value.register(native, creator_id="did:creator")
        assert third.generation not in {first.generation, second.generation}
        release.set()
        assert await asyncio.gather(sending, deleting) == [1, None]
        assert await value.configurations.get(task.id, native.id, creator_id="did:creator") == third
        for old in (first, second):
            assert await value.backend.fetch_one(
                "SELECT state, ciphertext FROM a2a_push_configs WHERE generation = ?", (old.generation,)
            ) == ("deleted", None)
        # The new consumer may ACK an older queued reference before its own
        # snapshot. That prefix must cause no callback and the backlog must end.
        for _ in range(3):
            if await runner.run_once() == 0:
                break
        else:
            pytest.fail("replacement notification backlog did not drain")
        assert [post[0] for post in outbound_tls.posts] == ["/original", "/third"]
    finally:
        release.set()
        witness.cancel()
        await asyncio.gather(witness, sending, *([deleting] if deleting else []), return_exceptions=True)


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 10}], indirect=True)
async def test_resident_retires_superseded_credentials_after_actual_http_close(outbox, outbound_tls):
    value, task = outbox
    native = callback(task)
    native.url = outbound_tls.url + "/callback"
    first = await value.register(native, creator_id="did:creator")
    runner = worker(value)
    assert await runner.run_once() == 1
    assert await value.backend.fetch_val("SELECT retired FROM a2a_push_delivery_attempts") == 1
    native.token = "replacement-token"
    second = await value.register(native, creator_id="did:creator")
    assert second.generation != first.generation
    assert await runner.run_once() == 1
    assert len(outbound_tls.posts) == 2
    assert await value.backend.fetch_one(
        "SELECT state, ciphertext FROM a2a_push_configs WHERE generation = ?", (first.generation,)
    ) == ("deleted", None)
    assert await value.configurations.get(task.id, "callback", creator_id="did:creator") == second
    assert await value.backend.fetch_all("SELECT generation FROM a2a_push_delivery_attempts") == [(second.generation,)]


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
async def test_invalid_retirement_batch_never_reads_or_removes_configuration(push_store, limit):
    store, task = push_store
    original = await store.create(config(task), creator_id="did:creator")
    with pytest.raises(ValueError, match="positive"):
        await store.retire_ready(limit=limit)
    assert await store.get(task.id, "callback", creator_id="did:creator") == original


async def test_retirement_batches_preserve_active_values_and_unretired_generations(push_store):
    store, task = push_store
    originals, current = [], []
    attempt = None
    for index in range(4):
        native = config(task, f"callback-{index}")
        original = await store.create(native, creator_id="did:creator")
        originals.append(original)
        if index == 0:
            attempt = await admit(store, original, executor_id="original-runtime")
        native.token = "replacement-token"
        current.append(await store.create(native, creator_id="did:creator"))
    assert await store.retire_ready(limit=2) == 2
    assert await store.backend.fetch_val("SELECT COUNT(*) FROM a2a_push_configs WHERE state = 'deleted'") == 2
    assert await store.retire_ready(limit=2) == 1
    assert await store.retire_ready(limit=2) == 0
    assert not await store.finish_delete(originals[0])
    assert await store.retire_delivery(attempt)
    assert await store.retire_ready(limit=2) == 1
    assert await store.retire_ready(limit=2) == 0
    assert await store.list(task.id, creator_id="did:creator", limit=10) == current
    assert await store.backend.fetch_val("SELECT COUNT(*) FROM a2a_push_delivery_attempts") == 0


async def test_idle_generation_retirement_uses_its_partial_index(push_store, monkeypatch):
    store, _ = push_store
    if store.is_postgres:
        rows = await store.backend.fetch_all(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() "
            "AND indexname = 'a2a_push_retiring_configs'"
        )
        assert len(rows) == 1 and "WHERE" in rows[0][0] and "deleting" in rows[0][0]
        return
    original = store.backend.fetch_all
    queries = []

    async def fetch(query, parameters=()):
        rows = await original(query, parameters)
        queries.append((query, parameters))
        return rows

    monkeypatch.setattr(store.backend, "fetch_all", fetch)
    assert await store.retire_ready(limit=1) == 0
    query, parameters = queries[-1]
    plan = await original("EXPLAIN QUERY PLAN " + query, parameters)
    assert any("SEARCH config" in row[-1] and "_retiring_configs" in row[-1] for row in plan), plan


async def test_replacement_preserves_the_old_ciphertext_and_original_attempt(push_store):
    store, task = push_store
    original = await store.create(config(task), creator_id="did:creator")
    attempt = await admit(store, original, executor_id="original-runtime")
    original_ciphertext = await store.backend.fetch_val("SELECT ciphertext FROM a2a_push_configs")
    changed = config(task)
    changed.token = "replacement-token"
    replacement = await store.create(changed, creator_id="did:creator")
    assert replacement.generation != original.generation
    assert (await store.get(task.id, "callback", creator_id="did:creator")) == replacement
    assert not await store.delivery_is_current(attempt)
    assert await admit(store, original, executor_id="late-runtime") is None
    assert await store.backend.fetch_one(
        "SELECT state, ciphertext FROM a2a_push_configs WHERE generation = ?",
        (original.generation,),
    ) == ("deleting", original_ciphertext)
    assert not await store.finish_delete(original)
    assert await store.retire_delivery(attempt)
    assert await store.finish_delete(original)
    assert await store.get(task.id, "callback", creator_id="did:creator") == replacement


async def test_delete_inventory_cannot_erase_a_generation_created_after_revocation(push_store):
    store, task = push_store
    first = await store.create(config(task), creator_id="did:creator")
    attempt = await admit(store, first, executor_id="original-runtime")
    changed = config(task)
    changed.token = "second-token"
    second = await store.create(changed, creator_id="did:creator")
    inventory = await store.begin_delete(task.id, "callback", creator_id="did:creator")
    assert {entry.generation for entry in inventory} == {first.generation, second.generation}
    changed.token = "third-token"
    third = await store.create(changed, creator_id="did:creator")
    assert not await store.finish_delete(first)
    assert await store.retire_delivery(attempt)
    for entry in inventory:
        assert await store.finish_delete(entry)
        assert await store.finish_delete(entry)
    assert await store.get(task.id, "callback", creator_id="did:creator") == third
    assert await store.backend.fetch_val(
        "SELECT COUNT(*) FROM a2a_push_delivery_attempts"
    ) == 0
