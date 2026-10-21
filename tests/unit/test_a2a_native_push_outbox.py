"""Atomic encrypted registration and durable metadata-only native push fan-out."""

import asyncio

import pytest
import pytest_asyncio
from kestrel_sdk.security.exceptions import SSRFError

from kestrel_sovereign.a2a.push_outbox import NativePushOutbox
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.base import NativeStoreConflict
from kestrel_sovereign.a2a.stores.unified.push_config_store import (
    PushConfigConflict,
    PushConfigStore,
)
from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
from kestrel_sovereign.signals.durable import DurableSignalStore
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError
from tests.unit.test_a2a_native_push_config_store import config
from tests.unit.test_a2a_native_push_config_store import (
    push_store as push_store,  # noqa: PLC0414 - pytest fixture re-export
)


@pytest_asyncio.fixture
async def outbox(push_store):
    store, task = push_store
    signals = DurableSignalStore(store.backend)
    await signals.initialize()
    value = NativePushOutbox(store, signals, retention_days=30, lease_seconds=60)
    await value.initialize()
    return value, task


def callback(task, config_id="callback"):
    value = config(task, config_id)
    # A public numeric address exercises real URL validation without DNS or TCP.
    value.url = "https://1.1.1.1/callback?private=never-journal-this"
    return value


async def test_registration_commits_encrypted_config_consumer_and_initial_reference(
    outbox,
):
    value, task = outbox
    created = await value.register(callback(task), creator_id="did:creator")
    subscriptions = await value.subscriptions()
    assert len(subscriptions) == 1
    subscription = subscriptions[0]
    assert await value.backend.fetch_one(
        "SELECT COUNT(*) FROM durable_signal_consumers"
    ) == (1,)
    assert (
        subscription.task_id,
        subscription.config_id,
        subscription.generation,
        subscription.creator_id,
    ) == (task.id, created.config.id, created.generation, "did:creator")
    pending = await value.changes.pending(limit=10)
    assert len(pending) == 2 and pending[0].sequence == subscription.after_sequence
    assert pending[1].sequence > subscription.after_sequence
    await value.forward_pending(limit=10)
    rows = await value.backend.fetch_all(
        "SELECT payload, caller_identity FROM durable_signal_events"
    )
    assert len(rows) == 2
    assert all(
        "never-journal-this" not in repr(row) and "callback-secret" not in repr(row)
        for row in rows
    )
    claimed = await value.signals.claim_delivery(
        agent_id=value.agent_id,
        consumer_id=subscription.consumer_id,
        executor_id="executor",
    )
    assert claimed.max_attempts == 0
    assert claimed.event.payload == {
        "task_id": task.id,
        "journal_sequence": pending[0].sequence,
    }
    assert claimed.event.caller_identity is None


async def test_registration_is_atomic_when_its_enclosing_task_admission_rolls_back(
    outbox,
):
    value, task = outbox
    before = await value.changes.pending(limit=10)
    original = RuntimeError("admission refused after registration")
    with pytest.raises(TransactionError):
        async with value.write_transaction():
            await value.register(callback(task), creator_id="did:creator")
            raise original
    assert await value.subscriptions() == []
    assert (
        await value.configurations.list(task.id, creator_id="did:creator", limit=10)
        == []
    )
    assert await value.backend.fetch_all("SELECT * FROM durable_signal_consumers") == []
    assert await value.changes.pending(limit=10) == before


async def test_forwarding_and_journal_receipt_rollback_together(outbox):
    value, task = outbox
    await value.register(callback(task), creator_id="did:creator")
    pending = await value.changes.pending(limit=10)
    with pytest.raises(TransactionError):
        async with value.write_transaction():
            assert await value.forward_pending(limit=10) == 2
            raise RuntimeError("producer stops before its commit")
    assert await value.changes.pending(limit=10) == pending
    assert await value.backend.fetch_all("SELECT * FROM durable_signal_events") == []
    restarted = NativePushOutbox(
        value.configurations, value.signals, retention_days=30, lease_seconds=60
    )
    await restarted.initialize()
    assert await restarted.forward_pending(limit=10) == 2
    assert await restarted.forward_pending(limit=10) == 0
    assert (
        len(await value.backend.fetch_all("SELECT * FROM durable_signal_events")) == 2
    )
    assert (
        len(await value.backend.fetch_all("SELECT * FROM durable_signal_deliveries"))
        == 2
    )


async def test_lost_forward_receipt_ack_cannot_duplicate_native_deliveries(outbox):
    value, task = outbox
    await value.register(callback(task), creator_id="did:creator")
    assert await value.forward_pending(limit=10) == 2
    events = await value.backend.fetch_all(
        "SELECT event_id FROM durable_signal_events ORDER BY event_id"
    )
    # Simulate a journal receipt restored without its matching event database.
    # Stable source IDs still select original events and original deliveries.
    await value.backend.execute("UPDATE a2a_task_changes SET forwarded_event_id = NULL")
    assert await value.forward_pending(limit=10) == 2
    assert (
        await value.backend.fetch_all(
            "SELECT event_id FROM durable_signal_events ORDER BY event_id"
        )
        == events
    )
    assert (
        len(await value.backend.fetch_all("SELECT * FROM durable_signal_deliveries"))
        == 2
    )


@pytest.mark.parametrize(
    "url",
    ["http://1.1.1.1", "https://127.0.0.1", "https://169.254.169.254/latest/meta-data"],
)
async def test_unsafe_callback_is_refused_before_registration_or_initial_enqueue(
    outbox, url
):
    value, task = outbox
    before = await value.changes.pending(limit=10)
    native = callback(task)
    native.url = url
    with pytest.raises(SSRFError):
        await value.register(native, creator_id="did:creator")
    assert await value.subscriptions() == []
    assert (
        await value.configurations.list(task.id, creator_id="did:creator", limit=10)
        == []
    )
    assert await value.changes.pending(limit=10) == before


@pytest.mark.parametrize("field", ["token", "credentials", "scheme"])
async def test_invalid_callback_header_refuses_before_any_registration(outbox, field):
    value, task = outbox
    native = callback(task)
    if field == "token":
        native.token = "header\r\ninjection"
    elif field == "credentials":
        native.authentication.credentials = "invalid\x00credential"
    else:
        native.authentication.scheme = "not a scheme"
    with pytest.raises(PushConfigConflict):
        await value.register(native, creator_id="did:creator")
    assert await value.subscriptions() == []
    assert (
        await value.configurations.list(task.id, creator_id="did:creator", limit=10)
        == []
    )


async def test_foreign_creator_cannot_register_or_create_consumers(outbox):
    value, task = outbox
    with pytest.raises(NativeStoreConflict):
        await value.register(callback(task), creator_id="did:impostor")
    assert await value.backend.fetch_all("SELECT * FROM durable_signal_consumers") == []


async def refuse_original_database_write(backend, *, table, event):
    """Inject a real database constraint failure without replacing the producer."""
    name = "test_refuse_" + table + "_" + event.lower()
    if backend.backend_type == "sqlite":
        await backend.execute_script(f"""
            CREATE TRIGGER {name} BEFORE {event} ON {table}
            BEGIN SELECT RAISE(ABORT, 'original test storage refusal'); END
        """)
    else:
        await backend.execute_script(f"""
            CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'original test storage refusal' USING ERRCODE = '23514'; END;
            $$;
            CREATE TRIGGER {name} BEFORE {event} ON {table}
                FOR EACH ROW EXECUTE FUNCTION {name}()
        """)


async def test_real_consumer_storage_refusal_rolls_back_the_encrypted_registration(
    outbox,
):
    value, task = outbox
    before = await value.changes.pending(limit=10)
    await refuse_original_database_write(
        value.backend, table="durable_signal_consumers", event="INSERT"
    )
    with pytest.raises(TransactionError):
        await value.register(callback(task), creator_id="did:creator")
    assert (
        await value.configurations.list(task.id, creator_id="did:creator", limit=10)
        == []
    )
    assert await value.subscriptions() == []
    assert await value.changes.pending(limit=10) == before


async def test_real_journal_receipt_refusal_rolls_back_the_event_and_initial_deliveries(
    outbox,
):
    value, task = outbox
    await value.register(callback(task), creator_id="did:creator")
    before = await value.changes.pending(limit=10)
    await refuse_original_database_write(
        value.backend, table="a2a_task_changes", event="UPDATE"
    )
    with pytest.raises(TransactionError):
        await value.forward_pending(limit=10)
    assert await value.backend.fetch_all("SELECT * FROM durable_signal_events") == []
    assert (
        await value.backend.fetch_all("SELECT * FROM durable_signal_deliveries") == []
    )
    assert await value.changes.pending(limit=10) == before


async def test_registration_validation_owns_an_immutable_callback_copy(
    outbox, monkeypatch
):
    from kestrel_sovereign.a2a import push_transport

    value, task = outbox
    entered, release = asyncio.Event(), asyncio.Event()
    validate = push_transport.assert_safe_url

    async def pause(*args, **kwargs):
        entered.set()
        await release.wait()
        return await validate(*args, **kwargs)

    monkeypatch.setattr(push_transport, "assert_safe_url", pause)
    native = callback(task)
    original = native.SerializeToString()
    registration = asyncio.create_task(value.register(native, creator_id="did:creator"))
    try:
        await entered.wait()
        native.url = "https://127.0.0.1"
        native.authentication.credentials = "swapped-secret"
        release.set()
        created = await registration
        assert created.config.SerializeToString() == original
    finally:
        release.set()
        await asyncio.gather(registration, return_exceptions=True)


async def test_prepared_inline_registration_has_no_dns_inside_the_task_transaction(
    outbox, monkeypatch
):
    from kestrel_sovereign.a2a import push_transport

    value, task = outbox
    native = callback(task)
    native.task_id = ""
    prepared = await value.prepare(native)
    native.url = "https://127.0.0.1"
    prepared.config.authentication.credentials = "mutated-reader"
    assert "callback-secret" not in repr(prepared)

    async def refuse_dns(*args, **kwargs):
        raise AssertionError(
            "Push registration attempted DNS inside its task transaction"
        )

    monkeypatch.setattr(push_transport, "assert_safe_url", refuse_dns)
    async with value.write_transaction():
        created = await value.register_prepared(
            prepared, creator_id="did:creator", task_id=task.id
        )
    assert created.config.task_id == task.id
    assert created.config.url == callback(task).url
    assert created.config.authentication.credentials == "callback-secret"
    assert (await value.subscriptions())[0].generation == created.generation


async def test_prepared_registration_cannot_change_recipient_tenant_or_selected_task(
    outbox,
):
    value, task = outbox
    prepared = await value.prepare(callback(task))
    foreign = NativePushOutbox(
        PushConfigStore(
            TaskStore(
                value.backend,
                scope=NativeRecipientScope("did:other", "tenant"),
                owns_backend=False,
            )
        ),
        value.signals,
        retention_days=30,
        lease_seconds=60,
    )
    with pytest.raises(PushConfigConflict, match="recipient scope"):
        await foreign.register_prepared(prepared, creator_id="did:creator")
    with pytest.raises(PushConfigConflict, match="another task"):
        await value.register_prepared(
            prepared, creator_id="did:creator", task_id="another-task"
        )
    assert await value.subscriptions() == []


async def test_unbound_inline_registration_requires_the_server_admitted_task_id(outbox):
    value, task = outbox
    native = callback(task)
    native.task_id = ""
    prepared = await value.prepare(native)
    with pytest.raises(PushConfigConflict, match="admitted task ID"):
        await value.register_prepared(prepared, creator_id="did:creator")


async def test_foreign_tenant_preparation_refuses_before_dns(outbox, monkeypatch):
    from kestrel_sovereign.a2a import push_transport

    value, task = outbox
    native = callback(task)
    native.tenant = "another-tenant"

    async def refuse_dns(*args, **kwargs):
        raise AssertionError("Another tenant reached callback DNS validation")

    monkeypatch.setattr(push_transport, "assert_safe_url", refuse_dns)
    with pytest.raises(PushConfigConflict, match="another tenant"):
        await value.prepare(native)


@pytest.mark.parametrize("bad", [None, {}, "private-config"])
async def test_prepared_boundary_requires_native_values_and_prepared_handoffs(
    outbox, bad
):
    value, _ = outbox
    with pytest.raises(TypeError, match="official"):
        await value.prepare(bad)
    with pytest.raises(TypeError, match="prepared"):
        await value.register_prepared(bad, creator_id="did:creator")


@pytest.mark.parametrize("bad", ["", " ", True])
async def test_inline_task_assignment_requires_a_concrete_server_id(outbox, bad):
    value, task = outbox
    native = callback(task)
    native.task_id = ""
    prepared = await value.prepare(native)
    with pytest.raises(ValueError, match="admitted task ID"):
        await value.register_prepared(prepared, creator_id="did:creator", task_id=bad)


async def test_tenant_scopes_are_distinct_even_on_the_same_backend(outbox):
    value, task = outbox
    await value.register(callback(task), creator_id="did:creator")
    other = NativePushOutbox(
        PushConfigStore(
            TaskStore(
                value.backend,
                scope=NativeRecipientScope(
                    value.configurations.scope.recipient_id, "other"
                ),
                owns_backend=False,
            )
        ),
        value.signals,
        retention_days=30,
        lease_seconds=60,
    )
    assert other.agent_id != value.agent_id
    assert await other.subscriptions() == []
    assert await other.forward_pending(limit=10) == 0


@pytest.mark.parametrize(
    "retention,lease", [(0, 60), (-1, 60), (True, 60), (30, 0), (30, False)]
)
async def test_invalid_delivery_policy_is_refused_before_storage(
    push_store, retention, lease
):
    store, _ = push_store
    with pytest.raises(ValueError):
        NativePushOutbox(
            store,
            DurableSignalStore(store.backend),
            retention_days=retention,
            lease_seconds=lease,
        )


async def test_push_schema_completion_covers_the_whole_atomic_bundle(
    push_store, monkeypatch
):
    store, _ = push_store
    signals = DurableSignalStore(store.backend)
    await signals.initialize()
    await store.backend.execute("DROP TABLE a2a_push_delivery_attempts")
    await store.backend.execute("DROP TABLE a2a_push_configs")
    original = store.backend.execute_script

    async def refuse_subscription_ddl(script):
        if "CREATE TABLE IF NOT EXISTS a2a_push_subscriptions" in script:
            # Exercise real DDL followed by a real database error. The schema
            # transaction must also roll back the earlier config/attempt DDL.
            script += "; SELECT original_missing_subscription_ddl_function()"
        return await original(script)

    monkeypatch.setattr(store.backend, "execute_script", refuse_subscription_ddl)
    value = NativePushOutbox(store, signals, retention_days=30, lease_seconds=60)
    with pytest.raises((QueryError, TransactionError)):
        await value.initialize()
    for table in (
        "a2a_push_configs",
        "a2a_push_delivery_attempts",
        "a2a_push_subscriptions",
    ):
        assert not await store.backend.table_exists(table)
    assert (
        await store.backend.fetch_one(
            "SELECT revision FROM a2a_native_store_schema WHERE component = 'push'"
        )
        is None
    )


async def test_completed_push_boot_keeps_its_schema_lease_without_replaying_ddl(
    outbox, monkeypatch
):
    from kestrel_sovereign.a2a.native_schema import NativeStoreSchema

    value, _ = outbox
    assert (
        await value.backend.fetch_one(
            "SELECT component, revision FROM a2a_native_store_schema WHERE component = 'push'"
        )
        == NativeStoreSchema.PUSH.value
    )

    async def refuse_data_ddl():
        raise AssertionError("Completed push boot replayed data-table migration DDL")

    monkeypatch.setattr(value.configurations, "initialize", refuse_data_ddl)
    await value.initialize()


async def test_missing_service_key_refuses_push_boot_before_any_schema_change(
    push_store, monkeypatch
):
    from kestrel_sdk.security.exceptions import MasterKeyNotConfiguredError

    from kestrel_sovereign.a2a.stores.unified import push_config_store

    store, _ = push_store
    signals = DurableSignalStore(store.backend)
    await signals.initialize()
    original = MasterKeyNotConfiguredError("original missing service key")

    def unavailable_key(*args):
        raise original

    monkeypatch.setattr(push_config_store, "get_agent_key", unavailable_key)
    with pytest.raises(MasterKeyNotConfiguredError) as caught:
        await NativePushOutbox(
            store, signals, retention_days=30, lease_seconds=60
        ).initialize()
    assert caught.value is original
    assert not await store.backend.table_exists("a2a_push_subscriptions")
    assert (
        await store.backend.fetch_one(
            "SELECT revision FROM a2a_native_store_schema WHERE component = 'push'"
        )
        is None
    )
