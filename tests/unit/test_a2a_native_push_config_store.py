"""Native task authority, real AEAD storage and retryable push retirement."""

from dataclasses import replace

import pytest
import pytest_asyncio
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
from kestrel_sdk.native_a2a.types import Role, SendMessageRequest, TaskPushNotificationConfig
from kestrel_sdk.security.exceptions import DecryptionError, MasterKeyNotConfiguredError

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.push_config_store import (
    PushConfigConflict,
    PushConfigStore,
    PushTaskNotFound,
)
from tests.utils.native_a2a import manager_for
from tests.utils.native_push import configure_push_test_key


@pytest_asyncio.fixture
async def push_store(native_a2a_backend_factory, monkeypatch):
    configure_push_test_key(monkeypatch)
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:recipient", "tenant"))
    try:
        await manager.initialize()
        task = await manager.create_task(
            SendMessageRequest(
                tenant="tenant", message=new_text_message("work", role=Role.ROLE_USER)
            ),
            creator_agent_id="did:creator",
        )
        store = PushConfigStore(manager.task_store)
        await store.initialize()
        yield store, task
    finally:
        await manager.close()


def config(task, config_id="callback"):
    value = TaskPushNotificationConfig(
        tenant="tenant",
        task_id=task.id,
        id=config_id,
        url="https://callback.example/secret-path",
        token="notification-secret",
    )
    value.authentication.scheme = "Bearer"
    value.authentication.credentials = "callback-secret"
    return value


async def admit(store, configuration, *, executor_id):
    attempt = store.prepare_delivery(configuration, executor_id=executor_id)
    return attempt if await store.admit_delivery(attempt) else None


async def test_native_config_round_trip_encrypts_all_callback_material(push_store):
    store, task = push_store
    value = config(task)
    before = value.SerializeToString()
    created = await store.create(value, creator_id="did:creator")
    value.url = "https://changed.example"
    read = await store.get(task.id, created.config.id, creator_id="did:creator")
    assert read.config.SerializeToString() == before
    assert read.generation == created.generation
    assert "callback-secret" not in repr(read)
    row = await store.backend.fetch_one("SELECT ciphertext FROM a2a_push_configs")
    assert row[0].startswith("KSAv2:")
    assert all(
        secret not in row[0]
        for secret in [
            "callback.example",
            "secret-path",
            "notification-secret",
            "callback-secret",
        ]
    )
    read.config.url = "https://mutated.example"
    assert read.config.SerializeToString() == before
    assert (
        await store.get(task.id, created.config.id, creator_id="did:creator")
    ).config.SerializeToString() == before


@pytest.mark.parametrize("operation", ["create", "get", "list", "begin_delete"])
async def test_task_creator_authority_cannot_be_forged_by_another_principal(
    push_store, operation
):
    store, task = push_store
    await store.create(config(task), creator_id="did:creator")
    with pytest.raises(PushTaskNotFound):
        if operation == "create":
            await store.create(config(task, "forged"), creator_id="did:other")
        elif operation == "list":
            await store.list(task.id, creator_id="did:other", limit=10)
        else:
            await getattr(store, operation)(task.id, "callback", creator_id="did:other")
    assert (await store.get(task.id, "callback", creator_id="did:creator")) is not None


@pytest.mark.parametrize(
    "field", ["creator_id", "generation", "config_id", "tenant", "recipient_id"]
)
async def test_ciphertext_is_bound_to_every_ownership_column(push_store, field):
    store, task = push_store
    await store.create(config(task), creator_id="did:creator")
    # Copy the encrypted row to another locator without decrypting it. Read
    # through the decoder with that independently selected locator must fail.
    row = list(
        await store.backend.fetch_one(
            "SELECT task_id, config_id, creator_id, generation, ciphertext FROM a2a_push_configs"
        )
    )
    if field in {"creator_id", "generation", "config_id"}:
        row[{"config_id": 1, "creator_id": 2, "generation": 3}[field]] = "substituted"
    else:
        from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore

        scope = NativeRecipientScope(
            "did:other" if field == "recipient_id" else "did:recipient",
            "other" if field == "tenant" else "tenant",
        )
        store = PushConfigStore(
            TaskStore(store.backend, scope=scope, owns_backend=False)
        )
    with pytest.raises(DecryptionError):
        store._decode(tuple(row))


async def test_deletion_revokes_new_reads_and_retains_original_generation_until_retirement(
    push_store,
):
    store, task = push_store
    created = await store.create(config(task), creator_id="did:creator")
    retiring, = await store.begin_delete(task.id, "callback", creator_id="did:creator")
    assert retiring.generation == created.generation
    assert await store.get(task.id, "callback", creator_id="did:creator") is None
    assert await store.list(task.id, creator_id="did:creator", limit=10) == []
    # A restarted service can resume the same retirement; it does not create a
    # new active generation or forget credentials before resource retirement.
    restarted = PushConfigStore(store.task_store)
    again, = await restarted.begin_delete(task.id, "callback", creator_id="did:creator")
    assert again.generation == retiring.generation
    assert not await restarted.finish_delete(replace(again, generation="stale"))
    assert await restarted.finish_delete(again)
    assert (
        await restarted.begin_delete(task.id, "callback", creator_id="did:creator")
        == ()
    )
    assert await store.backend.fetch_one(
        "SELECT state, ciphertext FROM a2a_push_configs"
    ) == ("deleted", None)
    replacement = await store.create(config(task), creator_id="did:creator")
    assert replacement.generation != retiring.generation
    assert await store.get(task.id, "callback", creator_id="did:creator") == replacement


async def test_config_id_default_pagination_and_idempotent_set(push_store):
    store, task = push_store
    generated = await store.create(config(task, ""), creator_id="did:creator")
    assert generated.config.id == task.id
    explicit = await store.create(config(task, "z"), creator_id="did:creator")
    first = await store.list(task.id, creator_id="did:creator", limit=1)
    second = await store.list(
        task.id, creator_id="did:creator", limit=1, after=first[0].config.id
    )
    assert [first[0].config.id, second[0].config.id] == [generated.config.id, "z"]
    assert await store.create(config(task, "z"), creator_id="did:creator") == explicit


async def test_wrong_tenant_refuses_without_persistence(push_store):
    store, task = push_store
    value = config(task)
    value.tenant = "different"
    with pytest.raises(PushConfigConflict, match="tenant"):
        await store.create(value, creator_id="did:creator")
    assert await store.backend.fetch_all("SELECT config_id FROM a2a_push_configs") == []


async def test_borrowed_store_close_leaves_primary_storage_live(push_store):
    store, task = push_store
    await store.close()
    assert store.backend.is_connected
    assert (
        await store.task_store.get_for_creator(
            task.id, "did:creator", recipient_agent_id="did:recipient"
        )
    ).id == task.id


async def test_missing_secret_key_fails_before_any_configuration_write(
    push_store, monkeypatch
):
    store, task = push_store
    import kestrel_sovereign.a2a.stores.unified.push_config_store as module

    original = MasterKeyNotConfiguredError("test missing key")

    def missing_key(*_args):
        raise original

    monkeypatch.setattr(module, "get_agent_key", missing_key)
    with pytest.raises(MasterKeyNotConfiguredError) as captured:
        await store.create(config(task), creator_id="did:creator")
    assert captured.value is original
    assert await store.backend.fetch_all("SELECT config_id FROM a2a_push_configs") == []


async def test_an_unretired_external_request_blocks_delete_even_from_another_connection(
    push_store,
    native_a2a_backend_factory,
):
    store, task = push_store
    created = await store.create(config(task), creator_id="did:creator")
    attempt = await admit(store, created, executor_id="original-runtime")
    assert attempt is not None and await store.delivery_is_current(attempt)
    other = native_a2a_backend_factory()
    await other.connect()
    try:
        from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore

        independent = PushConfigStore(
            TaskStore(other, scope=store.scope, owns_backend=False)
        )
        retiring, = await independent.begin_delete(
            task.id, "callback", creator_id="did:creator"
        )
        assert await admit(independent, created, executor_id="other-runtime") is None
        assert not await store.delivery_is_current(attempt)
        assert not await independent.finish_delete(retiring)
        assert not await independent.retire_delivery(
            replace(attempt, executor_id="impostor")
        )
        assert not await independent.finish_delete(retiring)
        assert await store.retire_delivery(attempt)
        assert await store.retire_delivery(attempt)
        assert await independent.finish_delete(retiring)
    finally:
        await other.close()


async def test_concurrent_delete_and_admission_have_one_durable_order(
    push_store,
    native_a2a_backend_factory,
):
    import asyncio

    from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore

    store, task = push_store
    created = await store.create(config(task), creator_id="did:creator")
    other = native_a2a_backend_factory()
    await other.connect()
    try:
        independent = PushConfigStore(
            TaskStore(other, scope=store.scope, owns_backend=False)
        )
        attempt, inventory = await asyncio.gather(
            admit(store, created, executor_id="original-runtime"),
            independent.begin_delete(task.id, "callback", creator_id="did:creator"),
        )
        retiring, = inventory
        if attempt is not None:
            assert not await independent.finish_delete(retiring)
            assert await store.retire_delivery(attempt)
        assert await independent.finish_delete(retiring)
        assert await admit(store, created, executor_id="late-runtime") is None
    finally:
        await other.close()


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
async def test_invalid_page_sizes_fail_explicitly(push_store, limit):
    store, task = push_store
    with pytest.raises(ValueError, match="positive"):
        await store.list(task.id, creator_id="did:creator", limit=limit)


async def test_task_cleanup_cannot_erase_a_live_delivery_and_succeeds_after_delete(
    push_store,
):
    from kestrel_sovereign.storage.db.interface import QueryError

    store, task = push_store
    created = await store.create(config(task), creator_id="did:creator")
    attempt = await admit(store, created, executor_id="original-runtime")
    with pytest.raises(QueryError):
        await store.task_store.delete(task.id)
    assert await store.delivery_is_current(attempt)
    retiring, = await store.begin_delete(task.id, "callback", creator_id="did:creator")
    assert not await store.finish_delete(retiring)
    assert await store.retire_delivery(attempt)
    assert await store.finish_delete(retiring)
    assert await store.task_store.delete(task.id)
    assert await store.backend.fetch_all("SELECT config_id FROM a2a_push_configs") == []
    assert (
        await store.backend.fetch_all(
            "SELECT attempt_id FROM a2a_push_delivery_attempts"
        )
        == []
    )


@pytest.mark.parametrize(
    "scope",
    [
        NativeRecipientScope("did:other", "tenant"),
        NativeRecipientScope("did:recipient", "other"),
    ],
)
async def test_other_recipient_or_tenant_cannot_read_an_existing_task_configuration(
    push_store, scope
):
    from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore

    store, task = push_store
    await store.create(config(task), creator_id="did:creator")
    other = PushConfigStore(TaskStore(store.backend, scope=scope, owns_backend=False))
    with pytest.raises(PushTaskNotFound):
        await other.get(task.id, "callback", creator_id="did:creator")


async def test_an_old_generation_cannot_admit_a_new_request(push_store):
    store, task = push_store
    created = await store.create(config(task), creator_id="did:creator")
    assert (
        await admit(store, replace(created, generation="stale"), executor_id="runtime")
        is None
    )
    assert (
        await store.backend.fetch_all(
            "SELECT attempt_id FROM a2a_push_delivery_attempts"
        )
        == []
    )
