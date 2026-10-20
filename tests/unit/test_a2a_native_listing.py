"""Native creator-scoped keyset pagination on actual SQLite and PostgreSQL."""

import base64

import pytest
import pytest_asyncio
from a2a.helpers.proto_helpers import new_data_artifact, new_task, new_text_message
from a2a.server.routes.common import serialize_list_tasks_response
from a2a.types import ListTasksRequest, TaskState
from a2a.utils.errors import InvalidParamsError
from a2a.utils.task import ListTasksCursor, encode_list_tasks_cursor
from google.protobuf.timestamp_pb2 import Timestamp

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore

SENDER = "did:kestrel:sender"
RECIPIENT = "did:kestrel:recipient"
TENANT = "tenant-alpha"
ORDER = ["max", "epoch-b", "epoch-a", "pre-epoch", "null-z", "null-a"]


async def create(
    store,
    identity,
    stamp,
    *,
    sender=SENDER,
    context="context-one",
    state=TaskState.TASK_STATE_SUBMITTED,
):
    value = new_task(
        identity,
        context,
        state,
        history=[
            new_text_message("request"),
            new_text_message("reply", task_id=identity, context_id=context),
        ],
    )
    value.status.ClearField("timestamp")
    if stamp is not None:
        value.status.timestamp.CopyFrom(Timestamp(seconds=stamp[0], nanos=stamp[1]))
    value.artifacts.append(
        new_data_artifact("result", {"nested": [None, False, 0, {}, []]})
    )
    value.metadata.update({"sender": SENDER, "creator_agent_id": SENDER})
    await store.create(
        value, creator_agent_id=sender, recipient_agent_id=store.scope.recipient_id
    )


@pytest_asyncio.fixture
async def listing(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    store = TaskStore(backend, scope=NativeRecipientScope(RECIPIENT, TENANT))
    try:
        await store.initialize()
        await create(store, "max", (253402300799, 999999999))
        await create(store, "epoch-b", (0, 1), state=TaskState.TASK_STATE_WORKING)
        await create(store, "epoch-a", (0, 1), state=TaskState.TASK_STATE_COMPLETED)
        await create(store, "pre-epoch", (-1, 999999999), context="context-two")
        await create(store, "null-z", None)
        await create(store, "null-a", None, context="context-two")
        await create(
            store,
            "foreign-sender",
            (253402300799, 999999999),
            sender="did:kestrel:other-sender",
        )
        foreign = TaskStore(
            backend,
            scope=NativeRecipientScope("did:kestrel:other-recipient", TENANT),
            owns_backend=False,
        )
        await create(foreign, "foreign-recipient", (253402300799, 999999999))
        foreign = TaskStore(
            backend,
            scope=NativeRecipientScope(RECIPIENT, "other-tenant"),
            owns_backend=False,
        )
        await create(foreign, "foreign-tenant", (253402300799, 999999999))
        yield store
    finally:
        await backend.close()


async def page(store, *, creator=SENDER, recipient=RECIPIENT, **fields):
    return await store.list_for_creator(
        ListTasksRequest(tenant=TENANT, **fields),
        creator_agent_id=creator,
        recipient_agent_id=recipient,
    )


async def test_native_pages_are_scoped_sorted_and_preserve_nanoseconds_and_nulls(
    listing,
):
    token = ""
    received = []
    for index in range(4):
        result = await page(
            listing,
            page_size=2,
            page_token=token,
            include_artifacts=True,
            history_length=1,
        )
        assert result.page_size == 2 and result.total_size == 6
        assert all(
            len(task.history) == 1 and len(task.artifacts) == 1 for task in result.tasks
        )
        received.extend(task.id for task in result.tasks)
        body = serialize_list_tasks_response(result, include_artifacts=True)
        assert body["nextPageToken"] == result.next_page_token
        token = result.next_page_token
        if not token:
            break
    assert index == 2
    assert received == ORDER
    assert len(set(received)) == len(received)
    assert body["nextPageToken"] == ""


async def test_list_defaults_and_projection_use_official_sdk_semantics(listing):
    result = await page(listing, history_length=0)
    assert result.page_size == 50 and result.total_size == 6
    assert [task.id for task in result.tasks] == ORDER
    assert all(not task.history and not task.artifacts for task in result.tasks)
    body = serialize_list_tasks_response(result, include_artifacts=False)
    assert all("artifacts" not in task for task in body["tasks"])
    assert body["nextPageToken"] == ""
    stored = await listing.get_for_creator("max", SENDER, recipient_agent_id=RECIPIENT)
    assert len(stored.history) == 2 and len(stored.artifacts) == 1


async def test_native_filters_preserve_inclusive_subsecond_boundaries(listing):
    result = await page(listing, status_timestamp_after=Timestamp(seconds=0, nanos=1))
    assert [task.id for task in result.tasks] == ["max", "epoch-b", "epoch-a"]
    assert result.total_size == 3
    result = await page(listing, context_id="context-two")
    assert [task.id for task in result.tasks] == ["pre-epoch", "null-a"]
    assert result.total_size == 2
    result = await page(listing, status=TaskState.TASK_STATE_WORKING)
    assert [task.id for task in result.tasks] == ["epoch-b"] and result.total_size == 1


async def test_a_keyset_cursor_survives_deletion_of_its_last_task(listing):
    first = await page(listing, page_size=2)
    assert [task.id for task in first.tasks] == ["max", "epoch-b"]
    assert await listing.delete("epoch-b")
    second = await page(listing, page_size=2, page_token=first.next_page_token)
    assert [task.id for task in second.tasks] == ["epoch-a", "pre-epoch"]
    assert second.total_size == 5


@pytest.mark.parametrize(
    "fields",
    [
        {"creator": "did:kestrel:unknown"},
        {"recipient": "did:kestrel:other-recipient"},
        {"creator": "' OR 1=1 --"},
    ],
)
async def test_no_visible_task_is_not_a_cross_principal_existence_probe(
    listing, fields
):
    result = await page(listing, **fields)
    assert list(result.tasks) == [] and result.total_size == 0
    assert (
        serialize_list_tasks_response(result, include_artifacts=False)["nextPageToken"]
        == ""
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"page_size": 0},
        {"page_size": -1},
        {"page_size": 101},
        {"history_length": -1},
        {"status": 999},
        {"page_token": "not a cursor"},
        {"page_token": base64.b64encode(b"epoch-b").decode()},
        {
            "page_token": encode_list_tasks_cursor(
                ListTasksCursor(timestamp_ns=10**100, task_id="epoch-b")
            )
        },
        {"status_timestamp_after": Timestamp(seconds=0, nanos=-1)},
    ],
)
async def test_malformed_native_list_parameters_fail_clearly(listing, fields):
    with pytest.raises(InvalidParamsError):
        await page(listing, **fields)


async def test_a_request_tenant_cannot_choose_the_store_scope(listing):
    with pytest.raises(InvalidParamsError, match="tenant"):
        await listing.list_for_creator(
            ListTasksRequest(tenant="other-tenant"),
            creator_agent_id=SENDER,
            recipient_agent_id=RECIPIENT,
        )
