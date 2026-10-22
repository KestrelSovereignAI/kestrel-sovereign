"""Standard push CRUD and inline admission through both official SDK bindings."""

import asyncio
import base64
import sys

import pytest
import pytest_asyncio
from a2a.types import (
    DeleteTaskPushNotificationConfigRequest,
    GetTaskPushNotificationConfigRequest,
    ListTaskPushNotificationConfigsRequest,
    TaskPushNotificationConfig,
)
from a2a.utils.errors import InternalError, InvalidParamsError, TaskNotFoundError

from kestrel_sovereign.a2a.push_worker import NativePushWorker
from kestrel_sovereign.a2a.request_handler import NativeA2ARequestHandler, _push_errors
from kestrel_sovereign.a2a.stores.unified.push_config_store import PushConfigConflict
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_commit_outcome,
)
from tests.unit.test_a2a_native_http import submit
from tests.unit.test_a2a_native_http import (
    wire as wire,  # noqa: PLC0414 - fixture re-export
)
from tests.utils.native_a2a_security import PRINCIPAL, SENDER, send
from tests.utils.native_push import managed_push_service
from tests.utils.outbound_tls import (
    outbound_close_barrier as outbound_close_barrier,  # noqa: PLC0414 - fixture dependency
)
from tests.utils.outbound_tls import (
    outbound_tls as outbound_tls,  # noqa: PLC0414 - fixture dependency
)


@pytest_asyncio.fixture
async def native_push_runtime(native_request_admission, monkeypatch):
    async with managed_push_service(
        native_request_admission.manager, monkeypatch, pause_delivery=True
    ) as service:
        yield service


def callback(wire, task_id, config_id=""):
    value = TaskPushNotificationConfig(
        task_id=task_id,
        tenant=wire.admission.scope.tenant,
        id=config_id,
        url="https://8.8.8.8/callback",
        token="native-push-secret",
    )
    value.authentication.scheme = "Bearer"
    value.authentication.credentials = "native-push-authentication"
    return value


def request(wire, cls, task_id, **kwargs):
    return cls(tenant=wire.admission.scope.tenant, task_id=task_id, **kwargs)


async def test_standard_create_get_pagination_and_idempotent_delete(wire):
    task = await submit(wire)
    client = wire.client()
    created = []
    for config_id in ("a", "b", "c"):
        value = callback(wire, task.id, config_id)
        before = value.SerializeToString(deterministic=True)
        created.append(await client.create_task_push_notification_config(value))
        assert value.SerializeToString(deterministic=True) == before
    assert wire.handler.agent_card.capabilities.push_notifications
    fetched = await client.get_task_push_notification_config(
        request(wire, GetTaskPushNotificationConfigRequest, task.id, id="b")
    )
    assert fetched == created[1]
    page = await client.list_task_push_notification_configs(
        request(wire, ListTaskPushNotificationConfigsRequest, task.id, page_size=2)
    )
    assert list(page.configs) == created[:2] and page.next_page_token
    rest = await client.list_task_push_notification_configs(
        request(
            wire,
            ListTaskPushNotificationConfigsRequest,
            task.id,
            page_size=2,
            page_token=page.next_page_token,
        )
    )
    assert list(rest.configs) == created[2:] and not rest.next_page_token
    for _ in range(2):
        await client.delete_task_push_notification_config(
            request(wire, DeleteTaskPushNotificationConfigRequest, task.id, id="b")
        )
    with pytest.raises(TaskNotFoundError):
        await client.get_task_push_notification_config(
            request(wire, GetTaskPushNotificationConfigRequest, task.id, id="b")
        )
    row = await wire.admission.backend.fetch_one(
        "SELECT state, ciphertext FROM a2a_push_configs WHERE config_id = ?", ("b",)
    )
    assert row == ("deleted", None)


async def test_inline_push_is_registered_by_the_admitted_task_before_response(wire):
    params = send(wire.admission)
    params.configuration.return_immediately = True  # SDK polling policy is explicit
    params.configuration.task_push_notification_config.CopyFrom(callback(wire, ""))
    before = params.SerializeToString(deterministic=True)
    events = [event async for event in wire.client().send_message(params)]
    task = events[0].task
    assert params.SerializeToString(deterministic=True) == before
    page = await wire.client().list_task_push_notification_configs(
        request(wire, ListTaskPushNotificationConfigsRequest, task.id)
    )
    assert len(page.configs) == 1
    assert page.configs[0].task_id == task.id and page.configs[0].id
    assert page.configs[0].token == "native-push-secret"
    snapshot = await wire.manager.get_task_snapshot(
        task.id, recipient_agent_id=wire.admission.scope.recipient_id
    )
    assert "native-push-secret" not in str(snapshot.input_request)


@pytest.mark.parametrize("operation", ["create", "get", "list", "delete"])
async def test_push_operations_use_current_task_creator_authority(wire, operation):
    task = await submit(wire)
    client = wire.client()
    await client.create_task_push_notification_config(callback(wire, task.id, "owned"))
    await wire.admission.backend.execute(
        "UPDATE native_test_principals SET principal = ? WHERE signing_did = ?",
        ("did:other-authorized-principal", SENDER),
    )
    methods = {
        "create": (
            client.create_task_push_notification_config,
            callback(wire, task.id, "forged"),
        ),
        "get": (
            client.get_task_push_notification_config,
            request(wire, GetTaskPushNotificationConfigRequest, task.id, id="owned"),
        ),
        "list": (
            client.list_task_push_notification_configs,
            request(wire, ListTaskPushNotificationConfigsRequest, task.id),
        ),
        "delete": (
            client.delete_task_push_notification_config,
            request(wire, DeleteTaskPushNotificationConfigRequest, task.id, id="owned"),
        ),
    }
    method, params = methods[operation]
    with pytest.raises(TaskNotFoundError):
        await method(params)
    stored = await wire.manager.push_service.worker.outbox.configurations.get(
        task.id, "owned", creator_id=PRINCIPAL
    )
    assert stored is not None


@pytest.mark.parametrize(
    "fault",
    [
        "unsafe-url",
        "page-negative",
        "bad-cursor",
        "other-task-cursor",
        "recursive-cursor",
    ],
)
async def test_push_refusals_are_standard_invalid_params(wire, fault):
    task = await submit(wire)
    client = wire.client()
    for config_id in ("a", "b"):
        await client.create_task_push_notification_config(
            callback(wire, task.id, config_id)
        )
    if fault == "unsafe-url":
        value = callback(wire, task.id, "refused")
        value.url = "https://127.0.0.1/private"
        with pytest.raises(InvalidParamsError):
            await client.create_task_push_notification_config(value)
    else:
        page = await client.list_task_push_notification_configs(
            request(wire, ListTaskPushNotificationConfigsRequest, task.id, page_size=1)
        )
        selected_task = (
            (await submit(wire)).id if fault == "other-task-cursor" else task.id
        )
        token = "invalid!" if fault == "bad-cursor" else page.next_page_token
        if fault == "recursive-cursor":
            depth = sys.getrecursionlimit() + 50
            token = base64.urlsafe_b64encode(b"[" * depth + b"]" * depth).decode()
        params = request(
            wire,
            ListTaskPushNotificationConfigsRequest,
            selected_task,
            page_size=-1 if fault == "page-negative" else 1,
            page_token=token,
        )
        with pytest.raises(InvalidParamsError):
            await client.list_task_push_notification_configs(params)


async def test_handler_capability_requires_its_exact_live_managed_service(wire):
    task = await submit(wire)
    card = wire.handler.agent_card
    card.capabilities.push_notifications = False
    with pytest.raises(ValueError, match="capability"):
        NativeA2ARequestHandler(wire.manager, wire.admission.guard, card)
    service = wire.manager.push_service
    await service.close()
    with pytest.raises(RuntimeError):
        NativeA2ARequestHandler(
            wire.manager, wire.admission.guard, wire.handler.agent_card
        )
    with pytest.raises(InternalError):
        await wire.client().list_task_push_notification_configs(
            request(wire, ListTaskPushNotificationConfigsRequest, task.id)
        )


def test_push_domain_translation_cannot_erase_original_commit_control():
    error = PushConfigConflict("retained callback refusal")
    original = ExecutionCommitOutcomeError("unknown")
    error.__cause__ = original
    with pytest.raises(PushConfigConflict) as caught, _push_errors():
        raise error
    assert caught.value is error and caught.value.__cause__ is original
    assert execution_commit_outcome(caught.value) == "unknown"


async def test_remote_unretired_attempt_is_an_internal_refusal_until_physically_retired(
    wire,
):
    task = await submit(wire)
    value = await wire.client().create_task_push_notification_config(
        callback(wire, task.id, "remote")
    )
    store = wire.manager.push_service.worker.outbox.configurations
    configuration = await store.get(task.id, value.id, creator_id=PRINCIPAL)
    attempt = store.prepare_delivery(
        configuration, executor_id="original-other-runtime"
    )
    assert await store.admit_delivery(attempt)
    deletion = request(
        wire, DeleteTaskPushNotificationConfigRequest, task.id, id=value.id
    )
    with pytest.raises(InternalError):
        await wire.client().delete_task_push_notification_config(deletion)
    assert await wire.admission.backend.fetch_one(
        "SELECT state, ciphertext IS NOT NULL FROM a2a_push_configs"
    ) == ("deleting", True)
    assert await store.retire_delivery(attempt)
    await wire.client().delete_task_push_notification_config(deletion)


@pytest.mark.parametrize("outbound_tls", [{"dns_validations": 2}], indirect=True)
async def test_public_delete_waits_for_actual_https_response_retirement(
    wire, outbound_tls, request, monkeypatch
):
    task = await submit(wire)
    value = callback(wire, task.id, "physical")
    value.url = outbound_tls.url + "/callback"
    created = await wire.client().create_task_push_notification_config(value)
    barrier = request.getfixturevalue("outbound_close_barrier")
    worker = wire.manager.push_service.worker
    # The first queued reference is this registration's own initial snapshot.
    # Its original HTTP owner is admitted only by the monitored iteration below.
    assert not outbound_tls.posts
    original_revoke = worker.outbox.revoke
    revoked = asyncio.Event()

    async def observe_revoke(*args, **kwargs):
        result = await original_revoke(*args, **kwargs)
        revoked.set()
        return result

    monkeypatch.setattr(worker.outbox, "revoke", observe_revoke)
    # The resident is held before claims by this fixture. Drive one original
    # iteration while retaining its actual delivery inventory and HTTP owner.
    sending = asyncio.create_task(NativePushWorker.run_once(worker))
    entering = asyncio.create_task(barrier.entered.wait())
    deleting = None
    try:
        await asyncio.wait((sending, entering), return_when=asyncio.FIRST_COMPLETED)
        if sending.done():
            await sending
            assert barrier.entered.is_set(), (
                "original callback iteration returned without HTTP"
            )
        await entering
        deleting = asyncio.create_task(
            wire.client().delete_task_push_notification_config(
                DeleteTaskPushNotificationConfigRequest(
                    tenant=wire.admission.scope.tenant, task_id=task.id, id=created.id
                )
            )
        )
        await revoked.wait()
        assert not deleting.done() and not sending.done()
        assert await wire.admission.backend.fetch_one(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == (0,)
        barrier.release.set()
        await asyncio.gather(sending, deleting)
        assert (
            outbound_tls.posts
            and outbound_tls.posts[0][1]["Authorization"]
            == "Bearer native-push-authentication"
        )
        assert await wire.admission.backend.fetch_one(
            "SELECT state, ciphertext FROM a2a_push_configs"
        ) == ("deleted", None)
    finally:
        barrier.release.set()
        if not entering.done():
            entering.cancel()
        await asyncio.gather(entering, return_exceptions=True)
        await asyncio.gather(
            sending, *([deleting] if deleting else []), return_exceptions=True
        )
