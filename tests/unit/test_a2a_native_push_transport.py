"""Real pinned TLS, standard push events and durable HTTP retirement receipts."""

from datetime import timedelta

import httpx
import pytest
from a2a.helpers.proto_helpers import new_data_artifact
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils.proto_utils import to_stream_response
from google.protobuf.json_format import MessageToDict
from kestrel_sdk.security.outbound import SSRFError

from kestrel_sovereign.a2a.push_transport import (
    NativePushTransport,
    PushDeliveryRevoked,
)
from kestrel_sovereign.a2a.stores.unified.push_config_store import PushConfigConflict
from tests.unit.test_a2a_native_push_config_store import config
from tests.unit.test_a2a_native_push_config_store import (
    push_store as push_store,  # noqa: PLC0414 - pytest fixture re-export
)
from tests.utils.outbound_tls import (
    outbound_close_barrier as outbound_close_barrier,  # noqa: PLC0414 - pytest fixture re-export
)
from tests.utils.outbound_tls import (
    outbound_tls as outbound_tls,  # noqa: PLC0414 - pytest fixture re-export
)


async def configured(push_store, url):
    store, task = push_store
    value = config(task)
    value.url = url
    created = await store.create(value, creator_id="did:creator")
    return NativePushTransport(store, timeout=timedelta(seconds=5)), created, task


async def test_cancel_during_admission_joins_original_commit_without_http(
    push_store, outbound_tls, monkeypatch
):
    import asyncio

    transport, configuration, task = await configured(
        push_store, outbound_tls.url + "/callback"
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original_admit = transport.store.admit_delivery

    async def original_transaction(attempt):
        async with transport.store.write_transaction():
            result = await original_admit(attempt)
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(transport.store, "admit_delivery", original_transaction)
    sending = asyncio.create_task(
        transport.send(configuration, task, executor_id="original-runtime")
    )
    try:
        await entered.wait()
        sending.cancel("original admission observer cancellation")
        await asyncio.sleep(0)
        assert not sending.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await sending
        assert caught.value.args == ("original admission observer cancellation",)
        assert await transport.store.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == [(1,)]
        assert outbound_tls.dns == [] and outbound_tls.dials == []
    finally:
        release.set()
        await asyncio.gather(sending, return_exceptions=True)


@pytest.mark.parametrize("event_kind", ["task", "status", "artifact"])
async def test_push_sends_standard_stream_response_and_retires_real_https(
    push_store, outbound_tls, event_kind
):
    peer = outbound_tls
    transport, configuration, task = await configured(
        push_store, peer.url + "/callback"
    )
    event = task
    if event_kind == "status":
        event = TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
        )
    elif event_kind == "artifact":
        event = TaskArtifactUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            artifact=new_data_artifact("result", {"answer": 42}, artifact_id="result"),
        )
    attempt = await transport.send(configuration, event, executor_id="runtime")
    assert await transport.store.backend.fetch_one(
        "SELECT retired FROM a2a_push_delivery_attempts WHERE attempt_id = ?",
        (attempt.attempt_id,),
    ) == (1,)
    assert peer.dns == [peer.host]
    assert peer.dials == [("8.8.8.8", peer.port)]
    assert peer.sni == [peer.host]
    assert len(peer.posts) == 1
    path, headers, body = peer.posts[0]
    assert path == "/callback"
    assert headers["Host"] == f"{peer.host}:{peer.port}"
    assert headers["Content-Type"] == "application/a2a+json"
    assert headers.get("A2A-Version") == "1.0"
    assert headers.get("Authorization") == "Bearer callback-secret"
    assert headers.get("X-A2A-Notification-Token") == "notification-secret"
    assert body == MessageToDict(to_stream_response(event))


@pytest.mark.parametrize("path,status", [("/redirect", 302), ("/failure", 503)])
async def test_callback_http_failure_is_not_success_or_a_redirect_retry(
    push_store, outbound_tls, path, status
):
    transport, configuration, task = await configured(
        push_store, outbound_tls.url + path
    )
    with pytest.raises(httpx.HTTPStatusError) as failure:
        await transport.send(configuration, task, executor_id="runtime")
    assert failure.value.response.status_code == status
    assert len(outbound_tls.posts) == 1
    assert outbound_tls.posts[0][0] == path
    assert await transport.store.backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(1,)]


@pytest.mark.parametrize("outbound_tls", ["wrong.example"], indirect=True)
async def test_callback_certificate_failure_retires_without_sending_credentials(
    push_store, outbound_tls
):
    transport, configuration, task = await configured(
        push_store, outbound_tls.url + "/callback"
    )
    with pytest.raises(httpx.ConnectError):
        await transport.send(configuration, task, executor_id="runtime")
    assert outbound_tls.posts == []
    assert await transport.store.backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(1,)]


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8/callback",
        "https://127.0.0.1/callback",
        "https://user:password@example.com/callback",
    ],
)
async def test_callback_url_refusal_has_no_tcp_effect(push_store, url, monkeypatch):
    from httpcore._backends.anyio import AnyIOBackend

    async def refuse_tcp(*_args, **_kwargs):
        raise AssertionError("Unsafe callback attempted a TCP connection")

    monkeypatch.setattr(AnyIOBackend, "connect_tcp", refuse_tcp)
    transport, configuration, task = await configured(push_store, url)
    with pytest.raises(SSRFError):
        await transport.send(configuration, task, executor_id="runtime")
    assert await transport.store.backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(1,)]


async def test_deleted_generation_refuses_before_dns_or_https(push_store, outbound_tls):
    transport, configuration, task = await configured(
        push_store, outbound_tls.url + "/callback"
    )
    await transport.store.begin_delete(
        task.id, configuration.config.id, creator_id="did:creator"
    )
    with pytest.raises(PushDeliveryRevoked):
        await transport.send(configuration, task, executor_id="runtime")
    assert outbound_tls.dns == [] and outbound_tls.posts == []
    assert (
        await transport.store.backend.fetch_all(
            "SELECT attempt_id FROM a2a_push_delivery_attempts"
        )
        == []
    )


@pytest.mark.parametrize(
    "fault", ["token", "scheme", "credentials", "missing-credentials"]
)
async def test_invalid_callback_headers_refuse_before_delivery_admission(
    push_store, outbound_tls, fault
):
    store, task = push_store
    value = config(task)
    value.url = outbound_tls.url + "/callback"
    if fault == "token":
        value.token = "secret\r\nInjected: yes"
    elif fault == "scheme":
        value.authentication.scheme = "Bearer bad"
    elif fault == "credentials":
        value.authentication.credentials = "secret\nInjected: yes"
    else:
        value.authentication.credentials = ""
    configuration = await store.create(value, creator_id="did:creator")
    with pytest.raises(PushConfigConflict):
        await NativePushTransport(store, timeout=timedelta(seconds=5)).send(
            configuration, task, executor_id="runtime"
        )
    assert outbound_tls.dns == [] and outbound_tls.posts == []
    assert (
        await store.backend.fetch_all(
            "SELECT attempt_id FROM a2a_push_delivery_attempts"
        )
        == []
    )


async def test_lost_admission_commit_ack_keeps_original_error_without_stranding_http_custody(
    push_store,
    outbound_tls,
    monkeypatch,
):
    from contextlib import asynccontextmanager

    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    transport, configuration, task = await configured(
        push_store, outbound_tls.url + "/callback"
    )
    backend = transport.store.backend
    native_transaction = backend.transaction
    original = ExecutionCommitOutcomeError("unknown")
    armed = True

    @asynccontextmanager
    async def lost_commit_ack(*args, **kwargs):
        nonlocal armed
        async with native_transaction(*args, **kwargs):
            yield
        if armed:
            armed = False
            # The real native COMMIT already completed. Only its caller's
            # acknowledgement is missing; a rollback would be a false proof.
            raise original

    monkeypatch.setattr(backend, "transaction", lost_commit_ack)
    with pytest.raises(ExecutionCommitOutcomeError) as captured:
        await transport.send(configuration, task, executor_id="runtime")
    assert captured.value is original
    assert outbound_tls.dns == [] and outbound_tls.posts == []
    assert await backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(1,)]


async def test_revocation_after_dns_is_checked_again_at_actual_dispatch(
    push_store, outbound_tls, monkeypatch
):
    import kestrel_sovereign.a2a.push_transport as module

    transport, configuration, task = await configured(
        push_store, outbound_tls.url + "/callback"
    )
    native_validate = module.assert_safe_url

    async def revoke_after_dns(*args, **kwargs):
        validated = await native_validate(*args, **kwargs)
        await transport.store.begin_delete(
            task.id, configuration.config.id, creator_id="did:creator"
        )
        return validated

    monkeypatch.setattr(module, "assert_safe_url", revoke_after_dns)
    with pytest.raises(PushDeliveryRevoked):
        await transport.send(configuration, task, executor_id="runtime")
    assert outbound_tls.posts == [] and outbound_tls.dials == []
    assert await transport.store.backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(1,)]


@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_delete_and_delivery_ack_wait_for_original_response_close(
    push_store, outbound_tls, outbound_close_barrier, cancel_caller
):
    import asyncio

    transport, configuration, task = await configured(
        push_store, outbound_tls.url + "/callback"
    )
    close_entered, release_close = (
        outbound_close_barrier.entered,
        outbound_close_barrier.release,
    )
    sending = asyncio.create_task(
        transport.send(configuration, task, executor_id="runtime")
    )
    witness = asyncio.create_task(close_entered.wait())
    try:
        await asyncio.wait_for(
            asyncio.wait({sending, witness}, return_when=asyncio.FIRST_COMPLETED), 5
        )
        assert close_entered.is_set(), (
            "Push returned without closing its original response"
        )
        assert not sending.done()
        assert await transport.store.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == [(0,)]
        retiring, = await transport.store.begin_delete(
            task.id, configuration.config.id, creator_id="did:creator"
        )
        assert not await transport.store.finish_delete(retiring)
        if cancel_caller:
            sending.cancel("push caller withdrew during physical close")
        release_close.set()
        if cancel_caller:
            with pytest.raises(asyncio.CancelledError) as captured:
                await sending
            assert captured.value.args == (
                "push caller withdrew during physical close",
            )
        else:
            await sending
        assert await transport.store.finish_delete(retiring)
    finally:
        release_close.set()
        await asyncio.gather(sending, witness, return_exceptions=True)


async def test_failed_pool_close_keeps_attempt_unretired_and_blocks_delete(
    push_store, outbound_tls, monkeypatch
):
    import kestrel_sovereign.a2a.push_transport as module
    from kestrel_sovereign.execution_custody import execution_cleanup_debts

    transport, configuration, task = await configured(
        push_store, outbound_tls.url + "/callback"
    )
    native_transport = module.pinned_httpx_async_transport
    original = OSError("original push pool close acknowledgement failed")
    closes = []

    def failing_pool(validated):
        pool = native_transport(validated)
        native_close = pool.aclose

        async def failed_close():
            await native_close()
            closes.append("physical-pool-close")
            raise original

        pool.aclose = failed_close
        return pool

    monkeypatch.setattr(module, "pinned_httpx_async_transport", failing_pool)
    with pytest.raises(BaseException) as captured:
        await transport.send(configuration, task, executor_id="runtime")
    assert closes == ["physical-pool-close"]
    assert execution_cleanup_debts(captured.value)
    assert len(outbound_tls.posts) == 1
    assert await transport.store.backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(0,)]
    retiring, = await transport.store.begin_delete(
        task.id, configuration.config.id, creator_id="did:creator"
    )
    assert not await transport.store.finish_delete(retiring)


async def test_slow_callback_headers_are_bounded_and_original_socket_retires(
    push_store, outbound_tls
):
    transport, configuration, task = await configured(
        push_store, outbound_tls.url + "/callback"
    )
    transport = NativePushTransport(
        transport.store, timeout=timedelta(milliseconds=100)
    )
    outbound_tls.reply.clear()
    try:
        with pytest.raises((TimeoutError, httpx.ReadTimeout)):
            await transport.send(configuration, task, executor_id="runtime")
        assert len(outbound_tls.posts) == 1
        assert await transport.store.backend.fetch_all(
            "SELECT retired FROM a2a_push_delivery_attempts"
        ) == [(1,)]
    finally:
        outbound_tls.reply.set()


@pytest.mark.parametrize("timeout", [None, timedelta(), timedelta(seconds=-1)])
async def test_invalid_timeout_policy_fails_without_admission(push_store, timeout):
    store, _task = push_store
    with pytest.raises(ValueError, match="positive"):
        NativePushTransport(store, timeout=timeout)
    assert (
        await store.backend.fetch_all(
            "SELECT attempt_id FROM a2a_push_delivery_attempts"
        )
        == []
    )


@pytest.mark.parametrize("retry_delay", [timedelta(), timedelta(seconds=-1), True])
async def test_invalid_retirement_retry_delay_refuses_before_admission(push_store, retry_delay):
    store, _task = push_store
    with pytest.raises(ValueError, match="retirement retry delay"):
        NativePushTransport(store, timeout=timedelta(seconds=5), retry_delay=retry_delay)
    assert await store.backend.fetch_all("SELECT attempt_id FROM a2a_push_delivery_attempts") == []


async def test_untyped_retirement_policy_refuses_before_admission(push_store):
    store, _task = push_store
    with pytest.raises(TypeError, match="retirement requires its database retry policy"):
        NativePushTransport(store, timeout=timedelta(seconds=5), database_retry_policy=True)
    assert await store.backend.fetch_all("SELECT attempt_id FROM a2a_push_delivery_attempts") == []


@pytest.mark.parametrize("fault", ["task", "private-model"])
async def test_push_cannot_send_a_different_task_or_a_private_wire_shape(
    push_store, outbound_tls, fault
):
    from a2a.types import Task

    transport, configuration, task = await configured(
        push_store, outbound_tls.url + "/callback"
    )
    event = Task.FromString(task.SerializeToString())
    if fault == "task":
        event.id = "other-task"
    else:
        event = {"id": task.id, "sessionId": "draft"}
    with pytest.raises(PushConfigConflict if fault == "task" else TypeError):
        await transport.send(configuration, event, executor_id="runtime")
    assert outbound_tls.posts == [] and outbound_tls.dns == []
    assert (
        await transport.store.backend.fetch_all(
            "SELECT attempt_id FROM a2a_push_delivery_attempts"
        )
        == []
    )


async def test_incremental_headers_cannot_extend_the_total_exchange_deadline(
    push_store, outbound_tls
):
    transport, configuration, task = await configured(
        push_store, outbound_tls.url + "/drip-headers"
    )
    transport = NativePushTransport(transport.store, timeout=timedelta(seconds=1))
    with pytest.raises(TimeoutError):
        await transport.send(configuration, task, executor_id="runtime")
    assert len(outbound_tls.posts) == 1
    assert await transport.store.backend.fetch_all(
        "SELECT retired FROM a2a_push_delivery_attempts"
    ) == [(1,)]
