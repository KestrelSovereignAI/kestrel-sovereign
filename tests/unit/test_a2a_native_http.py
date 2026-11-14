"""Actual SDK clients and HTTP bindings over native proof, policy and storage."""

import asyncio
import contextvars
import inspect
import os
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from kestrel_sdk.native_a2a.client import ClientConfig, ClientFactory
from kestrel_sdk.native_a2a.client.client import ClientCallContext
from kestrel_sdk.native_a2a.client.errors import A2AClientError
from kestrel_sdk.native_a2a.client.interceptors import BeforeArgs
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_artifact
from kestrel_sdk.native_a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
    CancelTaskRequest,
    DeleteTaskPushNotificationConfigRequest,
    GetExtendedAgentCardRequest,
    GetTaskPushNotificationConfigRequest,
    GetTaskRequest,
    ListTaskPushNotificationConfigsRequest,
    ListTasksRequest,
    SubscribeToTaskRequest,
    TaskPushNotificationConfig,
    TaskState,
)
from kestrel_sdk.native_a2a.utils.errors import (
    ContentTypeNotSupportedError,
    InvalidParamsError,
    PushNotificationNotSupportedError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from kestrel_sovereign.a2a.native_client import NativeSenderInterceptor
from kestrel_sovereign._async_ownership import OwnedTaskAdmission
from kestrel_sovereign.a2a.native_routes import (
    NativeA2AHTTP,
    NativeA2AHTTPRegistry,
    native_http_interfaces,
)
from kestrel_sovereign.a2a.native_sender import (
    SENDER_PROOF_HEADER,
    NativeSenderProofError,
)
from kestrel_sovereign.a2a.request_handler import NativeA2ARequestHandler
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import bind_native_execution_admission
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock
from tests.utils.native_a2a import manager_for
from tests.utils.native_a2a_security import (
    EXTENSIONS,
    NOW,
    PRINCIPAL,
    SENDER,
    send,
    signed_context,
)


@pytest_asyncio.fixture(params=["JSONRPC", "HTTP+JSON"])
async def wire(native_request_admission, request, native_push_runtime):
    admission = native_request_admission
    protocol = request.param
    root = "http://native-core.test"
    card = AgentCard(
        name="Native Kestrel",
        description="Native Kestrel HTTP contract",
        version="1",
        supported_interfaces=native_http_interfaces(
            root, routing_key=admission.guard.routing_key, scope=admission.scope
        ),
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=native_push_runtime is not None,
            extensions=[
                AgentExtension(uri=uri, required=True) for uri in sorted(EXTENSIONS)
            ],
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="message", name="Message", description="Receive work", tags=["test"]
            )
        ],
    )
    handler = NativeA2ARequestHandler(admission.manager, admission.guard, card)
    submitted = asyncio.Event()
    submissions = []

    def announce(snapshot):
        submissions.append(snapshot)
        submitted.set()

    admission.manager._on_task_submitted = announce
    binding = NativeA2AHTTP(handler, public_base_url=root)
    interceptor = NativeSenderInterceptor(
        keypair=admission.signing[0],
        sender=SENDER,
        recipient=admission.scope,
        extensions=EXTENSIONS,
        policy=admission.guard.proof_policy,
        classical_kid="key-1",
        pq_kid="key-2",
        clock=lambda: NOW,
        nonce=lambda: uuid4().hex,
    )
    trace = []

    async def capture(outgoing):
        trace.append(
            {
                "method": outgoing.method,
                "path": outgoing.url.path,
                "body": outgoing.content,
                "signed": "Kestrel-A2A-Sender-Proof" in outgoing.headers,
                "version": outgoing.headers.get("A2A-Version"),
            }
        )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(Starlette(routes=[binding.route])),
        base_url=root,
        headers={"A2A-Version": "1.0", "A2A-Extensions": ",".join(sorted(EXTENSIONS))},
        event_hooks={"request": [capture]},
    ) as http:

        def client(
            *,
            streaming=False,
            polling=True,
            selected_card=None,
            http_client=None,
            sender_interceptor=None,
        ):
            return ClientFactory(
                ClientConfig(
                    httpx_client=http if http_client is None else http_client,
                    streaming=streaming,
                    polling=polling,
                    supported_protocol_bindings=[protocol],
                )
            ).create(
                card if selected_card is None else selected_card,
                interceptors=[
                    interceptor if sender_interceptor is None else sender_interceptor
                ],
            )

        yield SimpleNamespace(
            admission=admission,
            manager=admission.manager,
            handler=handler,
            card=card,
            client=client,
            http=http,
            protocol=protocol,
            trace=trace,
            interceptor=interceptor,
            binding=binding,
            submitted=submitted,
            submissions=submissions,
        )


async def submit(wire):
    params = send(wire.admission)
    events = [event async for event in wire.client().send_message(params)]
    assert len(events) == 1 and events[0].HasField("task")
    return events[0].task


@asynccontextmanager
async def observed_native_http(wire, *, binding=None):
    """Observe real server control separately from its serialized HTTP status."""
    failures, statuses, bodies = [], [], []
    original_app = Starlette(routes=[(wire.binding if binding is None else binding).route])

    async def checked_app(scope, receive, send_response):
        async def checked_send(message):
            if message["type"] == "http.response.start":
                assert not wire.admission.lock.locked()
                statuses.append(message["status"])
            elif message["type"] == "http.response.body":
                bodies.append(message.get("body", b""))
            await send_response(message)

        try:
            await original_app(scope, receive, checked_send)
        except BaseException as error:
            failures.append(error)
            raise

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(checked_app, raise_app_exceptions=False),
        base_url="http://native-core.test",
        headers=dict(wire.http.headers),
    ) as actual:
        yield SimpleNamespace(http=actual, failures=failures, statuses=statuses, bodies=bodies)


@pytest.mark.parametrize(
    "headers,query,accepted",
    [
        ([], ["1.0.9"], True),
        ([], ["0.3"], False),
        (["1.0"], ["0.3"], False),
        (["0.3"], ["1.0"], False),
        (["1.0", "0.3"], ["1.0"], False),
        ([], ["1.0", "0.3"], False),
        (["1.0"], ["1.0.9"], True),
        ([], [], False),
    ],
)
async def test_signed_http_version_carriers_are_validated_before_task_admission(
    wire, headers, query, accepted
):
    from kestrel_sdk.native_a2a.utils.errors import VersionNotSupportedError

    async def carry_version(outgoing):
        # Leave the actual SDK request and DID proof intact. Choose only the
        # standard HTTP version carriers at the real transport boundary.
        outgoing.headers.pop("A2A-Version", None)
        outgoing.headers = httpx.Headers(
            [
                *outgoing.headers.multi_items(),
                *(("A2A-Version", value) for value in headers),
            ]
        )
        url = outgoing.url
        for value in query:
            url = url.copy_add_param("A2A-Version", value)
        outgoing.url = url

    wire.http.event_hooks["request"].append(carry_version)
    if accepted:
        task = await submit(wire)
        assert task.id and task.status.HasField("timestamp")
        assert len(wire.submissions) == 1
    else:
        with pytest.raises(VersionNotSupportedError):
            await submit(wire)
        assert wire.submissions == []
        assert (
            await wire.manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks")
            == []
        )


async def test_inline_callback_requires_the_advertised_push_capability(wire):
    params = send(wire.admission)
    params.configuration.task_push_notification_config.url = "https://8.8.8.8/callback"
    with pytest.raises(PushNotificationNotSupportedError):
        _ = [event async for event in wire.client().send_message(params)]
    assert wire.submissions == []


async def test_empty_text_returns_native_invalid_params_before_admission(wire):
    params = send(wire.admission)
    params.message.parts[0].text = ""
    with pytest.raises(InvalidParamsError):
        _ = [event async for event in wire.client().send_message(params)]
    assert not wire.submissions
    assert (
        await wire.manager.task_store.backend.fetch_all("SELECT id FROM a2a_tasks")
        == []
    )
    assert (
        await wire.manager.task_store.backend.fetch_all(
            "SELECT context_id FROM a2a_contexts"
        )
        == []
    )


async def test_sdk_send_get_list_and_cancel_share_native_durable_authority(wire):
    task = await submit(wire)
    assert task.status.state == TaskState.TASK_STATE_SUBMITTED
    assert task.id and task.context_id
    client = wire.client()
    fetched = await client.get_task(
        GetTaskRequest(id=task.id, tenant=wire.admission.scope.tenant, history_length=0)
    )
    assert fetched.id == task.id and not fetched.history
    listing = await client.list_tasks(
        ListTasksRequest(
            tenant=wire.admission.scope.tenant, page_size=1, history_length=0
        )
    )
    assert [item.id for item in listing.tasks] == [task.id]
    assert (
        listing.total_size == 1
        and listing.page_size == 1
        and listing.next_page_token == ""
    )
    canceled = await client.cancel_task(
        CancelTaskRequest(id=task.id, tenant=wire.admission.scope.tenant)
    )
    assert canceled.status.state == TaskState.TASK_STATE_CANCELED
    snapshot = await wire.manager.get_task_snapshot(
        task.id, recipient_agent_id=wire.admission.scope.recipient_id
    )
    assert snapshot.ownership.creator_agent_id == PRINCIPAL
    assert snapshot.task == canceled
    assert all(row["signed"] and row["version"] == "1.0" for row in wire.trace)
    assert (
        b"sessionId" not in wire.trace[0]["body"]
        and b'"type"' not in wire.trace[0]["body"]
    )


async def test_unknown_or_foreign_task_has_the_same_native_not_found_result(wire):
    other = await wire.manager.create_task(
        send(wire.admission), creator_agent_id="did:kestrel:other-principal"
    )
    for identity in ("absent", other.id):
        with pytest.raises(TaskNotFoundError):
            await wire.client().get_task(
                GetTaskRequest(id=identity, tenant=wire.admission.scope.tenant)
            )
    listing = await wire.client().list_tasks(
        ListTasksRequest(tenant=wire.admission.scope.tenant)
    )
    assert not listing.tasks and listing.total_size == 0


@pytest.mark.parametrize("streaming", [False, True])
async def test_foreign_context_is_denied_without_a_task_or_submission_wake(
    wire, streaming
):
    other = await wire.manager.create_task(
        send(wire.admission), creator_agent_id="did:kestrel:other-principal"
    )
    wire.submissions.clear()
    params = send(wire.admission)
    params.message.context_id = other.context_id
    with pytest.raises(A2AClientError, match="403"):
        async for _ in wire.client(streaming=streaming).send_message(params):
            pytest.fail("A foreign context admitted a task")
    rows = await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks")
    assert rows == [(other.id,)]
    assert not wire.submissions
    ownership = await wire.manager.context_store.require_owned(
        other.context_id,
        wire.admission.scope,
        creator_agent_id="did:kestrel:other-principal",
    )
    assert ownership.creator_agent_id == "did:kestrel:other-principal"


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate"])
async def test_handler_card_and_guard_require_the_same_extensions(wire, fault):
    card = AgentCard.FromString(wire.card.SerializeToString())
    if fault == "missing":
        del card.capabilities.extensions[0]
    elif fault == "extra":
        card.capabilities.extensions.append(
            AgentExtension(uri="https://peer.test/unsupported", required=True)
        )
    else:
        card.capabilities.extensions.append(card.capabilities.extensions[0])
    with pytest.raises(ValueError, match="extensions"):
        NativeA2ARequestHandler(wire.manager, wire.admission.guard, card)


async def test_peer_admission_uses_only_complete_registered_sdk_routes(wire):
    prefix = wire.binding.prefix
    tenant = wire.admission.scope.tenant
    allowed = [
        ("POST", "/rpc"),
        ("POST", f"/{tenant}/message:send"),
        ("POST", f"/{tenant}/message:stream"),
        ("GET", f"/{tenant}/tasks"),
        ("GET", f"/{tenant}/tasks/task"),
        ("POST", f"/{tenant}/tasks/task:cancel"),
        ("GET", f"/{tenant}/tasks/task:subscribe"),
        ("POST", f"/{tenant}/tasks/task:subscribe"),
        ("POST", f"/{tenant}/tasks/task/pushNotificationConfigs"),
        ("GET", f"/{tenant}/tasks/task/pushNotificationConfigs"),
        ("GET", f"/{tenant}/tasks/task/pushNotificationConfigs/push"),
        ("DELETE", f"/{tenant}/tasks/task/pushNotificationConfigs/push"),
        ("GET", f"/{tenant}/extendedAgentCard"),
    ]
    denied = [
        ("GET", "/rpc"),
        ("POST", "/tasks/send"),
        ("GET", "/tasks/task"),
        ("POST", f"/{tenant}/tasks/task"),
        ("GET", "/foreign/tasks/task"),
        ("GET", f"/{tenant}/unknown"),
        ("POST", f"/{tenant}/rpc"),
        ("GET", f"/{tenant}/tasks/task/subscribe"),
    ]
    for expected, cases in ((True, allowed), (False, denied)):
        for method, suffix in cases:
            for root_path in ("", "/reverse-proxy"):
                scope = {
                    "type": "http",
                    "method": method,
                    "path": root_path + prefix + suffix,
                    "root_path": root_path,
                }
                before = dict(scope)
                assert wire.binding.admits(scope) is expected, (method, suffix)
                assert scope == before
    for path in (
        "/api/agent/tasks/send",
        "/api/agents/other/api/agent/tasks/send",
        prefix + "-other/rpc",
        prefix + "/unknown/tasks/task",
    ):
        assert not wire.binding.admits({"type": "http", "method": "POST", "path": path})
    assert not wire.binding.admits({"type": "websocket", "path": prefix + "/rpc"})
    for path in ("/foreign/tasks/task", "/tasks/task", f"/{tenant}/unknown"):
        response = await wire.http.get(prefix + path)
        assert response.status_code == 404
    assert wire.admission.state.authorizations == []


async def test_route_registration_rejects_a_card_pointing_at_another_mount(wire):
    card = wire.handler.agent_card
    card.supported_interfaces[0].url += "/wrong"
    other = NativeA2ARequestHandler(wire.manager, wire.admission.guard, card)
    with pytest.raises(ValueError, match="interfaces"):
        NativeA2AHTTP(other, public_base_url="http://native-core.test")
    assert (
        wire.handler.agent_card.supported_interfaces == wire.card.supported_interfaces
    )


async def test_default_tenant_uses_the_sdk_base_routes_without_tenant_aliases(wire):
    scope = NativeRecipientScope(wire.admission.scope.recipient_id)
    guard = replace(wire.admission.guard, scope=scope)
    manager = manager_for(wire.admission.backend, scope, owns_backend=False)
    await manager.initialize()
    try:
        card = wire.handler.agent_card
        del card.supported_interfaces[:]
        card.supported_interfaces.extend(
            native_http_interfaces(
                "http://native-core.test", routing_key=guard.routing_key, scope=scope
            )
        )
        handler = NativeA2ARequestHandler(manager, guard, card)
        binding = NativeA2AHTTP(handler, public_base_url="http://native-core.test")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(Starlette(routes=[binding.route])),
            base_url="http://native-core.test",
            headers={"A2A-Version": "1.0"},
        ) as http:
            client = wire.client(
                http_client=http,
                selected_card=card,
                sender_interceptor=replace(wire.interceptor, recipient=scope),
            )
            params = send(wire.admission)
            params.tenant = ""
            events = [event async for event in client.send_message(params)]
            task = await client.get_task(GetTaskRequest(id=events[0].task.id))
            assert task.id == events[0].task.id
            assert binding.admits(
                {"type": "http", "method": "GET", "path": binding.prefix + "/tasks"}
            )
            assert not binding.admits(
                {
                    "type": "http",
                    "method": "GET",
                    "path": binding.prefix + "/tenant-alpha/tasks",
                }
            )
            response = await http.get(binding.prefix + "/tenant-alpha/tasks")
            assert response.status_code == 404
    finally:
        await manager.close()


async def test_actual_dispatcher_preserves_duplicate_proof_headers_for_rejection(wire):
    params = GetTaskRequest(id="unknown", tenant=wire.admission.scope.tenant)
    args = BeforeArgs(params, "get_task", wire.card)
    await wire.interceptor.before(args)
    headers = list(args.context.service_parameters.items())
    headers.append(
        (SENDER_PROOF_HEADER, args.context.service_parameters[SENDER_PROOF_HEADER])
    )
    prefix = wire.binding.prefix
    if wire.protocol == "JSONRPC":
        response = await wire.http.post(
            prefix + "/rpc",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": "request",
                "method": "GetTask",
                "params": {"id": params.id, "tenant": params.tenant},
            },
        )
    else:
        response = await wire.http.get(
            prefix + "/tenant-alpha/tasks/unknown", headers=headers
        )
    assert response.status_code == 401
    assert not wire.admission.state.authorizations
    assert (
        await wire.admission.backend.fetch_all("SELECT nonce FROM a2a_replay_nonces")
        == []
    )


async def test_persistence_failure_is_control_without_exposing_sql_details(
    wire,
):
    await wire.admission.backend.execute("DROP TABLE native_test_principals")
    from kestrel_sovereign.execution_custody import execution_evidence_carriers, is_execution_control_error
    from kestrel_sovereign.storage.db.interface import QueryError

    async with observed_native_http(wire) as observed:
        with pytest.raises(A2AClientError, match="500") as failed:
            await wire.client(http_client=observed.http).get_task(
                GetTaskRequest(id="unknown", tenant=wire.admission.scope.tenant)
            )
    assert observed.statuses == [500] and len(observed.failures) == 1
    assert is_execution_control_error(observed.failures[0])
    assert any(isinstance(error, QueryError) for error in execution_evidence_carriers(observed.failures[0]))
    assert "native_test_principals" not in str(failed.value)
    assert "SELECT" not in str(failed.value)
    assert b"native_test_principals" not in b"".join(observed.bodies)
    assert b"SELECT" not in b"".join(observed.bodies)


async def test_registry_withdrawal_cannot_remove_a_newer_registration(wire):
    application = Starlette()
    registry = NativeA2AHTTPRegistry(application)
    scope = {
        "type": "http",
        "method": "POST",
        "path": wire.binding.prefix + "/rpc",
        "app": application,
    }
    assert not registry.admits(scope)
    registry.publish(wire.binding)
    assert registry.admits(scope)
    replacement = NativeA2AHTTP(wire.handler, public_base_url="http://native-core.test")
    registry.publish(replacement)
    assert application.routes == [replacement.route]
    assert not registry.withdraw(wire.binding)
    assert registry.admits(scope)
    assert registry.withdraw(replacement)
    assert not registry.admits(scope) and not application.routes


async def test_registry_refuses_foreign_apps_and_lost_or_conflicting_routes(wire):
    application = Starlette()
    registry = NativeA2AHTTPRegistry(application)
    scope = {
        "type": "http",
        "method": "POST",
        "path": wire.binding.prefix + "/rpc",
        "app": Starlette(),
    }
    with pytest.raises(ValueError, match="different application"):
        registry.admits(scope)
    registry.publish(wire.binding)
    other = NativeA2AHTTPRegistry(application)
    with pytest.raises(RuntimeError, match="already owned"):
        other.publish(wire.binding)
    application.routes.clear()
    scope["app"] = application
    with pytest.raises(RuntimeError, match="lost its published route"):
        registry.admits(scope)


@pytest.mark.parametrize("managed", [False, True])
async def test_real_host_auth_keeps_sdk_calls_on_the_peer_lane(
    wire, monkeypatch, managed
):
    from fastapi import FastAPI, Request
    from starlette.middleware.sessions import SessionMiddleware

    from kestrel_sovereign import server
    from kestrel_sovereign.auth import AuthMethod

    monkeypatch.setenv("KESTREL_A2A_TRANSPORT_ONLY", "true" if managed else "false")
    operator_checks = []

    def operator_key():
        operator_checks.append(True)
        return "native-test-operator-key"

    monkeypatch.setattr(server, "get_api_key", operator_key)
    application = FastAPI()
    registry = NativeA2AHTTPRegistry(application)
    registry.publish(wire.binding)
    application.state.native_a2a_http = registry
    callers = []

    @application.middleware("http")
    async def observe(request: Request, call_next):
        callers.append(request.state.caller)
        return await call_next(request)

    application.middleware("http")(server.auth_middleware)
    application.add_middleware(SessionMiddleware, secret_key="native-test-session")

    @application.get("/native-operator-check")
    async def operator_route():
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(application),
        base_url="http://native-core.test",
        headers={"A2A-Version": "1.0", "A2A-Extensions": ",".join(sorted(EXTENSIONS))},
    ) as http:
        client = wire.client(http_client=http)
        params = send(wire.admission)
        for credential in (None, "native-test-operator-key", "invalid-operator-key"):
            context = ClientCallContext(
                service_parameters={"X-API-Key": credential} if credential else {}
            )
            events = [
                event async for event in client.send_message(params, context=context)
            ]
            assert events[0].task.id
            assert callers[-1].auth_method == AuthMethod.A2A_TRANSPORT
            assert not callers[-1].is_sovereign
            assert callers[-1].credential_fingerprint is None
        assert not operator_checks
        path = wire.binding.prefix + (
            "/rpc" if wire.protocol == "JSONRPC" else "/tenant-alpha/tasks/unknown"
        )
        response = await http.request(
            "POST" if wire.protocol == "JSONRPC" else "GET",
            path,
            headers={"X-API-Key": "native-test-operator-key"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "GetTask",
                "params": {"id": "unknown", "tenant": wire.admission.scope.tenant},
            }
            if wire.protocol == "JSONRPC"
            else None,
        )
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith(
            "Kestrel-A2A-Sender-Proof"
        )
        assert not operator_checks
        response = await http.get(
            "/native-operator-check", headers={"Kestrel-A2A-Sender-Proof": "irrelevant"}
        )
        assert response.status_code == 401
        response = await http.get(
            "/native-operator-check", headers={"X-API-Key": "native-test-operator-key"}
        )
        assert response.status_code == (401 if managed else 200)
        if not managed:
            assert callers[-1].is_sovereign


async def test_revocation_before_the_first_stream_event_exposes_no_snapshot(wire):
    params = send(wire.admission)
    guard = wire.admission.guard
    async with AsyncExitStack() as streams:
        async with guard.admit(
            params, "SendMessage", signed_context(wire.admission, params)
        ) as admitted:
            task = await wire.manager.create_task(params, creator_agent_id=PRINCIPAL)
            stream, initial = await streams.enter_async_context(
                wire.handler._open_stream(task.id, PRINCIPAL)
            )
        wire.admission.state.current = False
        events = wire.handler._events(stream, initial, admitted)
        try:
            with pytest.raises(HTTPException) as denied:
                await anext(events)
            assert denied.value.status_code == 403
        finally:
            await events.aclose()
    assert not wire.manager._subscribers


async def test_terminal_cancel_and_subscribe_return_standard_lifecycle_errors(
    wire, monkeypatch
):
    sources = []
    subscribe = wire.manager.subscribe

    def observe_original(*args, **kwargs):
        source = subscribe(*args, **kwargs)
        sources.append(source)
        return source

    monkeypatch.setattr(wire.manager, "subscribe", observe_original)
    task = await submit(wire)
    await wire.manager.complete_task(
        task.id, "done", recipient_agent_id=wire.admission.scope.recipient_id
    )
    with pytest.raises(TaskNotCancelableError):
        await wire.client().cancel_task(
            CancelTaskRequest(id=task.id, tenant=wire.admission.scope.tenant)
        )
    with pytest.raises(UnsupportedOperationError):
        async for _ in wire.client(streaming=True).subscribe(
            SubscribeToTaskRequest(id=task.id, tenant=wire.admission.scope.tenant)
        ):
            pytest.fail("A terminal task started a new subscription")
    assert len(sources) == 1 and sources[0].ag_frame is None
    assert not wire.manager._subscribers


@pytest.mark.parametrize("operation", ["send", "subscribe"])
async def test_streaming_capability_refusal_keeps_standard_error_after_authority_exit(
    wire, operation
):
    # The real caller card remains streaming-capable to exercise the server's
    # independently held configuration rather than a client-side refusal.
    wire.handler._card.capabilities.streaming = False
    client = wire.client(streaming=True)
    with pytest.raises(UnsupportedOperationError):
        if operation == "send":
            async for _ in client.send_message(send(wire.admission)):
                pytest.fail("Unadvertised streaming dispatched task work")
        else:
            async for _ in client.subscribe(
                SubscribeToTaskRequest(id="missing", tenant=wire.admission.scope.tenant)
            ):
                pytest.fail("Unadvertised streaming opened a subscription")
    assert wire.admission.state.authorizations == [SENDER]
    assert not wire.admission.lock.locked()
    assert not wire.manager._subscribers
    assert await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []


async def test_missing_subscription_keeps_standard_error_after_original_source_close(
    wire,
):
    with pytest.raises(TaskNotFoundError):
        async for _ in wire.client(streaming=True).subscribe(
            SubscribeToTaskRequest(id="missing", tenant=wire.admission.scope.tenant)
        ):
            pytest.fail("A missing task exposed a subscription snapshot")
    assert not wire.admission.lock.locked()
    assert not wire.manager._subscribers


@pytest.mark.parametrize(
    "refusal,error_type",
    [
        ("history", InvalidParamsError),
        ("output-mode", ContentTypeNotSupportedError),
        ("input-mode", ContentTypeNotSupportedError),
        ("missing-continuation", TaskNotFoundError),
    ],
)
async def test_stream_send_validation_refusals_keep_protocol_errors(
    wire, refusal, error_type
):
    params = send(wire.admission)
    if refusal == "history":
        params.configuration.history_length = -1
    elif refusal == "output-mode":
        params.configuration.accepted_output_modes.append("image/png")
    elif refusal == "input-mode":
        params.message.parts[0].media_type = "image/png"
    else:
        params.message.task_id = "missing"
    with pytest.raises(error_type):
        async for _ in wire.client(streaming=True).send_message(params):
            pytest.fail("Invalid streaming input dispatched business work")
    assert not wire.admission.lock.locked()
    assert not wire.manager._subscribers
    assert await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []


async def test_domain_typed_actual_authority_exit_failure_is_not_protocol_refusal(
    wire, monkeypatch
):
    from contextlib import asynccontextmanager
    from kestrel_sovereign.a2a.stores.unified.task_store import TaskInputConflictError
    from kestrel_sovereign.execution_custody import (
        execution_evidence_carriers,
        is_execution_control_error,
    )

    guard_type = type(wire.admission.guard)
    hold = guard_type.hold_authority
    reserve = wire.manager.context_store.reserve
    original = TaskInputConflictError(
        "Actual native authority exit acknowledgement failed"
    )
    armed = False
    witnesses = []

    async def arm_after_original_reservation(*args, **kwargs):
        nonlocal armed
        result = await reserve(*args, **kwargs)
        armed = True
        return result

    @asynccontextmanager
    async def lose_original_authority_ack(guard, admitted, **options):
        nonlocal armed
        async with hold(guard, admitted, **options) as decision:
            yield decision
        if guard is wire.admission.guard and armed:
            armed = False
            assert not wire.admission.lock.locked()
            witnesses.append(asyncio.current_task())
            raise original

    monkeypatch.setattr(
        wire.manager.context_store, "reserve", arm_after_original_reservation
    )
    monkeypatch.setattr(guard_type, "hold_authority", lose_original_authority_ack)
    with pytest.raises(BaseException) as failed:
        async for _ in wire.client(streaming=True).send_message(send(wire.admission)):
            pytest.fail("An actual failed authority exit became protocol data")
    assert len(witnesses) == 1
    assert is_execution_control_error(failed.value)
    assert any(error is original for error in execution_evidence_carriers(failed.value))


@pytest.mark.parametrize(
    "domain",
    ["input-conflict", "input-unsupported", "push-unavailable", "context-conflict"],
)
async def test_domain_typed_actual_postgres_release_failure_is_not_protocol_refusal(
    wire, monkeypatch, domain
):
    """Physical error origin, not its domain class, decides source failure."""
    if wire.admission.backend.backend_type != "postgres":
        pytest.skip("The release witness is the actual PostgreSQL pool context")
    import asyncpg
    from kestrel_sovereign.a2a.stores.unified.context_store import ContextOwnershipError
    from kestrel_sovereign.a2a.push_service import PushServiceUnavailable
    from kestrel_sovereign.a2a.stores.unified.task_store import (
        TaskInputConflictError,
        TaskInputUnsupportedError,
    )
    from kestrel_sovereign.execution_custody import (
        execution_evidence_carriers,
        is_execution_control_error,
    )

    error_type = {
        "input-conflict": TaskInputConflictError,
        "input-unsupported": TaskInputUnsupportedError,
        "push-unavailable": PushServiceUnavailable,
        "context-conflict": ContextOwnershipError,
    }[domain]
    original = error_type("Actual PostgreSQL release acknowledgement failed")
    reserve = wire.manager.context_store.reserve
    exit_pool = asyncpg.pool.PoolAcquireContext.__aexit__
    armed = False
    witnesses = []

    async def arm_after_original_reservation(*args, **kwargs):
        nonlocal armed
        result = await reserve(*args, **kwargs)
        armed = True
        return result

    async def lose_original_release_ack(checkout, typ, error, traceback):
        nonlocal armed
        result = await exit_pool(checkout, typ, error, traceback)
        if armed and typ is None:
            armed = False
            witnesses.append(asyncio.current_task())
            raise original
        return result

    monkeypatch.setattr(
        wire.manager.context_store, "reserve", arm_after_original_reservation
    )
    with monkeypatch.context() as fault:
        fault.setattr(
            asyncpg.pool.PoolAcquireContext, "__aexit__", lose_original_release_ack
        )
        with pytest.raises(BaseException) as failed:
            async for _ in wire.client(streaming=True).send_message(
                send(wire.admission)
            ):
                pytest.fail("An actual failed database release became protocol data")
    assert len(witnesses) == 1
    assert is_execution_control_error(failed.value)
    assert any(error is original for error in execution_evidence_carriers(failed.value))


@pytest.mark.parametrize(
    "method,request_type,fields,error_type",
    [
        (
            "create_task_push_notification_config",
            TaskPushNotificationConfig,
            {"task_id": "task", "id": "push", "url": "https://callback.test/push"},
            PushNotificationNotSupportedError,
        ),
        (
            "get_task_push_notification_config",
            GetTaskPushNotificationConfigRequest,
            {"task_id": "task", "id": "push"},
            PushNotificationNotSupportedError,
        ),
        (
            "list_task_push_notification_configs",
            ListTaskPushNotificationConfigsRequest,
            {"task_id": "task"},
            PushNotificationNotSupportedError,
        ),
        (
            "delete_task_push_notification_config",
            DeleteTaskPushNotificationConfigRequest,
            {"task_id": "task", "id": "push"},
            PushNotificationNotSupportedError,
        ),
        (
            "get_extended_agent_card",
            GetExtendedAgentCardRequest,
            {},
            UnsupportedOperationError,
        ),
    ],
)
async def test_unadvertised_operations_authenticate_before_the_capability_error(
    wire, method, request_type, fields, error_type
):
    params = request_type(tenant=wire.admission.scope.tenant, **fields)
    selected_card = AgentCard.FromString(wire.card.SerializeToString())
    if method == "get_extended_agent_card":
        # Force an adversarial caller past the SDK's public-card cache. The
        # actual server card still does not advertise the extended operation.
        selected_card.capabilities.extended_agent_card = True
    operation = getattr(wire.client(selected_card=selected_card), method)
    with pytest.raises(error_type):
        await operation(params)
    assert wire.admission.state.authorizations == [SENDER]
    assert await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []
    await wire.admission.backend.execute("DELETE FROM native_test_principals")
    with pytest.raises(A2AClientError, match="403"):
        await operation(params)


@pytest.mark.parametrize("fault", ["input", "tenant", "method", "descriptor"])
async def test_sender_interceptor_rejects_invalid_calls_without_changing_context(
    wire, fault
):
    params = GetTaskRequest(id="task", tenant=wire.admission.scope.tenant)
    method = "get_task"
    error_type = NativeSenderProofError
    if fault == "input":
        params = {"id": "task"}
        error_type = TypeError
    elif fault == "tenant":
        params.tenant = "foreign"
    elif fault == "method":
        method = "nonexistent_operation"
    else:
        method = "cancel_task"
    context = ClientCallContext(service_parameters={"X-Call-Test": "present"})
    args = BeforeArgs(params, method, wire.card, context)
    with pytest.raises(error_type):
        await wire.interceptor.before(args)
    assert args.context is context
    assert context.service_parameters == {"X-Call-Test": "present"}
    assert not wire.trace
    assert "keypair=" not in repr(wire.interceptor)


async def test_blocking_sdk_send_returns_the_committed_result_without_holding_topology(
    wire,
):
    params = send(wire.admission)

    async def request():
        return [
            event async for event in wire.client(polling=False).send_message(params)
        ]

    pending = asyncio.create_task(request())
    try:
        await asyncio.wait_for(wire.submitted.wait(), 5)
        async with asyncio.timeout(5):
            async with wire.admission.lock:
                # Waiting for background completion releases the topology lease.
                pass
        task_id = wire.submissions[0].task.id
        await wire.manager.complete_task(
            task_id,
            "committed result",
            artifacts=[new_text_artifact("result", "committed result")],
            recipient_agent_id=wire.admission.scope.recipient_id,
        )
        events = await asyncio.wait_for(pending, 5)
        assert len(events) == 1
        assert events[0].task.status.state == TaskState.TASK_STATE_COMPLETED
        assert events[0].task.artifacts[0].parts[0].text == "committed result"
        assert not wire.manager._subscribers
    finally:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


async def test_streaming_sdk_send_preserves_task_artifacts_and_terminal_event(wire):
    async def request():
        return [
            event
            async for event in wire.client(streaming=True).send_message(
                send(wire.admission)
            )
        ]

    pending = asyncio.create_task(request())
    try:
        await asyncio.wait_for(wire.submitted.wait(), 5)
        task_id = wire.submissions[0].task.id
        await wire.manager.complete_task(
            task_id,
            "streamed result",
            artifacts=[new_text_artifact("result", "streamed result")],
            recipient_agent_id=wire.admission.scope.recipient_id,
        )
        events = await asyncio.wait_for(pending, 5)
        assert events[0].HasField("task") and events[0].task.id == task_id
        stored = await wire.manager.get_task_for_recipient(
            task_id, wire.admission.scope.recipient_id
        )
        assert stored.status.state == TaskState.TASK_STATE_COMPLETED
        assert stored.artifacts[0].parts[0].text == "streamed result"
        assert (
            events[-1].task.status.state == TaskState.TASK_STATE_COMPLETED
            if events[-1].HasField("task")
            else events[-1].status_update.status.state == TaskState.TASK_STATE_COMPLETED
        )
        assert not wire.manager._subscribers
    finally:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


async def test_native_http_stream_uses_actual_admitted_core_source_owner(wire):
    coordinator = ReentrantTransitionLock()
    root = asyncio.current_task()
    observed = []

    async def complete_original():
        await wire.submitted.wait()
        await wire.manager.complete_task(
            wire.submissions[0].task.id,
            "original native snapshot",
            recipient_agent_id=wire.admission.scope.recipient_id,
        )

    completion = asyncio.create_task(complete_original(), context=contextvars.Context())
    try:
        async with coordinator:
            scope = coordinator.current_native_scope()
            assert scope is not None

            async def require_original_membership():
                coordinator.require_native_scope(scope, retained_owners=(root,))
                observed.append(asyncio.current_task())

            with bind_native_execution_admission(require_original_membership):
                events = [
                    event
                    async for event in wire.client(streaming=True).send_message(
                        send(wire.admission)
                    )
                ]
            assert events and events[0].task.id == wire.submissions[0].task.id
            originals = set(observed) - {root}
            assert originals, "The HTTP source must execute in an actual admitted child"
            for task in originals:
                receipt = getattr(task, "_kestrel_owned_admission", None)
                assert type(receipt) is OwnedTaskAdmission and receipt.task is task
                assert task.done()
                assert inspect.getcoroutinestate(task.get_coro()) == inspect.CORO_CLOSED
        await completion
    finally:
        if not completion.done():
            completion.cancel()
        await asyncio.gather(completion, return_exceptions=True)


async def test_missing_proof_and_recipient_denial_keep_http_401_and_403(wire):
    path = wire.binding.prefix + (
        "/rpc" if wire.protocol == "JSONRPC" else "/tenant-alpha/tasks/unknown"
    )
    if wire.protocol == "JSONRPC":
        response = await wire.http.post(
            path,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "GetTask",
                "params": {"id": "unknown", "tenant": wire.admission.scope.tenant},
            },
        )
    else:
        response = await wire.http.get(path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Kestrel-A2A-Sender-Proof")
    await wire.admission.backend.execute("DELETE FROM native_test_principals")
    with pytest.raises(A2AClientError, match="403"):
        await wire.client().get_task(
            GetTaskRequest(id="unknown", tenant=wire.admission.scope.tenant)
        )
    assert await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []


async def test_sender_interceptor_preserves_the_callers_context_and_signs_each_request(
    wire,
):
    task = await submit(wire)
    context = ClientCallContext(
        state={"request": object()},
        timeout=2,
        service_parameters={"X-Call-Test": "present"},
    )
    original = dict(context.service_parameters)
    params = GetTaskRequest(id=task.id, tenant=wire.admission.scope.tenant)
    before = params.SerializeToString(deterministic=True)
    for _ in range(2):
        assert (await wire.client().get_task(params, context=context)).id == task.id
    assert params.SerializeToString(deterministic=True) == before
    assert context.service_parameters == original
    assert (
        len(
            await wire.admission.backend.fetch_all(
                "SELECT nonce FROM a2a_replay_nonces"
            )
        )
        == 3
    )


@pytest.mark.parametrize("mime", ["image/png", "bad", "*/json"])
async def test_native_server_refuses_unsupported_or_malformed_input_before_task_admission(
    wire, mime
):
    from kestrel_sdk.native_a2a.types import Part
    from kestrel_sdk.native_a2a.utils.errors import (
        ContentTypeNotSupportedError,
        InvalidParamsError,
    )

    params = send(wire.admission)
    params.message.parts.append(Part(raw=b"not fetched or admitted", media_type=mime))
    params.configuration.return_immediately = True
    with pytest.raises(
        ContentTypeNotSupportedError if mime == "image/png" else InvalidParamsError
    ):
        async for _ in wire.client(polling=False).send_message(params):
            pytest.fail("Invalid input must not return an accepted native task")
    assert not wire.submissions
    listed = await wire.manager.task_store.list_for_creator(
        ListTasksRequest(tenant=wire.admission.scope.tenant),
        creator_agent_id=PRINCIPAL,
        recipient_agent_id=wire.admission.scope.recipient_id,
    )
    assert not listed.tasks


async def test_native_metadata_cannot_activate_its_own_optional_extension(wire):
    from kestrel_sdk.native_a2a.utils.errors import InvalidParamsError

    from kestrel_sovereign.a2a.extensions import CAUSATION_URI

    params = send(wire.admission)
    params.configuration.return_immediately = True
    params.metadata[CAUSATION_URI] = {"version": 1, "chain": []}
    with pytest.raises(InvalidParamsError, match="not activated"):
        async for _ in wire.client(polling=False).send_message(params):
            pytest.fail("URI metadata is not extension negotiation")
    assert not wire.submissions


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("invalid", ["agent-role", "empty-text"])
async def test_canonical_input_refusals_remain_standard_before_any_admission(
    wire, streaming, invalid
):
    from kestrel_sdk.native_a2a.types import Role
    from kestrel_sovereign.execution_custody import is_execution_control_error

    params = send(wire.admission)
    if invalid == "agent-role":
        params.message.role = Role.ROLE_AGENT
    else:
        params.message.parts[0].text = ""
    with pytest.raises(InvalidParamsError) as refused:
        async for _ in wire.client(streaming=streaming).send_message(params):
            pytest.fail("Invalid input must not admit or return a task")
    assert not is_execution_control_error(refused.value)
    assert not wire.submissions
    assert await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []
    assert not wire.manager._subscribers
    assert not wire.admission.lock.locked()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("fail_original_exit", [False, True])
async def test_pre_effect_policy_recheck_settles_before_http_refusal(
    wire, monkeypatch, streaming, continuation, fail_original_exit
):
    from contextlib import asynccontextmanager
    from kestrel_sovereign.execution_custody import (
        execution_evidence_carriers,
        is_execution_control_error,
    )

    params = send(wire.admission)
    params.configuration.return_immediately = True
    if continuation:
        original_task = await wire.manager.create_task(
            send(wire.admission), creator_agent_id=PRINCIPAL
        )
        params.message.task_id = original_task.id
        params.message.context_id = original_task.context_id
    before = await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks")
    submissions = len(wire.submissions)
    original_lease = wire.handler.guard.lifecycle_lease
    exits = []
    physical = HTTPException(403, detail="Actual recheck lease exit failed")

    @asynccontextmanager
    async def revoke_after_initial_lease():
        try:
            async with original_lease():
                yield
        finally:
            exits.append(asyncio.current_task())
            if len(exits) == 1:
                await wire.admission.backend.execute("DELETE FROM native_test_principals")
            elif len(exits) == 2 and fail_original_exit:
                raise physical

    monkeypatch.setattr(
        wire.handler,
        "guard",
        replace(wire.handler.guard, lifecycle_lease=revoke_after_initial_lease),
    )
    async with observed_native_http(wire) as observed:
        with pytest.raises(A2AClientError, match="500" if fail_original_exit else "403"):
            async for _ in wire.client(streaming=streaming, http_client=observed.http).send_message(params):
                pytest.fail("A revoked pre-effect policy dispatched business work")
    assert observed.statuses == [500 if fail_original_exit else 403]
    assert len(exits) == 2
    if fail_original_exit:
        assert len(observed.failures) == 1 and is_execution_control_error(observed.failures[0])
        assert any(error is physical for error in execution_evidence_carriers(observed.failures[0]))
    else:
        assert not observed.failures
    assert len(wire.submissions) == submissions
    assert await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == before
    assert not wire.manager._subscribers
    assert not wire.admission.lock.locked()


@pytest.mark.parametrize("operation", ["unary-immediate", "unary-blocking", "streaming"])
@pytest.mark.parametrize("continuation", [False, True])
async def test_post_commit_policy_loss_retains_outcome_not_ordinary_refusal(
    wire, monkeypatch, operation, continuation
):
    from kestrel_sovereign.execution_custody import (
        execution_commit_outcome,
        is_execution_control_error,
    )

    params = send(wire.admission)
    params.configuration.return_immediately = operation == "unary-immediate"
    if continuation:
        original = await wire.manager.create_task(send(wire.admission), creator_agent_id=PRINCIPAL)
        params.message.task_id = original.id
        params.message.context_id = original.context_id
    submissions = len(wire.submissions)
    announce = wire.manager._on_task_submitted

    def revoke_after_commit(snapshot):
        announce(snapshot)
        wire.admission.state.current = False

    monkeypatch.setattr(wire.manager, "_on_task_submitted", revoke_after_commit)
    async with observed_native_http(wire) as observed:
        with pytest.raises(A2AClientError, match="500"):
            async for _ in wire.client(
                streaming=operation == "streaming", polling=False, http_client=observed.http
            ).send_message(params):
                pytest.fail("Post-commit authority loss delivered a normal response")
    assert observed.statuses == [500]
    assert len(observed.failures) == 1 and is_execution_control_error(observed.failures[0])
    assert execution_commit_outcome(observed.failures[0]) == "committed"
    assert len(wire.submissions) == submissions + 1
    rows = await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks")
    assert len(rows) == 1 and rows[0][0] == wire.submissions[-1].task.id
    assert not wire.manager._subscribers and not wire.admission.lock.locked()


async def test_blocking_final_response_recheck_retains_committed_outcome(wire, monkeypatch):
    from kestrel_sovereign.execution_custody import (
        execution_commit_outcome,
        is_execution_control_error,
    )

    subscribe = wire.manager.subscribe

    async def revoke_after_original_eof(*args, **kwargs):
        source = subscribe(*args, **kwargs)
        try:
            async for event in source:
                yield event
        finally:
            await source.aclose()
            wire.admission.state.current = False

    monkeypatch.setattr(wire.manager, "subscribe", revoke_after_original_eof)
    params = send(wire.admission)
    params.configuration.return_immediately = False
    async with observed_native_http(wire) as observed:
        async def request():
            async for _ in wire.client(polling=False, http_client=observed.http).send_message(params):
                pytest.fail("Final post-commit authority loss delivered a normal response")

        original_request = asyncio.create_task(request())
        try:
            async with asyncio.timeout(5):
                await wire.submitted.wait()
            task_id = wire.submissions[-1].task.id
            # Actual durable completion, not a fake EOF or source cancellation.
            await wire.manager.complete_task(
                task_id, "actual test completion", recipient_agent_id=wire.admission.scope.recipient_id
            )
            outcome, = await asyncio.gather(original_request, return_exceptions=True)
            assert isinstance(outcome, A2AClientError) and "500" in str(outcome)
        finally:
            async with asyncio.timeout(5):
                await asyncio.gather(original_request, return_exceptions=True)
    assert observed.statuses == [500]
    assert len(observed.failures) == 1 and is_execution_control_error(observed.failures[0])
    assert execution_commit_outcome(observed.failures[0]) == "committed"
    snapshot = await wire.manager.get_task_snapshot(task_id, recipient_agent_id=wire.admission.scope.recipient_id)
    assert snapshot.task.status.state == TaskState.TASK_STATE_COMPLETED
    assert len(wire.submissions) == 1
    assert not wire.manager._subscribers and not wire.admission.lock.locked()


@pytest.mark.parametrize("streaming", [False, True])
async def test_post_commit_missing_subscription_retains_committed_outcome(wire, monkeypatch, streaming):
    from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error

    subscribe = wire.manager.subscribe
    originals = []

    async def erased_subscription(task_id, **kwargs):
        # Erase ONLY this test's actual committed row after submission and
        # before the canonical authorized read, never fabricate an empty source.
        assert len(wire.submissions) == 1 and wire.submissions[0].task.id == task_id
        await wire.admission.backend.execute("DELETE FROM a2a_tasks WHERE id = ?", (task_id,))
        source = subscribe(task_id, **kwargs)
        originals.append(source)
        try:
            async for event in source:
                yield event
        finally:
            await source.aclose()

    monkeypatch.setattr(wire.manager, "subscribe", erased_subscription)
    params = send(wire.admission)
    params.configuration.return_immediately = False
    async with observed_native_http(wire) as observed:
        with pytest.raises(A2AClientError, match="500"):
            async for _ in wire.client(streaming=streaming, polling=False, http_client=observed.http).send_message(params):
                pytest.fail("A committed task disappearance delivered a normal response")
    assert observed.statuses == [500]
    assert len(observed.failures) == 1 and is_execution_control_error(observed.failures[0])
    assert execution_commit_outcome(observed.failures[0]) == "committed"
    assert len(originals) == 1 and originals[0].ag_frame is None
    assert len(wire.submissions) == 1
    assert await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []
    assert not wire.manager._subscribers and not wire.admission.lock.locked()


@pytest.mark.parametrize("operation,streaming", [("send", False), ("send", True), ("subscribe", True)])
async def test_first_event_policy_recheck_retains_committed_outcome(wire, monkeypatch, operation, streaming):
    from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error

    original = wire.handler.guard.lifecycle_lease

    @asynccontextmanager
    async def revoke_after_acquisition():
        async with original():
            yield
        if wire.manager._subscribers:
            wire.admission.state.current = False

    monkeypatch.setattr(wire.handler, "guard", replace(wire.handler.guard, lifecycle_lease=revoke_after_acquisition))
    params = send(wire.admission)
    params.configuration.return_immediately = False
    if operation == "subscribe":
        task = await wire.manager.create_task(send(wire.admission), creator_agent_id=PRINCIPAL)
        params = SubscribeToTaskRequest(id=task.id, tenant=wire.admission.scope.tenant)
    async with observed_native_http(wire) as observed:
        with pytest.raises(A2AClientError, match="500"):
            client = wire.client(streaming=streaming, polling=False, http_client=observed.http)
            source = client.send_message(params) if operation == "send" else client.subscribe(params)
            async for _ in source:
                pytest.fail("Post-acquisition authority loss delivered a normal response")
    assert observed.statuses == [500]
    assert len(observed.failures) == 1 and is_execution_control_error(observed.failures[0])
    assert execution_commit_outcome(observed.failures[0]) == ("committed" if operation == "send" else None)
    assert len(wire.submissions) == 1
    rows = await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks")
    assert len(rows) == 1 and rows[0][0] == wire.submissions[0].task.id
    assert not wire.manager._subscribers and not wire.admission.lock.locked()


@pytest.mark.parametrize("phase", ["admission", "acquisition"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("continuation", [False, True])
async def test_actual_lease_exit_after_commit_retains_outcome_and_submission(
    wire, monkeypatch, phase, streaming, continuation
):
    from kestrel_sovereign.execution_custody import (
        execution_commit_outcome,
        execution_evidence_carriers,
        is_execution_control_error,
    )

    params = send(wire.admission)
    params.configuration.return_immediately = False
    if continuation:
        original_task = await wire.manager.create_task(send(wire.admission), creator_agent_id=PRINCIPAL)
        params.message.task_id = original_task.id
        params.message.context_id = original_task.context_id
    before = len(wire.submissions)
    original = wire.handler.guard.lifecycle_lease
    physical = HTTPException(403, detail="Actual post-commit original " + phase + " exit")
    exits = []
    armed = True

    @asynccontextmanager
    async def failed_original_exit():
        nonlocal armed
        async with original():
            yield
        exits.append(asyncio.current_task())
        if armed and ((phase == "admission" and len(exits) == 2) or (phase == "acquisition" and wire.manager._subscribers)):
            armed = False
            raise physical

    monkeypatch.setattr(wire.handler, "guard", replace(wire.handler.guard, lifecycle_lease=failed_original_exit))
    async with observed_native_http(wire) as observed:
        with pytest.raises(A2AClientError, match="500"):
            async for _ in wire.client(streaming=streaming, polling=False, http_client=observed.http).send_message(params):
                pytest.fail("Actual failed lease exit delivered a normal reply")
    assert not armed and observed.statuses == [500]
    assert len(observed.failures) == 1 and is_execution_control_error(observed.failures[0])
    assert execution_commit_outcome(observed.failures[0]) == "committed"
    assert any(error is physical for error in execution_evidence_carriers(observed.failures[0]))
    assert len(wire.submissions) == before + 1
    rows = await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks")
    assert len(rows) == 1 and rows[0][0] == wire.submissions[-1].task.id
    assert not wire.manager._subscribers and not wire.admission.lock.locked()


@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("mutate_result", [False, True])
async def test_continuation_notification_follows_actual_outer_commit(native_request_admission, monkeypatch, rollback, mutate_result):
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    from kestrel_sovereign.storage.db.interface import TransactionError

    a = native_request_admission
    notifications = []
    monkeypatch.setattr(a.manager, "_on_task_submitted", notifications.append)
    task = await a.manager.create_task(send(a), creator_agent_id=PRINCIPAL)
    assert len(notifications) == 1
    params = send(a)
    params.message.task_id = task.id
    params.message.context_id = task.context_id
    before_revision = await a.backend.fetch_one("SELECT input_revision FROM a2a_tasks WHERE id = ?", (task.id,))

    class RollBackOriginal(RuntimeError):
        pass

    original_error = RollBackOriginal("Actual outer test rollback")
    rolled_back = False
    try:
        async with a.backend.transaction():
            result = await a.manager.continue_task(params, creator_agent_id=PRINCIPAL)
            assert result.id == task.id
            assert len(notifications) == 1, "An uncommitted input woke the recipient"
            if mutate_result:
                result.id = "caller-return-value"
                result.context_id = "caller-return-context"
                result.history[-1].parts[0].text = "Caller changed its return projection"
                result.metadata["caller_projection"] = True
            if rollback:
                raise original_error
    except (RollBackOriginal, TransactionError) as error:
        assert rollback and any(node is original_error for node in execution_evidence_carriers(error))
        rolled_back = True
    assert rolled_back is rollback
    assert len(notifications) == (1 if rollback else 2)
    revision = await a.backend.fetch_one("SELECT input_revision FROM a2a_tasks WHERE id = ?", (task.id,))
    assert revision[0] == before_revision[0] + (0 if rollback else 1)
    canonical = await a.manager.task_store.get_task_snapshot(task.id, recipient_agent_id=a.scope.recipient_id)
    assert notifications[-1].task.SerializeToString(deterministic=True) == canonical.task.SerializeToString(deterministic=True), (
        "The committed wake retained a caller-mutable return projection instead of its admitted snapshot"
    )


@pytest.mark.parametrize("source", ["resolver-sync", "resolver-async", "replay", "authorization"])
@pytest.mark.parametrize("error_kind", ["proof", "http", "opaque"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_original_proof_io_exit_is_not_an_authentication_refusal(wire, monkeypatch, source, error_kind, streaming):
    from kestrel_sovereign.execution_custody import execution_evidence_carriers, is_execution_control_error

    authority = wire.handler.guard.authority_factory()
    physical, originals, owners, descriptors = [], [], [], []

    @contextmanager
    def original_io():
        descriptor = os.open(os.devnull, os.O_WRONLY)
        descriptors.append(descriptor)
        owner = asyncio.current_task()
        try:
            yield
        finally:
            owners.append((owner, asyncio.current_task()))
            try:
                os.read(descriptor, 1)
            except OSError as error:
                physical.append(error)
                original = (
                    NativeSenderProofError("Actual original proof resource closer failed") if error_kind == "proof"
                    else HTTPException(401, detail="Actual original authentication resource closer failed") if error_kind == "http"
                    else ValueError("Actual opaque authentication resource closer failed")
                )
                originals.append(original)
                raise original from error
            finally:
                os.close(descriptor)

    def resolve_sync(did):
        with original_io():
            return authority.resolver(did)

    async def resolve_async(did):
        await asyncio.sleep(0)
        return resolve_sync(did)

    store = wire.handler.guard.replay_store

    class OriginalReplay:
        async def reserve(self, *args, **kwargs):
            with original_io():
                return await store.reserve(*args, **kwargs)

    async def authorize(did):
        with original_io():
            return await authority.authorize_sender(did)

    if source == "replay":
        guard = replace(wire.handler.guard, replay_store=OriginalReplay())
    elif source == "authorization":
        guard = replace(wire.handler.guard, authority_factory=lambda: replace(authority, authorize_sender=authorize))
    else:
        guard = replace(wire.handler.guard, authority_factory=lambda: replace(
            authority, resolver=resolve_sync if source == "resolver-sync" else resolve_async))
    monkeypatch.setattr(wire.handler, "guard", guard)
    async with observed_native_http(wire) as observed:
        with pytest.raises(A2AClientError, match="500"):
            async for _ in wire.client(streaming=streaming, polling=False, http_client=observed.http).send_message(send(wire.admission)):
                pytest.fail("A failed original proof resource delivered a business reply")
    assert len(physical) == len(originals) == len(owners) == len(descriptors) == 1
    assert owners[0][0] is owners[0][1]
    assert observed.statuses == [500] and len(observed.failures) == 1
    assert is_execution_control_error(observed.failures[0])
    evidence = tuple(execution_evidence_carriers(observed.failures[0]))
    assert any(error is originals[0] for error in evidence)
    assert any(error is physical[0] for error in evidence)
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    assert not wire.submissions and not wire.manager._subscribers and not wire.admission.lock.locked()
    assert await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []


async def test_input_admission_refuses_unobservable_commit_before_write(native_request_admission, monkeypatch):
    from kestrel_sovereign.execution_custody import execution_evidence_carriers

    a = native_request_admission
    notifications = []
    monkeypatch.setattr(a.manager, "_on_task_submitted", notifications.append)
    monkeypatch.setattr(type(a.backend), "call_after_commit", None)
    with pytest.raises(Exception) as caught:
        await a.manager.create_task(send(a), creator_agent_id=PRINCIPAL)
    assert any(
        type(error) is RuntimeError and str(error) == "Native task admission requires actual backend commit observation"
        for error in execution_evidence_carriers(caught.value)
    )
    assert not notifications
    assert await a.backend.fetch_all("SELECT id FROM a2a_tasks") == []
    assert await a.backend.fetch_all("SELECT task_id FROM a2a_task_inputs") == []


async def test_fenced_admission_never_treats_joined_transaction_as_commit(native_request_admission):
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError, execution_commit_outcome

    a = native_request_admission

    @asynccontextmanager
    async def original_admission():
        yield

    async with a.backend.transaction():
        with pytest.raises(ExecutionCommitOutcomeError) as pending:
            async with a.manager._task_input_admission(None, original_admission):
                await a.backend.execute(
                    "INSERT INTO native_test_principals VALUES (?, ?)", ("did:staged-input", PRINCIPAL)
                )
        assert execution_commit_outcome(pending.value) == "unknown"
    assert await a.backend.fetch_one(
        "SELECT principal FROM native_test_principals WHERE signing_did = ?", ("did:staged-input",)
    ) == (PRINCIPAL,)


@pytest.mark.parametrize("streaming", [False, True])
async def test_raw_http_invalid_tenant_is_protocol_data_not_source_failure(
    wire, streaming
):
    from google.protobuf.json_format import MessageToDict

    params = send(wire.admission)
    params.tenant = "foreign"
    body = MessageToDict(params)
    if wire.protocol == "JSONRPC":
        path = wire.binding.prefix + "/rpc"
        body = {
            "jsonrpc": "2.0",
            "id": "tenant-refusal",
            "method": "SendStreamingMessage" if streaming else "SendMessage",
            "params": body,
        }
    else:
        path = (
            wire.binding.prefix
            + "/tenant-alpha/"
            + ("message:stream" if streaming else "message:send")
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(
            Starlette(routes=[wire.binding.route]), raise_app_exceptions=False
        ),
        base_url="http://native-core.test",
        headers=dict(wire.http.headers),
    ) as actual:
        response = await actual.post(path, json=body)
    assert response.status_code == (200 if wire.protocol == "JSONRPC" else 400)
    assert "tenant" in response.text.lower()
    assert not wire.submissions
    assert await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []
    assert not wire.admission.lock.locked()


@pytest.mark.parametrize("operation", ["send", "subscribe"])
@pytest.mark.parametrize("refusal", ["missing-proof", "policy-denial"])
@pytest.mark.parametrize("fail_original_exit", [False, True])
async def test_stream_admission_http_refusal_retires_before_status_and_challenge(
    wire, operation, refusal, fail_original_exit, monkeypatch
):
    from contextlib import asynccontextmanager
    from google.protobuf.json_format import MessageToDict
    from kestrel_sovereign.execution_custody import (
        execution_evidence_carriers,
        is_execution_control_error,
    )

    statuses = []
    failures = []
    physical = HTTPException(401, detail="actual original lease exit failed")
    if fail_original_exit:
        original_lease = wire.handler.guard.lifecycle_lease

        @asynccontextmanager
        async def failed_original_exit():
            async with original_lease():
                yield
            raise physical

        monkeypatch.setattr(
            wire.handler,
            "guard",
            replace(wire.handler.guard, lifecycle_lease=failed_original_exit),
        )
    original_app = Starlette(routes=[wire.binding.route])

    async def checked_app(scope, receive, send_response):
        async def checked_send(message):
            if message["type"] == "http.response.start":
                assert not wire.admission.lock.locked()
                statuses.append(message["status"])
            await send_response(message)

        try:
            await original_app(scope, receive, checked_send)
        except BaseException as error:
            failures.append(error)
            raise

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(checked_app, raise_app_exceptions=False),
        base_url="http://native-core.test",
        headers=dict(wire.http.headers),
    ) as actual:
        if refusal == "missing-proof":
            params = (
                send(wire.admission)
                if operation == "send"
                else SubscribeToTaskRequest(
                    id="unknown", tenant=wire.admission.scope.tenant
                )
            )
            if wire.protocol == "JSONRPC":
                path = wire.binding.prefix + "/rpc"
                body = {
                    "jsonrpc": "2.0",
                    "id": "refusal",
                    "method": "SendStreamingMessage"
                    if operation == "send"
                    else "SubscribeToTask",
                    "params": MessageToDict(params),
                }
            else:
                path = (
                    wire.binding.prefix
                    + "/tenant-alpha/"
                    + (
                        "message:stream"
                        if operation == "send"
                        else "tasks/unknown:subscribe"
                    )
                )
                body = MessageToDict(params) if operation == "send" else {}
            response = await actual.post(path, json=body)
            assert response.status_code == (500 if fail_original_exit else 401)
            if not fail_original_exit:
                assert response.headers["www-authenticate"].startswith(
                    "Kestrel-A2A-Sender-Proof"
                )
        else:
            await wire.admission.backend.execute("DELETE FROM native_test_principals")
            client = wire.client(streaming=True, http_client=actual)
            stream = (
                client.send_message(send(wire.admission))
                if operation == "send"
                else client.subscribe(
                    SubscribeToTaskRequest(
                        id="unknown", tenant=wire.admission.scope.tenant
                    )
                )
            )
            with pytest.raises(
                A2AClientError, match="500" if fail_original_exit else "403"
            ):
                async for _ in stream:
                    pytest.fail("A denied sender must not receive a native event")
    assert statuses == [
        500 if fail_original_exit else 401 if refusal == "missing-proof" else 403
    ]
    if fail_original_exit:
        assert len(failures) == 1 and is_execution_control_error(failures[0])
        assert any(
            error is physical for error in execution_evidence_carriers(failures[0])
        )
    else:
        assert not failures
    assert not wire.submissions
    assert await wire.admission.backend.fetch_all("SELECT id FROM a2a_tasks") == []
    assert not wire.manager._subscribers
    assert not wire.admission.lock.locked()
