"""Actual SDK clients and HTTP bindings over native proof, policy and storage."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from a2a.client import ClientConfig, ClientFactory
from a2a.client.client import ClientCallContext
from a2a.client.errors import A2AClientError
from a2a.client.interceptors import BeforeArgs
from a2a.helpers.proto_helpers import new_text_artifact
from a2a.types import (
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
from a2a.utils.errors import (
    InternalError,
    PushNotificationNotSupportedError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from kestrel_sovereign.a2a.native_client import NativeSenderInterceptor
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
async def wire(native_request_admission, request):
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


async def test_foreign_context_is_denied_without_a_task_or_submission_wake(wire):
    other = await wire.manager.create_task(
        send(wire.admission), creator_agent_id="did:kestrel:other-principal"
    )
    wire.submissions.clear()
    params = send(wire.admission)
    params.message.context_id = other.context_id
    with pytest.raises(A2AClientError, match="403"):
        async for _ in wire.client().send_message(params):
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


async def test_persistence_failure_is_a_standard_internal_error_without_sql_details(
    wire,
):
    await wire.admission.backend.execute("DROP TABLE native_test_principals")
    with pytest.raises(InternalError, match="A2A persistence is unavailable") as failed:
        await wire.client().get_task(
            GetTaskRequest(id="unknown", tenant=wire.admission.scope.tenant)
        )
    assert "native_test_principals" not in str(failed.value)
    assert "SELECT" not in str(failed.value)


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
    async with guard.admit(
        params, "SendMessage", signed_context(wire.admission, params)
    ) as admitted:
        task = await wire.manager.create_task(params, creator_agent_id=PRINCIPAL)
        stream, initial = await wire.handler._open_stream(task.id, PRINCIPAL)
    wire.admission.state.current = False
    events = wire.handler._events(stream, initial, admitted)
    try:
        with pytest.raises(HTTPException) as denied:
            await anext(events)
        assert denied.value.status_code == 403
        assert not wire.manager._subscribers
    finally:
        await events.aclose()


async def test_terminal_cancel_and_subscribe_return_standard_lifecycle_errors(wire):
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
    from a2a.types import Part
    from a2a.utils.errors import ContentTypeNotSupportedError, InvalidParamsError

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
