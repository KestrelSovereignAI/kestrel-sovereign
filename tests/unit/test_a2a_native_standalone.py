"""Single runtime authority drives the same official SDK bindings and primary DB."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from a2a.client import ClientConfig, ClientFactory
from a2a.types import AgentSkill, GetTaskRequest
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from kestrel_sovereign.a2a.inbound_authorization import RecipientA2ASenderAuthorizer
from kestrel_sovereign.a2a.native_card import verify_native_agent_card
from kestrel_sovereign.a2a.native_client import NativeSenderInterceptor
from kestrel_sovereign.a2a.native_publication import (
    NativeA2APublication,
    NativeHostingConfiguration,
)
from kestrel_sovereign.a2a.native_routes import NativeA2AHTTPRegistry
from kestrel_sovereign.a2a.native_standalone import SingleAgentNativeAuthority
from kestrel_sovereign.features.peers.directory import PeerRequester
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from tests.utils.native_a2a import signing_identity
from tests.utils.native_a2a_security import (
    EXTENSIONS,
    NOW,
    PRINCIPAL,
    SENDER,
    send,
    signed_context,
)


@pytest.fixture(scope="module")
def single_identity():
    pair, document = signing_identity("did:web:single.native.test:agent")
    return AgentIdentity(
        hybrid_keypair=pair,
        new_did=document["id"],
        new_verification_methods=document["verificationMethod"],
    ), document


@pytest_asyncio.fixture
async def single(native_request_admission, single_identity):
    a = native_request_admission
    ready = asyncio.Event()
    ready.set()
    recipient = SimpleNamespace(
        did=a.scope.recipient_id,
        agent_id=a.scope.recipient_id,
        task_manager=a.manager,
        identity=replace(
            single_identity[0],
            new_verification_methods=deepcopy(
                single_identity[0].new_verification_methods
            ),
        ),
        _runtime_publication_ready=ready,
        _execution_custody=None,
    )

    class DirectoryPolicy:
        async def authorize_inbound_sender(self, requester, sender_id):
            assert requester is recipient.peer_requester
            row = await a.backend.fetch_one(
                "SELECT principal FROM native_test_principals WHERE principal = ?",
                (sender_id,),
            )
            return row is not None

    recipient.peer_directory_router = DirectoryPolicy()
    recipient.peer_requester = PeerRequester(a.scope.recipient_id, object())
    recipient.a2a_did_resolver = lambda did: a.signing[1] if did == SENDER else None
    recipient.a2a_inbound_sender_authorizer = RecipientA2ASenderAuthorizer(
        None,
        recipient=recipient,
        sender_id_resolver=lambda did: PRINCIPAL if did == SENDER else None,
    )

    async def profile():
        return {"name": "Single Native", "description": "One explicit runtime"}

    recipient.get_public_profile = profile
    recipient.resolve_effective_name = lambda *, default: default
    a.manager.register_provider(
        "single-test",
        [
            AgentSkill(
                id="message", name="Message", description="Receive work", tags=["test"]
            )
        ],
        object(),
    )
    owner = SingleAgentNativeAuthority(
        recipient, "stable-single", proof_policy=a.guard.proof_policy, clock=lambda: NOW
    )
    app = Starlette()
    registry = NativeA2AHTTPRegistry(app)
    publication = await NativeA2APublication.prepare(
        registry,
        recipient,
        "stable-single",
        NativeHostingConfiguration("http://single.native.test", "stable-single"),
        guard_factory=owner.create_guard,
    )
    await owner.publish(publication)
    try:
        yield SimpleNamespace(
            admission=a,
            recipient=recipient,
            owner=owner,
            publication=publication,
            app=app,
            document=single_identity[1],
        )
    finally:
        await owner.close()


@pytest.mark.parametrize("protocol", ["JSONRPC", "HTTP+JSON"])
async def test_one_runtime_serves_signed_card_and_sdk_tasks_without_an_agent_manager(
    single, protocol
):
    p = single
    card = await p.publication.get_card()
    verify_native_agent_card(
        card, signing_did=p.document["id"], trusted_document=p.document
    )
    interceptor = NativeSenderInterceptor(
        keypair=p.admission.signing[0],
        sender=SENDER,
        recipient=p.admission.scope,
        extensions=EXTENSIONS,
        policy=p.owner.proof_policy,
        classical_kid="key-1",
        pq_kid="key-2",
        clock=lambda: NOW,
        nonce=lambda: uuid4().hex,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(p.app)) as http:
        client = ClientFactory(
            ClientConfig(
                httpx_client=http,
                streaming=False,
                polling=True,
                supported_protocol_bindings=[protocol],
            )
        ).create(card, interceptors=[interceptor])
        events = [event async for event in client.send_message(send(p.admission))]
        task = events[0].task
        fetched = await client.get_task(
            GetTaskRequest(id=task.id, tenant=p.admission.scope.tenant)
        )
        assert fetched.id == task.id
    snapshot = await p.admission.manager.get_task_snapshot(
        task.id, recipient_agent_id=p.admission.scope.recipient_id
    )
    assert snapshot.ownership.creator_agent_id == PRINCIPAL
    assert p.publication.is_published and len(p.app.routes) == 3
    await p.owner.close()
    assert not p.publication.is_published and p.app.routes == []
    with pytest.raises(HTTPException):
        await p.publication.get_card()


@pytest.mark.parametrize(
    "fault",
    [
        "task-manager",
        "identity",
        "keys",
        "ready",
        "ready-object",
        "runtime",
        "resolver",
        "policy",
        "router",
        "requester",
        "authority",
        "publication",
    ],
)
async def test_retained_guard_refuses_replaced_runtime_authority(single, fault):
    p = single
    guard = p.publication._guard
    agent = p.recipient
    if fault == "task-manager":
        agent.task_manager = object()
    elif fault == "identity":
        agent.identity = replace(agent.identity)
    elif fault == "keys":
        agent.identity.new_verification_methods[0]["id"] += "-changed"
    elif fault == "ready":
        agent._runtime_publication_ready.clear()
    elif fault == "ready-object":
        agent._runtime_publication_ready = asyncio.Event()
        agent._runtime_publication_ready.set()
    elif fault == "runtime":
        agent._execution_custody = object()
    elif fault == "resolver":
        agent.a2a_did_resolver = lambda did: p.admission.signing[1]
    elif fault == "policy":
        agent.a2a_inbound_sender_authorizer = object()
    elif fault == "router":
        agent.peer_directory_router = object()
    elif fault == "requester":
        agent.peer_requester = PeerRequester(agent.did, object())
    elif fault == "authority":
        agent._native_a2a_runtime_authority = object()
    else:
        agent._native_a2a_publication = object()
    params = GetTaskRequest(id="missing", tenant=p.admission.scope.tenant)
    selected = SimpleNamespace(**(vars(p.admission) | {"guard": guard}))
    context = signed_context(selected, params, operation="GetTask")
    with pytest.raises(HTTPException) as caught:
        async with guard.admit(params, "GetTask", context):
            pytest.fail("replaced runtime reached task access")
    assert caught.value.status_code == 403


async def test_close_joins_original_readers_and_closes_admission_before_withdrawal(
    single,
):
    p = single
    entered, release = asyncio.Event(), asyncio.Event()

    async def original_reader():
        async with p.publication._guard.lifecycle_lease():
            entered.set()
            await release.wait()

    reading = asyncio.create_task(original_reader())
    await entered.wait()
    closing = asyncio.create_task(p.owner.close())
    try:
        await asyncio.sleep(0)
        assert not closing.done() and p.publication.is_published
        with pytest.raises(HTTPException):
            p.owner.require_current()
        release.set()
        await asyncio.gather(reading, closing)
        assert not p.publication.is_published
    finally:
        release.set()
        await asyncio.gather(reading, closing, return_exceptions=True)


async def test_another_runtime_owner_cannot_replace_the_original_generation(single):
    p = single
    with pytest.raises(RuntimeError, match="fresh agent generation"):
        SingleAgentNativeAuthority(
            p.recipient,
            "replacement",
            proof_policy=p.owner.proof_policy,
            clock=lambda: NOW,
        )


async def test_agent_task_manager_close_joins_publication_before_any_primary_store_close(
    single,
    native_a2a_backend_factory,
):
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    p = single
    entered, release = asyncio.Event(), asyncio.Event()

    async def reader():
        async with p.publication._guard.lifecycle_lease():
            entered.set()
            await release.wait()

    independent = native_a2a_backend_factory()
    await independent.connect()
    reading = asyncio.create_task(reader())
    await entered.wait()
    closing = asyncio.create_task(
        KestrelAgent._close_native_task_manager(p.recipient, p.admission.manager)
    )
    try:
        await asyncio.sleep(0)
        assert not closing.done() and p.admission.backend.is_connected
        assert await independent.fetch_one("SELECT 1") == (1,)
        release.set()
        await asyncio.gather(reading, closing)
        assert not p.publication.is_published
        assert not p.admission.backend.is_connected
    finally:
        release.set()
        await asyncio.gather(reading, closing, return_exceptions=True)
        await independent.close()


async def test_unrecognized_publication_owner_cannot_discharge_storage(single):
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    p = single
    original = p.recipient._native_a2a_runtime_authority
    p.recipient._native_a2a_runtime_authority = object()
    try:
        with pytest.raises(TypeError, match="original authority owner"):
            await KestrelAgent._close_native_task_manager(
                p.recipient, p.admission.manager
            )
        assert p.admission.backend.is_connected
    finally:
        p.recipient._native_a2a_runtime_authority = original


async def test_cancelled_close_cannot_release_primary_storage_or_reopen_admission(
    single,
):
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    p = single
    entered, release = asyncio.Event(), asyncio.Event()

    async def reader():
        async with p.publication._guard.lifecycle_lease():
            entered.set()
            await release.wait()

    reading = asyncio.create_task(reader())
    await entered.wait()
    closing = asyncio.create_task(
        KestrelAgent._close_native_task_manager(p.recipient, p.admission.manager)
    )
    try:
        await asyncio.sleep(0)
        closing.cancel("original shutdown caller withdrew")
        with pytest.raises(asyncio.CancelledError) as caught:
            await closing
        assert caught.value.args == ("original shutdown caller withdrew",)
        assert p.admission.backend.is_connected and p.publication.is_published
        with pytest.raises(HTTPException):
            p.owner.require_current()
        release.set()
        await reading
        await KestrelAgent._close_native_task_manager(p.recipient, p.admission.manager)
        assert not p.admission.backend.is_connected and not p.publication.is_published
    finally:
        release.set()
        await asyncio.gather(reading, closing, return_exceptions=True)


@pytest.mark.parametrize("fault", ["policy", "clock"])
def test_invalid_proof_configuration_is_refused_before_binding_an_owner(fault):
    from kestrel_sovereign.a2a.native_sender import NativeProofPolicy

    recipient = SimpleNamespace()
    with pytest.raises(TypeError, match="proof policy and clock"):
        SingleAgentNativeAuthority(
            recipient,
            "single",
            proof_policy=None if fault == "policy" else NativeProofPolicy(),
            clock=None if fault == "clock" else lambda: NOW,
        )
    assert not hasattr(recipient, "_native_a2a_runtime_authority")


async def test_fresh_runtime_pointer_cannot_erase_original_native_revocation(single):
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError,
        ExecutionCustody,
        ProcessRuntimeExecutionFence,
    )

    p = single
    original = ExecutionCustody(
        ProcessRuntimeExecutionFence(agent_id=p.admission.scope.recipient_id)
    )
    clone = SimpleNamespace(
        **{
            key: value
            for key, value in vars(p.recipient).items()
            if not key.startswith("_native_a2a_")
        }
    )
    clone._execution_custody = original
    owner = SingleAgentNativeAuthority(
        clone, "original-runtime", proof_policy=p.owner.proof_policy, clock=lambda: NOW
    )
    original.revoke("original standalone runtime retired")
    clone._execution_custody = ExecutionCustody(
        ProcessRuntimeExecutionFence(agent_id=p.admission.scope.recipient_id)
    )
    with pytest.raises(
        ExecutionAuthorityError, match="original standalone runtime retired"
    ):
        owner.create_guard()
    await owner.close()


@pytest.mark.parametrize(
    "fault",
    [
        "manager",
        "recipient",
        "ready",
        "hybrid",
        "resolver",
        "authorizer",
        "router",
        "requester",
        "runtime",
    ],
)
async def test_invalid_runtime_contract_is_refused_before_owner_binding(single, fault):
    p = single
    clone = SimpleNamespace(
        **{
            key: value
            for key, value in vars(p.recipient).items()
            if not key.startswith("_native_a2a_")
        }
    )
    if fault == "manager":
        clone.task_manager = object()
    elif fault == "recipient":
        clone.did = "did:another-recipient"
    elif fault == "ready":
        clone._runtime_publication_ready = asyncio.Event()
    elif fault == "hybrid":
        clone.identity = SimpleNamespace(is_hybrid=False)
    elif fault == "resolver":
        clone.a2a_did_resolver = None
    elif fault == "authorizer":
        clone.a2a_inbound_sender_authorizer = None
    elif fault == "router":
        clone.peer_directory_router = None
    elif fault == "requester":
        clone.peer_requester = None
    else:
        clone._execution_custody = object()
    with pytest.raises(TypeError if fault in {"manager", "runtime"} else ValueError):
        SingleAgentNativeAuthority(
            clone, "refused", proof_policy=p.owner.proof_policy, clock=lambda: NOW
        )
    assert not hasattr(clone, "_native_a2a_runtime_authority")


async def test_foreign_typed_authority_cannot_discharge_another_runtime_storage(single):
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    p = single
    clone = SimpleNamespace(
        **{
            key: value
            for key, value in vars(p.recipient).items()
            if not key.startswith("_native_a2a_")
        }
    )
    foreign = SingleAgentNativeAuthority(
        clone, "another-runtime", proof_policy=p.owner.proof_policy, clock=lambda: NOW
    )
    p.recipient._native_a2a_runtime_authority = foreign
    try:
        with pytest.raises(ValueError, match="another runtime or task manager"):
            await KestrelAgent._close_native_task_manager(
                p.recipient, p.admission.manager
            )
        assert p.admission.backend.is_connected and p.publication.is_published
    finally:
        p.recipient._native_a2a_runtime_authority = p.owner
        await foreign.close()


async def test_foreign_task_manager_cannot_spend_the_publication_close_receipt(
    single, native_a2a_backend_factory
):
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from tests.utils.native_a2a import manager_for

    p = single
    backend = native_a2a_backend_factory()
    await backend.connect()
    foreign = manager_for(backend, p.admission.scope)
    await foreign.initialize()
    try:
        with pytest.raises(ValueError, match="another runtime or task manager"):
            await KestrelAgent._close_native_task_manager(p.recipient, foreign)
        assert backend.is_connected and p.admission.backend.is_connected
    finally:
        await foreign.close()


@pytest.mark.parametrize("fault", ["type", "routing", "replacement"])
async def test_runtime_publish_cannot_adopt_a_foreign_or_replacement_contract(
    single, fault
):
    p = single
    if fault == "type":
        with pytest.raises(TypeError, match="shared publication"):
            await p.owner.publish(object())
    else:
        candidate = await NativeA2APublication.prepare(
            p.publication.registry,
            p.recipient,
            "foreign-route" if fault == "routing" else p.owner.routing_key,
            p.publication.configuration,
            guard_factory=p.owner.create_guard,
        )
        with pytest.raises(ValueError if fault == "routing" else RuntimeError):
            await p.owner.publish(candidate)
    assert p.publication.is_published and len(p.app.routes) == 3
