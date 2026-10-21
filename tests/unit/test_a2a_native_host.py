"""Native manager construction and the real host's signal boot phase."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
import pytest_asyncio
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, SendMessageRequest
from a2a.utils.errors import InternalError

from kestrel_sovereign.a2a.did_registry import HostA2ADidResolver
from kestrel_sovereign.a2a.inbound_authorization import RecipientA2ASenderAuthorizer
from kestrel_sovereign.a2a.native_host import create_hosted_native_guard
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.task_manager import TaskManager, create_task_manager
from kestrel_sovereign.storage.db.interface import QueryError
from tests.utils.native_a2a import signing_identity
from tests.utils.native_a2a_security import (
    EXTENSIONS,
    NOW,
    PRINCIPAL,
    SENDER,
    send,
    signed_context,
)


@pytest_asyncio.fixture
async def publication(hosted, hosted_recipient_signing):
    from a2a.types import AgentSkill
    from fastapi import FastAPI

    from kestrel_sovereign.a2a.native_publication import (
        prepare_hosted_native_publication,
    )
    from kestrel_sovereign.identity.runtime_identity import AgentIdentity
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    recipient = hosted.recipient
    pair, document = hosted_recipient_signing
    recipient.identity = AgentIdentity(
        hybrid_keypair=pair,
        new_did=document["id"],
        new_verification_methods=document["verificationMethod"],
    )
    recipient.storage = None
    recipient.get_public_profile = KestrelAgent.get_public_profile.__get__(recipient)
    recipient.features = {}
    recipient._agent_name = "Live Name"
    recipient.resolve_effective_name = KestrelAgent.resolve_effective_name.__get__(
        recipient
    )
    hosted.manager.register_provider(
        "test",
        [AgentSkill(id="ask", name="Ask", description="Ask", tags=["chat"])],
        object(),
    )
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    native = await prepare_hosted_native_publication(
        app,
        hosted.host,
        "registered-agent",
        recipient,
        {
            "KESTREL_BASE_URL": "https://native.test",
            "KESTREL_A2A_DISCOVERY_AGENT": "registered-agent",
        },
    )
    recipient._native_a2a_publication = native
    yield SimpleNamespace(**vars(hosted), app=app, publication=native)
    async with hosted.host.a2a_lifecycle_lease():
        native.withdraw()


async def test_manager_commits_native_routes_and_card_as_one_publication(publication):
    import httpx
    from a2a.types import AgentCard
    from google.protobuf.json_format import ParseDict
    from starlette.exceptions import HTTPException

    from kestrel_sovereign.a2a.native_card import verify_native_agent_card
    from kestrel_sovereign.a2a.native_routes import native_card_path

    p = publication
    assert p.app.routes == []
    async with p.host.a2a_lifecycle_lease():
        p.host._register_agent("registered-agent", p.recipient)
    assert len(p.app.routes) == 3
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(p.app), base_url="https://native.test"
    ) as http:
        direct = await http.get(native_card_path("registered-agent"))
        root = await http.get("/.well-known/agent-card.json")
        assert direct.status_code == root.status_code == 200
        assert direct.json() == root.json()
        verify_native_agent_card(
            ParseDict(root.json(), AgentCard()),
            signing_did=p.recipient.identity.signing_did,
            trusted_document={
                "id": p.recipient.identity.signing_did,
                "verificationMethod": p.recipient.identity.new_verification_methods,
            },
        )
        etag = direct.headers["etag"]
        assert (
            await http.get(
                native_card_path("registered-agent"), headers={"If-None-Match": etag}
            )
        ).status_code == 304
        p.recipient._agent_name = "Session Rename"
        renamed = await http.get(
            native_card_path("registered-agent"), headers={"If-None-Match": etag}
        )
        assert renamed.status_code == 200 and renamed.json()["name"] == "Session Rename"
        assert renamed.headers["etag"] != etag
        assert (
            renamed.json()["supportedInterfaces"]
            == direct.json()["supportedInterfaces"]
        )
        async with p.host.a2a_lifecycle_lease():
            p.host._withdraw_initialized_agent("registered-agent", p.recipient)
        assert p.app.routes == []
        assert (await http.get(native_card_path("registered-agent"))).status_code == 404
        with pytest.raises(HTTPException) as denied:
            await p.publication.get_card()
        assert denied.value.status_code == 404


@pytest.mark.parametrize("fault", ["policy", "identity", "registration"])
async def test_public_card_revalidates_the_exact_recipient_generation(
    publication, fault
):
    from starlette.exceptions import HTTPException

    p = publication
    async with p.host.a2a_lifecycle_lease():
        p.host._register_agent("registered-agent", p.recipient)
        if fault == "policy":
            p.host.install_a2a_hosted_policy(
                p.recipient,
                resolver=p.policy.resolver,
                authorizer=p.policy.authorizer,
                router=p.policy.router,
                requester=p.policy.requester,
            )
        elif fault == "identity":
            p.recipient.identity = replace(
                p.recipient.identity, new_verification_methods=[]
            )
        else:
            p.host._agents["registered-agent"] = SimpleNamespace(**vars(p.recipient))
    with pytest.raises(HTTPException) as denied:
        await p.publication.get_card()
    assert denied.value.status_code == 403


async def test_native_publication_failure_withdraws_manager_registration(publication):
    from starlette.responses import Response
    from starlette.routing import Route

    p = publication
    conflict = Route("/.well-known/agent-card.json", lambda request: Response())
    p.app.routes.append(conflict)
    async with p.host.a2a_lifecycle_lease():
        with pytest.raises(RuntimeError, match="already owned"):
            p.host._register_agent("registered-agent", p.recipient)
    assert p.host.get_agent("registered-agent") is None
    assert p.host.a2a_hosted_policy_for(p.recipient) is None
    assert p.app.routes == [conflict]


async def test_public_card_holds_generation_through_async_profile_read(publication):
    p = publication
    async with p.host.a2a_lifecycle_lease():
        p.host._register_agent("registered-agent", p.recipient)
    reading, resume, attempted, changed = (asyncio.Event() for _ in range(4))

    class Storage:
        async def get_node(self, agent_id):
            assert agent_id == p.recipient.agent_id
            reading.set()
            await resume.wait()
            # This read also uses the actual native SQLite/PostgreSQL connection.
            await p.backend.fetch_one("SELECT principal FROM native_test_principals")
            return SimpleNamespace(properties={"description": "Public description"})

    async def replace_policy():
        attempted.set()
        async with p.host.a2a_lifecycle_lease():
            changed.set()
            p.host.install_a2a_hosted_policy(
                p.recipient,
                resolver=p.policy.resolver,
                authorizer=p.policy.authorizer,
                router=p.policy.router,
                requester=p.policy.requester,
            )

    p.recipient.storage = Storage()
    card_read = asyncio.create_task(p.publication.get_card())
    writer = None
    try:
        await asyncio.wait_for(reading.wait(), timeout=2)
        writer = asyncio.create_task(replace_policy())
        await asyncio.wait_for(attempted.wait(), timeout=2)
        assert not changed.is_set(), (
            "Card profile I/O must retain its recipient generation"
        )
        resume.set()
        card = await asyncio.wait_for(card_read, timeout=2)
        assert card.description == "Public description"
        await asyncio.wait_for(writer, timeout=2)
    finally:
        resume.set()
        await asyncio.gather(
            card_read, *([writer] if writer is not None else []), return_exceptions=True
        )


async def test_private_native_preparation_refuses_bad_identity_before_routes(
    publication,
):
    from kestrel_sovereign.a2a.native_publication import (
        HostedNativeA2APublication,
        NativeHostingConfiguration,
    )

    p = publication
    p.recipient.identity = None
    with pytest.raises(ValueError, match="hybrid recipient identity"):
        await HostedNativeA2APublication.prepare(
            p.host,
            p.app.state.native_a2a_http,
            p.recipient,
            "registered-agent",
            NativeHostingConfiguration("https://native.test"),
        )
    assert p.app.routes == []


async def test_native_preparation_refuses_registry_from_another_app(publication):
    from starlette.applications import Starlette

    from kestrel_sovereign.a2a.native_publication import (
        prepare_hosted_native_publication,
    )

    other_app = Starlette()
    other_app.state.native_a2a_http = publication.app.state.native_a2a_http
    with pytest.raises(ValueError, match="different application"):
        await prepare_hosted_native_publication(
            other_app,
            publication.host,
            "registered-agent",
            publication.recipient,
            {"KESTREL_BASE_URL": "https://native.test"},
        )
    assert other_app.routes == publication.app.routes == []


async def test_policy_revocation_precedes_route_cleanup_failure(publication):
    from starlette.exceptions import HTTPException

    p = publication
    async with p.host.a2a_lifecycle_lease():
        p.host._register_agent("registered-agent", p.recipient)
    binding = p.publication._binding
    missing_route = binding.card_routes[0]
    p.app.routes.remove(missing_route)
    try:
        async with p.host.a2a_lifecycle_lease():
            with pytest.raises(RuntimeError, match="lost its published route"):
                p.host._revoke_a2a_hosted_policy(p.recipient)
        assert p.host.a2a_hosted_policy_for(p.recipient) is None
        # Even a retained SDK handler must fail its original policy guard.
        async with p.host.a2a_execution_lease():
            with pytest.raises(HTTPException) as denied:
                p.publication._guard.authority_factory().require_current(None)
        assert denied.value.status_code == 403
    finally:
        p.app.routes.append(missing_route)


@pytest.mark.parametrize("protocol", ["JSONRPC", "HTTP+JSON"])
async def test_actual_host_onboarding_publication_discovery_and_sdk_request(
    publication, monkeypatch, protocol
):
    import time
    from uuid import uuid4

    import httpx
    from a2a.client import ClientConfig, ClientFactory
    from a2a.client.card_resolver import A2ACardResolver
    from a2a.types import GetTaskRequest
    from a2a.utils.errors import TaskNotFoundError
    from fastapi import FastAPI
    from starlette.middleware.sessions import SessionMiddleware

    from kestrel_sovereign import server
    from kestrel_sovereign.a2a.native_card import (
        REQUIRED_NATIVE_EXTENSIONS,
        verify_native_agent_card,
    )
    from kestrel_sovereign.a2a.native_client import NativeSenderInterceptor
    from kestrel_sovereign.a2a.native_routes import native_card_path

    p = publication
    monkeypatch.setenv("KESTREL_BASE_URL", "https://native.test")
    monkeypatch.setenv("KESTREL_A2A_DISCOVERY_AGENT", "registered-agent")
    operator_checks = []

    def operator_key():
        operator_checks.append(True)
        return "operator-key"

    monkeypatch.setattr(server, "get_api_key", operator_key)
    app = FastAPI()
    app.state.agent_manager = p.host
    app.middleware("http")(server.auth_middleware)
    app.add_middleware(SessionMiddleware, secret_key="native-publication-test")
    async with p.host.a2a_lifecycle_lease():
        p.host._agents.pop("registered-agent")
        p.host._agent_names.pop(p.recipient.agent_id)
        p.host._onboarding_agents["registered-agent"] = p.recipient
        rollback = await server._onboard_host_registered_agent(
            app, p.host, "registered-agent", p.recipient
        )
        native = p.recipient._native_a2a_publication
        assert not app.state.native_a2a_http._registrations
        assert p.host.get_agent("registered-agent") is None
        p.host._register_agent("registered-agent", p.recipient)
        p.host._onboarding_agents.pop("registered-agent")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app),
        base_url="https://native.test",
        headers={"A2A-Version": "1.0"},
    ) as http:

        def verifier(card):
            verify_native_agent_card(
                card,
                signing_did=p.recipient.identity.signing_did,
                trusted_document={
                    "id": p.recipient.identity.signing_did,
                    "verificationMethod": p.recipient.identity.new_verification_methods,
                },
            )

        card = await A2ACardResolver(http, "https://native.test").get_agent_card(
            signature_verifier=verifier
        )
        assert await native.get_card() == card
        interceptor = NativeSenderInterceptor(
            keypair=p.signing[0],
            sender=SENDER,
            recipient=p.scope,
            extensions=REQUIRED_NATIVE_EXTENSIONS,
            policy=p.guard.proof_policy,
            classical_kid="key-1",
            pq_kid="key-2",
            clock=time.time,
            nonce=lambda: uuid4().hex,
        )
        sdk = ClientFactory(
            ClientConfig(
                httpx_client=http,
                streaming=False,
                supported_protocol_bindings=[protocol],
            )
        ).create(card, interceptors=[interceptor])
        with pytest.raises(TaskNotFoundError):
            await sdk.get_task(GetTaskRequest(id="missing", tenant=p.scope.tenant))
        assert not operator_checks, (
            "Published discovery and SDK routes must not enter operator auth"
        )
        # Only the installed card route's actual read methods are public.
        assert (
            await http.post(native_card_path("registered-agent"))
        ).status_code == 401
        assert operator_checks
        async with p.host.a2a_lifecycle_lease():
            p.host._withdraw_initialized_agent("registered-agent", p.recipient)
            await rollback()
        assert not app.state.native_a2a_http._registrations


@pytest.fixture(scope="module")
def hosted_recipient_signing():
    return signing_identity("did:web:recipient.test:agent")


@pytest_asyncio.fixture
async def hosted(native_request_admission, hosted_recipient_signing, tmp_path):
    from kestrel_sovereign.features.peers.directory import PeerRequester
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    admission = native_request_admission
    host = AgentManager(
        base_data_dir=tmp_path, startup_runtime_env={}, startup_roster_enabled=False
    )

    def agent(did, document):
        return SimpleNamespace(
            did=did,
            agent_id=did,
            identity=SimpleNamespace(
                is_hybrid=True,
                signing_did=document["id"],
                new_verification_methods=document["verificationMethod"],
            ),
        )

    recipient = agent(admission.scope.recipient_id, hosted_recipient_signing[1])
    recipient.task_manager = admission.manager
    sender = agent(PRINCIPAL, admission.signing[1])
    host._agents.update({"registered-agent": recipient, "sender": sender})
    host._agent_names.update({recipient.did: "registered-agent", sender.did: "sender"})
    requester = PeerRequester(recipient.did, object())

    class DirectoryPolicy:
        async def authorize_inbound_sender(self, selected_requester, sender_id):
            assert host.a2a_lifecycle_lease().locked()
            assert selected_requester is requester
            row = await admission.backend.fetch_one(
                "SELECT principal FROM native_test_principals WHERE principal = ?",
                (sender_id,),
            )
            return row is not None

    router = DirectoryPolicy()
    recipient.peer_directory_router = router
    recipient.peer_requester = requester
    resolver = HostA2ADidResolver(host, recipient=recipient)
    authorizer = RecipientA2ASenderAuthorizer(host, recipient=recipient)
    async with host.a2a_lifecycle_lease():
        policy = host.install_a2a_hosted_policy(
            recipient,
            resolver=resolver.resolve,
            authorizer=authorizer,
            router=router,
            requester=requester,
        )
        guard = create_hosted_native_guard(
            host,
            routing_key="registered-agent",
            recipient=recipient,
            required_extensions=EXTENSIONS,
            proof_policy=admission.guard.proof_policy,
            clock=lambda: NOW,
        )
    yield SimpleNamespace(
        **(
            vars(admission)
            | {
                "host": host,
                "recipient": recipient,
                "sender": sender,
                "guard": guard,
                "policy": policy,
                "actor": agent,
            }
        )
    )


@pytest.mark.parametrize("protocol", ["JSONRPC", "HTTP+JSON"])
async def test_hosted_native_guard_uses_actual_directory_and_stable_principal(
    hosted, protocol
):
    params = send(hosted)
    async with hosted.guard.admit(
        params, "SendMessage", signed_context(hosted, params, protocol=protocol)
    ) as admitted:
        assert admitted.principal_id == PRINCIPAL
        task = await hosted.manager.create_task(
            admitted.request, creator_agent_id=admitted.principal_id
        )
    snapshot = await hosted.manager.get_task_snapshot(
        task.id, recipient_agent_id=hosted.scope.recipient_id
    )
    assert snapshot.ownership.creator_agent_id == PRINCIPAL
    assert snapshot.ownership.tenant == hosted.scope.tenant


@pytest.mark.parametrize(
    "fault", ["policy", "recipient", "recipient-identity", "task-manager", "task-store"]
)
async def test_retired_host_generation_never_admits_or_revalidates_a_request(
    hosted, fault
):
    from starlette.exceptions import HTTPException

    params = send(hosted)
    async with hosted.guard.admit(
        params, "SendMessage", signed_context(hosted, params)
    ) as admitted:
        pass
    async with hosted.host.a2a_lifecycle_lease():
        if fault == "policy":
            hosted.host.install_a2a_hosted_policy(
                hosted.recipient,
                resolver=hosted.policy.resolver,
                authorizer=hosted.policy.authorizer,
                router=hosted.policy.router,
                requester=hosted.policy.requester,
            )
        elif fault == "recipient":
            hosted.host._agents["registered-agent"] = SimpleNamespace(
                **vars(hosted.recipient)
            )
        elif fault == "recipient-identity":
            hosted.recipient.identity = SimpleNamespace(
                **vars(hosted.recipient.identity)
            )
        elif fault == "task-manager":
            hosted.recipient.task_manager = None
        else:
            from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore

            hosted.manager.task_store = TaskStore(
                hosted.backend, scope=hosted.scope, owns_backend=False
            )
    with pytest.raises(HTTPException) as denied:
        async with hosted.guard.hold_authority(admitted):
            pytest.fail("An old request retained a retired host generation")
    assert denied.value.status_code == 403
    with pytest.raises(HTTPException) as denied:
        async with hosted.guard.admit(
            params, "SendMessage", signed_context(hosted, params)
        ):
            pytest.fail("A retired host generation admitted a new request")
    assert denied.value.status_code == 403
    assert await hosted.backend.fetch_all("SELECT id FROM a2a_tasks") == []


@pytest.mark.parametrize("phase", ["resolve", "authorize", "wait"])
async def test_replacing_sender_with_same_keys_does_not_retain_authority(hosted, phase):
    from starlette.exceptions import HTTPException

    def replace_sender():
        hosted.host._agents["sender"] = SimpleNamespace(**vars(hosted.sender))

    async def authorize(did):
        principal = await hosted.policy.authorizer.authorize_principal_with_policy(
            did, router=hosted.policy.router, requester=hosted.policy.requester
        )
        replace_sender()
        return principal

    if phase != "wait":
        factory = hosted.guard.authority_factory

        def authority():
            original = factory()
            if phase == "resolve":

                async def witnessed_resolve(did):
                    document = await original.resolver(did)
                    replace_sender()
                    return document

                return replace(original, resolver=witnessed_resolve)
            return replace(original, authorize_sender=authorize)

        guard = replace(hosted.guard, authority_factory=authority)
    else:
        guard = hosted.guard
    params = send(hosted)
    if phase == "wait":
        async with guard.admit(
            params, "SendMessage", signed_context(hosted, params)
        ) as admitted:
            pass
        replace_sender()
        operation = guard.hold_authority(admitted)
    else:
        operation = guard.admit(params, "SendMessage", signed_context(hosted, params))
    with pytest.raises(HTTPException) as denied:
        async with operation:
            pytest.fail("A replacement sender inherited an older request's authority")
    assert denied.value.status_code == 403


async def test_local_sender_requires_the_hosts_exact_verification_document(hosted):
    from starlette.exceptions import HTTPException

    substituted = signing_identity(SENDER)
    factory = hosted.guard.authority_factory

    def authority():
        original = factory()

        async def resolve(did):
            await original.resolver(did)
            return substituted[1]

        return replace(original, resolver=resolve)

    guard = replace(hosted.guard, authority_factory=authority)
    adversarial = SimpleNamespace(
        **(vars(hosted) | {"guard": guard, "signing": substituted})
    )
    params = send(adversarial)
    with pytest.raises(HTTPException) as denied:
        async with guard.admit(
            params, "SendMessage", signed_context(adversarial, params)
        ):
            pytest.fail(
                "A substituted DID document replaced the host's sender identity"
            )
    assert denied.value.status_code == 403
    assert await hosted.backend.fetch_all("SELECT id FROM a2a_tasks") == []


async def test_host_directory_storage_failure_is_not_a_policy_denial(hosted):
    await hosted.backend.execute("DROP TABLE native_test_principals")
    params = send(hosted)
    with pytest.raises(InternalError, match="persistence is unavailable") as failed:
        async with hosted.guard.admit(
            params, "SendMessage", signed_context(hosted, params)
        ):
            pytest.fail("A broken directory admitted a request")
    assert isinstance(failed.value.__cause__, QueryError)


@pytest.mark.parametrize(
    "fault,error",
    [
        ("routing-key", ValueError),
        ("policy", ValueError),
        ("hybrid", ValueError),
        ("ambiguous", ValueError),
        ("task-manager", TypeError),
    ],
)
async def test_host_guard_cannot_publish_without_concrete_identity_and_policy(
    hosted, fault, error
):
    routing_key = "registered-agent"
    async with hosted.host.a2a_lifecycle_lease():
        if fault == "routing-key":
            routing_key = "REGISTERED-AGENT"
        elif fault == "policy":
            hosted.host._a2a_hosted_policies.clear()
        elif fault == "hybrid":
            hosted.recipient.identity.is_hybrid = False
        elif fault == "ambiguous":
            hosted.host._agents["duplicate"] = hosted.actor(
                "did:kestrel:duplicate",
                {
                    "id": hosted.recipient.identity.signing_did,
                    "verificationMethod": hosted.recipient.identity.new_verification_methods,
                },
            )
        else:
            hosted.recipient.task_manager = None
        with pytest.raises(error, match="Native hosting requires"):
            create_hosted_native_guard(
                hosted.host,
                routing_key=routing_key,
                recipient=hosted.recipient,
                required_extensions=EXTENSIONS,
                proof_policy=hosted.guard.proof_policy,
                clock=lambda: NOW,
            )


async def test_overlapping_sender_verifications_keep_separate_identity_witnesses(
    hosted,
):
    second_did = "did:web:second.test:agent"
    second_principal = "did:kestrel:second-sender"
    signing = signing_identity(second_did)
    entered = asyncio.Event()
    release = asyncio.Event()

    class CoordinatedResolver:
        async def resolve(self, did):
            document = hosted.policy.resolver(did)
            if did == SENDER:
                entered.set()
                await release.wait()
            return document

    await hosted.backend.execute(
        "INSERT INTO native_test_principals VALUES (?, ?)",
        (second_did, second_principal),
    )
    async with hosted.host.a2a_lifecycle_lease():
        hosted.host._agents["second"] = hosted.actor(second_principal, signing[1])
        hosted.host._agent_names[second_principal] = "second"
        hosted.host.install_a2a_hosted_policy(
            hosted.recipient,
            resolver=CoordinatedResolver().resolve,
            authorizer=hosted.policy.authorizer,
            router=hosted.policy.router,
            requester=hosted.policy.requester,
        )
        guard = create_hosted_native_guard(
            hosted.host,
            routing_key="registered-agent",
            recipient=hosted.recipient,
            required_extensions=EXTENSIONS,
            proof_policy=hosted.guard.proof_policy,
            clock=lambda: NOW,
        )
    first = SimpleNamespace(**(vars(hosted) | {"guard": guard}))
    second = SimpleNamespace(**(vars(first) | {"signing": signing}))

    async def admit(admission, sender):
        params = send(admission)
        context = signed_context(admission, params, proof_changes={"sender": sender})
        async with guard.admit(params, "SendMessage", context) as granted:
            return granted

    pending = asyncio.create_task(admit(first, SENDER))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        other = await asyncio.wait_for(admit(second, second_did), 5)
        release.set()
        original = await asyncio.wait_for(pending, 5)
        assert original.principal_id == PRINCIPAL
        assert other.principal_id == second_principal
        assert original.authority is not other.authority
        async with guard.hold_authority(original):
            pass
        async with guard.hold_authority(other):
            pass
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("owns_backend", [False, True])
async def test_native_factory_initializes_one_backend_and_honors_its_owner(
    native_a2a_backend_factory,
    owns_backend,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    submitted = []
    scope = NativeRecipientScope("did:kestrel:recipient", "tenant")
    manager = TaskManager.from_backend(
        backend,
        scope=scope,
        owns_backend=owns_backend,
        on_task_submitted=submitted.append,
    )
    try:
        stores = [
            manager.task_store,
            manager.context_store,
            manager.session_service,
            manager.observability_store,
            manager.memory_service,
            manager.feedback_store,
        ]
        assert all(store.backend is backend for store in stores)
        await manager.initialize()
        task = await manager.create_task(
            SendMessageRequest(
                tenant="tenant",
                message=new_text_message("native request", role=Role.ROLE_USER),
            ),
            creator_agent_id="did:kestrel:sender",
        )
        assert submitted[0].task == task
        assert submitted[0].ownership.recipient_agent_id == scope.recipient_id
        assert submitted[0].ownership.tenant == scope.tenant
        await manager.close()
        assert backend.is_connected is (not owns_backend)
        if not owns_backend:
            assert await backend.fetch_one("SELECT 1") == (1,)
    finally:
        await manager.close()
        await backend.close()


async def test_owned_sqlite_factory_uses_the_same_native_admission_contract(tmp_path):
    scope = NativeRecipientScope("did:kestrel:standalone")
    manager = await create_task_manager(str(tmp_path / "standalone.db"), scope=scope)
    backend = manager.task_store.backend
    try:
        task = await manager.create_task(
            SendMessageRequest(
                message=new_text_message("native request", role=Role.ROLE_USER)
            ),
            creator_agent_id="did:kestrel:sender",
        )
        view = await manager.get_task_snapshot(
            task.id, recipient_agent_id=scope.recipient_id
        )
        assert view.ownership.creator_agent_id == "did:kestrel:sender"
    finally:
        await manager.close()
    assert not backend.is_connected


@pytest.mark.parametrize("broken_session_schema", [False, True])
async def test_actual_host_signal_boot_borrows_native_storage_and_rolls_back(
    native_a2a_backend_factory,
    tmp_path,
    broken_session_schema,
):
    # Imports happen after the suite has isolated every runtime/config path.
    from kestrel_sovereign.agent.boot import BootContext
    from kestrel_sovereign.features.base import Feature, tool
    from kestrel_sovereign.features.privacy import PrivacyAgent
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage import AsyncStorage, PrivacyEnforcingStorage

    backend = native_a2a_backend_factory()
    storage = AsyncStorage(backend=backend, agent_id="did:kestrel:host")
    await storage.initialize()
    llm = LLMService(usage_db=storage.db, agent_data_dir=tmp_path)
    agent = KestrelAgent(
        "did:kestrel:host",
        storage_path=str(tmp_path / "host.db"),
        llm_service=llm,
        db_backend=backend.backend_type,
        sync_enabled=False,
        allowed_features=set(),
    )
    agent._raw_storage = storage
    agent.storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    agent.privacy_agent = PrivacyAgent(storage, PrivacyMode.NORMAL)
    ctx = BootContext(custody=agent._custody)
    try:
        if broken_session_schema:
            # A real SQL failure after task/context initialization proves the
            # rollback owner was registered before initialize's first await.
            await backend.execute_script(
                "CREATE TABLE a2a_sessions (id TEXT PRIMARY KEY)"
            )
            with pytest.raises(QueryError, match="agent_name"):
                await agent._boot_phase_a2a_observability_signals(ctx)
            assert "task_manager" in ctx.rollback_labels
            assert agent.task_manager is not None
        else:
            await agent._boot_phase_a2a_observability_signals(ctx)
            manager = agent.task_manager
            assert manager.task_store.scope == NativeRecipientScope(agent.did)
            assert (
                manager.context_store.backend is manager.task_store.backend is backend
            )
            assert manager.session_service.backend is backend
            assert manager.feedback_store is agent.feedback_store
            assert manager.observability_store is agent.observability_store
            assert agent.dispatcher is not None
            assert agent.signal_registry.get("a2a.task_submitted") is not None
            assert agent.signal_registry.get("a2a.task_complete") is not None

            class NativeHostFeature(Feature):
                @property
                def tool_description(self):
                    return "Native host skill contribution"

                async def initialize(self):
                    pass

                @tool(
                    name="native_host_echo",
                    description="Echo JSON unchanged",
                    command_prefix="!native_host_echo",
                )
                async def echo(self, value: dict) -> dict:
                    return value

            feature = NativeHostFeature(agent)
            await agent._register_feature(feature)
            assert [skill.id for skill in manager.get_agent_skills()] == [
                "native_host_echo"
            ]
            assert manager.get_agent_for_skill("native_host_echo") == feature.name
            assert manager.get_agent_for_command("!native_host_echo") == (
                feature.name,
                "native_host_echo",
            )
            await agent._unregister_feature_runtime(feature)
            assert manager.get_agent_skills() == []
            assert manager.get_agent_for_command("!native_host_echo") is None
        await ctx.run_rollback()
        assert not ctx.rollback_errors
        assert agent.task_manager is None
        assert backend.is_connected
        assert await backend.fetch_one("SELECT 1") == (1,)
    finally:
        await ctx.run_rollback()
        await llm.close()
        await storage.close()


@pytest_asyncio.fixture(params=["", "/proxy"])
async def local_native_peer(publication, request):
    from kestrel_sovereign.a2a.did_registry import HostA2ADidResolver
    from kestrel_sovereign.a2a.inbound_authorization import RecipientA2ASenderAuthorizer
    from kestrel_sovereign.features.peers.directory import PeerIdentity, PeerRequester
    from kestrel_sovereign.identity.runtime_identity import AgentIdentity
    from tests.utils.native_a2a import manager_for

    p = publication
    if request.param:
        from kestrel_sovereign.a2a.native_publication import NativeHostingConfiguration

        p.publication.configuration = NativeHostingConfiguration(
            "https://native.test" + request.param
        )
    p.sender.identity = AgentIdentity(
        hybrid_keypair=p.signing[0],
        new_did=SENDER,
        new_verification_methods=p.signing[1]["verificationMethod"],
    )
    sender_manager = manager_for(
        p.backend, NativeRecipientScope(PRINCIPAL, "sender-tenant"), owns_backend=False
    )
    await sender_manager.initialize()
    p.sender.task_manager = sender_manager
    requester = PeerRequester(PRINCIPAL, object())
    peer = PeerIdentity(p.recipient.did, "peer", "registered-agent", name="Peer")
    state = SimpleNamespace(allowed=True)

    class SenderDirectory:
        async def resolve_peer_by_agent_id(self, selected, agent_id):
            assert selected is requester
            return peer if state.allowed and agent_id == peer.agent_id else None

        async def resolve_peer(self, selected, name):
            assert selected is requester
            return peer if state.allowed and name == "Peer" else None

        async def authorize_inbound_sender(self, selected, sender_id):
            return selected is requester and state.allowed

        def open_a2a_session(self, selected, selected_peer):
            import httpx

            from kestrel_sovereign.a2a.native_publication import (
                open_hosted_native_peer_session,
            )

            return open_hosted_native_peer_session(
                p.app,
                p.host,
                sender=p.sender,
                requester=selected,
                peer=selected_peer,
                timeout=httpx.Timeout(10),
            )

    directory = SenderDirectory()
    p.sender.peer_directory_router = directory
    p.sender.peer_requester = requester
    resolver = HostA2ADidResolver(p.host, recipient=p.sender)
    authorizer = RecipientA2ASenderAuthorizer(p.host, recipient=p.sender)
    async with p.host.a2a_lifecycle_lease():
        p.host.install_a2a_hosted_policy(
            p.sender,
            resolver=resolver.resolve,
            authorizer=authorizer,
            router=directory,
            requester=requester,
        )
        p.host._register_agent("registered-agent", p.recipient)
    try:
        yield SimpleNamespace(
            **vars(p),
            sender_manager=sender_manager,
            requester=requester,
            peer=peer,
            directory=directory,
            route_state=state,
        )
    finally:
        await sender_manager.close()


async def test_same_host_session_uses_real_sdk_binding_and_creator_ledger(
    local_native_peer,
):
    from a2a.types import Message, Part, Role, SendMessageRequest

    p = local_native_peer
    async with p.directory.open_a2a_session(p.requester, p.peer) as session:
        request = SendMessageRequest(
            tenant=session.recipient_scope.tenant,
            message=Message(
                message_id="hosted-native-message",
                role=Role.ROLE_USER,
                parts=[Part(text="Native same-host delivery")],
            ),
            configuration={"return_immediately": True},
        )
        dispatch, wire = await session.reserve(
            request, dispatch_tool="send_a2a_message", verb="message"
        )
        receipt = await session.send(dispatch, wire)
        assert receipt.dispatch.remote_task_id == receipt.response.id
        assert dispatch.dispatch_id != receipt.response.id
        snapshot = await p.manager.get_task_snapshot(
            receipt.response.id, recipient_agent_id=p.recipient.did
        )
        assert snapshot.ownership.creator_agent_id == PRINCIPAL
        assert snapshot.ownership.tenant == p.scope.tenant
        assert (
            await p.sender_manager.outbound_store.get(dispatch.dispatch_id)
        ) == receipt.dispatch
        observed = await session.get_task(receipt.dispatch)
        assert observed.response == receipt.response
        assert observed.dispatch.revision > receipt.dispatch.revision


async def test_native_peer_message_tool_reaches_official_host_binding(
    local_native_peer,
):
    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    result = await feature.send_a2a_message("Peer", "Message from real peer tool")
    assert result.data["sent"] is True
    assert result.data["dispatch_id"] != result.data["remote_task_id"]
    assert result.data["response_kind"] == "task"
    task = await p.manager.get_task_for_creator(
        result.data["remote_task_id"], PRINCIPAL, recipient_agent_id=p.recipient.did
    )
    assert task.history[0].parts[0].text == "Message from real peer tool"
    assert task.context_id == result.data["context_id"]


async def test_same_host_session_rechecks_sender_directory_before_submission(
    local_native_peer,
):
    from a2a.types import Message, Part, Role, SendMessageRequest

    from kestrel_sovereign.features.peers.directory import PeerAccessDeniedError

    p = local_native_peer
    async with p.directory.open_a2a_session(p.requester, p.peer) as session:
        dispatch, wire = await session.reserve(
            SendMessageRequest(
                tenant=p.scope.tenant,
                message=Message(
                    message_id="revoked-hosted-message",
                    role=Role.ROLE_USER,
                    parts=[Part(text="Refuse before submission")],
                ),
                configuration={"return_immediately": True},
            ),
            dispatch_tool="send_a2a_message",
            verb="message",
        )
        p.route_state.allowed = False
        with pytest.raises(PeerAccessDeniedError, match="sender's directory"):
            await session.send(dispatch, wire)
        assert (
            await p.sender_manager.outbound_store.get(dispatch.dispatch_id)
        ).state.value == "reserved"
        from a2a.types import ListTasksRequest

        listed = await p.manager.task_store.list_for_creator(
            ListTasksRequest(tenant=p.scope.tenant),
            creator_agent_id=PRINCIPAL,
            recipient_agent_id=p.recipient.did,
        )
        assert list(listed.tasks) == []


async def test_native_message_tool_reports_lost_ack_as_unknown(local_native_peer):
    import httpx
    from a2a.types import ListTasksRequest

    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer

    class LostAcknowledgement:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            await self.app(scope, receive, send)
            raise httpx.ReadError("Lost acknowledgement after committed native task")

    p.app.add_middleware(LostAcknowledgement)
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    result = await feature.send_a2a_message(
        "Peer", "Recipient committed, acknowledgement lost"
    )
    assert result.data["sent"] is None
    assert result.data["delivery_state"] == "unknown"
    current = await p.sender_manager.outbound_store.get(result.data["dispatch_id"])
    assert current.state.value == "unknown" and current.remote_task_id is None
    listed = await p.manager.task_store.list_for_creator(
        ListTasksRequest(tenant=p.scope.tenant),
        creator_agent_id=PRINCIPAL,
        recipient_agent_id=p.recipient.did,
    )
    assert len(listed.tasks) == 1


async def test_actual_server_refresh_wires_native_message_tool_and_local_directory(
    local_native_peer,
):
    import httpx

    from kestrel_sovereign import server
    from kestrel_sovereign.a2a.did_registry import HostA2ADidResolver
    from kestrel_sovereign.a2a.inbound_authorization import RecipientA2ASenderAuthorizer
    from kestrel_sovereign.endpoints import models
    from kestrel_sovereign.features.peers.feature import PeersFeature
    from kestrel_sovereign.identity.runtime_identity import AgentIdentity
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    p = local_native_peer
    assert isinstance(p.sender.identity, AgentIdentity)
    p.sender.storage = None
    p.sender._agent_name = "Sender"
    p.sender.get_public_profile = KestrelAgent.get_public_profile.__get__(p.sender)
    p.sender.resolve_effective_name = KestrelAgent.resolve_effective_name.__get__(
        p.sender
    )
    feature = PeersFeature(p.sender)
    feature._own_name = "Sender"
    feature._host_url = "http://local-host.test"
    p.sender.features = {"PeersFeature": feature}
    p.app.state.agent_manager = p.host
    p.app.state.multi_agent_config = SimpleNamespace(
        host=SimpleNamespace(bind="local-host.test", port=8888)
    )
    p.app.routes.extend(models.router.routes)
    router, requester = server._hosted_peer_directory_context(
        p.app, p.sender, manager=p.host
    )
    # Both directory HTTP and peer HTTP explicitly use the actual owned app.
    router._client_factory = lambda **kwargs: httpx.AsyncClient(
        transport=httpx.ASGITransport(p.app), **kwargs
    )
    resolver = HostA2ADidResolver(p.host, recipient=p.sender)
    authorizer = RecipientA2ASenderAuthorizer(p.host, recipient=p.sender)
    async with p.host.a2a_lifecycle_lease():
        p.host.install_a2a_hosted_policy(
            p.sender,
            resolver=resolver.resolve,
            authorizer=authorizer,
            router=router,
            requester=requester,
        )
    result = await feature.send_a2a_message(
        "Live Name", "Through actual server and local directory"
    )
    assert result.data["sent"] is True
    task = await p.manager.get_task_for_creator(
        result.data["remote_task_id"], PRINCIPAL, recipient_agent_id=p.recipient.did
    )
    assert task.history[0].parts[0].text == "Through actual server and local directory"


async def test_native_tool_does_not_report_acknowledged_race_as_not_sent(
    local_native_peer,
):
    from a2a.types import ListTasksRequest

    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer

    class ReconciliationWins:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            await self.app(scope, receive, send)
            # Another sender-owned observer reconciles after recipient commit,
            # before the original sender receives its acknowledgement.
            row = await p.backend.fetch_one(
                "SELECT dispatch_id FROM a2a_outbound_dispatches WHERE sender_id = ? AND sender_tenant = ?",
                (PRINCIPAL, "sender-tenant"),
            )
            store = p.sender_manager.outbound_store
            submitting = await store.get(row[0])
            unknown = await store.mark_unknown(
                submitting, operation_id="reader-observed-uncertainty"
            )
            listed = await p.manager.task_store.list_for_creator(
                ListTasksRequest(tenant=p.scope.tenant),
                creator_agent_id=PRINCIPAL,
                recipient_agent_id=p.recipient.did,
            )
            assert await store.reconcile(
                unknown, list(listed.tasks), operation_id="read-witness"
            )

    p.app.add_middleware(ReconciliationWins)
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    result = await feature.send_a2a_message(
        "Peer", "Acknowledged by the independent reader"
    )
    assert result.data["sent"] is True
    assert result.data["delivery_state"] == "acknowledged"
    assert result.data["remote_task_id"]


@pytest.mark.parametrize(
    "fault",
    [
        "sender-policy",
        "recipient-policy",
        "sender-identity",
        "recipient-identity",
        "sender-store",
    ],
)
async def test_same_host_session_cannot_spend_a_replaced_generation(
    local_native_peer, fault
):
    from a2a.types import Message, Part, Role, SendMessageRequest

    from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore
    from kestrel_sovereign.features.peers.directory import PeerAccessDeniedError
    from kestrel_sovereign.identity.runtime_identity import AgentIdentity

    p = local_native_peer
    async with p.directory.open_a2a_session(p.requester, p.peer) as session:
        dispatch, wire = await session.reserve(
            SendMessageRequest(
                tenant=p.scope.tenant,
                message=Message(
                    message_id="generation-message",
                    role=Role.ROLE_USER,
                    parts=[Part(text="Do not submit after replacement")],
                ),
                configuration={"return_immediately": True},
            ),
            dispatch_tool="send_a2a_message",
            verb="message",
        )
        async with p.host.a2a_lifecycle_lease():
            selected = p.sender if fault.startswith("sender") else p.recipient
            if fault.endswith("policy"):
                old = p.host.a2a_hosted_policy_for(selected)
                p.host.install_a2a_hosted_policy(
                    selected,
                    resolver=old.resolver,
                    authorizer=old.authorizer,
                    router=old.router,
                    requester=old.requester,
                )
            elif fault.endswith("identity"):
                old = selected.identity
                selected.identity = AgentIdentity(
                    hybrid_keypair=old.hybrid_keypair,
                    new_did=old.signing_did,
                    new_verification_methods=old.new_verification_methods,
                )
            else:
                p.sender_manager.task_store = TaskStore(
                    p.backend,
                    scope=NativeRecipientScope(PRINCIPAL, "other-tenant"),
                    owns_backend=False,
                )
        with pytest.raises(
            PeerAccessDeniedError, match="registration or identity changed"
        ):
            await session.send(dispatch, wire)
        assert (
            await p.sender_manager.outbound_store.get(dispatch.dispatch_id)
        ).state.value == "reserved"


async def test_native_message_tool_audit_lists_immutable_dispatch_and_remote_id(
    local_native_peer,
):
    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    sent = await feature.send_a2a_message("Peer", "Native audit visibility")
    listed = await feature.list_outbound_a2a_tasks(recipient="Peer")
    assert listed.data["count"] == 1
    row = listed.data["rows"][0]
    assert row["dispatch_id"] == sent.data["dispatch_id"]
    assert row["remote_task_id"] == sent.data["remote_task_id"]
    assert row["sender_tenant"] == "sender-tenant" and row["tenant"] == p.scope.tenant
    assert row["state"] == "acknowledged" and row["verb"] == "message"
    assert "Native audit visibility" not in str(row)


async def test_native_message_audit_refuses_another_local_tenant_store(
    local_native_peer,
):
    from kestrel_sovereign.a2a.stores.unified.outbound_store import OutboundStore
    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer
    p.sender_manager.outbound_store = OutboundStore(
        p.backend, sender_scope=NativeRecipientScope(PRINCIPAL, "other-tenant")
    )
    feature = PeersFeature(p.sender)
    result = await feature.list_outbound_a2a_tasks()
    assert result.data == {"rows": [], "count": 0}
    assert "different local actor, tenant or primary backend" in result.error


async def test_host_session_refuses_wrong_sender_tenant_before_any_sdk_request(
    local_native_peer,
):
    from kestrel_sovereign.a2a.stores.unified.outbound_store import OutboundStore
    from kestrel_sovereign.features.peers.directory import PeerAccessDeniedError

    p = local_native_peer
    p.sender_manager.outbound_store = OutboundStore(
        p.backend, sender_scope=NativeRecipientScope(PRINCIPAL, "other-tenant")
    )
    with pytest.raises(
        PeerAccessDeniedError, match="different local actor, tenant or primary backend"
    ):
        async with p.directory.open_a2a_session(p.requester, p.peer):
            pytest.fail("A foreign sender ledger cannot open a native session")


async def test_tracked_task_tool_uses_native_input_parts_and_advisory_hint(
    local_native_peer,
):
    from google.protobuf.json_format import MessageToDict

    from kestrel_sovereign.a2a.extensions import SKILL_HINT_URI
    from kestrel_sovereign.features.peers.feature import PeersFeature
    from kestrel_sovereign.features.tasks.feature import TaskFeature

    p = local_native_peer
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    result = await feature.send_a2a_task(
        "Peer",
        "Read the plan and memory reference",
        skill_id="ask",
        parts=[
            {"text": "Full plan", "filename": "plan.txt"},
            {"data": {"ref_type": "memory", "id": "memory-42", "flags": [False, None]}},
        ],
    )
    assert result.data["sent"] is True and result.data["state"] == "submitted"
    assert result.data["dispatch_id"] != result.data["remote_task_id"]
    task = await p.manager.get_task_for_creator(
        result.data["remote_task_id"], PRINCIPAL, recipient_agent_id=p.recipient.did
    )
    assert not task.artifacts, "Input handoffs are Message Parts; Artifacts are outputs"
    assert [part.WhichOneof("content") for part in task.history[0].parts] == [
        "text",
        "text",
        "data",
    ]
    assert MessageToDict(task.metadata)[SKILL_HINT_URI] == {
        "version": 1,
        "skill_id": "ask",
    }
    inbox = TaskFeature(p.recipient)
    inbox.set_task_manager(p.recipient.task_manager)
    inspected = await inbox._get_task_status_data(task.id)
    assert inspected["ok"] and inspected["skill_hint"] == "ask"
    assert inspected["request"]["parts"][1] == {
        "text": "Full plan",
        "filename": "plan.txt",
    }
    assert inspected["request"]["parts"][2]["data"] == {
        "ref_type": "memory",
        "id": "memory-42",
        "flags": [False, None],
    }
    assert "metadata" not in inspected, (
        "Internal execution metadata is not inbox presentation"
    )


@pytest.mark.parametrize(
    "parts",
    [
        "legacy-json-string",
        [{"type": "text", "text": "draft"}],
        [False],
        [{"data": {}, "text": "two contents"}],
    ],
)
async def test_tracked_task_tool_refuses_draft_or_malformed_parts_before_reservation(
    local_native_peer, parts
):
    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    result = await feature.send_a2a_task(
        "Peer", "Bad parts must not dispatch", parts=parts
    )
    assert result.status.value == "error"
    assert not await p.sender_manager.outbound_store.list_recent()


async def test_tracked_task_unknown_advisory_skill_cannot_select_execution(
    local_native_peer,
):
    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    result = await feature.send_a2a_task(
        "Peer", "Unknown published skill", skill_id="operator.shell"
    )
    assert result.data["sent"] is False
    assert result.data["delivery_state"] == "not_submitted"
    assert not await p.sender_manager.outbound_store.list_recent()


async def test_native_result_and_public_cancel_use_local_dispatch_handle(
    local_native_peer,
):
    from kestrel_sovereign.features.peers.feature import PeersFeature
    from kestrel_sovereign.features.tasks.feature import TaskFeature

    p = local_native_peer
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    p.sender.features = {"peers": feature}
    task_feature = TaskFeature(p.sender)
    task_feature.set_task_manager(p.sender_manager)
    sent = await feature.send_a2a_task("Peer", "Tracked native assignment")
    dispatch_id = sent.data["dispatch_id"]
    remote_id = sent.data["remote_task_id"]
    read = await feature.get_peer_task_result(dispatch_id)
    assert read.data["task"]["id"] == remote_id
    assert read.data["task"]["status"]["state"] == "TASK_STATE_SUBMITTED"
    # A recipient's remote ID never grants the sender a route to cancel it.
    refused = await task_feature.cancel_task(remote_id)
    assert refused.status.value == "error"
    canceled = await task_feature.cancel_task(
        dispatch_id, reason="Assignment withdrawn"
    )
    assert canceled.data["cancellation_confirmed"] is True
    assert canceled.data["remote_task_id"] == remote_id
    read = await feature.get_peer_task_result(dispatch_id)
    assert read.data["state"] == "canceled"
    assert read.data["task"]["status"]["state"] == "TASK_STATE_CANCELED"
    row = await p.sender_manager.outbound_store.get(dispatch_id)
    assert row.remote_task_id == remote_id and row.cancel_state.value == "complete"
    # Repeated cancellation is a read of the retained terminal task, never a replay.
    confirmed = await task_feature.cancel_task(dispatch_id)
    assert confirmed.data["cancellation_confirmed"] is True


async def test_native_retained_tool_cannot_retarget_a_revoked_peer(local_native_peer):
    from a2a.types import TaskState

    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    sent = await feature.send_a2a_task("Peer", "Retained authorization")
    row = await p.sender_manager.outbound_store.get(sent.data["dispatch_id"])
    p.route_state.allowed = False
    read = await feature.get_peer_task_result(row.dispatch_id)
    cancel = await feature.cancel_outbound_task(row.dispatch_id)
    assert read.status.value == cancel.status.value == "error"
    assert await p.sender_manager.outbound_store.get(row.dispatch_id) == row
    task = await p.manager.get_task_for_creator(
        row.remote_task_id, PRINCIPAL, recipient_agent_id=p.recipient.did
    )
    assert task.status.state == TaskState.TASK_STATE_SUBMITTED


async def test_synchronous_ask_uses_native_blocking_send_and_recipient_response(
    local_native_peer,
    monkeypatch,
):
    from kestrel_sovereign.a2a.request_handler import NativeA2ARequestHandler
    from kestrel_sovereign.features.peers.feature import PeersFeature
    from kestrel_sovereign.features.tasks.feature import TaskFeature

    p = local_native_peer
    native_flags = []
    native_send = NativeA2ARequestHandler.on_message_send

    async def observe_native_send(handler, params, context):
        native_flags.append(params.configuration.return_immediately)
        return await native_send(handler, params, context)

    monkeypatch.setattr(NativeA2ARequestHandler, "on_message_send", observe_native_send)
    submitted = asyncio.Event()
    received = []
    original = p.manager._on_task_submitted

    def announce(snapshot):
        if original is not None:
            original(snapshot)
        received.append(snapshot)
        submitted.set()

    p.manager._on_task_submitted = announce
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    inbox = TaskFeature(p.recipient)
    inbox.set_task_manager(p.manager)
    asking = asyncio.create_task(
        feature.ask_agent("Peer", "Synchronous native question")
    )
    try:
        await asyncio.wait_for(submitted.wait(), 5)
        assert native_flags == [False], "The SDK must send a blocking native request"
        assert not asking.done(), (
            "Native blocking send waits for a terminal/interrupted task"
        )
        task = received[0].task
        answered = await inbox.respond_to_a2a_task(task.id, "Native peer answer")
        assert answered.data["state"] == "completed"
        result = await asyncio.wait_for(asking, 5)
        assert result.data["response"] == "Native peer answer"
        assert result.data["state"] == "completed"
        assert result.data["native_response"]["id"] == task.id
        retained = await p.sender_manager.outbound_store.get(result.data["dispatch_id"])
        assert retained.remote_task_id == task.id and retained.state.value == "complete"
    finally:
        if not asking.done():
            asking.cancel()
        await asyncio.gather(asking, return_exceptions=True)
