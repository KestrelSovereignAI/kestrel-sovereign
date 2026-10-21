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
    from kestrel_sovereign.a2a.native_publication import NativeHostingConfiguration
    from tests.utils.native_a2a import serve_native_app

    async with serve_native_app(p.app, root_path=request.param) as public_base:
        p.publication.configuration = NativeHostingConfiguration(public_base)
        p.app.state.agent_manager = p.host
        p.sender.identity = AgentIdentity(
            hybrid_keypair=p.signing[0],
            new_did=SENDER,
            new_verification_methods=p.signing[1]["verificationMethod"],
        )
        sender_manager = manager_for(
            p.backend,
            NativeRecipientScope(PRINCIPAL, "sender-tenant"),
            owns_backend=False,
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
    from a2a.types import ListTasksRequest

    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer

    class LostAcknowledgement:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope.get("method") != "POST":
                return await self.app(scope, receive, send)

            async def withhold_response(_message):
                return None

            await self.app(scope, receive, withhold_response)
            raise RuntimeError("Lost acknowledgement after committed native task")

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
            if scope.get("method") != "POST":
                return await self.app(scope, receive, send)
            frames = []

            async def retain_response(frame):
                frames.append(frame)

            await self.app(scope, receive, retain_response)
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
            for frame in frames:
                await send(frame)

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


async def test_native_cancellation_reports_a_durably_acknowledged_race_as_confirmed(
    local_native_peer, monkeypatch
):
    from kestrel_sovereign.a2a.stores.unified.outbound_store import (
        OutboundDispatchConflict,
    )
    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    sent = await feature.send_a2a_task("Peer", "Retain the cancellation receipt")
    store = p.sender_manager.outbound_store
    original = store.acknowledge

    async def lose_ack_after_commit(dispatch, response, **kwargs):
        committed = await original(dispatch, response, **kwargs)
        assert committed is not None and committed.remote_task_state == "canceled"
        raise OutboundDispatchConflict(
            "A cancellation acknowledgement raced its durable commit"
        )

    monkeypatch.setattr(store, "acknowledge", lose_ack_after_commit)
    result = await feature.cancel_outbound_task(sent.data["dispatch_id"])
    assert result.data["cancellation_confirmed"] is True
    assert result.data["remote_task_id"] == sent.data["remote_task_id"]


async def test_native_question_requires_pre_dispatch_intent_and_correlates_recipient_answer(
    local_native_peer,
):
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from a2a.types import Message, Part, Role, SendMessageRequest
    from google.protobuf.json_format import MessageToDict

    from kestrel_sovereign.a2a.extensions import (
        QUESTION_CORRELATION_URI,
        question_correlation_metadata,
    )
    from kestrel_sovereign.a2a.stores.unified.question_store import QuestionConflict
    from kestrel_sovereign.features.tasks.feature import TaskFeature

    p = local_native_peer
    questions = p.sender_manager.questions_for_actor(PRINCIPAL)
    question_id = uuid4().hex
    async with p.directory.open_a2a_session(p.requester, p.peer) as session:
        params = SendMessageRequest(
            tenant=p.scope.tenant,
            metadata=question_correlation_metadata("question", question_id),
            message=Message(
                message_id=uuid4().hex,
                role=Role.ROLE_USER,
                parts=[Part(text="A native correlated question")],
            ),
            configuration={"return_immediately": True},
        )
        dispatch, wire = await session.reserve(
            params, dispatch_tool="send_a2a_question", verb="question"
        )
        with pytest.raises(
            QuestionConflict, match="current sender-owned peer dispatch"
        ):
            await session.send(dispatch, wire)
        assert (
            await p.sender_manager.outbound_store.get(dispatch.dispatch_id)
        ).state.value == "reserved"
        question = await questions.reserve(
            dispatch,
            wire,
            origin_turn_id="origin-turn",
            origin_session_id="local-chat-session",
            deadline=datetime.now(UTC) + timedelta(seconds=60),
        )
        sent = await session.send(dispatch, wire)
        inbox = TaskFeature(p.recipient)
        inbox.set_task_manager(p.manager)
        result = await inbox.respond_to_a2a_task(
            sent.response.id, "The correlated native answer"
        )
        assert result.data["state"] == "completed"
        observed = await session.get_task(sent.dispatch)
        assert MessageToDict(observed.response.status.message.metadata)[
            QUESTION_CORRELATION_URI
        ] == {"version": 1, "kind": "answer", "question_id": question_id}
        settled = await questions.get(question_id)
        assert settled.status == "RESOLVED"
        assert await questions.retained_response(observed.dispatch) == observed.response
        assert (
            await questions.record_answer(
                question_id, observed.dispatch, observed.response
            )
            == settled
        )
        wake = (await questions.pending_wakes())[0]
        import json

        payload = json.loads(wake.payload_json)
        assert payload["origin_session_id"] == "local-chat-session"
        assert payload["reply_text"] == "The correlated native answer"
        assert question.dispatch_id != sent.response.id


async def test_native_question_deadline_is_rechecked_at_actual_http_dispatch(
    local_native_peer,
):
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from a2a.types import ListTasksRequest, Message, Part, Role, SendMessageRequest

    from kestrel_sovereign.a2a.extensions import question_correlation_metadata
    from kestrel_sovereign.a2a.stores.unified.question_store import QuestionConflict

    p = local_native_peer
    questions = p.sender_manager.questions_for_actor(PRINCIPAL)
    deadline = datetime.now(UTC) + timedelta(seconds=60)
    async with p.directory.open_a2a_session(p.requester, p.peer) as session:
        params = SendMessageRequest(
            tenant=p.scope.tenant,
            metadata=question_correlation_metadata("question", uuid4().hex),
            message=Message(
                message_id=uuid4().hex,
                role=Role.ROLE_USER,
                parts=[Part(text="Expire before native admission")],
            ),
            configuration={"return_immediately": True},
        )
        dispatch, wire = await session.reserve(
            params, dispatch_tool="send_a2a_question", verb="question"
        )
        await questions.reserve(
            dispatch,
            wire,
            origin_turn_id="turn",
            origin_session_id="chat",
            deadline=deadline,
        )

        async def deadline_passes(_request):
            questions._clock = lambda: deadline + timedelta(seconds=1)

        session._http.event_hooks["request"].append(deadline_passes)
        with pytest.raises(QuestionConflict, match="no longer live"):
            await session.send(dispatch, wire)
        assert (
            await p.sender_manager.outbound_store.get(dispatch.dispatch_id)
        ).state.value == "not-sent"
        tasks = await p.manager.task_store.list_for_creator(
            ListTasksRequest(tenant=p.scope.tenant),
            creator_agent_id=PRINCIPAL,
            recipient_agent_id=p.recipient.did,
        )
        assert not tasks.tasks


async def test_unsolicited_answer_request_cannot_create_a_native_task_or_wake(
    local_native_peer,
):
    from uuid import uuid4

    from a2a.types import ListTasksRequest, Message, Part, Role, SendMessageRequest
    from a2a.utils.errors import InvalidParamsError

    from kestrel_sovereign.a2a.extensions import question_correlation_metadata

    p = local_native_peer
    async with p.directory.open_a2a_session(p.requester, p.peer) as session:
        params = SendMessageRequest(
            tenant=p.scope.tenant,
            metadata=question_correlation_metadata("answer", "somebody-elses-question"),
            message=Message(
                message_id=uuid4().hex,
                role=Role.ROLE_USER,
                parts=[Part(text="Forged standalone answer")],
            ),
            configuration={"return_immediately": True},
        )
        # An external standards client need not use Kestrel's sender ledger.
        # Exercise the server's own check through the actual official SDK client.
        with pytest.raises(InvalidParamsError, match="original task response"):
            async with session._network():
                async for _ in session._client.send_message(params):
                    pytest.fail("An unsolicited answer cannot be admitted")
    tasks = await p.manager.task_store.list_for_creator(
        ListTasksRequest(tenant=p.scope.tenant),
        creator_agent_id=PRINCIPAL,
        recipient_agent_id=p.recipient.did,
    )
    assert not tasks.tasks and not await p.sender_manager.question_store.pending_wakes()


async def test_committed_dispatch_label_cannot_bypass_actual_question_intent(
    local_native_peer,
):
    from uuid import uuid4

    from a2a.types import ListTasksRequest, Message, Part, Role, SendMessageRequest

    from kestrel_sovereign.a2a.extensions import question_correlation_metadata
    from kestrel_sovereign.a2a.native_client import NativePeerProtocolError

    p = local_native_peer
    async with p.directory.open_a2a_session(p.requester, p.peer) as session:
        params = SendMessageRequest(
            tenant=p.scope.tenant,
            metadata=question_correlation_metadata("question", uuid4().hex),
            message=Message(
                message_id=uuid4().hex,
                role=Role.ROLE_USER,
                parts=[Part(text="A label cannot bypass correlation")],
            ),
            configuration={"return_immediately": True},
        )
        interface = next(
            item
            for item in session.agent_card.supported_interfaces
            if item.protocol_binding == "JSONRPC"
        )
        dispatch = await p.sender_manager.outbound_store.reserve(
            uuid4().hex,
            params,
            recipient=session.recipient_scope,
            signing_did=p.recipient.identity.signing_did,
            interface=interface,
            dispatch_tool="send_a2a_message",
            verb="message",
        )
        with pytest.raises(NativePeerProtocolError, match="correlation contract"):
            await session.send(dispatch, params)
        assert (
            await p.sender_manager.outbound_store.get(dispatch.dispatch_id) == dispatch
        )
    tasks = await p.manager.task_store.list_for_creator(
        ListTasksRequest(tenant=p.scope.tenant),
        creator_agent_id=PRINCIPAL,
        recipient_agent_id=p.recipient.did,
    )
    assert not tasks.tasks


async def test_actual_host_middlewares_preserve_published_native_routes(
    local_native_peer, monkeypatch
):
    from kestrel_sovereign import server
    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer
    monkeypatch.setattr(server, "app", p.app)
    p.app.add_middleware(server.MultiAgentAgentRoutingMiddleware)
    p.app.middleware("http")(server.agent_routing_middleware)
    p.app.middleware("http")(server.auth_middleware)
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    result = await feature.send_a2a_message(
        "Peer", "Use the actual host middleware stack"
    )
    assert result.data["sent"] is True
    task = await p.manager.get_task_for_creator(
        result.data["remote_task_id"], PRINCIPAL, recipient_agent_id=p.recipient.did
    )
    assert task.history[0].parts[0].text == "Use the actual host middleware stack"


async def test_network_discovery_and_recipient_admission_do_not_inherit_sender_context(
    local_native_peer,
):
    import contextvars

    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer
    marker = contextvars.ContextVar("native_sender_turn_marker", default=None)
    observed = []

    class PublicProfileStorage:
        async def get_node(self, _agent_id):
            observed.append(("public-profile", marker.get()))
            return SimpleNamespace(properties={"description": "Public native profile"})

    p.recipient.storage = PublicProfileStorage()
    p.manager._on_task_submitted = lambda _snapshot: observed.append(
        ("recipient-admission", marker.get())
    )
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    token = marker.set("sender's original turn context")
    try:
        result = await feature.send_a2a_message(
            "Peer", "Recipient work owns its own request context"
        )
        assert marker.get() == "sender's original turn context"
    finally:
        marker.reset(token)
    assert result.data["sent"] is True
    assert {kind for kind, _ in observed} == {"public-profile", "recipient-admission"}
    assert all(value is None for _, value in observed)


async def test_blocking_peer_timeout_retires_http_subscription_without_replaying_task(
    local_native_peer,
):
    import httpx
    from a2a.types import ListTasksRequest

    from kestrel_sovereign.a2a.native_publication import open_hosted_native_peer_session
    from kestrel_sovereign.features.peers.feature import PeersFeature

    p = local_native_peer
    p.directory.open_a2a_session = lambda selected, peer: (
        open_hosted_native_peer_session(
            p.app,
            p.host,
            sender=p.sender,
            requester=selected,
            peer=peer,
            timeout=httpx.Timeout(0.1),
        )
    )
    feature = PeersFeature(p.sender)
    feature._peer_router = p.directory
    feature._peer_requester = p.requester
    feature._own_name = "Sender"
    result = await asyncio.wait_for(
        feature.ask_agent("Peer", "An unanswered blocking native ask"), 5
    )
    assert result.data["sent"] is None and result.data["delivery_state"] == "unknown"
    async with asyncio.timeout(5):
        while p.manager._subscribers:
            await asyncio.sleep(0.001)
    rows = await p.sender_manager.outbound_store.list_recent()
    assert len(rows) == 1 and rows[0].state.value == "unknown"
    tasks = await p.manager.task_store.list_for_creator(
        ListTasksRequest(tenant=p.scope.tenant),
        creator_agent_id=PRINCIPAL,
        recipient_agent_id=p.recipient.did,
    )
    assert (
        len(tasks.tasks) == 1
        and tasks.tasks[0].status.state
        == __import__(
            "a2a.types", fromlist=["TaskState"]
        ).TaskState.TASK_STATE_SUBMITTED
    )


async def test_public_question_tool_uses_native_intent_and_restarts_without_legacy_rows(
    local_native_peer, monkeypatch
):
    from kestrel_sovereign.features.peers.feature import PeersFeature
    from tests.unit.test_a2a_native_question_supervisor import dispatcher_for
    from kestrel_sovereign.a2a.extensions import QUESTION_CORRELATION_URI
    from google.protobuf.json_format import MessageToDict

    p = local_native_peer
    state = SimpleNamespace(backend=p.backend, store=p.sender_manager.question_store)
    agent, dispatcher = await dispatcher_for(state)
    p.sender.dispatcher = dispatcher
    p.sender._track_background_task = agent._track_background_task
    p.sender.get_current_turn_id = lambda: "asking-turn"
    p.sender.get_turn_bound_session_id = lambda: "asking-session"
    monkeypatch.setenv("KESTREL_A2A_QUESTION_POLL_SECONDS", "0.01")

    def feature():
        result = PeersFeature(p.sender)
        result._peer_router = p.directory
        result._peer_requester = p.requester
        result._own_name = "Sender"
        return result

    first = feature()
    second = None
    try:
        await first.post_all_features_loaded(p.sender)
        result = await first.send_a2a_question(
            "Peer",
            "Native question through the public tool",
            parts=[{"data": {"value": 42}}],
        )
        assert result.data["sent"] is True and result.data["awaiting_reply"]
        question = await state.store.get(result.data["question_id"])
        assert question.dispatch_id == result.data["dispatch_id"]
        assert (question.origin_turn_id, question.origin_session_id) == (
            "asking-turn",
            "asking-session",
        )
        task = await p.manager.get_task_for_creator(
            result.data["remote_task_id"], PRINCIPAL, recipient_agent_id=p.recipient.did
        )
        assert (
            MessageToDict(task.metadata)[QUESTION_CORRELATION_URI]["question_id"]
            == question.question_id
        )
        assert MessageToDict(task.history[0].parts[1])["data"] == {"value": 42}
        assert not await p.backend.table_exists("a2a_outbound_tasks")
        await first.shutdown()
        # No in-memory subscriber or caller turn survives the feature restart.
        await p.manager.complete_task(
            task.id, "The native answer", recipient_agent_id=p.recipient.did
        )
        second = feature()
        await second.post_all_features_loaded(p.sender)
        async with asyncio.timeout(5):
            while (
                not (await state.store.get(question.question_id)).receiver_settled
                or await state.store.pending_wakes()
            ):
                await asyncio.sleep(0.01)
        assert (
            await p.backend.fetch_one(
                "SELECT COUNT(*) FROM durable_signal_events WHERE source = ?",
                ("a2a.question_answered",),
            )
        )[0] == 1
        assert (
            await p.sender_manager.outbound_store.get(question.dispatch_id)
        ).remote_task_state == "completed"
    finally:
        await first.shutdown()
        if second is not None:
            await second.shutdown()
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def test_peer_session_allows_queued_topology_writer_before_card_discovery(
    local_native_peer, monkeypatch
):
    p = local_native_peer
    directory_entered = asyncio.Event()
    release_directory = asyncio.Event()
    writer_finished = asyncio.Event()
    original = p.directory.resolve_peer_by_agent_id
    calls = 0

    async def paused_directory(requester, agent_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            directory_entered.set()
            await release_directory.wait()
        return await original(requester, agent_id)

    monkeypatch.setattr(p.directory, "resolve_peer_by_agent_id", paused_directory)

    async def open_session():
        async with p.directory.open_a2a_session(p.requester, p.peer) as session:
            assert session.recipient_scope.recipient_id == p.recipient.did

    async def mutate_topology():
        async with p.host.a2a_lifecycle_lease():
            writer_finished.set()

    opening = asyncio.create_task(open_session())
    writer = None
    try:
        async with asyncio.timeout(5):
            await directory_entered.wait()
            writer = asyncio.create_task(mutate_topology())
            while not p.host._a2a_lifecycle_lock._waiting_writers:
                await asyncio.sleep(0)
            assert not writer_finished.is_set()
            release_directory.set()
            await opening
            await writer
        assert writer_finished.is_set()
    finally:
        release_directory.set()
        for task in (opening, writer):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (opening, writer) if task is not None),
            return_exceptions=True,
        )


async def test_dead_question_resident_refuses_new_sends_before_delivery(
    local_native_peer,
):
    from kestrel_sdk.tools.result import ToolResultStatus
    from kestrel_sovereign.features.peers.feature import PeersFeature
    from tests.unit.test_a2a_native_question_supervisor import dispatcher_for

    p = local_native_peer
    state = SimpleNamespace(backend=p.backend, store=p.sender_manager.question_store)
    agent, dispatcher = await dispatcher_for(state)
    p.sender.dispatcher = dispatcher
    p.sender._track_background_task = agent._track_background_task
    p.sender.get_current_turn_id = lambda: "turn"
    feature = PeersFeature(p.sender)
    feature._peer_router, feature._peer_requester = p.directory, p.requester
    feature._own_name = "Sender"
    try:
        await feature.post_all_features_loaded(p.sender)
        owner = feature._question_supervisor_task
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
        result = await feature.send_a2a_question(
            "Peer", "Refuse if resident recovery failed"
        )
        assert result.status is ToolResultStatus.ERROR
        assert "recovery is not running" in result.error
        assert await p.sender_manager.outbound_store.list_recent() == []
    finally:
        await feature.shutdown()
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*agent.tasks, return_exceptions=True)


@pytest_asyncio.fixture
async def native_question_peer(local_native_peer):
    from kestrel_sovereign.features.peers.feature import PeersFeature
    from tests.unit.test_a2a_native_question_supervisor import dispatcher_for

    p = local_native_peer
    signal_agent, dispatcher = await dispatcher_for(
        SimpleNamespace(backend=p.backend, store=p.sender_manager.question_store)
    )
    p.sender.dispatcher = dispatcher
    p.sender._track_background_task = signal_agent._track_background_task
    p.sender.get_current_turn_id = lambda: "asking-turn"
    p.sender.get_turn_bound_session_id = lambda: "asking-session"
    feature = PeersFeature(p.sender)
    feature._peer_router, feature._peer_requester = p.directory, p.requester
    feature._own_name = "Sender"
    try:
        await feature.post_all_features_loaded(p.sender)
        yield SimpleNamespace(**vars(p), feature=feature, dispatcher=dispatcher)
    finally:
        await feature.shutdown()
        await dispatcher.shutdown_durable_delivery()
        await asyncio.gather(*signal_agent.tasks, return_exceptions=True)


async def test_public_question_sender_retains_live_submission_ownership_until_receipt(
    native_question_peer,
):
    p = native_question_peer
    accepted, release = asyncio.Event(), asyncio.Event()

    class HoldNativeAcknowledgement:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            async def delayed(frame):
                if (
                    scope.get("method") == "POST"
                    and frame["type"] == "http.response.start"
                ):
                    accepted.set()
                    await release.wait()
                await send(frame)

            await self.app(scope, receive, delayed)

    p.app.add_middleware(HoldNativeAcknowledgement)
    sending = asyncio.create_task(
        p.feature.send_a2a_question("Peer", "Retain the physical original sender")
    )
    try:
        async with asyncio.timeout(5):
            await accepted.wait()
        [question] = await p.sender_manager.question_store.unsettled()
        assert p.sender_manager.outbound_store.submission_active(question.dispatch_id)
        dispatch = await p.sender_manager.outbound_store.get(question.dispatch_id)
        assert dispatch.state.value == "submitting" and not sending.done()
        release.set()
        result = await sending
        assert result.data["sent"] is True
        assert not p.sender_manager.outbound_store.submission_active(
            question.dispatch_id
        )
    finally:
        release.set()
        if not sending.done():
            sending.cancel()
        await asyncio.gather(sending, return_exceptions=True)
