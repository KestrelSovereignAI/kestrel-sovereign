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
            resolver=resolver,
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
            document = hosted.policy.resolver.resolve(did)
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
            resolver=CoordinatedResolver(),
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
