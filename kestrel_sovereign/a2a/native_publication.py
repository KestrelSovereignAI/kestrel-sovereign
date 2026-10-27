"""Host-owned publication of one current recipient's signed native contract."""

import os
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
from kestrel_sdk.native_a2a.client import A2ACardResolver
from kestrel_sdk.native_a2a.types import AgentCard
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from kestrel_sovereign import __version__
from kestrel_sovereign.a2a.did_registry import same_a2a_sender_identity_witness
from kestrel_sovereign.a2a.native_card import (
    REQUIRED_NATIVE_EXTENSIONS,
    create_native_agent_card,
)
from kestrel_sovereign.a2a.native_client import (
    NativePeerSession,
    NativeReconciliationPolicy,
    NativeSenderInterceptor,
)
from kestrel_sovereign.a2a.native_host import create_hosted_native_guard
from kestrel_sovereign.a2a.native_routes import (
    NativeA2AHTTP,
    NativeA2AHTTPRegistry,
    native_http_interfaces,
)
from kestrel_sovereign.a2a.native_security import NativeDIDRequestGuard
from kestrel_sovereign.a2a.native_sender import NativeProofPolicy
from kestrel_sovereign.a2a.push_policy import NativePushPolicy
from kestrel_sovereign.a2a.push_service import start_native_push_service
from kestrel_sovereign.a2a.request_handler import NativeA2ARequestHandler
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.task_manager import OutboundActorScopeError
from kestrel_sovereign.execution_custody import (
    bind_native_http_execution_admission,
    is_execution_control_error,
    native_execution_response_owner,
    owned_native_http_client,
)
from kestrel_sovereign.features.peers.directory import (
    PeerAccessDeniedError,
    PeerIdentity,
    PeerUnavailableError,
)
from kestrel_sovereign.multi_agent.route_name import encode_agent_route_name

if TYPE_CHECKING:
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.multi_agent.config import MultiAgentConfig

PUBLIC_BASE_URL_ENV = "KESTREL_BASE_URL"


def _host_publishes_native_a2a(environment: Mapping[str, str]) -> bool:
    """Only an explicit public base URL publishes; a listen address never does."""
    return environment.get(PUBLIC_BASE_URL_ENV) is not None


def unpublished_cohost_warning(
    environment: Mapping[str, str], config: "MultiAgentConfig"
) -> str | None:
    """Name a multi-agent host whose co-hosted agents cannot reach each other.

    Core 0.53 defaulted the public URL to ``http://localhost:8888``. Native
    publication requires it explicitly, so a host upgraded without it starts
    healthy and every co-hosted send fails only when an agent first tries one
    (#3872). The port is advice for the operator, never a substitute URL.
    """
    agents = len(config.get_local_agents())
    if agents < 2 or _host_publishes_native_a2a(environment):
        return None
    return (
        f"{PUBLIC_BASE_URL_ENV} is unset: none of this host's {agents} agents "
        "publishes a native A2A interface, so co-hosted agents cannot send A2A "
        "tasks, messages, questions or stops to each other. For a local host "
        f"set {PUBLIC_BASE_URL_ENV}=http://127.0.0.1:{config.host.port} and "
        "KESTREL_A2A_PUSH_ENABLED=false."
    )


def unpublished_peer_detail(environment: Mapping[str, str]) -> str:
    """Why a co-hosted recipient has no native interface to open."""
    if _host_publishes_native_a2a(environment):
        return "Peer has no published native A2A interface"
    return (
        "Peer has no published native A2A interface: this host has no "
        f"{PUBLIC_BASE_URL_ENV}, so none of its agents publishes one. The "
        f"operator must set {PUBLIC_BASE_URL_ENV} and restart the host."
    )


@dataclass(frozen=True)
class NativeHostingConfiguration:
    """No public URL or discovery recipient is inferred from an incoming request."""

    public_base_url: str
    discovery_routing_key: str | None = None
    push: NativePushPolicy | None = None

    def __post_init__(self):
        if self.push is not None and not isinstance(self.push, NativePushPolicy):
            raise TypeError("Native hosting requires a validated push policy")
        native_http_interfaces(
            self.public_base_url,
            routing_key="configuration",
            scope=NativeRecipientScope("configuration"),
        )
        if self.discovery_routing_key is not None:
            encode_agent_route_name(self.discovery_routing_key)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]):
        """Unconfigured hosts run operator APIs without public peer discovery."""
        base = environment.get(PUBLIC_BASE_URL_ENV)
        discovery = environment.get("KESTREL_A2A_DISCOVERY_AGENT")
        if not _host_publishes_native_a2a(environment):
            if discovery is not None:
                raise ValueError(f"A2A discovery requires {PUBLIC_BASE_URL_ENV}")
            return None
        return cls(base, discovery, NativePushPolicy.from_environment(environment))


class NativeA2APublication:
    """One native card, handler and route publication for every host owner.

    The embedding host supplies its request guard factory and owns the matching
    lifecycle writer during synchronous publish/withdraw. Requests and card
    reads use that guard's reader. Preparation owns no application routes.
    """

    def __init__(
        self,
        registry: NativeA2AHTTPRegistry,
        recipient: "KestrelAgent",
        routing_key: str,
        configuration: NativeHostingConfiguration,
        profile: dict[str, str],
        *,
        guard_factory: Callable[[], NativeDIDRequestGuard],
    ):
        if not callable(guard_factory):
            raise TypeError("Native publication requires its host-owned guard factory")
        self._guard_factory = guard_factory
        self.registry = registry
        self.recipient = recipient
        self.routing_key = routing_key
        self.configuration = configuration
        self._profile = profile
        self._binding: NativeA2AHTTP | None = None
        self._guard = None
        self._push_service = None

    @classmethod
    async def prepare(
        cls, registry, recipient, routing_key, configuration, *, guard_factory
    ):
        profile = await recipient.get_public_profile()
        publication = cls(
            registry,
            recipient,
            routing_key,
            configuration,
            profile,
            guard_factory=guard_factory,
        )
        # Bad configuration, keys or skills refuse private onboarding, before
        # AgentManager installs any routing entry for this candidate.
        publication._build_card(profile, preparing=True)
        if configuration.push is not None:
            publication._push_service = await start_native_push_service(
                recipient.task_manager,
                recipient.dispatcher.durable_store,
                policy=configuration.push,
                task_factory=recipient._track_runtime_task,
                authorize_creator=publication._authorize_push_creator,
            )
        publication._build_card(profile)
        return publication

    async def _authorize_push_creator(self, principal_id: str) -> bool:
        guard, binding = self._guard, self._binding
        if guard is None or binding is None:
            return False
        try:
            async with guard.hold_retained_authority(principal_id):
                return self._guard is guard and self._binding is binding and self.registry.is_current(binding)
        except HTTPException as error:
            if is_execution_control_error(error) or error.status_code not in {401, 403, 404}:
                raise
            return False

    def _build_card(
        self, profile: dict[str, str], *, preparing: bool = False
    ) -> AgentCard:
        service = self.recipient.task_manager.push_service
        if service is not None:
            service.require_ready()
        if not preparing and service is not self._push_service:
            raise RuntimeError("Native publication lost its original push service")
        if not preparing and (service is not None) != (
            self.configuration.push is not None
        ):
            raise RuntimeError(
                "Published push configuration differs from its bound service"
            )
        return create_native_agent_card(
            identity=self.recipient.identity,
            scope=self.recipient.task_manager.task_store.scope,
            routing_key=self.routing_key,
            public_base_url=self.configuration.public_base_url,
            name=self.recipient.resolve_effective_name(default=profile["name"]),
            description=profile["description"],
            version=__version__,
            skills=self.recipient.task_manager.get_agent_skills(),
            peer_stop=True,
            push_notifications=self.configuration.push is not None,
        )

    def publish(self) -> None:
        """The host owns its lifecycle writer for this synchronous commit."""
        if self._binding is not None:
            if self.registry.is_current(self._binding):
                return
            raise RuntimeError("A retired native publication cannot be republished")
        guard = self._guard_factory()
        if not isinstance(guard, NativeDIDRequestGuard):
            raise TypeError("Native publication requires its canonical request guard")
        card = self._build_card(self._profile)
        binding = NativeA2AHTTP(
            NativeA2ARequestHandler(
                self.recipient.task_manager,
                guard,
                card,
                peer_stop_target=self.recipient,
            ),
            public_base_url=self.configuration.public_base_url,
            card_modifier=self.get_card,
            public_discovery=(
                self.configuration.discovery_routing_key == self.routing_key
            ),
        )
        self.registry.publish(binding)
        self._guard = guard
        self._binding = binding

    def withdraw(self) -> None:
        if self._binding is not None:
            self.registry.withdraw(self._binding)

    @property
    def is_published(self) -> bool:
        """Whether this exact route binding remains registered with its host."""
        return self._binding is not None and self.registry.is_current(self._binding)

    async def get_card(self, _base: AgentCard | None = None) -> AgentCard:
        """Refresh public display/skills under the same live identity/policy lease."""
        if self._guard is None or self._binding is None:
            raise HTTPException(
                status_code=404, detail="Native recipient is not published"
            )
        async with self._guard.lifecycle_lease():
            if not self.registry.is_current(self._binding):
                raise HTTPException(
                    status_code=404, detail="Native recipient was retired"
                )
            authority = self._guard.authority_factory()
            authority.require_current(None)
            profile = await self.recipient.get_public_profile()
            card = self._build_card(profile)
            authority.require_current(None)
            return card


def native_publication_registry(app: Starlette) -> NativeA2AHTTPRegistry:
    """Own the one SDK route registry for this exact application."""
    registry = getattr(app.state, "native_a2a_http", None)
    if registry is None:
        registry = NativeA2AHTTPRegistry(app)
        app.state.native_a2a_http = registry
    if not isinstance(registry, NativeA2AHTTPRegistry):
        raise TypeError("Host native A2A routes require their published registry")
    registry.require_application(app)
    return registry


async def prepare_hosted_native_publication(
    app: Starlette,
    manager: "AgentManager",
    routing_key: str,
    recipient: "KestrelAgent",
    environment: Mapping[str, str],
) -> NativeA2APublication | None:
    """The startup/cold-onboarding seam; no public routes exist until commit."""
    configuration = NativeHostingConfiguration.from_environment(environment)
    if configuration is None:
        return None
    registry = native_publication_registry(app)
    return await NativeA2APublication.prepare(
        registry,
        recipient,
        routing_key,
        configuration,
        guard_factory=partial(
            create_hosted_native_guard,
            manager,
            routing_key=routing_key,
            recipient=recipient,
            required_extensions=REQUIRED_NATIVE_EXTENSIONS,
            proof_policy=NativeProofPolicy(),
            clock=time.time,
        ),
    )


@asynccontextmanager
async def open_hosted_native_peer_session(
    app: Starlette,
    manager: "AgentManager",
    *,
    sender,
    requester,
    peer: PeerIdentity,
    timeout: httpx.Timeout,
):
    """Route a same-host peer through the installed official SDK HTTP binding.

    The SDK uses the published HTTP routes for discovery and every operation.
    Sender egress is authorized here; recipient authorization runs only at the
    receiving native handler, in its own request context. No operator credential
    or in-process sender custody crosses that boundary.
    """
    async with manager.a2a_execution_lease():
        _, recipient = manager._resolve_hosted_a2a_route_binding(
            sender=sender,
            requester=requester,
            peer=peer,
        )
        sender_policy = manager.a2a_hosted_policy_for(sender)
        recipient_policy = manager.a2a_hosted_policy_for(recipient)
        publication = getattr(recipient, "_native_a2a_publication", None)
        if not isinstance(publication, NativeA2APublication):
            # Host onboarding decides publication from this same environment.
            raise PeerUnavailableError(unpublished_peer_detail(os.environ))
        publication.registry.require_application(app)
        sender_identity = sender.identity
        recipient_identity = recipient.identity
        sender_witness = manager.a2a_sender_identity_witness(
            sender_identity.signing_did
        )
        recipient_witness = manager.a2a_sender_identity_witness(
            recipient_identity.signing_did
        )
        sender_manager = sender.task_manager
        sender_task_store = sender_manager.task_store
        sender_scope = sender_task_store.scope
        try:
            outbound = sender_manager.outbound_for_actor(requester.identity)
            questions = sender_manager.questions_for_actor(requester.identity)
        except OutboundActorScopeError as error:
            raise PeerAccessDeniedError(str(error)) from error
        scope = recipient.task_manager.task_store.scope
        signing_document = {
            "id": recipient_identity.signing_did,
            "verificationMethod": recipient_identity.new_verification_methods,
        }

    def require_current():
        if (
            manager.a2a_hosted_policy_for(sender) is not sender_policy
            or manager.a2a_hosted_policy_for(recipient) is not recipient_policy
            or sender.task_manager is not sender_manager
            or sender_manager.outbound_store is not outbound
            or sender_manager.question_store is not questions
            or sender_manager.task_store is not sender_task_store
            or sender_task_store.scope != sender_scope
            or getattr(recipient, "_native_a2a_publication", None) is not publication
            or not publication.registry.is_current(publication._binding)
            or not same_a2a_sender_identity_witness(
                sender_witness,
                manager.a2a_sender_identity_witness(sender_identity.signing_did),
            )
            or not same_a2a_sender_identity_witness(
                recipient_witness,
                manager.a2a_sender_identity_witness(recipient_identity.signing_did),
            )
        ):
            raise PeerAccessDeniedError(
                "Native peer session registration or identity changed"
            )

    async def authorize():
        async with manager.a2a_execution_lease():
            require_current()
            current_peer = await sender_policy.router.resolve_peer_by_agent_id(
                requester, peer.agent_id
            )
            require_current()
            if not isinstance(current_peer, PeerIdentity) or (
                current_peer.agent_id,
                current_peer.routing_key,
            ) != (peer.agent_id, peer.routing_key):
                raise PeerAccessDeniedError(
                    "Recipient is no longer in the sender's directory"
                )
            (
                _,
                current_recipient,
            ) = manager._resolve_hosted_a2a_route_binding(
                sender=sender,
                requester=requester,
                peer=current_peer,
            )
            require_current()
            if current_recipient is not recipient:
                raise PeerAccessDeniedError(
                    "Native peer recipient registration changed"
                )

    async with open_native_peer_session(
        public_base_url=publication.configuration.public_base_url,
        routing_key=peer.routing_key,
        sender_identity=sender_identity,
        signing_did=recipient_identity.signing_did,
        signing_document=signing_document,
        scope=scope,
        outbound=outbound,
        questions=questions,
        authorize=authorize,
        timeout=timeout,
    ) as session:
        yield session


@asynccontextmanager
async def open_native_peer_session(
    *,
    public_base_url,
    routing_key,
    sender_identity,
    signing_did,
    signing_document,
    scope,
    outbound,
    questions,
    authorize,
    timeout,
):
    """Discover and open the one SDK session under its embedding's authority.

    The embedding supplies trusted routing, identity and durable stores. This
    common path owns HTTP acquisition, card discovery, SDK setup and retirement
    for in-process and separately hosted agents alike.
    """
    reconciliation_policy = NativeReconciliationPolicy.from_environment(os.environ)
    await authorize()
    async with owned_native_http_client(
        httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        )
    ) as http:
        from kestrel_sovereign.a2a.native_routes import native_card_path

        with bind_native_http_execution_admission(http, authorize):
            async with native_execution_response_owner(http):
                card = await A2ACardResolver(
                    http,
                    public_base_url,
                    agent_card_path=native_card_path(routing_key),
                ).get_agent_card()
        await authorize()
        interfaces = [
            item
            for item in card.supported_interfaces
            if item.protocol_binding == "JSONRPC"
        ]
        if len(interfaces) != 1:
            raise PeerUnavailableError(
                "Published peer has no unique native JSON-RPC interface"
            )
        interface = interfaces[0]
        expected = native_http_interfaces(
            public_base_url, routing_key=routing_key, scope=scope
        )[0]
        if interface != expected:
            raise PeerUnavailableError(
                "Signed peer interface differs from its trusted host route"
            )
        signer = NativeSenderInterceptor.for_identity(
            sender_identity,
            recipient=scope,
            extensions=frozenset(item.uri for item in card.capabilities.extensions),
            policy=NativeProofPolicy(),
            clock=time.time,
            nonce=lambda: uuid4().hex,
        )
        session = NativePeerSession(
            card=card,
            signing_did=signing_did,
            trusted_document=signing_document,
            recipient=scope,
            interface=interface,
            signer=signer,
            http_client=http,
            outbound_store=outbound,
            question_store=questions,
            reconciliation_policy=reconciliation_policy,
            authorize=authorize,
        )
        try:
            yield session
        finally:
            await session.close()
