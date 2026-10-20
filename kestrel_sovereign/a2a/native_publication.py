"""Host-owned publication of one current recipient's signed native contract."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from a2a.types import AgentCard
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from kestrel_sovereign import __version__
from kestrel_sovereign.a2a.native_card import (
    REQUIRED_NATIVE_EXTENSIONS,
    create_native_agent_card,
)
from kestrel_sovereign.a2a.native_host import create_hosted_native_guard
from kestrel_sovereign.a2a.native_routes import (
    NativeA2AHTTP,
    NativeA2AHTTPRegistry,
    native_http_interfaces,
)
from kestrel_sovereign.a2a.native_sender import NativeProofPolicy
from kestrel_sovereign.a2a.request_handler import NativeA2ARequestHandler
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.multi_agent.route_name import encode_agent_route_name

if TYPE_CHECKING:
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager


@dataclass(frozen=True)
class NativeHostingConfiguration:
    """No public URL or discovery recipient is inferred from an incoming request."""

    public_base_url: str
    discovery_routing_key: str | None = None

    def __post_init__(self):
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
        base = environment.get("KESTREL_BASE_URL")
        discovery = environment.get("KESTREL_A2A_DISCOVERY_AGENT")
        if base is None:
            if discovery is not None:
                raise ValueError("A2A discovery requires KESTREL_BASE_URL")
            return None
        return cls(base, discovery)


class HostedNativeA2APublication:
    """Prepare privately, publish without an await, withdraw by exact ownership.

    AgentManager calls ``publish`` while retaining its topology writer, after
    the candidate's registration and identity become visible to its own policy
    checks. Requests and card reads acquire the matching reader, so no request
    can observe an intermediate registration. Preparation owns no app routes.
    """

    def __init__(
        self,
        manager: "AgentManager",
        registry: NativeA2AHTTPRegistry,
        recipient: "KestrelAgent",
        routing_key: str,
        configuration: NativeHostingConfiguration,
        profile: dict[str, str],
    ):
        self.manager = manager
        self.registry = registry
        self.recipient = recipient
        self.routing_key = routing_key
        self.configuration = configuration
        self._profile = profile
        self._binding: NativeA2AHTTP | None = None
        self._guard = None

    @classmethod
    async def prepare(cls, manager, registry, recipient, routing_key, configuration):
        profile = await recipient.get_public_profile()
        publication = cls(
            manager, registry, recipient, routing_key, configuration, profile
        )
        # Bad configuration, keys or skills refuse private onboarding, before
        # AgentManager installs any routing entry for this candidate.
        publication._build_card(profile)
        return publication

    def _build_card(self, profile: dict[str, str]) -> AgentCard:
        return create_native_agent_card(
            identity=self.recipient.identity,
            scope=self.recipient.task_manager.task_store.scope,
            routing_key=self.routing_key,
            public_base_url=self.configuration.public_base_url,
            name=self.recipient.resolve_effective_name(default=profile["name"]),
            description=profile["description"],
            version=__version__,
            skills=self.recipient.task_manager.get_agent_skills(),
        )

    def publish(self) -> None:
        """The manager owns its exclusive topology lease for this synchronous commit."""
        if self._binding is not None:
            if self.registry.is_current(self._binding):
                return
            raise RuntimeError("A retired native publication cannot be republished")
        guard = create_hosted_native_guard(
            self.manager,
            routing_key=self.routing_key,
            recipient=self.recipient,
            required_extensions=REQUIRED_NATIVE_EXTENSIONS,
            proof_policy=NativeProofPolicy(),
            clock=time.time,
        )
        card = self._build_card(self._profile)
        binding = NativeA2AHTTP(
            NativeA2ARequestHandler(self.recipient.task_manager, guard, card),
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


async def prepare_hosted_native_publication(
    app: Starlette,
    manager: "AgentManager",
    routing_key: str,
    recipient: "KestrelAgent",
    environment: Mapping[str, str],
) -> HostedNativeA2APublication | None:
    """The startup/cold-onboarding seam; no public routes exist until commit."""
    configuration = NativeHostingConfiguration.from_environment(environment)
    if configuration is None:
        return None
    registry = getattr(app.state, "native_a2a_http", None)
    if registry is None:
        registry = NativeA2AHTTPRegistry(app)
        app.state.native_a2a_http = registry
    if not isinstance(registry, NativeA2AHTTPRegistry):
        raise TypeError("Host native A2A routes require their published registry")
    registry.require_application(app)
    return await HostedNativeA2APublication.prepare(
        manager, registry, recipient, routing_key, configuration
    )
