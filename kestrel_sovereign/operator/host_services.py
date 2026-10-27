"""Retained host service discovery; this supplies no operator/content grant."""

from contextlib import asynccontextmanager

from kestrel_sdk.operator import OperatorAuthorizationError
from .runtime import OperatorRuntimeRegistry, ServiceOperation, ServiceDiscoveryIdentity


class HostServiceBinding:
    """One agent's explicit binding to the original host registry lifecycle."""

    host_service_identity_api_version = 1

    def __init__(self, agent, resolver=None):
        self._agent = agent
        self._resolver = None
        self._generation = 0
        self._retired = False
        self.bind(resolver)

    @staticmethod
    def validate(resolver):
        if resolver is not None and not isinstance(resolver, OperatorRuntimeRegistry):
            raise TypeError(
                "host services require the original OperatorRuntimeRegistry"
            )

    def validate_binding(self, resolver):
        self.validate(resolver)
        if self._retired and resolver is not None:
            raise OperatorAuthorizationError("agent host service binding retired")

    def bind(self, resolver):
        self.validate_binding(resolver)
        self._generation += 1
        self._resolver = resolver

    def clear(self, expected):
        if self._resolver is not expected:
            return False
        self.bind(None)
        return True

    def retire(self):
        self._retired = True
        self._generation += 1
        self._resolver = None

    @asynccontextmanager
    async def operation(self, requirement):
        agent, resolver = self._agent, self._resolver
        generation, did = self._generation, agent.did
        publication = getattr(agent, "_host_context_publication_state", None)
        published_generation = getattr(publication, "generation", None)
        manager = getattr(agent, "_agent_manager", None)
        name = None if manager is None else manager.get_agent_name(did)
        active = True

        def current():
            from kestrel_sovereign.execution_custody import require_execution_work

            require_execution_work(agent)
            gate = getattr(agent, "_host_context_publication_gate", None)
            if (
                not active
                or self._retired
                or self._generation != generation
                or self._resolver is not resolver
                or resolver is None
                or self._agent is not agent
                or agent.did != did
                or getattr(agent, "_agent_manager", None) is not manager
                or (
                    manager is not None
                    and (name is None or manager.get_agent(name) is not agent)
                )
                or getattr(agent, "_agent_manager_published", None) is False
                or (gate is not None and not gate.is_set())
                or getattr(agent, "_host_context_publication_state", None)
                is not publication
                or (
                    publication is not None
                    and (
                        getattr(publication, "generation", None) != published_generation
                        or getattr(publication, "service_resolver", None)
                        is not resolver
                    )
                )
            ):
                raise OperatorAuthorizationError(
                    "original host service discovery is unavailable"
                )

        try:
            current()
            async with resolver.host_service_operation(requirement) as original:

                def check():
                    current()
                    original.check_current()

                check()
                identity = getattr(original, "discovery_identity", None)
                if type(identity) is ServiceDiscoveryIdentity:
                    original.require_issued()
                    identity = identity._with_host_binding(
                        self, generation, publication, published_generation
                    )
                else:
                    # Preserve operation API1; only the separately advertised
                    # identity consumer requires genuine identity metadata.
                    identity = None
                if identity is None:
                    yield ServiceOperation(original.service, check)
                else:
                    # The outer frame retains the genuine original registry
                    # checker, not another record's copied metadata/callback.
                    def issued_check():
                        original.require_issued()
                        check()
                    yield ServiceOperation._issue(original.service, issued_check, identity)
                check()
        finally:
            active = False
