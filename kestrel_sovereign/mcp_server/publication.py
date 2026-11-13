"""Host-owned native MCP preparation, committed routing and physical retirement."""

import asyncio
from contextlib import asynccontextmanager

from kestrel_sdk.security.encryption import get_agent_key
from starlette.routing import Route

from kestrel_sovereign._async_ownership import (
    OwnedOperation,
    raise_owned_outcome,
)
from kestrel_sovereign.agent.custody import ReleaseOutcome
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    execution_finite_error,
    require_execution_work,
)
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.multi_agent.route_name import encode_agent_route_name
from kestrel_sovereign.security.capability_authority import (
    AsyncReaderWriterLock,
    HostCapabilityAccessDenied,
)

from .consent import MCPConsentAuthority
from .console import MCPConsentController, consent_page
from .grants import MCPAuthorizationGrants
from .pending import PendingMCPAuthorizations
from .policy import MCPServerPolicy
from .transport import NativeMCPHTTP, NativeMCPHTTPRegistry, NativeMCPServer


def native_mcp_registry(app):
    """Resolve the original registry owned by this exact Core application."""
    registry = getattr(app.state, "native_mcp_http", None)
    if registry is None:
        registry = NativeMCPHTTPRegistry(app)
        app.state.native_mcp_http = registry
    if not isinstance(registry, NativeMCPHTTPRegistry):
        raise TypeError("Original native MCP HTTP registry required")
    registry.require_application(app)
    return registry


class NativeMCPPublication:
    """One original agent incarnation, borrowed primary backend and SDK task."""

    def __init__(self, app, agent, routing_key, policy, *, sovereign_key, manager=None):
        if vars(agent).get("_native_mcp_publication") is not None:
            raise RuntimeError("Retire the original native MCP publication before replacement")
        self.app, self.agent, self.manager, self.routing_key = app, agent, manager, routing_key
        self.policy, self._sovereign_key = policy, sovereign_key
        self._did, self._identity = agent.did, agent.identity
        self._execution_custody = getattr(agent, "_execution_custody", None)
        if self._execution_custody is not None and not isinstance(self._execution_custody, ExecutionCustody):
            raise TypeError("Native MCP requires its original execution custody")
        self._ready = agent._runtime_publication_ready
        self.backend = resolve_feature_database(agent).backend
        self.registry = native_mcp_registry(app)
        self._preparer = asyncio.current_task()
        self._preparing, self._prepared, self._published, self._retired = True, False, False, False
        self._resident_operation = self._close_operation = None
        self._stop = asyncio.Event()
        self._requests = AsyncReaderWriterLock()
        self.runtime = self.controller = self.binding = None
        self.resource = policy.console_origin + "/api/agent-routes/" + encode_agent_route_name(routing_key) + "/mcp"

    @property
    def _resident(self):
        return self._resident_operation.task if self._resident_operation is not None else None

    @property
    def _close_task(self):
        return self._close_operation.task if self._close_operation is not None else None

    def _runtime_current(self):
        if self._execution_custody is not None:
            self._execution_custody.require_work()
        require_execution_work(self.agent)
        if (self.agent.did != self._did or self.agent.identity is not self._identity
                or getattr(self.agent, "_execution_custody", None) is not self._execution_custody
                or self.agent._runtime_publication_ready is not self._ready or not self._ready.is_set()
                or resolve_feature_database(self.agent).backend is not self.backend
                or vars(self.agent).get("_native_mcp_publication") is not self or self._retired):
            raise HostCapabilityAccessDenied("Original native MCP agent runtime is unavailable")

    def require_current(self):
        self._runtime_current()
        if self._preparing and asyncio.current_task() in (self._preparer, self._resident):
            return
        if (not self._published or self._resident is None or self._resident.done()
                or not self.registry.is_current(self.binding)):
            raise HostCapabilityAccessDenied("Original native MCP publication is unavailable")
        if self.manager is None:
            current = getattr(self.app.state, "agent", None) is self.agent
        else:
            current = (getattr(self.app.state, "agent_manager", None) is self.manager
                and self.manager.get_agent(self.routing_key) is self.agent
                and vars(self.agent).get("_agent_manager_published") is True)
        if not current:
            raise HostCapabilityAccessDenied("Original native MCP agent was withdrawn")

    @classmethod
    async def prepare(cls, app, agent, routing_key, policy: MCPServerPolicy, *, sovereign_key, manager=None):
        publication = cls(app, agent, routing_key, policy, sovereign_key=sovereign_key, manager=manager)
        agent._native_mcp_publication = publication
        custody = agent._resource_custody()
        custody.acquire("native_mcp_server")
        try:
            publication._runtime_current()
            key = sovereign_key()
            from kestrel_sovereign.security.sovereign_key import (
                is_ephemeral_sovereign_key,
            )
            if not key or is_ephemeral_sovereign_key(key):
                raise ValueError("Native MCP publication requires a durable configured sovereign key")
            get_agent_key(agent.did, "service-keys")
            pending = PendingMCPAuthorizations(publication.backend, agent.did, publication.resource, policy)
            await pending.initialize()
            consent = MCPConsentAuthority(agent.did, publication.resource, policy,
                require_current=publication.require_current, sovereign_key=sovereign_key)
            grants = MCPAuthorizationGrants(pending, publication.resource + "/oauth", consent_authority=consent)
            await grants.initialize()
            publication.runtime = NativeMCPServer(grants, name=agent.agent_name,
                consent_url=publication.resource + "/consent")
            publication.controller = MCPConsentController(agent, publication.runtime.authorization,
                require_current=publication.require_current)
            prefix = "" if manager is None else "/api/agent-routes/" + encode_agent_route_name(routing_key)

            async def shell(_request):
                publication.require_current()
                return consent_page(prefix)

            publication.runtime.app.router.routes.append(Route(
                "/api/agent-routes/" + encode_agent_route_name(routing_key) + "/mcp/consent", shell, methods=["GET"]))
            publication.binding = NativeMCPHTTP(routing_key, publication.runtime,
                request_lifetime=publication.request_lifetime)
            ready = asyncio.get_running_loop().create_future()

            async def resident():
                async with publication.runtime.app.router.lifespan_context(publication.runtime.app):
                    ready.set_result(None)
                    await publication._stop.wait()

            resident_context = agent._runtime_owner_context()
            publication._resident_operation = OwnedOperation(resident(), name="native-mcp-server:" + routing_key,
                context=resident_context, retain_originals=True, retirement_resource=publication)
            if publication._resident is None:
                outcome = await publication._resident_operation.observe()
                raise_owned_outcome(outcome, operation="native MCP SDK admission", retain_originals=True)
                raise RuntimeError("Original native MCP SDK admission did not retain its task")
            await asyncio.wait({ready, publication._resident}, return_when=asyncio.FIRST_COMPLETED)
            if publication._resident.done():
                outcome = await publication._resident_operation.observe()
                raise_owned_outcome(outcome, operation="native MCP SDK startup", retain_originals=True)
                raise RuntimeError("The original native MCP SDK runtime ended during preparation")
            publication._runtime_current()
            publication._prepared = True
            return publication
        except BaseException as original:
            try:
                await publication.close()
            except BaseException as retirement:  # noqa: BLE001 - retain startup and original native retirement evidence
                raise execution_finite_error(original, retirement, retain_originals=True) from None
            raise
        finally:
            publication._preparing = False
            publication._preparer = None

    def publish(self):
        self._runtime_current()
        if not self._prepared or self._published or self._resident.done():
            raise RuntimeError("Original prepared native MCP runtime required for publication")
        if self.manager is None:
            current = getattr(self.app.state, "agent", None) is self.agent
        else:
            current = (getattr(self.app.state, "agent_manager", None) is self.manager
                and self.manager.get_agent(self.routing_key) is self.agent)
        if not current:
            raise HostCapabilityAccessDenied("Original native MCP agent registration is unavailable")
        self.registry.publish(self.binding)
        self._published = True

    def withdraw(self):
        self._retired, self._published, self._preparing = True, False, False
        self._stop.set()
        if self.binding is not None:
            self.registry.withdraw(self.binding)

    @asynccontextmanager
    async def request_lifetime(self):
        self.require_current()
        async with self._requests.read():
            self.require_current()
            yield self.require_current

    async def close(self):
        failure = None
        try:
            self.withdraw()
        except BaseException as error:  # noqa: BLE001 - drain original readers and SDK before surfacing withdrawal
            failure = error
        if self._close_operation is None:
            async def drain():
                async with self._requests:
                    if self.controller is not None:
                        async with self.controller._gate:
                            pass
                resident_failure = None
                if self._resident_operation is not None:
                    outcome = await self._resident_operation.observe()
                    try:
                        raise_owned_outcome(outcome, operation="native MCP SDK retirement", retain_originals=True)
                    except BaseException as error:  # noqa: BLE001 - preserve the joined original SDK completion receipt
                        resident_failure = error
                terminal = execution_finite_error(failure, resident_failure, retain_originals=True)
                if terminal is not None:
                    raise terminal
                self.agent._resource_custody().settle("native_mcp_server", ReleaseOutcome.RELEASED)
                if vars(self.agent).get("_native_mcp_publication") is self:
                    self.agent._native_mcp_publication = None
                return ReleaseOutcome.RELEASED
            self._close_operation = OwnedOperation(drain(), name="native-mcp-server-close:" + self.routing_key,
                retain_originals=True, retirement_resource=self)
        outcome = await self._close_operation.observe()
        return raise_owned_outcome(outcome, operation="native MCP publication close", retain_originals=True)
