"""Authenticated operator input for native MCP continuations.

Core owns private readers, durable checkpoint custody and fresh tool authority.
The feature owns native SDK input types, schema checks and connection identity.
Human waits never retain a conversation lock or an old execution turn.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from kestrel_sdk.operator import OperatorContext
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.responses import JSONResponse

from kestrel_sovereign.auth import caller_context_scope, caller_identity_digest, current_caller_context
from kestrel_sovereign.execution_custody import require_execution_work
from kestrel_sovereign.mcp_input_store import (
    MCPInputBinding, MCPInputClaim, MCPInputConflict, MCPInputRecord, MCPInputStore,
)
from kestrel_sovereign.security.capability_authority import (
    AsyncReaderWriterLock, HostCapabilityAccessDenied, HostCapabilityAuthority,
    require_current_feature_host, require_raw_capability_disclosure,
)
from kestrel_sovereign.security.csrf import CSRFError
from kestrel_sovereign.security.private_delivery import AuthorizedCapabilityResponse
from kestrel_sovereign.security.request_auth import RequestAuthenticationService, current_request_authentication
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate
from kestrel_sovereign.tools.effect_journal import current_tool_effect_state, require_persistent_tool_effects


class MCPInputHostPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False,
        hide_input_in_errors=True)
    max_pending: int = Field(default=64, gt=0, le=4096)
    max_payload_bytes: int = Field(default=1_048_576, gt=0, le=16_777_216)
    max_request_bytes: int = Field(default=65_536, gt=0, le=1_048_576)
    retention_seconds: float = Field(default=900.0, gt=0, le=86400)
    operation_seconds: float = Field(default=30.0, gt=0, le=300)


class _Action(StrEnum):
    READ = 'mcp-input.read'
    RESPOND = 'mcp-input.respond'


class _Response(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, hide_input_in_errors=True)
    revision: int = Field(ge=0)
    operation_id: str = Field(min_length=1, max_length=256)
    responses: dict


@dataclass(frozen=True)
class MCPInputOrigin:
    operation_id: str
    session_id: str | None
    principal_id: str | None
    allow_blocking: bool
    require_current: Callable[[], None] = field(repr=False)


@dataclass
class _Reader:
    context: OperatorContext
    session_id: str
    authentication: object
    request: object = field(repr=False)
    borrowed: AsyncExitStack = field(repr=False)
    response_claim: MCPInputClaim | None = field(default=None, repr=False)
    app_view: object = field(default=None, repr=False)
    app_transport: object = field(default=None, repr=False)
    active: bool = True
    check: Callable[[], None] | None = None


def _error(error, prepared):
    from kestrel_sovereign.api_errors import api_error_response, execution_control_api_error

    control = execution_control_api_error(error)
    if control is not None:
        return api_error_response(status_code=control.status_code, code=control.code,
            message=control.message, details=control.details)
    if isinstance(error, (HostCapabilityAccessDenied, CSRFError)):
        status, code = 403, 'mcp_input_access_denied'
    elif isinstance(error, (ResourceOutcomeUnknown, TimeoutError)) or prepared:
        status, code = 503, 'mcp_input_outcome_unknown_do_not_replay'
    elif isinstance(error, (MCPInputConflict, ResourceConflict)):
        status, code = 409, 'mcp_input_unavailable'
    elif isinstance(error, (ValueError, ValidationError)):
        status, code = 400, 'invalid_mcp_input_response'
    else:
        status, code = 500, 'mcp_input_service_error'
    return JSONResponse({'error': {'code': code}}, status_code=status)


class MCPInputHost:
    def __init__(self, agent, feature, *, configuration: dict, is_current: Callable[[], bool]):
        if not callable(is_current):
            raise TypeError('MCP input requires its original feature lifetime')
        self.agent, self.feature = agent, feature
        self._agent_id, self._is_current = agent.did, is_current
        self.policy = MCPInputHostPolicy.model_validate(configuration)
        database = require_persistent_tool_effects(agent).database
        self.store = MCPInputStore(database, agent_did=agent.did,
            max_pending=self.policy.max_pending, max_payload_bytes=self.policy.max_payload_bytes)
        self.authority = HostCapabilityAuthority(capability='mcp-input', label='MCP input')
        self._gate = AsyncReaderWriterLock()
        self._reader = ContextVar('mcp_input_private_reader', default=None)
        self._runtime = None
        self._accepting, self.retired = True, False

    def require_prepared(self):
        """Schema preparation precedes feature activation; it grants no I/O."""
        if (not self._accepting or self.retired or self.agent.did != self._agent_id
                or self._is_current() is not True
                or require_persistent_tool_effects(self.agent).database is not self.store.database):
            raise HostCapabilityAccessDenied('Original MCP input preparation owner changed')

    def require_current(self, request=None):
        self.require_prepared()
        require_current_feature_host(self.agent, self.feature, self._agent_id,
            available=self._accepting and not self.retired, is_current=self._is_current,
            label='MCP input', request=request)

    async def initialize(self):
        self.require_prepared()
        await self.store.initialize()
        self.require_prepared()

    def attach(self, runtime):
        if self._runtime is not None or any(not callable(getattr(runtime, name, None))
                for name in ('describe', 'respond')):
            raise TypeError('MCP input requires one native feature runtime')
        self._runtime = runtime

    def origin(self) -> MCPInputOrigin:
        """Capture an actual admitted effect, never a caller-supplied label."""
        from kestrel_sovereign.agent.turn_lifecycle import current_turn_is_signal_driven
        from kestrel_sovereign.features.security.hooks import NON_INTERACTIVE_SESSION_IDS

        self.require_current()
        turn, claim = current_tool_effect_state(self.agent)
        if turn is None or claim is None or claim.attempt is not turn.attempt:
            raise HostCapabilityAccessDenied('Original governed MCP dispatch is required')
        turn.require_live(self.agent)
        require_execution_work(self.agent)
        session = self.agent.get_turn_bound_session_id()
        if session is None:
            from kestrel_sovereign.agent.parts import current_part_collector, current_tool_publication_context
            collector = current_part_collector()
            destination = getattr(collector, 'publication_context', None)
            if destination is not None:
                original = current_tool_publication_context(self.agent, destination.session_id)
                session = original.session_id if original is not None else None
        reader = self._reader.get()
        if session is None and reader is not None:
            self._require_reader()
            session = reader.session_id
        if session is None:
            for host in getattr(self.agent, '_mcp_apps_hosts', ()):
                session = host.current_call_session_id()
                if session is not None:
                    break
        authentication = current_request_authentication()
        caller = current_caller_context()
        interactive = (isinstance(session, str) and bool(session.strip())
            and session not in NON_INTERACTIVE_SESSION_IDS and not current_turn_is_signal_driven()
            and authentication is not None and caller is authentication.caller)

        def check():
            self.require_current()
            actual_turn, actual_claim = current_tool_effect_state(self.agent)
            if actual_turn is not turn or actual_claim is not claim:
                raise HostCapabilityAccessDenied('Original MCP dispatch changed')
            turn.require_live(self.agent)
            require_execution_work(self.agent)
            if interactive:
                authentication.require_current()
                if current_caller_context() is not authentication.caller:
                    raise HostCapabilityAccessDenied('Original MCP input caller changed')
        check()
        return MCPInputOrigin(claim.claim_id, session,
            caller_identity_digest(caller) if interactive else None, interactive, check)

    def _revalidate_origin(self, description: MCPInputOrigin) -> MCPInputOrigin:
        """A description/callback cannot replace the host's actual admission."""
        if type(description) is not MCPInputOrigin:
            raise HostCapabilityAccessDenied('Original MCP input description is required')
        actual = self.origin()
        if ((description.operation_id, description.session_id, description.principal_id, description.allow_blocking)
                != (actual.operation_id, actual.session_id, actual.principal_id, actual.allow_blocking)):
            raise HostCapabilityAccessDenied('Original MCP input admission changed')
        actual.require_current()
        return actual

    async def audit(self, origin: MCPInputOrigin, checkpoint_digest: str, decisions):
        origin = self._revalidate_origin(origin)
        await self.store.audit_decisions(origin.operation_id, checkpoint_digest,
            tuple(decisions), now=self.authority.clock().timestamp())
        self._revalidate_origin(origin)

    async def audit_inline_decline(self, connection_id: str, request_id: str):
        """Legacy/unsolicited callbacks have no restartable original checkpoint."""
        # Long-lived native clients can call this outside any governed turn.
        # Read policy only after admission and retain it through native commit.
        async with require_durable_persistence_gate(self.agent).shared():
            self.require_current()
            await self.store.audit_decisions(connection_id, 'inline',
                ((request_id, 'no_restartable_state'),), now=self.authority.clock().timestamp())
            self.require_current()

    async def suspend(self, origin: MCPInputOrigin, *, connection_id: str,
                      contract_digest: str, method: str, payload: dict) -> str:
        origin = self._revalidate_origin(origin)
        if not origin.allow_blocking or origin.principal_id is None or origin.session_id is None:
            raise HostCapabilityAccessDenied('Unattended MCP work cannot create an operator wait')
        require_raw_capability_disclosure(self.agent, 'mcp-input')
        binding = MCPInputBinding(self._agent_id, origin.principal_id, origin.session_id,
            origin.operation_id, connection_id, contract_digest, method)
        now = self.authority.clock().timestamp()
        reader = self._reader.get()
        predecessor = reader.response_claim if reader is not None else None
        record = await self.store.create(binding, payload, now=now,
            expires_at=now + self.policy.retention_seconds, predecessor=predecessor)
        self._revalidate_origin(origin)
        from kestrel_sovereign.tools.effect_journal import record_tool_effect_input_required
        record_tool_effect_input_required(self.agent, record.request_id)
        return record.request_id

    def _require_reader(self):
        self.require_current()
        reader = self._reader.get()
        if reader is None or not reader.active or reader.check is None:
            raise HostCapabilityAccessDenied('Original MCP input reader is required')
        reader.check()
        return reader

    async def claim_response(self, record: MCPInputRecord, responses: dict) -> MCPInputClaim:
        origin = self.origin()
        reader = self._require_reader()
        if (not origin.allow_blocking or record.binding.principal_id != origin.principal_id
                or record.binding.session_id != reader.session_id):
            raise HostCapabilityAccessDenied('Original MCP input response owner changed')
        if reader.response_claim is not None:
            raise MCPInputConflict('Original MCP input response already owns a claim')
        claim = await self.store.claim_response(record, responses, now=self.authority.clock().timestamp())
        origin.require_current()
        reader.response_claim = claim
        return claim

    async def settle_response(self, claim: MCPInputClaim, *, state: str, continuation: dict):
        reader = self._require_reader()
        if (claim.record.binding.principal_id != reader.context.principal_id
                or claim.record.binding.session_id != reader.session_id):
            raise HostCapabilityAccessDenied('Original MCP input response owner changed')
        result = await self.store.settle(claim, state=state, continuation=continuation)
        reader.check()
        return result

    async def current_record(self, record: MCPInputRecord):
        reader = self._require_reader()
        result = await self.store.read_for(reader.context.principal_id, reader.session_id, record.request_id)
        reader.check()
        return result

    @asynccontextmanager
    async def borrow_app_view(self, view):
        reader = self._require_reader()
        if reader.app_view is not None:
            if reader.app_view is not view:
                raise ResourceConflict('MCP input cannot change its original App View')
            view.require_current()
            yield
            reader.check()
            return
        for host in getattr(self.agent, '_mcp_apps_hosts', ()):
            if any(physical.view is view for physical in host._views.values()):
                # Enter on the response producer and retain through its actual
                # outer send. An ended execution borrow cannot deliver _meta.
                reader.app_transport = await reader.borrowed.enter_async_context(
                    host.borrow_input_view(view, request=reader.request))
                reader.app_view = view
                yield
                reader.check()
                return
        raise ResourceConflict('Original MCP App View is unavailable')

    @asynccontextmanager
    async def _transport_operation(self, _context, _value, check):
        reader = self._require_reader()
        yield reader.app_transport
        check()

    @asynccontextmanager
    async def _resolve_context(self, request):
        self.require_current(request)
        session = request.query_params.get('session_id')
        if (len(request.query_params.getlist('session_id')) != 1 or not isinstance(session, str)
                or not session.strip() or len(session) > 256):
            raise ValueError('One original chat selector is required')
        service = getattr(request.app.state, 'request_authentication', None)
        if not isinstance(service, RequestAuthenticationService):
            raise HostCapabilityAccessDenied('Interactive request authentication is unavailable')
        with service.bind(request) as authentication, caller_context_scope(authentication.caller):
            now = self.authority.clock()
            deadline = now + timedelta(seconds=self.policy.operation_seconds)
            if authentication.expires_at is not None:
                deadline = min(deadline, authentication.expires_at)
            context = OperatorContext(tenant_id=self._agent_id, acting_agent_id=self._agent_id,
                principal_id=caller_identity_digest(authentication.caller),
                granted_actions=frozenset(action.value for action in _Action),
                granted_capabilities=frozenset({'mcp-input'}),
                permitted_boundary_ids=frozenset({self._agent_id}), correlation_id=uuid4().hex,
                issued_at=now, expires_at=deadline)
            reader = _Reader(context, session, authentication, request, AsyncExitStack())
            epoch = self.agent._get_privacy_policy_epoch()

            def current(original, _action, target):
                self.require_current(request)
                authentication.require_current()
                if (not reader.active or original is not context or target is not None
                        or self.agent._get_privacy_policy_epoch() != epoch):
                    raise HostCapabilityAccessDenied('Original MCP input request expired')
            with self.authority.bind_authenticated(context, privacy_holder=self.agent,
                    check_current=current, authority_gate=self._gate):
                token = self._reader.set(reader)
                try:
                    from kestrel_sovereign.agent.parts import part_collector
                    with part_collector(host=self.agent, session_id=session):
                        async with reader.borrowed:
                            yield context
                finally:
                    reader.active = False
                    self._reader.reset(token)

    @asynccontextmanager
    async def _operation(self, context, action):
        reader = self._reader.get()
        if action is _Action.RESPOND:
            # Dispatch takes this mutex before the persistence gate. Retain
            # its original span outside later App borrows, so they settle
            # before closing the span joins their native owners.
            await reader.borrowed.enter_async_context(self.agent._get_privacy_transition_lock())
        async with self.authority.operation(context, action, self._agent_id) as check:
            reader.check = check
            try:
                yield check
            finally:
                reader.check = None

    async def _body(self, request):
        if request.headers.get('content-type', '').split(';', 1)[0].strip() != 'application/json':
            raise ValueError('JSON response required')
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > self.policy.max_request_bytes:
                raise ValueError('MCP input response exceeded its byte limit')
            body.extend(chunk)
        return _Response.model_validate_json(bytes(body))

    @staticmethod
    def router_for(resolve_host):
        router = APIRouter(prefix='/api/mcp/inputs', tags=['mcp-input'])
        def host():
            value = resolve_host()
            if value is None:
                raise HTTPException(status_code=404, detail='mcp_input_unavailable')
            if not isinstance(value, MCPInputHost) or value._runtime is None:
                raise TypeError('Original attached MCP input host required')
            return value

        def response(owner, request, action, build, *, prepared=False):
            return AuthorizedCapabilityResponse(request, authority=owner.authority,
                resolve_context=owner._resolve_context,
                operation=lambda context, _value: owner._operation(context, action),
                build=build, error_response=_error, timeout_seconds=owner.policy.operation_seconds,
                prepare=(lambda: owner._body(request)) if prepared else None,
                transport_operation=owner._transport_operation)

        @router.get('')
        async def list_inputs(request: Request):
            owner = host()
            async def build(context):
                reader = owner._require_reader()
                rows = await owner.store.list_for(context.principal_id, reader.session_id,
                    now=owner.authority.clock().timestamp())
                values = [dict(request_id=row.request_id, revision=row.revision, state=row.state,
                    expires_at=row.expires_at, **owner._runtime.describe(row)) for row in rows]
                reader.check()
                return JSONResponse({'inputs': values})
            return response(owner, request, _Action.READ, build)

        @router.post('/{request_id}/responses')
        async def respond(request_id: str, request: Request):
            owner = host()
            async def build(context, value):
                reader = owner._require_reader()
                row = await owner.store.read_for(context.principal_id, reader.session_id, request_id)
                if (row.state != 'WAITING' or row.revision != value.revision
                        or row.expires_at <= owner.authority.clock().timestamp()):
                    raise MCPInputConflict('MCP input waiting revision is unavailable')
                reader.check()
                result = await owner._runtime.respond(row, value.responses, operation_id=value.operation_id)
                reader.check()
                return JSONResponse(result)
            return response(owner, request, _Action.RESPOND, build, prepared=True)

        @router.get('/{request_id}')
        async def read_input(request_id: str, request: Request):
            owner = host()
            async def build(context):
                reader = owner._require_reader()
                row = await owner.store.read_for(context.principal_id, reader.session_id, request_id)
                value = dict(request_id=row.request_id, revision=row.revision,
                    state=('EXPIRED' if row.state == 'WAITING' and row.expires_at <= owner.authority.clock().timestamp()
                        else row.state), expires_at=row.expires_at, **owner._runtime.describe(row))
                reader.check()
                return JSONResponse(value)
            return response(owner, request, _Action.READ, build)
        return router

    def withdraw(self):
        self._accepting = False

    async def close(self):
        self.withdraw()
        self.retired = True


class MCPInputHostMixin:
    supports_mcp_input_host = True

    def validate_mcp_input_host_configuration(self, configuration):
        return MCPInputHostPolicy.model_validate(configuration).model_dump(mode='json')

    def create_mcp_input_host(self, feature, *, configuration, is_current):
        if any(not host.retired for host in getattr(self, '_mcp_input_hosts', ())):
            raise ResourceConflict('Close the original MCP input host before replacement')
        host = MCPInputHost(self, feature, configuration=configuration, is_current=is_current)
        self._mcp_input_hosts = [host]
        return host

    def create_mcp_input_router(self, feature, *, resolve_host):
        def original():
            host = resolve_host()
            if host is not None and (not isinstance(host, MCPInputHost)
                    or host.agent is not self or host.feature is not feature):
                raise HostCapabilityAccessDenied('Original MCP input host was replaced')
            return host
        return MCPInputHost.router_for(original)
