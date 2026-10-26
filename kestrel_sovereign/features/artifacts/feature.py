"""Bind the artifact service to an original agent and current host authority.

The first host policy admits the configured sovereign API key only. OAuth,
JWT, peer, unattended and delegated access require a future explicit grant
resolver; authentication or an agent address alone grants no artifact access.
Context links are not granted until their destination has an authority resolver.
Encryption-key maintenance remains offline: stop the host before rotation.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections.abc import Callable
from datetime import timedelta
from uuid import uuid4

from kestrel_sdk.operator import OperatorAuthorizationError, OperatorContext
from kestrel_sdk.security.encryption import get_fernet

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.artifacts.http import create_artifact_router, _delivery_error
from kestrel_sovereign.artifacts.preview import DockerArtifactRenderer, host_preview_pool
from kestrel_sovereign.artifacts.repository import ArtifactIntegrityError, ArtifactRepository
from kestrel_sovereign.artifacts.service import (
    ArtifactAccessDenied,
    ArtifactAction,
    ArtifactAuthority,
    AuthorizedArtifactService,
)
from kestrel_sovereign.auth import (
    AuthMethod, CallerContext, caller_context_scope, current_caller_context,
)
from kestrel_sovereign.endpoints.agent_helpers import get_caller
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.storage_access import (
    resolve_feature_database,
    resolve_scoped_agent_did,
)
from kestrel_sovereign.security.host_authority import (
    HostAuthorityError, require_sovereign_caller, stable_sovereign_secret,
)
from kestrel_sovereign.security.private_delivery import AuthorizedCapabilityResponse
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate
from kestrel_sovereign.turn_scope import capture_turn_scope
from .producers import ArtifactProducerTools
from .workspace import ArtifactWorkspaceTools, create_workspace_router


@dataclass(frozen=True)
class ArtifactOperation:
    """One retained original operation, never a portable grant or credential.

    ``check_current`` is synchronous and usable only while the original
    operation frame remains active, including original owned database work.
    Consumers retain their independent policy and combine its restriction.
    """

    service: AuthorizedArtifactService
    context: OperatorContext
    check_current: Callable[[], None]

    @property
    def expires_at(self):
        return self.context.expires_at


class ArtifactsFeature(ArtifactProducerTools, ArtifactWorkspaceTools, Feature):
    """Encrypted revisions served through the host's live feature router."""

    supports_subagent_dispatch = False
    promote_tools_on_startup = True
    artifact_bridge_api_version = 3

    def __init__(self, agent):
        super().__init__(agent)
        # Optional bundled features are contained during discovery, whereas an
        # initialize failure aborts agent boot. Refuse construction before any
        # DB work so supported keyless hosts retain their other capabilities.
        if get_fernet() is None:
            reason = "Artifacts require KESTREL_DATA_KEY; configure it and reload the feature"
            recorder = getattr(agent, "record_feature_unavailable", None)
            if callable(recorder):
                recorder(feature=self, feature_name=self.name, reason=reason)
            raise ArtifactIntegrityError(reason)
        self._accepting = False
        self._generation = 0
        self._gate = AsyncReaderWriterLock()
        self._service: AuthorizedArtifactService | None = None
        self._router = None
        self._agent_id: str | None = None
        self._sweep_task = None

    _sweep_seconds = 60

    @property
    def tool_description(self) -> str:
        return "Publish, discover and reopen encrypted durable artifacts with exact revision references"

    async def initialize(self):
        if self._service is not None:
            if (resolve_scoped_agent_did(self.agent) != self._agent_id
                    or resolve_feature_database(self.agent) is not self._service.repository.db):
                raise RuntimeError("Artifact runtime owner changed")
            return  # The canonical host re-enables the SAME feature instance.
        # Resolve once from the original runtime. Requests cannot choose the
        # tenant, principal, database, storage boundary, or acting identity.
        agent_id = resolve_scoped_agent_did(self.agent)
        require_durable_persistence_gate(self.agent)
        db = resolve_feature_database(self.agent)
        if db is None:
            raise RuntimeError("Artifacts require the agent's durable database")
        repository = ArtifactRepository(db, storage_id="agent-artifacts")
        self._agent_id = agent_id
        self._service = AuthorizedArtifactService(repository, ArtifactAuthority())
        self._service.preview_renderer = DockerArtifactRenderer(os.environ.get("KESTREL_ARTIFACT_RENDERER_IMAGE"))
        self._router = create_artifact_router(
            self._service, resolve_context=self._resolve_context,
        )
        self._router.include_router(create_workspace_router(self))

    def get_router(self):
        return self._router

    @asynccontextmanager
    async def artifact_operation(self, request=None):
        """Original-caller service bridge; no run/delegated grant.

        Yield an ``ArtifactOperation`` while retaining the original
        privacy and current feature/caller READ lease. Service operations still
        check their own actions. A context cannot be reused after this frame.
        HTTP producers must retain this frame through actual private delivery,
        or use artifact_response. Independent policies retain their own lease.
        """
        resolver = self._resolve_context(request) if request is not None else self._resolve_tool_context()
        async with resolver as context:
            service = self._service
            async with service.authority.operation(context, ArtifactAction.READ, service.repository.storage_id) as check:
                active = True
                def current():
                    if not active:
                        raise ArtifactAccessDenied('artifact operation has ended')
                    check()
                try:
                    current()
                    yield ArtifactOperation(service, context, current)
                    current()
                finally:
                    active = False

    def artifact_response(self, request, build, *, check_current=None):
        """Retain original artifact authority and restrictions through delivery.

        ``await build(operation)`` receives the typed retained operation and
        returns a Response. A
        workflow wrapper must separately retain its original run resolver
        through the Response's ASGI call. Its synchronous ``check_current``
        restricts canonical service work and the outer buffered ASGI delivery;
        it grants nothing. A run scope/lease never substitutes for artifact
        authority. A resolver closed inside build cannot be resurrected here.
        """
        if not callable(build):
            raise TypeError('artifact response builder required')
        if check_current is not None and not callable(check_current):
            raise TypeError('artifact response restriction must be callable')
        service = self._service
        if service is None:
            raise ArtifactAccessDenied('artifact runtime is unavailable')
        @asynccontextmanager
        async def resolve(current_request):
            async with self.artifact_operation(current_request) as original:
                if original.service is not service:
                    raise ArtifactAccessDenied('artifact runtime changed')
                def combined_check():
                    original.check_current()
                    if check_current is not None:
                        restriction_check()
                yield ArtifactOperation(service, original.context, combined_check)
        async def body(admission):
            return await build(admission)
        def restriction_check():
            try:
                check_current()
            except OperatorAuthorizationError:
                # Keep a consumer's denial recognizable through canonical
                # database rollback, without disclosing its policy details.
                raise ArtifactAccessDenied('additional artifact policy is unavailable') from None
        @asynccontextmanager
        async def operation(admission, value):
            async with service.restrict_current(admission.context, admission.check_current):
                admission.check_current()
                yield admission.check_current
                admission.check_current()
        return AuthorizedCapabilityResponse(request, authority=service.authority,
            resolve_context=resolve,
            operation=operation,
            build=body, access_denied=ArtifactAccessDenied,
            error_response=lambda error,prepared:_delivery_error(error, prepared, request))

    async def on_enable(self):
        if self._service is None:
            raise RuntimeError("Artifacts have not initialized")
        async with self._gate:
            self._generation += 1
            self._accepting = True
        if self._sweep_task is None or self._sweep_task.done():
            self._sweep_task = self._track_owned_background_task(
                self._sweep_staging(), name="artifact-staging-expiry", _resident_owner=True,
            )

    async def _sweep_staging(self):
        while self._accepting:
            try:
                async with self._gate.read():
                    if not self._accepting:
                        return
                    # Host maintenance only: erase expired unreferenced bytes,
                    # including owners who never return. Erasure is allowed in
                    # volatile privacy modes; nothing is read or disclosed.
                    repository = self._service.repository
                    # Do not open an unused store or unwrap a key merely to
                    # boot a host. Existing staging can be erased without keys.
                    if await repository.db.table_exists("artifact_staging"):
                        await repository.collect_expired_staging(owner_limit=100)
            except Exception as error:
                if is_execution_control_error(error):
                    raise
                logging.getLogger(__name__).warning("Artifact staging expiry failed; retrying next sweep")
            await asyncio.sleep(self._sweep_seconds)

    async def _withdraw(self):
        # Revoke before waiting, including uploads not holding a read lease.
        # Re-enable cannot resurrect a context admitted in this generation.
        self._accepting = False
        self._generation += 1
        async with self._gate:
            pass  # Join every operation/response before lifecycle release.
        await self._cancel_owned_background_tasks()

    async def on_disable(self):
        await self._withdraw()

    async def shutdown(self):
        await self._withdraw()
        # The agent owns the DB. Do not close it, delete saved artifacts, or
        # clear retained native-operation custody during feature withdrawal.
        return await super().shutdown()

    def _require_live(self, request, generation: int) -> None:
        if (
            not self._accepting or generation != self._generation
            or getattr(self, "enabled", True) is not True
            or self.agent.features.get(self.name) is not self
            or resolve_scoped_agent_did(self.agent) != self._agent_id
            or resolve_feature_database(self.agent) is not self._service.repository.db
        ):
            raise ArtifactAccessDenied("artifact runtime is no longer current")
        # An explicit route must never fall back to this mount's first agent.
        if request is None:
            return  # Tool admission is bound to this original live feature.
        routed = getattr(request.state, "agent", None)
        if routed is not None and routed is not self.agent:
            raise ArtifactAccessDenied("artifact runtime does not match route")
        manager = getattr(request.app.state, "agent_manager", None)
        if manager is not None:
            if not any(agent is self.agent for agent in manager.list_agents().values()):
                raise ArtifactAccessDenied("artifact runtime was withdrawn")
        elif getattr(request.app.state, "agent", None) is not self.agent:
            raise ArtifactAccessDenied("artifact runtime was replaced")

    @asynccontextmanager
    async def _resolve_context(self, request):
        caller = get_caller(request)
        try:
            secret = stable_sovereign_secret("artifact access")
            bearer = request.headers.get("authorization", "")
            presented = (request.headers.get("x-api-key", ""),
                         bearer[7:] if bearer.startswith("Bearer ") else "")
            if not any(hmac.compare_digest(value.encode("utf-8"), secret) for value in presented):
                raise HostAuthorityError("explicit artifact credential required")
        except HostAuthorityError:
            raise ArtifactAccessDenied("artifact authority is unavailable") from None
        async with self._bind_caller(caller, request=request) as context:
            self._service.preview_pool = host_preview_pool(request.app)
            yield context

    @asynccontextmanager
    async def _resolve_tool_context(self):
        # A tool cannot select a principal, agent, credential or delegated grant.
        # The endpoint-bound original caller has the same revocable lifetime as
        # HTTP admission, even when execution moves to a registered tool task.
        async with self._bind_caller(current_caller_context()) as context:
            yield context

    @asynccontextmanager
    async def _bind_caller(self, caller, *, request=None):
        if (not isinstance(caller, CallerContext)
                or caller.auth_method is not AuthMethod.API_KEY
                or caller.is_sovereign is not True):
            raise ArtifactAccessDenied("artifact access requires sovereign authority")
        generation = self._generation
        self._require_live(request, generation)
        scope = capture_turn_scope(self.agent)
        if request is None and current_caller_context() is not caller:
            raise ArtifactAccessDenied("artifact tool caller is no longer current")
        # HTTP owns this response lifetime. Tools MUST reuse the endpoint's
        # existing binding; minting a scope here would outlive its revocation.
        with (scope.bind() if request is None else caller_context_scope(caller)):
            try:
                require_sovereign_caller("artifact access")
            except HostAuthorityError:
                raise ArtifactAccessDenied("artifact authority is unavailable") from None
            request_scope = capture_turn_scope(self.agent)
            authority = self._service.authority
            now = authority.clock()
            context = OperatorContext(
                tenant_id=self._agent_id, acting_agent_id=self._agent_id,
                principal_id="host-sovereign",
                granted_actions=frozenset({
                    ArtifactAction.READ.value, ArtifactAction.WRITE.value,
                    ArtifactAction.DELETE.value,
                    ArtifactAction.TRANSFER.value,
                }),
                granted_capabilities=frozenset({"artifacts"}),
                permitted_boundary_ids=frozenset({self._service.repository.storage_id}),
                correlation_id=uuid4().hex, issued_at=now,
                expires_at=now + timedelta(seconds=30),
            )

            def check_current(context, action, link):
                self._require_live(request, generation)
                if (request is not None and get_caller(request) is not caller) or link is not None:
                    raise ArtifactAccessDenied("artifact authority changed")
                with request_scope.bind():
                    require_sovereign_caller("artifact access")

            with authority.bind_authenticated(
                context, privacy_holder=self.agent,
                check_current=check_current, authority_gate=self._gate,
            ):
                yield context
