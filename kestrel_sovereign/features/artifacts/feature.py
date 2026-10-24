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
from contextlib import asynccontextmanager
from datetime import timedelta
from uuid import uuid4

from kestrel_sdk.operator import OperatorContext
from kestrel_sdk.security.encryption import get_fernet

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.artifacts.http import create_artifact_router
from kestrel_sovereign.artifacts.repository import ArtifactIntegrityError, ArtifactRepository
from kestrel_sovereign.artifacts.service import (
    ArtifactAccessDenied,
    ArtifactAction,
    ArtifactAuthority,
    AuthorizedArtifactService,
)
from kestrel_sovereign.auth import (
    AuthMethod, CallerContext, caller_context_scope,
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
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate
from kestrel_sovereign.turn_scope import capture_turn_scope


class ArtifactsFeature(Feature):
    """Encrypted revisions served through the host's live feature router."""

    supports_subagent_dispatch = False

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
        return "HTTP service for encrypted artifact revisions; no model-facing producer tools"

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
        self._router = create_artifact_router(
            self._service, resolve_context=self._resolve_context,
        )

    def get_router(self):
        return self._router

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
        if (
            not isinstance(caller, CallerContext)
            or caller.auth_method is not AuthMethod.API_KEY
            or caller.is_sovereign is not True
        ):
            raise ArtifactAccessDenied("artifact access requires sovereign authority")
        generation = self._generation
        self._require_live(request, generation)
        # Only header/bearer API credentials are admitted, so browsers cannot
        # attach the credential implicitly. Cookie/JWT callers are denied even
        # with a CSRF token; a token must not create a missing artifact grant.
        with caller_context_scope(caller):
            try:
                require_sovereign_caller("artifact access")
                secret = stable_sovereign_secret("artifact access")
                bearer = request.headers.get("authorization", "")
                presented = (request.headers.get("x-api-key", ""),
                             bearer[7:] if bearer.startswith("Bearer ") else "")
                if not any(hmac.compare_digest(value.encode("utf-8"), secret) for value in presented):
                    raise HostAuthorityError("explicit artifact credential required")
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
                }),
                granted_capabilities=frozenset({"artifacts"}),
                permitted_boundary_ids=frozenset({self._service.repository.storage_id}),
                correlation_id=uuid4().hex, issued_at=now,
                expires_at=now + timedelta(seconds=30),
            )

            def check_current(context, action, link):
                self._require_live(request, generation)
                if get_caller(request) is not caller or link is not None:
                    raise ArtifactAccessDenied("artifact authority changed")
                # The outer transport task predates this request's context.
                # Carry every declared context through the canonical registry,
                # including the SAME revocable caller lifetime, never a new one.
                with request_scope.bind():
                    require_sovereign_caller("artifact access")

            with authority.bind_authenticated(
                context, privacy_holder=self.agent,
                check_current=check_current, authority_gate=self._gate,
            ):
                yield context
