"""Host-authorized artifact operations, independent of any computer session.

The host resolves authentication and live grants, then binds that exact SDK
context for the request/turn lifetime. Deserialized contexts are never bearer
credentials. A binding's synchronous policy checker must inspect CURRENT
caller/grants, including the target of link/unlink, and raise on refusal.
It must not cache an admission decision. Privacy uses the existing host gate;
the host's grant/key mutations must take the supplied authority gate exclusive.
Lock order is privacy gate, authority gate, database. A host that cannot fence
its grant mutations this way must not enable this service. Single host process
ownership is required; these in-process gates do not coordinate other hosts.

This module does not register HTTP routes, mint identities, or initialize a
store. Host lifecycle wiring must use the same database as key rotation and
quiesce this service for maintenance. Python extensions are trusted host code;
this is not a Python sandbox. No service entry point accepts an owner selector.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from enum import StrEnum

from kestrel_sdk.artifacts import (
    ArtifactCommit,
    ArtifactFile,
    ArtifactLink,
    ArtifactPage,
    ArtifactRef,
    ArtifactRevision,
    PublicationReceipt,
)
from kestrel_sdk.operator import OperatorAuthorizationError, OperatorContext
from kestrel_sdk.resources import ResourceError, ResourceOutcomeUnknown, ResourceScope

from kestrel_sovereign.security.capability_authority import HostCapabilityAuthority

from .repository import ArtifactRepository


class ArtifactAccessDenied(ResourceError, OperatorAuthorizationError):
    """No authorized result is disclosed; the message contains no object data."""


class ArtifactAction(StrEnum):
    READ = "artifacts.read"
    WRITE = "artifacts.write"
    LINK = "artifacts.link"
    DELETE = "artifacts.delete"


CurrentAuthority = Callable[[OperatorContext, ArtifactAction, ArtifactLink | None], None]


class ArtifactAuthority(HostCapabilityAuthority):
    """Artifact policy over the canonical original-context host binding."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None):
        super().__init__(capability="artifacts", label="artifact",
                         non_disclosing_actions=frozenset({ArtifactAction.DELETE.value}),
                         access_denied=ArtifactAccessDenied, clock=clock)

    @asynccontextmanager
    async def operation(self, context: OperatorContext, action: ArtifactAction,
                        storage_id: str, *, link: ArtifactLink | None = None):
        async with super().operation(context, action, storage_id, target=link) as check:
            yield check


class AuthorizedArtifactService:
    """Implement SDK ArtifactService with current, owner-exact host authority."""

    def __init__(self, repository: ArtifactRepository, authority: ArtifactAuthority):
        self.repository = repository
        self.authority = authority

    @staticmethod
    def _scope(context: OperatorContext) -> ResourceScope:
        return ResourceScope(tenant_id=context.tenant_id, principal_id=context.principal_id,
                             agent_id=context.acting_agent_id)

    async def _perform(self, context, action, operation, *, mutation=False, link=None):
        async with self.authority.operation(context, action, self.repository.storage_id,
                                            link=link) as check:
            await self.repository.ensure_initialized()
            check()
            result = await operation(self._scope(context), check)
            try:
                # Native COMMIT can await after the repository's final check.
                # Never disclose bytes or a receipt under an expired authority.
                check()
            except Exception:
                if mutation:
                    raise ResourceOutcomeUnknown(
                        "artifact operation completed without delivery authority; reconcile with current authority"
                    ) from None
                raise ArtifactAccessDenied("artifact result authority is unavailable") from None
            return result

    async def stage(self, context: OperatorContext, content: bytes, *, publication_key: str,
                    path: str, media_type: str) -> ArtifactFile:
        return await self._perform(context, ArtifactAction.WRITE,
            lambda scope, check: self.repository.stage(scope, content, publication_key=publication_key,
                path=path, media_type=media_type, guard=check), mutation=True)

    async def commit(self, context: OperatorContext, request: ArtifactCommit) -> PublicationReceipt:
        return await self._perform(context, ArtifactAction.WRITE,
            lambda scope, check: self.repository.commit(scope, request, guard=check), mutation=True)

    async def publication_status(self, context: OperatorContext, idempotency_key: str) -> PublicationReceipt | None:
        return await self._perform(context, ArtifactAction.WRITE,
            lambda scope, check: self.repository.publication_status(scope, idempotency_key, guard=check))

    async def get(self, context: OperatorContext, ref: ArtifactRef) -> ArtifactRevision:
        return await self._perform(context, ArtifactAction.READ,
            lambda scope, check: self.repository.get(scope, ref, guard=check))

    async def head(self, context: OperatorContext, artifact_id: str) -> ArtifactRevision:
        return await self._perform(context, ArtifactAction.READ,
            lambda scope, check: self.repository.head(scope, artifact_id, guard=check))

    async def read(self, context: OperatorContext, ref: ArtifactRef, path: str) -> bytes:
        return await self._perform(context, ArtifactAction.READ,
            lambda scope, check: self.repository.read(scope, ref, path, guard=check))

    async def list(self, context: OperatorContext, *, cursor: str | None = None,
                   limit: int = 50, query: str = "") -> ArtifactPage:
        return await self._perform(context, ArtifactAction.READ,
            lambda scope, check: self.repository.page(scope, cursor=cursor, limit=limit, query=query, guard=check))

    async def history(self, context: OperatorContext, artifact_id: str, *,
                      cursor: str | None = None, limit: int = 50) -> ArtifactPage:
        return await self._perform(context, ArtifactAction.READ,
            lambda scope, check: self.repository.page(scope, artifact_id=artifact_id,
                cursor=cursor, limit=limit, guard=check))

    async def link(self, context: OperatorContext, link: ArtifactLink) -> None:
        link = ArtifactLink.model_validate(link)
        await self._perform(context, ArtifactAction.LINK,
            lambda scope, check: self.repository.link(scope, link, guard=check), mutation=True, link=link)

    async def unlink(self, context: OperatorContext, link: ArtifactLink) -> None:
        link = ArtifactLink.model_validate(link)
        await self._perform(context, ArtifactAction.LINK,
            lambda scope, check: self.repository.link(scope, link, remove=True, guard=check),
            mutation=True, link=link)

    async def delete(self, context: OperatorContext, expected_head: ArtifactRef) -> None:
        await self._perform(context, ArtifactAction.DELETE,
            lambda scope, check: self.repository.delete(scope, expected_head, guard=check), mutation=True)
