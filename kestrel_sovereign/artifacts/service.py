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

import asyncio
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
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

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.privacy import PrivacyConfig, PrivacyMode
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate

from .repository import ArtifactRepository


class ArtifactAccessDenied(ResourceError, OperatorAuthorizationError):
    """No authorized result is disclosed; the message contains no object data."""


class ArtifactAction(StrEnum):
    READ = "artifacts.read"
    WRITE = "artifacts.write"
    LINK = "artifacts.link"
    DELETE = "artifacts.delete"


CurrentAuthority = Callable[[OperatorContext, ArtifactAction, ArtifactLink | None], None]


@dataclass
class _Binding:
    context: OperatorContext
    privacy_holder: object
    check_current: CurrentAuthority
    authority_gate: AsyncReaderWriterLock
    active: bool = True


class ArtifactAuthority:
    """Request-lifetime provenance for contexts resolved by trusted host code.

Use one instance per host service. Bindings are removed on request exit and
invalidate copied task contexts immediately. No external registration API may
expose ``bind_authenticated``. The checker is required even for a sovereign
caller: role alone does not prove a still-current credential or live grant.
"""

    def __init__(self, *, clock: Callable[[], datetime] | None = None):
        self.clock = clock or (lambda: datetime.now(UTC))
        self._bindings: dict[int, _Binding] = {}
        self._operations: dict[asyncio.Task, _Binding] = {}

    @contextmanager
    def bind_authenticated(
        self, context: OperatorContext, *, privacy_holder: object,
        check_current: CurrentAuthority, authority_gate: AsyncReaderWriterLock,
    ) -> Iterator[OperatorContext]:
        if not isinstance(context, OperatorContext) or not callable(check_current):
            raise TypeError("host-resolved context and current authority checker required")
        if not isinstance(authority_gate, AsyncReaderWriterLock):
            raise TypeError("host authority transition gate required")
        context.require_fresh(self.clock())
        require_durable_persistence_gate(privacy_holder)
        if id(context) in self._bindings:
            raise ValueError("artifact context already has an active lifetime")
        binding = _Binding(context, privacy_holder, check_current, authority_gate)
        self._bindings[id(context)] = binding
        try:
            yield context
        finally:
            binding.active = False
            self._bindings.pop(id(context), None)

    def _check(self, binding: _Binding, context: OperatorContext,
               action: ArtifactAction, storage_id: str,
               link: ArtifactLink | None) -> None:
        if (not binding.active or binding.context is not context
                or self._bindings.get(id(context)) is not binding):
            raise ArtifactAccessDenied("artifact context is not active")
        try:
            context.require_fresh(self.clock())
            context.require_action(action.value)
            context.require_capability("artifacts")
            context.require_boundary(storage_id)
            if binding.check_current(context, action, link) is not None:
                raise TypeError("artifact authority checker must return None or raise")
        except Exception:
            # A failed current-grant lookup is a denial, including plain
            # PermissionError or infrastructure failures. Never turn a checker
            # failure after durable acceptance into an ordinary save failure.
            raise ArtifactAccessDenied("artifact authority is unavailable") from None
        # Generic binary/source artifacts have no PII/deidentification transform.
        # Refuse those modes instead of treating persistent storage as consent
        # to retain or reveal arbitrary originals. Explicit erasure stays usable.
        if action is not ArtifactAction.DELETE:
            holder = binding.privacy_holder
            config = getattr(holder, "privacy_config", None)
            if (getattr(holder, "privacy_mode", None) not in (PrivacyMode.NORMAL, PrivacyMode.PUBLIC)
                    or not isinstance(config, PrivacyConfig)
                    or config.storage != "full" or config.assurance != "none"
                    or config.audit != "optional"):
                raise ArtifactAccessDenied("privacy policy does not permit raw artifacts")

    @asynccontextmanager
    async def operation(self, context: OperatorContext, action: ArtifactAction,
                        storage_id: str, *, link: ArtifactLink | None = None):
        binding = self._bindings.get(id(context))
        if binding is None or binding.context is not context:
            raise ArtifactAccessDenied("artifact context was not resolved by this host")

        def check() -> None:
            self._check(binding, context, action, storage_id, link)

        check()
        task = asyncio.current_task()
        existing = self._operations.get(task)
        if existing is not None:
            # A response retains the gates across service calls and ASGI send.
            # Re-enter only on the SAME task and exact binding; copied task
            # context does not grant a lease. Still check each nested action.
            if existing is not binding:
                raise ArtifactAccessDenied("nested artifact owner does not match")
            yield check
            return
        gate = require_durable_persistence_gate(binding.privacy_holder)
        async with gate.shared():
            async with binding.authority_gate.read():
                check()
                self._operations[task] = binding
                try:
                    yield check
                finally:
                    self._operations.pop(task, None)


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
                   limit: int = 50) -> ArtifactPage:
        return await self._perform(context, ArtifactAction.READ,
            lambda scope, check: self.repository.page(scope, cursor=cursor, limit=limit, guard=check))

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
