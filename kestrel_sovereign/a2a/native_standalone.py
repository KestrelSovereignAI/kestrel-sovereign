"""Original single-agent runtime authority over the shared native publication."""

import asyncio
import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from starlette.exceptions import HTTPException

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.a2a.inbound_authorization import mark_a2a_inbound_scoped_policy
from kestrel_sovereign.a2a.native_card import REQUIRED_NATIVE_EXTENSIONS
from kestrel_sovereign.a2a.native_security import (
    NativeDIDRequestGuard,
    NativeRequestAuthority,
)
from kestrel_sovereign.a2a.native_sender import (
    NativeProofPolicy,
    NativeSenderProofError,
    NativeVerifiedSender,
    verification_document_fingerprint,
)
from kestrel_sovereign.a2a.replay_store import SharedReplayNonceStore
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    is_execution_control_error,
    require_execution_work,
)
from kestrel_sovereign.features.storage_access import resolve_scoped_agent_did
from kestrel_sovereign.multi_agent.route_name import encode_agent_route_name

if TYPE_CHECKING:
    from kestrel_sovereign.a2a.native_publication import NativeA2APublication
    from kestrel_sovereign.kestrel_agent import KestrelAgent


class _SingleAgentNativeRequest:
    """Keep one resolved sender document in its original request lifetime."""

    def __init__(self, owner: "SingleAgentNativeAuthority"):
        self.owner = owner
        self.sender_id = None
        self.fingerprint = None

    def require_current(self, sender: NativeVerifiedSender | None):
        self.owner.require_current()
        if sender is not None and (
            sender.sender != self.sender_id
            or sender.verification_document_fingerprint != self.fingerprint
        ):
            raise HTTPException(
                status_code=403, detail="Native sender document changed"
            )

    async def resolve(self, did):
        self.require_current(None)
        if self.sender_id is not None:
            raise RuntimeError("A native request cannot replace its sender witness")
        self.sender_id = did
        document = self.owner.resolver(did)
        if inspect.isawaitable(document):
            document = await document
        self.require_current(None)
        if isinstance(document, Mapping):
            try:
                self.fingerprint = verification_document_fingerprint(document)
            except (TypeError, ValueError) as error:
                if is_execution_control_error(error):
                    raise
                raise NativeSenderProofError(
                    "Native sender document cannot be fingerprinted"
                ) from error
        return document

    async def authorize(self, did):
        return await self.owner.authorizer.authorize_principal_with_policy(
            did, router=self.owner.router, requester=self.owner.requester
        )


class SingleAgentNativeAuthority:
    """Bind request authority to one original runtime and explicit peer policy.

    This owns no wire implementation. Its guard feeds NativeA2APublication,
    which owns the same card, SDK routes and handler as the AgentManager host.
    The caller supplies a trusted DID resolver and scoped recipient policy;
    a verified DID alone never supplies authorization. Close stops new
    admissions and joins existing readers before withdrawing the exact routes.
    """

    def __init__(
        self,
        recipient: "KestrelAgent",
        routing_key: str,
        *,
        proof_policy: NativeProofPolicy,
        clock: Callable[[], float],
    ):
        if not isinstance(proof_policy, NativeProofPolicy) or not callable(clock):
            raise TypeError("Native runtime requires its proof policy and clock")
        if getattr(recipient, "_native_a2a_runtime_authority", None) is not None:
            raise RuntimeError(
                "Native runtime authority requires a fresh agent generation"
            )
        encode_agent_route_name(routing_key)
        manager = getattr(recipient, "task_manager", None)
        ready = getattr(recipient, "_runtime_publication_ready", None)
        if not isinstance(manager, TaskManager):
            raise TypeError("Native runtime requires its canonical task manager")
        if manager.task_store.scope.recipient_id != resolve_scoped_agent_did(recipient):
            raise ValueError(
                "Native task manager belongs to another recipient identity"
            )
        if not isinstance(ready, asyncio.Event) or not ready.is_set():
            raise ValueError("Native runtime publication requires READY")
        identity = getattr(recipient, "identity", None)
        if getattr(identity, "is_hybrid", False) is not True:
            raise ValueError("Native runtime publication requires a hybrid identity")
        resolver = getattr(recipient, "a2a_did_resolver", None)
        authorizer = getattr(recipient, "a2a_inbound_sender_authorizer", None)
        router = getattr(recipient, "peer_directory_router", None)
        requester = getattr(recipient, "peer_requester", None)
        mark_a2a_inbound_scoped_policy(recipient, required=True)
        if (
            not callable(resolver)
            or not callable(getattr(authorizer, "has_valid_policy_scope", None))
            or not callable(
                getattr(authorizer, "authorize_principal_with_policy", None)
            )
            or authorizer.has_valid_policy_scope(router, requester) is not True
        ):
            raise ValueError(
                "Native runtime requires an explicit scoped peer policy and DID resolver"
            )
        self.recipient = recipient
        self.routing_key = routing_key
        self.manager = manager
        self.task_store = manager.task_store
        self.scope = self.task_store.scope
        self.identity = identity
        self.identity_fingerprint = verification_document_fingerprint(self._document())
        runtime = getattr(recipient, "_execution_custody", None)
        if runtime is not None and not isinstance(runtime, ExecutionCustody):
            raise TypeError("Native runtime requires its original execution custody")
        require_execution_work(recipient)
        self.runtime = runtime
        self.ready = ready
        self.resolver, self.authorizer = resolver, authorizer
        self.router, self.requester = router, requester
        self.proof_policy, self.clock = proof_policy, clock
        self._lifecycle = AsyncReaderWriterLock()
        self._publication: NativeA2APublication | None = None
        self._closing = False
        self._retired = False
        recipient._native_a2a_runtime_authority = self

    def _document(self):
        return {
            "id": self.identity.signing_did,
            "verificationMethod": self.identity.new_verification_methods,
        }

    def _require_runtime(self):
        if self.runtime is not None:
            self.runtime.require_work()
        require_execution_work(self.recipient)
        if (
            self._closing
            or getattr(self.recipient, "_native_a2a_runtime_authority", None)
            is not self
            or getattr(self.recipient, "task_manager", None) is not self.manager
            or self.manager.task_store is not self.task_store
            or self.task_store.scope != self.scope
            or getattr(self.recipient, "identity", None) is not self.identity
            or verification_document_fingerprint(self._document())
            != self.identity_fingerprint
            or getattr(self.recipient, "_execution_custody", None) is not self.runtime
            or getattr(self.recipient, "_runtime_publication_ready", None)
            is not self.ready
            or not self.ready.is_set()
            or getattr(self.recipient, "a2a_did_resolver", None) is not self.resolver
            or getattr(self.recipient, "a2a_inbound_sender_authorizer", None)
            is not self.authorizer
            or getattr(self.recipient, "peer_directory_router", None) is not self.router
            or getattr(self.recipient, "peer_requester", None) is not self.requester
            or self.authorizer.has_valid_policy_scope(self.router, self.requester)
            is not True
            or (
                self._publication is not None
                and getattr(self.recipient, "_native_a2a_publication", None)
                is not self._publication
            )
        ):
            raise HTTPException(
                status_code=403, detail="Native recipient runtime changed"
            )

    def require_current(self):
        self._require_runtime()
        if self._publication is None or not self._publication.is_published:
            raise HTTPException(
                status_code=403, detail="Native runtime is not published"
            )

    def create_guard(self) -> NativeDIDRequestGuard:
        """Capture this runtime without inventing an AgentManager registration."""
        self._require_runtime()

        def authority():
            request = _SingleAgentNativeRequest(self)
            return NativeRequestAuthority(
                request.resolve, request.authorize, request.require_current
            )

        return NativeDIDRequestGuard(
            routing_key=self.routing_key,
            scope=self.scope,
            required_extensions=REQUIRED_NATIVE_EXTENSIONS,
            proof_policy=self.proof_policy,
            authority_factory=authority,
            replay_store=SharedReplayNonceStore(self.task_store.backend),
            lifecycle_lease=self._lifecycle.read,
            clock=self.clock,
        )

    async def publish(self, publication: "NativeA2APublication"):
        from kestrel_sovereign.a2a.native_publication import NativeA2APublication

        if not isinstance(publication, NativeA2APublication):
            raise TypeError("Native runtime requires the shared publication")
        if (
            publication.recipient is not self.recipient
            or publication.routing_key != self.routing_key
        ):
            raise ValueError("Native publication belongs to another runtime")
        async with self._lifecycle:
            self._require_runtime()
            if self._publication is not None and self._publication is not publication:
                raise RuntimeError("Native runtime cannot replace its publication")
            installed = getattr(self.recipient, "_native_a2a_publication", None)
            if installed is not None and installed is not publication:
                raise RuntimeError(
                    "Native runtime cannot replace an installed publication"
                )
            self._publication = publication
            self.recipient._native_a2a_publication = publication
            publication.publish()

    async def close(self):
        """Retire the original route owner only after its admitted readers finish."""
        self._closing = True
        async with self._lifecycle:
            if not self._retired:
                if self._publication is not None:
                    self._publication.withdraw()
                self._retired = True
