"""Native request admission under the host's current recipient policy lease."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

from kestrel_sovereign._vendor.a2a.server.context import ServerCallContext
from kestrel_sovereign._vendor.a2a.utils.errors import ExtensionSupportRequiredError, InternalError
from google.protobuf.message import Message as ProtoMessage
from starlette.exceptions import HTTPException

from kestrel_sovereign.a2a.extensions import SIGNED_SENDER_URI
from kestrel_sovereign.a2a.native_context import (
    bind_native_request,
    native_header_values,
)
from kestrel_sovereign.a2a.native_sender import (
    SENDER_PROOF_HEADER,
    NativeProofPolicy,
    NativeSenderProofError,
    NativeVerifiedSender,
    ReplayNonceStore,
    Resolver,
    verify_native_request,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.storage.db.interface import DatabaseError

_Request = TypeVar("_Request", bound=ProtoMessage)


@asynccontextmanager
async def _native_persistence_errors() -> AsyncIterator[None]:
    """Expose a standard internal error, retaining the original for host diagnostics."""
    try:
        yield
    except DatabaseError as error:
        if is_execution_control_error(error):
            raise
        logging.getLogger(__name__).exception("Native A2A persistence failed")
        raise InternalError(message="A2A persistence is unavailable") from error


@dataclass(frozen=True)
class NativeRequestAuthority:
    """Request-owned policy callbacks retained across waits and stream reads."""

    resolver: Resolver
    authorize_sender: Callable[[str], Awaitable[str | None]]
    require_current: Callable[[NativeVerifiedSender | None], None]
    authorize_retained_principal: Callable[[str], Awaitable[bool]]


@dataclass(frozen=True)
class AuthorizedNativeRequest(Generic[_Request]):
    """A native request and the durable principal selected by recipient policy."""

    request: _Request
    principal_id: str
    sender: NativeVerifiedSender
    scope: NativeRecipientScope
    routing_key: str
    authority: NativeRequestAuthority
    issuer: "NativeDIDRequestGuard"
    activated_extensions: frozenset[str]


@dataclass(frozen=True)
class NativeDIDRequestGuard:
    """Verify and authorize before exposing a task or admitting any mutation.

    The host supplies its current policy check and shared topology lease. The
    authorization callback is the recipient's existing scoped authorizer; the
    guard never interprets caller metadata as a principal or a permission.
    """

    routing_key: str
    scope: NativeRecipientScope
    required_extensions: frozenset[str]
    proof_policy: NativeProofPolicy
    authority_factory: Callable[[], NativeRequestAuthority]
    replay_store: ReplayNonceStore
    lifecycle_lease: Callable[[], AbstractAsyncContextManager[object]]
    clock: Callable[[], float]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, NativeRecipientScope):
            raise TypeError("Native admission requires a host-owned recipient scope")
        if not isinstance(self.routing_key, str) or not self.routing_key.strip():
            raise ValueError("Native admission requires a concrete routing key")
        if not isinstance(self.required_extensions, frozenset):
            raise TypeError("Native required extensions must be immutable")
        if SIGNED_SENDER_URI not in self.required_extensions:
            raise ValueError("The DID guard requires the signed-sender extension")

    def _new_authority(self) -> NativeRequestAuthority:
        authority = self.authority_factory()
        if not isinstance(authority, NativeRequestAuthority):
            raise TypeError("Native admission requires request-owned policy authority")
        return authority

    @asynccontextmanager
    async def hold_retained_authority(self, principal_id: str) -> AsyncIterator[None]:
        """Reauthorize a canonical callback creator under this publication lease.

        No original request nonce is replayed and no new identity is verified.
        The caller must supply the principal from its retained server-side row.
        """
        if not isinstance(principal_id, str) or not principal_id.strip():
            raise ValueError("Retained native authority requires a concrete principal")
        # Resident polling retains typed native database outcomes for its
        # idempotent retry policy; HTTP error translation belongs to requests.
        async with self.lifecycle_lease():
            authority = self._new_authority()
            authority.require_current(None)
            allowed = await authority.authorize_retained_principal(principal_id)
            authority.require_current(None)
            if allowed is not True:
                raise HTTPException(
                    status_code=403, detail="A2A recipient policy refused this principal"
                )
            yield

    @asynccontextmanager
    async def admit(
        self, params: _Request, operation: str, context: ServerCallContext
    ) -> AsyncIterator[AuthorizedNativeRequest[_Request]]:
        request = await bind_native_request(
            params,
            context,
            routing_key=self.routing_key,
            scope=self.scope,
        )
        activated_extensions = frozenset(context.requested_extensions)
        if not self.required_extensions <= activated_extensions:
            raise ExtensionSupportRequiredError(
                message="Required Agent Card extensions were not activated"
            )
        async with _native_persistence_errors(), self.lifecycle_lease():
            authority = self._new_authority()
            authority.require_current(None)
            try:
                sender = await verify_native_request(
                    native_header_values(context, SENDER_PROOF_HEADER),
                    request,
                    recipient=self.scope.recipient_id,
                    operation=operation,
                    extensions=activated_extensions,
                    resolver=authority.resolver,
                    replay_store=self.replay_store,
                    policy=self.proof_policy,
                    now_ts=int(self.clock()),
                )
            except NativeSenderProofError as error:
                if is_execution_control_error(error):
                    raise
                raise HTTPException(
                    status_code=401,
                    detail="A2A sender proof was not accepted",
                    headers={
                        "WWW-Authenticate": 'Kestrel-A2A-Sender-Proof realm="a2a"'
                    },
                ) from error
            principal = await self._authorize(authority, sender)
            yield AuthorizedNativeRequest(
                request,
                principal,
                sender,
                self.scope,
                self.routing_key,
                authority,
                self,
                activated_extensions,
            )

    async def _authorize(
        self, authority: NativeRequestAuthority, sender: NativeVerifiedSender
    ) -> str:
        authority.require_current(sender)
        principal = await authority.authorize_sender(sender.sender)
        authority.require_current(sender)
        if not isinstance(principal, str) or not principal.strip():
            raise HTTPException(
                status_code=403, detail="A2A recipient policy refused this principal"
            )
        return principal

    @asynccontextmanager
    async def hold_authority(
        self, admitted: AuthorizedNativeRequest[_Request]
    ) -> AsyncIterator[None]:
        """Recheck a waiting request before its next authorized read or event.

        The handler retains this value within the admitted request. It is not
        an independent proof and must never be accepted from caller metadata.
        Waiting for background work holds no topology lease; final reads and
        streamed events reacquire it without spending the same nonce again.
        """
        if (
            admitted.issuer is not self
            or admitted.scope != self.scope
            or admitted.routing_key != self.routing_key
        ):
            raise HTTPException(
                status_code=403,
                detail="Native request belongs to a different recipient registration",
            )
        async with _native_persistence_errors(), self.lifecycle_lease():
            principal = await self._authorize(admitted.authority, admitted.sender)
            if principal != admitted.principal_id:
                raise HTTPException(
                    status_code=403,
                    detail="Native request principal changed while waiting",
                )
            yield
