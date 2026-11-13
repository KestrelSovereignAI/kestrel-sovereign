"""Native request admission under the host's current recipient policy lease."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

from kestrel_sdk.native_a2a.server.context import ServerCallContext
from kestrel_sdk.native_a2a.utils.errors import (
    ExtensionSupportRequiredError,
    InternalError,
)
from kestrel_sdk.native_a2a.utils.stream_refusal import StreamRefusal
from google.protobuf.message import Message as ProtoMessage
from starlette.exceptions import HTTPException

from kestrel_sovereign._async_ownership import OriginalContextExit
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
                    status_code=403,
                    detail="A2A recipient policy refused this principal",
                )
            yield

    @asynccontextmanager
    async def admit(
        self,
        params: _Request,
        operation: str,
        context: ServerCallContext,
        *,
        refusal_values: bool = False,
    ) -> AsyncIterator[AuthorizedNativeRequest[_Request] | StreamRefusal]:
        request = await bind_native_request(
            params,
            context,
            routing_key=self.routing_key,
            scope=self.scope,
            refusal_values=True,
        )
        if isinstance(request, StreamRefusal):
            if refusal_values:
                yield request
            else:
                raise request.error
            return
        activated_extensions = frozenset(context.requested_extensions)
        refusal = None
        if not self.required_extensions <= activated_extensions:
            refusal = StreamRefusal(
                ExtensionSupportRequiredError(
                    message="Required Agent Card extensions were not activated"
                )
            )
        else:
            async with (
                _native_persistence_errors(),
                OriginalContextExit(self.lifecycle_lease()),
            ):
                authority = self._new_authority()
                refusal = self._current_decision(authority, None)
                if refusal is None:
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
                        denied = HTTPException(
                            status_code=401,
                            detail="A2A sender proof was not accepted",
                            headers={
                                "WWW-Authenticate": 'Kestrel-A2A-Sender-Proof realm="a2a"'
                            },
                        )
                        denied.__cause__ = error
                        refusal = StreamRefusal(denied)
                    else:
                        principal = await self._authorization_decision(
                            authority, sender
                        )
                        if isinstance(principal, StreamRefusal):
                            refusal = principal
                        else:
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
        if refusal is not None:
            # These values arise only at explicit decision branches. An actual
            # original lease exit failure never reaches this settled publication.
            if refusal_values:
                yield refusal
            else:
                raise refusal.error

    @staticmethod
    def _current_decision(
        authority: NativeRequestAuthority,
        sender: NativeVerifiedSender | None,
    ) -> StreamRefusal | None:
        """Read the synchronous host policy decision, not a resource/I/O scope."""
        try:
            authority.require_current(sender)
        except HTTPException as error:
            if is_execution_control_error(error):
                raise
            return StreamRefusal(error)
        return None

    async def _authorization_decision(
        self,
        authority: NativeRequestAuthority,
        sender: NativeVerifiedSender,
    ) -> str | StreamRefusal:
        refused = self._current_decision(authority, sender)
        if refused is not None:
            return refused
        principal = await authority.authorize_sender(sender.sender)
        refused = self._current_decision(authority, sender)
        if refused is not None:
            return refused
        if not isinstance(principal, str) or not principal.strip():
            return StreamRefusal(
                HTTPException(
                    status_code=403,
                    detail="A2A recipient policy refused this principal",
                )
            )
        return principal

    async def _authorize(
        self, authority: NativeRequestAuthority, sender: NativeVerifiedSender
    ) -> str:
        principal = await self._authorization_decision(authority, sender)
        if isinstance(principal, StreamRefusal):
            raise principal.error
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
