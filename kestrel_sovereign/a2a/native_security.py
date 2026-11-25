"""Native request admission under the host's current recipient policy lease."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from typing import Generic, TypeVar

from google.protobuf.message import Message as ProtoMessage
from kestrel_sdk.native_a2a.server.context import ServerCallContext
from kestrel_sdk.native_a2a.utils.errors import (
    ExtensionSupportRequiredError,
    InternalError,
)
from kestrel_sdk.native_a2a.utils.stream_refusal import StreamRefusal
from starlette.exceptions import HTTPException

from kestrel_sovereign._async_ownership import OriginalContextExit
from kestrel_sovereign.a2a.extensions import SIGNED_SENDER_URI
from kestrel_sovereign.a2a.native_context import (
    NativeLocalSender,
    bind_native_request,
    native_header_values,
    native_local_sender,
)
from kestrel_sovereign.a2a.native_sender import (
    SENDER_PROOF_HEADER,
    NativeProofPolicy,
    NativeSenderProofError,
    NativeVerifiedSender,
    ReplayNonceStore,
    Resolver,
    authenticate_native_request,
    reserve_native_sender_nonce,
)
from kestrel_sovereign.a2a.replay_store import SharedReplayNonceStore
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    execution_error_with_cause,
    is_execution_control_error,
)
from kestrel_sovereign.storage.db.interface import DatabaseError

_Request = TypeVar("_Request", bound=ProtoMessage)


@asynccontextmanager
async def _original_authentication_io() -> AsyncIterator[None]:
    """Opaque original I/O failures are control, never proof/refusal data.

    Same caller Task and exact original graph; no resource-close or commit ACK
    is invented. Only explicit decision values outside this boundary can be
    translated by the HTTP layer. Already-classified native evidence stays exact.
    """
    try:
        yield
    except BaseException as error:
        if is_execution_control_error(error):
            raise
        raise execution_error_with_cause(
            ExecutionAuthorityError("Original native authentication I/O failed"), error
        )


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


class NativeRequestKind(Enum):
    """Semantically classified native admission, never an HTTP header."""

    WORK = "work"
    CANCEL_TASK = "cancel-task"
    PEER_STOP = "peer-stop"


@dataclass(frozen=True)
class NativeVerifiedAdmission:
    """Original local sender selected by the host and matched to signed proof."""

    sender: NativeVerifiedSender
    origin: NativeLocalSender
    kind: NativeRequestKind


@dataclass(frozen=True)
class NativeHostedLocalAdmission:
    """Split Core proof verification from the embedding's durable admission.

    Verification must not take the embedding's workload or control slot. The
    verified lease enters the host fence before the original Core reader and
    holds both source and recipient custody through an effect or authorized
    read. Discovery and durable-principal callbacks still use lifecycle_lease.
    """

    verification_lease: Callable[[], AbstractAsyncContextManager[object]]
    acquire: Callable[
        [NativeVerifiedAdmission, Callable[[], None] | None],
        AbstractAsyncContextManager[StreamRefusal | None],
    ]


class NativeAdmissionExpired(NativeSenderProofError):
    """Initial proof freshness was lost before the host's admission point."""


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
    verified_admission: NativeVerifiedAdmission | None = None


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
    check_recipient_custody: Callable[[], Awaitable[StreamRefusal | None]] | None = None
    local_admission: NativeHostedLocalAdmission | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, NativeRecipientScope):
            raise TypeError("Native admission requires a host-owned recipient scope")
        if not isinstance(self.routing_key, str) or not self.routing_key.strip():
            raise ValueError("Native admission requires a concrete routing key")
        if not isinstance(self.required_extensions, frozenset):
            raise TypeError("Native required extensions must be immutable")
        if SIGNED_SENDER_URI not in self.required_extensions:
            raise ValueError("The DID guard requires the signed-sender extension")
        if self.local_admission is not None and not isinstance(
            self.local_admission, NativeHostedLocalAdmission
        ):
            raise TypeError("Hosted local admission requires its typed lease contract")
        if self.local_admission is not None and not isinstance(self.replay_store, SharedReplayNonceStore):
            raise TypeError("Hosted local admission requires durable shared nonce storage")

    def _new_authority(self) -> NativeRequestAuthority:
        authority = self.authority_factory()
        if not isinstance(authority, NativeRequestAuthority):
            raise TypeError("Native admission requires request-owned policy authority")
        return authority

    async def _recipient_custody_refusal(self) -> StreamRefusal | None:
        if self.check_recipient_custody is not None:
            return await self.check_recipient_custody()
        return None

    async def require_recipient_custody(self) -> None:
        refusal = await self._recipient_custody_refusal()
        if refusal is not None:
            raise refusal.error

    async def _authenticate(
        self, request: ProtoMessage, operation: str, context: ServerCallContext,
        authority: NativeRequestAuthority, extensions: frozenset[str], now_ts: int,
    ) -> NativeVerifiedSender | NativeSenderProofError:
        async with _original_authentication_io():
            return await authenticate_native_request(
                native_header_values(context, SENDER_PROOF_HEADER), request,
                recipient=self.scope.recipient_id, operation=operation,
                extensions=extensions, resolver=authority.resolver,
                policy=self.proof_policy, now_ts=now_ts, refusal_values=True,
            )

    async def _spend_under_recipient_custody(
        self, sender: NativeVerifiedSender, now_ts: int,
    ) -> NativeVerifiedSender | NativeSenderProofError | StreamRefusal:
        refusal = await self._recipient_custody_refusal()
        if refusal is not None:
            return refusal
        async with _original_authentication_io():
            return await reserve_native_sender_nonce(
                sender, replay_store=self.replay_store, policy=self.proof_policy,
                now_ts=now_ts, refusal_values=True,
            )

    @staticmethod
    def _proof_refusal(error: NativeSenderProofError) -> StreamRefusal:
        denied = HTTPException(
            status_code=401,
            detail="A2A sender proof was not accepted",
            headers={"WWW-Authenticate": 'Kestrel-A2A-Sender-Proof realm="a2a"'},
        )
        denied.__cause__ = error
        return StreamRefusal(denied)

    @staticmethod
    def _local_sender_decision(
        sender: NativeVerifiedSender, origin: NativeLocalSender | None,
    ) -> StreamRefusal | None:
        if (
            origin is None
            or origin.signing_did != sender.sender
            or origin.verification_document_fingerprint != sender.verification_document_fingerprint
            or not origin.is_current()
        ):
            return StreamRefusal(HTTPException(
                status_code=403, detail="A2A original local sender is not current",
            ))
        return None

    def _require_fresh_proof(self, sender: NativeVerifiedSender) -> None:
        if (
            sender.issued_at is None
            or abs(int(self.clock()) - sender.issued_at) > self.proof_policy.max_age_seconds
        ):
            raise NativeAdmissionExpired("Sender proof expired during admission")

    @asynccontextmanager
    async def hold_local_source(self) -> AsyncIterator[StreamRefusal | None]:
        """Retain the exact published source's original Core reader."""
        if self.local_admission is None:
            raise RuntimeError("Native local source has no hosted admission contract")
        async with OriginalContextExit(self.local_admission.verification_lease()):
            refusal = self._current_decision(self._new_authority(), None)
            if refusal is None:
                yield None
                return
        yield refusal

    @asynccontextmanager
    async def _verified_local_lease(
        self, verified: NativeVerifiedAdmission, *, initial: bool = False,
    ) -> AsyncIterator[StreamRefusal | None]:
        if self.local_admission is None:
            raise RuntimeError("Native hosted admission lease was not configured")
        refused = self._local_sender_decision(verified.sender, verified.origin)
        if refused is None:
            async with verified.origin.authority_lease() as refused:
                pass
        if refused is None:
            freshness = partial(self._require_fresh_proof, verified.sender) if initial else None
            async with AsyncExitStack() as leases:
                try:
                    if freshness is not None:
                        freshness()
                    refused = await leases.enter_async_context(OriginalContextExit(
                        self.local_admission.acquire(verified, freshness),
                    ))
                except NativeAdmissionExpired as error:
                    refused = self._proof_refusal(error)
                if refused is not None and not isinstance(refused, StreamRefusal):
                    raise TypeError("Hosted admission lease returned an invalid decision")
                if refused is None:
                    # Both distributed generations precede the original Core
                    # readers; the stable route order applies to reciprocal
                    # same-process calls without turning shared locks into wait cycles.
                    readers = (
                        (verified.origin.authority_lease, self.hold_local_source)
                        if verified.origin.publication.routing_key < self.routing_key
                        else (self.hold_local_source, verified.origin.authority_lease)
                    )
                    for reader in readers:
                        refused = await leases.enter_async_context(reader())
                        if refused is not None:
                            break
                if refused is None:
                    refused = self._local_sender_decision(verified.sender, verified.origin)
                if refused is None:
                    yield None
                    return
        yield refused

    @asynccontextmanager
    async def _admit_original(
        self, request: _Request, operation: str, context: ServerCallContext,
        extensions: frozenset[str],
        classify_peer_stop: Callable[[_Request, frozenset[str]], bool] | None,
    ) -> AsyncIterator[AuthorizedNativeRequest[_Request] | StreamRefusal]:
        local = self.local_admission
        refusal = None
        # The standalone/fleet path keeps its original continuous lease. The
        # hosted local path releases the read-only verifier before waiting on
        # the host's admission fence, maintaining host-before-Core lock order.
        verification_lease = self.lifecycle_lease if local is None else local.verification_lease
        async with OriginalContextExit(verification_lease()):
            authority = self._new_authority()
            refusal = self._current_decision(authority, None)
            if refusal is None:
                now_ts = int(self.clock())
                sender = await self._authenticate(
                    request, operation, context, authority, extensions, now_ts,
                )
                if isinstance(sender, NativeSenderProofError):
                    refusal = self._proof_refusal(sender)
                elif local is None:
                    result = await self._authorize_initial(
                        request, sender, authority, extensions, now_ts,
                    )
                    if isinstance(result, StreamRefusal):
                        refusal = result
                    else:
                        yield result
                        return
                else:
                    origin = native_local_sender(context)
                    refusal = self._local_sender_decision(sender, origin)
                    if refusal is None:
                        self.replay_store.require_outermost_reservation()
                        reserved = await self._spend_under_recipient_custody(sender, now_ts)
                        if isinstance(reserved, StreamRefusal):
                            refusal = reserved
                        elif isinstance(reserved, NativeSenderProofError):
                            refusal = self._proof_refusal(reserved)
        if refusal is not None:
            yield refusal
            return

        assert local is not None and origin is not None
        kind = NativeRequestKind.WORK
        if operation == "CancelTask":
            kind = NativeRequestKind.CANCEL_TASK
        elif (
            operation in {"SendMessage", "SendStreamingMessage"}
            and classify_peer_stop is not None
            and classify_peer_stop(request, extensions)
        ):
            kind = NativeRequestKind.PEER_STOP
        verified = NativeVerifiedAdmission(sender, origin, kind)
        async with self._verified_local_lease(verified, initial=True) as refusal:
            if refusal is None:
                refusal = self._current_decision(authority, sender)
            if refusal is None:
                try:
                    self._require_fresh_proof(sender)
                except NativeAdmissionExpired as error:
                    refusal = self._proof_refusal(error)
                if refusal is None:
                    result = await self._authorized_result(
                        request, sender, authority, extensions, verified=verified,
                    )
                    if isinstance(result, StreamRefusal):
                        refusal = result
                    else:
                        yield result
                        return
        assert refusal is not None
        yield refusal

    async def _authorize_initial(
        self, request: _Request, sender: NativeVerifiedSender,
        authority: NativeRequestAuthority, extensions: frozenset[str], now_ts: int,
    ) -> AuthorizedNativeRequest[_Request] | StreamRefusal:
        result = await self._spend_under_recipient_custody(sender, now_ts)
        if isinstance(result, StreamRefusal):
            return result
        if isinstance(result, NativeSenderProofError):
            return self._proof_refusal(result)
        return await self._authorized_result(request, sender, authority, extensions)

    async def _authorized_result(
        self, request: _Request, sender: NativeVerifiedSender,
        authority: NativeRequestAuthority, extensions: frozenset[str],
        *, verified: NativeVerifiedAdmission | None = None,
    ) -> AuthorizedNativeRequest[_Request] | StreamRefusal:
        principal = await self._authorization_decision(authority, sender)
        if isinstance(principal, StreamRefusal):
            return principal
        if verified is not None:
            refusal = self._local_sender_decision(sender, verified.origin)
            if refusal is not None:
                return refusal
        return AuthorizedNativeRequest(
            request, principal, sender, self.scope, self.routing_key, authority,
            self, extensions, verified,
        )

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
            await self.require_recipient_custody()
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
        classify_peer_stop: Callable[[_Request, frozenset[str]], bool] | None = None,
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
                self._admit_original(
                    request, operation, context, activated_extensions, classify_peer_stop,
                ) as decision,
            ):
                if isinstance(decision, StreamRefusal):
                    refusal = decision
                else:
                    yield decision
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
        async with _original_authentication_io():
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
    async def _retained_request_lease(
        self, admitted: AuthorizedNativeRequest[_Request], *, workload_only: bool,
        first_effect: bool,
    ) -> AsyncIterator[StreamRefusal | None]:
        verified = admitted.verified_admission
        if self.local_admission is None:
            async with OriginalContextExit(self.lifecycle_lease()):
                yield None
        elif verified is None or verified.sender is not admitted.sender:
            yield StreamRefusal(HTTPException(
                status_code=403, detail="A2A request lacks original hosted admission",
            ))
        else:
            if workload_only and verified.kind is not NativeRequestKind.WORK:
                verified = replace(verified, kind=NativeRequestKind.WORK)
            async with self._verified_local_lease(verified, initial=first_effect) as refused:
                yield refused

    @asynccontextmanager
    async def hold_authority(
        self, admitted: AuthorizedNativeRequest[_Request], *,
        refusal_values: bool = False, workload_only: bool = False, first_effect: bool = False,
    ) -> AsyncIterator[StreamRefusal | None]:
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
            refused = StreamRefusal(
                HTTPException(
                    status_code=403,
                    detail="Native request belongs to a different recipient registration",
                )
            )
        else:
            refused = None
            async with (
                _native_persistence_errors(),
                self._retained_request_lease(
                    admitted, workload_only=workload_only, first_effect=first_effect,
                ) as lease_refusal,
            ):
                refused = lease_refusal
                if refused is None and self.check_recipient_custody is not None:
                    refused = self._current_decision(admitted.authority, admitted.sender)
                    if refused is None:
                        refused = await self._recipient_custody_refusal()
                if refused is None:
                    principal = await self._authorization_decision(admitted.authority, admitted.sender)
                    if isinstance(principal, StreamRefusal):
                        refused = principal
                    elif principal != admitted.principal_id:
                        refused = StreamRefusal(
                            HTTPException(
                                status_code=403,
                                detail="Native request principal changed while waiting",
                            )
                        )
                    elif admitted.verified_admission is not None:
                        refused = self._local_sender_decision(
                            admitted.sender, admitted.verified_admission.origin,
                        )
                    if refused is None:
                        yield None
        if refused is not None:
            # Publish a known pre-effect decision only AFTER the actual lease
            # exits. Actual exit/I/O/body failures never become refusal values.
            if refusal_values:
                yield refused
            else:
                raise refused.error
