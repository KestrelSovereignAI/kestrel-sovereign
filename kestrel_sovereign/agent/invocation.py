"""Task-local identity for one top-level agent invocation.

Request cancellation has legacy agent-global state because a stop request can
arrive from another task. Provenance cannot use that mutable state: a queued
turn may observe a later request's id. This module carries the immutable
operation identity with the task that is actually executing the turn.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from contextvars import ContextVar, copy_context
from functools import wraps
import hashlib
import inspect
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Mapping, TypeVar
from urllib.parse import quote, unquote_to_bytes
import uuid


MAX_INVOCATION_ID_LENGTH = 256
_HEADER_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)

_current_invocation_id: ContextVar[str | None] = ContextVar(
    "kestrel_current_invocation_id",
    default=None,
)


@dataclass(frozen=True, slots=True)
class InvocationProvenance:
    """Trusted transport context bound to one top-level agent invocation.

    This is deliberately constructed by an endpoint or other trusted entry
    point, never from a tool's arguments.  Semantic producers can therefore
    record the authenticated requester and transport occurrence without
    accepting authority-bearing provenance fields from model/tool content.
    """

    actor: str | None
    source_kind: str
    source_locator: str
    received_at: str


_current_invocation_provenance: ContextVar[InvocationProvenance | None] = ContextVar(
    "kestrel_current_invocation_provenance",
    default=None,
)

_T = TypeVar("_T")


class InvocationCancelledError(Exception):
    """An isolated turn ended without cancelling its long-lived caller."""


def validate_invocation_id(value: object) -> str:
    """Return a bounded opaque invocation id or reject an invalid one."""
    if not isinstance(value, str) or not (1 <= len(value) <= MAX_INVOCATION_ID_LENGTH):
        raise ValueError(
            "invocation id must be a non-empty string no longer than "
            f"{MAX_INVOCATION_ID_LENGTH} characters"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("invocation id must be valid Unicode text") from error
    return value


def invocation_id_response_header(value: object) -> str:
    """Return an ASCII-safe representation of an accepted invocation ID.

    Invocation IDs are opaque UTF-8 strings because callers can use their own
    retry keys.  HTTP response headers, however, are Latin-1 in Starlette.  A
    percent-encoded representation preserves that opaque identity on the wire
    without turning a valid request into a post-mutation 500.  Unreserved
    ASCII IDs retain their familiar spelling.
    """
    return quote(validate_invocation_id(value), safe="-._~")


def invocation_id_from_request_header(value: object) -> str:
    """Decode the one-pass wire form used by ``X-Request-ID``.

    Request IDs are opaque Unicode *values*, while HTTP headers are an ASCII
    transport.  ``X-Request-ID`` therefore always uses RFC 3986 percent
    encoding: a client retries with the response value verbatim, and the
    server decodes it exactly once.  Body ``request_id`` / ``invocation_id`` /
    ``id`` values remain literal Unicode and retain their existing precedence.

    This is deliberately not a best-effort decoder.  A literal percent in a
    header value must be sent as ``%25``; otherwise a value such as ``%E2`` is
    ambiguous between an identifier and a wire escape.  Rejecting malformed
    wire forms prevents a double-decoding or a silently forked retry key.
    """
    if not isinstance(value, str):
        raise ValueError("request-id header must be text")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(
            "request-id header must use ASCII percent-encoding for Unicode text"
        ) from error

    index = 0
    while index < len(value):
        character = value[index]
        if character in _HEADER_UNRESERVED:
            index += 1
            continue
        if character != "%" or (
            index + 2 >= len(value)
            or value[index + 1] not in "0123456789abcdefABCDEF"
            or value[index + 2] not in "0123456789abcdefABCDEF"
        ):
            raise ValueError(
                "request-id header must percent-encode reserved characters"
            )
        index += 3
    try:
        decoded = unquote_to_bytes(value).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            "request-id header percent escapes must decode to valid UTF-8"
        ) from error
    return validate_invocation_id(decoded)


def invocation_log_correlation(value: object) -> str:
    """Return a non-reversible operator-log correlation for an invocation.

    Transport retry identifiers are client-controlled and can contain private
    content.  They remain opaque to application logs; this short digest lets
    operators correlate related events without persisting the identifier.
    """
    invocation_id = validate_invocation_id(value)
    return hashlib.sha256(invocation_id.encode("utf-8")).hexdigest()[:16]


def new_invocation_id() -> str:
    """Create an opaque identity for a producer without a transport id."""
    return str(uuid.uuid4())


def ensure_invocation_id(value: object = None) -> str:
    """Validate a supplied identity or generate one for a local producer."""
    return new_invocation_id() if value is None else validate_invocation_id(value)


def resolve_transport_invocation_id(
    body: Mapping[str, object] | object | None,
    header_request_id: object = None,
) -> str:
    """Resolve one client retry identity from the shared HTTP contract.

    Explicit body fields win over the header for backwards compatibility with
    ``/api/agent``.  ``id`` is accepted for OpenAI-compatible clients, while
    ``invocation_id`` lets non-Kestrel gateways name the same opaque operation
    without overloading a provider's response identifier.
    """
    for field in ("request_id", "invocation_id", "id"):
        if isinstance(body, Mapping):
            value = body.get(field)
        else:
            value = getattr(body, field, None) if body is not None else None
        if value is not None:
            return ensure_invocation_id(value)
    if header_request_id is None:
        return ensure_invocation_id()
    return invocation_id_from_request_header(header_request_id)


def request_provenance(
    *,
    actor: object = None,
    source_kind: str,
    source_locator: str,
    received_at: str | None = None,
) -> InvocationProvenance:
    """Create bounded, endpoint-owned provenance for the current request.

    ``actor`` is optional because a token-authenticated webhook can identify a
    trusted transport without claiming that an untrusted payload sender is the
    actor.  Endpoint code supplies the authenticated ``CallerContext``
    identity when one exists.
    """
    if actor is not None and not isinstance(actor, str):
        raise ValueError("provenance actor must be text when supplied")
    if actor is not None and not actor:
        actor = None
    source_kind = validate_invocation_id(source_kind)
    source_locator = validate_invocation_id(source_locator)
    received_at = received_at or datetime.now(timezone.utc).isoformat()
    return InvocationProvenance(
        actor=actor,
        source_kind=source_kind,
        source_locator=source_locator,
        received_at=received_at,
    )


def new_stream_delivery_id() -> str:
    """Return a server-owned delivery identity distinct from retry identity."""
    return f"stream:{uuid.uuid4()}"


def current_invocation_id() -> str | None:
    """Return the identity bound to this task, never an agent-global fallback."""
    return _current_invocation_id.get()


def current_invocation_provenance() -> InvocationProvenance | None:
    """Return task-local trusted request provenance, if the entry point bound it."""
    return _current_invocation_provenance.get()


@contextmanager
def _exact_invocation_scope(
    invocation_id: str,
    provenance: InvocationProvenance | None,
) -> Iterator[str]:
    """Bind an already-resolved identity and an exact provenance value."""

    id_token = _current_invocation_id.set(invocation_id)
    provenance_token = _current_invocation_provenance.set(provenance)
    try:
        yield invocation_id
    finally:
        _current_invocation_provenance.reset(provenance_token)
        _current_invocation_id.reset(id_token)


@contextmanager
def invocation_scope(
    invocation_id: object = None,
    *,
    provenance: InvocationProvenance | None = None,
) -> Iterator[str]:
    """Bind one validated identity for the current async task and its awaits."""
    if provenance is not None and not isinstance(provenance, InvocationProvenance):
        raise TypeError("invocation provenance must be endpoint-owned InvocationProvenance")
    effective_id = ensure_invocation_id(invocation_id)
    effective_provenance = (
        provenance
        if provenance is not None
        else _current_invocation_provenance.get()
    )
    with _exact_invocation_scope(effective_id, effective_provenance):
        yield effective_id


def bind_async_invocation(
    parameter: str,
    *,
    track_request_lifecycle: bool = False,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Bind/generate ``parameter`` for an async top-level turn method.

    ``track_request_lifecycle`` makes the decorated method the canonical
    inventory boundary for every transport that calls it. This avoids a Stop
    implementation that works only for endpoints which remembered to install
    their own lifecycle wrapper.
    """
    def decorate(function: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        signature = inspect.signature(function)

        @wraps(function)
        async def wrapped(*args: Any, **kwargs: Any) -> _T:
            bound = signature.bind_partial(*args, **kwargs)
            with invocation_scope(
                bound.arguments.get(parameter),
                provenance=bound.arguments.get("invocation_provenance"),
            ) as invocation_id:
                bound.arguments[parameter] = invocation_id
                lifecycle_owner = args[0] if args else None
                registered = False
                if track_request_lifecycle and lifecycle_owner is not None:
                    register = getattr(
                        type(lifecycle_owner),
                        "register_active_request",
                        None,
                    )
                    if callable(register):
                        register(lifecycle_owner, invocation_id)
                        registered = True
                try:
                    if registered:
                        bind_operation = getattr(
                            type(lifecycle_owner),
                            "bind_request_operation",
                            None,
                        )
                        parent_context = copy_context()
                        operation_context = parent_context.copy()
                        operation = asyncio.create_task(
                            function(*bound.args, **bound.kwargs),
                            name=(
                                "invocation-turn:"
                                f"{invocation_log_correlation(invocation_id)}"
                            ),
                            context=operation_context,
                        )
                        if callable(bind_operation):
                            bind_operation(
                                lifecycle_owner,
                                invocation_id,
                                operation,
                            )
                        try:
                            return await operation
                        except asyncio.CancelledError as error:
                            caller = asyncio.current_task()
                            if caller is not None and caller.cancelling():
                                raise
                            raise InvocationCancelledError(
                                "isolated invocation was cancelled "
                                f"({invocation_log_correlation(invocation_id)})"
                            ) from error
                        finally:
                            # A normal ``await function(...)`` shares ContextVar
                            # updates with its caller. Isolating the cancellable
                            # task must preserve that contract (notably the
                            # constitution-injection audit read immediately
                            # after process_input returns).
                            missing = object()
                            for variable in operation_context:
                                child_value = operation_context.get(
                                    variable, missing
                                )
                                parent_value = parent_context.get(
                                    variable, missing
                                )
                                if (
                                    child_value is not missing
                                    and child_value != parent_value
                                ):
                                    variable.set(child_value)
                    return await function(*bound.args, **bound.kwargs)
                finally:
                    if registered:
                        lifecycle_owner._cleanup_cancelled_request(invocation_id)

        return wrapped

    return decorate


def bind_async_generator_invocation(
    parameter: str,
) -> Callable[[Callable[..., AsyncIterator[_T]]], Callable[..., AsyncIterator[_T]]]:
    """Bind/generate ``parameter`` for an async-generator top-level turn.

    Streaming ``request_id`` also drives legacy UI/cancellation routing.  Its
    absence intentionally remains observable to preserve that fallback; use
    the task-local context for provenance without injecting a generated value
    into the method unless the caller explicitly asks for it.
    """
    def decorate(
        function: Callable[..., AsyncIterator[_T]],
    ) -> Callable[..., AsyncIterator[_T]]:
        signature = inspect.signature(function)

        @wraps(function)
        async def wrapped(*args: Any, **kwargs: Any) -> AsyncIterator[_T]:
            bound = signature.bind_partial(*args, **kwargs)
            # Do not hold ContextVar tokens across an outward ``yield``. The
            # consumer is allowed to close this iterator from a cancellation-
            # safe owner task; resetting a token created by the original task
            # from that closer's Context raises ValueError and can falsely make
            # request cleanup look complete. Bind only while advancing or
            # closing the underlying generator, and retain one effective id and
            # provenance for every advance even if different tasks drive it.
            effective_id = ensure_invocation_id(bound.arguments.get(parameter))
            supplied_provenance = bound.arguments.get("invocation_provenance")
            if supplied_provenance is not None and not isinstance(
                supplied_provenance, InvocationProvenance
            ):
                raise TypeError(
                    "invocation provenance must be endpoint-owned "
                    "InvocationProvenance"
                )
            effective_provenance = (
                supplied_provenance
                if supplied_provenance is not None
                else _current_invocation_provenance.get()
            )
            iterator = function(*bound.args, **bound.kwargs)
            try:
                while True:
                    with _exact_invocation_scope(
                        effective_id,
                        effective_provenance,
                    ):
                        try:
                            item = await anext(iterator)
                        except StopAsyncIteration:
                            return
                    yield item
            finally:
                close_iterator = getattr(iterator, "aclose", None)
                if callable(close_iterator):
                    with _exact_invocation_scope(
                        effective_id,
                        effective_provenance,
                    ):
                        await close_iterator()

        return wrapped

    return decorate
