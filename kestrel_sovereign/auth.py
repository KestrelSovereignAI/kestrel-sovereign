"""Caller context for threading authentication identity into agent operations."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from kestrel_sovereign.security.sovereign_key import sovereign_key_fingerprint


def normalize_api_key(value: Optional[str]) -> Optional[str]:
    """Normalize a configured API credential exactly once for every auth lane."""

    if value is None:
        return None
    # Docker ``--env-file`` retains matching surrounding quotes literally.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
    # Callers use ``None`` as the canonical "no credential" state. Returning
    # an empty string here would let a literal Docker env-file value of ``""``
    # compare equal to an empty bearer token after normalization.
    return value or None


class AuthMethod(str, Enum):
    """How the caller authenticated."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH_SESSION = "oauth_session"
    A2A_TRANSPORT = "a2a_transport"
    INTERNAL = "internal"  # Agent-to-agent or system calls


class CallerRole(str, Enum):
    """Authority level of the caller."""
    SOVEREIGN = "sovereign"      # API key holder — full control
    AUTHENTICATED = "authenticated"  # OAuth/JWT user — may be restricted
    ANONYMOUS = "anonymous"      # No identity resolved


@dataclass(frozen=True)
class CallerContext:
    """Identity and authority of the caller making a request.

    Threaded from auth middleware → endpoint → process_input → command handler.
    """
    role: CallerRole = CallerRole.ANONYMOUS
    auth_method: AuthMethod = AuthMethod.INTERNAL
    identity: Optional[str] = None  # email, "api_key", or None
    # Present only when the endpoint authenticated an actual sovereign API-key
    # credential. It binds later mutation authority to that entry credential
    # without retaining the secret itself.
    credential_fingerprint: Optional[str] = None

    @staticmethod
    def sovereign(
        auth_method: AuthMethod = AuthMethod.API_KEY,
        identity: str = None,
        *,
        credential: str | None = None,
    ) -> "CallerContext":
        return CallerContext(
            role=CallerRole.SOVEREIGN,
            auth_method=auth_method,
            identity=identity or "api_key",
            credential_fingerprint=(
                sovereign_key_fingerprint(credential)
                if credential is not None
                else None
            ),
        )

    @staticmethod
    def authenticated(identity: str, auth_method: AuthMethod = AuthMethod.OAUTH_SESSION) -> "CallerContext":
        return CallerContext(role=CallerRole.AUTHENTICATED, auth_method=auth_method, identity=identity)

    @staticmethod
    def a2a_transport() -> "CallerContext":
        """Transport admission only; never sovereign or task authority."""

        return CallerContext(
            role=CallerRole.AUTHENTICATED,
            auth_method=AuthMethod.A2A_TRANSPORT,
            identity="a2a_transport",
        )

    @staticmethod
    def anonymous() -> "CallerContext":
        return CallerContext(role=CallerRole.ANONYMOUS, auth_method=AuthMethod.INTERNAL)

    @property
    def is_sovereign(self) -> bool:
        return self.role == CallerRole.SOVEREIGN


@dataclass
class _CallerContextBinding:
    """Revocable authority shared by every copied task context."""

    caller: Optional[CallerContext]
    active: bool = True


_current_caller_context: ContextVar[Optional[_CallerContextBinding]] = ContextVar(
    "kestrel_current_caller_context",
    default=None,
)


def current_caller_context() -> Optional[CallerContext]:
    """Return the caller while its endpoint-owned scope remains active."""

    binding = _current_caller_context.get()
    if binding is None or not binding.active:
        return None
    return binding.caller


def capture_caller_context_binding() -> Optional[_CallerContextBinding]:
    """Capture the current revocable endpoint binding for a callback task.

    The returned object is deliberately the binding, not merely its immutable
    :class:`CallerContext`.  A persistent transport may execute a turn-owned
    callback on a task whose ContextVar snapshot predates that turn.  Rebinding
    this same object carries the exact caller while it remains owned by the
    endpoint, and observes revocation as soon as the endpoint scope exits.
    """

    binding = _current_caller_context.get()
    if binding is None or not binding.active:
        return None
    return binding


@contextmanager
def caller_context_lifetime(caller: Optional[CallerContext]):
    """Own one revocable caller binding without implicitly publishing it.

    Async-generator endpoints need a binding whose lifetime spans every
    ``anext`` call, while publishing it only while their own generator body is
    running.  Callback closures can capture the yielded binding and explicitly
    re-present it on foreign transport tasks.  Exiting this lifetime revokes all
    copied/re-presented views at once.
    """

    if caller is not None and not isinstance(caller, CallerContext):
        raise TypeError("caller context must be endpoint-owned CallerContext")
    binding = _CallerContextBinding(caller=caller)
    try:
        yield binding
    finally:
        binding.active = False


@contextmanager
def caller_context_binding_scope(
    binding: Optional[_CallerContextBinding],
):
    """Re-present an existing revocable binding in the current task.

    ``None`` explicitly clears any ambient caller.  This scope never extends a
    binding's lifetime and never reactivates an expired binding; the owner from
    :func:`caller_context_lifetime` remains the sole revocation authority.
    """

    if binding is not None and not isinstance(binding, _CallerContextBinding):
        raise TypeError("caller binding must come from caller_context_lifetime")
    token = _current_caller_context.set(binding)
    try:
        yield current_caller_context()
    finally:
        _current_caller_context.reset(token)


@contextmanager
def caller_context_scope(caller: Optional[CallerContext]):
    """Bind one endpoint-owned caller, explicitly clearing absent authority.

    Signal-dispatch tasks may be created during an authenticated turn and thus
    inherit its Python context. Setting ``None`` is intentional: an unattended
    wake must not inherit sovereign authority from the task that enqueued it.
    """

    with caller_context_lifetime(caller) as binding:
        with caller_context_binding_scope(binding):
            yield caller
