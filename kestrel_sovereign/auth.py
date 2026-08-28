"""Caller context for threading authentication identity into agent operations."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Optional


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

    @staticmethod
    def sovereign(auth_method: AuthMethod = AuthMethod.API_KEY, identity: str = None) -> "CallerContext":
        return CallerContext(role=CallerRole.SOVEREIGN, auth_method=auth_method, identity=identity or "api_key")

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


_current_caller_context: ContextVar[Optional[CallerContext]] = ContextVar(
    "kestrel_current_caller_context",
    default=None,
)


def current_caller_context() -> Optional[CallerContext]:
    """Return the authenticated caller bound to this exact invocation task."""

    return _current_caller_context.get()


@contextmanager
def caller_context_scope(caller: Optional[CallerContext]):
    """Bind one endpoint-owned caller, explicitly clearing absent authority.

    Signal-dispatch tasks may be created during an authenticated turn and thus
    inherit its Python context. Setting ``None`` is intentional: an unattended
    wake must not inherit sovereign authority from the task that enqueued it.
    """

    if caller is not None and not isinstance(caller, CallerContext):
        raise TypeError("caller context must be endpoint-owned CallerContext")
    token = _current_caller_context.set(caller)
    try:
        yield caller
    finally:
        _current_caller_context.reset(token)
