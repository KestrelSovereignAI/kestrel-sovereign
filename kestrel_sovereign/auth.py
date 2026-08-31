"""Caller context for threading authentication identity into agent operations."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
