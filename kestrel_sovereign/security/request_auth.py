"""Request-owned interactive credentials for private response delivery.

The normal middleware selects the authentication lane. This service rechecks
that SAME lane through delivery; it never promotes a descriptor, URL or session
selector into credentials and never retries authentication with another lane.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import hmac
import math

from itsdangerous import BadSignature
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import HTTPConnection, Request

from kestrel_sovereign.auth import AuthMethod, CallerContext
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.csrf import enforce_csrf
from kestrel_sovereign.security.sovereign_key import sovereign_key_fingerprint
from kestrel_sovereign.turn_scope import turn_scoped


_SESSION_KEY = "kestrel.authenticated_session"
_CURRENT: ContextVar[InteractiveRequestAuthentication | None] = ContextVar(
    "kestrel_interactive_request_authentication", default=None)


@dataclass(frozen=True)
class _SignedSession:
    scope: dict = field(repr=False)
    session: dict = field(repr=False)
    credential: bytes = field(repr=False)
    middleware: SessionMiddleware = field(repr=False)
    identity: str
    expires_at: datetime | None

    def require_current(self, identity: str) -> None:
        if (self.scope.get("session") is not self.session
                or self.session.get("user_email") != self.identity or identity != self.identity):
            raise HostCapabilityAccessDenied("interactive session changed")
        try:
            self.middleware.signer.unsign(self.credential, max_age=self.middleware.max_age)
        except BadSignature as error:
            raise HostCapabilityAccessDenied("interactive session expired") from error


class AuthenticatedSessionMiddleware(SessionMiddleware):
    """Use Starlette's native session codec and retain its original expiry.

    Starlette still owns parsing, verification, mutation and cookie emission.
    Its public signer supplies the authenticated timestamp; no second cookie
    format, serializer or signing policy is introduced here.
    """

    def __init__(self, app, **settings):
        async def retain_session(scope, receive, send):
            session = scope.get("session")
            identity = session.get("user_email") if session is not None else None
            if scope["type"] == "http" and isinstance(identity, str) and identity:
                raw = HTTPConnection(scope).cookies[self.session_cookie].encode("utf-8")
                _, issued_at = self.signer.unsign(raw, max_age=self.max_age, return_timestamp=True)
                expires_at = (issued_at + timedelta(seconds=self.max_age)
                              if self.max_age is not None else None)
                scope[_SESSION_KEY] = _SignedSession(scope, session, raw, self, identity, expires_at)
            await app(scope, receive, send)
        super().__init__(retain_session, **settings)


@dataclass
class InteractiveRequestAuthentication:
    caller: CallerContext
    expires_at: datetime | None
    _check: Callable[[], None] = field(repr=False)
    _active: bool = field(default=True, repr=False)

    def require_current(self) -> None:
        if not self._active:
            raise HostCapabilityAccessDenied("interactive request ended")
        if self.expires_at is not None and datetime.now(UTC) >= self.expires_at:
            raise HostCapabilityAccessDenied("interactive credential expired")
        self._check()


class RequestAuthenticationService:
    """Revalidate an actual middleware-authenticated interactive request."""

    def __init__(self, *, api_key: Callable[[], str | None]):
        self._api_key = api_key

    @contextmanager
    def bind(self, request, *, require_csrf: bool = True):
        from kestrel_sovereign.endpoints.auth_oauth import _email_authorized, _verify_jwt

        caller = getattr(request.state, "caller", None)
        if (not isinstance(caller, CallerContext) or not isinstance(caller.identity, str)
                or not caller.identity.strip()):
            raise HostCapabilityAccessDenied("interactive authentication required")
        expires_at = None
        if caller.auth_method is AuthMethod.API_KEY:
            if not caller.is_sovereign or not caller.credential_fingerprint:
                raise HostCapabilityAccessDenied("original API credential required")

            def credential_current():
                key = self._api_key()
                if not key or not hmac.compare_digest(
                        caller.credential_fingerprint, sovereign_key_fingerprint(key)):
                    raise HostCapabilityAccessDenied("original API credential changed")
                bearer = request.headers.get("authorization", "")
                presented = (request.headers.get("x-api-key", ""),
                             bearer[7:] if bearer.startswith("Bearer ") else "")
                if not any(hmac.compare_digest(value.encode(), key.encode()) for value in presented):
                    raise HostCapabilityAccessDenied("explicit API credential required")
        elif caller.auth_method is AuthMethod.JWT:
            bearer = request.headers.get("authorization", "")
            token = bearer[7:] if bearer.startswith("Bearer ") else ""
            payload = _verify_jwt(token) if token else None
            expiry = payload.get("exp") if payload is not None else None
            if (isinstance(expiry, bool) or not isinstance(expiry, (int, float))
                    or not math.isfinite(expiry)):
                raise HostCapabilityAccessDenied("expiring JWT credential required")
            expires_at = datetime.fromtimestamp(expiry, UTC)

            def credential_current():
                current = _verify_jwt(token)
                if (current is None or current.get("sub") != caller.identity
                        or current.get("exp") != expiry or not _email_authorized(caller.identity)):
                    raise HostCapabilityAccessDenied("original JWT credential is unavailable")
        elif caller.auth_method is AuthMethod.OAUTH_SESSION:
            session = request.scope.get(_SESSION_KEY)
            if not isinstance(session, _SignedSession):
                raise HostCapabilityAccessDenied("original signed interactive session required")
            expires_at = session.expires_at
            if require_csrf:
                enforce_csrf(request, authed_via_cookie=True)

            def credential_current():
                if request.scope.get("session") is not session.session:
                    raise HostCapabilityAccessDenied("original interactive session changed")
                session.require_current(caller.identity)
                if not _email_authorized(caller.identity):
                    raise HostCapabilityAccessDenied("interactive user authorization changed")
        else:
            raise HostCapabilityAccessDenied("interactive authentication required")

        def check():
            if (getattr(request.state, "caller", None) is not caller
                    or getattr(request.app.state, "request_authentication", None) is not self):
                raise HostCapabilityAccessDenied("original request authentication changed")
            credential_current()

        binding = InteractiveRequestAuthentication(caller, expires_at, check)
        binding.require_current()
        try:
            yield binding
        finally:
            binding._active = False


def current_request_authentication() -> InteractiveRequestAuthentication | None:
    """The exact active HTTP owner, never a serialized caller descriptor."""
    binding = _CURRENT.get()
    if binding is not None:
        binding.require_current()
    return binding


@contextmanager
def _bind_request_authentication(binding):
    """Re-present the original owner's revocable grant on a foreign turn task."""
    if binding is not None and not isinstance(binding, InteractiveRequestAuthentication):
        raise TypeError("original interactive request binding required")
    token = _CURRENT.set(binding)
    try:
        yield
    finally:
        _CURRENT.reset(token)


turn_scoped("interactive_request_authentication", variables=(_CURRENT,),
            capture=lambda _agent: _CURRENT.get(), bind=_bind_request_authentication)


class InteractiveAuthenticationLifetimeMiddleware:
    """Retain initial publication's credential owner through ASGI production.

    Installed inside the normal authentication middleware. Private response
    controllers separately retain their own grant through the OUTER actual send;
    a buffering acknowledgement here is not proof of private delivery.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = _CURRENT.set(None)
        try:
            await self._http(scope, receive, send)
        finally:
            _CURRENT.reset(token)

    async def _http(self, scope, receive, send):
        request = Request(scope, receive)
        caller = getattr(request.state, "caller", None)
        # Public, native peer, Phoenix embed and EventSource credentials supply
        # no interactive grant. An extra header cannot promote another lane.
        explicit = request.headers.get("x-api-key") or request.headers.get("authorization")
        if (caller is None or caller.auth_method not in {
                AuthMethod.API_KEY, AuthMethod.JWT, AuthMethod.OAUTH_SESSION}
                or (caller.auth_method is AuthMethod.API_KEY
                    and (not explicit or not caller.credential_fingerprint))):
            await self.app(scope, receive, send)
            return
        service = getattr(request.app.state, "request_authentication", None)
        if not isinstance(service, RequestAuthenticationService):
            raise RuntimeError("host request authentication service is unavailable")
        with ExitStack() as owner:
            # Existing chat CSRF policy is unchanged. Each private View mutation
            # uses bind()'s default canonical CSRF check before its own body.
            try:
                binding = owner.enter_context(service.bind(request, require_csrf=False))
            except HostCapabilityAccessDenied:
                # This middleware supplies optional initial-publication
                # metadata. Private endpoints independently REQUIRE bind().
                # An unrelated header or unavailable private grant cannot
                # promote a query credential or relabel another endpoint.
                binding = None
            token = _CURRENT.set(binding)
            try:
                await self.app(scope, receive, send)
            finally:
                _CURRENT.reset(token)
