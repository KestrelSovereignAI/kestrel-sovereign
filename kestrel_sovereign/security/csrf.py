"""Double-submit-cookie CSRF protection (#2293).

The host serves state-changing endpoints that authenticate via the OAuth
**session cookie** (browsers auto-attach cookies), which makes them
CSRF-susceptible: a malicious cross-origin page can trigger a state-changing
request and the browser will attach the victim's session cookie. The current
posture is ``SameSite=lax`` only, which is not sufficient for top-level POST
navigations and provides no defense on older browsers.

This module implements the **double-submit cookie** pattern, chosen (Q1 of the
issue) because it is *stateless* — no server-side session store to plumb:

* A non-``HttpOnly`` CSRF cookie (:data:`CSRF_COOKIE_NAME`) carries a random
  token. Because it is readable by JavaScript, the console can echo it back in
  the :data:`CSRF_HEADER_NAME` header on state-changing requests.
* A state-changing request authenticated by the session cookie must present a
  ``X-CSRF-Token`` header whose value matches the cookie. A cross-origin
  attacker can cause the cookie to be *sent* but cannot *read* it (same-origin
  policy) and so cannot set the matching header.

**API-key / machine callers are exempt.** An ``X-API-Key`` / bearer request is
not CSRF-susceptible — a browser never auto-attaches those headers, so a
cross-site page cannot forge such a request. Requiring a CSRF token from them
would only break legitimate scripted clients.

This helper is deliberately transport-agnostic and reusable: the host wires it
today (:mod:`kestrel_sovereign.host`), and per-agent routes can adopt the same
:func:`enforce_csrf` / :func:`issue_csrf_cookie` primitives later without
re-implementing the check.
"""

from __future__ import annotations

import os
import secrets
from typing import Optional
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import Response

#: Name of the non-HttpOnly cookie carrying the CSRF token (readable by JS so
#: the console can echo it back in the header).
CSRF_COOKIE_NAME = "kestrel_csrf"

#: Header the client must echo the cookie value in on state-changing requests.
CSRF_HEADER_NAME = "X-CSRF-Token"

#: HTTP methods that mutate state and therefore require a CSRF token when the
#: caller authenticated via the session cookie. Safe (read-only) methods are
#: exempt per RFC 7231.
STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Bytes of entropy for a minted token.
_TOKEN_BYTES = 32


class CSRFError(Exception):
    """Raised by :func:`enforce_csrf` when the CSRF check fails.

    The caller (middleware / dependency) maps this to an HTTP 403 with
    :attr:`detail`.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def generate_csrf_token() -> str:
    """Return a fresh, URL-safe CSRF token."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def _is_production() -> bool:
    return os.environ.get("KESTREL_ENV", "development") == "production"


def ensure_csrf_cookie(request: Request, response: Response) -> str:
    """Ensure the response carries a CSRF cookie, returning its token.

    If the request already presents a valid-looking CSRF cookie, it is reused
    (so a page's in-flight token stays stable); otherwise a new token is minted
    and set. Called on safe requests (e.g. serving the console / a bootstrap
    endpoint) so the browser has a token to echo back later.
    """
    existing = request.cookies.get(CSRF_COOKIE_NAME)
    token = existing if existing else generate_csrf_token()
    issue_csrf_cookie(response, token)
    return token


def issue_csrf_cookie(response: Response, token: Optional[str] = None) -> str:
    """Attach a CSRF cookie to ``response``; return the token written.

    The cookie is intentionally **not** ``HttpOnly`` — the double-submit
    pattern requires client JS to read it and echo it in the header. It is
    ``SameSite=lax`` and ``Secure`` in production, matching the session cookie.
    """
    token = token or generate_csrf_token()
    response.set_cookie(
        CSRF_COOKIE_NAME,
        token,
        max_age=7 * 24 * 3600,
        samesite="lax",
        httponly=False,
        secure=_is_production(),
        path="/",
    )
    return token


def _origin_matches(request: Request) -> bool:
    """Best-effort Origin/Referer sameness check (defense-in-depth).

    Returns ``True`` when there is no Origin/Referer to check (non-browser
    client) or when it matches the request host; ``False`` only when a header
    is present and its host differs. This never *grants* access on its own — it
    is an additional gate layered under the token check.
    """
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return True
    try:
        origin_host = urlparse(origin).netloc.split("@")[-1]
    except Exception:  # noqa: BLE001 - malformed header → treat as mismatch
        return False
    if not origin_host:
        return False
    request_host = request.headers.get("host", "")
    return secrets.compare_digest(origin_host, request_host)


def enforce_csrf(request: Request, *, authed_via_cookie: bool) -> None:
    """Raise :class:`CSRFError` if this request fails the CSRF check.

    Args:
        request: The incoming request.
        authed_via_cookie: Whether the request authenticated via the session
            cookie (as opposed to an API key / bearer token). Only cookie-authed
            state-changing requests are CSRF-checked; API-key/machine callers
            are exempt (they are not CSRF-susceptible — see module docstring).

    Enforcement rules:
        * Safe methods (GET/HEAD/OPTIONS/TRACE) are never checked.
        * API-key/machine callers are exempt.
        * A cookie-authed state-changing request must present a
          ``X-CSRF-Token`` header equal to the ``kestrel_csrf`` cookie.
    """
    if request.method not in STATE_CHANGING_METHODS:
        return
    if not authed_via_cookie:
        # Machine caller (API key / bearer). Not CSRF-susceptible.
        return

    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get(CSRF_HEADER_NAME)
    if not cookie_token or not header_token:
        raise CSRFError("Missing CSRF token")
    if not secrets.compare_digest(cookie_token, header_token):
        raise CSRFError("CSRF token mismatch")
    # Defense-in-depth: reject an obvious cross-origin submission even if the
    # token somehow matched (e.g. a subdomain leak). Absent Origin/Referer is
    # allowed (non-browser clients don't send them).
    if not _origin_matches(request):
        raise CSRFError("Origin/Referer mismatch")


__all__ = [
    "CSRF_COOKIE_NAME",
    "CSRF_HEADER_NAME",
    "STATE_CHANGING_METHODS",
    "CSRFError",
    "generate_csrf_token",
    "issue_csrf_cookie",
    "ensure_csrf_cookie",
    "enforce_csrf",
]
