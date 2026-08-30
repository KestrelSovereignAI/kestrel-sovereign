"""Shared rate limiter instance for the Kestrel server.

This module exists to avoid circular imports between server.py and endpoint
modules.  Both import the same ``limiter`` singleton so that ``@limiter.limit``
decorators in router files work correctly with the SlowAPI middleware
registered in ``server.py``.
"""

import hashlib

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


STOP_ADMISSION_RATE_LIMIT = "60/minute"
_STOP_RATE_KEY_DOMAIN = b"kestrel:durable-stop-rate-limit:v1\0"


def durable_stop_rate_limit_key(request: Request) -> str:
    """Return a private, per-caller key for durable Stop admissions.

    Authentication middleware binds ``request.state.caller`` before endpoint
    dispatch. Prefer the credential fingerprint so aliases of one sovereign
    key share a bucket, and hash every fallback so rate-limit logs never repeat
    a principal identifier. A narrow network fallback protects standalone apps
    that deliberately mount a Stop router without authentication middleware.
    """

    caller = getattr(request.state, "caller", None)
    credential_fingerprint = getattr(caller, "credential_fingerprint", None)
    identity = getattr(caller, "identity", None)
    role = getattr(getattr(caller, "role", None), "value", None)
    auth_method = getattr(getattr(caller, "auth_method", None), "value", None)
    if (
        isinstance(credential_fingerprint, str)
        and credential_fingerprint.strip()
    ):
        principal = f"credential:{credential_fingerprint}"
    elif isinstance(identity, str) and identity.strip():
        principal = f"identity:{role}:{auth_method}:{identity}"
    else:
        principal = f"transport:{get_remote_address(request)}"
    payload = principal.encode("utf-8", errors="surrogatepass")
    return "stop-principal:" + hashlib.sha256(
        _STOP_RATE_KEY_DOMAIN + payload
    ).hexdigest()

limiter = Limiter(key_func=get_remote_address)


__all__ = [
    "STOP_ADMISSION_RATE_LIMIT",
    "durable_stop_rate_limit_key",
    "limiter",
]
