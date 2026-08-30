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
    dispatch. Hash the identity so SlowAPI's exceeded-limit log never repeats a
    principal identifier. A narrow network fallback retains protection for
    standalone/library apps that deliberately mount the router without the
    production authentication middleware. The app namespace keeps independent
    in-process hosts and test applications from sharing one accidental bucket.
    """

    caller = getattr(request.state, "caller", None)
    identity = getattr(caller, "identity", None)
    auth_method = getattr(getattr(caller, "auth_method", None), "value", "")
    if isinstance(identity, str) and identity.strip():
        principal = f"caller:{auth_method}:{identity}"
    else:
        principal = f"network:{get_remote_address(request)}"
    payload = (
        f"app:{id(request.app)}\0{principal}"
    ).encode("utf-8", errors="surrogatepass")
    return "sha256:" + hashlib.sha256(
        _STOP_RATE_KEY_DOMAIN + payload
    ).hexdigest()

limiter = Limiter(key_func=get_remote_address)


__all__ = [
    "STOP_ADMISSION_RATE_LIMIT",
    "durable_stop_rate_limit_key",
    "limiter",
]
