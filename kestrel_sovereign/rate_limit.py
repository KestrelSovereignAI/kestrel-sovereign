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

STOP_ADMISSION_RATE_LIMIT_COUNT = 120
STOP_ADMISSION_RATE_LIMIT_WINDOW_SECONDS = 60
STOP_ADMISSION_RATE_LIMIT = f"{STOP_ADMISSION_RATE_LIMIT_COUNT}/minute"
STOP_ADMISSION_RATE_LIMIT_SCOPE = "durable-stop-admission"
_STOP_RATE_KEY_DOMAIN = b"kestrel:durable-stop-rate-limit:v1\0"

HOLD_ADMISSION_RATE_LIMIT_COUNT = 60
HOLD_ADMISSION_RATE_LIMIT_WINDOW_SECONDS = 60
HOLD_ADMISSION_RATE_LIMIT = f"{HOLD_ADMISSION_RATE_LIMIT_COUNT}/minute"
HOLD_ADMISSION_RATE_LIMIT_SCOPE = "durable-hold-admission"


def durable_control_plane_rate_limit_key(request: Request) -> str:
    """Return a private, per-caller key for durable control-plane admissions.

    Authentication middleware binds ``request.state.caller`` before endpoint
    dispatch. Prefer the credential fingerprint so aliases of one sovereign
    key share a bucket, and hash every fallback so rate-limit logs never repeat
    a principal identifier. A narrow network fallback protects standalone apps
    that deliberately mount a Stop router without authentication middleware.

    The derivation is the caller's identity, not one verb's semantics, so Stop
    and Hold share it. Their BUDGETS stay separate (each has its own limiter
    scope): a Hold storm must not spend the andon cord's admissions.
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


# The historical name of the shared derivation above, kept because callers
# outside this repository import it.
durable_stop_rate_limit_key = durable_control_plane_rate_limit_key

limiter = Limiter(key_func=get_remote_address)
stop_admission_rate_limit = limiter.shared_limit(
    STOP_ADMISSION_RATE_LIMIT,
    scope=STOP_ADMISSION_RATE_LIMIT_SCOPE,
    key_func=durable_control_plane_rate_limit_key,
)
hold_admission_rate_limit = limiter.shared_limit(
    HOLD_ADMISSION_RATE_LIMIT,
    scope=HOLD_ADMISSION_RATE_LIMIT_SCOPE,
    key_func=durable_control_plane_rate_limit_key,
)


__all__ = [
    "HOLD_ADMISSION_RATE_LIMIT",
    "HOLD_ADMISSION_RATE_LIMIT_COUNT",
    "HOLD_ADMISSION_RATE_LIMIT_SCOPE",
    "HOLD_ADMISSION_RATE_LIMIT_WINDOW_SECONDS",
    "STOP_ADMISSION_RATE_LIMIT",
    "STOP_ADMISSION_RATE_LIMIT_COUNT",
    "STOP_ADMISSION_RATE_LIMIT_SCOPE",
    "STOP_ADMISSION_RATE_LIMIT_WINDOW_SECONDS",
    "durable_control_plane_rate_limit_key",
    "durable_stop_rate_limit_key",
    "hold_admission_rate_limit",
    "limiter",
    "stop_admission_rate_limit",
]
