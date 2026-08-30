"""Shared authorization helpers for sovereign host-control doors."""

from __future__ import annotations

from fastapi import Request

from kestrel_sovereign.api_errors import ApiHTTPException
from kestrel_sovereign.auth import CallerContext


def require_sovereign_actor(request: Request, *, action: str) -> str:
    """Return the authenticated sovereign actor or refuse before side effects."""

    caller = getattr(request.state, "caller", None)
    if not isinstance(caller, CallerContext) or not caller.is_sovereign:
        raise ApiHTTPException(
            status_code=403,
            code="sovereign_authority_required",
            message=f"{action} requires sovereign authority.",
        )
    identity = caller.identity
    return identity if isinstance(identity, str) and identity.strip() else "api_key"


__all__ = ["require_sovereign_actor"]
