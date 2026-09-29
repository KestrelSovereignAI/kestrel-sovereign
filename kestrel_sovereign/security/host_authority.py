"""Sovereign authority for mutations of shared host resources.

Some things an agent's tool can reach are not the agent's: the Ollama
daemon every co-hosted agent loads models from, the Cloud Run fleet a
multi-agent profile deploys, the host's restart. Being an agent, being
routed to, being asked by an authenticated user, or having a tool's ASK
promoted to AUTO is operational consent; none of it is authority over
the host (the two-axis doctrine, #3138; the audit, #3143).

The predicate here is the one the whole-host restart already enforced
(#3148), lifted out so the model service (#3221) and fleet deployment
(#3223) ask the same question rather than a paraphrase of it: the turn
must carry an endpoint-bound **sovereign-key** caller, and the credential
that caller authenticated with must still be the host's stable sovereign
key. A turn admitted under key A cannot mint authority under a rotated key
B just because it is still running; a scheduler wake, a signal, an A2A
peer, or an OAuth user carries no sovereign caller and is refused.
"""

from __future__ import annotations

import hmac
import os

from kestrel_sovereign.auth import current_caller_context
from kestrel_sovereign.security.sovereign_key import (
    is_ephemeral_sovereign_key,
    normalize_sovereign_api_key,
    sovereign_key_fingerprint,
)


class HostAuthorityError(ValueError):
    """A host-scoped mutation was asked for without verifiable sovereign authority."""


def stable_sovereign_secret(operation: str) -> bytes:
    """The host's configured, non-ephemeral sovereign key, or raise.

    ``operation`` names the mutation in the message ("whole-host restart",
    "shared local model installation") so a refusal says what was refused.
    """
    raw = normalize_sovereign_api_key(os.environ.get("KESTREL_API_KEY") or "")
    if not raw:
        raise HostAuthorityError(
            f"{operation} authority is unavailable: no stable sovereign key"
        )
    try:
        if is_ephemeral_sovereign_key(raw):
            raise HostAuthorityError(
                f"{operation} authority is unavailable: the server generated "
                "a temporary sovereign key; configure a stable KESTREL_API_KEY"
            )
        return raw.encode("utf-8")
    except UnicodeEncodeError as error:
        raise HostAuthorityError(
            f"{operation} authority is unavailable: the sovereign key is "
            "not valid UTF-8"
        ) from error


def require_sovereign_caller(operation: str) -> str:
    """Return the sovereign actor identity for ``operation``, or raise.

    Reads the endpoint-bound caller of the current turn. Nothing about the
    agent, its consent state, or the request path substitutes for it.
    """
    caller = current_caller_context()
    if caller is None or caller.is_sovereign is not True:
        raise HostAuthorityError(
            f"{operation} requires an authenticated sovereign-key caller"
        )
    actor = caller.identity
    if not isinstance(actor, str) or not actor.strip():
        raise HostAuthorityError("sovereign caller has no durable actor identity")
    # Validate the signing key before any further inspection, then bind it
    # to the credential the endpoint actually authenticated.
    secret = stable_sovereign_secret(operation)
    authenticated_fingerprint = caller.credential_fingerprint
    if not isinstance(authenticated_fingerprint, str) or not hmac.compare_digest(
        authenticated_fingerprint,
        sovereign_key_fingerprint(secret.decode("utf-8")),
    ):
        raise HostAuthorityError(
            f"{operation} authority no longer matches the authenticated "
            "credential at request entry"
        )
    return actor.strip()
