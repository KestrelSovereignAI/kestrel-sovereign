"""Durable sovereign authority evidence for whole-host restart requests.

RestartCoordinator is a host mutation surface, not a capability conferred by
being an agent, a peer, a task recipient, or the cause of a turn. A request is
accepted only while an endpoint-bound sovereign caller is present. The host
then seals the exact request bounds with the sovereign API key; every executor
re-verifies that seal immediately before update and restart boundaries.

Rotating ``KESTREL_API_KEY`` revokes pending evidence. Unsigned legacy rows and
rows whose immutable request fields were edited fail closed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from kestrel_sovereign.auth import current_caller_context
from kestrel_sovereign.security.sovereign_key import (
    is_ephemeral_sovereign_key,
    normalize_sovereign_api_key,
)


AUTHORITY_KIND = "sovereign_api_key_hmac_v1"
AUTHORITY_VERSION = 1
_DOMAIN = b"kestrel/restart-authority/v1\x00"


class RestartAuthorityError(ValueError):
    """A whole-host request lacks verifiable sovereign authority."""


def _sovereign_secret() -> bytes:
    raw = normalize_sovereign_api_key(os.environ.get("KESTREL_API_KEY") or "")
    if not raw:
        raise RestartAuthorityError(
            "whole-host restart authority is unavailable: no stable sovereign key"
        )
    try:
        if is_ephemeral_sovereign_key(raw):
            raise RestartAuthorityError(
                "whole-host restart authority is unavailable: the server generated "
                "a temporary sovereign key; configure a stable KESTREL_API_KEY"
            )
        return raw.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RestartAuthorityError(
            "whole-host restart authority is unavailable: the sovereign key is "
            "not valid UTF-8"
        ) from error


def _request_claims(
    *,
    request_id: str,
    requested_by_agent: str,
    reason: str,
    urgency: str,
    policy: str,
    desired_window: str,
    operation: str,
    update_repo_path: str,
    update_target_ref: str,
    update_profile: str,
    update_allow_migrations: bool,
    requester_request_id: str,
    origin_session_id: str,
    requested_at: str,
    first_blocked_at: str,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "requested_by_agent": requested_by_agent,
        "reason": reason,
        "urgency": urgency,
        "policy": policy,
        "desired_window": desired_window,
        "operation": operation,
        "update_repo_path": update_repo_path,
        "update_target_ref": update_target_ref,
        "update_profile": update_profile,
        "update_allow_migrations": bool(update_allow_migrations),
        "requester_request_id": requester_request_id,
        "origin_session_id": origin_session_id,
        "requested_at": requested_at,
        "first_blocked_at": first_blocked_at,
    }


def _canonical(document: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except UnicodeEncodeError as error:
        raise RestartAuthorityError(
            "restart authority evidence is not valid UTF-8"
        ) from error


def _signature(document: Mapping[str, Any]) -> str:
    return hmac.new(
        _sovereign_secret(),
        _DOMAIN + _canonical(document),
        hashlib.sha256,
    ).hexdigest()


def issue_restart_authority(
    *,
    request_id: str,
    requested_by_agent: str,
    reason: str,
    urgency: str,
    policy: str,
    desired_window: str,
    operation: str,
    update_repo_path: str,
    update_target_ref: str,
    update_profile: str,
    update_allow_migrations: bool,
    requester_request_id: str,
    origin_session_id: str,
    requested_at: str,
    first_blocked_at: str = "",
) -> tuple[str, str]:
    """Seal exact request bounds for the current sovereign caller."""

    caller = current_caller_context()
    if caller is None or caller.is_sovereign is not True:
        raise RestartAuthorityError(
            "whole-host restart requires an authenticated sovereign-key caller"
        )
    actor = caller.identity
    if not isinstance(actor, str) or not actor.strip():
        raise RestartAuthorityError("sovereign caller has no durable actor identity")
    document = {
        "version": AUTHORITY_VERSION,
        "kind": AUTHORITY_KIND,
        "actor": actor.strip(),
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "request": _request_claims(
            request_id=request_id,
            requested_by_agent=requested_by_agent,
            reason=reason,
            urgency=urgency,
            policy=policy,
            desired_window=desired_window,
            operation=operation,
            update_repo_path=update_repo_path,
            update_target_ref=update_target_ref,
            update_profile=update_profile,
            update_allow_migrations=update_allow_migrations,
            requester_request_id=requester_request_id,
            origin_session_id=origin_session_id,
            requested_at=requested_at,
            first_blocked_at=first_blocked_at,
        ),
    }
    evidence = _canonical(document).decode("utf-8")
    return evidence, _signature(document)


def reseal_restart_safety_state(
    request: Any,
    *,
    first_blocked_at: str,
) -> tuple[str, str]:
    """Authenticate a host-owned deferral-clock transition.

    This is deliberately not a new request-authority door: the existing seal
    must verify first, and every immutable request claim and sovereign actor is
    preserved. The coordinator uses it only while atomically changing the
    safety timestamp whose age may release an idle-only gate.
    """

    verified, reason = verify_restart_authority(request)
    if not verified:
        raise RestartAuthorityError(reason)
    document = json.loads(getattr(request, "authority_evidence"))
    document["safety_state_updated_at"] = datetime.now(timezone.utc).isoformat()
    document["request"] = _request_claims(
        request_id=str(getattr(request, "id", "")),
        requested_by_agent=str(getattr(request, "requested_by_agent", "")),
        reason=str(getattr(request, "reason", "")),
        urgency=str(getattr(request, "urgency", "")),
        policy=str(getattr(request, "policy", "")),
        desired_window=str(getattr(request, "desired_window", "")),
        operation=str(getattr(request, "operation", "")),
        update_repo_path=str(getattr(request, "update_repo_path", "")),
        update_target_ref=str(getattr(request, "update_target_ref", "")),
        update_profile=str(getattr(request, "update_profile", "")),
        update_allow_migrations=bool(
            getattr(request, "update_allow_migrations", False)
        ),
        requester_request_id=str(getattr(request, "requester_request_id", "")),
        origin_session_id=str(getattr(request, "origin_session_id", "")),
        requested_at=str(getattr(request, "requested_at", "")),
        first_blocked_at=first_blocked_at,
    )
    evidence = _canonical(document).decode("utf-8")
    return evidence, _signature(document)


def verify_restart_authority(request: Any) -> tuple[bool, str]:
    """Re-verify one durable row against its exact fields and current key."""

    evidence = getattr(request, "authority_evidence", "")
    signature = getattr(request, "authority_signature", "")
    if not isinstance(evidence, str) or not evidence:
        return False, "unsigned legacy restart request"
    if not isinstance(signature, str) or not signature:
        return False, "restart authority signature is absent"
    if re.fullmatch(r"[0-9a-f]{64}", signature) is None:
        return False, "restart authority signature is malformed"
    try:
        document = json.loads(evidence)
    except (TypeError, ValueError):
        return False, "restart authority evidence is malformed"
    if not isinstance(document, dict):
        return False, "restart authority evidence is not an object"
    if document.get("version") != AUTHORITY_VERSION:
        return False, "restart authority evidence version is unsupported"
    if document.get("kind") != AUTHORITY_KIND:
        return False, "restart authority kind is not sovereign-key authority"
    actor = document.get("actor")
    if not isinstance(actor, str) or not actor:
        return False, "restart authority actor is absent"

    expected_claims = _request_claims(
        request_id=str(getattr(request, "id", "")),
        requested_by_agent=str(getattr(request, "requested_by_agent", "")),
        reason=str(getattr(request, "reason", "")),
        urgency=str(getattr(request, "urgency", "")),
        policy=str(getattr(request, "policy", "")),
        desired_window=str(getattr(request, "desired_window", "")),
        operation=str(getattr(request, "operation", "")),
        update_repo_path=str(getattr(request, "update_repo_path", "")),
        update_target_ref=str(getattr(request, "update_target_ref", "")),
        update_profile=str(getattr(request, "update_profile", "")),
        update_allow_migrations=bool(
            getattr(request, "update_allow_migrations", False)
        ),
        requester_request_id=str(getattr(request, "requester_request_id", "")),
        origin_session_id=str(getattr(request, "origin_session_id", "")),
        requested_at=str(getattr(request, "requested_at", "")),
        first_blocked_at=str(getattr(request, "first_blocked_at", "")),
    )
    if document.get("request") != expected_claims:
        return False, "restart request fields do not match signed authority bounds"
    try:
        expected_signature = _signature(document)
    except RestartAuthorityError as error:
        return False, str(error)
    if not hmac.compare_digest(signature, expected_signature):
        return False, "restart authority signature verification failed"
    return True, "verified sovereign-key authority"
