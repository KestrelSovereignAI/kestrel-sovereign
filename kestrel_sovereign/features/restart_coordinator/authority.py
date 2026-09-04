"""Durable sovereign and delegated authority for whole-host restarts.

RestartCoordinator is a host mutation surface, not a capability conferred by
being an agent, a peer, a task recipient, or the cause of a turn. A request is
accepted only for an endpoint-bound sovereign caller or for the exact subject
of a narrow sovereign-signed delegation. The host seals the request bounds and
every executor re-verifies both that seal and any delegation immediately before
update and restart boundaries.

Rotating ``KESTREL_API_KEY`` revokes pending evidence. Unsigned legacy rows and
rows whose immutable request fields were edited fail closed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from kestrel_sovereign.auth import current_caller_context
from kestrel_sovereign.security.sovereign_key import (
    is_ephemeral_sovereign_key,
    normalize_sovereign_api_key,
    sovereign_key_fingerprint,
)


AUTHORITY_KIND = "sovereign_api_key_hmac_v2"
AUTHORITY_VERSION = 2
_DOMAIN = b"kestrel/restart-authority/v2\x00"
DELEGATION_KIND = "sovereign_restart_delegation_v1"
DELEGATION_VERSION = 1
_DELEGATION_DOMAIN = b"kestrel/restart-delegation/v1\x00"
REVOCATION_KIND = "sovereign_restart_delegation_revocation_v1"
REVOCATION_VERSION = 1
_REVOCATION_DOMAIN = b"kestrel/restart-delegation-revocation/v1\x00"
_GENERATION_RE = re.compile(r"[0-9a-f]{32}")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class RestartDelegation:
    """One parsed, sovereign-signed, exact restart delegation."""

    delegation_id: str
    issuer: str
    subject_agent_did: str
    operation: str
    update_repo_path: str
    update_target_ref: str
    update_profile: str
    update_allow_migrations: bool
    issued_at: str
    expires_at: str
    evidence: str
    signature: str

    def to_public_dict(self) -> dict[str, Any]:
        """Return auditable bounds without returning replayable signed bytes."""

        return {
            "delegation_id": self.delegation_id,
            "issuer": self.issuer,
            "subject_agent_did": self.subject_agent_did,
            "operation": self.operation,
            "update_repo_path": self.update_repo_path,
            "update_target_ref": self.update_target_ref,
            "update_profile": self.update_profile,
            "update_allow_migrations": self.update_allow_migrations,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


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


def require_restart_request_authority() -> str:
    """Return the current sovereign actor after validating durable key custody."""

    caller = current_caller_context()
    if caller is None or caller.is_sovereign is not True:
        raise RestartAuthorityError(
            "whole-host restart requires an authenticated sovereign-key caller"
        )
    actor = caller.identity
    if not isinstance(actor, str) or not actor.strip():
        raise RestartAuthorityError("sovereign caller has no durable actor identity")
    # Validate the signing key before callers perform any update-path
    # inspection, then bind it to the credential the endpoint actually
    # authenticated. A request admitted under key A cannot mint authority under
    # newly rotated key B merely because its agent turn is still running.
    secret = _sovereign_secret()
    authenticated_fingerprint = caller.credential_fingerprint
    if not isinstance(authenticated_fingerprint, str) or not hmac.compare_digest(
        authenticated_fingerprint,
        sovereign_key_fingerprint(secret.decode("utf-8")),
    ):
        raise RestartAuthorityError(
            "whole-host restart authority no longer matches the authenticated "
            "credential at request entry"
        )
    return actor.strip()


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


def _domain_signature(document: Mapping[str, Any], domain: bytes) -> str:
    return hmac.new(
        _sovereign_secret(),
        domain + _canonical(document),
        hashlib.sha256,
    ).hexdigest()


def _signature(document: Mapping[str, Any]) -> str:
    return _domain_signature(document, _DOMAIN)


def issue_restart_delegation(
    *,
    subject_agent_did: str,
    operation: str,
    update_repo_path: str,
    update_target_ref: str,
    update_profile: str,
    update_allow_migrations: bool,
    issued_at: str,
    expires_at: str,
) -> tuple[str, str]:
    """Issue one narrow delegation for the authenticated sovereign caller."""

    issuer = require_restart_request_authority()
    document = {
        "version": DELEGATION_VERSION,
        "kind": DELEGATION_KIND,
        "delegation_id": secrets.token_hex(16),
        "issuer": issuer,
        "subject_agent_did": subject_agent_did,
        "operation": operation,
        "update_repo_path": update_repo_path,
        "update_target_ref": update_target_ref,
        "update_profile": update_profile,
        "update_allow_migrations": bool(update_allow_migrations),
        "issued_at": issued_at,
        "expires_at": expires_at,
    }
    evidence = _canonical(document).decode("utf-8")
    signature = _domain_signature(document, _DELEGATION_DOMAIN)
    parsed, reason = verify_restart_delegation(evidence, signature)
    if parsed is None:
        raise RestartAuthorityError(reason)
    return evidence, signature


def _aware_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RestartAuthorityError(f"restart delegation {field} is absent")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RestartAuthorityError(
            f"restart delegation {field} is malformed"
        ) from error
    if parsed.tzinfo is None:
        raise RestartAuthorityError(
            f"restart delegation {field} must include a timezone"
        )
    return parsed


def verify_restart_delegation(
    evidence: Any,
    signature: Any,
) -> tuple[RestartDelegation | None, str]:
    """Verify signed delegation bytes and return their exact typed bounds."""

    if not isinstance(evidence, str) or not evidence:
        return None, "restart delegation evidence is absent"
    if not isinstance(signature, str) or _SIGNATURE_RE.fullmatch(signature) is None:
        return None, "restart delegation signature is absent or malformed"
    try:
        document = json.loads(evidence)
    except (TypeError, ValueError):
        return None, "restart delegation evidence is malformed"
    if not isinstance(document, dict):
        return None, "restart delegation evidence is not an object"
    expected_fields = {
        "version",
        "kind",
        "delegation_id",
        "issuer",
        "subject_agent_did",
        "operation",
        "update_repo_path",
        "update_target_ref",
        "update_profile",
        "update_allow_migrations",
        "issued_at",
        "expires_at",
    }
    if set(document) != expected_fields:
        return None, "restart delegation evidence schema is unsupported"
    if document.get("version") != DELEGATION_VERSION:
        return None, "restart delegation evidence version is unsupported"
    if document.get("kind") != DELEGATION_KIND:
        return None, "restart delegation evidence kind is unsupported"
    delegation_id = document.get("delegation_id")
    if not isinstance(delegation_id, str) or _GENERATION_RE.fullmatch(
        delegation_id
    ) is None:
        return None, "restart delegation id is absent or malformed"
    for field in ("issuer", "subject_agent_did"):
        value = document.get(field)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            return None, f"restart delegation {field} is absent or malformed"
    operation = document.get("operation")
    if operation not in {"restart_only", "update_then_restart"}:
        return None, "restart delegation operation is unsupported"
    path = document.get("update_repo_path")
    target_ref = document.get("update_target_ref")
    profile = document.get("update_profile")
    allow_migrations = document.get("update_allow_migrations")
    if not all(isinstance(value, str) for value in (path, target_ref, profile)):
        return None, "restart delegation update bounds are malformed"
    if not isinstance(allow_migrations, bool):
        return None, "restart delegation migration bound is malformed"
    if operation == "restart_only" and any(
        (path, target_ref, profile, allow_migrations)
    ):
        return None, "restart-only delegation carries unauthorized update bounds"
    if operation == "update_then_restart" and not all((path, target_ref, profile)):
        return None, "update delegation bounds must be explicit"
    try:
        issued = _aware_datetime(document.get("issued_at"), "issued_at")
        expires = _aware_datetime(document.get("expires_at"), "expires_at")
        if expires <= issued:
            return None, "restart delegation expiry must follow issuance"
        expected_signature = _domain_signature(document, _DELEGATION_DOMAIN)
    except RestartAuthorityError as error:
        return None, str(error)
    if not hmac.compare_digest(signature, expected_signature):
        return None, "restart delegation signature verification failed"
    return RestartDelegation(
        delegation_id=delegation_id,
        issuer=document["issuer"],
        subject_agent_did=document["subject_agent_did"],
        operation=operation,
        update_repo_path=path,
        update_target_ref=target_ref,
        update_profile=profile,
        update_allow_migrations=allow_migrations,
        issued_at=document["issued_at"],
        expires_at=document["expires_at"],
        evidence=evidence,
        signature=signature,
    ), "verified sovereign-signed restart delegation"


def restart_delegation_allows(
    delegation: RestartDelegation,
    *,
    subject_agent_did: str,
    operation: str,
    update_repo_path: str,
    update_target_ref: str,
    update_profile: str,
    update_allow_migrations: bool,
) -> tuple[bool, str]:
    """Check exact subject and mutation bounds without widening aliases."""

    if not hmac.compare_digest(delegation.subject_agent_did, subject_agent_did):
        return False, "restart delegation subject does not match this agent"
    expected = (
        delegation.operation,
        delegation.update_repo_path,
        delegation.update_target_ref,
        delegation.update_profile,
        delegation.update_allow_migrations,
    )
    actual = (
        operation,
        update_repo_path,
        update_target_ref,
        update_profile,
        bool(update_allow_migrations),
    )
    if expected != actual:
        return False, "restart request exceeds its delegated operation bounds"
    return True, "restart request is within delegated bounds"


def issue_restart_delegation_revocation(
    *, delegation_id: str, revoked_at: str,
) -> tuple[str, str]:
    """Sign an immutable revocation receipt under the current sovereign key."""

    actor = require_restart_request_authority()
    document = {
        "version": REVOCATION_VERSION,
        "kind": REVOCATION_KIND,
        "delegation_id": delegation_id,
        "revoked_at": revoked_at,
        "revoked_by": actor,
    }
    evidence = _canonical(document).decode("utf-8")
    return evidence, _domain_signature(document, _REVOCATION_DOMAIN)


def verify_restart_delegation_revocation(
    evidence: Any,
    signature: Any,
    *,
    delegation_id: str,
) -> tuple[dict[str, str] | None, str]:
    """Verify one signed revocation receipt against its durable lookup key."""

    if not isinstance(evidence, str) or not evidence:
        return None, "restart delegation revocation evidence is absent"
    if not isinstance(signature, str) or _SIGNATURE_RE.fullmatch(signature) is None:
        return None, "restart delegation revocation signature is malformed"
    try:
        document = json.loads(evidence)
    except (TypeError, ValueError):
        return None, "restart delegation revocation evidence is malformed"
    if not isinstance(document, dict) or set(document) != {
        "version", "kind", "delegation_id", "revoked_at", "revoked_by",
    }:
        return None, "restart delegation revocation schema is unsupported"
    if (
        document.get("version") != REVOCATION_VERSION
        or document.get("kind") != REVOCATION_KIND
        or document.get("delegation_id") != delegation_id
    ):
        return None, "restart delegation revocation binding is invalid"
    revoked_by = document.get("revoked_by")
    if (
        not isinstance(revoked_by, str)
        or not revoked_by.strip()
        or revoked_by != revoked_by.strip()
    ):
        return None, "restart delegation revocation actor is malformed"
    try:
        _aware_datetime(document.get("revoked_at"), "revoked_at")
        expected = _domain_signature(document, _REVOCATION_DOMAIN)
    except RestartAuthorityError as error:
        return None, str(error)
    if not hmac.compare_digest(signature, expected):
        return None, "restart delegation revocation signature verification failed"
    return {
        "delegation_id": delegation_id,
        "revoked_at": document["revoked_at"],
        "revoked_by": revoked_by,
    }, "verified sovereign-signed restart delegation revocation"


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
    delegation: RestartDelegation | None = None,
) -> tuple[str, str]:
    """Seal exact bounds for a sovereign caller or verified delegation."""

    if delegation is None:
        actor = require_restart_request_authority()
        delegation_binding = None
    else:
        parsed, reason = verify_restart_delegation(
            delegation.evidence, delegation.signature
        )
        if parsed != delegation:
            raise RestartAuthorityError(
                "restart delegation object does not match its signed evidence"
            )
        allowed, reason = restart_delegation_allows(
            delegation,
            subject_agent_did=requested_by_agent,
            operation=operation,
            update_repo_path=update_repo_path,
            update_target_ref=update_target_ref,
            update_profile=update_profile,
            update_allow_migrations=update_allow_migrations,
        )
        if not allowed:
            raise RestartAuthorityError(reason)
        actor = delegation.subject_agent_did
        delegation_binding = {
            "delegation_id": delegation.delegation_id,
            "delegation_signature": delegation.signature,
        }
    document = {
        "version": AUTHORITY_VERSION,
        "kind": AUTHORITY_KIND,
        "actor": actor,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        # A seal authorizes exactly one lifecycle attempt. The store consumes
        # this unpredictable generation in a separate durable ledger before a
        # host mutation can begin; status edits to restart_requests cannot make
        # a consumed authorization executable again.
        "lifecycle_generation": secrets.token_hex(16),
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
    if delegation_binding is not None:
        document["delegation"] = delegation_binding
    evidence = _canonical(document).decode("utf-8")
    return evidence, _signature(document)


def restart_authority_evidence_generation(evidence: Any) -> str:
    """Return the structurally valid lifecycle generation in evidence bytes."""

    try:
        document = json.loads(evidence)
    except (TypeError, ValueError) as error:
        raise RestartAuthorityError(
            "restart authority evidence is malformed"
        ) from error
    if not isinstance(document, dict):
        raise RestartAuthorityError(
            "restart authority evidence is malformed"
        )
    generation = document.get("lifecycle_generation")
    if not isinstance(generation, str) or _GENERATION_RE.fullmatch(generation) is None:
        raise RestartAuthorityError(
            "restart authority lifecycle generation is absent or malformed"
        )
    return generation


def restart_authority_generation(request: Any) -> str:
    """Return the structurally valid lifecycle generation in one request."""

    return restart_authority_evidence_generation(
        getattr(request, "authority_evidence", "")
    )


def rotate_restart_authority_generation(request: Any) -> tuple[str, str]:
    """Reseal an authenticated retry onto a fresh single-use generation."""

    verified, reason = verify_restart_authority(request)
    if not verified:
        raise RestartAuthorityError(reason)
    document = json.loads(getattr(request, "authority_evidence"))
    document["lifecycle_generation"] = secrets.token_hex(16)
    document["lifecycle_reissued_at"] = datetime.now(timezone.utc).isoformat()
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
    if _SIGNATURE_RE.fullmatch(signature) is None:
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
    generation = document.get("lifecycle_generation")
    if not isinstance(generation, str) or _GENERATION_RE.fullmatch(generation) is None:
        return False, "restart authority lifecycle generation is absent or malformed"
    actor = document.get("actor")
    if not isinstance(actor, str) or not actor:
        return False, "restart authority actor is absent"
    delegation = document.get("delegation")
    if delegation is not None:
        if not isinstance(delegation, dict) or set(delegation) != {
            "delegation_id",
            "delegation_signature",
        }:
            return False, "restart authority delegation binding is malformed"
        delegation_id = delegation.get("delegation_id")
        delegation_signature = delegation.get("delegation_signature")
        if not isinstance(delegation_id, str) or _GENERATION_RE.fullmatch(
            delegation_id
        ) is None:
            return False, "restart authority delegation id is malformed"
        if not isinstance(
            delegation_signature, str
        ) or _SIGNATURE_RE.fullmatch(delegation_signature) is None:
            return False, "restart authority delegation signature is malformed"
        if actor != str(getattr(request, "requested_by_agent", "")):
            return False, "restart authority delegation actor is not the requester"

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
    return True, (
        "verified sovereign-signed delegated request seal"
        if delegation is not None
        else "verified sovereign-key authority"
    )


def restart_authority_delegation_binding(
    request: Any,
) -> tuple[str, str] | None:
    """Return the delegation id/signature bound into a verified request seal."""

    verified, reason = verify_restart_authority(request)
    if not verified:
        raise RestartAuthorityError(reason)
    document = json.loads(getattr(request, "authority_evidence"))
    binding = document.get("delegation")
    if binding is None:
        return None
    return binding["delegation_id"], binding["delegation_signature"]
