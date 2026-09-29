"""Durable authority for whole-host restarts.

A request is sealed on one of three bases:

* an endpoint-bound **sovereign-key caller** — any bounds;
* the exact subject of a narrow **sovereign-signed delegation** — the
  delegation's bounds;
* an **agent request** (#3339) — an agent may ask for a whole-host restart
  from its own work (a scheduler wake, a signal, any autonomous turn), but
  only inside the *agent-requestable bounds*: a plain restart, or the default
  update profile landing the default Sovereign checkout's default branch with
  no migrations. The coordinator's idle/timeout gate is the control for these;
  anything wider still needs a sovereign caller or a delegation.

The host seals the request bounds (and, for an agent request, the basis and
the requesting agent as actor) under the sovereign key, and every executor
re-verifies that seal, any delegation, and an agent request's bounds
immediately before update and restart boundaries.

Rotating ``KESTREL_API_KEY`` revokes pending evidence. Unsigned legacy rows and
rows whose immutable request fields were edited fail closed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from kestrel_sovereign.security.host_authority import (
    HostAuthorityError,
    require_sovereign_caller,
    stable_sovereign_secret,
)

from . import update_profiles


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

# The seal's ``basis`` for a request an agent filed with no sovereign caller
# and no delegation (#3339). Sovereign-caller and delegated seals carry no
# ``basis`` field, exactly as before, so every existing row verifies unchanged.
AGENT_REQUEST_BASIS = "agent_request"

# Policies an agent may choose for its own request. Every one of them leaves
# execution to the coordinator's idle/timeout gate (``manual_only`` never
# auto-executes at all).
AGENT_REQUESTABLE_POLICIES = frozenset(
    {"idle_agents_only", "allow_busy_after_timeout", "manual_only"}
)


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


_OPERATION = "whole-host restart"


def _sovereign_secret() -> bytes:
    try:
        return stable_sovereign_secret(_OPERATION)
    except HostAuthorityError as error:
        raise RestartAuthorityError(str(error)) from error


def require_restart_request_authority() -> str:
    """Return the current sovereign actor after validating durable key custody.

    The predicate itself lives in :mod:`kestrel_sovereign.security.host_authority`
    since #3221/#3223, shared with the model service and fleet deployment; this
    keeps the restart-specific error type and messages callers already handle.
    """
    try:
        return require_sovereign_caller(_OPERATION)
    except HostAuthorityError as error:
        raise RestartAuthorityError(str(error)) from error


def _canonical_path(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return ""


def agent_request_bounds_violation(
    *,
    operation: str,
    policy: str,
    update_repo_path: str,
    update_target_ref: str,
    update_profile: str,
    update_allow_migrations: bool,
    fresh: bool = False,
) -> tuple[str, str] | None:
    """Return ``(bound, detail)`` for the first exceeded agent bound, else None.

    The agent-requestable bounds (#3339): ``restart_only`` with no update
    fields, or ``update_then_restart`` with a known update profile, the
    default Sovereign checkout, that checkout's default branch (``origin/HEAD``)
    as a branch no tag shadows, and no migrations; any policy the coordinator
    gates. The inputs are the values a durable row stores, so the same
    predicate serves request-time screening, sealing, and executor
    re-verification. It never inspects a caller-supplied path: the only
    filesystem it reads is the default checkout. ``fresh=True`` bypasses the
    short git-answer cache; the update boundary uses it.
    """

    if policy not in AGENT_REQUESTABLE_POLICIES:
        return "policy", f"policy {policy!r} is not agent-requestable"
    if bool(update_allow_migrations):
        return "allow_migrations", "an agent request cannot allow migrations"
    if operation == "restart_only":
        if update_repo_path or update_target_ref or update_profile:
            return "operation", "a restart_only request carries update bounds"
        return None
    if operation != "update_then_restart":
        return "operation", f"operation {operation!r} is not agent-requestable"
    if update_profile not in update_profiles.KNOWN_UPDATE_PROFILES:
        return "update_profile", (
            f"update_profile {update_profile!r} is not a known update profile"
        )
    default_repo = update_profiles.default_sovereign_repo_path()
    default_repo = _canonical_path(default_repo) if default_repo else ""
    if not default_repo:
        return "repo_path", "this host has no default Sovereign checkout"
    if update_repo_path != default_repo:
        return "repo_path", "repo_path is not the default Sovereign checkout"
    default_branch = update_profiles.checkout_default_branch(
        default_repo, fresh=fresh,
    )
    if not default_branch:
        return "target_ref", (
            "the default Sovereign checkout's default branch is unknown "
            "(origin/HEAD is not set)"
        )
    if update_target_ref != default_branch:
        return "target_ref", (
            "target_ref must be the default Sovereign checkout's default "
            f"branch {default_branch!r}; got {update_target_ref!r}"
        )
    # The profile runs ``git fetch origin <name>`` and lands on FETCH_HEAD;
    # when a tag and a branch share the name, git fetches the TAG. An agent
    # request is bound to the branch, so a same-named tag refuses it.
    shadowing_tag = update_profiles.checkout_has_tag(
        default_repo, default_branch, fresh=fresh,
    )
    if shadowing_tag is None:
        return "target_ref", (
            "could not read the default Sovereign checkout's tag namespace "
            f"to confirm no tag shadows branch {default_branch!r}"
        )
    if shadowing_tag:
        return "target_ref", (
            f"a tag named {default_branch!r} exists beside the default branch; "
            "git fetch would land on the tag, not the branch"
        )
    return None


def is_agent_request_seal(request: Any) -> bool:
    """Whether a row's evidence is sealed on the agent-request basis.

    Structural only; callers use it after ``verify_restart_authority``.
    """

    try:
        document = json.loads(getattr(request, "authority_evidence", ""))
    except (TypeError, ValueError):
        return False
    return isinstance(document, dict) and (
        document.get("basis") == AGENT_REQUEST_BASIS
    )


def agent_update_boundary_violation(request: Any) -> tuple[str, str] | None:
    """Re-check an agent-requested update's bounds with no cached answers.

    For the executor, immediately before the update profile runs: the
    per-tick verification may reuse a git answer for a few seconds, this
    re-reads ``origin/HEAD`` and the local tag namespace.
    """

    return agent_request_bounds_violation(
        operation=str(getattr(request, "operation", "")),
        policy=str(getattr(request, "policy", "")),
        update_repo_path=str(getattr(request, "update_repo_path", "")),
        update_target_ref=str(getattr(request, "update_target_ref", "")),
        update_profile=str(getattr(request, "update_profile", "")),
        update_allow_migrations=bool(
            getattr(request, "update_allow_migrations", False)
        ),
        fresh=True,
    )


def agent_request_refusal(bound: str, detail: str) -> str:
    """The refusal for an agent request outside the agent-requestable bounds."""

    return (
        f"restart request exceeds the agent-requestable bound on {bound}: "
        f"{detail}; a request beyond those bounds requires an authenticated "
        "sovereign-key caller or a sovereign-signed restart delegation"
    )


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
    allow_agent_request: bool = False,
) -> tuple[str, str]:
    """Seal exact bounds for a sovereign caller, delegation, or agent request.

    ``allow_agent_request`` lets a request with neither a sovereign caller nor
    a delegation be sealed on the agent-request basis, but only inside the
    agent-requestable bounds; its actor is ``requested_by_agent``, the
    filing tool's own scoped DID, never a caller-supplied value.
    """

    basis = None
    if delegation is None:
        delegation_binding = None
        try:
            actor = require_restart_request_authority()
        except RestartAuthorityError:
            if not allow_agent_request:
                raise
            exceeded = agent_request_bounds_violation(
                operation=operation,
                policy=policy,
                update_repo_path=update_repo_path,
                update_target_ref=update_target_ref,
                update_profile=update_profile,
                update_allow_migrations=update_allow_migrations,
            )
            if exceeded is not None:
                raise RestartAuthorityError(
                    agent_request_refusal(*exceeded)
                ) from None
            if not isinstance(requested_by_agent, str) or not (
                requested_by_agent.strip()
            ):
                raise RestartAuthorityError(
                    "agent restart request has no durable requesting agent"
                ) from None
            actor = requested_by_agent
            basis = AGENT_REQUEST_BASIS
    else:
        parsed, _verification_reason = verify_restart_delegation(
            delegation.evidence, delegation.signature
        )
        if parsed != delegation:
            raise RestartAuthorityError(
                "restart delegation object does not match its signed evidence"
            )
        allowed, delegation_reason = restart_delegation_allows(
            delegation,
            subject_agent_did=requested_by_agent,
            operation=operation,
            update_repo_path=update_repo_path,
            update_target_ref=update_target_ref,
            update_profile=update_profile,
            update_allow_migrations=update_allow_migrations,
        )
        if not allowed:
            raise RestartAuthorityError(delegation_reason)
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
    if basis is not None:
        # Signed with everything else: an editor can neither add, drop, nor
        # change the basis without failing the HMAC.
        document["basis"] = basis
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
    must verify first, and every immutable request claim, the actor, and the
    authority basis are preserved. The coordinator uses it only while
    atomically changing the safety timestamp whose age may release an
    idle-only gate.
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
    agent_request = "basis" in document
    if agent_request:
        if document.get("basis") != AGENT_REQUEST_BASIS:
            return False, "restart authority basis is unsupported"
        if delegation is not None:
            return False, "agent-request restart authority carries a delegation"
        if actor != str(getattr(request, "requested_by_agent", "")):
            return False, "agent-request restart authority actor is not the requester"

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
    if agent_request:
        # The seal proves what the agent asked for; it does not make those
        # bounds agent-requestable. Re-check the durable row against the
        # bounds as they stand now (default checkout, its default branch).
        exceeded = agent_request_bounds_violation(
            operation=expected_claims["operation"],
            policy=expected_claims["policy"],
            update_repo_path=expected_claims["update_repo_path"],
            update_target_ref=expected_claims["update_target_ref"],
            update_profile=expected_claims["update_profile"],
            update_allow_migrations=expected_claims["update_allow_migrations"],
        )
        if exceeded is not None:
            bound, detail = exceeded
            return False, (
                "agent-requested restart is outside the agent-requestable "
                f"bound on {bound}: {detail}"
            )
        return True, "verified agent-requested restart within agent bounds"
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
