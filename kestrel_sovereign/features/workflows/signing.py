"""Workflow definition + transition signing helpers.

Implements the small surface called out in design §6 Phase 0:

> DID signing + verification helpers (reuse existing ``kestrel-talon``
> IdentityPackage primitives)

The ceremony-grade signing primitives in
:mod:`kestrel_sovereign.identity.signing` are package-specific to
``AgentIdentityPackage``. Workflow specs and stage-link transitions
need a thinner, format-agnostic surface, so this module wires the
existing :mod:`kestrel_sovereign.security.crypto_suite` and
:class:`kestrel_sovereign.identity.runtime_identity.AgentIdentity`
primitives directly.

This module ships:

- :func:`sign_workflow_spec` — populate ``spec.spec_hash`` and
  ``author_sig``. Legacy agents use ECDSA secp256k1; hybrid agents can
  opt into a ``hybrid:`` Ed25519 + ML-DSA-65 signature bundle.
- :func:`verify_workflow_spec` — recompute the hash, verify the
  signature against an author public key bytes blob.
- :func:`canonical_transition_payload` — the deterministic byte form
  signed for each :class:`StageLink`.
- :func:`sign_stage_transition` — produces the actor signature for a
  stage transition.
- :func:`verify_stage_transition` — verifies that signature.

What this module deliberately does NOT do:

- DID-document resolution. The ``public_key_resolver`` parameter is a
  pluggable callable so callers can resolve over ``did:web``,
  ``did:pkh``, in-memory test fixtures, or :class:`AgentIdentity`'s
  legacy keypair without this module owning that lookup. The
  ``verification_methods_resolver`` parameter does the same for hybrid
  Multikey verification methods.
- Anchor-style hashing. Workflow definitions sign the canonical JSON
  bytes directly; we do not Merkle-anchor or chain.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

from kestrel_sovereign.features.workflows.models import (
    RevocationReason,
    StageLink,
    WorkflowDefinitionError,
    WorkflowSpec,
)
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
    CryptoSuiteError,
    get_suite,
)

logger = logging.getLogger(__name__)


# Pluggable resolver: caller maps a DID to the public-key bytes used by
# :meth:`CryptoSuite.deserialize_public_key`. The legacy uncompressed
# X9.62 form is the right shape for secp256k1; callers that source
# their keys from W3C Multikey VMs use
# ``serialize_public_key_for_multikey``'s inverse.
PublicKeyResolver = Callable[[str], bytes]
VerificationMethodsResolver = Callable[[str], list[Mapping[str, Any]]]

_HYBRID_PREFIX = "hybrid:"


# ---------------------------------------------------------------------------
# Definition revocation signing
# ---------------------------------------------------------------------------


def canonical_force_abort_payload(*, run_id: str, reason: str) -> bytes:
    """Canonical payload signed by the sovereign for emergency abort."""

    body = json.dumps(
        {"reason": reason, "run_id": run_id},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"workflow.force_abort.v1\n{body}".encode("utf-8")


def canonical_definition_revocation_payload(
    *,
    name: str,
    version: int,
    reason: RevocationReason | str,
    revoked_at: str,
) -> bytes:
    """Canonical payload signed for a workflow definition revocation."""

    payload = {
        "name": name,
        "reason": RevocationReason(reason).value,
        "revoked_at": revoked_at,
        "version": version,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sign_definition_revocation(
    *,
    name: str,
    version: int,
    reason: RevocationReason | str,
    revoked_at: str,
    agent_identity: AgentIdentity,
) -> tuple[str, str]:
    """Sign a definition revocation with the legacy authority DID.

    Phase 1 stores the signature and authority DID with the revocation
    event. Hybrid revocation signatures can layer on this helper later,
    but the default path mirrors Phase 0 definition signing: legacy DID
    plus ECDSA secp256k1.
    """

    if agent_identity.legacy_keypair.private_key is None:
        raise WorkflowDefinitionError(
            "sign_definition_revocation: AgentIdentity has no legacy private key"
        )
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    payload = canonical_definition_revocation_payload(
        name=name,
        version=version,
        reason=reason,
        revoked_at=revoked_at,
    )
    try:
        sig = suite.sign(payload, agent_identity.legacy_keypair.private_key)
    except CryptoSuiteError as exc:
        raise WorkflowDefinitionError(
            f"sign_definition_revocation failed: {exc}"
        ) from exc
    return agent_identity.legacy_did, sig.hex()


# ---------------------------------------------------------------------------
# WorkflowSpec signing
# ---------------------------------------------------------------------------


def sign_workflow_spec(
    spec: WorkflowSpec,
    agent_identity: AgentIdentity,
    *,
    use_hybrid: bool = False,
) -> WorkflowSpec:
    """Sign ``spec`` with the agent's legacy ECDSA secp256k1 key.

    The author DID is taken from :attr:`AgentIdentity.signing_did` —
    that's the post-ceremony ``did:web`` identity for hybrid agents and
    the legacy ``did:pkh`` for pre-ceremony agents. The legacy ECDSA
    private key is used to produce the signature in either case
    (hybrid signatures are a Phase 1 concern; this Phase 0 helper
    reproduces the legacy field).

    The signed bytes are the lowercase-hex SHA-256 of the canonical
    payload. We sign the hex bytes (not the raw 32-byte digest) so the
    signature is over a stable string that auditors can recompute and
    log without normalization decisions.

    Returns a new :class:`WorkflowSpec` (frozen dataclass) with
    ``spec_hash`` and ``author_sig`` populated. ``author_did`` is
    overwritten only when the input was empty — callers may pre-set
    it for delegated signing scenarios where the spec author and the
    runner-on-disk agent differ.
    """
    if not isinstance(spec, WorkflowSpec):
        raise TypeError("sign_workflow_spec requires a WorkflowSpec instance")

    if use_hybrid and agent_identity.is_hybrid:
        chosen_author = agent_identity.signing_did
    else:
        chosen_author = agent_identity.legacy_did

    if (
        not (use_hybrid and agent_identity.is_hybrid)
        and agent_identity.legacy_keypair.private_key is None
    ):
        raise WorkflowDefinitionError(
            "sign_workflow_spec: AgentIdentity has no legacy private key "
            "(post-destruction state). Phase 1 will use the hybrid keypair "
            "via sign_hybrid; until then this helper requires a legacy key."
        )

    # Codex chunk-D round-2 P2: Phase 0 always signed with the legacy
    # ECDSA key, so ``author_did`` MUST be the *legacy* DID — not
    # ``signing_did``. For pre-ceremony agents these are equal. For
    # post-ceremony hybrid agents, ``signing_did`` is the new did:web
    # whose VMs publish ed25519 + ml-dsa-65 — a resolver mapping that
    # DID would NOT return the legacy ECDSA public key, so verification
    # would fail. The signed spec must carry the DID whose VMs cover
    # the actual signing algorithm. Phase 1 adds the hybrid path
    # (sign_hybrid + new did:web author) once the runner exercises
    # signatures end-to-end.
    # Phase 1 keeps that legacy path as the default for backward
    # compatibility, and enables a caller-selected hybrid path for
    # post-ceremony agents.

    # Codex chunk-D round-1 P2 (carried): a pre-set ``author_did``
    # must match the DID we'll actually sign for. For Phase 0 that's
    # the legacy DID. Delegated-signing flows where the spec author
    # differs from the local signer belong in a Phase 1+ multisig
    # surface, not this helper.
    if spec.author_did and spec.author_did != chosen_author:
        raise WorkflowDefinitionError(
            f"sign_workflow_spec: spec.author_did {spec.author_did!r} does "
            f"not match the DID {chosen_author!r} this helper signs as. "
            "Use a delegated-signing helper (Phase 1+) for cross-author "
            "signatures, or clear spec.author_did so this helper sets it "
            "from the signing identity."
        )
    signing_did = chosen_author

    # The canonical payload INCLUDES ``author_did`` (it's part of what
    # the author signs). Therefore set the author_did first, recompute
    # the hash on the *to-be-signed* form, then sign that hash. If we
    # signed before setting author_did, ``verify_workflow_spec`` would
    # later recompute a hash that doesn't match the stored one and
    # fail. Frozen-dataclass: use ``replace`` to materialize the
    # pre-sign canonical form.
    pre_sign = replace(
        spec,
        author_did=signing_did,
        author_sig="",
        spec_hash="",
    )
    spec_hash = pre_sign.compute_spec_hash()

    if use_hybrid and agent_identity.is_hybrid:
        from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid

        vms = agent_identity.new_verification_methods or []
        classical_kid = vms[0]["id"].rsplit("#", 1)[-1] if vms else "key-1"
        pq_kid = vms[1]["id"].rsplit("#", 1)[-1] if len(vms) > 1 else "key-2"
        sig = _encode_hybrid_signatures(
            sign_hybrid(
                spec_hash_to_bytes(spec_hash),
                agent_identity.hybrid_keypair,
                classical_kid=classical_kid,
                pq_kid=pq_kid,
            )
        )
    else:
        suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
        try:
            legacy_sig = suite.sign(
                spec_hash.encode("utf-8"),
                agent_identity.legacy_keypair.private_key,
            )
        except CryptoSuiteError as exc:
            raise WorkflowDefinitionError(
                f"sign_workflow_spec failed: {exc}"
            ) from exc
        sig = legacy_sig.hex()

    return replace(
        pre_sign,
        spec_hash=spec_hash,
        author_sig=sig,
    )


def verify_workflow_spec(
    spec: WorkflowSpec,
    public_key_resolver: PublicKeyResolver,
    *,
    verification_methods_resolver: Optional[VerificationMethodsResolver] = None,
) -> bool:
    """Recompute the canonical hash and verify ``spec.author_sig``.

    Returns ``True`` iff the signature verifies. Returns ``False`` (not
    raises) on any of: missing author_did/author_sig/spec_hash, hash
    mismatch, signature mismatch, malformed signature bytes,
    unresolvable author DID. The verifier never throws on bad input —
    callers integrate it into a runner-side gate where ``False`` means
    refuse to run.

    Hash mismatch and signature mismatch are deliberately
    indistinguishable in the return value because either represents an
    untrusted input; surfaces that need to log the cause use the side
    channel of the canary log already established by #1137.
    """
    if not isinstance(spec, WorkflowSpec):
        return False
    if not spec.author_did or not spec.author_sig or not spec.spec_hash:
        return False

    expected_hash = spec.compute_spec_hash()
    if expected_hash != spec.spec_hash:
        return False

    if spec.author_sig.startswith(_HYBRID_PREFIX):
        if verification_methods_resolver is None:
            return False
        try:
            from kestrel_sovereign.identity.hybrid_keypair import verify_hybrid

            signatures = _decode_hybrid_signatures(spec.author_sig)
            methods = verification_methods_resolver(spec.author_did)
            return verify_hybrid(
                spec_hash_to_bytes(spec.spec_hash),
                signatures,
                methods,
            ).ok
        except Exception as exc:  # noqa: BLE001 - verifier is fail-closed
            logger.debug("verify_workflow_spec hybrid failed: %s", exc)
            return False

    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)

    try:
        public_key_bytes = public_key_resolver(spec.author_did)
        public_key = suite.deserialize_public_key(public_key_bytes)
    except Exception as exc:  # noqa: BLE001 — resolver is caller-supplied
        logger.debug(
            "verify_workflow_spec: public-key resolution failed for %s: %s",
            spec.author_did,
            exc,
        )
        return False

    try:
        sig_bytes = bytes.fromhex(spec.author_sig)
    except ValueError:
        return False

    try:
        return suite.verify(spec_hash_to_bytes(spec.spec_hash), sig_bytes, public_key)
    except CryptoSuiteError:
        return False


def spec_hash_to_bytes(spec_hash: str) -> bytes:
    """Helper exposed for callers signing across the same hash form
    (sign and verify must encode identically). Returns the lowercase-
    hex string as UTF-8 bytes — NOT the raw 32-byte digest. See the
    sign_workflow_spec docstring for the rationale.
    """
    return spec_hash.encode("utf-8")


# ---------------------------------------------------------------------------
# Stage-transition signing
# ---------------------------------------------------------------------------


def canonical_transition_payload(
    *,
    run_id: str,
    stage_name: str,
    attempt_number: int,
    signal_id: Optional[str],
    gate_outcome: Optional[str],
) -> bytes:
    """Deterministic byte form for a stage transition signature (design
    §5 ``actor_sig`` definition).

    Format: a JSON object with sorted keys, no whitespace, ``ensure_ascii``
    off, prefixed with the ``workflow.stage_transition.v1`` discriminator
    line. JSON's distinction between ``null`` and the string ``"null"``
    means a pre-dispatch ``signal_id=None`` cannot produce the same
    bytes as any persisted string value — closing the round-1 codex
    P2 sentinel-collision gap (the prior format used the literal
    string ``"none"`` which collided with a persisted ``signal_id="none"``).

    The ``v1`` discriminator front-loads the format version so future
    additions (extra fields) can ship ``v2`` without colliding with
    v1 signatures.
    """
    import json as _json

    body = _json.dumps(
        {
            "run_id": run_id,
            "stage_name": stage_name,
            "attempt_number": attempt_number,
            "signal_id": signal_id,
            "gate_outcome": gate_outcome,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"workflow.stage_transition.v1\n{body}".encode("utf-8")


def sign_stage_transition(
    *,
    run_id: str,
    stage_name: str,
    attempt_number: int,
    signal_id: Optional[str],
    gate_outcome: Optional[str],
    agent_identity: AgentIdentity,
    use_hybrid: bool = False,
) -> tuple[str, str]:
    """Returns ``(actor_did, actor_sig_hex)`` for a stage transition.

    Same Phase 0 invariant as :func:`sign_workflow_spec`: we sign with
    the legacy ECDSA key, so the returned ``actor_did`` is the agent's
    *legacy* DID. Caller writes both into ``StageLink.actor_did`` /
    ``StageLink.actor_sig`` so verification resolves the matching
    public key. Phase 1 hybrid signing returns the new DID + a v2
    signature array.
    """
    if use_hybrid and agent_identity.is_hybrid:
        from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid

        payload = canonical_transition_payload(
            run_id=run_id,
            stage_name=stage_name,
            attempt_number=attempt_number,
            signal_id=signal_id,
            gate_outcome=gate_outcome,
        )
        vms = agent_identity.new_verification_methods or []
        classical_kid = vms[0]["id"].rsplit("#", 1)[-1] if vms else "key-1"
        pq_kid = vms[1]["id"].rsplit("#", 1)[-1] if len(vms) > 1 else "key-2"
        return (
            agent_identity.signing_did,
            _encode_hybrid_signatures(
                sign_hybrid(
                    payload,
                    agent_identity.hybrid_keypair,
                    classical_kid=classical_kid,
                    pq_kid=pq_kid,
                )
            ),
        )

    if agent_identity.legacy_keypair.private_key is None:
        raise WorkflowDefinitionError(
            "sign_stage_transition: AgentIdentity has no legacy private key"
        )
    payload = canonical_transition_payload(
        run_id=run_id,
        stage_name=stage_name,
        attempt_number=attempt_number,
        signal_id=signal_id,
        gate_outcome=gate_outcome,
    )
    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    try:
        sig = suite.sign(payload, agent_identity.legacy_keypair.private_key)
    except CryptoSuiteError as exc:
        raise WorkflowDefinitionError(
            f"sign_stage_transition failed: {exc}"
        ) from exc
    return agent_identity.legacy_did, sig.hex()


def verify_stage_transition(
    link: StageLink,
    public_key_resolver: PublicKeyResolver,
    *,
    verification_methods_resolver: Optional[VerificationMethodsResolver] = None,
) -> bool:
    """Verify a stage transition's actor signature.

    Returns False (not raises) on any failure mode — same posture as
    :func:`verify_workflow_spec`. The runner-side gate treats False
    as "refuse to advance the workflow" and logs a structured event;
    a bad signature on a stage transition is auditor-grade evidence
    of either tampering or a stale identity rotation that the
    operator must explicitly resolve.
    """
    if not isinstance(link, StageLink):
        return False
    if not link.actor_did or not link.actor_sig:
        return False

    payload = canonical_transition_payload(
        run_id=link.run_id,
        stage_name=link.stage_name,
        attempt_number=link.attempt_number,
        signal_id=link.signal_id,
        gate_outcome=(
            link.gate_outcome.value
            if link.gate_outcome is not None
            else None
        ),
    )

    if link.actor_sig.startswith(_HYBRID_PREFIX):
        if verification_methods_resolver is None:
            return False
        try:
            from kestrel_sovereign.identity.hybrid_keypair import verify_hybrid

            return verify_hybrid(
                payload,
                _decode_hybrid_signatures(link.actor_sig),
                verification_methods_resolver(link.actor_did),
            ).ok
        except Exception as exc:  # noqa: BLE001 - verifier is fail-closed
            logger.debug("verify_stage_transition hybrid failed: %s", exc)
            return False

    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)

    try:
        public_key_bytes = public_key_resolver(link.actor_did)
        public_key = suite.deserialize_public_key(public_key_bytes)
    except Exception as exc:  # noqa: BLE001 — resolver is caller-supplied
        logger.debug(
            "verify_stage_transition: public-key resolution failed for %s: %s",
            link.actor_did,
            exc,
        )
        return False

    try:
        sig_bytes = bytes.fromhex(link.actor_sig)
    except ValueError:
        return False

    try:
        return suite.verify(payload, sig_bytes, public_key)
    except CryptoSuiteError:
        return False


def _encode_hybrid_signatures(signatures: list[dict]) -> str:
    return _HYBRID_PREFIX + base64.b64encode(
        json.dumps(signatures, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


def _decode_hybrid_signatures(signature: str) -> list[Mapping[str, str]]:
    if not signature.startswith(_HYBRID_PREFIX):
        raise ValueError("not a hybrid workflow signature")
    decoded = base64.b64decode(signature[len(_HYBRID_PREFIX):]).decode()
    payload = json.loads(decoded)
    if not isinstance(payload, list):
        raise ValueError("hybrid workflow signature payload must be a list")
    return payload


__all__ = [
    "PublicKeyResolver",
    "VerificationMethodsResolver",
    "canonical_transition_payload",
    "sign_stage_transition",
    "sign_workflow_spec",
    "spec_hash_to_bytes",
    "verify_stage_transition",
    "verify_workflow_spec",
]
