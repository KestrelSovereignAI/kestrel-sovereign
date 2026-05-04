"""
Verify-policy modes for signed identity artifacts.

Wave 1 sub-PR 4 of the Quantum Hardening epic (#921, #916). Defines
the policy enum every signature verifier consumes and the per-context
default policies. Composes with the temporal-validity rules in
``docs/architecture/security/SERIALIZATION_COMPATIBILITY.md`` —
specifically the **post-succession-cutoff** rule that prevents
post-quantum forgeries of historical statements.

Three policy modes
------------------

- ``LEGACY_ALLOWED`` — accept any signature in the package's
  ``signatures`` array. Used for archival import / offline recovery
  through the migration window. Does NOT short-circuit other checks
  (the signature still has to verify cryptographically, and
  ``post_cutoff_classical_allowed=False`` still applies).
- ``HYBRID_REQUIRED`` — require at least one classical signature
  AND at least one post-quantum signature, both verifying. Default
  for live identity assertion after Wave 2.
- ``PQ_REQUIRED`` — require at least one post-quantum signature
  verifying; classical signatures are optional. Reserved for narrow
  long-horizon contexts after Wave 3.

Per-context defaults
--------------------

The schedule from PRD-v2 §7-§8 maps a ``Context`` enum value to its
current default ``VerifyPolicy``. Defaults tighten over releases as
each wave lands; the schedule below records the intent. Caller code
SHOULD use ``default_policy_for(context)`` rather than hardcoding,
so a global default-bump release becomes a one-line edit here.

Composition with succession-chain temporal validity
---------------------------------------------------

Wave 3 ships succession statements with an ``effective_from`` field
that bounds when each key was authoritative. The verify-policy alone
cannot decide whether a classical-only signature post-cutoff is
forged — it needs the temporal context. ``evaluate_signatures`` takes
an optional ``post_cutoff_classical_allowed`` flag (default True for
backwards compat through Wave 2) so Wave 3's chain walker can supply
``False`` once it determines an artifact is dated after the agent's
succession ``effective_from``. This is the hook that closes the
"LEGACY_ALLOWED + post-quantum forgery of historical statements"
attack the second feedback note flagged.

Today (pre-Wave-3) every caller passes the default; Wave 3 wires
the cutoff check at the chain-walker level.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Mapping, Optional

from .crypto_suite import _REGISTRY, CryptoSuite, get_suite


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VerifyPolicy(Enum):
    """Policy for accepting v2 signatures (and synthetic-v2 from v1)."""

    LEGACY_ALLOWED = "legacy_allowed"
    HYBRID_REQUIRED = "hybrid_required"
    PQ_REQUIRED = "pq_required"


class Context(Enum):
    """Caller context that selects a default policy.

    Each value maps to a per-context default in ``_CONTEXT_DEFAULTS``.
    Adding a new context is a one-line addition there; bumping a
    default for a release is a one-line edit there.
    """

    ARCHIVAL_IMPORT = "archival_import"
    LIVE_IDENTITY_ASSERTION = "live_identity_assertion"
    NEW_IDENTITY_ISSUANCE = "new_identity_issuance"
    SPAWN_MANDATE_VERIFICATION = "spawn_mandate_verification"
    CONSTITUTION_AUDIT_ROUTINE = "constitution_audit_routine"
    CONSTITUTION_CHECKPOINT = "constitution_checkpoint"


# ---------------------------------------------------------------------------
# Per-context defaults
# ---------------------------------------------------------------------------
#
# Schedule from PRD-v2 §7. Defaults reflect the *current* release (Wave 1).
# Each comment names the planned tightening release.

_CONTEXT_DEFAULTS: Mapping[Context, VerifyPolicy] = {
    # Read-only archival import stays permissive forever — sub-PR 4
    # consumers (legacy export readers, restore tools) need to see v1
    # data through the migration window.
    Context.ARCHIVAL_IMPORT: VerifyPolicy.LEGACY_ALLOWED,
    # Live identity assertion stays permissive until Wave 2 lands a
    # hybrid-issuing path; flips to HYBRID_REQUIRED with the v0.Z
    # release that bumps the new-agent default to did:web.
    Context.LIVE_IDENTITY_ASSERTION: VerifyPolicy.LEGACY_ALLOWED,
    # New identity issuance is HYBRID_REQUIRED from Wave 2 onward.
    # Pre-Wave-2 the call site doesn't exist yet, so the default here
    # is what the new-agent path will use when it ships.
    Context.NEW_IDENTITY_ISSUANCE: VerifyPolicy.HYBRID_REQUIRED,
    # Spawn-mandate verification stays permissive until the parent
    # has rotated via Wave 3; the chain walker tightens per parent.
    Context.SPAWN_MANDATE_VERIFICATION: VerifyPolicy.LEGACY_ALLOWED,
    # Routine constitution audits inherit the agent's identity policy.
    # Surface this as LEGACY_ALLOWED here; callers SHOULD pass the
    # agent's actual policy when they know it.
    Context.CONSTITUTION_AUDIT_ROUTINE: VerifyPolicy.LEGACY_ALLOWED,
    # Constitution checkpoint events (rotations) require PQ from
    # Wave 3 onward — irrevocable, long-horizon, conservative.
    Context.CONSTITUTION_CHECKPOINT: VerifyPolicy.PQ_REQUIRED,
}


def default_policy_for(context: Context) -> VerifyPolicy:
    """Return the current default policy for ``context``.

    Callers SHOULD use this instead of hardcoding a policy, so the
    release-by-release default-tightening schedule lives in one place.
    """
    return _CONTEXT_DEFAULTS[context]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyResult:
    """Outcome of a policy evaluation against a set of signatures.

    ``ok`` is the boolean verdict; ``reason`` carries a human-readable
    explanation suitable for log lines and error messages. ``alg_ids_seen``
    is the set of suite identifiers from the input — useful for
    diagnostics and telemetry.
    """
    ok: bool
    reason: str
    alg_ids_seen: frozenset[str]


def _classify(alg_id: str) -> Optional[bool]:
    """Return True if ``alg_id`` is a registered PQ suite, False if
    classical, None if unknown.

    Unknown alg_ids are NOT silently treated as classical or PQ —
    they fail the policy with an explicit "unknown alg" reason. A
    future suite that ships a token under a brand-new alg_id this
    build doesn't recognize must fail loud, not silently mis-classify.
    """
    suite = _REGISTRY.get(alg_id)
    if suite is None:
        return None
    return bool(suite.is_post_quantum)


def evaluate_signatures(
    signatures: Iterable[Mapping[str, str]],
    policy: VerifyPolicy,
    *,
    post_cutoff_classical_allowed: bool = True,
) -> PolicyResult:
    """Evaluate a list of v2 ``signatures`` array entries against ``policy``.

    Each entry must be a mapping with at minimum an ``alg`` key — the
    function does NOT verify the signatures cryptographically. That's
    the caller's job; this function decides whether the *set* of suite
    identifiers satisfies the policy. Pair it with per-signature
    crypto verification at the call site.

    Args:
        signatures: iterable of ``{alg, kid, sig}`` dicts. Synthetic v2
            entries from v1 packages are accepted (they're tagged
            ``ecdsa-secp256k1-sha256``).
        policy: the ``VerifyPolicy`` mode to enforce.
        post_cutoff_classical_allowed: True (default) → no temporal
            cutoff applies. False → reject if the only acceptable
            signatures are classical-only. Wave 3 chain walkers pass
            False once they determine the artifact is dated after the
            agent's succession ``effective_from``.

    Returns:
        ``PolicyResult`` with ``ok``, ``reason``, and the seen alg_ids.
    """
    seen: List[str] = []
    has_classical = False
    has_pq = False
    has_unknown = False

    for entry in signatures:
        alg = entry.get("alg") if isinstance(entry, Mapping) else None
        if not alg:
            return PolicyResult(
                ok=False,
                reason="signature entry missing 'alg' field",
                alg_ids_seen=frozenset(seen),
            )
        seen.append(alg)
        kind = _classify(alg)
        if kind is None:
            has_unknown = True
        elif kind:
            has_pq = True
        else:
            has_classical = True

    seen_frozen = frozenset(seen)

    if not seen:
        return PolicyResult(
            ok=False,
            reason="no signatures present",
            alg_ids_seen=seen_frozen,
        )

    if has_unknown:
        return PolicyResult(
            ok=False,
            reason=(
                f"unknown alg_id in signatures (registered: "
                f"{sorted(_REGISTRY)}); refusing rather than silently "
                f"misclassifying"
            ),
            alg_ids_seen=seen_frozen,
        )

    # Temporal cutoff: a chain segment dated after a known
    # post-quantum succession effective_from cannot rely on classical
    # signatures alone — Shor-broken old keys could otherwise forge it.
    # PQ_REQUIRED already implies has_pq, so this only matters under
    # LEGACY_ALLOWED and HYBRID_REQUIRED.
    if not post_cutoff_classical_allowed and not has_pq:
        return PolicyResult(
            ok=False,
            reason=(
                "post-cutoff artifact has no post-quantum signature; "
                "classical-only signatures are not trusted after the "
                "agent's succession effective_from"
            ),
            alg_ids_seen=seen_frozen,
        )

    if policy is VerifyPolicy.LEGACY_ALLOWED:
        # Any (known) signature suffices.
        return PolicyResult(
            ok=True,
            reason="LEGACY_ALLOWED satisfied",
            alg_ids_seen=seen_frozen,
        )

    if policy is VerifyPolicy.HYBRID_REQUIRED:
        if has_classical and has_pq:
            return PolicyResult(
                ok=True,
                reason="HYBRID_REQUIRED satisfied (classical + PQ both present)",
                alg_ids_seen=seen_frozen,
            )
        missing = []
        if not has_classical:
            missing.append("classical")
        if not has_pq:
            missing.append("post-quantum")
        return PolicyResult(
            ok=False,
            reason=f"HYBRID_REQUIRED needs both classical and PQ; missing: {', '.join(missing)}",
            alg_ids_seen=seen_frozen,
        )

    if policy is VerifyPolicy.PQ_REQUIRED:
        if has_pq:
            return PolicyResult(
                ok=True,
                reason="PQ_REQUIRED satisfied",
                alg_ids_seen=seen_frozen,
            )
        return PolicyResult(
            ok=False,
            reason="PQ_REQUIRED needs at least one post-quantum signature",
            alg_ids_seen=seen_frozen,
        )

    # Unreachable; guard against a future enum value being added without
    # a branch here.
    raise AssertionError(f"unhandled VerifyPolicy: {policy!r}")


__all__ = [
    "Context",
    "PolicyResult",
    "VerifyPolicy",
    "default_policy_for",
    "evaluate_signatures",
]
