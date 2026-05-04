"""
Succession chain walker — Wave 3 sub-PR 3 of Quantum Hardening (#921, #918).

The chain walker is the verifier-side primitive that lets any consumer
of a signed artifact (constitution audit, identity-package verification,
spawn-mandate check, capsule import) decide which keys are authoritative
at the artifact's timestamp, and reject classical-only signatures that
fall after a successor cutoff.

Why this lives in a separate module from :mod:`succession`
----------------------------------------------------------

``succession.py`` ships the data structure and the per-statement
sign / verify primitives. The chain walker composes those primitives
with **temporal context**: an artifact's timestamp determines which
predecessor / successor is authoritative and whether the
``post_cutoff_classical_allowed=False`` rule kicks in. Putting it in
its own module keeps each layer's responsibility crisp:

- ``succession.py``: a single rotation event
- ``succession_chain.py``: temporal resolution across many events
- (Wave 3 sub-PR 4): the actual rotation ceremony script

Threat model the cutoff blocks
------------------------------

Without the cutoff, a future Shor-equipped adversary could:

1. Recover a legacy ECDSA private key from the predecessor's public key.
2. Sign a *new* artifact (e.g. a constitution checkpoint) under that key,
   back-dated to a timestamp the adversary picks.
3. Present the artifact + a forged "succession statement" pointing at
   an attacker-controlled successor.
4. Convince a verifier that the legitimate successor was never
   authorized.

The chain walker rejects step 2 (artifact signed by classical key after
``effective_from``) and rejects step 3 (the legitimate, signed-while-
classical-was-trusted succession statement is incompatible with the
forged one because the chain walker's structural checks notice that
two successions both claim ``effective_from`` in the same window).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from kestrel_sovereign.identity.succession import (
    SuccessionStatement,
    SuccessionVerifyResult,
    verify_succession,
)
from kestrel_sovereign.security.crypto_suite import CryptoSuiteError
from kestrel_sovereign.security.multikey import multibase_to_public_key
from kestrel_sovereign.security.verify_policy import (
    PolicyResult,
    VerifyPolicy,
    evaluate_signatures,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SuccessionChainError(Exception):
    """Raised when a chain fails structural validation."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso8601_utc(s: str) -> datetime:
    """Parse an ISO 8601 timestamp into a tz-aware UTC datetime.

    Tolerates the trailing ``Z`` suffix that some serializers emit.
    Rejects naive timestamps AND non-UTC offsets — temporal
    comparisons across successions must be unambiguous about
    timezone, and the schema documents UTC-only.
    """
    from datetime import timedelta
    if not isinstance(s, str) or not s:
        raise SuccessionChainError(f"timestamp must be a non-empty string; got {s!r}")
    candidate = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError as e:
        raise SuccessionChainError(f"invalid ISO 8601 timestamp {s!r}: {e}") from e
    if dt.tzinfo is None:
        raise SuccessionChainError(
            f"timestamp {s!r} is timezone-naive; chain walker requires UTC-explicit "
            f"timestamps to compare successions unambiguously"
        )
    if dt.utcoffset() != timedelta(0):
        raise SuccessionChainError(
            f"timestamp {s!r} is not UTC (offset {dt.utcoffset()}); chain "
            f"walker requires UTC-only timestamps for cross-statement "
            f"comparisons"
        )
    return dt


# ---------------------------------------------------------------------------
# SuccessionChain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuccessionChain:
    """An ordered, structurally-validated sequence of succession statements.

    Empty chains are valid — they represent an agent that has not (yet)
    rotated. A single-element chain is the most common Wave 3 ceremony
    case (one ECDSA → hybrid migration). Multi-element chains arise
    when an agent rotates more than once.
    """

    statements: Tuple[SuccessionStatement, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.statements)

    def is_empty(self) -> bool:
        return not self.statements


def _vm_dict_key(vm: Mapping) -> tuple:
    """Stable identity key for a verification method (id + multibase).

    Used by chain VM-linkage check. Compares only the fields that
    matter for binding — ``id`` and ``publicKeyMultibase``. ``type``
    is fixed (``Multikey``) and ``controller`` is administrative.
    """
    return (vm.get("id"), vm.get("publicKeyMultibase"))


def build_chain(statements: Iterable[SuccessionStatement]) -> SuccessionChain:
    """Validate + freeze a list of statements into a :class:`SuccessionChain`.

    Validation rules:

    - **Linkage**: each statement's ``predecessor_did`` MUST equal the
      previous statement's ``successor_did``. A chain with a fork or
      gap is rejected.
    - **VM linkage** (added per #963 codex review): each statement's
      ``predecessor_verification_methods`` MUST equal the previous
      statement's ``successor_verification_methods`` (matched by
      ``id`` + ``publicKeyMultibase``). Without this, an attacker
      could fork the chain at link N+1 by claiming the right
      predecessor_did but embedding their OWN keys; the per-statement
      DID-binding check on a did:web predecessor can't catch that
      without resolution. Chain linkage closes the gap because the
      previous statement's successor signatures (which we already
      verify) attest to those exact VMs.
    - **Temporal monotonicity**: each statement's ``effective_from``
      MUST be strictly after the previous one's. Chains that go
      backward in time would let an adversary unwind a rotation.
    - **Self-loop guard**: ``predecessor_did != successor_did`` per
      statement. A self-succession is meaningless and we refuse it
      rather than silently accept.

    Per-statement *signature* validity (i.e. each statement's
    crypto-verify) is NOT checked here — :func:`build_chain` is purely
    structural. Callers that need signature validation pair the chain
    with :func:`verify_chain_signatures` (or use
    :func:`verify_artifact_against_chain` which composes both).
    """
    seq = tuple(statements)
    for i, s in enumerate(seq):
        if s.predecessor_did == s.successor_did:
            raise SuccessionChainError(
                f"statement[{i}] is a self-succession "
                f"({s.predecessor_did!r}); refusing"
            )
        # Codex P2 round 12: validate every statement's timestamp,
        # including the first one. Otherwise build_chain([single]) lets
        # a malformed timestamp through and resolve_active_identity
        # raises later — turning structured verification into an
        # exception. Parse for side-effect (raises on malformed input);
        # re-parsed in the comparison branch below for the dt value.
        _parse_iso8601_utc(s.effective_from)
        if i == 0:
            continue
        prev = seq[i - 1]
        if s.predecessor_did != prev.successor_did:
            raise SuccessionChainError(
                f"chain link broken at statement[{i}]: "
                f"predecessor_did={s.predecessor_did!r} != prev.successor_did="
                f"{prev.successor_did!r}"
            )
        prev_succ_keys = sorted(
            _vm_dict_key(vm) for vm in prev.successor_verification_methods
        )
        cur_pred_keys = sorted(
            _vm_dict_key(vm) for vm in s.predecessor_verification_methods
        )
        if prev_succ_keys != cur_pred_keys:
            raise SuccessionChainError(
                f"chain VM linkage broken at statement[{i}]: "
                f"predecessor_verification_methods don't match "
                f"prev.successor_verification_methods (id + multibase). "
                f"This blocks an attacker from forking the chain by "
                f"substituting their own keys at a later link."
            )
        prev_t = _parse_iso8601_utc(prev.effective_from)
        cur_t = _parse_iso8601_utc(s.effective_from)
        if cur_t <= prev_t:
            raise SuccessionChainError(
                f"chain temporal monotonicity violated at statement[{i}]: "
                f"effective_from={s.effective_from!r} <= prev.effective_from="
                f"{prev.effective_from!r}"
            )
    return SuccessionChain(statements=seq)


# ---------------------------------------------------------------------------
# Active-identity resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActiveIdentity:
    """The authoritative DID + verification methods at a given timestamp."""

    did: str
    verification_methods: Tuple[Mapping, ...]
    is_root: bool  # True if no succession has fired yet at this timestamp
    succession_index: Optional[int]  # which statement provided this (None if root)
    post_cutoff: bool  # True if at least one succession has fired


def resolve_active_identity(
    root_did: str,
    root_verification_methods: Sequence[Mapping],
    chain: SuccessionChain,
    artifact_timestamp: str,
) -> ActiveIdentity:
    """At ``artifact_timestamp``, return the authoritative DID + VMs.

    Walk rules:

    - If ``artifact_timestamp < statements[0].effective_from`` (or chain
      is empty): the root identity is active. ``post_cutoff`` is False —
      classical-only signatures are still trusted.
    - Otherwise: find the LARGEST ``i`` such that
      ``statements[i].effective_from <= artifact_timestamp``. That
      statement's successor is the active identity. ``post_cutoff`` is
      True; classical-only artifacts dated after a known successor
      cutoff fail the policy unless they pre-date the cutoff.

    Note: a future succession (``effective_from > artifact_timestamp``)
    does NOT push an artifact into post-cutoff. The cutoff is a function
    of what HAS happened, not what WILL happen — otherwise a back-dated
    artifact at the moment of rotation would be retroactively rejected.
    """
    artifact_t = _parse_iso8601_utc(artifact_timestamp)

    if chain.is_empty():
        return ActiveIdentity(
            did=root_did,
            verification_methods=tuple(dict(m) for m in root_verification_methods),
            is_root=True,
            succession_index=None,
            post_cutoff=False,
        )

    # Find latest succession whose effective_from <= artifact_t.
    active_idx: Optional[int] = None
    for i, s in enumerate(chain.statements):
        eff_t = _parse_iso8601_utc(s.effective_from)
        if eff_t <= artifact_t:
            active_idx = i
        else:
            break  # statements monotonically increasing

    if active_idx is None:
        # Artifact is older than the first succession — root is active.
        return ActiveIdentity(
            did=root_did,
            verification_methods=tuple(dict(m) for m in root_verification_methods),
            is_root=True,
            succession_index=None,
            post_cutoff=False,
        )

    active = chain.statements[active_idx]
    return ActiveIdentity(
        did=active.successor_did,
        verification_methods=tuple(
            dict(m) for m in active.successor_verification_methods
        ),
        is_root=False,
        succession_index=active_idx,
        post_cutoff=True,
    )


# ---------------------------------------------------------------------------
# Chain signature verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainSignaturesResult:
    """Outcome of verifying every statement's signatures in a chain."""

    ok: bool
    per_statement: Tuple[SuccessionVerifyResult, ...]
    reason: str


def _build_chain_internal_resolver(
    chain: SuccessionChain,
    user_resolver: Optional[Callable[[str], Mapping[str, object]]] = None,
):
    """Build a chain-aware did:web resolver.

    Codex P1 chain-walker review: the previous version returned the
    embedded successor VMs for EVERY chain successor — including the
    chain TIP (the current active identity) and any successor being
    bound at signing time. That made successor binding tautological
    for did:web (the binding check confirmed embedded == embedded).

    New behavior:
    - **Chain-internal linkage** (predecessor_did of statement[i+1]
      equals successor_did of statement[i] for some i): return a
      synthetic doc containing the previous successor's VMs. Honest
      because :func:`build_chain` already enforced VM linkage and the
      previous statement's successor signatures attested to those VMs.
    - **Non-linkage DIDs** (chain tip's successor, root predecessor,
      external references): defer to ``user_resolver``. If no user
      resolver is provided, raise so binding fails-closed.

    Linkage DIDs are exactly: ``successor_did`` of statements[0:n-1]
    (i.e. all chain successors except the last). The last successor
    is the active identity; its publication is the source of truth.
    """
    if not chain.statements:
        linkage_dids: dict = {}
    else:
        linkage_dids = {
            s.successor_did: tuple(dict(vm) for vm in s.successor_verification_methods)
            for s in chain.statements[:-1]
        }

    def _resolver(did: str) -> Mapping[str, object]:
        if did in linkage_dids:
            return {"id": did, "verificationMethod": list(linkage_dids[did])}
        if user_resolver is not None:
            return user_resolver(did)
        raise SuccessionChainError(
            f"chain-aware resolver cannot resolve {did!r}: not an "
            f"internal-linkage DID and no user_resolver was provided "
            f"to handle the chain tip / external lookup"
        )

    return _resolver


def verify_chain_signatures(
    chain: SuccessionChain,
    *,
    predecessor_policy: Optional[VerifyPolicy] = None,
    successor_policy: VerifyPolicy = VerifyPolicy.HYBRID_REQUIRED,
    require_archival: bool = False,
    did_web_resolver: Optional[Callable[[str], Mapping[str, object]]] = None,
) -> ChainSignaturesResult:
    """Verify every statement in the chain individually.

    Each statement's :func:`verify_succession` runs with the supplied
    policies. For chain-internal did:web predecessors (statement[i+1]'s
    predecessor matches statement[i]'s successor), binding is satisfied
    via the chain-internal resolver — chain VM linkage was already
    verified at :func:`build_chain` time. For NON-linkage DIDs (the
    chain tip's successor, etc.) the wrapper defers to
    ``did_web_resolver`` if provided.

    Codex round-11 chain review:
    - ``predecessor_policy`` defaults to ``None`` so
      ``verify_succession``'s adaptive auto-promotion to HYBRID_REQUIRED
      kicks in for hybrid-predecessor links (was hardcoded LEGACY_ALLOWED
      which regressed quantum-hardening for post-migration rotations).
    - The chain-internal resolver no longer self-resolves did:web
      successors; callers must pass ``did_web_resolver`` for the chain
      tip / any external lookup.

    The composite ``ok`` is True only if every statement passes.
    Per-statement results are returned for diagnostics.
    """
    chain_resolver = _build_chain_internal_resolver(chain, user_resolver=did_web_resolver)

    per_results: List[SuccessionVerifyResult] = []
    all_ok = True
    failure_summaries: List[str] = []

    for i, s in enumerate(chain.statements):
        r = verify_succession(
            s,
            predecessor_policy=predecessor_policy,
            successor_policy=successor_policy,
            require_archival=require_archival,
            did_web_resolver=chain_resolver,
        )
        per_results.append(r)
        if not r.ok:
            all_ok = False
            failure_summaries.append(f"statement[{i}]: {r.reason}")

    return ChainSignaturesResult(
        ok=all_ok,
        per_statement=tuple(per_results),
        reason="all chain signatures verified" if all_ok
        else "; ".join(failure_summaries),
    )


# ---------------------------------------------------------------------------
# Artifact verification under a chain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactChainVerifyResult:
    """Composite verdict for verifying an artifact against a chain.

    - ``ok``: True only if all of: chain structure valid, every chain
      signature verified, and artifact signatures satisfy the policy
      (with the post-cutoff rule applied where it bites).
    - ``active_identity``: who's authoritative for this artifact.
    - ``chain_signatures``: per-statement results.
    - ``policy_result``: outcome of the artifact-side policy evaluation.
    - ``reason``: human-readable summary for log lines.
    """

    ok: bool
    active_identity: ActiveIdentity
    chain_signatures: ChainSignaturesResult
    policy_result: PolicyResult
    reason: str


def _verify_artifact_signatures(
    payload: bytes,
    signatures: Iterable[Mapping[str, str]],
    verification_methods: Iterable[Mapping[str, object]],
) -> List[dict]:
    """Drop entries that don't crypto-verify; return survivors. Mirrors
    :func:`succession._verify_signatures_against` but accepts the
    bytes-payload form artifacts use directly.

    Defense-in-depth (codex P1 follow-up): refuse VM lists with
    duplicate kid fragments. Same attack as in succession.py — a
    later attacker-controlled VM silently overwrites a legitimate one
    in ``methods_by_kid``.
    """
    vms_list = list(verification_methods)
    seen_kids: dict = {}
    for vm in vms_list:
        if not isinstance(vm, Mapping):
            continue
        vm_id = vm.get("id") or ""
        kid = vm_id.rsplit("#", 1)[-1] if "#" in vm_id else vm_id
        if not kid:
            continue
        if kid in seen_kids:
            return []
        seen_kids[kid] = vm_id

    methods_by_kid: dict = {}
    for vm in vms_list:
        if not isinstance(vm, Mapping):
            continue
        vm_id = vm.get("id") or ""
        kid = vm_id.rsplit("#", 1)[-1] if "#" in vm_id else vm_id
        methods_by_kid[kid] = vm

    verified: List[dict] = []
    for entry in signatures:
        if not isinstance(entry, Mapping):
            continue
        alg = entry.get("alg")
        kid = entry.get("kid")
        sig_hex = entry.get("sig")
        if not alg or not kid or not sig_hex:
            continue
        vm = methods_by_kid.get(kid)
        if not vm:
            continue
        multibase = vm.get("publicKeyMultibase")
        if not isinstance(multibase, str):
            continue
        try:
            suite, pub = multibase_to_public_key(multibase)
        except CryptoSuiteError:
            continue
        if suite.alg_id != alg:
            continue
        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except ValueError:
            continue
        if suite.verify(payload, sig_bytes, pub):
            verified.append(dict(entry))
    return verified


def verify_artifact_against_chain(
    *,
    root_did: str,
    root_verification_methods: Sequence[Mapping],
    chain: SuccessionChain,
    artifact_timestamp: str,
    artifact_payload: bytes,
    artifact_signatures: Sequence[Mapping[str, str]],
    policy: VerifyPolicy = VerifyPolicy.HYBRID_REQUIRED,
    verify_chain: bool = True,
    did_web_resolver: Optional[Callable[[str], Mapping]] = None,
) -> ArtifactChainVerifyResult:
    """Verify an artifact's signatures against the active identity at
    ``artifact_timestamp``, applying the post-cutoff rule.

    Steps:

    1. (Optional, default on) Verify every chain statement's signatures
       under the standard predecessor/successor policies.
    2. Resolve the active identity at ``artifact_timestamp`` via
       :func:`resolve_active_identity`.
    3. Crypto-verify each ``artifact_signatures`` entry against the
       active identity's verification methods. Drop entries that don't
       verify.
    4. Run :func:`evaluate_signatures` over the surviving set with
       ``policy`` and ``post_cutoff_classical_allowed=not active.post_cutoff``.
       Once a succession has fired at the artifact's timestamp,
       classical-only signatures are no longer trusted on their own —
       they must be paired with at least one post-quantum signature.

    Args:
        root_did: the original (root) DID before any successions.
        root_verification_methods: VMs that were authoritative for
            ``root_did`` before any successions. May be classical-only
            for legacy agents.
        chain: validated :class:`SuccessionChain` (build via
            :func:`build_chain`).
        artifact_timestamp: ISO 8601 UTC timestamp of the artifact.
            Used to find the active identity AND to gate the cutoff.
        artifact_payload: canonical bytes that were signed.
        artifact_signatures: the v2 ``signatures`` array from the
            artifact (each ``{alg, kid, sig}`` dict).
        policy: the policy mode the artifact is required to satisfy.
            Default ``HYBRID_REQUIRED`` for live identity assertion.
        verify_chain: if True (default), validate every chain
            statement's signatures before doing anything else.

    Returns:
        :class:`ArtifactChainVerifyResult` with active identity,
        per-statement diagnostics, and the policy verdict.
    """
    # Step 0: anchor the chain to the supplied root (codex P1 round 11
    # chain review). Without this, an attacker could hand the verifier
    # a valid succession chain for AN UNRELATED identity along with an
    # artifact, and artifacts signed by that chain's successor would
    # verify as though they belonged to the supplied root_did.
    #
    # Anchor rule: when chain is non-empty, statements[0].predecessor_did
    # MUST equal root_did, and statements[0].predecessor_verification_
    # methods MUST equal root_verification_methods (matched by
    # id + multibase).
    if not chain.is_empty():
        first = chain.statements[0]
        if first.predecessor_did != root_did:
            return ArtifactChainVerifyResult(
                ok=False,
                active_identity=ActiveIdentity(
                    did=root_did, verification_methods=tuple(),
                    is_root=True, succession_index=None, post_cutoff=False,
                ),
                chain_signatures=ChainSignaturesResult(
                    ok=False, per_statement=tuple(),
                    reason="chain not anchored to supplied root",
                ),
                policy_result=PolicyResult(
                    ok=False, reason="not evaluated", alg_ids_seen=frozenset(),
                ),
                reason=(
                    f"chain anchor mismatch: chain[0].predecessor_did="
                    f"{first.predecessor_did!r} but root_did={root_did!r}"
                ),
            )
        root_keys = sorted(
            (vm.get("id"), vm.get("publicKeyMultibase"))
            for vm in root_verification_methods
            if isinstance(vm, Mapping)
        )
        first_pred_keys = sorted(
            (vm.get("id"), vm.get("publicKeyMultibase"))
            for vm in first.predecessor_verification_methods
            if isinstance(vm, Mapping)
        )
        if root_keys != first_pred_keys:
            return ArtifactChainVerifyResult(
                ok=False,
                active_identity=ActiveIdentity(
                    did=root_did, verification_methods=tuple(),
                    is_root=True, succession_index=None, post_cutoff=False,
                ),
                chain_signatures=ChainSignaturesResult(
                    ok=False, per_statement=tuple(),
                    reason="chain not anchored to supplied root VMs",
                ),
                policy_result=PolicyResult(
                    ok=False, reason="not evaluated", alg_ids_seen=frozenset(),
                ),
                reason=(
                    "chain anchor mismatch: chain[0].predecessor_verification"
                    "_methods do not match supplied root_verification_methods "
                    "(id + multibase). Refusing to treat an unrelated chain "
                    "as belonging to this root."
                ),
            )

    # Step 1: chain structure was validated at build_chain() time. Now
    # validate chain signatures too (unless disabled for performance —
    # e.g. a hot-path verifier that already cached the chain result).
    if verify_chain:
        chain_sigs = verify_chain_signatures(chain, did_web_resolver=did_web_resolver)
    else:
        chain_sigs = ChainSignaturesResult(
            ok=True,
            per_statement=tuple(),
            reason="chain signature verification skipped per caller",
        )

    # Step 2: who's authoritative right now?
    active = resolve_active_identity(
        root_did=root_did,
        root_verification_methods=root_verification_methods,
        chain=chain,
        artifact_timestamp=artifact_timestamp,
    )

    # Step 3: crypto-verify artifact signatures
    verified = _verify_artifact_signatures(
        artifact_payload,
        artifact_signatures,
        active.verification_methods,
    )

    # Step 4: apply policy with post-cutoff bit
    policy_result = evaluate_signatures(
        verified,
        policy,
        post_cutoff_classical_allowed=not active.post_cutoff,
    )

    composite_ok = chain_sigs.ok and policy_result.ok

    if composite_ok:
        reason = (
            f"artifact verified against active identity {active.did} "
            f"({'root' if active.is_root else f'successor[{active.succession_index}]'})"
        )
    else:
        parts = []
        if not chain_sigs.ok:
            parts.append(f"chain: {chain_sigs.reason}")
        if not policy_result.ok:
            parts.append(f"artifact: {policy_result.reason}")
        reason = "; ".join(parts) or "unknown failure"

    return ArtifactChainVerifyResult(
        ok=composite_ok,
        active_identity=active,
        chain_signatures=chain_sigs,
        policy_result=policy_result,
        reason=reason,
    )


__all__ = [
    "ActiveIdentity",
    "ArtifactChainVerifyResult",
    "ChainSignaturesResult",
    "SuccessionChain",
    "SuccessionChainError",
    "build_chain",
    "resolve_active_identity",
    "verify_artifact_against_chain",
    "verify_chain_signatures",
]
