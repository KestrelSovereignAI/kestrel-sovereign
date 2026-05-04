"""
Release manifest — Wave 5 sub-PR 1 of Quantum Hardening (#921, #920).

A release manifest is a signed inventory of the files that constitute
a Kestrel release. Each entry records the file's path, SHA-256 hash,
and size. The manifest itself is signed with SLH-DSA-SHA2-128s
(NIST FIPS 205, hash-based PQ — the conservative-tier signature suite
shipped in Wave 3 sub-PR 1, #960).

Why SLH-DSA for release signing
-------------------------------

Releases are infrequent and the integrity guarantee must hold for
years, possibly decades. The cost of forgery is permanent: an
attacker who recovers a release-signing private key can sign a
compromised wheel that every future user installs. SLH-DSA's
hash-based security argument relies only on the underlying hash
function's collision/preimage resistance — which Grover halves but
does NOT break the way Shor breaks ECC and lattice schemes.

Signature size (~7856 bytes) is irrelevant for release artifacts:
each release ships once and the signature lives in the manifest
forever. The high-throughput signing path (identity assertion,
constitution audits) stays hybrid Ed25519 + ML-DSA-65.

Wire format
-----------

JSON, sorted-keys for byte stability::

    {
      "format": "kestrel-release-manifest-v1",
      "version": 1,
      "release_tag": "v1.2.3",
      "released_at": "2026-05-04T20:00:00+00:00",
      "signer_did": "did:web:kestrel-sovereign.example",
      "artifacts": [
        {"path": "kestrel_sovereign-1.2.3-py3-none-any.whl",
         "sha256": "<hex>", "size": 1234567},
        {"path": "kestrel_sovereign-1.2.3.tar.gz",
         "sha256": "<hex>", "size": 987654}
      ],
      "manifest_id": "<sha256-hex of signable_payload>",
      "signatures": [
        {"alg": "slh-dsa-sha2-128s", "kid": "release-key-1", "sig": "<hex>"}
      ]
    }

Signatures may also include an Ed25519 + ML-DSA-65 hybrid pair
(useful during the rollout period when verifiers may not yet
recognize the SLH-DSA codec). The verifier's policy decides which
algorithms must be present — see :func:`verify_manifest`.

What this module DOES NOT do
----------------------------

- It does not produce or upload release artifacts; the caller hands
  the artifact bytes to :func:`add_artifact_entry` to record their
  hash. The actual build/upload tooling is separate (Wave 5 sub-PR
  2 will add the CLI; sub-PR 3 the GitHub Action).
- It does not pin a key-distribution mechanism. The signer's
  expected public key is provided to :func:`verify_manifest` by the
  verifier (typically a known multibase string baked into release
  documentation, or fetched from the signer's did:web document).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Mapping, Optional

from kestrel_sovereign.security.crypto_suite import (
    ALG_SLH_DSA_SHA2_128S,
    CryptoSuiteError,
    Keypair,
    get_suite,
)
from kestrel_sovereign.security.multikey import (
    multibase_to_public_key,
    public_key_to_multibase,
)
from kestrel_sovereign.security.verify_policy import (
    PolicyResult,
    VerifyPolicy,
    evaluate_signatures,
)


MANIFEST_FORMAT_ID = "kestrel-release-manifest-v1"
MANIFEST_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactEntry:
    """A single artifact's path, hash, and size.

    ``path`` is the relative file name as it will appear in the
    release distribution (no leading ``/``, no ``..``). The path is
    bytewise-compared at verify time, so consistent normalization is
    the producer's responsibility — pick one convention (lowercase
    extensions, forward slashes) and stick to it.
    """

    path: str
    sha256: str
    size: int

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ArtifactEntry":
        return cls(
            path=d["path"],
            sha256=d["sha256"],
            size=int(d["size"]),
        )


@dataclass(frozen=True)
class ReleaseManifest:
    """An immutable signed inventory of release artifacts."""

    release_tag: str
    released_at: str
    signer_did: str = ""
    artifacts: List[ArtifactEntry] = field(default_factory=list)
    signatures: List[dict] = field(default_factory=list)
    manifest_id: str = ""

    # Format / version are constants but exposed in to_dict / from_dict
    # so a v2 reader can detect a v1 producer cleanly.

    def to_dict(self) -> dict:
        return {
            "format": MANIFEST_FORMAT_ID,
            "version": MANIFEST_FORMAT_VERSION,
            "release_tag": self.release_tag,
            "released_at": self.released_at,
            "signer_did": self.signer_did,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "manifest_id": self.manifest_id,
            "signatures": [dict(s) for s in self.signatures],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReleaseManifest":
        if d.get("format") != MANIFEST_FORMAT_ID:
            raise ReleaseManifestError(
                f"unknown manifest format {d.get('format')!r}; expected "
                f"{MANIFEST_FORMAT_ID!r}"
            )
        if d.get("version") != MANIFEST_FORMAT_VERSION:
            raise ReleaseManifestError(
                f"unknown manifest version {d.get('version')!r}; expected "
                f"{MANIFEST_FORMAT_VERSION}"
            )
        return cls(
            release_tag=d["release_tag"],
            released_at=d["released_at"],
            signer_did=d.get("signer_did", ""),
            artifacts=[
                ArtifactEntry.from_dict(a) for a in d.get("artifacts", [])
            ],
            signatures=[dict(s) for s in d.get("signatures", [])],
            manifest_id=d.get("manifest_id", ""),
        )


class ReleaseManifestError(Exception):
    """Raised on manifest format errors, signing/verify failures, or
    artifact-hash mismatches."""


# ---------------------------------------------------------------------------
# Canonical signable payload + manifest_id
# ---------------------------------------------------------------------------
#
# Excludes:
# - signatures: signing your own signature is circular
# - manifest_id: derived from the payload; would also be circular

_SIGNED_FIELDS = ("release_tag", "released_at", "signer_did", "artifacts")


def signable_payload(manifest: ReleaseManifest) -> bytes:
    """Canonical UTF-8 bytes for signing/verification.

    Sorted-keys compact JSON; deterministic across signers/verifiers.
    """
    payload = {
        "format": MANIFEST_FORMAT_ID,
        "version": MANIFEST_FORMAT_VERSION,
        "release_tag": manifest.release_tag,
        "released_at": manifest.released_at,
        "signer_did": manifest.signer_did,
        "artifacts": [a.to_dict() for a in manifest.artifacts],
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def compute_manifest_id(manifest: ReleaseManifest) -> str:
    """SHA-256 hex of the signable payload."""
    return hashlib.sha256(signable_payload(manifest)).hexdigest()


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _validate_iso8601_utc(s: str) -> None:
    """Reject naive / non-UTC / malformed timestamps. Same rule as
    Wave 3's succession-statement validator: archival comparisons
    require unambiguous UTC."""
    if not isinstance(s, str) or not s:
        raise ReleaseManifestError(
            f"released_at must be a non-empty string; got {s!r}"
        )
    candidate = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError as e:
        raise ReleaseManifestError(
            f"released_at is not valid ISO 8601: {e}"
        ) from e
    if dt.tzinfo is None:
        raise ReleaseManifestError(
            f"released_at {s!r} is timezone-naive; must be UTC-explicit"
        )
    if dt.utcoffset() != timedelta(0):
        raise ReleaseManifestError(
            f"released_at {s!r} is not UTC (offset {dt.utcoffset()}); "
            f"manifest schema requires UTC"
        )


def _validate_artifact_path(path: str) -> None:
    """Refuse path-traversal and absolute paths.

    Verifiers iterate ``manifest.artifacts`` and look up each path on
    disk; an attacker who controls a manifest could otherwise direct
    the verifier at ``/etc/passwd`` or ``../../../something``. Refuse
    at construction time so a malformed manifest never gets signed.

    Windows drive-qualified paths (``C:\\Users\\...``, ``C:/Windows/...``)
    are also rejected (codex P2 round 2): they're absolute on Windows
    even though they don't start with ``/`` or ``\\``.
    """
    if not isinstance(path, str) or not path:
        raise ReleaseManifestError(f"artifact path must be a non-empty string; got {path!r}")
    if path.startswith("/") or path.startswith("\\"):
        raise ReleaseManifestError(f"artifact path must not be absolute: {path!r}")
    # Windows drive prefix: a single letter followed by ``:`` at the
    # very start (e.g. ``C:``, ``c:foo``, ``D:\\path``). Per Windows
    # path rules, ``X:`` is the volume-relative or absolute prefix.
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        raise ReleaseManifestError(
            f"artifact path must not be absolute (Windows drive prefix): {path!r}"
        )
    if ".." in path.replace("\\", "/").split("/"):
        raise ReleaseManifestError(f"artifact path must not contain '..': {path!r}")
    if "\x00" in path:
        raise ReleaseManifestError(f"artifact path must not contain NUL: {path!r}")


def new_manifest(
    *,
    release_tag: str,
    released_at: Optional[str] = None,
    signer_did: str = "",
) -> ReleaseManifest:
    """Start a fresh manifest. Add artifacts via :func:`add_artifact_entry`,
    then sign via :func:`sign_manifest`, then finalize via
    :func:`finalize`."""
    if not isinstance(release_tag, str) or not release_tag:
        raise ReleaseManifestError(
            f"release_tag must be a non-empty string; got {release_tag!r}"
        )
    if released_at is None:
        released_at = datetime.now(timezone.utc).isoformat()
    _validate_iso8601_utc(released_at)
    return ReleaseManifest(
        release_tag=release_tag,
        released_at=released_at,
        signer_did=signer_did,
    )


def _validate_manifest_invariants(manifest: ReleaseManifest) -> None:
    """Re-run construction-time invariants on a parsed/constructed
    manifest. Codex P2: ``from_dict`` previously trusted whatever was
    in the JSON, so a signed manifest with ``../escape`` paths could
    be accepted by ``verify_manifest`` and drive downstream artifact
    lookups outside the release directory.

    Run this before signing AND inside the verifier so neither path
    can produce/consume a malformed manifest.
    """
    if not isinstance(manifest.release_tag, str) or not manifest.release_tag:
        raise ReleaseManifestError(
            f"manifest.release_tag must be a non-empty string; "
            f"got {manifest.release_tag!r}"
        )
    _validate_iso8601_utc(manifest.released_at)
    seen_paths = set()
    for i, a in enumerate(manifest.artifacts):
        if not isinstance(a, ArtifactEntry):
            raise ReleaseManifestError(
                f"artifacts[{i}] must be an ArtifactEntry; got {type(a).__name__}"
            )
        _validate_artifact_path(a.path)
        if a.path in seen_paths:
            raise ReleaseManifestError(
                f"duplicate artifact path: {a.path!r}"
            )
        seen_paths.add(a.path)
        if not isinstance(a.sha256, str) or len(a.sha256) != 64:
            raise ReleaseManifestError(
                f"artifacts[{i}].sha256 must be 64-char hex; got {a.sha256!r}"
            )
        try:
            int(a.sha256, 16)
        except ValueError as e:
            raise ReleaseManifestError(
                f"artifacts[{i}].sha256 is not valid hex: {e}"
            ) from e
        if not isinstance(a.size, int) or a.size < 0:
            raise ReleaseManifestError(
                f"artifacts[{i}].size must be a non-negative int; got {a.size!r}"
            )


def add_artifact_entry(
    manifest: ReleaseManifest,
    path: str,
    content: bytes,
) -> ReleaseManifest:
    """Append an artifact to ``manifest`` by hashing ``content``.

    Returns a new manifest; the input is unchanged (frozen dataclass).
    Refuses duplicate paths so a producer can't accidentally include
    the same file twice with different hashes.
    """
    _validate_artifact_path(path)
    if not isinstance(content, (bytes, bytearray)):
        raise ReleaseManifestError(
            f"content must be bytes; got {type(content).__name__}"
        )
    for existing in manifest.artifacts:
        if existing.path == path:
            raise ReleaseManifestError(
                f"duplicate artifact path: {path!r}"
            )
    sha256 = hashlib.sha256(bytes(content)).hexdigest()
    entry = ArtifactEntry(path=path, sha256=sha256, size=len(content))
    return replace(manifest, artifacts=list(manifest.artifacts) + [entry])


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

def sign_manifest(
    manifest: ReleaseManifest,
    keypair: Keypair,
    kid: str,
) -> ReleaseManifest:
    """Apply a signature using ``keypair``. Multiple signatures are
    supported by chaining calls (e.g. SLH-DSA + Ed25519 hybrid).

    Re-validates manifest invariants before signing so a producer
    cannot accidentally produce a signed-but-malformed manifest
    (codex P2 review).
    """
    if not isinstance(kid, str) or not kid:
        raise ReleaseManifestError(f"kid must be a non-empty string; got {kid!r}")
    _validate_manifest_invariants(manifest)
    suite = get_suite(keypair.suite_id)
    payload = signable_payload(manifest)
    sig = suite.sign(payload, keypair.private_key)
    new_sigs = list(manifest.signatures) + [{
        "alg": suite.alg_id,
        "kid": kid,
        "sig": sig.hex(),
    }]
    return replace(manifest, signatures=new_sigs)


def finalize(manifest: ReleaseManifest) -> ReleaseManifest:
    """Stamp ``manifest_id``. Excluded from the signable payload, so
    calling this AFTER signing does not invalidate signatures."""
    return replace(manifest, manifest_id=compute_manifest_id(manifest))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifestVerifyResult:
    ok: bool
    signature_policy: PolicyResult
    manifest_id_consistent: bool
    signer_match: bool
    reason: str


def verify_manifest(
    manifest: ReleaseManifest,
    *,
    trusted_signer_multibase: Optional[str] = None,
    trusted_signer_multibases: Optional[List[str]] = None,
    trusted_signer_alg: str = ALG_SLH_DSA_SHA2_128S,
    policy: VerifyPolicy = VerifyPolicy.PQ_REQUIRED,
) -> ManifestVerifyResult:
    """Verify a manifest's signatures against pinned trusted signers.

    Args:
        manifest: the parsed manifest (e.g. via ``ReleaseManifest.from_dict(json.loads(s))``)
        trusted_signer_multibase: legacy single-key form. Equivalent
            to ``trusted_signer_multibases=[trusted_signer_multibase]``.
        trusted_signer_multibases: list of pinned public keys. Each
            decoded suite-id must be unique within the list. Codex P2
            round 3: the previous single-key API made the documented
            ``HYBRID_REQUIRED`` policy functionally unsatisfiable
            because only one alg ever made it into the verified set.
            Pass one trusted key per algorithm to use a hybrid policy.
        trusted_signer_alg: required suite for the SINGLE-key
            signature path. Ignored when ``trusted_signer_multibases``
            is provided (each key's alg is checked individually).
            Default ``slh-dsa-sha2-128s``.
        policy: signature policy. Default ``PQ_REQUIRED`` — release
            signatures are long-horizon and classical-only would be
            Shor-vulnerable. ``HYBRID_REQUIRED`` requires at least
            one classical AND one PQ trusted key (not just trusted
            signature claim).

    Returns:
        ``ManifestVerifyResult`` with composite ``ok``, the per-policy
        result, ``signer_match`` (at least one trusted key actually
        verified a signature), and ``manifest_id_consistent``.
    """
    # Normalize to a list of (multibase, expected_alg or None) entries.
    if trusted_signer_multibases is not None and trusted_signer_multibase is not None:
        return ManifestVerifyResult(
            ok=False,
            signature_policy=PolicyResult(
                ok=False, reason="conflicting trusted-signer arguments",
                alg_ids_seen=frozenset(),
            ),
            manifest_id_consistent=False,
            signer_match=False,
            reason=(
                "pass either trusted_signer_multibase (single) OR "
                "trusted_signer_multibases (list), not both"
            ),
        )
    if trusted_signer_multibases is None:
        if trusted_signer_multibase is None:
            return ManifestVerifyResult(
                ok=False,
                signature_policy=PolicyResult(
                    ok=False, reason="no trusted signer provided",
                    alg_ids_seen=frozenset(),
                ),
                manifest_id_consistent=False,
                signer_match=False,
                reason="must provide trusted_signer_multibase or trusted_signer_multibases",
            )
        trusted_signer_multibases = [trusted_signer_multibase]
        # Single-key mode also enforces the alg pin
        single_key_expected_alg = trusted_signer_alg
    else:
        single_key_expected_alg = None
    # Re-run invariants on the parsed/passed manifest. Without this a
    # signed manifest with ``../escape`` paths could verify ok=True
    # and drive consumer code outside the release directory (codex P2).
    try:
        _validate_manifest_invariants(manifest)
    except ReleaseManifestError as e:
        return ManifestVerifyResult(
            ok=False,
            signature_policy=PolicyResult(
                ok=False, reason="manifest invariant violated",
                alg_ids_seen=frozenset(),
            ),
            manifest_id_consistent=False,
            signer_match=False,
            reason=f"manifest invariant violated: {e}",
        )

    payload = signable_payload(manifest)

    # Decode every trusted signer's public key. Index by alg_id so we
    # can look up the right key when verifying a signature entry.
    # Refuse duplicates (two keys for the same alg) — that ambiguity
    # has no good resolution.
    trusted_keys_by_alg: dict = {}
    for i, mb in enumerate(trusted_signer_multibases):
        try:
            ts, tp = multibase_to_public_key(mb)
        except (CryptoSuiteError, ValueError) as e:
            return ManifestVerifyResult(
                ok=False,
                signature_policy=PolicyResult(
                    ok=False, reason="trusted_signer_multibase decode failed",
                    alg_ids_seen=frozenset(),
                ),
                manifest_id_consistent=False,
                signer_match=False,
                reason=f"trusted_signer_multibase[{i}] decode failed: {e}",
            )
        if ts.alg_id in trusted_keys_by_alg:
            return ManifestVerifyResult(
                ok=False,
                signature_policy=PolicyResult(
                    ok=False, reason="duplicate trusted alg",
                    alg_ids_seen=frozenset(),
                ),
                manifest_id_consistent=False,
                signer_match=False,
                reason=(
                    f"two trusted keys for alg {ts.alg_id!r}; refusing "
                    f"because there is no good way to disambiguate"
                ),
            )
        trusted_keys_by_alg[ts.alg_id] = (ts, tp)

    # Single-key mode pins the alg; multi-key mode trusts each key
    # under its own decoded alg.
    if single_key_expected_alg is not None:
        # Exactly one key in trusted_keys_by_alg
        only_alg = next(iter(trusted_keys_by_alg))
        if only_alg != single_key_expected_alg:
            return ManifestVerifyResult(
                ok=False,
                signature_policy=PolicyResult(
                    ok=False, reason="trusted_signer_alg mismatch",
                    alg_ids_seen=frozenset(),
                ),
                manifest_id_consistent=False,
                signer_match=False,
                reason=(
                    f"trusted_signer_multibase resolves to alg "
                    f"{only_alg!r}, but trusted_signer_alg="
                    f"{single_key_expected_alg!r}"
                ),
            )

    # Verify each signature; collect verified entries. Be defensive
    # about non-string fields — codex P2 round 2 flagged that
    # ``ReleaseManifest.from_dict`` doesn't strictly validate the
    # ``signatures`` array shape, so a manifest carrying e.g.
    # ``{"alg": ["list"], "sig": [1,2]}`` could TypeError out of
    # ``bytes.fromhex`` or ``get_suite`` rather than producing a
    # structured failure.
    verified: List[dict] = []
    signer_match = False
    for entry in manifest.signatures:
        if not isinstance(entry, Mapping):
            continue
        alg = entry.get("alg")
        sig_hex = entry.get("sig")
        if not isinstance(alg, str) or not alg:
            continue
        if not isinstance(sig_hex, str) or not sig_hex:
            continue
        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except (ValueError, TypeError):
            continue
        try:
            suite = get_suite(alg)
        except (CryptoSuiteError, TypeError):
            continue
        # Attempt verify against each trusted key whose alg matches.
        # Multi-key mode (codex P2 round 3) lets HYBRID_REQUIRED
        # actually be satisfiable: if the manifest carries SLH-DSA
        # AND Ed25519 sigs and the caller provided trusted keys for
        # both, both verified entries land in ``verified``.
        trusted = trusted_keys_by_alg.get(alg)
        if trusted is None:
            # Manifest carries a sig with an alg we don't have a
            # trusted key for. Don't substitute it.
            continue
        ts_suite, ts_pub = trusted
        if ts_suite.verify(payload, sig_bytes, ts_pub):
            verified.append(dict(entry))
            signer_match = True

    policy_result = evaluate_signatures(verified, policy)

    # Manifest_id integrity
    expected_id = compute_manifest_id(manifest)
    id_ok = bool(manifest.manifest_id) and manifest.manifest_id == expected_id

    composite_ok = signer_match and policy_result.ok and id_ok

    if composite_ok:
        reason = "release manifest verified"
    else:
        parts = []
        if not signer_match:
            parts.append("trusted signer did not verify any signature")
        if not policy_result.ok:
            parts.append(f"policy: {policy_result.reason}")
        if not id_ok:
            if not manifest.manifest_id:
                parts.append("manifest_id is empty (call finalize() before verify)")
            else:
                parts.append(
                    f"manifest_id mismatch: stored={manifest.manifest_id!r} "
                    f"computed={expected_id!r}"
                )
        reason = "; ".join(parts) or "unknown failure"

    return ManifestVerifyResult(
        ok=composite_ok,
        signature_policy=policy_result,
        manifest_id_consistent=id_ok,
        signer_match=signer_match,
        reason=reason,
    )


def verify_artifact_bytes(
    manifest: ReleaseManifest,
    path: str,
    content: bytes,
) -> bool:
    """Confirm that ``content`` matches the manifest's recorded hash
    for ``path``.

    Returns True iff the artifact entry exists AND content's SHA-256
    matches AND its byte length matches. Used after :func:`verify_manifest`:
    a manifest's signature only attests to the recorded hashes; the
    consumer must separately verify each downloaded artifact byte
    against its hash.
    """
    if not isinstance(content, (bytes, bytearray)):
        return False
    actual = hashlib.sha256(bytes(content)).hexdigest()
    actual_size = len(content)
    for entry in manifest.artifacts:
        if entry.path == path:
            return entry.sha256 == actual and entry.size == actual_size
    return False


__all__ = [
    "ArtifactEntry",
    "MANIFEST_FORMAT_ID",
    "MANIFEST_FORMAT_VERSION",
    "ManifestVerifyResult",
    "ReleaseManifest",
    "ReleaseManifestError",
    "add_artifact_entry",
    "compute_manifest_id",
    "finalize",
    "new_manifest",
    "sign_manifest",
    "signable_payload",
    "verify_artifact_bytes",
    "verify_manifest",
]
