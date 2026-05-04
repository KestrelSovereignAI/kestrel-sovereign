"""
Release manifest tests — Wave 5 sub-PR 1 (#920).

Covers:
- new_manifest / add_artifact_entry / sign_manifest / finalize round-trip
- Canonical signable payload byte-stable across to_dict/from_dict
- manifest_id stable across JSON round-trips
- Validators:
  * release_tag must be non-empty
  * released_at must be UTC ISO 8601 (naive / non-UTC / malformed rejected)
  * artifact path: empty, absolute, ".." segments, NUL byte rejected
  * duplicate artifact path rejected
- Signing: SLH-DSA primary; multiple sigs (e.g. SLH-DSA + Ed25519) supported
- verify_manifest:
  * Happy path with pinned trusted signer
  * Wrong trusted_signer_multibase fails
  * Trusted_signer_alg mismatch fails (trusted key is different alg
    than declared)
  * Tampered manifest field invalidates signatures
  * Tampered artifact entry invalidates signatures
  * Missing manifest_id (unfinalized) rejected
  * Tampered manifest_id rejected
  * Other-alg signatures (Ed25519 alongside SLH-DSA) don't substitute
    for the trusted signer's signature
- verify_artifact_bytes: matches/mismatches by path, hash, and size
"""

from __future__ import annotations

import hashlib
import json

import pytest

from kestrel_sovereign.security.crypto_suite import (
    ALG_ED25519,
    ALG_SLH_DSA_SHA2_128S,
    Ed25519Suite,
    SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.multikey import public_key_to_multibase
from kestrel_sovereign.security.release_manifest import (
    MANIFEST_FORMAT_ID,
    MANIFEST_FORMAT_VERSION,
    ReleaseManifest,
    ReleaseManifestError,
    add_artifact_entry,
    compute_manifest_id,
    finalize,
    new_manifest,
    sign_manifest,
    signable_payload,
    verify_artifact_bytes,
    verify_manifest,
)
from kestrel_sovereign.security.verify_policy import VerifyPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def slh_kp():
    return SLHDSASHA2128sSuite().generate_keypair()


@pytest.fixture(scope="module")
def slh_pub_multibase(slh_kp):
    return public_key_to_multibase(SLHDSASHA2128sSuite(), slh_kp.public_key)


@pytest.fixture(scope="module")
def ed_kp():
    return Ed25519Suite().generate_keypair()


@pytest.fixture
def base_manifest():
    return new_manifest(
        release_tag="v1.2.3",
        released_at="2026-05-04T20:00:00+00:00",
        signer_did="did:web:kestrel-sovereign.example",
    )


@pytest.fixture
def signed_manifest(base_manifest, slh_kp):
    """A representative signed-and-finalized manifest with two artifacts."""
    m = add_artifact_entry(base_manifest, "wheel.whl", b"wheel-bytes")
    m = add_artifact_entry(m, "src.tar.gz", b"tarball-bytes")
    m = sign_manifest(m, slh_kp, kid="release-key-1")
    return finalize(m)


# ---------------------------------------------------------------------------
# Construction validators
# ---------------------------------------------------------------------------

def test_new_manifest_requires_release_tag():
    with pytest.raises(ReleaseManifestError, match="release_tag"):
        new_manifest(release_tag="", released_at="2026-05-04T20:00:00+00:00")


def test_new_manifest_default_released_at_is_utc():
    """No released_at → now (UTC, tz-aware)."""
    m = new_manifest(release_tag="v0.0.1")
    assert m.released_at.endswith("+00:00") or m.released_at.endswith("Z")


def test_naive_released_at_rejected():
    with pytest.raises(ReleaseManifestError, match="timezone-naive"):
        new_manifest(release_tag="v1", released_at="2026-05-04T20:00:00")


def test_non_utc_released_at_rejected():
    with pytest.raises(ReleaseManifestError, match="not UTC"):
        new_manifest(release_tag="v1", released_at="2026-05-04T20:00:00+05:00")


def test_malformed_released_at_rejected():
    with pytest.raises(ReleaseManifestError, match="not valid ISO 8601"):
        new_manifest(release_tag="v1", released_at="not-a-date")


# ---------------------------------------------------------------------------
# Artifact validators
# ---------------------------------------------------------------------------

def test_artifact_path_must_be_relative(base_manifest):
    with pytest.raises(ReleaseManifestError, match="absolute"):
        add_artifact_entry(base_manifest, "/etc/passwd", b"x")


def test_artifact_path_no_dot_dot(base_manifest):
    with pytest.raises(ReleaseManifestError, match="'\\.\\.'"):
        add_artifact_entry(base_manifest, "../escape", b"x")


def test_artifact_path_no_dot_dot_in_middle(base_manifest):
    with pytest.raises(ReleaseManifestError, match="'\\.\\.'"):
        add_artifact_entry(base_manifest, "ok/../escape", b"x")


def test_artifact_path_no_nul_byte(base_manifest):
    with pytest.raises(ReleaseManifestError, match="NUL"):
        add_artifact_entry(base_manifest, "ok\x00.whl", b"x")


def test_artifact_path_empty_rejected(base_manifest):
    with pytest.raises(ReleaseManifestError, match="non-empty"):
        add_artifact_entry(base_manifest, "", b"x")


def test_artifact_duplicate_path_rejected(base_manifest):
    m = add_artifact_entry(base_manifest, "wheel.whl", b"a")
    with pytest.raises(ReleaseManifestError, match="duplicate artifact path"):
        add_artifact_entry(m, "wheel.whl", b"b")


def test_artifact_content_must_be_bytes(base_manifest):
    with pytest.raises(ReleaseManifestError, match="must be bytes"):
        add_artifact_entry(base_manifest, "wheel.whl", "not-bytes")  # type: ignore[arg-type]


def test_artifact_records_correct_hash_and_size(base_manifest):
    content = b"hello world"
    m = add_artifact_entry(base_manifest, "hello.txt", content)
    assert len(m.artifacts) == 1
    entry = m.artifacts[0]
    assert entry.path == "hello.txt"
    assert entry.sha256 == hashlib.sha256(content).hexdigest()
    assert entry.size == len(content)


# ---------------------------------------------------------------------------
# Canonical payload + manifest_id
# ---------------------------------------------------------------------------

def test_signable_payload_is_deterministic(base_manifest):
    a = signable_payload(base_manifest)
    b = signable_payload(base_manifest)
    assert a == b


def test_signable_payload_excludes_signatures_and_id(base_manifest, slh_kp):
    """Signing then finalizing must not change the signable payload."""
    before = signable_payload(base_manifest)
    signed = sign_manifest(base_manifest, slh_kp, kid="k1")
    after_sign = signable_payload(signed)
    final = finalize(signed)
    after_final = signable_payload(final)
    assert before == after_sign == after_final


def test_signable_payload_changes_when_signed_field_changes(base_manifest):
    other = new_manifest(
        release_tag="v9.9.9",
        released_at="2026-05-04T20:00:00+00:00",
    )
    assert signable_payload(other) != signable_payload(base_manifest)


def test_manifest_id_round_trips_through_dict(signed_manifest):
    raw = json.dumps(signed_manifest.to_dict())
    rehydrated = ReleaseManifest.from_dict(json.loads(raw))
    assert compute_manifest_id(rehydrated) == compute_manifest_id(signed_manifest)


def test_from_dict_rejects_unknown_format(base_manifest):
    bad = base_manifest.to_dict()
    bad["format"] = "kestrel-release-manifest-v999"
    with pytest.raises(ReleaseManifestError, match="unknown manifest format"):
        ReleaseManifest.from_dict(bad)


def test_from_dict_rejects_unknown_version(base_manifest):
    bad = base_manifest.to_dict()
    bad["version"] = 99
    with pytest.raises(ReleaseManifestError, match="unknown manifest version"):
        ReleaseManifest.from_dict(bad)


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

def test_sign_manifest_appends_entry(base_manifest, slh_kp):
    m = sign_manifest(base_manifest, slh_kp, kid="release-key-1")
    assert len(m.signatures) == 1
    entry = m.signatures[0]
    assert entry["alg"] == ALG_SLH_DSA_SHA2_128S
    assert entry["kid"] == "release-key-1"
    bytes.fromhex(entry["sig"])  # valid hex


def test_sign_manifest_rejects_empty_kid(base_manifest, slh_kp):
    with pytest.raises(ReleaseManifestError, match="kid"):
        sign_manifest(base_manifest, slh_kp, kid="")


def test_multiple_signatures_chain(base_manifest, slh_kp, ed_kp):
    """Hybrid release: SLH-DSA primary + Ed25519 for transitional verifiers."""
    m = sign_manifest(base_manifest, slh_kp, kid="release-key-1")
    m = sign_manifest(m, ed_kp, kid="ed25519-aux")
    assert len(m.signatures) == 2
    algs = {s["alg"] for s in m.signatures}
    assert algs == {ALG_SLH_DSA_SHA2_128S, ALG_ED25519}


# ---------------------------------------------------------------------------
# verify_manifest — happy path
# ---------------------------------------------------------------------------

def test_verify_happy_path(signed_manifest, slh_pub_multibase):
    result = verify_manifest(
        signed_manifest,
        trusted_signer_multibase=slh_pub_multibase,
    )
    assert result.ok, result.reason
    assert result.signer_match
    assert result.manifest_id_consistent


def test_verify_with_hybrid_signatures_uses_pinned_alg(
    base_manifest, slh_kp, ed_kp, slh_pub_multibase,
):
    """A manifest with both SLH-DSA and Ed25519 signatures verifies
    cleanly when the trusted signer is the SLH-DSA key. The Ed25519
    entry is accepted by the policy enumerator but isn't verified
    against the trusted key (different alg)."""
    m = add_artifact_entry(base_manifest, "wheel.whl", b"data")
    m = sign_manifest(m, slh_kp, kid="slh-1")
    m = sign_manifest(m, ed_kp, kid="ed-1")
    m = finalize(m)
    result = verify_manifest(m, trusted_signer_multibase=slh_pub_multibase)
    assert result.ok, result.reason


# ---------------------------------------------------------------------------
# verify_manifest — failure modes
# ---------------------------------------------------------------------------

def test_verify_wrong_trusted_signer_fails(signed_manifest):
    """Different SLH-DSA key → trusted signer doesn't verify any sig."""
    other_kp = SLHDSASHA2128sSuite().generate_keypair()
    other_mb = public_key_to_multibase(SLHDSASHA2128sSuite(), other_kp.public_key)
    result = verify_manifest(signed_manifest, trusted_signer_multibase=other_mb)
    assert not result.ok
    assert not result.signer_match


def test_verify_trusted_signer_alg_mismatch_fails(signed_manifest, ed_kp):
    """Caller pins an Ed25519 multibase but says trusted_signer_alg=
    slh-dsa-sha2-128s. The decoder resolves alg ed25519, mismatch
    detected before signature verification."""
    ed_mb = public_key_to_multibase(Ed25519Suite(), ed_kp.public_key)
    result = verify_manifest(
        signed_manifest,
        trusted_signer_multibase=ed_mb,
        trusted_signer_alg=ALG_SLH_DSA_SHA2_128S,
    )
    assert not result.ok
    assert "alg" in result.reason


def test_verify_tampered_release_tag_fails(signed_manifest, slh_pub_multibase):
    from dataclasses import replace
    tampered = replace(signed_manifest, release_tag="v9.9.9")
    result = verify_manifest(tampered, trusted_signer_multibase=slh_pub_multibase)
    assert not result.ok


def test_verify_tampered_artifact_entry_fails(signed_manifest, slh_pub_multibase):
    from dataclasses import replace
    from kestrel_sovereign.security.release_manifest import ArtifactEntry
    bad_artifact = ArtifactEntry(path="wheel.whl", sha256="0" * 64, size=999)
    tampered = replace(
        signed_manifest,
        artifacts=[bad_artifact] + list(signed_manifest.artifacts[1:]),
    )
    result = verify_manifest(tampered, trusted_signer_multibase=slh_pub_multibase)
    assert not result.ok


def test_verify_unfinalized_manifest_fails(base_manifest, slh_kp, slh_pub_multibase):
    """Empty manifest_id is no longer treated as 'consistent' — the
    same strictness as Wave 3's succession verifier."""
    m = sign_manifest(base_manifest, slh_kp, kid="k1")
    # NOT calling finalize() — manifest_id stays empty
    assert not m.manifest_id
    result = verify_manifest(m, trusted_signer_multibase=slh_pub_multibase)
    assert not result.ok
    assert "manifest_id is empty" in result.reason


def test_verify_tampered_manifest_id_fails(signed_manifest, slh_pub_multibase):
    from dataclasses import replace
    spoofed = replace(signed_manifest, manifest_id="0" * 64)
    result = verify_manifest(spoofed, trusted_signer_multibase=slh_pub_multibase)
    assert not result.ok
    assert not result.manifest_id_consistent


def test_other_alg_signatures_dont_substitute_for_trusted_key(
    base_manifest, ed_kp, slh_pub_multibase,
):
    """A manifest signed ONLY with Ed25519 must NOT verify against an
    SLH-DSA pinned signer, even though Ed25519 is registered. This is
    the key distinction codex-aware verifier code must enforce: pinned
    trust beats present-but-untrusted signatures."""
    m = add_artifact_entry(base_manifest, "wheel.whl", b"data")
    m = sign_manifest(m, ed_kp, kid="ed-1")
    m = finalize(m)
    result = verify_manifest(m, trusted_signer_multibase=slh_pub_multibase)
    assert not result.ok
    assert not result.signer_match


# ---------------------------------------------------------------------------
# verify_artifact_bytes
# ---------------------------------------------------------------------------

def test_verify_artifact_bytes_match(signed_manifest):
    assert verify_artifact_bytes(signed_manifest, "wheel.whl", b"wheel-bytes")
    assert verify_artifact_bytes(signed_manifest, "src.tar.gz", b"tarball-bytes")


def test_verify_artifact_bytes_hash_mismatch(signed_manifest):
    assert not verify_artifact_bytes(signed_manifest, "wheel.whl", b"WRONG-bytes")


def test_verify_artifact_bytes_size_mismatch(signed_manifest):
    """Even if the truncated content somehow had the same hash (which
    SHA-256 makes infeasible), the size check independently catches it.
    Using zero-length here for clarity."""
    assert not verify_artifact_bytes(signed_manifest, "wheel.whl", b"")


def test_verify_artifact_bytes_unknown_path(signed_manifest):
    assert not verify_artifact_bytes(signed_manifest, "not-in-manifest.txt", b"x")


def test_verify_artifact_bytes_rejects_non_bytes(signed_manifest):
    assert not verify_artifact_bytes(signed_manifest, "wheel.whl", "not-bytes")  # type: ignore[arg-type]
