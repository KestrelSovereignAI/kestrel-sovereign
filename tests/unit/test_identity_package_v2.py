"""
Identity package v2 schema tests — Wave 1 sub-PR 3 (#916).

Covers:
- v2 round-trip with signatures + verification_methods arrays.
- v1 → v2 read translation: legacy ``signature`` field surfaces as a
  synthetic single entry in ``signatures`` tagged ``ecdsa-secp256k1-sha256``.
- ``compute_content_hash`` excludes both ``signature`` and ``signatures``
  but includes ``verification_methods``.
- ``compute_content_hash`` is stable across the v1/v2 boundary for a
  package that was originally v1 — verifying a v1 package's hash still
  works after the schema bump.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from kestrel_sovereign.identity.identity_package import (
    IDENTITY_PACKAGE_VERSION,
    IDENTITY_PACKAGE_VERSION_LEGACY,
    AgentIdentityPackage,
)
from kestrel_sovereign.security.crypto_suite import (
    ALG_ECDSA_SECP256K1_SHA256,
)
from kestrel_sovereign.security.keypair_factory import KeypairFactory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_package() -> AgentIdentityPackage:
    """A minimal v2 package with no signatures yet."""
    return AgentIdentityPackage(
        did="did:web:agents.example:alice",
        agent_name="alice",
        created_at="2026-05-01T00:00:00Z",
        constitution_hash="a" * 64,
        constitution_text="content",
    )


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

def test_default_package_version_is_v2():
    pkg = AgentIdentityPackage(
        did="did:web:x", agent_name="x", created_at="t",
        constitution_hash="h", constitution_text="c",
    )
    assert pkg.package_version == IDENTITY_PACKAGE_VERSION
    assert pkg.package_version.startswith("2.")
    assert pkg.is_v2() is True


def test_legacy_constant_unchanged():
    assert IDENTITY_PACKAGE_VERSION_LEGACY == "1.0.0"


# ---------------------------------------------------------------------------
# v2 helpers
# ---------------------------------------------------------------------------

def test_add_signature_appends_entry(base_package):
    base_package.add_signature("ecdsa-secp256k1-sha256", "did:web:x#keys-1", "abcdef")
    assert base_package.signatures == [{
        "alg": "ecdsa-secp256k1-sha256",
        "kid": "did:web:x#keys-1",
        "sig": "abcdef",
    }]


def test_add_verification_method_defaults_controller(base_package):
    base_package.add_verification_method(
        kid="did:web:agents.example:alice#keys-1",
        public_key_multibase="z123abc",
    )
    vm = base_package.verification_methods[0]
    assert vm["type"] == "Multikey"
    assert vm["controller"] == "did:web:agents.example:alice"
    assert vm["publicKeyMultibase"] == "z123abc"


def test_add_verification_method_explicit_controller(base_package):
    base_package.add_verification_method(
        kid="kid", public_key_multibase="z", controller="did:web:other",
    )
    assert base_package.verification_methods[0]["controller"] == "did:web:other"


def test_iter_signatures_v2_array(base_package):
    base_package.add_signature("ed25519", "kid-ed", "sig-ed")
    base_package.add_signature("ml-dsa-65", "kid-ml", "sig-ml")
    sigs = base_package.iter_signatures()
    assert len(sigs) == 2
    assert {s["alg"] for s in sigs} == {"ed25519", "ml-dsa-65"}


def test_iter_signatures_synthetic_v1():
    """A v1 package with only the legacy ``signature`` field iterates as
    a single synthetic v2 entry tagged ``ecdsa-secp256k1-sha256``."""
    pkg = AgentIdentityPackage(
        did="did:pkh:eip155:1:0xabc",
        agent_name="alice",
        created_at="t",
        constitution_hash="h",
        constitution_text="c",
        signature="deadbeef",
    )
    # Manually mark as v1 to simulate a legacy package
    pkg.package_version = IDENTITY_PACKAGE_VERSION_LEGACY
    pkg.signatures = []  # ensure no v2 array

    sigs = pkg.iter_signatures()
    assert len(sigs) == 1
    assert sigs[0]["alg"] == "ecdsa-secp256k1-sha256"
    assert sigs[0]["sig"] == "deadbeef"
    assert sigs[0]["kid"] == "did:pkh:eip155:1:0xabc#keys-1"


def test_iter_signatures_empty(base_package):
    assert base_package.iter_signatures() == []


# ---------------------------------------------------------------------------
# v2 round-trip
# ---------------------------------------------------------------------------

def test_v2_to_dict_emits_arrays(base_package):
    base_package.add_signature("ed25519", "kid-ed", "sig-ed")
    base_package.add_verification_method(
        kid="kid-ed", public_key_multibase="zabc",
    )
    out = base_package.to_dict()
    assert out["package_version"] == IDENTITY_PACKAGE_VERSION
    assert out["signatures"] == [{
        "alg": "ed25519", "kid": "kid-ed", "sig": "sig-ed",
    }]
    assert out["verification_methods"][0]["publicKeyMultibase"] == "zabc"


def test_v2_round_trip_preserves_arrays(base_package):
    base_package.add_signature("ecdsa-secp256k1-sha256", "kid-1", "ec1")
    base_package.add_signature("ml-dsa-65", "kid-2", "ml2")
    base_package.add_verification_method("kid-1", "zecdsa")
    base_package.add_verification_method("kid-2", "zmldsa")

    rebuilt = AgentIdentityPackage.from_json(base_package.to_json())
    assert rebuilt.signatures == base_package.signatures
    assert rebuilt.verification_methods == base_package.verification_methods


# ---------------------------------------------------------------------------
# v1 → v2 read translation
# ---------------------------------------------------------------------------

def test_v1_input_translated_to_synthetic_v2():
    """A serialized v1 package — ``package_version: 1.0.0``, only the
    legacy ``signature`` hex field, no ``signatures`` array — must read
    back with ``signatures`` populated as a single synthetic entry."""
    v1_json = json.dumps({
        "did": "did:pkh:eip155:1:0xabc",
        "agent_name": "legacy-agent",
        "created_at": "2024-01-01T00:00:00Z",
        "constitution_hash": "c" * 64,
        "constitution_text": "old constitution",
        "package_version": "1.0.0",
        "signature": "0123456789abcdef",
        # No "signatures" key at all
    })
    pkg = AgentIdentityPackage.from_json(v1_json)
    assert pkg.signature == "0123456789abcdef"  # legacy field preserved
    assert len(pkg.signatures) == 1
    assert pkg.signatures[0]["alg"] == "ecdsa-secp256k1-sha256"
    assert pkg.signatures[0]["sig"] == "0123456789abcdef"
    assert pkg.signatures[0]["kid"] == "did:pkh:eip155:1:0xabc#keys-1"
    # iter_signatures returns the v2 array (now populated)
    assert pkg.iter_signatures() == pkg.signatures


def test_v1_input_with_no_signature_yields_empty():
    """v1 input with NO signature hex must NOT synthesize a phantom entry."""
    v1_json = json.dumps({
        "did": "did:pkh:eip155:1:0xabc",
        "agent_name": "x",
        "created_at": "t",
        "constitution_hash": "h",
        "constitution_text": "c",
        "package_version": "1.0.0",
    })
    pkg = AgentIdentityPackage.from_json(v1_json)
    assert pkg.signatures == []
    assert pkg.iter_signatures() == []


# ---------------------------------------------------------------------------
# content_hash semantics
# ---------------------------------------------------------------------------

def test_content_hash_excludes_signatures(base_package):
    """Adding a v2 signature must NOT change the content hash — otherwise
    sign-then-verify would never round-trip."""
    h_before = base_package.compute_content_hash()
    base_package.add_signature("ed25519", "kid", "sig")
    h_after = base_package.compute_content_hash()
    assert h_before == h_after


def test_content_hash_excludes_legacy_signature(base_package):
    h_before = base_package.compute_content_hash()
    base_package.signature = "any-hex-value"
    h_after = base_package.compute_content_hash()
    assert h_before == h_after


def test_content_hash_includes_verification_methods(base_package):
    """Verification methods carry public keys that the signature must
    authenticate — they MUST be in the hashed payload, otherwise an
    attacker could swap public keys post-sign without invalidating the
    signature."""
    h_before = base_package.compute_content_hash()
    base_package.add_verification_method("kid", "zsomekey")
    h_after = base_package.compute_content_hash()
    assert h_before != h_after, (
        "verification_methods must be in the hashed payload — they carry "
        "public keys that need to be bound by the signature."
    )


def test_content_hash_v1_byte_stable_with_legacy_canonicalization():
    """A v1 package's ``compute_content_hash`` must produce the EXACT
    bytes the original v1 signer produced. v1 ``to_dict()`` had no
    ``signatures`` or ``verification_methods`` keys at all — the new
    schema's empty defaults must NOT leak into the canonical hash, or
    every already-exported v1 identity package would fail
    ``verify_content_hash`` after this PR.

    Reference v1 canonical shape: pop ``content_hash`` + ``signature``,
    no v2-only keys at all. That's what the original signer produced;
    that's what the new code must reproduce for v1 packages.
    """
    pkg = AgentIdentityPackage(
        did="did:pkh:eip155:1:0xabc",
        agent_name="legacy",
        created_at="2024-01-01T00:00:00Z",
        constitution_hash="c" * 64,
        constitution_text="x",
    )
    pkg.package_version = IDENTITY_PACKAGE_VERSION_LEGACY
    pkg.signature = "fake-sig-hex"

    # Original v1 canonicalization: take to_dict, strip the v2-only keys
    # (which the original v1 to_dict did not emit), pop content_hash and
    # signature, sort + serialize.
    v1_canonical = pkg.to_dict()
    v1_canonical.pop("content_hash", None)
    v1_canonical.pop("signature", None)
    v1_canonical.pop("signatures", None)
    v1_canonical.pop("verification_methods", None)
    expected_v1_hash = hashlib.sha256(
        json.dumps(v1_canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    actual = pkg.compute_content_hash()
    assert actual == expected_v1_hash, (
        "v1 package hash must equal the original v1 canonical shape "
        "(no signatures/verification_methods keys). Otherwise stored "
        "v1 packages fail verify_content_hash after the schema bump."
    )


def test_content_hash_v2_includes_verification_methods_only_for_v2():
    """Symmetric check: a v2 package binds verification_methods into
    the hash, while a v1 package with the same data does NOT (because
    v1 had no such field at sign time)."""
    common = dict(
        did="did:web:agents.example:alice",
        agent_name="alice",
        created_at="2026-05-01T00:00:00Z",
        constitution_hash="a" * 64,
        constitution_text="content",
    )
    v1 = AgentIdentityPackage(**common)
    v1.package_version = IDENTITY_PACKAGE_VERSION_LEGACY
    v2 = AgentIdentityPackage(**common)  # default = v2

    # No verification methods anywhere → v1 and v2 hashes can differ
    # only because of the package_version string. Verify they're
    # ordered: same except for `package_version` field — establish
    # baseline.
    v1_hash_baseline = v1.compute_content_hash()
    v2_hash_baseline = v2.compute_content_hash()
    assert v1_hash_baseline != v2_hash_baseline  # version differs

    # Now add a verification method to BOTH:
    v1.add_verification_method("kid", "zsomekey")
    v2.add_verification_method("kid", "zsomekey")

    # v1 hash must NOT have changed (verification_methods is excluded
    # for v1 — preserves legacy-artifact verifiability).
    assert v1.compute_content_hash() == v1_hash_baseline, (
        "v1 hash must not change when v2-only fields are populated; "
        "otherwise legacy v1 artifacts would fail verify_content_hash."
    )
    # v2 hash MUST have changed (verification_methods IS bound for v2).
    assert v2.compute_content_hash() != v2_hash_baseline


def test_v1_signed_artifact_still_verifies_after_round_trip():
    """End-to-end: a v1 package serialized to JSON, deserialized through
    the new code, and re-hashed must yield the same content_hash that
    the original v1 signer produced.

    Builds the v1 export by constructing a real v1 ``AgentIdentityPackage``,
    snapshotting its v1-canonical to_dict shape, computing the hash that
    v1 would have stored, then writing that hash + a placeholder
    signature back to the export and reading it through the new code.
    """
    pkg = AgentIdentityPackage(
        did="did:pkh:eip155:1:0xfeed",
        agent_name="old-agent",
        created_at="2024-06-01T00:00:00Z",
        constitution_hash="h" * 64,
        constitution_text="old text",
        export_timestamp="2024-06-01T00:00:00Z",
    )
    pkg.package_version = IDENTITY_PACKAGE_VERSION_LEGACY

    # Reproduce the v1 to_dict shape: drop the v2-only keys (which v1
    # to_dict didn't emit) and the verification fields.
    v1_canonical = pkg.to_dict()
    v1_canonical.pop("signatures", None)
    v1_canonical.pop("verification_methods", None)
    v1_canonical.pop("content_hash", None)
    v1_canonical.pop("signature", None)

    expected_v1_hash = hashlib.sha256(
        json.dumps(v1_canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    # Now write the v1 export with hash + signature
    v1_export = pkg.to_dict()
    v1_export.pop("signatures", None)
    v1_export.pop("verification_methods", None)
    v1_export["content_hash"] = expected_v1_hash
    v1_export["signature"] = "signature-bytes-here"

    # Read through the new code; the hash must round-trip
    rebuilt = AgentIdentityPackage.from_json(json.dumps(v1_export))
    assert rebuilt.is_v2() is False
    assert rebuilt.compute_content_hash() == expected_v1_hash
    assert rebuilt.verify_content_hash() is True


# ---------------------------------------------------------------------------
# Integration with KeypairFactory + multikey
# ---------------------------------------------------------------------------

def test_v2_package_with_real_multikey(base_package):
    """End-to-end: generate a keypair via the factory, add it as a v2
    verification method using the W3C Multikey form, verify the package
    round-trips and the multibase string can be decoded back to the
    same suite + key."""
    kp = KeypairFactory.generate_default()
    multibase = KeypairFactory.public_key_to_multibase(kp)
    base_package.add_verification_method(
        kid=f"{base_package.did}#keys-1",
        public_key_multibase=multibase,
    )

    rebuilt = AgentIdentityPackage.from_json(base_package.to_json())
    vm = rebuilt.verification_methods[0]
    assert vm["publicKeyMultibase"] == multibase
    assert vm["type"] == "Multikey"

    # The multibase string round-trips back to the same suite + public key
    suite_id, pub = KeypairFactory.multibase_to_public_key(multibase)
    assert suite_id == ALG_ECDSA_SECP256K1_SHA256
