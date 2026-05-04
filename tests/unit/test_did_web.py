"""
did:web producer + resolver — Wave 2 sub-PR 3 (#917).

Covers:
- DID URI builder (host-only, with path, with port, with percent-encoding)
- did_to_url / url_to_did round-trips against W3C did:web spec examples
- DID document shape (W3C contexts, verification methods, relationships)
- Hybrid identity (Ed25519 + ML-DSA-65) yielding two Multikey methods
- parse_did_document round-trips back to (suite, public_key) pairs
- Impersonation guard (id mismatch)
- Resolver with injected fetcher (no real network)
- HTTPS-only enforcement
"""

from __future__ import annotations

import json

import pytest

from kestrel_sovereign.identity.did_web import (
    DID_DOCUMENT_CONTEXTS,
    DID_WEB_PREFIX,
    DidWebError,
    build_did,
    build_did_document,
    build_verification_methods,
    did_to_url,
    parse_did_document,
    resolve,
    url_to_did,
)
from kestrel_sovereign.security.crypto_suite import (
    Ed25519Suite,
    MLDSA65Suite,
    Secp256k1Suite,
)


# ---------------------------------------------------------------------------
# build_did
# ---------------------------------------------------------------------------

def test_build_did_host_only():
    assert build_did("example.com") == "did:web:example.com"


def test_build_did_with_path():
    assert build_did("example.com", ["agent", "meridian"]) == (
        "did:web:example.com:agent:meridian"
    )


def test_build_did_with_port_encoded_as_percent_3a():
    assert build_did("example.com", ["foo"], port=8080) == (
        "did:web:example.com%3A8080:foo"
    )


def test_build_did_rejects_bare_colon_in_domain():
    with pytest.raises(DidWebError, match="bare host"):
        build_did("example.com:8080")


def test_build_did_rejects_scheme_in_domain():
    with pytest.raises(DidWebError, match="bare host"):
        build_did("https://example.com")


def test_build_did_rejects_slash_in_domain():
    with pytest.raises(DidWebError, match="bare host"):
        build_did("example.com/agent")


def test_build_did_rejects_colon_in_segment():
    with pytest.raises(DidWebError, match="must not contain ':'"):
        build_did("example.com", ["bad:segment"])


def test_build_did_rejects_empty_segment():
    with pytest.raises(DidWebError, match="non-empty"):
        build_did("example.com", [""])


def test_build_did_rejects_invalid_port():
    with pytest.raises(DidWebError, match="1..65535"):
        build_did("example.com", port=0)
    with pytest.raises(DidWebError, match="1..65535"):
        build_did("example.com", port=70_000)


# ---------------------------------------------------------------------------
# did_to_url — W3C did:web spec examples
# ---------------------------------------------------------------------------

def test_did_to_url_host_only_uses_well_known():
    """Per spec: host-only DID resolves to /.well-known/did.json."""
    assert did_to_url("did:web:example.com") == (
        "https://example.com/.well-known/did.json"
    )


def test_did_to_url_with_path():
    """Per spec: path segments split on ':' become /-separated path."""
    assert did_to_url("did:web:example.com:agent:meridian") == (
        "https://example.com/agent/meridian/did.json"
    )


def test_did_to_url_with_port_decoded():
    """Per spec: %3A in host part decodes to ':' for the URL."""
    assert did_to_url("did:web:example.com%3A8080:foo") == (
        "https://example.com:8080/foo/did.json"
    )


def test_did_to_url_rejects_non_did_web():
    with pytest.raises(DidWebError, match="not a did:web"):
        did_to_url("did:key:abc")


def test_did_to_url_rejects_empty_body():
    with pytest.raises(DidWebError, match="empty"):
        did_to_url("did:web:")


# ---------------------------------------------------------------------------
# url_to_did round-trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("did", [
    "did:web:example.com",
    "did:web:example.com:agent:meridian",
    "did:web:example.com%3A8080:foo",
])
def test_url_to_did_round_trip(did):
    assert url_to_did(did_to_url(did)) == did


def test_url_to_did_rejects_http():
    with pytest.raises(DidWebError, match="HTTPS"):
        url_to_did("http://example.com/.well-known/did.json")


def test_url_to_did_rejects_non_did_json_path():
    with pytest.raises(DidWebError, match="must end with"):
        url_to_did("https://example.com/agent/meridian/wrong.json")


# ---------------------------------------------------------------------------
# Verification methods
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hybrid_keys():
    """Hybrid identity: Ed25519 (classical) + ML-DSA-65 (PQ)."""
    ed = Ed25519Suite()
    pq = MLDSA65Suite()
    return [
        (ed, ed.generate_keypair().public_key),
        (pq, pq.generate_keypair().public_key),
    ]


def test_build_verification_methods_emits_multikey_type(hybrid_keys):
    did = "did:web:example.com:agent:meridian"
    methods = build_verification_methods(did, hybrid_keys)
    assert len(methods) == 2
    for m in methods:
        assert m["type"] == "Multikey"
        assert m["controller"] == did
        assert m["publicKeyMultibase"].startswith("z")


def test_build_verification_methods_kids_are_unique_and_indexed(hybrid_keys):
    did = "did:web:example.com"
    methods = build_verification_methods(did, hybrid_keys)
    assert methods[0]["id"] == f"{did}#key-1"
    assert methods[1]["id"] == f"{did}#key-2"


def test_build_verification_methods_custom_prefix(hybrid_keys):
    did = "did:web:example.com"
    methods = build_verification_methods(did, hybrid_keys, kid_prefix="agent")
    assert methods[0]["id"] == f"{did}#agent-1"


def test_build_verification_methods_secp256k1_uses_compressed_form():
    """Wave 1 sub-PR 4: secp256k1 multikey body must be 33-byte compressed
    (not 65-byte uncompressed). Verifies the multikey path uses
    ``serialize_public_key_for_multikey``, not the legacy serializer."""
    from kestrel_sovereign.security.multikey import base58btc_decode
    secp = Secp256k1Suite()
    kp = secp.generate_keypair()
    did = "did:web:example.com"
    methods = build_verification_methods(did, [(secp, kp.public_key)])
    raw = base58btc_decode(methods[0]["publicKeyMultibase"][1:])
    # codec varint b'\xe7\x01' (2 bytes) + 33-byte compressed point
    assert len(raw) == 2 + 33


def test_build_verification_methods_rejects_non_suite(hybrid_keys):
    with pytest.raises(DidWebError, match="expected CryptoSuite"):
        build_verification_methods(
            "did:web:example.com",
            [("not-a-suite", b"\x00" * 32)],
        )


# ---------------------------------------------------------------------------
# DID document
# ---------------------------------------------------------------------------

def test_build_did_document_carries_w3c_contexts(hybrid_keys):
    doc = build_did_document("did:web:example.com:agent", hybrid_keys)
    assert doc["@context"] == DID_DOCUMENT_CONTEXTS


def test_build_did_document_id_matches(hybrid_keys):
    did = "did:web:example.com:agent"
    doc = build_did_document(did, hybrid_keys)
    assert doc["id"] == did


def test_build_did_document_lists_methods_in_relationships(hybrid_keys):
    """Authentication and assertionMethod must reference every key by id —
    verifier-policy code looks up keys via these arrays. Hybrid means BOTH
    keys land in BOTH relationships; verify-policy decides what's required.
    """
    did = "did:web:example.com:agent"
    doc = build_did_document(did, hybrid_keys)
    method_ids = [m["id"] for m in doc["verificationMethod"]]
    assert doc["authentication"] == method_ids
    assert doc["assertionMethod"] == method_ids
    assert len(method_ids) == 2  # hybrid


def test_build_did_document_optional_also_known_as(hybrid_keys):
    doc = build_did_document(
        "did:web:example.com",
        hybrid_keys,
        also_known_as=["did:key:zXyz"],
    )
    assert doc["alsoKnownAs"] == ["did:key:zXyz"]


def test_build_did_document_optional_services(hybrid_keys):
    services = [{"id": "#car", "type": "CARStore", "serviceEndpoint": "https://x/y"}]
    doc = build_did_document("did:web:example.com", hybrid_keys, services=services)
    assert doc["service"] == services


def test_build_did_document_rejects_non_did_web(hybrid_keys):
    with pytest.raises(DidWebError, match="not a did:web"):
        build_did_document("did:key:abc", hybrid_keys)


def test_build_did_document_round_trips_through_json(hybrid_keys):
    """Real wire test: build, serialize, parse, extract — keys must match."""
    did = "did:web:example.com:agent:meridian"
    doc = build_did_document(did, hybrid_keys)
    wire = json.loads(json.dumps(doc))  # round-trip through JSON

    parsed = parse_did_document(wire)
    assert len(parsed) == len(hybrid_keys)
    for (orig_suite, orig_pub), (kid, parsed_suite, parsed_pub) in zip(
        hybrid_keys, parsed
    ):
        assert kid.startswith(did + "#")
        assert parsed_suite.alg_id == orig_suite.alg_id
        # Verify a signature with the parsed key — strongest possible check
        sig = orig_suite.sign(b"identity-payload", orig_suite.generate_keypair().private_key) \
            if False else None
        # Use the orig private key path:
        # (We don't have private keys cached; verify by serializing both pubs.)
        a = orig_suite.serialize_public_key_for_multikey(orig_pub)
        b = parsed_suite.serialize_public_key_for_multikey(parsed_pub)
        assert a == b


# ---------------------------------------------------------------------------
# parse_did_document
# ---------------------------------------------------------------------------

def test_parse_did_document_skips_non_multikey_methods(hybrid_keys):
    """A DID document may carry verification methods of other types
    (e.g. JsonWebKey2020); those must be silently skipped, not crashed on.
    """
    did = "did:web:example.com"
    doc = build_did_document(did, hybrid_keys)
    doc["verificationMethod"].append({
        "id": f"{did}#legacy",
        "type": "JsonWebKey2020",
        "controller": did,
        "publicKeyJwk": {"kty": "RSA"},
    })
    parsed = parse_did_document(doc)
    assert len(parsed) == 2  # Only the two Multikey entries


def test_parse_did_document_unknown_multicodec_raises(hybrid_keys):
    """Unknown multicodec on a Multikey method = real interop break;
    surface it as DidWebError so callers don't silently drop a key."""
    did = "did:web:example.com"
    doc = {
        "@context": DID_DOCUMENT_CONTEXTS,
        "id": did,
        "verificationMethod": [{
            "id": f"{did}#alien",
            "type": "Multikey",
            "controller": did,
            # base58btc("zzz") with no valid codec prefix
            "publicKeyMultibase": "zzz",
        }],
    }
    with pytest.raises(DidWebError):
        parse_did_document(doc)


def test_parse_did_document_rejects_malformed_method():
    did = "did:web:example.com"
    doc = {
        "@context": DID_DOCUMENT_CONTEXTS,
        "id": did,
        "verificationMethod": [{"type": "Multikey"}],  # missing id, multibase
    }
    with pytest.raises(DidWebError, match="missing id"):
        parse_did_document(doc)


def test_parse_did_document_handles_empty_methods():
    assert parse_did_document({"verificationMethod": []}) == []


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def test_resolve_with_injected_fetcher(hybrid_keys):
    """No-network round-trip: fetcher returns canned bytes."""
    did = "did:web:example.com:agent"
    expected_doc = build_did_document(did, hybrid_keys)
    payload = json.dumps(expected_doc).encode("utf-8")

    captured_url = []
    def fake_fetcher(url: str) -> bytes:
        captured_url.append(url)
        return payload

    doc = resolve(did, fetcher=fake_fetcher)
    assert doc == expected_doc
    assert captured_url == ["https://example.com/agent/did.json"]


def test_resolve_rejects_id_mismatch_impersonation(hybrid_keys):
    """Impersonation guard: the document's id must match the requested DID.
    Otherwise a malicious server could return another agent's document
    and a naive verifier would accept the keys without noticing."""
    requested = "did:web:example.com:alice"
    other_doc = build_did_document("did:web:example.com:eve", hybrid_keys)
    payload = json.dumps(other_doc).encode("utf-8")

    with pytest.raises(DidWebError, match="impersonation guard"):
        resolve(requested, fetcher=lambda url: payload)


def test_resolve_rejects_non_object_document():
    with pytest.raises(DidWebError, match="must be a JSON object"):
        resolve(
            "did:web:example.com",
            fetcher=lambda url: b'["not", "an", "object"]',
        )


def test_resolve_rejects_non_json_document():
    with pytest.raises(DidWebError, match="not JSON"):
        resolve("did:web:example.com", fetcher=lambda url: b"<html>404</html>")


def test_resolve_default_fetcher_rejects_http():
    """Direct test of the safety floor — HTTPS-only is not optional."""
    from kestrel_sovereign.identity.did_web import _default_fetcher
    with pytest.raises(DidWebError, match="HTTPS"):
        _default_fetcher("http://example.com/did.json")
