"""Tests for the out-of-DB Sovereign trust-root resolver (#2499)."""

from __future__ import annotations

import json

import pytest

from kestrel_sovereign.constitution.amendment_artifact import (
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.constitution.trust_root import (
    MAX_TRUST_ROOT_BYTES,
    SOVEREIGN_TRUST_ROOT_ENV,
    SovereignTrustRootError,
    load_sovereign_trust_root,
)
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite


def _write_root(path, did):
    keypair = Secp256k1Suite().generate_keypair()
    document = did_document_from_legacy_public_key(did, keypair.public_key)
    path.write_text(json.dumps(document), encoding="utf-8")
    return document


def test_missing_root_fails_with_migration_guidance():
    with pytest.raises(SovereignTrustRootError) as exc_info:
        load_sovereign_trust_root(environ={})
    message = str(exc_info.value)
    assert SOVEREIGN_TRUST_ROOT_ENV in message
    assert "operator-owned JSON file" in message
    assert "audit data only" in message


def test_loads_operator_pinned_document(tmp_path):
    path = tmp_path / "root.did.json"
    expected = _write_root(path, "did:example:sovereign")
    assert load_sovereign_trust_root(
        environ={SOVEREIGN_TRUST_ROOT_ENV: str(path)}
    ) == expected


def test_same_explicit_and_environment_path_is_unambiguous(tmp_path):
    path = tmp_path / "root.did.json"
    expected = _write_root(path, "did:example:sovereign")
    assert load_sovereign_trust_root(
        explicit_path=path,
        environ={SOVEREIGN_TRUST_ROOT_ENV: str(path)},
    ) == expected


def test_conflicting_explicit_and_environment_paths_fail_closed(tmp_path):
    first = tmp_path / "first.did.json"
    second = tmp_path / "second.did.json"
    _write_root(first, "did:example:first")
    _write_root(second, "did:example:second")
    with pytest.raises(SovereignTrustRootError, match="Ambiguous"):
        load_sovereign_trust_root(
            explicit_path=first,
            environ={SOVEREIGN_TRUST_ROOT_ENV: str(second)},
        )


@pytest.mark.parametrize(
    "content,match",
    [
        ("not json", "Cannot read"),
        ("[]", "one JSON DID-document object"),
        ('{"id": "not-a-did", "publicKey": [{}]}', "no valid DID id"),
        ('{"id": "did:example:root"}', "no publicKey or verificationMethod"),
    ],
)
def test_malformed_root_fails_closed(tmp_path, content, match):
    path = tmp_path / "root.did.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SovereignTrustRootError, match=match):
        load_sovereign_trust_root(explicit_path=path, environ={})


def test_agent_owned_root_is_rejected(tmp_path):
    path = tmp_path / "root.did.json"
    agent_did = "did:example:agent"
    _write_root(path, agent_did)
    with pytest.raises(SovereignTrustRootError, match="agent-owned DID"):
        load_sovereign_trust_root(
            explicit_path=path,
            environ={},
            agent_dids={agent_did},
        )


def test_oversized_root_is_rejected_with_bounded_read(tmp_path):
    path = tmp_path / "oversized.did.json"
    path.write_bytes(b"x" * (MAX_TRUST_ROOT_BYTES + 1))
    with pytest.raises(SovereignTrustRootError, match="exceeds"):
        load_sovereign_trust_root(explicit_path=path, environ={})
