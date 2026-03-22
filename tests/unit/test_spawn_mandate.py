"""Tests for SpawnMandate data structure and DID delegation chains."""

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from kestrel_sovereign.spawn.mandate import (
    SpawnMandate,
    sign_mandate,
    verify_mandate,
    create_child_did_document,
)
from kestrel_sovereign.inception_service import generate_secp256k1_keypair


@pytest.fixture
def parent_keys():
    return generate_secp256k1_keypair()


@pytest.fixture
def child_keys():
    return generate_secp256k1_keypair()


@pytest.fixture
def sample_mandate():
    return SpawnMandate(
        parent_did="did:pkh:eip155:1:0xParent123",
        child_did="did:pkh:eip155:1:0xChild456",
        constitution_hash="abc123def456",
        additional_constraints={"max_tokens": 1000},
        budget_allocation=10.0,
        ttl_seconds=7200,
        features_allowed=["chat", "memory"],
        purpose="research assistant",
        max_child_depth=2,
    )


class TestSpawnMandate:
    def test_mandate_creation(self, sample_mandate):
        assert sample_mandate.parent_did == "did:pkh:eip155:1:0xParent123"
        assert sample_mandate.child_did == "did:pkh:eip155:1:0xChild456"
        assert sample_mandate.purpose == "research assistant"
        assert sample_mandate.ttl_seconds == 7200
        assert sample_mandate.max_child_depth == 2
        assert sample_mandate.parent_signature is None

    def test_mandate_defaults(self):
        mandate = SpawnMandate(parent_did="did:pkh:eip155:1:0xTest")
        assert mandate.child_did is None
        assert mandate.constitution_hash == ""
        assert mandate.budget_allocation == 0.0
        assert mandate.ttl_seconds == 3600
        assert mandate.features_allowed == []
        assert mandate.purpose == ""
        assert mandate.max_child_depth == 0
        assert mandate.parent_signature is None
        assert mandate.created_at  # should be set

    def test_mandate_to_dict(self, sample_mandate):
        d = sample_mandate.to_dict()
        assert d["parent_did"] == "did:pkh:eip155:1:0xParent123"
        assert d["budget_allocation"] == 10.0
        assert isinstance(d, dict)


class TestMandateSigning:
    def test_sign_and_verify_roundtrip(self, sample_mandate, parent_keys):
        private_key, public_key = parent_keys

        signed = sign_mandate(sample_mandate, private_key)
        assert signed.parent_signature is not None
        assert len(signed.parent_signature) > 0

        assert verify_mandate(signed, public_key) is True

    def test_verify_with_wrong_key_fails(self, sample_mandate, parent_keys, child_keys):
        parent_private, _ = parent_keys
        _, wrong_public = child_keys

        signed = sign_mandate(sample_mandate, parent_private)
        assert verify_mandate(signed, wrong_public) is False

    def test_verify_unsigned_mandate_fails(self, sample_mandate, parent_keys):
        _, public_key = parent_keys
        assert verify_mandate(sample_mandate, public_key) is False

    def test_tampered_mandate_fails(self, sample_mandate, parent_keys):
        private_key, public_key = parent_keys

        signed = sign_mandate(sample_mandate, private_key)
        assert verify_mandate(signed, public_key) is True

        # Tamper with the mandate after signing
        signed.purpose = "malicious purpose"
        assert verify_mandate(signed, public_key) is False

    def test_invalid_signature_hex_fails(self, sample_mandate, parent_keys):
        _, public_key = parent_keys
        sample_mandate.parent_signature = "deadbeef"
        assert verify_mandate(sample_mandate, public_key) is False


class TestChildDIDDocument:
    def test_child_did_has_controller(self, child_keys):
        _, child_public = child_keys
        parent_did = "did:pkh:eip155:1:0xParentAddr"

        doc = create_child_did_document(parent_did, child_public)

        assert doc["controller"] == parent_did
        assert doc["id"].startswith("did:pkh:eip155:1:")
        assert "@context" in doc
        assert "publicKey" in doc

    def test_child_did_is_valid_format(self, child_keys):
        _, child_public = child_keys
        parent_did = "did:pkh:eip155:1:0xParentAddr"

        doc = create_child_did_document(parent_did, child_public)

        assert doc["id"] != parent_did
        assert len(doc["publicKey"]) > 0
        assert doc["publicKey"][0]["type"] == "EcdsaSecp256k1VerificationKey2019"
