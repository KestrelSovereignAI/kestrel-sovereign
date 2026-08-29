"""Tests for SpawnMandate data structure and DID delegation chains."""

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from kestrel_sovereign.spawn.mandate import (
    SpawnMandate,
    sign_mandate,
    verify_mandate,
    create_child_did_document,
)
from kestrel_sovereign.inception_service import generate_secp256k1_keypair
from kestrel_sovereign.identity.succession import SuccessionStatement
from kestrel_sovereign.identity.succession_chain import build_chain


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

    def test_decimal_budget_normalizes_mandate_to_signed_json_number(self):
        """Runtime budget and durable receipt use one normalized value."""
        mandate = SpawnMandate(
            parent_did="did:test:parent",
            budget_allocation=Decimal("0.100000000000000005"),
        )

        payload = json.loads(mandate._signable_payload())
        edge = mandate.to_edge_properties()

        assert mandate.budget_allocation == 0.1
        assert payload["budget_allocation"] == mandate.budget_allocation
        assert edge["budget_allocation"] == mandate.budget_allocation

    def test_decimal_budget_rejects_float_overflow_before_signing(self):
        mandate = SpawnMandate(
            parent_did="did:test:parent",
            budget_allocation=Decimal("1e10000"),
        )

        with pytest.raises(ValueError, match="JSON numeric range"):
            mandate._signable_payload()

    def test_decimal_budget_rejects_float_underflow_before_signing(self):
        mandate = SpawnMandate(
            parent_did="did:test:parent",
            budget_allocation=Decimal("1e-400"),
        )

        with pytest.raises(ValueError, match="JSON numeric range"):
            mandate._signable_payload()


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

    def test_rotated_parent_accepts_classical_mandate_from_before_cutoff(
        self, parent_keys
    ):
        parent_private, parent_public = parent_keys
        legacy_did = "did:pkh:eip155:1:0xPreRotationParent"
        statement = SuccessionStatement(
            predecessor_did=legacy_did,
            successor_did="did:web:example.test:rotated-parent",
            effective_from="2026-08-20T00:00:00+00:00",
            reason="test rotation",
        )
        identity = SimpleNamespace(
            is_hybrid=True,
            legacy_did=legacy_did,
            succession_chain=build_chain([statement]),
        )
        mandate = SpawnMandate(
            parent_did=legacy_did,
            child_did="did:test:child",
            created_at="2026-08-19T23:59:59+00:00",
        )
        sign_mandate(mandate, parent_private)

        assert verify_mandate(
            mandate, parent_public, parent_identity=identity
        ) is True

    def test_rotated_parent_rejects_classical_mandate_at_or_after_cutoff(
        self, parent_keys
    ):
        parent_private, parent_public = parent_keys
        legacy_did = "did:pkh:eip155:1:0xPostRotationParent"
        statement = SuccessionStatement(
            predecessor_did=legacy_did,
            successor_did="did:web:example.test:rotated-parent",
            effective_from="2026-08-20T00:00:00+00:00",
            reason="test rotation",
        )
        identity = SimpleNamespace(
            is_hybrid=True,
            legacy_did=legacy_did,
            succession_chain=build_chain([statement]),
        )
        mandate = SpawnMandate(
            parent_did=legacy_did,
            child_did="did:test:child",
            created_at="2026-08-20T00:00:00+00:00",
        )
        sign_mandate(mandate, parent_private)

        assert verify_mandate(
            mandate, parent_public, parent_identity=identity
        ) is False


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
