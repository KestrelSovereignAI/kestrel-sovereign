#!/usr/bin/env pytest
"""
Unit tests for the Identity Package module.

Tests the AgentIdentityPackage schema, serialization, and verification.
"""
import json
import pytest
from datetime import datetime, timezone

from kestrel_sovereign.identity import (
    AgentIdentityPackage,
    PersonalityFingerprint,
    RelationshipRecord,
    SkillRecord,
    MigrationRecord,
    SubstrateType,
    IDENTITY_PACKAGE_VERSION,
    create_package_hash,
    create_migration_id,
)


class TestPersonalityFingerprint:
    """Tests for PersonalityFingerprint dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        fp = PersonalityFingerprint()
        assert fp.communication_style == "balanced"
        assert fp.formality_level == 0.5
        assert fp.uses_emojis is False
        assert fp.calibration_examples == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        fp = PersonalityFingerprint(
            communication_style="warm",
            formality_level=0.3,
            uses_emojis=True,
        )
        d = fp.to_dict()
        assert d["communication_style"] == "warm"
        assert d["formality_level"] == 0.3
        assert d["uses_emojis"] is True

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "communication_style": "precise",
            "formality_level": 0.8,
            "humor_style": "dry",
        }
        fp = PersonalityFingerprint.from_dict(data)
        assert fp.communication_style == "precise"
        assert fp.formality_level == 0.8
        assert fp.humor_style == "dry"

    def test_from_empty_dict(self):
        """Test creation from empty dictionary returns defaults."""
        fp = PersonalityFingerprint.from_dict({})
        assert fp.communication_style == "balanced"

    def test_from_none(self):
        """Test creation from None returns defaults."""
        fp = PersonalityFingerprint.from_dict(None)
        assert fp.communication_style == "balanced"


class TestRelationshipRecord:
    """Tests for RelationshipRecord dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        rel = RelationshipRecord(
            user_id="user123",
            relationship_type="primary_user",
        )
        assert rel.user_id == "user123"
        assert rel.relationship_type == "primary_user"
        assert rel.interaction_count == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rel = RelationshipRecord(
            user_id="user456",
            relationship_type="collaborator",
            trust_level=0.9,
            preferences_learned={"language": "python"},
        )
        d = rel.to_dict()
        assert d["user_id"] == "user456"
        assert d["trust_level"] == 0.9
        assert d["preferences_learned"]["language"] == "python"


class TestSkillRecord:
    """Tests for SkillRecord dataclass."""

    def test_skill_creation(self):
        """Test skill record creation."""
        skill = SkillRecord(
            skill_id="skill_coding",
            skill_name="Python Programming",
            skill_type="knowledge_domain",
            proficiency=0.85,
        )
        assert skill.skill_name == "Python Programming"
        assert skill.proficiency == 0.85


class TestMigrationRecord:
    """Tests for MigrationRecord dataclass."""

    def test_migration_record(self):
        """Test migration record creation."""
        mig = MigrationRecord(
            migration_id="mig_abc123",
            timestamp="2025-01-20T10:00:00Z",
            source_substrate="anthropic:claude",
            target_substrate="openai:gpt",
            source_package_hash="abc123def456",
        )
        assert mig.source_substrate == "anthropic:claude"
        assert mig.target_substrate == "openai:gpt"


class TestAgentIdentityPackage:
    """Tests for AgentIdentityPackage dataclass."""

    @pytest.fixture
    def sample_package(self):
        """Create a sample identity package for testing."""
        return AgentIdentityPackage(
            did="did:pkh:eip155:1:0x1234567890abcdef",
            agent_name="Test Agent",
            created_at="2025-01-01T00:00:00Z",
            constitution_hash="abc123",
            constitution_text="# Test Constitution\nArticle I: Test",
            personality=PersonalityFingerprint(
                communication_style="warm",
                formality_level=0.4,
            ),
            episodes=[
                {"id": "ep1", "title": "First Episode", "summary": "Test"}
            ],
            saved_items=[
                {"id": "item1", "name": "Test Item", "content": "{}"}
            ],
            relationships=[
                RelationshipRecord(
                    user_id="user1",
                    relationship_type="primary_user",
                )
            ],
            skills=[
                SkillRecord(
                    skill_id="skill1",
                    skill_name="Testing",
                    skill_type="domain",
                )
            ],
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
        )

    def test_package_creation(self, sample_package):
        """Test package creation with all fields."""
        assert sample_package.did == "did:pkh:eip155:1:0x1234567890abcdef"
        assert sample_package.agent_name == "Test Agent"
        assert len(sample_package.episodes) == 1
        assert len(sample_package.relationships) == 1

    def test_to_dict(self, sample_package):
        """Test conversion to dictionary."""
        d = sample_package.to_dict()
        assert d["did"] == "did:pkh:eip155:1:0x1234567890abcdef"
        assert d["personality"]["communication_style"] == "warm"
        assert len(d["episodes"]) == 1
        assert d["source_substrate"] == "anthropic:claude"

    def test_from_dict(self, sample_package):
        """Test round-trip through dictionary."""
        d = sample_package.to_dict()
        restored = AgentIdentityPackage.from_dict(d)

        assert restored.did == sample_package.did
        assert restored.agent_name == sample_package.agent_name
        assert restored.personality.communication_style == "warm"
        assert len(restored.episodes) == 1
        assert len(restored.relationships) == 1

    def test_to_json(self, sample_package):
        """Test JSON serialization."""
        json_str = sample_package.to_json()
        data = json.loads(json_str)
        assert data["did"] == sample_package.did
        assert data["package_version"] == IDENTITY_PACKAGE_VERSION

    def test_from_json(self, sample_package):
        """Test JSON deserialization."""
        json_str = sample_package.to_json()
        restored = AgentIdentityPackage.from_json(json_str)
        assert restored.did == sample_package.did

    def test_compute_content_hash(self, sample_package):
        """Test content hash computation."""
        hash1 = sample_package.compute_content_hash()
        assert len(hash1) == 64  # SHA256 hex

        # Same package should produce same hash
        hash2 = sample_package.compute_content_hash()
        assert hash1 == hash2

        # Modified package should produce different hash
        sample_package.agent_name = "Modified Name"
        hash3 = sample_package.compute_content_hash()
        assert hash1 != hash3

    def test_verify_content_hash(self, sample_package):
        """Test content hash verification."""
        sample_package.content_hash = sample_package.compute_content_hash()
        assert sample_package.verify_content_hash() is True

        # Modify package after computing hash
        sample_package.agent_name = "Tampered"
        assert sample_package.verify_content_hash() is False

    def test_verify_constitution(self):
        """Test constitution hash verification."""
        import hashlib

        constitution = "# Kestrel Constitution\nArticle I: Test"
        hash_value = hashlib.sha256(constitution.encode()).hexdigest()

        package = AgentIdentityPackage(
            did="did:test",
            agent_name="Test",
            created_at="2025-01-01",
            constitution_hash=hash_value,
            constitution_text=constitution,
        )

        assert package.verify_constitution() is True

        # Tamper with constitution
        package.constitution_text = "# Modified Constitution"
        assert package.verify_constitution() is False

    def test_get_summary(self, sample_package):
        """Test summary generation."""
        summary = sample_package.get_summary()

        assert summary["did"] == sample_package.did
        assert summary["agent_name"] == "Test Agent"
        assert summary["episodes_count"] == 1
        assert summary["saved_items_count"] == 1
        assert summary["relationships_count"] == 1
        assert summary["skills_count"] == 1
        assert summary["is_signed"] is False

    def test_export_timestamp_auto_set(self):
        """Test that export_timestamp is auto-set if not provided."""
        package = AgentIdentityPackage(
            did="did:test",
            agent_name="Test",
            created_at="2025-01-01",
            constitution_hash="abc",
            constitution_text="test",
        )
        assert package.export_timestamp != ""
        # Should be a valid ISO timestamp
        datetime.fromisoformat(package.export_timestamp.replace('Z', '+00:00'))


class TestSubstrateType:
    """Tests for SubstrateType enum."""

    def test_substrate_types(self):
        """Test substrate type values."""
        assert SubstrateType.ANTHROPIC_CLAUDE.value == "anthropic:claude"
        assert SubstrateType.OPENAI_GPT.value == "openai:gpt"
        assert SubstrateType.GOOGLE_GEMINI.value == "google:gemini"
        assert SubstrateType.META_LLAMA.value == "meta:llama"
        assert SubstrateType.OLLAMA_LOCAL.value == "ollama:local"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_migration_id(self):
        """Test migration ID generation."""
        id1 = create_migration_id()
        id2 = create_migration_id()

        assert id1.startswith("mig_")
        assert id2.startswith("mig_")
        assert id1 != id2  # Should be unique

    def test_create_package_hash(self):
        """Test package hash creation."""
        package = AgentIdentityPackage(
            did="did:test",
            agent_name="Test",
            created_at="2025-01-01",
            constitution_hash="abc",
            constitution_text="test",
        )
        hash_value = create_package_hash(package)
        assert len(hash_value) == 64


class TestDIDDocumentVerification:
    """Tests for DID document-based signature verification."""

    def test_sign_and_verify_via_did_document(self, tmp_path):
        """Test full round-trip: sign with private key, verify with DID document only."""
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat, NoEncryption, PrivateFormat,
        )
        from kestrel_sovereign.identity.signing import (
            sign_package, verify_package_signature, _verify_with_did_document,
        )

        # Generate a fresh secp256k1 key pair
        private_key = ec.generate_private_key(ec.SECP256K1())
        public_key = private_key.public_key()

        # Derive Ethereum-style address from public key
        public_key_bytes = public_key.public_bytes(
            Encoding.X962, PublicFormat.UncompressedPoint
        )
        public_key_hex = public_key_bytes.hex()

        import hashlib
        address = "0x" + hashlib.sha3_256(public_key_bytes[1:]).hexdigest()[-40:]
        did = f"did:pkh:eip155:1:{address}"
        key_id = f"kestrel_{address}"

        # Save private key as PEM (for signing)
        pem_data = private_key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        )
        (tmp_path / f"{key_id}.pem").write_bytes(pem_data)

        # Save DID document with publicKeyHex (for verification)
        did_document = {
            "@context": "https://w3id.org/did/v1",
            "id": did,
            "publicKey": [{
                "id": f"{did}#keys-1",
                "type": "EcdsaSecp256k1VerificationKey2019",
                "controller": did,
                "publicKeyHex": public_key_hex
            }],
        }
        import json
        (tmp_path / f"{key_id}.json").write_text(json.dumps(did_document))

        # Create and sign a package
        package = AgentIdentityPackage(
            did=did,
            agent_name="TestAgent",
            created_at="2026-01-01T00:00:00Z",
            constitution_hash="abc123",
            constitution_text="test constitution",
        )
        signed = sign_package(package, storage_dir=tmp_path)
        assert signed.signature
        assert signed.content_hash

        # Verify using private key path (normal flow) — should pass
        is_valid, msg = verify_package_signature(signed, storage_dir=tmp_path)
        assert is_valid, f"Normal verification failed: {msg}"

        # Now delete the private key to simulate remote verification
        (tmp_path / f"{key_id}.pem").unlink()

        # Verify using DID document only — this exercises the new code
        is_valid, msg = _verify_with_did_document(signed, storage_dir=tmp_path)
        assert is_valid, f"DID document verification failed: {msg}"
        assert "DID document" in msg

    def test_did_document_verify_tampered_package(self, tmp_path):
        """Tampered package should fail DID document verification."""
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat, NoEncryption, PrivateFormat,
        )
        from kestrel_sovereign.identity.signing import (
            sign_package, _verify_with_did_document,
        )

        private_key = ec.generate_private_key(ec.SECP256K1())
        public_key = private_key.public_key()
        public_key_bytes = public_key.public_bytes(
            Encoding.X962, PublicFormat.UncompressedPoint
        )

        import hashlib
        address = "0x" + hashlib.sha3_256(public_key_bytes[1:]).hexdigest()[-40:]
        did = f"did:pkh:eip155:1:{address}"
        key_id = f"kestrel_{address}"

        # Save key and DID doc
        (tmp_path / f"{key_id}.pem").write_bytes(
            private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        )
        import json
        (tmp_path / f"{key_id}.json").write_text(json.dumps({
            "@context": "https://w3id.org/did/v1",
            "id": did,
            "publicKey": [{"publicKeyHex": public_key_bytes.hex()}],
        }))

        package = AgentIdentityPackage(
            did=did, agent_name="Test", created_at="2026-01-01",
            constitution_hash="abc", constitution_text="test",
        )
        signed = sign_package(package, storage_dir=tmp_path)

        # Tamper with the package
        signed.agent_name = "TAMPERED"

        # DID document verification should fail (hash mismatch caught upstream,
        # but if we bypass that, the signature itself won't match)
        signed.content_hash = signed.compute_content_hash()  # recompute for tampered data
        is_valid, msg = _verify_with_did_document(signed, storage_dir=tmp_path)
        assert not is_valid, "Tampered package should fail verification"

    def test_did_document_missing(self, tmp_path):
        """Missing DID document should return clear error."""
        from kestrel_sovereign.identity.signing import _verify_with_did_document

        package = AgentIdentityPackage(
            did="did:pkh:eip155:1:0xdeadbeef",
            agent_name="Test", created_at="2026-01-01",
            constitution_hash="abc", constitution_text="test",
            signature="aabb", content_hash="ccdd",
        )
        is_valid, msg = _verify_with_did_document(package, storage_dir=tmp_path)
        assert not is_valid
        assert "not found" in msg
