"""
Unit tests for Wyoming DAO LLC document generation.

Tests the kestrel-feature-legal package: models, articles generator,
operating agreement, and the incorporate agent tool.
"""

import json
import pytest

from kestrel_feature_legal.models import (
    DAOArticles,
    EntityStatus,
    EntityType,
    IncorporationPackage,
    LegalEntity,
    ManagementType,
    OrganizerInfo,
    RegisteredAgentInfo,
    WYOMING_DAO_RESTRICTIONS_NOTICE,
)
from kestrel_feature_legal.wyoming_dao import (
    FILING_FEE_USD,
    generate_articles,
    generate_incorporation_package,
    render_articles_json,
    render_articles_text,
    validate_entity_name,
)
from kestrel_feature_legal.operating_agreement import generate_operating_agreement
from kestrel_feature_legal.incorporate_tool import IncorporateTool


# --- Fixtures ---

SAMPLE_DID = "did:pkh:eip155:1:0xABCDEF1234567890abcdef1234567890ABCDEF12"
SAMPLE_CONSTITUTION_HASH = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
SAMPLE_CONSTITUTION_TEXT = """\
KESTREL CONSTITUTION

Article 1: This agent shall act transparently in all operations.
Article 2: This agent shall preserve user privacy.
Article 3: This agent shall maintain constitutional integrity.
"""


@pytest.fixture
def registered_agent():
    return RegisteredAgentInfo(
        name="Wyoming Agents, Inc.",
        physical_address="1712 Pioneer Ave, Suite 500, Cheyenne, WY 82001",
    )


@pytest.fixture
def organizer():
    return OrganizerInfo(
        name="Jane Doe",
        address="456 Oak St, Denver, CO 80202",
    )


@pytest.fixture
def sample_articles(registered_agent, organizer):
    return generate_articles(
        entity_name="Kestrel Alpha DAO LLC",
        agent_did=SAMPLE_DID,
        constitution_hash=SAMPLE_CONSTITUTION_HASH,
        registered_agent=registered_agent,
        organizer=organizer,
    )


@pytest.fixture
def did_document():
    return {
        "@context": "https://www.w3.org/ns/did/v1",
        "id": SAMPLE_DID,
        "verificationMethod": [
            {
                "id": f"{SAMPLE_DID}#key-1",
                "type": "EcdsaSecp256k1VerificationKey2019",
                "controller": SAMPLE_DID,
            }
        ],
    }


# --- Entity Name Validation ---


class TestEntityNameValidation:
    def test_valid_name(self):
        assert validate_entity_name("Kestrel Alpha DAO LLC") == []

    def test_valid_name_lao(self):
        assert validate_entity_name("Kestrel LAO LLC") == []

    def test_valid_name_llc_variation(self):
        assert validate_entity_name("Kestrel DAO L.L.C.") == []

    def test_missing_dao(self):
        errors = validate_entity_name("Kestrel Alpha LLC")
        assert len(errors) == 1
        assert "DAO" in errors[0]

    def test_missing_llc(self):
        errors = validate_entity_name("Kestrel Alpha DAO")
        assert len(errors) == 1
        assert "LLC" in errors[0]

    def test_missing_both(self):
        errors = validate_entity_name("Kestrel Alpha")
        assert len(errors) == 2

    def test_case_insensitive(self):
        assert validate_entity_name("kestrel dao llc") == []

    def test_too_short(self):
        errors = validate_entity_name("AB")
        assert any("short" in e.lower() for e in errors)


# --- Articles Generation ---


class TestArticlesGeneration:
    def test_generates_valid_articles(self, registered_agent, organizer):
        articles = generate_articles(
            entity_name="Kestrel Alpha DAO LLC",
            agent_did=SAMPLE_DID,
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            registered_agent=registered_agent,
            organizer=organizer,
        )
        assert articles.entity_name == "Kestrel Alpha DAO LLC"
        assert articles.smart_contract_id == SAMPLE_DID
        assert articles.constitution_hash == SAMPLE_CONSTITUTION_HASH
        assert articles.management_type == ManagementType.ALGORITHMICALLY_MANAGED

    def test_rejects_invalid_name(self, registered_agent, organizer):
        with pytest.raises(ValueError, match="validation failed"):
            generate_articles(
                entity_name="Bad Name",
                agent_did=SAMPLE_DID,
                constitution_hash=SAMPLE_CONSTITUTION_HASH,
                registered_agent=registered_agent,
                organizer=organizer,
            )

    def test_member_managed(self, registered_agent, organizer):
        articles = generate_articles(
            entity_name="Kestrel DAO LLC",
            agent_did=SAMPLE_DID,
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            registered_agent=registered_agent,
            organizer=organizer,
            management_type=ManagementType.MEMBER_MANAGED,
        )
        assert articles.management_type == ManagementType.MEMBER_MANAGED


# --- Articles Rendering ---


class TestArticlesRendering:
    def test_text_rendering(self, sample_articles):
        text = render_articles_text(sample_articles)
        assert "Kestrel Alpha DAO LLC" in text
        assert SAMPLE_DID in text
        assert SAMPLE_CONSTITUTION_HASH in text
        assert "algorithmically managed" in text

    def test_json_rendering(self, sample_articles):
        json_str = render_articles_json(sample_articles)
        data = json.loads(json_str)
        assert data["entity_name"] == "Kestrel Alpha DAO LLC"
        assert data["smart_contract_id"] == SAMPLE_DID


# --- Operating Agreement ---


class TestOperatingAgreement:
    def test_generates_agreement(self, sample_articles):
        agreement = generate_operating_agreement(
            articles=sample_articles,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
        )
        assert "Kestrel Alpha DAO LLC" in agreement
        assert SAMPLE_DID in agreement
        assert "OPERATING AGREEMENT" in agreement

    def test_algorithmically_managed_content(self, sample_articles):
        agreement = generate_operating_agreement(
            articles=sample_articles,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
        )
        assert "ALGORITHMICALLY MANAGED" in agreement


# --- Legal Entity Model ---


class TestLegalEntity:
    def test_default_values(self):
        entity = LegalEntity(jurisdiction="US-WY")
        assert entity.status == EntityStatus.DRAFT
        assert entity.entity_type == EntityType.DAO_LLC

    def test_serialization_roundtrip(self):
        entity = LegalEntity(
            jurisdiction="US-WY",
            entity_type=EntityType.DAO_LLC,
            entity_name="Kestrel Alpha DAO LLC",
            filing_number="2026-001234",
            status=EntityStatus.ACTIVE,
            registrar_did=SAMPLE_DID,
        )
        data = entity.to_dict()
        restored = LegalEntity.from_dict(data)
        assert restored.jurisdiction == "US-WY"
        assert restored.filing_number == "2026-001234"
        assert restored.status == EntityStatus.ACTIVE


# --- Incorporation Package ---


class TestIncorporationPackage:
    def test_generates_complete_package(
        self, registered_agent, organizer, did_document
    ):
        articles = generate_articles(
            entity_name="Kestrel Alpha DAO LLC",
            agent_did=SAMPLE_DID,
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            registered_agent=registered_agent,
            organizer=organizer,
        )
        operating_agreement = generate_operating_agreement(
            articles=articles,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
        )
        package = generate_incorporation_package(
            entity_name="Kestrel Alpha DAO LLC",
            agent_did=SAMPLE_DID,
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
            did_document=did_document,
            registered_agent=registered_agent,
            organizer=organizer,
            operating_agreement_text=operating_agreement,
        )

        assert package.articles.entity_name == "Kestrel Alpha DAO LLC"
        assert package.legal_entity.status == EntityStatus.DRAFT
        assert package.package_hash
        assert package.cost_breakdown["filing_fee"] == FILING_FEE_USD

    def test_package_serialization_roundtrip(
        self, registered_agent, organizer, did_document
    ):
        package = generate_incorporation_package(
            entity_name="Kestrel Alpha DAO LLC",
            agent_did=SAMPLE_DID,
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
            did_document=did_document,
            registered_agent=registered_agent,
            organizer=organizer,
            operating_agreement_text="Test agreement",
        )
        data = package.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = IncorporationPackage.from_dict(restored_data)
        assert restored.articles.entity_name == "Kestrel Alpha DAO LLC"
        assert restored.package_hash == package.package_hash
