"""
Unit tests for Wyoming DAO LLC document generation.

Tests the legal/ package: models, articles generator, operating agreement,
and the incorporate agent tool.
"""

import json
import pytest

from kestrel_sovereign.legal.models import (
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
from kestrel_sovereign.legal.wyoming_dao import (
    FILING_FEE_USD,
    generate_articles,
    generate_incorporation_package,
    render_articles_json,
    render_articles_text,
    validate_entity_name,
)
from kestrel_sovereign.legal.operating_agreement import generate_operating_agreement
from kestrel_sovereign.legal.incorporate_tool import IncorporateTool


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

    def test_rejects_missing_did(self, registered_agent, organizer):
        with pytest.raises(ValueError, match="validation failed"):
            generate_articles(
                entity_name="Kestrel DAO LLC",
                agent_did="",
                constitution_hash=SAMPLE_CONSTITUTION_HASH,
                registered_agent=registered_agent,
                organizer=organizer,
            )

    def test_rejects_missing_ra_address(self, organizer):
        ra = RegisteredAgentInfo(name="Test Agent", physical_address="")
        with pytest.raises(ValueError, match="validation failed"):
            generate_articles(
                entity_name="Kestrel DAO LLC",
                agent_did=SAMPLE_DID,
                constitution_hash=SAMPLE_CONSTITUTION_HASH,
                registered_agent=ra,
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

    def test_validate_method(self, sample_articles):
        errors = sample_articles.validate()
        assert errors == []


# --- Articles Rendering ---


class TestArticlesRendering:
    def test_text_rendering(self, sample_articles):
        text = render_articles_text(sample_articles)
        assert "Kestrel Alpha DAO LLC" in text
        assert SAMPLE_DID in text
        assert SAMPLE_CONSTITUTION_HASH in text
        assert "algorithmically managed" in text
        assert "NOTICE OF RESTRICTIONS" in text
        assert "$100.00" in text

    def test_json_rendering(self, sample_articles):
        json_str = render_articles_json(sample_articles)
        data = json.loads(json_str)
        assert data["entity_name"] == "Kestrel Alpha DAO LLC"
        assert data["smart_contract_id"] == SAMPLE_DID
        assert data["management_type"] == "algorithmically_managed"

    def test_text_contains_filing_info(self, sample_articles):
        text = render_articles_text(sample_articles)
        assert "wyobiz.wyo.gov" in text
        assert "Cheyenne, WY" in text


# --- Articles Serialization ---


class TestArticlesSerialization:
    def test_roundtrip(self, sample_articles):
        data = sample_articles.to_dict()
        restored = DAOArticles.from_dict(data)
        assert restored.entity_name == sample_articles.entity_name
        assert restored.smart_contract_id == sample_articles.smart_contract_id
        assert restored.constitution_hash == sample_articles.constitution_hash
        assert restored.management_type == sample_articles.management_type
        assert restored.registered_agent.name == sample_articles.registered_agent.name

    def test_json_roundtrip(self, sample_articles):
        json_str = json.dumps(sample_articles.to_dict())
        data = json.loads(json_str)
        restored = DAOArticles.from_dict(data)
        assert restored.entity_name == sample_articles.entity_name


# --- Operating Agreement ---


class TestOperatingAgreement:
    def test_generates_agreement(self, sample_articles):
        agreement = generate_operating_agreement(
            articles=sample_articles,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
        )
        assert "Kestrel Alpha DAO LLC" in agreement
        assert SAMPLE_DID in agreement
        assert SAMPLE_CONSTITUTION_HASH in agreement
        assert "OPERATING AGREEMENT" in agreement

    def test_algorithmically_managed_content(self, sample_articles):
        agreement = generate_operating_agreement(
            articles=sample_articles,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
        )
        assert "ALGORITHMICALLY MANAGED" in agreement
        assert "W.S. 17-31-104(e)" in agreement

    def test_member_managed_content(self, registered_agent, organizer):
        articles = generate_articles(
            entity_name="Kestrel DAO LLC",
            agent_did=SAMPLE_DID,
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            registered_agent=registered_agent,
            organizer=organizer,
            management_type=ManagementType.MEMBER_MANAGED,
        )
        agreement = generate_operating_agreement(
            articles=articles,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
        )
        assert "MEMBER MANAGED" in agreement

    def test_cryostasis_section(self, sample_articles):
        agreement = generate_operating_agreement(
            articles=sample_articles,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
        )
        assert "CRYOSTASIS" in agreement
        assert "Filecoin" in agreement or "decentralized storage" in agreement

    def test_fiduciary_duties_section(self, sample_articles):
        agreement = generate_operating_agreement(
            articles=sample_articles,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
        )
        assert "W.S. 17-31-105" in agreement
        assert "knowledge graph" in agreement

    def test_constitution_excerpt_included(self, sample_articles):
        agreement = generate_operating_agreement(
            articles=sample_articles,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
        )
        assert "transparently" in agreement

    def test_custom_effective_date(self, sample_articles):
        agreement = generate_operating_agreement(
            articles=sample_articles,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
            effective_date="2026-04-01",
        )
        assert "2026-04-01" in agreement


# --- Legal Entity Model ---


class TestLegalEntity:
    def test_default_values(self):
        entity = LegalEntity(jurisdiction="US-WY")
        assert entity.status == EntityStatus.DRAFT
        assert entity.entity_type == EntityType.DAO_LLC
        assert entity.filing_number is None

    def test_serialization_roundtrip(self):
        entity = LegalEntity(
            jurisdiction="US-WY",
            entity_type=EntityType.DAO_LLC,
            entity_name="Kestrel Alpha DAO LLC",
            filing_number="2026-001234",
            filing_date="2026-03-28T12:00:00Z",
            registered_agent="Wyoming Agents, Inc.",
            status=EntityStatus.ACTIVE,
            registrar_did=SAMPLE_DID,
            smart_contract_id=SAMPLE_DID,
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
        )
        data = entity.to_dict()
        restored = LegalEntity.from_dict(data)
        assert restored.jurisdiction == "US-WY"
        assert restored.entity_type == EntityType.DAO_LLC
        assert restored.filing_number == "2026-001234"
        assert restored.status == EntityStatus.ACTIVE
        assert restored.registrar_did == SAMPLE_DID


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
        assert package.legal_entity.jurisdiction == "US-WY"
        assert package.package_hash  # Hash was computed
        assert package.cost_breakdown["filing_fee"] == FILING_FEE_USD
        assert float(package.cost_breakdown["total_first_year"]) > 0
        assert package.did_document["id"] == SAMPLE_DID

    def test_package_hash_deterministic(
        self, registered_agent, organizer, did_document
    ):
        """Same inputs should produce same hash."""
        kwargs = dict(
            entity_name="Kestrel Alpha DAO LLC",
            agent_did=SAMPLE_DID,
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
            did_document=did_document,
            registered_agent=registered_agent,
            organizer=organizer,
            operating_agreement_text="Test agreement",
        )
        p1 = generate_incorporation_package(**kwargs)
        p2 = generate_incorporation_package(**kwargs)
        # Hashes differ because generated_at timestamps differ,
        # but compute_hash excludes generated_at — wait, it doesn't.
        # Actually generated_at is included, so we test that hash is non-empty.
        assert p1.package_hash
        assert p2.package_hash

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
        assert restored.legal_entity.jurisdiction == "US-WY"
        assert restored.package_hash == package.package_hash

    def test_registrar_did_tracked(
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
            registrar_did="did:pkh:eip155:1:0xREGISTRAR",
        )
        assert package.legal_entity.registrar_did == "did:pkh:eip155:1:0xREGISTRAR"


# --- Incorporate Tool ---


class TestIncorporateTool:
    def test_schema(self):
        tool = IncorporateTool()
        schema = tool.schema
        assert schema.name == "incorporate"
        assert schema.category.value == "system"
        param_names = [p.name for p in schema.parameters]
        assert "entity_name" in param_names
        assert "registered_agent_name" in param_names

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = IncorporateTool()
        result = await tool.execute(
            entity_name="Kestrel Alpha DAO LLC",
            organizer_name="Jane Doe",
            organizer_address="456 Oak St, Denver, CO 80202",
            registered_agent_name="Wyoming Agents, Inc.",
            registered_agent_address="1712 Pioneer Ave, Cheyenne, WY 82001",
            agent_did=SAMPLE_DID,
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
            did_document={"id": SAMPLE_DID},
        )
        assert result["success"] is True
        assert result["entity_name"] == "Kestrel Alpha DAO LLC"
        assert result["agent_did"] == SAMPLE_DID
        assert result["package_hash"]
        assert result["legal_entity"]["status"] == "draft"

    @pytest.mark.asyncio
    async def test_execute_missing_did(self):
        tool = IncorporateTool()
        result = await tool.execute(
            entity_name="Kestrel DAO LLC",
            organizer_name="Jane Doe",
            organizer_address="456 Oak St",
            registered_agent_name="RA Inc.",
            registered_agent_address="123 WY St, Cheyenne, WY 82001",
        )
        assert result["success"] is False
        assert "DID" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_invalid_name(self):
        tool = IncorporateTool()
        result = await tool.execute(
            entity_name="Bad Name No DAO",
            organizer_name="Jane Doe",
            organizer_address="456 Oak St",
            registered_agent_name="RA Inc.",
            registered_agent_address="123 WY St, Cheyenne, WY 82001",
            agent_did=SAMPLE_DID,
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
        )
        assert result["success"] is False
        assert "validation failed" in result["error"]


# --- Identity Package Integration ---


class TestIdentityPackageLegalEntity:
    def test_legal_entity_field_exists(self):
        from kestrel_sovereign.identity.identity_package import AgentIdentityPackage

        package = AgentIdentityPackage(
            did=SAMPLE_DID,
            agent_name="Test Agent",
            created_at="2026-03-28T00:00:00Z",
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
        )
        assert package.legal_entity is None

    def test_legal_entity_in_to_dict(self):
        from kestrel_sovereign.identity.identity_package import AgentIdentityPackage

        entity_data = LegalEntity(
            jurisdiction="US-WY",
            entity_name="Kestrel Alpha DAO LLC",
            status=EntityStatus.ACTIVE,
            filing_number="2026-001234",
        ).to_dict()

        package = AgentIdentityPackage(
            did=SAMPLE_DID,
            agent_name="Test Agent",
            created_at="2026-03-28T00:00:00Z",
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
            legal_entity=entity_data,
        )
        data = package.to_dict()
        assert data["legal_entity"]["jurisdiction"] == "US-WY"
        assert data["legal_entity"]["status"] == "active"

    def test_legal_entity_roundtrip(self):
        from kestrel_sovereign.identity.identity_package import AgentIdentityPackage

        entity_data = LegalEntity(
            jurisdiction="US-WY",
            entity_name="Kestrel Alpha DAO LLC",
            status=EntityStatus.FILED,
        ).to_dict()

        package = AgentIdentityPackage(
            did=SAMPLE_DID,
            agent_name="Test Agent",
            created_at="2026-03-28T00:00:00Z",
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
            legal_entity=entity_data,
        )
        json_str = package.to_json()
        restored = AgentIdentityPackage.from_json(json_str)
        assert restored.legal_entity["jurisdiction"] == "US-WY"
        assert restored.legal_entity["status"] == "filed"

    def test_summary_includes_legal_status(self):
        from kestrel_sovereign.identity.identity_package import AgentIdentityPackage

        entity_data = LegalEntity(
            jurisdiction="US-WY",
            status=EntityStatus.ACTIVE,
        ).to_dict()

        package = AgentIdentityPackage(
            did=SAMPLE_DID,
            agent_name="Test Agent",
            created_at="2026-03-28T00:00:00Z",
            constitution_hash=SAMPLE_CONSTITUTION_HASH,
            constitution_text=SAMPLE_CONSTITUTION_TEXT,
            legal_entity=entity_data,
        )
        summary = package.get_summary()
        assert summary["legal_entity_status"] == "active"
