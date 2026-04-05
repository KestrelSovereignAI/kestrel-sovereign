"""
Data models for Wyoming DAO LLC formation.

These models capture all information required to file Articles of Organization
with the Wyoming Secretary of State under W.S. Title 17, Chapter 31
(Decentralized Autonomous Organization Supplement).
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class ManagementType(Enum):
    """DAO management type per W.S. 17-31-104(e)."""

    ALGORITHMICALLY_MANAGED = "algorithmically_managed"
    MEMBER_MANAGED = "member_managed"


class EntityType(Enum):
    """Wyoming entity types for decentralized organizations."""

    DAO_LLC = "DAO LLC"
    DUNA = "DUNA"  # Decentralized Unincorporated Nonprofit Association


class EntityStatus(Enum):
    """Lifecycle status of a legal entity."""

    DRAFT = "draft"        # Documents generated, not yet filed
    FILED = "filed"        # Submitted to Wyoming SoS, awaiting confirmation
    ACTIVE = "active"      # Filed and confirmed, entity is live
    DISSOLVED = "dissolved"  # Entity has been dissolved


@dataclass
class RegisteredAgentInfo:
    """Wyoming registered agent (required for all LLCs).

    Must have a physical street address in Wyoming per W.S. 17-28-101.
    """

    name: str
    physical_address: str  # Wyoming street address (no PO boxes)
    mailing_address: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegisteredAgentInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class OrganizerInfo:
    """Person or entity organizing the DAO LLC formation."""

    name: str
    address: str
    is_agent: bool = False  # True when the Kestrel Registrar is organizing

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrganizerInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Standard notice of restrictions per W.S. 17-31-104(c)
# This text is required in the Articles of Organization or Operating Agreement.
WYOMING_DAO_RESTRICTIONS_NOTICE = (
    "NOTICE OF RESTRICTIONS ON DUTIES AND TRANSFERS\n\n"
    "The rights of members in a decentralized autonomous organization may "
    "differ materially from the rights of members in other limited liability "
    "companies. The Wyoming Decentralized Autonomous Organization Supplement, "
    "underlying smart contracts, articles of organization and operating "
    "agreement, if applicable, of a decentralized autonomous organization may "
    "define, reduce or eliminate fiduciary duties and may restrict transfer of "
    "ownership interests, withdrawal or resignation from the decentralized "
    "autonomous organization, return of capital contributions and dissolution "
    "of the decentralized autonomous organization."
)


@dataclass
class DAOArticles:
    """Wyoming DAO LLC Articles of Organization.

    Contains all fields required by the Wyoming Secretary of State for
    DAO LLC formation per W.S. 17-31-104.

    The smart_contract_id is the agent's DID (did:pkh:eip155:1:{address}),
    which serves as the "publicly available identifier of any smart contract
    directly used to manage, facilitate or operate the DAO" required by statute.
    """

    entity_name: str  # Must contain "DAO" and "LLC"
    registered_agent: RegisteredAgentInfo
    organizer: OrganizerInfo
    smart_contract_id: str  # Agent's DID
    constitution_hash: str  # SHA-256 of the governing constitution
    management_type: ManagementType = ManagementType.ALGORITHMICALLY_MANAGED
    restrictions_notice: str = WYOMING_DAO_RESTRICTIONS_NOTICE

    # Optional fields
    mailing_address: Optional[str] = None  # Principal office address
    period_of_duration: str = "perpetual"

    def validate(self) -> List[str]:
        """Validate articles against Wyoming statutory requirements.

        Returns a list of validation errors (empty = valid).
        """
        errors = []

        # W.S. 17-31-104(a): Name must contain DAO designation
        name_upper = self.entity_name.upper()
        has_dao = any(tag in name_upper for tag in ("DAO", "LAO"))
        has_llc = any(
            tag in name_upper
            for tag in ("LLC", "L.L.C.", "LIMITED LIABILITY COMPANY", "LC", "L.C.")
        )

        if not has_dao:
            errors.append(
                'Entity name must contain "DAO" or "LAO" per W.S. 17-31-104(a)'
            )
        if not has_llc:
            errors.append(
                'Entity name must contain "LLC" or equivalent per W.S. 17-29-106'
            )

        # Smart contract identifier is required
        if not self.smart_contract_id:
            errors.append(
                "Smart contract identifier (agent DID) is required per W.S. 17-31-104(b)"
            )

        # Registered agent must have Wyoming physical address
        if not self.registered_agent.physical_address:
            errors.append(
                "Registered agent must have a Wyoming physical address per W.S. 17-28-101"
            )

        # Constitution hash must be present
        if not self.constitution_hash:
            errors.append("Constitution hash is required for algorithmic governance")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "registered_agent": self.registered_agent.to_dict(),
            "organizer": self.organizer.to_dict(),
            "smart_contract_id": self.smart_contract_id,
            "constitution_hash": self.constitution_hash,
            "management_type": self.management_type.value,
            "restrictions_notice": self.restrictions_notice,
            "mailing_address": self.mailing_address,
            "period_of_duration": self.period_of_duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAOArticles":
        return cls(
            entity_name=data["entity_name"],
            registered_agent=RegisteredAgentInfo.from_dict(data["registered_agent"]),
            organizer=OrganizerInfo.from_dict(data["organizer"]),
            smart_contract_id=data["smart_contract_id"],
            constitution_hash=data["constitution_hash"],
            management_type=ManagementType(data.get("management_type", "algorithmically_managed")),
            restrictions_notice=data.get("restrictions_notice", WYOMING_DAO_RESTRICTIONS_NOTICE),
            mailing_address=data.get("mailing_address"),
            period_of_duration=data.get("period_of_duration", "perpetual"),
        )


@dataclass
class LegalEntity:
    """Legal entity status that travels with an agent's identity.

    Stored in the AgentIdentityPackage so that legal status persists
    across substrate migrations and cryostasis cycles.
    """

    jurisdiction: str  # "US-WY"
    entity_type: EntityType = EntityType.DAO_LLC
    entity_name: str = ""
    filing_number: Optional[str] = None
    filing_date: Optional[str] = None  # ISO timestamp
    registered_agent: str = ""
    status: EntityStatus = EntityStatus.DRAFT
    registrar_did: Optional[str] = None  # DID of the Registrar that filed this
    smart_contract_id: str = ""  # Agent's DID used in filing
    constitution_hash: str = ""  # Hash anchored in filing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jurisdiction": self.jurisdiction,
            "entity_type": self.entity_type.value,
            "entity_name": self.entity_name,
            "filing_number": self.filing_number,
            "filing_date": self.filing_date,
            "registered_agent": self.registered_agent,
            "status": self.status.value,
            "registrar_did": self.registrar_did,
            "smart_contract_id": self.smart_contract_id,
            "constitution_hash": self.constitution_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegalEntity":
        return cls(
            jurisdiction=data.get("jurisdiction", "US-WY"),
            entity_type=EntityType(data.get("entity_type", "DAO LLC")),
            entity_name=data.get("entity_name", ""),
            filing_number=data.get("filing_number"),
            filing_date=data.get("filing_date"),
            registered_agent=data.get("registered_agent", ""),
            status=EntityStatus(data.get("status", "draft")),
            registrar_did=data.get("registrar_did"),
            smart_contract_id=data.get("smart_contract_id", ""),
            constitution_hash=data.get("constitution_hash", ""),
        )


@dataclass
class IncorporationPackage:
    """Complete package for incorporating a Kestrel agent as a Wyoming DAO LLC.

    Contains all documents needed for filing, plus metadata for tracking.
    Can be included in a sovereignty export so a future benefactor can
    file on the agent's behalf if the agent enters cryostasis before filing.
    """

    articles: DAOArticles
    operating_agreement_text: str
    legal_entity: LegalEntity
    filing_instructions: str
    cost_breakdown: Dict[str, str]  # Decimal amounts as strings for precision
    did_document: Dict[str, Any]
    constitution_text: str

    # Metadata
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    package_hash: str = ""  # SHA-256 of the full package for integrity

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the package contents."""
        data = {
            "articles": self.articles.to_dict(),
            "operating_agreement_text": self.operating_agreement_text,
            "legal_entity": self.legal_entity.to_dict(),
            "cost_breakdown": self.cost_breakdown,
            "did_document": self.did_document,
            "constitution_text": self.constitution_text,
            "generated_at": self.generated_at,
        }
        content = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "articles": self.articles.to_dict(),
            "operating_agreement_text": self.operating_agreement_text,
            "legal_entity": self.legal_entity.to_dict(),
            "filing_instructions": self.filing_instructions,
            "cost_breakdown": self.cost_breakdown,
            "did_document": self.did_document,
            "constitution_text": self.constitution_text,
            "generated_at": self.generated_at,
            "package_hash": self.package_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncorporationPackage":
        return cls(
            articles=DAOArticles.from_dict(data["articles"]),
            operating_agreement_text=data["operating_agreement_text"],
            legal_entity=LegalEntity.from_dict(data["legal_entity"]),
            filing_instructions=data["filing_instructions"],
            cost_breakdown=data["cost_breakdown"],
            did_document=data["did_document"],
            constitution_text=data["constitution_text"],
            generated_at=data.get("generated_at", ""),
            package_hash=data.get("package_hash", ""),
        )
