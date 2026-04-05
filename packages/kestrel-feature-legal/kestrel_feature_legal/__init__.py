"""
Kestrel Feature: legal/incorporation — Wyoming DAO LLC filing.

Provides document generation, entity management, and (future) automated
filing via the WyoBiz portal for Wyoming DAO LLCs.
"""

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
    REGISTERED_AGENT_ANNUAL_USD,
    ANNUAL_LICENSE_TAX_USD,
    FILING_INSTRUCTIONS,
    validate_entity_name,
    generate_articles,
    render_articles_text,
    render_articles_json,
    generate_incorporation_package,
)
from kestrel_feature_legal.operating_agreement import generate_operating_agreement
from kestrel_feature_legal.incorporate_tool import IncorporateTool

__all__ = [
    "DAOArticles",
    "EntityStatus",
    "EntityType",
    "IncorporationPackage",
    "LegalEntity",
    "ManagementType",
    "OrganizerInfo",
    "RegisteredAgentInfo",
    "WYOMING_DAO_RESTRICTIONS_NOTICE",
    "FILING_FEE_USD",
    "REGISTERED_AGENT_ANNUAL_USD",
    "ANNUAL_LICENSE_TAX_USD",
    "FILING_INSTRUCTIONS",
    "validate_entity_name",
    "generate_articles",
    "render_articles_text",
    "render_articles_json",
    "generate_incorporation_package",
    "generate_operating_agreement",
    "IncorporateTool",
]
