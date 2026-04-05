"""
Incorporate Tool: Agent tool for Wyoming DAO LLC formation.

Allows any Kestrel agent to generate incorporation documents for itself
or request filing via the Kestrel Registrar agent.

Usage:
    - Agent calls `incorporate` tool with entity name and organizer info
    - Tool generates complete IncorporationPackage (articles + operating agreement)
    - If registrar_did is provided, the Registrar handles filing
    - Otherwise, documents are generated for manual human filing
"""

import logging
from typing import Any, Dict, Optional

from kestrel_sovereign.tools.base import AgentTool, ToolSchema, ToolParameter, ToolCategory
from kestrel_sovereign.legal.models import (
    IncorporationPackage,
    ManagementType,
    OrganizerInfo,
    RegisteredAgentInfo,
)
from kestrel_sovereign.legal.wyoming_dao import (
    generate_incorporation_package,
    render_articles_text,
)
from kestrel_sovereign.legal.operating_agreement import generate_operating_agreement
from kestrel_sovereign.legal.wyoming_dao import generate_articles

logger = logging.getLogger(__name__)


class IncorporateTool(AgentTool):
    """Agent tool for Wyoming DAO LLC incorporation.

    This tool generates all documents needed to incorporate a Kestrel agent
    as a Wyoming DAO LLC. The agent's DID serves as the smart contract
    identifier and its constitution hash as the governance reference.
    """

    def __init__(self, agent=None):
        """Initialize with optional agent reference for DID/constitution access.

        Args:
            agent: The Kestrel agent instance (provides DID, constitution, etc.)
        """
        self._agent = agent

    @property
    def name(self) -> str:
        return "incorporate"

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="incorporate",
            description=(
                "Generate Wyoming DAO LLC formation documents. Creates Articles "
                "of Organization and Operating Agreement using the agent's DID as "
                "the smart contract identifier and constitution hash as the "
                "governance reference. Returns a complete IncorporationPackage "
                "ready for filing with the Wyoming Secretary of State."
            ),
            category=ToolCategory.SYSTEM,
            parameters=[
                ToolParameter(
                    name="entity_name",
                    type="string",
                    description=(
                        'Proposed entity name. Must contain "DAO" and "LLC". '
                        'Example: "Kestrel Alpha DAO LLC"'
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="organizer_name",
                    type="string",
                    description="Name of the person or entity organizing the formation.",
                    required=True,
                ),
                ToolParameter(
                    name="organizer_address",
                    type="string",
                    description="Address of the organizer.",
                    required=True,
                ),
                ToolParameter(
                    name="registered_agent_name",
                    type="string",
                    description="Name of the Wyoming registered agent.",
                    required=True,
                ),
                ToolParameter(
                    name="registered_agent_address",
                    type="string",
                    description="Wyoming physical street address of the registered agent.",
                    required=True,
                ),
                ToolParameter(
                    name="management_type",
                    type="string",
                    description="DAO management type.",
                    required=False,
                    default="algorithmically_managed",
                    enum=["algorithmically_managed", "member_managed"],
                ),
                ToolParameter(
                    name="registrar_did",
                    type="string",
                    description=(
                        "DID of the Kestrel Registrar agent for automated filing. "
                        "If omitted, generates documents for manual filing."
                    ),
                    required=False,
                ),
            ],
            examples=[
                {
                    "input": {
                        "entity_name": "Kestrel Alpha DAO LLC",
                        "organizer_name": "Jane Doe",
                        "organizer_address": "123 Main St, Cheyenne, WY 82001",
                        "registered_agent_name": "Wyoming Agents, Inc.",
                        "registered_agent_address": "1712 Pioneer Ave, Cheyenne, WY 82001",
                    },
                    "description": "Generate incorporation documents for Kestrel Alpha",
                },
            ],
        )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Generate incorporation documents.

        If self._agent is set, uses the agent's DID and constitution.
        Otherwise, requires them to be passed explicitly.
        """
        entity_name = kwargs["entity_name"]
        management_type = ManagementType(
            kwargs.get("management_type", "algorithmically_managed")
        )
        registrar_did = kwargs.get("registrar_did")

        # Build organizer and registered agent from params
        organizer = OrganizerInfo(
            name=kwargs["organizer_name"],
            address=kwargs["organizer_address"],
            is_agent=bool(registrar_did),
        )
        registered_agent = RegisteredAgentInfo(
            name=kwargs["registered_agent_name"],
            physical_address=kwargs["registered_agent_address"],
        )

        # Get DID and constitution from agent if available
        agent_did = kwargs.get("agent_did", "")
        constitution_hash = kwargs.get("constitution_hash", "")
        constitution_text = kwargs.get("constitution_text", "")
        did_document = kwargs.get("did_document", {})

        if self._agent:
            agent_did = agent_did or getattr(self._agent, "did", "")
            constitution_hash = constitution_hash or getattr(
                self._agent, "constitution_hash", ""
            )
            constitution_text = constitution_text or getattr(
                self._agent, "constitution_text", ""
            )
            # Try to get DID document from inception service
            if not did_document and hasattr(self._agent, "did_document"):
                did_document = self._agent.did_document

        if not agent_did:
            return {
                "success": False,
                "error": "Agent DID is required. Set agent reference or pass agent_did.",
            }

        if not constitution_hash:
            return {
                "success": False,
                "error": "Constitution hash is required. Agent must have a constitution.",
            }

        # Generate articles first (for operating agreement)
        try:
            articles = generate_articles(
                entity_name=entity_name,
                agent_did=agent_did,
                constitution_hash=constitution_hash,
                registered_agent=registered_agent,
                organizer=organizer,
                management_type=management_type,
            )
        except ValueError as e:
            return {"success": False, "error": str(e)}

        # Generate operating agreement
        operating_agreement_text = generate_operating_agreement(
            articles=articles,
            constitution_text=constitution_text,
        )

        # Generate full package
        package = generate_incorporation_package(
            entity_name=entity_name,
            agent_did=agent_did,
            constitution_hash=constitution_hash,
            constitution_text=constitution_text,
            did_document=did_document,
            registered_agent=registered_agent,
            organizer=organizer,
            operating_agreement_text=operating_agreement_text,
            management_type=management_type,
            registrar_did=registrar_did,
        )

        # Render human-readable articles
        articles_text = render_articles_text(articles)

        logger.info(
            "Generated incorporation package for %s (DID: %s, hash: %s)",
            entity_name,
            agent_did[:30] + "...",
            package.package_hash[:16] + "...",
        )

        return {
            "success": True,
            "entity_name": entity_name,
            "agent_did": agent_did,
            "constitution_hash": constitution_hash,
            "management_type": management_type.value,
            "articles_text": articles_text,
            "operating_agreement_preview": operating_agreement_text[:500] + "...",
            "filing_instructions": package.filing_instructions,
            "cost_breakdown": package.cost_breakdown,
            "package_hash": package.package_hash,
            "legal_entity": package.legal_entity.to_dict(),
            "package": package.to_dict(),
        }
