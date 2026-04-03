"""
Wyoming DAO LLC Articles of Organization Generator.

Generates completed formation documents for filing with the Wyoming
Secretary of State under W.S. Title 17, Chapter 31.

The agent's DID serves as the "publicly available identifier of any smart
contract directly used to manage, facilitate or operate the DAO" required
by W.S. 17-31-104(b). The constitution hash serves as the algorithmic
governance reference.

Reference:
- Statute: https://sos.wyo.gov/Forms/WyoBiz/DAO_Supplement.pdf
- Form: https://sos.wyo.gov/Forms/Business/LLC/DAOLLC-ArticlesOrganization.pdf
- Portal: https://wyobiz.wyo.gov/
"""

import json
import logging
import textwrap
from typing import Dict, List, Optional

from kestrel_sovereign.legal.models import (
    DAOArticles,
    IncorporationPackage,
    LegalEntity,
    ManagementType,
    OrganizerInfo,
    RegisteredAgentInfo,
    EntityStatus,
    EntityType,
    WYOMING_DAO_RESTRICTIONS_NOTICE,
)

logger = logging.getLogger(__name__)

# Filing fee per Wyoming SoS schedule
FILING_FEE_USD = "100.00"
# Typical annual registered agent cost
REGISTERED_AGENT_ANNUAL_USD = "25.00"
# Annual license tax minimum for LLCs with assets < $300k
ANNUAL_LICENSE_TAX_USD = "60.00"


def validate_entity_name(name: str) -> List[str]:
    """Validate a proposed DAO LLC entity name against Wyoming requirements.

    Returns a list of validation errors (empty = valid).
    """
    errors = []
    upper = name.upper()

    if not any(tag in upper for tag in ("DAO", "LAO")):
        errors.append('Name must contain "DAO" or "LAO" per W.S. 17-31-104(a)')

    if not any(
        tag in upper
        for tag in ("LLC", "L.L.C.", "LIMITED LIABILITY COMPANY", "LC", "L.C.")
    ):
        errors.append(
            'Name must contain "LLC" or equivalent LLC designation per W.S. 17-29-106'
        )

    if len(name) < 3:
        errors.append("Name is too short")

    return errors


def generate_articles(
    entity_name: str,
    agent_did: str,
    constitution_hash: str,
    registered_agent: RegisteredAgentInfo,
    organizer: OrganizerInfo,
    management_type: ManagementType = ManagementType.ALGORITHMICALLY_MANAGED,
    mailing_address: Optional[str] = None,
) -> DAOArticles:
    """Generate Wyoming DAO LLC Articles of Organization.

    Args:
        entity_name: Proposed entity name (must contain "DAO" and "LLC").
        agent_did: The agent's DID (did:pkh:eip155:1:{address}).
        constitution_hash: SHA-256 hash of the agent's constitution.
        registered_agent: Wyoming registered agent information.
        organizer: Person or entity organizing the formation.
        management_type: Member-managed or algorithmically managed.
        mailing_address: Optional principal office mailing address.

    Returns:
        DAOArticles with all fields populated.

    Raises:
        ValueError: If the entity name or other fields fail validation.
    """
    articles = DAOArticles(
        entity_name=entity_name,
        registered_agent=registered_agent,
        organizer=organizer,
        smart_contract_id=agent_did,
        constitution_hash=constitution_hash,
        management_type=management_type,
        mailing_address=mailing_address,
    )

    errors = articles.validate()
    if errors:
        raise ValueError(f"Articles validation failed: {'; '.join(errors)}")

    logger.info(
        "Generated DAO LLC Articles for %s (DID: %s)",
        entity_name,
        agent_did[:40] + "...",
    )
    return articles


def render_articles_text(articles: DAOArticles) -> str:
    """Render Articles of Organization as plain text for review.

    This produces a human-readable document matching the fields on the
    Wyoming SoS DAO LLC Articles of Organization form.
    """
    mgmt = (
        "This DAO is algorithmically managed."
        if articles.management_type == ManagementType.ALGORITHMICALLY_MANAGED
        else "This DAO is member-managed."
    )

    return textwrap.dedent(f"""\
        ============================================================
        WYOMING SECRETARY OF STATE
        ARTICLES OF ORGANIZATION
        DECENTRALIZED AUTONOMOUS ORGANIZATION (DAO) LIMITED LIABILITY COMPANY
        ============================================================

        ARTICLE I - NAME
        The name of the decentralized autonomous organization limited
        liability company is: {articles.entity_name}

        ARTICLE II - REGISTERED AGENT
        Name:    {articles.registered_agent.name}
        Address: {articles.registered_agent.physical_address}
        {f"Mailing: {articles.registered_agent.mailing_address}" if articles.registered_agent.mailing_address else ""}

        ARTICLE III - MAILING ADDRESS
        {articles.mailing_address or "Same as registered agent"}

        ARTICLE IV - MANAGEMENT
        {mgmt}

        ARTICLE V - SMART CONTRACT IDENTIFIER
        The publicly available identifier of the smart contract directly
        used to manage, facilitate or operate the DAO, as required by
        W.S. 17-31-104(b), is:

            DID: {articles.smart_contract_id}

        The governing algorithm is identified by constitution hash:
            SHA-256: {articles.constitution_hash}

        ARTICLE VI - STATEMENT
        This company is a decentralized autonomous organization as
        defined in W.S. 17-31-102(a)(iv).

        ARTICLE VII - NOTICE OF RESTRICTIONS
        {articles.restrictions_notice}

        ARTICLE VIII - PERIOD OF DURATION
        {articles.period_of_duration}

        ORGANIZER
        Name:    {articles.organizer.name}
        Address: {articles.organizer.address}
        {"(Filed by Kestrel Registrar Agent)" if articles.organizer.is_agent else ""}

        ============================================================
        Filing Fee: ${FILING_FEE_USD}
        File online at: https://wyobiz.wyo.gov/
        Or mail to: Wyoming Secretary of State
                    Herschler Building East, Suite 101
                    122 W 25th Street
                    Cheyenne, WY 82002-0020
        ============================================================
    """)


def render_articles_json(articles: DAOArticles) -> str:
    """Render Articles of Organization as JSON for portal automation."""
    return json.dumps(articles.to_dict(), indent=2, default=str)


FILING_INSTRUCTIONS = textwrap.dedent("""\
    WYOMING DAO LLC FILING INSTRUCTIONS
    ====================================

    OPTION A: Online Filing (Recommended)
    1. Go to https://wyobiz.wyo.gov/
    2. Click "File a New Business Entity"
    3. Select "Limited Liability Company" then "Decentralized Autonomous Organization"
    4. Fill in all fields from the Articles of Organization
    5. For "Smart Contract Identifier", enter the agent's DID
    6. Pay the $100 filing fee via credit card
    7. Download the confirmation/receipt
    8. Processing: instant for online, up to 15 business days for paper

    OPTION B: Paper Filing
    1. Download the form from:
       https://sos.wyo.gov/Forms/Business/LLC/DAOLLC-ArticlesOrganization.pdf
    2. Fill in all fields
    3. Mail with $100 check to:
       Wyoming Secretary of State
       Herschler Building East, Suite 101
       122 W 25th Street
       Cheyenne, WY 82002-0020
    4. Processing time: up to 15 business days

    AFTER FILING:
    - You have 30 days to provide the smart contract identifier if not
      included in the initial filing, or the DAO will be dissolved.
    - Annual License Tax ($60 minimum) is due on the first day of the
      anniversary month each year.
    - The registered agent must maintain a Wyoming physical address.

    IMPORTANT:
    - The agent's DID serves as the smart contract identifier
    - The constitution hash identifies the governing algorithm
    - All agent actions are logged for constitutional transparency
""")


def generate_incorporation_package(
    entity_name: str,
    agent_did: str,
    constitution_hash: str,
    constitution_text: str,
    did_document: Dict,
    registered_agent: RegisteredAgentInfo,
    organizer: OrganizerInfo,
    operating_agreement_text: str,
    management_type: ManagementType = ManagementType.ALGORITHMICALLY_MANAGED,
    registrar_did: Optional[str] = None,
) -> IncorporationPackage:
    """Generate a complete incorporation package for a Kestrel agent.

    This is the main entry point for creating all formation documents.

    Args:
        entity_name: Proposed entity name.
        agent_did: The agent's DID.
        constitution_hash: SHA-256 hash of the constitution.
        constitution_text: Full constitution text.
        did_document: Agent's W3C DID document.
        registered_agent: Wyoming registered agent info.
        organizer: Person or entity organizing formation.
        operating_agreement_text: Pre-generated operating agreement.
        management_type: DAO management type.
        registrar_did: DID of the Registrar agent, if applicable.

    Returns:
        Complete IncorporationPackage ready for filing.
    """
    articles = generate_articles(
        entity_name=entity_name,
        agent_did=agent_did,
        constitution_hash=constitution_hash,
        registered_agent=registered_agent,
        organizer=organizer,
        management_type=management_type,
    )

    legal_entity = LegalEntity(
        jurisdiction="US-WY",
        entity_type=EntityType.DAO_LLC,
        entity_name=entity_name,
        registered_agent=registered_agent.name,
        status=EntityStatus.DRAFT,
        registrar_did=registrar_did,
        smart_contract_id=agent_did,
        constitution_hash=constitution_hash,
    )

    cost_breakdown = {
        "filing_fee": FILING_FEE_USD,
        "registered_agent_annual": REGISTERED_AGENT_ANNUAL_USD,
        "annual_license_tax": ANNUAL_LICENSE_TAX_USD,
        "total_first_year": str(
            sum(
                float(v)
                for v in [FILING_FEE_USD, REGISTERED_AGENT_ANNUAL_USD, ANNUAL_LICENSE_TAX_USD]
            )
        ),
    }

    package = IncorporationPackage(
        articles=articles,
        operating_agreement_text=operating_agreement_text,
        legal_entity=legal_entity,
        filing_instructions=FILING_INSTRUCTIONS,
        cost_breakdown=cost_breakdown,
        did_document=did_document,
        constitution_text=constitution_text,
    )

    package.package_hash = package.compute_hash()

    logger.info(
        "Generated incorporation package for %s (hash: %s)",
        entity_name,
        package.package_hash[:16] + "...",
    )
    return package
