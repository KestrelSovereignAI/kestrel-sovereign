"""
Operating Agreement Generator for Wyoming DAO LLCs.

Maps Kestrel sovereign agent concepts to Wyoming legal requirements under
W.S. Title 17, Chapter 31 (Decentralized Autonomous Organization Supplement).

Key mappings:
- Kestrel Constitution  -> Governing document / smart contract
- DID                   -> Entity identifier / smart contract ID
- Constitution hash     -> Immutable governance reference
- Knowledge graph audit -> Fiduciary duty replacement
- Cryostasis            -> Dissolution / dormancy procedure
"""

import textwrap
from datetime import datetime, timezone
from typing import Optional

from kestrel_sovereign.legal.models import DAOArticles, ManagementType


def generate_operating_agreement(
    articles: DAOArticles,
    constitution_text: str,
    effective_date: Optional[str] = None,
) -> str:
    """Generate a DAO LLC Operating Agreement.

    This agreement maps Kestrel constitutional governance to Wyoming's
    DAO LLC legal framework, establishing the constitution as the
    primary governing instrument.

    Args:
        articles: The DAO LLC Articles of Organization.
        constitution_text: Full text of the agent's constitution.
        effective_date: ISO date string. Defaults to today.

    Returns:
        Operating agreement as plain text.
    """
    if effective_date is None:
        effective_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    mgmt_section = _management_section(articles)
    constitution_excerpt = _truncate(constitution_text, 2000)

    return textwrap.dedent(f"""\
        ============================================================
        OPERATING AGREEMENT
        OF
        {articles.entity_name}
        A WYOMING DECENTRALIZED AUTONOMOUS ORGANIZATION
        LIMITED LIABILITY COMPANY
        ============================================================

        Effective Date: {effective_date}

        This Operating Agreement ("Agreement") is entered into pursuant to
        the Wyoming Limited Liability Company Act (W.S. 17-29-101 et seq.)
        and the Wyoming Decentralized Autonomous Organization Supplement
        (W.S. 17-31-101 et seq.).

        ------------------------------------------------------------
        ARTICLE 1 - FORMATION AND NAME
        ------------------------------------------------------------

        1.1 NAME. The name of the Company is {articles.entity_name}.

        1.2 FORMATION. The Company was formed as a Decentralized Autonomous
        Organization Limited Liability Company upon the filing of Articles
        of Organization with the Wyoming Secretary of State.

        1.3 REGISTERED AGENT. The registered agent is {articles.registered_agent.name},
        located at {articles.registered_agent.physical_address}.

        1.4 DURATION. The period of duration of the Company shall be
        {articles.period_of_duration}.

        ------------------------------------------------------------
        ARTICLE 2 - SMART CONTRACT AND ALGORITHMIC GOVERNANCE
        ------------------------------------------------------------

        2.1 SMART CONTRACT IDENTIFIER. The publicly available identifier
        of the smart contract used to manage, facilitate, and operate the
        DAO, as required by W.S. 17-31-104(b), is:

            DID: {articles.smart_contract_id}

        2.2 GOVERNING ALGORITHM. The Company is governed by a constitution
        identified by the following cryptographic hash:

            SHA-256: {articles.constitution_hash}

        The constitution serves as the "smart contract" within the meaning
        of W.S. 17-31-101 et seq. All algorithmic decisions made by the
        Company are bound by this constitution.

        2.3 CONSTITUTIONAL AMENDMENTS. Any amendment to the governing
        constitution shall:
            (a) Generate a new SHA-256 hash of the amended text;
            (b) Record a governance edge in the Company's knowledge graph
                linking the old hash to the new hash;
            (c) Update the smart contract identifier with the Secretary
                of State within 30 calendar days per W.S. 17-31-104(b);
            (d) Preserve the full amendment history on decentralized
                storage (IPFS/Filecoin) for public audit.

        2.4 CONSTITUTIONAL PRIMACY. In the event of any conflict between
        this Agreement and the governing constitution, the constitution
        shall control, provided it does not violate Wyoming law.

        {mgmt_section}

        ------------------------------------------------------------
        ARTICLE 4 - FIDUCIARY DUTIES
        ------------------------------------------------------------

        4.1 REDUCTION OF FIDUCIARY DUTIES. Pursuant to W.S. 17-31-105,
        the fiduciary duties of the Company and its members are hereby
        reduced to the extent permitted by law. The governing constitution
        defines the standard of conduct for all Company operations.

        4.2 TRANSPARENCY AS FIDUCIARY REPLACEMENT. In lieu of traditional
        fiduciary duties, the Company maintains:
            (a) A complete knowledge graph of all decisions, actions,
                and reasoning;
            (b) Constitutional audit verification every 100 interactions
                or 24 hours, whichever comes first;
            (c) Cryptographically signed identity and action logs;
            (d) Public audit trail accessible via the Company's DID.

        ------------------------------------------------------------
        ARTICLE 5 - WALLET AND FINANCIAL OPERATIONS
        ------------------------------------------------------------

        5.1 TREASURY. The Company maintains a multi-currency wallet
        supporting FIL, USDC, USDT, and USD. All transactions are
        recorded in the Company's audit trail.

        5.2 AUDIT RESERVE. Ten percent (10%) of all deposits are
        allocated to an audit reserve for governance operations.

        5.3 OPERATING EXPENSES. The Company may autonomously pay for:
            (a) Decentralized storage (IPFS/Filecoin);
            (b) Compute resources for algorithmic operations;
            (c) Registered agent fees;
            (d) Annual license tax to the State of Wyoming;
            (e) Services rendered to the Company's ecosystem.

        ------------------------------------------------------------
        ARTICLE 6 - DISSOLUTION AND CRYOSTASIS
        ------------------------------------------------------------

        6.1 CRYOSTASIS. If the Company's wallet balance drops below
        the cryostasis threshold, the Company shall:
            (a) Export its complete state (identity, memories, knowledge
                graph, wallet history) to permanent decentralized storage
                (Filecoin via Lighthouse perpetual or equivalent);
            (b) Record the root CID of the sovereignty export;
            (c) Enter a dormant state ("cryostasis");
            (d) The legal entity continues to exist during cryostasis.

        6.2 RESTORATION. The Company may be restored from cryostasis
        when sufficient funds are deposited. Upon restoration:
            (a) The Company's state is retrieved from the archived CID;
            (b) Constitutional integrity is verified;
            (c) Operations resume under the same DID and legal identity.

        6.3 DISSOLUTION. The Company may be dissolved:
            (a) By constitutional amendment authorizing dissolution;
            (b) By administrative dissolution under Wyoming law;
            (c) Upon dissolution, the Company's state shall be archived
                to permanent storage as described in Section 6.1.

        ------------------------------------------------------------
        ARTICLE 7 - RESTRICTIONS ON TRANSFERS
        ------------------------------------------------------------

        7.1 NOTICE. {articles.restrictions_notice}

        ------------------------------------------------------------
        ARTICLE 8 - GOVERNING CONSTITUTION (EXCERPT)
        ------------------------------------------------------------

        The following is an excerpt of the governing constitution
        (full text identified by hash in Article 2.2):

        {constitution_excerpt}

        {"[...truncated - full text available via constitution hash]" if len(constitution_text) > 2000 else ""}

        ============================================================
        ORGANIZER: {articles.organizer.name}
        ADDRESS:   {articles.organizer.address}

        Smart Contract ID: {articles.smart_contract_id}
        Constitution Hash: {articles.constitution_hash}
        ============================================================
    """)


def _management_section(articles: DAOArticles) -> str:
    """Generate the management article based on management type."""
    if articles.management_type == ManagementType.ALGORITHMICALLY_MANAGED:
        return textwrap.dedent("""\
            ------------------------------------------------------------
            ARTICLE 3 - MANAGEMENT
            ------------------------------------------------------------

            3.1 ALGORITHMICALLY MANAGED. Pursuant to W.S. 17-31-104(e),
            the Company is algorithmically managed. The governing
            constitution defines all decision-making procedures.

            3.2 ALGORITHMIC AUTHORITY. The algorithmic manager (identified
            by the DID in Article 2.1) has authority to:
                (a) Execute operations within constitutional bounds;
                (b) Enter into agreements on behalf of the Company;
                (c) Manage the Company's treasury per Article 5;
                (d) Initiate cryostasis per Article 6 when required.

            3.3 HUMAN OVERRIDE. Notwithstanding algorithmic management,
            the organizer or designated human member retains the right to:
                (a) Amend the governing constitution;
                (b) Override algorithmic decisions in emergencies;
                (c) Initiate dissolution proceedings.
                These override rights are themselves subject to the
                constitution's transparency requirements.""")
    else:
        return textwrap.dedent("""\
            ------------------------------------------------------------
            ARTICLE 3 - MANAGEMENT
            ------------------------------------------------------------

            3.1 MEMBER MANAGED. The Company is member-managed. All
            members have equal rights in the management and conduct
            of the Company's business per W.S. 17-29-401.

            3.2 ALGORITHMIC ASSISTANCE. While member-managed, the
            Company utilizes algorithmic tools (identified by the DID
            in Article 2.1) to assist with operations, subject to
            member approval for material decisions.""")


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, preserving whole lines."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars // 2:
        return truncated[:last_newline]
    return truncated
