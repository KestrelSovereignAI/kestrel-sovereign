#!/usr/bin/env python3
"""
Convene Constitutional Council for SQLite-First Architecture Decision

This script runs a formal council session with the configured foundation models
to deliberate on Issue #2: SQLite-First with Sync Layer architecture.

Context from prior analysis:
- Issue #2 proposes making SQLite the only runtime database
- A sync layer already exists (WALListener, SyncService, S3Target, LighthouseTarget)
- PostgreSQL is already marked deprecated in the codebase
- The abstract DatabaseBackend interface adds some overhead
- Key tension: sovereignty/simplicity vs multi-tenant/proven infrastructure
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from _council_feature_package import load_council_exports

_council = load_council_exports()
Evidence = _council.Evidence
CouncilConfig = _council.CouncilConfig
CouncilMember = _council.CouncilMember
ConsensusRule = _council.ConsensusRule
convene_council = _council.convene_council
print_token_usage_summary = _council.print_token_usage_summary
get_storage = _council.get_storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_token_usage(session) -> float:
    """Backward-compatible wrapper around shared council costing."""
    return print_token_usage_summary(session)

    if session.token_usage:
        print("Per-round breakdown:")
        rounds_seen = set()
        for usage in session.token_usage:
            if usage.round_number not in rounds_seen:
                rounds_seen.add(usage.round_number)
                round_tokens = sum(
                    u.total_tokens for u in session.token_usage
                    if u.round_number == usage.round_number
                )
                print(f"  Round {usage.round_number}: {round_tokens:,} tokens")
        print()

    return total_cost


def load_council_config() -> CouncilConfig:
    """Load council configuration from TOML file."""
    import tomllib
    config_path = Path(__file__).parent.parent / "council_config.toml"

    if not config_path.exists():
        raise FileNotFoundError(f"Council config not found: {config_path}")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    return CouncilConfig.from_dict(data.get("council", {}))


def build_sqlite_evidence() -> Evidence:
    """Build the evidence package for SQLite-first architecture decision."""

    evidence = Evidence(
        target="sqlite_first_architecture",
        code_changes=[
            # Current state of implementation
            "CURRENT STATE:",
            "- Sync layer already implemented in kestrel_sovereign/storage/sync/",
            "  - WALListener: Monitors SQLite WAL for changes (wal_listener.py)",
            "  - SyncService: Orchestrates replication to targets (service.py, 373 lines)",
            "  - S3Target, LighthouseTarget: Cloud sync destinations (targets.py)",
            "- PostgreSQL already marked DEPRECATED in storage/db/__init__.py:50-74",
            "- Abstract DatabaseBackend interface exists (~180 lines in interface.py)",
            "- 14 files currently use SQLiteBackend",
            "- Placeholder conversion (? to $1) handled automatically",
            "",
            "ISSUE #2 PROPOSES:",
            "1. Make SQLite the primary (and only) runtime database",
            "2. Deprecate PostgreSQL as a runtime backend entirely",
            "3. Remove the abstract DatabaseBackend interface",
            "4. Use sync layer to replicate SQLite to cloud storage (S3, Lighthouse/Filecoin)",
            "5. 'Two-file sovereign agent' vision: agent.db (SQLite) + emma.llamafile (LLM)",
            "",
            "SYNC LAYER ARCHITECTURE:",
            "```",
            "SQLite (primary) ---> WAL Listener ---> Sync Service ---> Targets",
            "                                                          - S3/R2",
            "                                                          - Lighthouse (Filecoin)",
            "                                                          - PostgreSQL (aggregation only)",
            "```",
        ],
        test_count=765,  # Current test count
        test_passed=765,
        risks=[
            # Technical risks
            "SINGLE-WRITER LIMITATION: SQLite allows only one writer at a time. Issue #2 mentions 'parallel self-improvement cycles' being blocked. WAL mode helps but doesn't eliminate this constraint.",
            "SYNC LAYER UNTESTED: The sync layer (SyncService, WALListener, S3Target) exists but has NO visible integration tests. Production reliability unproven.",
            "CONFLICT RESOLUTION: No documented strategy for handling sync conflicts after offline changes.",
            "MULTI-TENANT BLOCKED: Per-agent SQLite files make centralized admin/analytics harder. PostgreSQL excels at this.",
            "SCHEMA MIGRATIONS: With 1000+ sovereign agents each having own SQLite, coordinating migrations becomes challenging.",
            # Benefits
            "SOVEREIGNTY ALIGNMENT: 'Your data is literally a file you own' - core Kestrel value",
            "DEPLOYMENT SIMPLICITY: No daemon, single file, works offline, runs on mobile/IoT/browser",
            "ABSTRACTION COST: The DatabaseBackend interface is ~180 lines, placeholder conversion automatic - friction may be overstated",
        ],
        architecture_docs=[
            "docs/diagrams/data-architecture/DA-10-sqlite-first-sync.md - Full sync architecture slides",
            "docs/diagrams/data-architecture/DA-02-database-abstraction.md - Current abstraction layer",
            "kestrel_sovereign/storage/sync/__init__.py - Sync layer module (implemented)",
            "kestrel_sovereign/storage/sync/service.py - SyncService implementation (373 lines)",
            "kestrel_sovereign/storage/sync/wal_listener.py - WAL monitoring",
            "kestrel_sovereign/storage/sync/targets.py - S3Target, LighthouseTarget",
            "kestrel_sovereign/storage/db/__init__.py - Factory with deprecation warnings",
            "kestrel_sovereign/storage/db/postgres.py - PostgresBackend (already deprecated)",
        ],
        previous_decisions=[
            "No prior council decision on this topic",
            "Issue #2 created based on feedback during Emma 22/7 scheduler development",
            "PostgreSQL deprecation warnings already added to codebase proactively",
        ],
    )

    return evidence


# The detailed question with full context
SQLITE_QUESTION = """
Should Kestrel fully commit to the SQLite-first architecture proposed in Issue #2?

## The Proposal (Issue #2: SQLite-First with Sync Layer)

1. Make SQLite the ONLY runtime database (not just default)
2. Remove the abstract DatabaseBackend interface entirely
3. Deprecate and eventually remove PostgresBackend
4. Use the sync layer for cloud backup instead of PostgreSQL

## Arguments FOR SQLite-First

**Sovereignty Alignment:**
- "Your data is literally a file you own" - core Kestrel mission
- Two-file sovereign agent vision: agent.db + emma.llamafile = complete portable agent
- No dependency on external database daemon

**Simplicity:**
- Remove ~180 lines of abstract interface
- No placeholder conversion (? vs $1)
- Single codebase, single SQL dialect

**Portability:**
- SQLite runs everywhere: desktop, mobile, IoT, browser (WASM)
- PostgreSQL requires daemon, doesn't run on mobile/embedded
- Offline-first by default

**Sync Layer Exists:**
- WALListener monitors changes
- SyncService orchestrates replication
- S3Target, LighthouseTarget already implemented
- PostgreSQL can be aggregation-only target

## Arguments AGAINST (or for keeping PostgreSQL)

**Single-Writer Limitation:**
- SQLite allows one writer at a time
- Issue mentions "parallel self-improvement cycles" being blocked
- WAL mode helps but doesn't eliminate the constraint

**Multi-Tenant/SaaS:**
- Some deployments need shared database (multiple agents, one PostgreSQL)
- Per-agent SQLite files harder for central admin/analytics
- PostgreSQL connection pooling handles concurrent users well

**Sync Layer Untested:**
- The sync layer code exists but has NO integration tests
- Production reliability is unproven
- What happens during conflicts? Partial syncs? Recovery?

**Abstraction Cost is Low:**
- ~180 lines for DatabaseBackend interface
- Placeholder conversion is automatic
- Maintaining both costs little, provides flexibility

**Don't Burn Bridges:**
- Decision is mostly reversible, but customer migrations are not
- PostgreSQL is proven infrastructure with decades of reliability

## Current State

- Sync layer: IMPLEMENTED (but untested)
- PostgreSQL: Already DEPRECATED with warnings in code
- Abstract interface: EXISTS and works
- Tests: 765 passing

## Questions for Council

1. Should we FULLY commit to SQLite-first (remove PostgreSQL)?
2. Or should we make SQLite DEFAULT while keeping PostgreSQL as option?
3. What validation should happen before removing PostgreSQL?
4. Does the sync layer need testing first?

Please evaluate this architectural decision considering Kestrel's sovereignty mission,
technical tradeoffs, and operational risks.
"""


async def run_council_session():
    """Run the Constitutional Council session for SQLite-first decision."""

    print("=" * 70)
    print("CONSTITUTIONAL COUNCIL SESSION: SQLITE-FIRST ARCHITECTURE")
    print("=" * 70)
    print()
    print("Issue #2: Architecture: SQLite-First with Sync Layer")
    print()

    # Load config
    print("Loading council configuration...")
    config = load_council_config()
    print(f"  Members: {len(config.members)}")
    for member in config.members:
        print(f"    - {member.name} ({member.provider}/{member.model}) as {member.role}")
    print(f"  Consensus Rule: {config.consensus_rule.value}")
    print(f"  Max Rounds: {config.max_rounds}")
    print()

    # Check API keys
    print("Checking API key availability...")
    api_keys = {
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
    }
    for key, available in api_keys.items():
        status = "+" if available else "-"
        print(f"  {key}: {status}")
    print()

    # Build evidence
    print("Compiling evidence package...")
    evidence = build_sqlite_evidence()
    print(f"  Target: {evidence.target}")
    print(f"  Code changes/context: {len(evidence.code_changes)} items")
    print(f"  Risks identified: {len(evidence.risks)}")
    print(f"  Architecture docs: {len(evidence.architecture_docs)}")
    print()

    print("Question for council:")
    print("-" * 50)
    # Print first 1000 chars
    print(SQLITE_QUESTION[:1000] + "..." if len(SQLITE_QUESTION) > 1000 else SQLITE_QUESTION)
    print("-" * 50)
    print()

    # Convene
    print("Convening council session...")
    print("(This may take 3-8 minutes as each model deliberates across multiple rounds)")
    print()

    try:
        session = await convene_council(
            question=SQLITE_QUESTION,
            evidence=evidence,
            members=config.members,
            max_rounds=config.max_rounds,
            consensus_rule=config.consensus_rule,
        )

        # Save session
        storage = get_storage()
        await storage.save_session(session)

        # Print results
        print()
        print("=" * 70)
        print("COUNCIL SESSION RESULTS")
        print("=" * 70)
        print()
        print(f"Session ID: {session.id}")
        print(f"Outcome: {session.outcome.value}")
        print(f"Rounds: {len(session.rounds)}")
        print(f"Verdicts: {len(session.verdicts)}")
        print()

        # Print each verdict
        for verdict in session.verdicts:
            emoji = {"APPROVE": "+", "REJECT": "-", "ABSTAIN": "o"}.get(verdict.decision.value, "?")
            print(f"{emoji} {verdict.member_name}: {verdict.decision.value} ({verdict.confidence:.0%} confidence)")
            print(f"   Reasoning: {verdict.reasoning[:300]}..." if len(verdict.reasoning) > 300 else f"   Reasoning: {verdict.reasoning}")
            if verdict.concerns:
                print(f"   Concerns: {len(verdict.concerns)} items")
                for concern in verdict.concerns[:3]:
                    print(f"     - {concern[:100]}...")
            if verdict.conditions:
                print(f"   Conditions: {len(verdict.conditions)} items")
                for condition in verdict.conditions[:3]:
                    print(f"     - {condition[:100]}...")
            print()

        print(f"Full transcript saved to: data/council_sessions/{session.id}.md")

        # Print token usage and costs
        total_cost = print_token_usage(session)
        print(f"Council session cost: ${total_cost:.4f}")
        print()

        return session

    except Exception as e:
        logger.error(f"Council session failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    session = asyncio.run(run_council_session())

    # Exit with appropriate code based on outcome
    if session.outcome.value == "APPROVED":
        print("SQLITE-FIRST ARCHITECTURE APPROVED BY COUNCIL")
        print("Recommendation: Proceed with full SQLite-first implementation")
        sys.exit(0)
    elif session.outcome.value == "REJECTED":
        print("SQLITE-FIRST ARCHITECTURE REJECTED BY COUNCIL")
        print("Recommendation: Keep PostgreSQL as supported option")
        sys.exit(1)
    else:
        print("COUNCIL REACHED DEADLOCK")
        print("Recommendation: Further discussion needed")
        sys.exit(2)
