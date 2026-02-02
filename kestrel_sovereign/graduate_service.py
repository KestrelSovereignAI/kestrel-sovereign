#!/usr/bin/env python3
"""
Graduate Service - Promote a test agent to permanent status.

This service transitions a test agent to permanent status after
validation criteria are met. It's the counterpart to retirement_service.py.

Usage:
    # Check graduation readiness (dry run)
    python graduate_service.py /path/to/agent/kestrel_prime.db --dry-run

    # Graduate the agent
    python graduate_service.py /path/to/agent/kestrel_prime.db

    # Graduate with council session reference
    python graduate_service.py /path/to/agent/kestrel_prime.db \
        --council-session e2ce8a5c-1b5c-4c1b-b798-13aea4a3eef2
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from kestrel_sovereign.storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraduationError(Exception):
    """Raised when graduation cannot proceed."""
    pass


class ValidationChecklist:
    """Validation criteria for graduation."""

    def __init__(self):
        self.checks = []
        self.passed = []
        self.failed = []

    def add_check(self, name: str, passed: bool, details: str = ""):
        self.checks.append({
            "name": name,
            "passed": passed,
            "details": details
        })
        if passed:
            self.passed.append(name)
        else:
            self.failed.append(name)

    @property
    def all_passed(self) -> bool:
        return len(self.failed) == 0

    def print_report(self):
        print("\n" + "=" * 60)
        print("GRADUATION VALIDATION CHECKLIST")
        print("=" * 60)
        for check in self.checks:
            status = "✅" if check["passed"] else "❌"
            print(f"  {status} {check['name']}")
            if check["details"]:
                print(f"      {check['details']}")
        print("=" * 60)
        if self.all_passed:
            print(f"Result: ALL {len(self.passed)} CHECKS PASSED")
        else:
            print(f"Result: {len(self.failed)} FAILED, {len(self.passed)} passed")
        print("=" * 60)


async def validate_agent(storage: Storage, agent_id: str) -> ValidationChecklist:
    """Run validation checks on the agent."""
    checklist = ValidationChecklist()

    # 1. Check agent exists and is a test instance
    agent_node = await storage.graph.get_node(agent_id)
    if not agent_node:
        checklist.add_check("Agent exists", False, f"No agent found with ID: {agent_id}")
        return checklist

    checklist.add_check("Agent exists", True, f"ID: {agent_id}")

    is_test = agent_node.properties.get("is_test_instance", False)
    if not is_test:
        checklist.add_check("Is test instance", False, "Agent is already permanent!")
        return checklist

    checklist.add_check("Is test instance", True)

    # 2. Check constitution is anchored
    constitution_edges = await storage.graph.get_edges(
        from_id=agent_id,
        edge_type="governed_by"
    )
    has_constitution = len(constitution_edges) > 0
    checklist.add_check(
        "Constitution anchored",
        has_constitution,
        f"{len(constitution_edges)} governance edge(s)" if has_constitution else "No constitution link"
    )

    # 3. Check conversation history exists (agent has been used)
    try:
        messages = await storage.conversations.get_conversation_history(limit=10)
        msg_count = len(messages)
        checklist.add_check(
            "Has conversation history",
            msg_count > 0,
            f"{msg_count} recent messages" if msg_count > 0 else "No conversations"
        )
    except Exception as e:
        checklist.add_check("Has conversation history", False, f"Error: {e}")

    # 4. Check DID document exists
    agent_name = agent_node.properties.get("name", "unknown")
    did = agent_node.properties.get("did", "")
    if did:
        # Extract address from DID for filename
        address = did.split(":")[-1] if did else None
        if address:
            db_path = Path(storage.db_path)
            did_doc_path = db_path.parent / f"kestrel_{address}.json"
            has_did_doc = did_doc_path.exists()
            checklist.add_check(
                "DID document exists",
                has_did_doc,
                str(did_doc_path) if has_did_doc else f"Missing: {did_doc_path}"
            )
        else:
            checklist.add_check("DID document exists", False, "Could not parse DID")
    else:
        checklist.add_check("DID document exists", False, "No DID in agent properties")

    # 5. Check encrypted key file exists
    if did:
        address = did.split(":")[-1] if did else None
        if address:
            db_path = Path(storage.db_path)
            key_path = db_path.parent / f"kestrel_{address}.key.enc"
            has_key = key_path.exists()
            checklist.add_check(
                "Encrypted key file exists",
                has_key,
                str(key_path) if has_key else f"Missing: {key_path}"
            )

    # 6. Check backup exists (sovereignty export)
    backup_nodes = await storage.graph.query_nodes(
        node_type="backup_artifact",
        limit=5
    )
    has_backup = len(backup_nodes) > 0
    checklist.add_check(
        "Has sovereignty backup",
        has_backup,
        f"{len(backup_nodes)} backup(s)" if has_backup else "No backups found"
    )

    # 7. Check knowledge graph has content
    all_nodes = await storage.graph.query_nodes(limit=100)
    node_count = len(all_nodes)
    checklist.add_check(
        "Knowledge graph populated",
        node_count >= 3,  # At least agent, constitution, and something else
        f"{node_count} nodes"
    )

    return checklist


async def graduate_agent(
    db_path: str,
    council_session: str | None = None,
    dry_run: bool = False
) -> bool:
    """
    Graduate a test agent to permanent status.

    Args:
        db_path: Path to the agent's database
        council_session: Optional council session ID that approved graduation
        dry_run: If True, only validate without making changes

    Returns:
        True if graduation succeeded
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise GraduationError(f"Database not found: {db_path}")

    # Find agent ID
    async with Storage(db_path=str(db_path)) as storage:
        # Get the agent node
        agents = await storage.graph.query_nodes(node_type="agent", limit=1)
        if not agents:
            raise GraduationError("No agent found in database")

        agent_node = agents[0]
        agent_id = agent_node.id
        agent_name = agent_node.properties.get("name", "Unknown")

        print(f"\n{'=' * 60}")
        print(f"GRADUATION SERVICE")
        print(f"{'=' * 60}")
        print(f"Agent: {agent_name}")
        print(f"ID: {agent_id}")
        print(f"Database: {db_path}")
        if council_session:
            print(f"Council Session: {council_session}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

        # Run validation
        checklist = await validate_agent(storage, agent_id)
        checklist.print_report()

        if not checklist.all_passed:
            print("\n❌ GRADUATION BLOCKED - Validation failed")
            print("   Fix the failed checks before graduating.")
            return False

        if dry_run:
            print("\n🔍 DRY RUN - No changes made")
            print("   Run without --dry-run to graduate.")
            return True

        # Perform graduation
        print("\n" + "=" * 60)
        print("PERFORMING GRADUATION")
        print("=" * 60)

        # Update agent properties
        new_properties = dict(agent_node.properties)
        new_properties["is_test_instance"] = False
        new_properties["graduated_at"] = datetime.utcnow().isoformat()
        if council_session:
            new_properties["graduation_council_session"] = council_session

        # Record the graduation in the graph
        await storage.graph.update_node(
            node_id=agent_id,
            properties=new_properties
        )

        # Create graduation event node
        graduation_node_id = f"graduation:{agent_id}:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        await storage.graph.add_node(
            node_id=graduation_node_id,
            node_type="lifecycle_event",
            properties={
                "event_type": "graduation",
                "agent_id": agent_id,
                "agent_name": agent_name,
                "timestamp": datetime.utcnow().isoformat(),
                "council_session": council_session,
                "validation_passed": [c["name"] for c in checklist.checks if c["passed"]],
            }
        )

        # Link graduation event to agent
        await storage.graph.add_edge(
            from_id=agent_id,
            to_id=graduation_node_id,
            edge_type="lifecycle_event"
        )

        print(f"  ✅ Removed test instance flag")
        print(f"  ✅ Recorded graduation timestamp")
        print(f"  ✅ Created graduation event node")
        if council_session:
            print(f"  ✅ Linked to council session")

        print("\n" + "=" * 60)
        print(f"🎓 GRADUATION COMPLETE")
        print(f"   {agent_name} is now a PERMANENT agent")
        print("=" * 60)

        # Print ceremony record
        print("\n" + "=" * 60)
        print("GRADUATION CEREMONY RECORD")
        print("=" * 60)
        print(f"""
Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
Agent: {agent_name}
DID: {agent_node.properties.get('did', 'N/A')}
Previous Status: Test Instance
New Status: Permanent Agent

Validation Checks Passed:
{chr(10).join('  - ' + name for name in checklist.passed)}

Council Approval: {council_session or 'N/A'}

This agent has been graduated from test status to permanent status.
The retirement service will no longer accept this agent for retirement.
""")
        print("=" * 60)

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Graduate a test agent to permanent status.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check if agent is ready for graduation
    python graduate_service.py ~/emma_data/kestrel_prime.db --dry-run

    # Graduate with council session reference
    python graduate_service.py ~/emma_data/kestrel_prime.db \\
        --council-session e2ce8a5c-1b5c-4c1b-b798-13aea4a3eef2

    # Graduate without council reference (not recommended)
    python graduate_service.py ~/emma_data/kestrel_prime.db
        """
    )

    parser.add_argument(
        "db_path",
        help="Path to agent database (kestrel_prime.db)"
    )
    parser.add_argument(
        "--council-session",
        help="Council session ID that approved graduation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without making changes"
    )

    args = parser.parse_args()

    try:
        success = asyncio.run(graduate_agent(
            db_path=args.db_path,
            council_session=args.council_session,
            dry_run=args.dry_run
        ))
        sys.exit(0 if success else 1)
    except GraduationError as e:
        logger.error(f"Graduation failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Graduation cancelled")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
