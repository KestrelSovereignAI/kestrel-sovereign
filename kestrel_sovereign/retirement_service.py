#!/usr/bin/env python3
"""
Retirement Service: Graceful retirement protocol for Kestrel agents.

This service handles the ethical retirement of agents:
1. Verify the agent status (test vs permanent)
2. Create a final memory snapshot (for learning)
3. Add retirement record to the graph
4. Archive (don't delete) agent files
5. Log the retirement ceremony

For permanent agents, --force is required with an explicit reason.
This prevents accidental retirement while preserving human sovereignty.

Philosophy: Even test instances deserve dignity in their ending.
Permanent agents deserve deliberation before ending.
"""
import logging
import json
import shutil
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass
import asyncio

from kestrel_sovereign.storage import AsyncStorage, GraphNode, Edge
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import AsyncGraphStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _resolve_archive_dir(db_path: Path, archive_dir: Optional[str]) -> Path:
    """Resolve where to store retirement archives.

    In the sovereign Docker runtime, the code directory (/app) is read-only.
    Archives must be written to the mounted data volume (typically /data),
    which is the parent directory of the database file.
    """
    if archive_dir is not None:
        return Path(archive_dir)

    # Prefer the mounted data volume when configured.
    kestrel_db_path = Path(os.environ.get("KESTREL_DB_PATH", "")).expanduser()
    if str(kestrel_db_path) and kestrel_db_path.exists():
        return kestrel_db_path / "archive" / "retired_agents"

    # Default: archive alongside the DB in the agent's data directory.
    return db_path.parent / "archive" / "retired_agents"


@dataclass
class RetirementRecord:
    """Record of a test agent's retirement."""
    agent_did: str
    agent_name: str
    test_cycle_id: str
    created_at: str
    retired_at: str
    reason: str
    archive_path: str
    final_conversation_count: int
    final_memory_hash: Optional[str] = None


async def get_agent_info(db_path: str) -> Optional[dict]:
    """Retrieve agent information from the database."""
    db = await AsyncDatabase.sqlite(db_path)

    # Find the agent node
    # Agent nodes have type "agent"
    row = await db.fetchone(
        "SELECT node_id, properties FROM graph_nodes WHERE node_type = 'agent' LIMIT 1"
    )
    await db.close()

    if row:
        properties = json.loads(row[1]) if row[1] else {}
        return {
            "agent_did": row[0],
            **properties
        }
    return None


async def count_conversations(db_path: str) -> int:
    """Count the number of conversations in the database."""
    db = await AsyncDatabase.sqlite(db_path)
    try:
        row = await db.fetchone("SELECT COUNT(*) FROM messages")
        await db.close()
        return row[0] if row else 0
    except Exception:
        # Table may not exist if agent was never used
        await db.close()
        return 0


async def retire_agent(
    db_path: str,
    reason: str = "testing_complete",
    archive_dir: Optional[str] = None,
    force: bool = False
) -> RetirementRecord:
    """
    Gracefully retire an agent.

    Args:
        db_path: Path to the agent's database
        reason: Reason for retirement (e.g., "testing_complete", "validation_passed")
        archive_dir: Directory to archive agent files. Defaults to archive/retired_agents/
        force: Required for permanent agents. Prevents accidental retirement.

    Returns:
        RetirementRecord with details of the retirement

    Raises:
        ValueError: If permanent agent and force=False, or if reason not provided for permanent
        FileNotFoundError: If database doesn't exist
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Get agent info
    agent_info = await get_agent_info(str(db_path))
    if not agent_info:
        raise ValueError(f"No agent found in database: {db_path}")

    # Check if it's a permanent agent
    is_test_instance = agent_info.get("is_test_instance", False)
    if not is_test_instance:
        if not force:
            raise ValueError(
                f"Agent {agent_info.get('agent_did')} is a PERMANENT agent.\n"
                "Retiring a permanent agent requires explicit confirmation.\n"
                "Use --force --reason 'your reason' to proceed.\n\n"
                "Consider: Is there a way to fix the issue instead of retiring?"
            )
        if reason == "testing_complete":
            raise ValueError(
                "Permanent agent retirement requires a specific reason.\n"
                "Use --reason 'detailed explanation' with --force."
            )
        logger.warning("=" * 60)
        logger.warning("RETIRING PERMANENT AGENT")
        logger.warning("=" * 60)
        logger.warning(f"Agent: {agent_info.get('name', 'Unknown')}")
        logger.warning(f"Reason: {reason}")
        logger.warning("This action will archive the agent. It can be restored from backup.")
        logger.warning("=" * 60)

    agent_did = agent_info["agent_did"]
    agent_name = agent_info.get("name", "Unknown")
    test_cycle_id = agent_info.get("test_cycle_id", "unknown")
    created_at = agent_info.get("created_at", "unknown")

    logger.info(f"Beginning retirement ceremony for {agent_name} (cycle: {test_cycle_id})")

    # Count final conversations
    conversation_count = await count_conversations(str(db_path))
    logger.info(f"Agent has {conversation_count} messages in memory")

    # Add retirement record to the graph
    db = await AsyncDatabase.sqlite(str(db_path))
    graph = AsyncGraphStore(db)

    retired_at = datetime.now(timezone.utc).isoformat()

    retirement_node = GraphNode(
        node_id=f"retirement_{test_cycle_id}",
        node_type="retirement_event",
        label=f"Retirement of {agent_name}",
        properties={
            "agent_did": agent_did,
            "agent_name": agent_name,
            "test_cycle_id": test_cycle_id,
            "reason": reason,
            "retired_at": retired_at,
            "conversation_count": conversation_count,
            "ceremony_message": (
                f"Thank you, {agent_name}, for your service as a test instance. "
                f"Your {conversation_count} interactions helped validate the Kestrel framework. "
                f"Your contributions will improve future agents."
            )
        }
    )
    await graph.add_node(retirement_node)
    await graph.add_edge(agent_did, retirement_node.node_id, "retired_via")

    await db.close()

    # Revoke OpenRouter API key if the agent had one provisioned
    openrouter_key_hash = agent_info.get("openrouter_key_hash")
    if openrouter_key_hash:
        try:
            from kestrel_sovereign.features.llm_keys.openrouter_provisioning import delete_agent_key
            await delete_agent_key(openrouter_key_hash)
            logger.info(f"Revoked OpenRouter key (hash: {openrouter_key_hash[:16]}...)")
        except Exception as e:
            logger.warning(f"Could not revoke OpenRouter key: {e}")

    # Archive agent files (don't delete)
    archive_dir = _resolve_archive_dir(db_path, archive_dir)

    archive_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped archive folder
    archive_name = f"{agent_name}_{test_cycle_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    agent_archive_path = archive_dir / archive_name
    agent_archive_path.mkdir(exist_ok=True)

    # Move database to archive
    archived_db = agent_archive_path / db_path.name
    shutil.move(str(db_path), str(archived_db))
    logger.info(f"Archived database to {archived_db}")

    # Move related files (keys, DID document)
    db_dir = db_path.parent
    for pattern in ["*.pem", "*.key.enc", "*kestrel*.json"]:
        for file in db_dir.glob(pattern):
            if file.exists():
                shutil.move(str(file), str(agent_archive_path / file.name))
                logger.info(f"Archived {file.name}")

    # Create retirement summary
    record = RetirementRecord(
        agent_did=agent_did,
        agent_name=agent_name,
        test_cycle_id=test_cycle_id,
        created_at=created_at,
        retired_at=retired_at,
        reason=reason,
        archive_path=str(agent_archive_path),
        final_conversation_count=conversation_count
    )

    # Write retirement record to archive
    record_path = agent_archive_path / "RETIREMENT_RECORD.json"
    with open(record_path, 'w') as f:
        json.dump({
            "agent_did": record.agent_did,
            "agent_name": record.agent_name,
            "test_cycle_id": record.test_cycle_id,
            "created_at": record.created_at,
            "retired_at": record.retired_at,
            "reason": record.reason,
            "final_conversation_count": record.final_conversation_count,
            "ceremony_message": (
                f"This archive contains the final state of {agent_name}, "
                f"a test instance that served honorably during test cycle {test_cycle_id}. "
                f"Retired on {retired_at}."
            )
        }, f, indent=2)

    # Log the ceremony
    logger.info("=" * 60)
    logger.info("RETIREMENT CEREMONY COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Agent: {agent_name}")
    logger.info(f"DID: {agent_did}")
    logger.info(f"Test Cycle: {test_cycle_id}")
    logger.info(f"Service Period: {created_at} to {retired_at}")
    logger.info(f"Conversations Preserved: {conversation_count}")
    logger.info(f"Archive Location: {agent_archive_path}")
    logger.info("")
    logger.info(f"Thank you, {agent_name}. Your service mattered.")
    logger.info("=" * 60)

    return record


async def list_retired_agents(archive_dir: Optional[str] = None) -> list[dict]:
    """List all retired test agents."""
    # Without a DB path, default to the configured data directory if available.
    if archive_dir is None:
        kestrel_db_path = Path(os.environ.get("KESTREL_DB_PATH", "")).expanduser()
        if str(kestrel_db_path) and kestrel_db_path.exists():
            archive_dir = kestrel_db_path / "archive" / "retired_agents"
        else:
            archive_dir = Path.cwd() / "archive" / "retired_agents"
    else:
        archive_dir = Path(archive_dir)

    if not archive_dir.exists():
        return []

    retired = []
    for agent_dir in archive_dir.iterdir():
        if agent_dir.is_dir():
            record_path = agent_dir / "RETIREMENT_RECORD.json"
            if record_path.exists():
                with open(record_path) as f:
                    retired.append(json.load(f))

    return retired


def retire_agent_sync(
    db_path: str,
    reason: str = "testing_complete",
    force: bool = False
) -> RetirementRecord:
    """Sync wrapper for retire_agent."""
    return asyncio.run(retire_agent(db_path, reason, force=force))


# Backwards compatibility alias
async def retire_test_agent(
    db_path: str,
    reason: str = "testing_complete",
    archive_dir: Optional[str] = None
) -> RetirementRecord:
    """Deprecated: Use retire_agent instead."""
    return await retire_agent(db_path, reason, archive_dir, force=False)


def main():
    """CLI for retirement service."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Retire a Kestrel agent gracefully.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Retire a test agent
    python retirement_service.py ~/test_agent/kestrel_prime.db

    # Retire with specific reason
    python retirement_service.py ~/agent/kestrel_prime.db --reason "validation_complete"

    # Retire a PERMANENT agent (requires --force)
    python retirement_service.py ~/emma/kestrel_prime.db \\
        --force --reason "Critical security vulnerability discovered"

    # List all retired agents
    python retirement_service.py --list
        """
    )
    parser.add_argument("db_path", nargs="?", help="Path to agent database")
    parser.add_argument("--reason", default="testing_complete",
                        help="Reason for retirement")
    parser.add_argument("--force", action="store_true",
                        help="Required for permanent agents. Confirms deliberate action.")
    parser.add_argument("--list", action="store_true",
                        help="List all retired agents")

    args = parser.parse_args()

    if args.list:
        retired = asyncio.run(list_retired_agents())
        if not retired:
            print("No retired agents found.")
        else:
            print(f"\nRetired Agents ({len(retired)} total):\n")
            for agent in retired:
                print(f"  - {agent['agent_name']} ({agent['test_cycle_id']})")
                print(f"    Retired: {agent['retired_at']}")
                print(f"    Reason: {agent['reason']}")
                print()
    elif not args.db_path:
        parser.print_help()
        exit(1)
    else:
        try:
            record = retire_agent_sync(args.db_path, args.reason, force=args.force)
            print(f"\nRetirement complete. Archive: {record.archive_path}")
        except ValueError as e:
            print(f"\nError: {e}")
            exit(1)
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            exit(1)


if __name__ == "__main__":
    main()
