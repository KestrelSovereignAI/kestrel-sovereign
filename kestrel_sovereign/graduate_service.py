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

Fleet-restart caveat
--------------------
The ``is_test_instance`` flag is read once at ``KestrelAgent`` boot and cached
on ``self._is_test_instance`` (see ``kestrel_agent.py``). Flipping the flag in
SQLite while the multi-agent fleet is running does **not** update the in-memory
copy. No runtime code path consults the flag after boot, so this is not a
correctness issue — but to see the change reflected on the live agent, restart
the fleet process (uvicorn ``kestrel_sovereign.server:app``) once graduation
has been recorded. A future ticket will promote this to a fleet admin endpoint
that reloads the agent in-place; until then, restart is the cleanest signal.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.async_graph_store import GraphNode

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
            status = "[PASS]" if check["passed"] else "[FAIL]"
            print(f"  {status} {check['name']}")
            if check["details"]:
                print(f"        {check['details']}")
        print("=" * 60)
        if self.all_passed:
            print(f"Result: ALL {len(self.passed)} CHECKS PASSED")
        else:
            print(f"Result: {len(self.failed)} FAILED, {len(self.passed)} passed")
        print("=" * 60)


def _resolve_did(agent_node) -> str:
    """Return the agent's DID, or an empty string if the node does not
    actually carry one.

    Canonical layout: the agent's ``node_id`` *is* the DID
    (kestrel_agent.py:530 uses ``AsyncStorage(path, agent_id=self.did)`` and
    inception writes the agent graph node with ``node_id=did``). Some agents
    additionally carry a ``properties['did']`` field; prefer that when set,
    otherwise fall back to ``node_id``. The validator originally only
    consulted ``properties['did']`` and so failed three gates on Emma's live
    DB even though her DID is right there on the node_id.

    The fallback is gated on ``node_id`` actually looking like a DID
    (``"did:"`` prefix) — otherwise a legacy non-DID node_id such as
    ``"agent:test-emma"`` would be treated as a DID, allowing the on-disk and
    tenant-scoped gates to be satisfied by arbitrary files / rows named after
    that non-DID ID. Better to fail the gate honestly than launder a non-DID
    through it. Codex caught this on PR #1325 review round 2.
    """
    candidate = agent_node.properties.get("did") or agent_node.node_id or ""
    return candidate if candidate.startswith("did:") else ""


async def validate_agent(storage: Storage, agent_id: str) -> ValidationChecklist:
    """Run validation checks on the agent."""
    checklist = ValidationChecklist()

    # 1. Agent node exists
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

    did = _resolve_did(agent_node)

    # 2. Constitution anchored — agent has an outgoing 'governed_by' edge
    out_edges = await storage.graph.get_edges(agent_id, direction="out")
    constitution_edges = [e for e in out_edges if e.label == "governed_by"]
    has_constitution = len(constitution_edges) > 0
    checklist.add_check(
        "Constitution anchored",
        has_constitution,
        f"{len(constitution_edges)} governance edge(s)" if has_constitution
        else "No 'governed_by' edge from agent"
    )

    # 3. Conversation history (agent has been used). Live agents write rows
    # under ``agent_id = self.did`` (per-tenant isolation in
    # ``AsyncConversationStore``), but this script opens ``Storage`` without an
    # agent_id, so ``storage.get_conversation_history()`` would only see rows
    # tagged with the empty-string default tenant. Query the table directly,
    # filtered by the agent's DID — that's the tenant the agent uses at boot
    # (kestrel_agent.py:530 instantiates ``AsyncStorage(path, agent_id=self.did)``).
    try:
        row = await storage.db.fetchone(
            "SELECT COUNT(*) FROM conversation_history "
            "WHERE agent_id = ? AND deleted_at IS NULL",
            (did,),
        )
        msg_count = int(row[0]) if row else 0
        checklist.add_check(
            "Has conversation history",
            msg_count > 0,
            f"{msg_count} messages under agent tenant"
            if msg_count > 0 else f"No conversations under agent_id={did!r}",
        )
    except Exception as e:
        checklist.add_check("Has conversation history", False, f"Error: {e}")

    # 4. DID document exists on disk
    address = did.split(":")[-1] if did else None
    db_path = Path(storage.db_path)
    if address:
        did_doc_path = db_path.parent / f"kestrel_{address}.json"
        has_did_doc = did_doc_path.exists()
        checklist.add_check(
            "DID document exists",
            has_did_doc,
            str(did_doc_path) if has_did_doc else f"Missing: {did_doc_path}"
        )
    else:
        checklist.add_check(
            "DID document exists",
            False,
            "Could not parse DID from agent node",
        )

    # 5. Encrypted key file exists on disk
    if address:
        key_path = db_path.parent / f"kestrel_{address}.key.enc"
        has_key = key_path.exists()
        checklist.add_check(
            "Encrypted key file exists",
            has_key,
            str(key_path) if has_key else f"Missing: {key_path}"
        )
    else:
        checklist.add_check("Encrypted key file exists", False, "No DID address available")

    # 6. Sovereignty backup exists — accept either surface:
    #    (a) backup_artifact graph nodes (discrete sovereignty exports via
    #        SovereigntyFeature -> record_backup_artifact), OR
    #    (b) on-disk storage-sync manifests written by the sync targets
    #        (continuous mirroring). Each sync target writes a manifest named
    #        after itself; enumerate every target that can be configured so
    #        sovereign-IPFS-only agents aren't blocked just because they don't
    #        also run GCS / Lighthouse. Sources:
    #        - storage/sync/gcs_target.py            -> .gcs_manifest_<did>.json
    #        - storage/sync/lighthouse_target.py     -> .lighthouse_manifest_<did>.json
    #        - storage/sync/sovereign_ipfs_target.py -> .sovereign_ipfs_manifest_<did>.json
    # Both prove "the agent's sovereign state is recoverable". The validator
    # originally required (a) only — Emma's live DB has 5 months of continuous
    # mirroring via (b) but zero discrete exports, so the strict graph-only
    # check blocked an agent whose sovereignty was demonstrably backed up.
    backup_nodes = await storage.graph.get_nodes_by_type("backup_artifact")
    manifest_filenames = [
        f".gcs_manifest_{did}.json",
        f".lighthouse_manifest_{did}.json",
        f".sovereign_ipfs_manifest_{did}.json",
    ]
    manifests_present = [
        n for n in manifest_filenames if (db_path.parent / n).exists()
    ]
    has_backup = bool(backup_nodes) or bool(manifests_present)
    if has_backup:
        parts = []
        if backup_nodes:
            parts.append(f"{len(backup_nodes)} backup_artifact node(s)")
        if manifests_present:
            parts.append(f"sync manifest(s): {', '.join(manifests_present)}")
        details = "; ".join(parts)
    else:
        details = "no backup_artifact nodes; no sync manifests on disk"
    checklist.add_check("Has sovereignty backup", has_backup, details)

    # 7. Knowledge graph populated — at least agent + 2 other nodes (any type)
    total_row = await storage.db.fetchone("SELECT COUNT(*) FROM graph_nodes")
    total_nodes = int(total_row[0]) if total_row else 0
    checklist.add_check(
        "Knowledge graph populated",
        total_nodes >= 3,
        f"{total_nodes} nodes"
    )

    return checklist


async def graduate_agent(
    db_path: str,
    dry_run: bool = False,
) -> bool:
    """
    Graduate a test agent to permanent status.

    Args:
        db_path: Path to the agent's database
        dry_run: If True, only validate without making changes

    Returns:
        True if graduation succeeded (or dry-run passed)
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise GraduationError(f"Database not found: {db_path}")

    async with Storage(db_path=str(db_path)) as storage:
        agents = await storage.graph.get_nodes_by_type("agent")
        if not agents:
            raise GraduationError("No agent found in database")

        agent_node = agents[0]
        agent_id = agent_node.node_id
        agent_name = agent_node.properties.get("name", "Unknown")

        print(f"\n{'=' * 60}")
        print("GRADUATION SERVICE")
        print(f"{'=' * 60}")
        print(f"Agent: {agent_name}")
        print(f"ID: {agent_id}")
        print(f"Database: {db_path}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

        checklist = await validate_agent(storage, agent_id)
        checklist.print_report()

        if not checklist.all_passed:
            print("\nGRADUATION BLOCKED - Validation failed")
            print("   Fix the failed checks before graduating.")
            return False

        if dry_run:
            print("\nDRY RUN - No changes made")
            print("   Run without --dry-run to graduate.")
            return True

        print("\n" + "=" * 60)
        print("PERFORMING GRADUATION")
        print("=" * 60)

        now_iso = datetime.now(timezone.utc).isoformat()
        timestamp_suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        # Upsert agent with new properties — add_node performs INSERT OR REPLACE
        # / ON CONFLICT UPDATE based on backend, so this updates in place.
        updated_agent = GraphNode(
            node_id=agent_node.node_id,
            node_type=agent_node.node_type,
            label=agent_node.label,
            properties={
                **agent_node.properties,
                "is_test_instance": False,
                "graduated_at": now_iso,
            },
        )
        await storage.graph.add_node(updated_agent)

        # Record the graduation as a lifecycle_event node
        graduation_node_id = f"graduation:{agent_id}:{timestamp_suffix}"
        graduation_event = GraphNode(
            node_id=graduation_node_id,
            node_type="lifecycle_event",
            label=f"Graduation of {agent_name}",
            properties={
                "event_type": "graduation",
                "agent_id": agent_id,
                "agent_name": agent_name,
                "timestamp": now_iso,
                "validation_passed": [c["name"] for c in checklist.checks if c["passed"]],
            },
        )
        await storage.graph.add_node(graduation_event)

        # Link agent -> graduation event
        await storage.graph.add_edge(
            source_id=agent_id,
            target_id=graduation_node_id,
            label="lifecycle_event",
        )

        print("  [OK] Removed test instance flag")
        print("  [OK] Recorded graduation timestamp")
        print("  [OK] Created graduation event node")
        print("  [OK] Linked graduation event to agent")

        print("\n" + "=" * 60)
        print("GRADUATION COMPLETE")
        print(f"   {agent_name} is now a PERMANENT agent")
        print("=" * 60)

        print("\n" + "=" * 60)
        print("GRADUATION CEREMONY RECORD")
        print("=" * 60)
        print(f"""
Date: {now_iso}
Agent: {agent_name}
DID: {_resolve_did(agent_node)}
Previous Status: Test Instance
New Status: Permanent Agent

Validation Checks Passed:
{chr(10).join('  - ' + name for name in checklist.passed)}

This agent has been graduated from test status to permanent status.
The retirement service will no longer accept this agent for retirement.

NOTE: If a multi-agent fleet (uvicorn kestrel_sovereign.server:app) is
running and has this agent loaded, restart it so the in-memory
_is_test_instance flag re-reads from disk. No runtime code path consults
the flag after boot, so this is observability, not correctness.
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
    python graduate_service.py /path/to/agent/kestrel_prime.db --dry-run

    # Graduate the agent
    python graduate_service.py /path/to/agent/kestrel_prime.db
        """
    )

    parser.add_argument(
        "db_path",
        help="Path to agent database (kestrel_prime.db)"
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
            dry_run=args.dry_run,
        ))
        sys.exit(0 if success else 1)
    except GraduationError as e:
        logger.error(f"Graduation failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nGraduation cancelled")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
