#!/usr/bin/env python
"""
Provision an OpenRouter API key for an agent and persist it.

Distinct from `manage_openrouter_keys.py provision`:
  - That tool creates an OpenRouter key and prints it.
  - This tool ALSO stores the key in the agent's ServiceKeyStorage
    (encrypted with the agent's derived key) and updates the agent's
    `openrouter_key_hash` metadata. After this runs, key resolution
    finds the agent's own key on subsequent calls.

Usage:
    python scripts/provision_agent_openrouter.py --db agent_data/<agent>.db \\
        [--limit-usd 100.0] [--limit-reset monthly] [--name <agent-label>]

Environment:
    OPENROUTER_MANAGEMENT_API_KEY  Required for OpenRouter provisioning.
    KESTREL_DATA_KEY               Required for ServiceKeyStorage encryption.

The agent's DID is read from the supplied database (graph_nodes
agent row, or agent_metadata key=agent_did fallback).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()


async def provision_agent_key(
    db_path: Path,
    agent_label: str,
    limit_usd: float,
    limit_reset: str,
) -> int:
    """Provision and persist an OpenRouter key for the agent at db_path.

    Returns process exit code.
    """
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    if not os.getenv("OPENROUTER_MANAGEMENT_API_KEY"):
        print("Error: OPENROUTER_MANAGEMENT_API_KEY not set", file=sys.stderr)
        return 1

    if not os.getenv("KESTREL_DATA_KEY"):
        print("Error: KESTREL_DATA_KEY not set", file=sys.stderr)
        return 1

    if not db_path.exists():
        print(f"Error: agent database not found at {db_path}", file=sys.stderr)
        return 1

    print(f"Agent database: {db_path}")

    db = await AsyncDatabase.sqlite(str(db_path))

    try:
        agent_did = None
        agent_node_id = None
        agent_properties_json = None

        # Prefer graph_nodes since that's the canonical location runtime
        # and retirement service read from. Fall back to agent_metadata for
        # exotic states.
        result = await db.fetchone(
            "SELECT node_id, properties FROM graph_nodes WHERE node_type = 'agent' LIMIT 1"
        )
        if result:
            agent_did = result[0]
            agent_node_id = result[0]
            agent_properties_json = result[1]
        else:
            result = await db.fetchone(
                "SELECT value FROM agent_metadata WHERE key = 'agent_did' LIMIT 1"
            )
            if result:
                agent_did = result[0]

        if not agent_did:
            print("Error: could not find agent DID in database", file=sys.stderr)
            return 1

        print(f"Agent DID: {agent_did}")

        key_storage = ServiceKeyStorage(db, agent_did)
        if await key_storage.has_key(provider_id="openrouter"):
            print("Agent already has an OpenRouter key configured")
            for key in await key_storage.list_keys():
                if key.provider_id == "openrouter":
                    print(f"  - ID: {key.id}")
                    print(f"  - Active: {key.is_active}")
                    print(f"  - Quota: {key.quota_used}/{key.quota_limit or 'unlimited'}")
                    print(f"  - Created: {key.created_at}")

            # Backfill graph_nodes.properties.openrouter_key_hash if it's
            # missing. Agents provisioned by the original (buggy) script wrote
            # the hash only to agent_metadata; runtime startup and retirement
            # both read from graph_nodes.properties, so without this backfill
            # they remain on the shared key path and retirement skips revocation.
            if agent_node_id is not None:
                properties = json.loads(agent_properties_json) if agent_properties_json else {}
                if not properties.get("openrouter_key_hash"):
                    legacy_row = await db.fetchone(
                        "SELECT value FROM agent_metadata WHERE key = 'openrouter_key_hash' LIMIT 1"
                    )
                    legacy_hash = legacy_row[0] if legacy_row else None
                    if legacy_hash:
                        properties["openrouter_key_hash"] = legacy_hash
                        await db.execute(
                            "UPDATE graph_nodes SET properties = ? WHERE node_id = ?",
                            (json.dumps(properties), agent_node_id),
                        )
                        print(
                            f"\nBackfilled graph_nodes.properties.openrouter_key_hash "
                            f"from legacy agent_metadata (hash {legacy_hash[:16]}...)."
                        )
                    else:
                        print(
                            "\nWARNING: graph_nodes.properties.openrouter_key_hash is missing "
                            "and no legacy agent_metadata fallback was found. Runtime startup "
                            "will not activate the stored agent key. To recover, either delete "
                            "the encrypted entry from ServiceKeyStorage and re-run this script, "
                            "or fetch the hash from OpenRouter and write it to graph_nodes "
                            "manually.",
                            file=sys.stderr,
                        )
            return 0

        print(f"\nProvisioning OpenRouter key (label={agent_label}, limit=${limit_usd:.2f}, reset={limit_reset})...")
        provisioning = OpenRouterProvisioningService()

        try:
            key_info = await provisioning.create_agent_key(
                agent_name=agent_label,
                limit_usd=limit_usd,
                limit_reset=limit_reset,
            )

            print("Key created.")
            print(f"  - Key hash: {key_info.key_hash}")
            print(f"  - Limit:    ${key_info.limit_usd:.2f}/month")

            print("\nStoring key with agent-derived encryption...")
            key_id = await key_storage.store_key(
                provider_id="openrouter",
                api_key=key_info.key,
                quota_limit=10000,
            )
            print(f"  - Internal key ID: {key_id}")

            # Update graph_nodes.properties — this is where kestrel_agent.py
            # and retirement_service.py read openrouter_key_hash. Writing
            # only to agent_metadata (as the original Emma script did)
            # left this hash invisible to the runtime; keys provisioned
            # that way silently fell through to the shared key path.
            if agent_node_id is not None:
                properties = json.loads(agent_properties_json) if agent_properties_json else {}
                properties["openrouter_key_hash"] = key_info.key_hash
                await db.execute(
                    "UPDATE graph_nodes SET properties = ? WHERE node_id = ?",
                    (json.dumps(properties), agent_node_id),
                )
                print("  - graph_nodes.properties.openrouter_key_hash updated")
            else:
                # Agent_metadata fallback path. retirement_service merges
                # agent_metadata into the agent_info dict, so writing here
                # at least covers retirement; runtime startup that reads
                # from graph_nodes.properties will not see it.
                await db.execute(
                    """
                    INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                    VALUES (?, 'openrouter_key_hash', ?, CURRENT_TIMESTAMP)
                    """,
                    (agent_did, key_info.key_hash),
                )
                print("  - agent_metadata.openrouter_key_hash updated (fallback)")

            retrieved_key = await key_storage.get_key(provider_id="openrouter")
            if retrieved_key == key_info.key:
                print("\nVerification passed: key stored and retrieved.")
            else:
                print("\nVerification failed: retrieved key did not match.", file=sys.stderr)
                return 1

            print("\n" + "=" * 60)
            print(f"Agent {agent_did} is now configured with its own OpenRouter API key.")
            print("=" * 60)
            return 0

        finally:
            await provisioning.close()

    finally:
        await db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision and persist an OpenRouter API key for an agent",
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the agent's database (e.g. agent_data/<agent>.db)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="OpenRouter key label (defaults to db filename stem)",
    )
    parser.add_argument(
        "--limit-usd",
        type=float,
        default=100.0,
        help="Monthly spending limit in USD (default: 100.0)",
    )
    parser.add_argument(
        "--limit-reset",
        choices=("daily", "weekly", "monthly"),
        default="monthly",
        help="Limit reset interval (default: monthly)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    db_path = Path(args.db)
    agent_label = args.name or db_path.stem
    rc = asyncio.run(
        provision_agent_key(
            db_path=db_path,
            agent_label=agent_label,
            limit_usd=args.limit_usd,
            limit_reset=args.limit_reset,
        )
    )
    sys.exit(rc)
