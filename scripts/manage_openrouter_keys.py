#!/usr/bin/env python3
"""
OpenRouter Key Management CLI.

List, provision, and delete OpenRouter API keys for Kestrel agents.

Usage:
    python scripts/manage_openrouter_keys.py list
    python scripts/manage_openrouter_keys.py provision --agent emma --limit 10.0
    python scripts/manage_openrouter_keys.py delete --hash abc123...
    python scripts/manage_openrouter_keys.py prune  # delete all unused keys

Environment:
    OPENROUTER_MANAGEMENT_API_KEY - Required for all operations
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


async def cmd_list(args):
    """List all OpenRouter keys."""
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    svc = OpenRouterProvisioningService()
    try:
        keys = await svc.list_keys()
    finally:
        await svc.close()

    if not keys:
        print("No keys found.")
        return

    print(f"\n{'Name':<45} {'Limit':>8} {'Used':>8} {'Created':<20} {'Hash (12)'}")
    print("-" * 105)
    for k in keys:
        name = k.get("name", "unnamed")
        limit = k.get("limit", "?")
        usage = k.get("usage", 0)
        created = k.get("created_at", "?")[:19]
        h = k.get("hash", "?")[:12]
        print(f"{name:<45} ${limit:>7} ${usage:>7} {created:<20} {h}")

    print(f"\nTotal: {len(keys)} keys")


async def cmd_provision(args):
    """Provision a new key for an agent."""
    from kestrel_sovereign.features.llm_keys.openrouter_provisioning import provision_agent_key

    print(f"Provisioning key for '{args.agent}' with ${args.limit:.2f}/{args.reset} limit...")
    key_info = await provision_agent_key(
        agent_name=args.agent,
        limit_usd=args.limit,
        limit_reset=args.reset,
    )

    print(f"Key created:")
    print(f"  Hash:  {key_info.key_hash}")
    print(f"  Limit: ${key_info.limit_usd:.2f}/{key_info.limit_reset}")
    print(f"\nStore the key securely - it won't be shown again.")
    print(f"Key: {key_info.key}")


async def cmd_delete(args):
    """Delete a specific key by hash."""
    from kestrel_sovereign.features.llm_keys.openrouter_provisioning import delete_agent_key

    print(f"Deleting key {args.hash[:16]}...")
    result = await delete_agent_key(args.hash)
    if result:
        print("Deleted.")
    else:
        print("Key not found or already deleted.")


async def cmd_prune(args):
    """Delete all agent keys (keeps management key)."""
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    svc = OpenRouterProvisioningService()
    try:
        total_deleted = 0
        while True:
            keys = await svc.list_keys()
            if not keys:
                break

            print(f"Deleting batch of {len(keys)} keys...")
            for k in keys:
                try:
                    await svc.delete_key(k["hash"])
                    total_deleted += 1
                    await asyncio.sleep(0.8)
                except Exception:
                    await asyncio.sleep(3.0)
                    try:
                        await svc.delete_key(k["hash"])
                        total_deleted += 1
                    except Exception as e:
                        print(f"  Failed: {k['hash'][:12]} - {e}")

            await asyncio.sleep(2.0)
    finally:
        await svc.close()

    print(f"\nDeleted {total_deleted} keys.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage OpenRouter API keys for Kestrel agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all provisioned keys")

    prov = sub.add_parser("provision", help="Provision a new key for an agent")
    prov.add_argument("--agent", required=True, help="Agent name")
    prov.add_argument("--limit", type=float, default=0.10, help="Spending limit in USD (default: $0.10)")
    prov.add_argument("--reset", default="monthly", choices=["daily", "weekly", "monthly"],
                       help="Limit reset interval (default: monthly)")

    delete = sub.add_parser("delete", help="Delete a key by hash")
    delete.add_argument("--hash", required=True, help="Key hash to delete")

    sub.add_parser("prune", help="Delete ALL provisioned agent keys")

    args = parser.parse_args()

    if not os.getenv("OPENROUTER_MANAGEMENT_API_KEY"):
        print("Error: OPENROUTER_MANAGEMENT_API_KEY not set")
        sys.exit(1)

    cmd_map = {
        "list": cmd_list,
        "provision": cmd_provision,
        "delete": cmd_delete,
        "prune": cmd_prune,
    }
    asyncio.run(cmd_map[args.command](args))


if __name__ == "__main__":
    main()
