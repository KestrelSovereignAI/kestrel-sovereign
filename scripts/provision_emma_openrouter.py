#!/usr/bin/env python
"""
Provision Emma's OpenRouter API Key.

This script provisions an OpenRouter API key for Emma (the genesis agent)
and stores it encrypted in her sovereign database.

Usage:
    python scripts/provision_emma_openrouter.py

Environment Variables Required:
    OPENROUTER_MANAGEMENT_API_KEY - Management key for provisioning new keys
    KESTREL_DATA_KEY - Encryption key for secure storage

The script will:
1. Find Emma's database in ./agent_data/kestrel_prime.db
2. Extract Emma's DID from the database
3. Create an OpenRouter API key with $100/month limit
4. Store the key encrypted with Emma's agent-derived encryption key
5. Update Emma's metadata with the key hash
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()


async def provision_emma_key() -> None:
    """Provision Emma's OpenRouter API key."""
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    # Check environment variables
    if not os.getenv("OPENROUTER_MANAGEMENT_API_KEY"):
        print("Error: OPENROUTER_MANAGEMENT_API_KEY not set")
        sys.exit(1)

    if not os.getenv("KESTREL_DATA_KEY"):
        print("Error: KESTREL_DATA_KEY not set")
        sys.exit(1)

    # Find Emma's database
    db_path = Path(os.getenv("KESTREL_DB_PATH", "./agent_data")) / "kestrel_prime.db"
    if not db_path.exists():
        print(f"Error: Emma's database not found at {db_path}")
        print("Run inception_service.py first to create Emma")
        sys.exit(1)

    print(f"Found Emma's database at {db_path}")

    # Open database
    db = await AsyncDatabase.sqlite(str(db_path))

    try:
        # Get Emma's DID from agent_metadata or graph_nodes
        emma_did = None

        # Try graph_nodes first (for agent nodes)
        result = await db.fetchone(
            "SELECT node_id FROM graph_nodes WHERE node_type = 'agent' LIMIT 1"
        )
        if result:
            emma_did = result[0]
        else:
            # Try agent_metadata
            result = await db.fetchone(
                "SELECT value FROM agent_metadata WHERE key = 'agent_did' LIMIT 1"
            )
            if result:
                emma_did = result[0]

        if not emma_did:
            print("Error: Could not find Emma's DID in database")
            sys.exit(1)

        print(f"Emma's DID: {emma_did}")

        # Check if Emma already has an OpenRouter key
        key_storage = ServiceKeyStorage(db, emma_did)
        has_key = await key_storage.has_key(provider_id="openrouter")

        if has_key:
            print("Emma already has an OpenRouter key configured")
            # Get key info
            keys = await key_storage.list_keys()
            for key in keys:
                if key.provider_id == "openrouter":
                    print(f"  - ID: {key.id}")
                    print(f"  - Active: {key.is_active}")
                    print(f"  - Quota: {key.quota_used}/{key.quota_limit or 'unlimited'}")
                    print(f"  - Created: {key.created_at}")
            return

        # Provision new OpenRouter key
        print("\nProvisioning new OpenRouter key for Emma...")
        provisioning = OpenRouterProvisioningService()

        try:
            key_info = await provisioning.create_agent_key(
                agent_name="emma-genesis",
                limit_usd=100.0,  # $100/month
                limit_reset="monthly",
            )

            print(f"Key created successfully!")
            print(f"  - Key hash: {key_info.key_hash}")
            print(f"  - Limit: ${key_info.limit_usd:.2f}/month")

            # Store the key in Emma's database
            print("\nStoring key with agent-derived encryption...")
            key_id = await key_storage.store_key(
                provider_id="openrouter",
                api_key=key_info.key,
                quota_limit=10000,  # 10,000 units (internal quota tracking)
            )

            print(f"Key stored successfully!")
            print(f"  - Internal key ID: {key_id}")

            # Update agent metadata with key hash
            await db.execute(
                """
                INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                VALUES (?, 'openrouter_key_hash', ?, CURRENT_TIMESTAMP)
                """,
                (emma_did, key_info.key_hash)
            )
            print(f"  - Metadata updated with key hash")

            # Verify key can be retrieved
            retrieved_key = await key_storage.get_key(provider_id="openrouter")
            if retrieved_key == key_info.key:
                print("\n✓ Verification passed: Key stored and retrieved successfully")
            else:
                print("\n✗ Verification failed: Retrieved key doesn't match")

            print("\n" + "=" * 60)
            print("Emma is now configured with her own OpenRouter API key!")
            print("=" * 60)

        finally:
            await provisioning.close()

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(provision_emma_key())
