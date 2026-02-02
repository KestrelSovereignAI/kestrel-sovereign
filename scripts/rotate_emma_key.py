#!/usr/bin/env python3
"""
Rotate Emma's encryption key from temp passphrase to a secure passphrase.

Usage:
    python scripts/rotate_emma_key.py --old-pass "THIS IS A TEMP KEY FOR TESTING" --new-pass "your-secure-passphrase"
"""
import asyncio
import argparse
import hashlib
import base64
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def passphrase_to_master_key(passphrase: str) -> bytes:
    """Convert a passphrase to a master key via SHA-256."""
    digest = hashlib.sha256(passphrase.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(digest)


def derive_agent_fernet(master_key: bytes, agent_id: str) -> Fernet:
    """Derive per-agent Fernet key using HKDF."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=agent_id.encode('utf-8'),
        info=b"kestrel-agent-v1"
    )
    derived = hkdf.derive(master_key)
    return Fernet(base64.urlsafe_b64encode(derived))


async def rotate_conversation_keys(db_path: str, old_pass: str, new_pass: str) -> dict:
    """
    Rotate encryption keys for all encrypted conversations.

    Uses per-agent HKDF-derived keys (the actual encryption scheme used by Kestrel).

    Returns:
        Stats dict with counts of processed/failed records
    """
    old_master = passphrase_to_master_key(old_pass)
    new_master = passphrase_to_master_key(new_pass)

    db = SQLiteBackend(db_path)
    await db.connect()

    stats = {
        "total": 0,
        "rotated": 0,
        "already_unencrypted": 0,
        "failed": 0,
        "errors": []
    }

    try:
        # Get all conversations with agent_id for per-agent key derivation
        rows = await db.fetch_all(
            "SELECT id, agent_id, content, metadata FROM conversation_history"
        )
        stats["total"] = len(rows)
        logger.info(f"Found {len(rows)} conversation records to process")

        # Cache derived keys per agent
        old_agent_keys = {}
        new_agent_keys = {}

        for row in rows:
            msg_id, agent_id, content, metadata_json = row[0], row[1], row[2], row[3]

            # Check if encrypted
            import json
            meta = {}
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                except json.JSONDecodeError:
                    pass

            if not meta.get('enc'):
                stats["already_unencrypted"] += 1
                continue

            # Get or create per-agent keys
            if agent_id not in old_agent_keys:
                old_agent_keys[agent_id] = derive_agent_fernet(old_master, agent_id)
                new_agent_keys[agent_id] = derive_agent_fernet(new_master, agent_id)

            old_fernet = old_agent_keys[agent_id]
            new_fernet = new_agent_keys[agent_id]

            # Decrypt with old key
            try:
                decrypted = old_fernet.decrypt(content.encode('utf-8')).decode('utf-8')
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"Record {msg_id}: Failed to decrypt - {e}")
                logger.error(f"Failed to decrypt record {msg_id}: {e}")
                continue

            # Re-encrypt with new key
            try:
                encrypted = new_fernet.encrypt(decrypted.encode('utf-8')).decode('utf-8')
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"Record {msg_id}: Failed to re-encrypt - {e}")
                logger.error(f"Failed to re-encrypt record {msg_id}: {e}")
                continue

            # Update record (execute auto-commits)
            await db.execute(
                "UPDATE conversation_history SET content = ? WHERE id = ?",
                (encrypted, msg_id)
            )
            stats["rotated"] += 1

            if stats["rotated"] % 10 == 0:
                logger.info(f"Rotated {stats['rotated']} records...")

        logger.info(f"Rotation complete: {stats['rotated']} rotated, {stats['failed']} failed")

    finally:
        await db.close()

    return stats


async def main():
    parser = argparse.ArgumentParser(
        description="Rotate Emma's encryption key from old passphrase to new passphrase"
    )
    parser.add_argument("--db", default="agent_data/kestrel_prime.db",
                        help="Path to agent database")
    parser.add_argument("--old-pass", required=True,
                        help="Old passphrase (the one used to encrypt)")
    parser.add_argument("--new-pass", required=True,
                        help="New secure passphrase")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")

    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}")
        sys.exit(1)

    print("=" * 60)
    print("EMMA KEY ROTATION")
    print("=" * 60)
    print(f"Database: {args.db}")
    print(f"Old passphrase: {args.old_pass[:20]}...")
    print(f"New passphrase: {args.new_pass[:20]}...")
    print()

    # Show derived keys (first 16 chars only for security)
    old_key = base64.urlsafe_b64encode(hashlib.sha256(args.old_pass.encode()).digest()).decode()
    new_key = base64.urlsafe_b64encode(hashlib.sha256(args.new_pass.encode()).digest()).decode()
    print(f"Old derived key: {old_key[:16]}...")
    print(f"New derived key: {new_key[:16]}...")
    print()

    if args.dry_run:
        print("[DRY RUN] No changes will be made")
        # Just count records
        db = SQLiteBackend(args.db)
        await db.connect()
        rows = await db.fetch_all("SELECT COUNT(*) FROM conversation_history WHERE metadata LIKE '%\"enc\": true%' OR metadata LIKE '%\"enc\":true%'")
        await db.close()
        print(f"Would rotate approximately {rows[0][0]} encrypted records")
        return

    # Confirm
    print("This will re-encrypt all conversation history with the new key.")
    print("Make sure to update KESTREL_DATA_KEY in your .env after completion.")
    response = input("\nProceed? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    # Do rotation
    stats = await rotate_conversation_keys(args.db, args.old_pass, args.new_pass)

    print()
    print("=" * 60)
    print("ROTATION COMPLETE")
    print("=" * 60)
    print(f"Total records: {stats['total']}")
    print(f"Rotated: {stats['rotated']}")
    print(f"Already unencrypted: {stats['already_unencrypted']}")
    print(f"Failed: {stats['failed']}")

    if stats['errors']:
        print("\nErrors:")
        for err in stats['errors'][:5]:
            print(f"  - {err}")

    print()
    print("NEXT STEPS:")
    print(f"1. Update .env: KESTREL_DATA_KEY=\"{args.new_pass}\"")
    print("2. Restart the server with the new key")
    print("3. Verify conversations load correctly in the UI")


if __name__ == "__main__":
    asyncio.run(main())
