#!/usr/bin/env python3
"""
Migration script for encrypting existing plaintext private keys.

This script finds all plaintext PEM files in agent_data/ and migrates them
to the new encrypted format using KESTREL_DATA_KEY.

Usage:
    # Set your master key first
    export KESTREL_DATA_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
    
    # Run migration (dry run first)
    python scripts/migrate_keys.py --dry-run
    
    # Actually migrate
    python scripts/migrate_keys.py
    
    # Migrate specific directory
    python scripts/migrate_keys.py --directory /path/to/keys

Security Notes:
    - SAVE YOUR KESTREL_DATA_KEY SECURELY - losing it means losing access to encrypted keys
    - This script creates backups of original files before migration
    - Original PEM files are securely overwritten before deletion
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kestrel_sovereign.security.key_storage import (
    SecureKeyStorage,
    migrate_all_plaintext_keys,
    MasterKeyNotConfiguredError,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_pem_files(directory: Path) -> list[Path]:
    """Find all PEM files in directory and subdirectories."""
    return list(directory.rglob("*.pem"))


def main():
    parser = argparse.ArgumentParser(
        description="Migrate plaintext PEM keys to encrypted format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--directory", "-d",
        type=Path,
        default=Path("agent_data"),
        help="Directory containing keys to migrate (default: agent_data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually doing it"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Migrate without confirmation prompt"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for master key
    if not os.environ.get("KESTREL_DATA_KEY"):
        logger.error(
            "KESTREL_DATA_KEY environment variable is not set.\n"
            "Generate one with:\n"
            "  export KESTREL_DATA_KEY=$(python -c \"import secrets; print(secrets.token_urlsafe(32))\")\n\n"
            "IMPORTANT: Save this key securely - you will need it to access encrypted keys!"
        )
        sys.exit(1)
    
    # Find PEM files
    directory = args.directory.resolve()
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    
    pem_files = find_pem_files(directory)
    
    if not pem_files:
        logger.info(f"No PEM files found in {directory}")
        return
    
    logger.info(f"Found {len(pem_files)} PEM file(s) in {directory}")
    
    if args.dry_run:
        logger.info("\n=== DRY RUN - No changes will be made ===\n")
        for pem in pem_files:
            logger.info(f"Would migrate: {pem}")
        logger.info(f"\nTotal: {len(pem_files)} files would be migrated")
        return
    
    # Confirmation prompt
    if not args.force:
        print(f"\nThis will migrate {len(pem_files)} plaintext key(s) to encrypted format.")
        print("Original files will be securely deleted after migration.")
        print("\nMake sure you have:")
        print("  1. Saved KESTREL_DATA_KEY securely")
        print("  2. Backed up your agent_data directory (optional but recommended)")
        
        confirm = input("\nProceed with migration? [y/N]: ").strip().lower()
        if confirm != 'y':
            logger.info("Migration cancelled")
            return
    
    # Run migration
    logger.info(f"\nMigrating keys in {directory}...")
    try:
        results = migrate_all_plaintext_keys(directory)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
    
    # Report results
    print("\n" + "=" * 50)
    print("Migration Results")
    print("=" * 50)
    print(f"  Migrated: {results['migrated']}")
    print(f"  Skipped (already encrypted): {results['skipped']}")
    print(f"  Errors: {len(results['errors'])}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error['file']}: {error['error']}")
    
    if results['migrated'] > 0:
        print("\n✓ Migration complete!")
        print("  Encrypted keys are stored with .key.enc extension")
        print("  Original PEM files have been securely deleted")
    else:
        print("\n✓ No files needed migration")


if __name__ == "__main__":
    main()
