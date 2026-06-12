#!/usr/bin/env python3
"""
Inception: A command-line tool for creating Kestrel agents.
"""
import argparse
import asyncio
import logging
from pathlib import Path
import inception_service

logger = logging.getLogger(__name__)

def main():
    """
    Command-line wrapper for the inception service.
    """
    parser = argparse.ArgumentParser(description="Create a new Kestrel Sovereign Agent.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="The directory where the agent's identity files will be stored."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing agent database (backs it up first). Without "
             "--force, inception refuses to overwrite an existing agent.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    logger.info(f"Starting Kestrel Inception in directory: '{output_dir}'")

    try:
        credentials = asyncio.run(inception_service.create_kestrel_identity(
            output_dir=str(output_dir),
            force=args.force,
        ))

        logger.info("Inception Complete.")
        logger.info(f"   Agent Name: {credentials.agent_name}")
        logger.info(f"   DID: {credentials.agent_did}")
        logger.info(f"   Memory: {credentials.db_path}")

        # Display the critical backup information
        logger.info("\n" + "="*80)
        logger.info(credentials.backup_prompt)
        logger.info("="*80)

    except FileExistsError as e:
        logger.error(f"ERROR: {e}")
        logger.error("   Use the --force flag if you wish to overwrite it.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during inception: {e}")

if __name__ == "__main__":
    main() 