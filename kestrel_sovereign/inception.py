#!/usr/bin/env python3
"""
Inception: A command-line tool for creating Kestrel agents.
"""
import argparse
import asyncio
from pathlib import Path
import inception_service

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
    args = parser.parse_args()

    output_dir = args.output_dir
    print(f"🚀 Starting Kestrel Inception in directory: '{output_dir}'")
    
    try:
        credentials = asyncio.run(inception_service.create_kestrel_identity(
            output_dir=str(output_dir)
        ))

        print("\n✅ Inception Complete.")
        print(f"   Agent Name: {credentials.agent_name}")
        print(f"   DID: {credentials.agent_did}")
        print(f"   Memory: {credentials.db_path}")

        # Display the critical backup information
        print("\n" + "="*80)
        print(credentials.backup_prompt)
        print("="*80)

    except FileExistsError as e:
        print(f"❌ ERROR: {e}")
        print("   Use the --force flag if you wish to overwrite it.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during inception: {e}")

if __name__ == "__main__":
    main() 