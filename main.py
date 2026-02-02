#!/usr/bin/env python3
"""
The main entry point for the Kestrel Agent.
"""
import argparse
import asyncio
import os
from pathlib import Path
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.encryption import DecryptionError
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.config import load_config, DEFAULT_LLM_CONFIG_PATH
from kestrel_sovereign.filecoin_adapter import FilecoinAdapter
import logging

from kestrel_sovereign.kestrel_config.constants import SHUTDOWN_TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_agent_did_async(storage_dir: str) -> str:
    """
    Retrieves the agent's DID from async storage.
    This function is critical for server startup and agent initialization.

    Args:
        storage_dir: Directory containing the agent database (not the file path itself).
                    The database file is expected to be 'kestrel_prime.db' inside this directory.
    """
    # Construct full database file path from directory
    db_path = os.path.join(storage_dir, "kestrel_prime.db")
    storage = AsyncStorage(db_path)
    await storage.initialize()
    try:
        agent_nodes = await storage.get_nodes_by_type("agent")
        if not agent_nodes:
            raise ValueError("No agent found in the database. Please run inception service first.")
        if len(agent_nodes) > 1:
            logging.warning(f"Multiple agents found, using the first one: {agent_nodes[0].node_id}")
        return agent_nodes[0].node_id
    finally:
        await storage.close()

# This is a placeholder for a more robust discovery mechanism.
# In a real system, this would involve a more complex lookup.
async def get_agent_by_did(did: str) -> KestrelAgent:
    """
    Retrieves an agent instance based on its DID.
    """
    storage_path = os.environ.get("KESTREL_DB_PATH", os.getcwd())
    llm_service = LLMService()
    agent = KestrelAgent(did=did, storage_path=storage_path, llm_service=llm_service)
    await agent.initialize()
    return agent

async def main():
    parser = argparse.ArgumentParser(description="Kestrel Sovereign Agent Interface")
    parser.add_argument(
        "db_path",
        nargs='?',
        type=str,
        default=None,
        help="Path to the agent's memory database. Overrides KESTREL_DB_PATH."
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default=DEFAULT_LLM_CONFIG_PATH,
        help=f"Path to LLM config. Defaults to {DEFAULT_LLM_CONFIG_PATH}"
    )
    parser.add_argument(
        "--app",
        type=str,
        default=None,
        choices=['elderly'],
        help="Load an application extension."
    )
    args = parser.parse_args()

    db_path = args.db_path or os.environ.get("KESTREL_DB_PATH")
    if not db_path:
        print("❌ Error: Agent database path not specified.")
        return

    if not os.path.exists(db_path):
        print(f"❌ Error: Database file not found at '{db_path}'.")
        return

    storage_dir = db_path
    agent_did = await get_agent_did_async(storage_dir)
    if not agent_did:
        print(f"❌ Error: Could not determine agent's DID from '{storage_dir}'.")
        return

    # Construct full database file path from directory (same as get_agent_did_async)
    storage_path = os.path.join(storage_dir, "kestrel_prime.db")
    llm_service = LLMService()
    agent = KestrelAgent(did=agent_did, storage_path=storage_path, llm_service=llm_service)
    await agent.initialize()

    if args.app:
        extension_class = None
        if args.app == 'elderly':
            from kestrel_sovereign.extensions.elderly_extension import ElderlyExtension
            extension_class = ElderlyExtension
        
        if extension_class:
            agent.extension = extension_class(agent)
            agent.app_context = args.app
            print(f"   Extension Loaded: {args.app}")

    print("✅ Kestrel Agent Initialized.")
    print(f"   DID: {agent.agent_id}")
    print(f"   Memory: {db_path}")

    decryption_error_count = 0
    MAX_DECRYPTION_ERRORS = 3

    try:
        while True:
            user_input = input("\n> ")
            if user_input.lower() == '!quit':
                break
            try:
                response = await agent.process_input(user_input)
                decryption_error_count = 0  # Reset on success
                print(f"\nKestrel: {response}")
            except DecryptionError as e:
                decryption_error_count += 1
                logger.error(f"DecryptionError during processing: {e}")
                print(f"\n🔐 DECRYPTION ERROR: Cannot read encrypted data.")
                print(f"   This usually means KESTREL_DATA_KEY is incorrect or missing.")
                print(f"   Error count: {decryption_error_count}/{MAX_DECRYPTION_ERRORS}")

                if decryption_error_count >= MAX_DECRYPTION_ERRORS:
                    print("\n⚠️  Too many decryption errors. Entering safe mode.")
                    print("   The agent cannot access encrypted memories.")
                    print("   Please verify KESTREL_DATA_KEY and restart.")
                    print("   Use !quit to exit.")
                    # Trigger safe mode in agent
                    if hasattr(agent, '_safe_mode'):
                        agent._safe_mode = True

    except KeyboardInterrupt:
        print("\nDeactivating agent...")
    finally:
        # Graceful shutdown with timeout
        try:
            await asyncio.wait_for(agent.shutdown(), timeout=SHUTDOWN_TIMEOUT)
            print("Agent deactivated.")
        except asyncio.TimeoutError:
            print(f"Agent shutdown timed out ({SHUTDOWN_TIMEOUT}s), forcing exit.")
        except asyncio.CancelledError:
            print("Agent shutdown cancelled.")
        except Exception as e:
            logger.debug(f"Error during shutdown: {e}")
            print("Agent deactivated (with errors).")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle double Ctrl+C gracefully
        print("\nForced exit.") 