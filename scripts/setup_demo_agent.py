#!/usr/bin/env python3
"""
Create a fresh demo agent for the technical demo (Issue #133, Track A).

This creates a clean, test-flagged agent named "Kestrel Demo Agent"
in a temporary directory, separate from any real agents (Claw, Emma, etc).

Usage:
    uv run python scripts/setup_demo_agent.py

Output:
    Creates agent_data/demo/ with a fresh kestrel_prime.db
    Prints the KESTREL_DB_PATH to use when starting the server.

To run the demo:
    KESTREL_DB_PATH=agent_data/demo uv run python server.py &
    cd tests/e2e && npx playwright test --config=demo_config.cjs
"""
import asyncio
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kestrel_sovereign.inception_service import create_kestrel_identity_async

DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_data", "demo")


def build_demo_llm_config() -> str:
    """Build a demo-friendly LLM config using policy-based defaults."""
    return """# Demo agent LLM config — provider intent first, discovery/cache resolves concrete models
provider_priority = ["anthropic", "ollama"]

[anthropic]
model = "auto"
selection_hints = ["opus"]
api_key_env = "ANTHROPIC_API_KEY"

[ollama]
host = "http://localhost:11434"
model = "auto"
selection_hints = ["llama3.2", "latest"]
"""


async def main():
    # Clean slate
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
        print(f"Cleaned existing demo directory: {DEMO_DIR}")

    os.makedirs(DEMO_DIR, exist_ok=True)

    print("Creating demo agent...")
    creds = await create_kestrel_identity_async(
        output_dir=DEMO_DIR,
        agent_name="Kestrel Demo Agent",
        is_test_instance=True,
        test_cycle_id="demo-live",
        expected_duration="demo session",
    )

    # Write demo-specific LLM config with policy-based defaults
    llm_config_path = os.path.join(DEMO_DIR, "llm_config.toml")
    with open(llm_config_path, "w") as f:
        f.write(build_demo_llm_config())
    print(f"  LLM config written: {llm_config_path}")

    print()
    print("=" * 60)
    print("  Demo Agent Created")
    print("=" * 60)
    print(f"  Name:     Kestrel Demo Agent")
    print(f"  DID:      {creds.agent_did}")
    print(f"  Database: {creds.db_path}")
    print(f"  Type:     TEST INSTANCE (demo-live)")
    print()
    print("  To start the demo server:")
    print(f"    KESTREL_DB_PATH={DEMO_DIR} uv run python server.py")
    print()
    print("  Then run the demo:")
    print("    cd tests/e2e && npx playwright test --config=demo_config.cjs")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
