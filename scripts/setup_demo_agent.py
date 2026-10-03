#!/usr/bin/env python3
"""
Create a fresh demo agent for the technical demo (Issue #133, Track A).

This creates a clean, test-flagged agent named "Kestrel Demo Agent"
in a temporary directory, separate from any real agents (Claw, Emma, etc).

Usage:
    uv run python scripts/setup_demo_agent.py
    uv run python scripts/setup_demo_agent.py --data-dir /tmp/fresh \
        --manifest /tmp/fresh-inception.json --local-llm-only

Output:
    Creates agent_data/demo/ with a fresh kestrel_prime.db
    Prints the KESTREL_DB_PATH to use when starting the server.

``--data-dir`` targets another directory, which must not exist yet or be
empty: only the default ``agent_data/demo`` sandbox is wiped for a clean
slate, so an operator-supplied path is never deleted. ``--manifest`` records
the inception facts (DID, database path) as JSON so a caller can prove which
database a server is serving. ``--local-llm-only`` configures only the local
Ollama route, so no paid provider can be selected (the Console smoke, #2682).

To run the demo:
    KESTREL_DB_PATH=agent_data/demo uv run python -m kestrel_sovereign.server \
        --host 127.0.0.1 --port 8900 &
    cd demos/technical && npx playwright test --config=config.cjs
"""
import argparse
import asyncio
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kestrel_sovereign.inception_service import create_kestrel_identity_async

DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_data", "demo")

DEMO_AGENT_NAME = "Kestrel Demo Agent"

_ANTHROPIC_ROUTE_TOML = """
[llm.vendors.anthropic]
is_cloud = true

[llm.vendors.anthropic.routes.api]
adapter         = "AnthropicAdapter"
api_key_env     = "ANTHROPIC_API_KEY"
model           = "auto"
selection_hints = ["opus"]
"""

_OLLAMA_ROUTE_TOML = """
[llm.vendors.ollama]
is_cloud = false

[llm.vendors.ollama.routes.local]
adapter         = "OllamaAdapter"
host            = "http://localhost:11434"
model           = "auto"
selection_hints = ["llama3.2", "latest"]
"""


def build_demo_kestrel_toml(local_llm_only: bool = False) -> str:
    """Build a demo-friendly kestrel.toml using vendor/route/model defaults.

    ``local_llm_only`` drops the cloud route, leaving local Ollama as the only
    configured provider: an agent built from it cannot select a paid LLM.
    """
    if local_llm_only:
        return (
            "# Smoke agent config — local Ollama is the only route; no paid provider\n"
            "[llm]\n"
            'route_priority = ["ollama:local"]\n'
            + _OLLAMA_ROUTE_TOML
        )
    return (
        "# Demo agent config — vendor/route/model architecture, discovery picks concrete IDs\n"
        "[llm]\n"
        'route_priority = ["anthropic:api", "ollama:local"]\n'
        + _ANTHROPIC_ROUTE_TOML
        + _OLLAMA_ROUTE_TOML
    )


def prepare_data_dir(data_dir: str) -> str:
    """Return the absolute agent directory, ready for a fresh inception.

    The default demo sandbox is wiped for a clean slate. Any other directory
    must be absent or empty; it is created but never deleted, so a mistyped
    ``--data-dir`` cannot destroy an existing agent.
    """
    data_dir = os.path.abspath(data_dir)
    if data_dir == os.path.abspath(DEMO_DIR):
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"Cleaned existing demo directory: {data_dir}")
    elif os.path.exists(data_dir) and (
        not os.path.isdir(data_dir) or os.listdir(data_dir)
    ):
        raise SystemExit(
            f"error: --data-dir {data_dir} already exists and is not empty; "
            "a demo agent is only created in a fresh directory."
        )
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def write_manifest(path: str, *, agent_did: str, db_path: str, data_dir: str) -> None:
    """Record the inception facts a caller needs to prove database origin."""
    manifest = {
        "agent_name": DEMO_AGENT_NAME,
        "agent_did": agent_did,
        "db_path": os.path.abspath(db_path),
        "data_dir": data_dir,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fresh demo agent.")
    parser.add_argument(
        "--data-dir",
        default=DEMO_DIR,
        help="Agent directory (default: agent_data/demo, wiped first). Any "
             "other directory must be absent or empty.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Write the new agent's DID and database path to this JSON file.",
    )
    parser.add_argument(
        "--local-llm-only",
        action="store_true",
        help="Configure only the local Ollama route (no paid provider).",
    )
    return parser.parse_args(argv)


async def main(argv=None):
    args = _parse_args(argv)
    data_dir = prepare_data_dir(args.data_dir)

    print("Creating demo agent...")
    creds = await create_kestrel_identity_async(
        output_dir=data_dir,
        agent_name=DEMO_AGENT_NAME,
        is_test_instance=True,
        test_cycle_id="demo-live",
        expected_duration="demo session",
        is_demo=True,  # #766: server-side guardrails permit destructive ops on this agent
    )

    # Write demo-specific kestrel.toml with policy-based defaults
    kestrel_toml_path = os.path.join(data_dir, "kestrel.toml")
    with open(kestrel_toml_path, "w") as f:
        f.write(build_demo_kestrel_toml(local_llm_only=args.local_llm_only))
    print(f"  kestrel.toml written: {kestrel_toml_path}")

    if args.manifest:
        write_manifest(
            args.manifest,
            agent_did=creds.agent_did,
            db_path=creds.db_path,
            data_dir=data_dir,
        )
        print(f"  manifest written: {args.manifest}")

    print()
    print("=" * 60)
    print("  Demo Agent Created")
    print("=" * 60)
    print(f"  Name:     {DEMO_AGENT_NAME}")
    print(f"  DID:      {creds.agent_did}")
    print(f"  Database: {creds.db_path}")
    print(f"  Type:     TEST INSTANCE (demo-live)")
    print()
    print("  To start the demo server:")
    print(
        f"    KESTREL_DB_PATH={data_dir} uv run python "
        "-m kestrel_sovereign.server --host 127.0.0.1 --port 8900"
    )
    print()
    print("  Then run the demo:")
    print("    cd demos/technical && npx playwright test --config=config.cjs")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
