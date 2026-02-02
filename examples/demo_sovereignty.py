#!/usr/bin/env python3
"""
Demonstration of Kestrel's Sovereignty System

This demonstrates the core value proposition:
"Your AI companion can never be taken away from you."

Even if:
- The platform shuts down
- Your device breaks
- You switch to a different service

As long as you have the CID, you can restore your agent.
"""

import asyncio
import logging
from kestrel_sovereign.storage import Storage, GraphNode
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.features.privacy import PrivacyMode
from kestrel_sovereign.kestrel_agent import KestrelAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_sovereignty_export():
    """
    Demo: Export an agent to IPFS and get sovereignty receipt.

    This simulates a user who:
    1. Has conversations with their AI companion
    2. Exports to IPFS for sovereignty
    3. Gets a CID (their proof of ownership)
    4. Can restore anywhere with that CID
    """

    print("=" * 70)
    print("KESTREL SOVEREIGNTY DEMONSTRATION")
    print("=" * 70)
    print()

    # Step 1: Create an agent with some data
    print("📝 Step 1: Creating agent with conversations...")
    print()

    storage = Storage(db_path=":memory:")  # In-memory for demo
    llm_service = LLMService()

    agent_did = "did:pkh:eip155:1:0xdemo123"

    # Create agent node
    agent_node = GraphNode(
        node_id=agent_did,
        node_type="agent",
        label="Demo Agent",
        properties={"purpose": "demonstrate sovereignty"}
    )
    storage.add_node(agent_node)

    # Add some conversations (simulating user interactions)
    conversations = [
        ("user", "Hi! What's your name?"),
        ("assistant", "I'm your Kestrel companion. How can I help you today?"),
        ("user", "Tell me about sovereignty."),
        ("assistant", "Sovereignty means you own and control your data and AI companion. No one can take it away from you!"),
        ("user", "That's amazing! How does it work?"),
        ("assistant", "Through IPFS and cryptographic proofs. Your data gets a unique CID that you control."),
    ]

    for role, content in conversations:
        storage.add_conversation(role, content, metadata={"demo": True})

    print(f"✅ Created agent: {agent_did}")
    print(f"✅ Added {len(conversations)} conversation messages")
    print()

    # Step 2: Initialize agent
    print("🤖 Step 2: Initializing KestrelAgent...")
    print()

    agent = KestrelAgent(
        did=agent_did,
        storage=storage,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )

    # Disable audit for demo (would need real LLM)
    agent.audit_enabled = False

    print("✅ Agent initialized")
    print()

    # Step 3: Export to IPFS for sovereignty
    print("🔐 Step 3: Exporting to IPFS for sovereignty...")
    print()
    print("Running command: !export-sovereignty")
    print()

    try:
        # This is what the user would type
        result = await agent.process_input("!export-sovereignty")

        print("COMMAND RESULT:")
        print("-" * 70)
        print(result)
        print("-" * 70)
        print()

    except Exception as e:
        print(f"⚠️  IPFS export failed: {e}")
        print()
        print("This is expected if IPFS is not running.")
        print()
        print("To run this demo successfully:")
        print("1. Install IPFS: https://docs.ipfs.tech/install/")
        print("2. Start IPFS daemon: ipfs daemon")
        print("3. Run this demo again")
        print()
        print("What WOULD happen if IPFS was running:")
        print("- Agent data packaged into snapshot")
        print("- Encrypted with Fernet key")
        print("- Uploaded to IPFS network")
        print("- CID returned (e.g., QmYwAPJzv5CZsnA636s8K6v...)")
        print("- User saves CID = sovereignty proof!")
        print()

    # Step 4: Check sovereignty status
    print("📋 Step 4: Checking sovereignty status...")
    print()

    try:
        status = await agent.process_input("!sovereignty-status")
        print("STATUS RESULT:")
        print("-" * 70)
        print(status)
        print("-" * 70)
        print()

    except Exception as e:
        print(f"Error: {e}")
        print()

    # Summary
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print()
    print("1. **Sovereignty = CID**")
    print("   The user gets an IPFS Content ID (CID)")
    print("   With this CID, they can restore their agent ANYWHERE")
    print()
    print("2. **Platform Independence**")
    print("   - Kestrel server shuts down? No problem, use CID")
    print("   - Switch to different platform? Import with CID")
    print("   - Device dies? Restore from CID on new device")
    print()
    print("3. **Inheritance Ready**")
    print("   - CID can be written in a will")
    print("   - Family gets CID, can talk to memories")
    print("   - Elderly stories preserved forever")
    print()
    print("4. **True Ownership**")
    print("   - No vendor lock-in")
    print("   - No subscription to 'keep your data'")
    print("   - Cryptographic proof of ownership")
    print()
    print("This is what makes Kestrel different from ChatGPT/Claude.")
    print("Your AI companion can NEVER be taken away from you.")
    print()


async def demo_sovereignty_workflow():
    """
    Full workflow demonstration:
    1. User creates agent
    2. Has conversations
    3. Exports to IPFS (gets CID)
    4. Simulates platform shutdown
    5. Imports from IPFS on 'new' platform
    6. Agent restored with all data!
    """

    print()
    print("=" * 70)
    print("FULL SOVEREIGNTY WORKFLOW")
    print("=" * 70)
    print()

    # This would be the full lifecycle demo
    # For now, we'll just show the export part
    await demo_sovereignty_export()

    print("Future: Import demo")
    print("  (Would show loading agent from CID)")
    print()


if __name__ == "__main__":
    print()
    print("🚀 Kestrel Sovereignty Demo")
    print()
    print("This demonstrates why Kestrel is revolutionary:")
    print("Your AI companion is YOURS. Forever. No matter what.")
    print()

    asyncio.run(demo_sovereignty_workflow())
