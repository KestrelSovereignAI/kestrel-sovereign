#!/usr/bin/env python3
"""
Demo: Emma's Autonomous Model Management

This script demonstrates the Kestrel agent's ability to autonomously manage
LLM models through the new tool system.

Emma (the agent) can:
- List available models from all providers
- Pull new models when needed
- Check storage status
- Clean up unused models

Uses REAL services (Ollama, LLM service) - NO MOCKS.
Uses small models only (qwen2.5:0.5b) for safety.
"""

import asyncio
from pathlib import Path
import sys

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode


async def main():
    """Demonstrate Emma's model management capabilities."""
    print("=" * 70)
    print("Emma's Autonomous Model Management Demo")
    print("=" * 70)
    print()

    # Create temporary storage for demo
    demo_db = Path("/tmp/emma_demo.db")
    if demo_db.exists():
        demo_db.unlink()

    print("🔧 Initializing Emma...")
    llm_service = LLMService()

    # Create Emma agent
    emma = KestrelAgent(
        did="did:kestrel:emma",
        storage_path=str(demo_db),
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )

    # Initialize async storage
    await emma.initialize()

    print("✅ Emma initialized successfully!")
    print()

    # Check that features are registered
    print(f"📋 Emma has {len(emma.features)} features available:")
    for name, feature in emma.features.items():
        print(f"   - {name}: {feature.tool_description}")
    print()

    # Demonstrate model management commands
    commands = [
        ("List available models", "!list-models"),
        ("Check storage status", "!storage-status"),
        ("Pull small model", "!pull-model qwen2.5:0.5b"),
        ("Verify model pulled", "!list-models"),
        ("Check storage again", "!storage-status"),
        ("Cleanup models (dry run)", "!cleanup-models --dry-run"),
    ]

    for description, command in commands:
        print(f"📝 {description}")
        print(f"   Command: {command}")
        print()

        # Execute command
        try:
            response = await emma.process_input(command)

            if response:
                print("   Response:")
                # Truncate long responses
                lines = response.split("\n")
                if len(lines) > 15:
                    print("   " + "\n   ".join(lines[:15]))
                    print(f"   ... ({len(lines) - 15} more lines)")
                else:
                    print("   " + "\n   ".join(lines))
            else:
                print("   ❌ Command not recognized")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        print()
        print("-" * 70)
        print()

    # Cleanup
    await emma.shutdown()
    await llm_service.close()
    if demo_db.exists():
        demo_db.unlink()

    print("=" * 70)
    print("Demo Complete!")
    print()
    print("Key Takeaways:")
    print("  ✅ Emma can list models from all providers (Ollama + cloud)")
    print("  ✅ Emma can pull new models autonomously")
    print("  ✅ Emma can monitor storage usage")
    print("  ✅ Emma can clean up old models to save space")
    print("  ✅ All operations use the new tool system")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
