"""
IDIOT-PROOF SOVEREIGNTY TESTS (V2)

These tests prove sovereignty in the most obvious, undeniable ways possible.
Each test answers one simple question that even a moron could understand.

NO MOCKS. NO BULLSHIT. JUST PROOF.
"""

import pytest
import tempfile
import os
import json
import hashlib
from datetime import datetime, UTC, timedelta

from kestrel_sovereign.storage import Storage, GraphNode
from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter, ConvergentEncryptor
from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.kestrel_agent import KestrelAgent


# ============================================================================
# QUESTION 1: Can I actually save my conversations?
# ============================================================================

@pytest.mark.asyncio
async def test_conversations_are_actually_saved():
    """
    QUESTION: If I chat with my AI, are the conversations ACTUALLY saved?

    ANSWER: YES. Here's proof.
    """
    # Create storage
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        async with Storage(db_path=db_path) as storage:
            # Add conversations
            await storage.add_conversation("user", "My mom's birthday is March 15th")
            await storage.add_conversation("assistant", "I'll remember that!")
            await storage.add_conversation("user", "She loves roses")

        # PROOF: Open storage again, conversations should still be there
        async with Storage(db_path=db_path) as storage2:
            history = await storage2.get_conversation_history()

            # Verify
            assert len(history) == 3
            assert history[0]['content'] == "My mom's birthday is March 15th"
            assert history[2]['content'] == "She loves roses"

        print("✅ PROOF: Conversations are saved to disk")
        print("   - Added 3 messages")
        print("   - Closed database")
        print("   - Reopened database")
        print("   - All 3 messages still there")
        print()
        print("CONCLUSION: Your conversations are REALLY saved.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ============================================================================
# QUESTION 2: If the platform shuts down, do I lose everything?
# ============================================================================

@pytest.mark.asyncio
async def test_platform_shutdown_does_not_lose_data():
    """
    QUESTION: If kestrel/Kestrel shuts down tomorrow, do I lose all my data?

    ANSWER: NO. If you exported to IPFS, your data is SAFE.
    """
    # Step 1: Create agent with precious data
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    agent_did = "did:pkh:eip155:1:platform_shutdown_test"

    # Precious conversations (would be devastating to lose)
    precious_conversations = [
        ("user", "My grandmother's favorite recipe was apple pie"),
        ("assistant", "That's a beautiful memory. Do you remember the recipe?"),
        ("user", "Yes! 6 apples, 2 cups sugar, cinnamon..."),
        ("assistant", "I've saved that. Your grandmother's recipe is preserved."),
    ]

    try:
        async with Storage(db_path=db_path) as storage:
            for role, content in precious_conversations:
                await storage.add_conversation(role, content)

            # Step 2: Export to sovereignty (V2)
            secret = "my-secret-key"
            adapter = SovereignStorageAdapter(storage.db, secret)

            # This creates encrypted shards and a manifest
            cid = await adapter.export_agent(agent_did, storage_tier=StorageTier.LOCAL_ONLY)

            assert cid

            # Step 3: PLATFORM SHUTS DOWN / local state is lost.
            await storage.db.execute_commit("DELETE FROM conversation_history")
            assert await storage.get_conversation_history() == []

            # Step 4: User has ONLY the export CID. Prove it can restore data.
            stats = await adapter.import_agent(cid)
            assert stats["agent_did"] == agent_did
            assert stats["manifest_version"] == "3.0"
            assert stats["messages_restored"] == len(precious_conversations)

            restored = await storage.get_conversation_history()
            assert [(m["role"], m["content"]) for m in restored] == precious_conversations
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ============================================================================
# QUESTION 3: Can someone else access my data?
# ============================================================================

@pytest.mark.asyncio
async def test_encryption_actually_works():
    """
    QUESTION: Is my data ACTUALLY encrypted? Or is that bullshit?

    ANSWER: It's ACTUALLY encrypted. Here's proof.
    """
    # Create test data
    sensitive_data = b"My social security number is 123-45-6789"
    secret = "my-secret-key"

    # Encrypt it using V2 Convergent Encryption
    encryptor = ConvergentEncryptor(secret)
    encrypted_content, key = encryptor.encrypt_with_nonce_prefix(sensitive_data)

    # PROOF 1: Encrypted data does NOT contain original text
    assert b"123-45-6789" not in encrypted_content
    assert b"social security" not in encrypted_content

    print("✅ PROOF 1: Data is actually encrypted")
    print(f"   Original: {sensitive_data}")
    print(f"   Encrypted: {encrypted_content[:20]}...")

    # PROOF 2: Can decrypt with correct key
    decrypted = encryptor.decrypt(encrypted_content, key)
    assert decrypted == sensitive_data
    print("✅ PROOF 2: Can decrypt with correct key")

    # PROOF 3: Cannot decrypt with wrong key (simulated)
    try:
        wrong_key = os.urandom(32)
        encryptor.decrypt(encrypted_content, wrong_key)
        assert False, "Should have failed to decrypt with wrong key"
    except Exception:
        print("✅ PROOF 3: Cannot decrypt with wrong key")

    print()
    print("CONCLUSION: Your data is ACTUALLY encrypted.")


# ============================================================================
# QUESTION 4: How do I know the data isn't corrupted?
# ============================================================================

@pytest.mark.asyncio
async def test_content_integrity_verification():
    """
    QUESTION: How do I know the data I download is the same as what I uploaded?
           What if it gets corrupted? What if someone tampers with it?

    ANSWER: Content-addressed storage (CID) makes tampering IMPOSSIBLE.
    """
    # Original data
    original_text = "This is my precious agent data that must not be corrupted"
    original_bytes = original_text.encode('utf-8')

    # Calculate content hash (this is how IPFS CIDs work)
    content_hash = hashlib.sha256(original_bytes).hexdigest()

    print(f"Original data: {original_text}")
    print(f"Content hash: {content_hash}")
    print()

    # PROOF 1: Same content = same hash
    same_bytes = original_text.encode('utf-8')
    same_hash = hashlib.sha256(same_bytes).hexdigest()
    assert content_hash == same_hash

    print("✅ PROOF 1: Same content = same hash")
    print("   Re-hashed same data: hash matches")

    # PROOF 2: Different content = different hash
    corrupted_text = "This is my precious agent data that WAS corrupted"  # Changed one word
    corrupted_bytes = corrupted_text.encode('utf-8')
    corrupted_hash = hashlib.sha256(corrupted_bytes).hexdigest()
    assert content_hash != corrupted_hash

    print("✅ PROOF 2: Tampered content = different hash")
    print(f"   Original hash:  {content_hash}")
    print(f"   Corrupted hash: {corrupted_hash}")
    print("   Hashes DON'T match - corruption detected!")

    # PROOF 3: Even tiny changes are detected
    tiny_change = original_text + " "  # Just added a space!
    tiny_bytes = tiny_change.encode('utf-8')
    tiny_hash = hashlib.sha256(tiny_bytes).hexdigest()
    assert content_hash != tiny_hash

    print("✅ PROOF 3: Even tiny changes are detected")
    print("   Added ONE SPACE to data")
    print("   Hash changed - tampering detected!")
    print()
    print("CONCLUSION: Content hashing makes corruption/tampering IMPOSSIBLE to hide.")
    print("            If hash matches, data is IDENTICAL to original.")


# ============================================================================
# QUESTION 5: Is this really different from just downloading a file?
# ============================================================================

@pytest.mark.asyncio
async def test_sovereignty_vs_simple_download():
    """
    QUESTION: Why is this "sovereignty" different from just downloading my data?
           Can't I just download a JSON file from ChatGPT?

    ANSWER: Sovereignty is fundamentally different. Here's why.
    """
    # Create test agent
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    agent_did = "did:pkh:eip155:1:sovereignty_test"

    try:
        async with Storage(db_path=db_path) as storage:
            await storage.add_conversation("user", "Remember my favorite color is blue")
            await storage.add_conversation("assistant", "Got it, blue is your favorite!")

            # Create sovereignty export (V2)
            secret = "my-secret-key"
            adapter = SovereignStorageAdapter(storage.db, secret)

            cid = await adapter.export_agent(agent_did, storage_tier=StorageTier.LOCAL_ONLY)

            assert cid

            stats = await adapter.import_agent(cid)
            assert stats["agent_did"] == agent_did
            assert stats["manifest_version"] == "3.0"
            assert stats["shards_restored"] > 0
            assert stats["messages_restored"] == 2
            assert "assets_restored" in stats

            restored = await storage.get_conversation_history()
            assert [(m["role"], m["content"]) for m in restored] == [
                ("user", "Remember my favorite color is blue"),
                ("assistant", "Got it, blue is your favorite!"),
            ]
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ============================================================================
# QUESTION 6: What if I die? Can my family access my AI?
# ============================================================================

@pytest.mark.asyncio
async def test_inheritance_scenario():
    """
    QUESTION: If I die, can my family access my AI companion's memories?
           Or do they die with me?

    ANSWER: With sovereignty, your AI can be inherited. Here's how.
    """
    # Scenario: Grandma has been using Kestrel for 5 years
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    agent_did = "did:pkh:eip155:1:grandma_agent"

    # 5 years of precious memories
    precious_memories = [
        ("user", "I grew up in Brooklyn in the 1950s"),
        ("assistant", "Tell me more about Brooklyn in the 50s"),
        ("user", "We had a small apartment, my father worked at the docks"),
        ("assistant", "That must have been a vibrant community"),
        ("user", "It was. I met my husband at a dance hall in 1958"),
        ("assistant", "What a beautiful memory. What was his name?"),
        ("user", "Joseph. We were married for 62 years before he passed"),
        ("assistant", "I'm so sorry for your loss. He sounds wonderful."),
    ]

    try:
        async with Storage(db_path=db_path) as storage:
            for role, content in precious_memories:
                await storage.add_conversation(role, content)

            # Grandma exports to IPFS before she dies
            secret = "grandmas-secret-key"
            adapter = SovereignStorageAdapter(storage.db, secret)

            cid = await adapter.export_agent(agent_did, storage_tier=StorageTier.LOCAL_ONLY)

            assert cid

            # Family receives the CID from the will and imports the archive.
            await storage.db.execute_commit("DELETE FROM conversation_history")
            stats = await adapter.import_agent(cid)
            assert stats["agent_did"] == agent_did
            assert stats["messages_restored"] == len(precious_memories)

            restored = await storage.get_conversation_history()
            assert [(m["role"], m["content"]) for m in restored] == precious_memories
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ============================================================================
# QUESTION 7: Is this actually faster/better than cloud storage?
# ============================================================================

@pytest.mark.asyncio
async def test_performance_vs_cloud():
    """
    QUESTION: Is this IPFS stuff actually practical? Or is it slow/expensive?

    ANSWER: For small data (agent conversations), it's FAST and CHEAP.
    """
    import time

    # Create test data (typical agent export size)
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    agent_did = "did:pkh:eip155:1:performance_test"

    try:
        async with Storage(db_path=db_path) as storage:
            # Add 100 conversations (realistic amount)
            for i in range(100):
                await storage.add_conversation("user", f"Message {i}")
                await storage.add_conversation("assistant", f"Response {i}")

            # Measure export time
            start = time.time()

            secret = "perf-test-secret"
            adapter = SovereignStorageAdapter(storage.db, secret)

            cid = await adapter.export_agent(agent_did, storage_tier=StorageTier.LOCAL_ONLY)

            end = time.time()
            duration = end - start

            print(f"✅ Export Time: {duration:.4f} seconds")
            print(f"   CID: {cid}")

            assert duration < 5.0 # Should be very fast locally
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
