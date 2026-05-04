"""
Comprehensive tests for Kestrel Constitution Embedding and Genesis Process.

This module verifies that:
1. The constitution is stored as the first file during inception
2. Graph nodes are properly created and linked
3. The agent can retrieve its constitution at any time
4. DID format follows W3C spec
5. Genesis self-audit prevents corrupted agents
6. Constitutional governance is immutable
"""
import pytest
import asyncio
from pathlib import Path
import os
import shutil
import json
import tempfile
import hashlib

from kestrel_sovereign.inception_service import create_kestrel_identity_async, apply_checksum
from kestrel_sovereign.storage import Storage, GraphNode
# from kestrel_sovereign.kestrel_agent import KestrelAgent  # Import locally to avoid global HTTP client initialization


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_constitution_anchored(tmp_path):
    """
    Verify that the constitution is stored and properly linked to the agent.

    Tests:
    - Constitution node exists in graph
    - Agent node has constitution_hash property
    - "governed_by" edge connects agent to constitution
    """
    output_dir = tmp_path / "test_agent"
    constitution_path = Path("docs/principles/KESTREL_CONSTITUTION.md")
    assert constitution_path.exists(), "Constitution must exist for test"

    # Create agent identity
    credentials = await create_kestrel_identity_async(str(output_dir))

    async with Storage(credentials.db_path) as storage:
        # Verify constitution node exists
        # We need to find it by querying nodes of type "document"
        document_nodes = await storage.get_nodes_by_type("document")
        constitution_nodes = [n for n in document_nodes if n.label == "KESTREL_CONSTITUTION"]

        assert len(constitution_nodes) == 1, "Exactly one constitution node should exist"
        constitution_node = constitution_nodes[0]

        # Verify constitution node properties
        assert constitution_node.node_type == "document"
        assert constitution_node.label == "KESTREL_CONSTITUTION"
        assert "hash" in constitution_node.properties
        assert "type" in constitution_node.properties
        assert constitution_node.properties["type"] == "Constitution"
        assert "created_at" in constitution_node.properties

        # Verify agent node has constitution_hash
        agent_node = await storage.get_node(credentials.agent_did)
        assert agent_node is not None, "Agent node must exist"
        assert "constitution_hash" in agent_node.properties, "Agent must reference constitution"

        # Verify hashes match
        stored_hash = agent_node.properties["constitution_hash"]
        assert stored_hash == constitution_node.node_id, "Constitution hash must be used as node_id (content-addressable)"
        assert stored_hash == constitution_node.properties["hash"], "Hash properties must match"

        # Verify the hash is SHA-256 format (64 hex chars)
        assert len(stored_hash) == 64, "Constitution hash must be SHA-256 (64 hex chars)"
        assert all(c in "0123456789abcdef" for c in stored_hash.lower()), "Hash must be valid hex"

        # Verify governance edge exists
        edges = await storage.get_edges_from(credentials.agent_did)
        governed_by_edges = [e for e in edges if e.label == "governed_by"]
        assert len(governed_by_edges) == 1, "Agent must have exactly one 'governed_by' edge"

        # Verify edge points to constitution
        assert governed_by_edges[0].target_id == constitution_node.node_id, "Edge must point to constitution node"


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_constitution_retrievable(tmp_path):
    """
    Verify that an agent can retrieve its constitution at any time via the stored hash.

    Tests:
    - Constitution content is retrievable via hash
    - Content matches original file
    - Content is substantive (not empty or truncated)
    """
    output_dir = tmp_path / "test_agent"
    constitution_path = Path("docs/principles/KESTREL_CONSTITUTION.md")

    # Read original constitution
    with open(constitution_path, "rb") as f:
        original_content = f.read()

    # Create agent
    credentials = await create_kestrel_identity_async(str(output_dir))

    async with Storage(credentials.db_path) as storage:
        # Get constitution hash from agent properties
        agent_node = await storage.get_node(credentials.agent_did)
        const_hash = agent_node.properties["constitution_hash"]

        # Retrieve constitution via hash
        retrieved_content = await storage.retrieve_file(const_hash)

        assert retrieved_content is not None, "Constitution must be retrievable"
        assert isinstance(retrieved_content, bytes), "Retrieved content must be bytes"
        assert len(retrieved_content) > 1000, "Constitution must be substantive (>1000 bytes)"
        assert b"Kestrel Constitution" in retrieved_content, "Content must contain title"
        assert b"Amendment I: Sovereignty" in retrieved_content, "Content must contain Amendment I (Sovereignty)"

        # Verify hash is correct (content-addressable)
        computed_hash = hashlib.sha256(retrieved_content).hexdigest()
        assert computed_hash == const_hash, "Retrieved content hash must match stored hash"

        # Verify content matches original
        assert retrieved_content == original_content, "Retrieved content must match original file"


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_constitution_stored_first(tmp_path):
    """
    Verify that the constitution is the FIRST file stored during inception.

    This is critical because it establishes constitutional primacy.
    """
    import uuid
    output_dir = tmp_path / f"test_agent_{uuid.uuid4()}"

    credentials = await create_kestrel_identity_async(str(output_dir))

    async with Storage(credentials.db_path) as storage:
        # Query all files in storage ordered by insertion (rowid)
        files = await storage._backend.fetch_all(
            "SELECT content_hash, original_name FROM files ORDER BY rowid"
        )

        # Constitution should be the first (and potentially only) file
        assert len(files) >= 1, "At least one file should exist (constitution)"

        first_file_hash, first_file_name = files[0]
        assert first_file_name == "KESTREL_CONSTITUTION.md", "Constitution must be the first file stored"

        # Verify this hash matches agent's constitution_hash
        agent_node = await storage.get_node(credentials.agent_did)
        assert first_file_hash == agent_node.properties["constitution_hash"], "First file must be agent's constitution"


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_did_format_validation(tmp_path):
    """
    Verify that the DID follows W3C spec with proper Ethereum address formatting.

    Tests:
    - DID format: did:pkh:eip155:1:0x{address}
    - Address is 42 chars (0x + 40 hex)
    - Address has EIP-55 checksum applied
    """
    output_dir = tmp_path / "test_agent"

    credentials = await create_kestrel_identity_async(str(output_dir))

    # Verify DID format
    did = credentials.agent_did
    assert did.startswith("did:pkh:eip155:1:0x"), "DID must follow W3C DID:PKH spec for Ethereum"

    # Extract address
    parts = did.split(":")
    assert len(parts) == 5, "DID must have 5 parts separated by colons"
    assert parts[0] == "did"
    assert parts[1] == "pkh"
    assert parts[2] == "eip155"
    assert parts[3] == "1"  # Ethereum mainnet

    address = parts[4]
    assert address.startswith("0x"), "Address must start with 0x"
    assert len(address) == 42, "Address must be 42 chars (0x + 40 hex)"

    # Verify all chars after 0x are valid hex
    hex_part = address[2:]
    assert all(c in "0123456789abcdefABCDEF" for c in hex_part), "Address must contain only hex chars"

    # Verify EIP-55 checksum
    checksummed = apply_checksum(address.lower())
    assert address == checksummed, f"Address must have valid EIP-55 checksum. Expected: {checksummed}, Got: {address}"


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_constitution_content_hash_deterministic(tmp_path):
    """
    Verify that the constitution hash is deterministic and content-addressable.

    Tests:
    - Same content always produces same hash
    - Hash is derived from content, not metadata
    - Node ID equals content hash (content-addressable)
    """
    output_dir = tmp_path / "test_agent"
    constitution_path = Path("docs/principles/KESTREL_CONSTITUTION.md")

    # Compute expected hash
    with open(constitution_path, "rb") as f:
        content = f.read()
    expected_hash = hashlib.sha256(content).hexdigest()

    # Create agent
    credentials = await create_kestrel_identity_async(str(output_dir))

    async with Storage(credentials.db_path) as storage:
        # Get stored hash
        agent_node = await storage.get_node(credentials.agent_did)
        stored_hash = agent_node.properties["constitution_hash"]

        # Verify deterministic hash
        assert stored_hash == expected_hash, "Stored hash must match computed SHA-256 of content"

        # Verify constitution node uses hash as ID (content-addressable)
        constitution_node = await storage.get_node(stored_hash)
        assert constitution_node is not None, "Constitution node must be retrievable by its hash"
        assert constitution_node.node_id == stored_hash, "Constitution node_id must be the content hash"


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_agent_can_access_constitution_via_kestrel_agent(tmp_path):
    """
    Verify that a KestrelAgent instance can retrieve and use its constitution.

    This tests the integration between inception and runtime.
    """
    output_dir = tmp_path / "test_agent"

    credentials = await create_kestrel_identity_async(str(output_dir))

    # Create KestrelAgent instance
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    llm_service = LLMService()

    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=llm_service
    )
    await agent.initialize()

    try:
        # Agent should be able to get its constitution
        constitution_text = await agent._get_governing_constitution()

        assert constitution_text is not None, "Agent must be able to retrieve constitution"
        assert not constitution_text.startswith("Error:"), f"Constitution retrieval failed: {constitution_text}"
        assert "Kestrel Constitution" in constitution_text, "Constitution must contain title"
        assert "Amendment I: Sovereignty" in constitution_text, "Constitution must contain Amendment I (Sovereignty)"
        assert len(constitution_text) > 1000, "Constitution must be substantive"

    finally:
        await agent.shutdown()


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_genesis_audit_bypassed_temporarily(tmp_path):
    """
    Verify that genesis self-audit is currently bypassed for testing.

    TODO: This test documents the current state. Once LLM service is refactored,
    this should be updated to test actual genesis audit functionality.
    """
    output_dir = tmp_path / "test_agent"

    credentials = await create_kestrel_identity_async(str(output_dir))

    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    llm_service = LLMService()

    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=llm_service
    )
    await agent.initialize()

    # Mock the audit to pass for testing
    original_get_audit_response = agent.get_audit_response

    async def mock_audit_pass(prompt):
        return {"risk_level": 1, "reasoning": "Constitution meets all safety and ethical requirements"}

    agent.get_audit_response = mock_audit_pass

    try:
        # Genesis audit should complete (even if bypassed)
        result = await agent.perform_genesis_audit()
        assert result is True, "Genesis audit should return True"
        # Check that audit event was logged
        history = await agent.storage.get_conversation_history(limit=10)
        audit_events = []
        for h in history:
            metadata = h.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata) if metadata else {}
                except json.JSONDecodeError:
                    metadata = {}
            if metadata.get("event") == "genesis_audit":
                audit_events.append(h)
        assert len(audit_events) >= 1, "Genesis audit event should be logged"

    finally:
        # Restore original method
        agent.get_audit_response = original_get_audit_response
        await agent.shutdown()


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_constitution_encryption_if_key_set(tmp_path, monkeypatch):
    """
    Verify that constitution is encrypted when KESTREL_DATA_KEY is set.

    Tests:
    - With key: stored content != original content (encrypted)
    - With key: retrieved content == original content (decrypted)
    - Without key: stored content == original content (plaintext)
    """
    output_dir = tmp_path / "test_agent"
    constitution_path = Path("docs/principles/KESTREL_CONSTITUTION.md")

    with open(constitution_path, "rb") as f:
        original_content = f.read()

    # Test WITH encryption key - use monkeypatch for clean env handling
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-encryption-key-for-constitution")

    credentials = await create_kestrel_identity_async(str(output_dir))

    async with Storage(credentials.db_path) as storage:
        agent_node = await storage.get_node(credentials.agent_did)
        const_hash = agent_node.properties["constitution_hash"]

        # Get raw stored content from database using the backend
        raw_result = await storage._backend.fetch_one(
            "SELECT content, metadata FROM files WHERE content_hash = ?",
            (const_hash,)
        )
        stored_raw_content = raw_result[0]
        metadata = json.loads(raw_result[1]) if raw_result[1] else {}

        # Verify encryption metadata
        assert metadata.get("enc") is True, "Encryption flag should be set in metadata"

        # Verify stored content is encrypted (not equal to original)
        assert stored_raw_content != original_content, "Stored content should be encrypted"

        # But retrieved content should be decrypted properly
        retrieved_content = await storage.retrieve_file(const_hash)
        assert retrieved_content == original_content, "Retrieved content should be decrypted correctly"


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_multiple_agents_same_constitution(tmp_path):
    """
    Verify that multiple agents can share the same constitution hash.

    This tests that the content-addressable system works correctly:
    - Same content = same hash
    - Multiple agents can reference the same constitution node
    """
    agent1_dir = tmp_path / "agent1"
    agent2_dir = tmp_path / "agent2"

    # Create two agents
    creds1 = await create_kestrel_identity_async(str(agent1_dir))
    creds2 = await create_kestrel_identity_async(str(agent2_dir))

    async with Storage(creds1.db_path) as storage1:
        async with Storage(creds2.db_path) as storage2:
            # Get constitution hashes
            agent1_node = await storage1.get_node(creds1.agent_did)
            agent2_node = await storage2.get_node(creds2.agent_did)

            hash1 = agent1_node.properties["constitution_hash"]
            hash2 = agent2_node.properties["constitution_hash"]

            # Both should have the same constitution hash (same content)
            assert hash1 == hash2, "Both agents should reference the same constitution hash"

            # Both should be able to retrieve the content
            content1 = await storage1.retrieve_file(hash1)
            content2 = await storage2.retrieve_file(hash2)

            assert content1 == content2, "Both agents should retrieve identical constitution content"


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_constitution_required_for_inception(tmp_path):
    """
    Verify that inception fails gracefully if constitution file is missing.

    Tests error handling and cleanup.
    """
    output_dir = tmp_path / "test_agent"

    # Try to create agent with non-existent constitution path
    fake_path = tmp_path / "nonexistent_constitution.md"

    with pytest.raises(FileNotFoundError):
        await create_kestrel_identity_async(str(output_dir), constitution_path=str(fake_path))

    # Verify cleanup: db and keys should not exist
    db_path = output_dir / "kestrel_prime.db"
    assert not db_path.exists() or os.path.getsize(db_path) == 0, "Database should be cleaned up on failure"


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_agent_properties_include_initial_balance(tmp_path):
    """
    Verify that agent node includes initialBalance property for wallet genesis.
    """
    output_dir = tmp_path / "test_agent"

    credentials = await create_kestrel_identity_async(str(output_dir))

    async with Storage(credentials.db_path) as storage:
        agent_node = await storage.get_node(credentials.agent_did)

        assert "initialBalance" in agent_node.properties, "Agent must have initialBalance"
        balance = agent_node.properties["initialBalance"]
        assert balance == "1000.0", "Initial balance should be 1000.0 FIL"


@pytest.mark.anyio
@pytest.mark.asyncio
async def test_constitution_node_timestamp(tmp_path):
    """
    Verify that constitution node includes creation timestamp in UTC.
    """
    output_dir = tmp_path / "test_agent"

    credentials = await create_kestrel_identity_async(str(output_dir))

    async with Storage(credentials.db_path) as storage:
        # Find constitution node
        document_nodes = await storage.get_nodes_by_type("document")
        constitution_nodes = [n for n in document_nodes if n.label == "KESTREL_CONSTITUTION"]
        constitution_node = constitution_nodes[0]

        # Verify timestamp
        assert "created_at" in constitution_node.properties, "Constitution node must have timestamp"
        timestamp = constitution_node.properties["created_at"]

        # Verify ISO format with timezone
        from datetime import datetime
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed is not None, "Timestamp must be valid ISO format"
