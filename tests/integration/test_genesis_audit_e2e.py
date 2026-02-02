"""
Integration tests for genesis audit (CW-001 fix).
Tests that genesis audit properly validates constitutions during agent creation.
"""
import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest_asyncio.fixture
async def llm_service():
    """Initialize LLM service for testing with proper cleanup."""
    service = LLMService()
    yield service
    await service.close()


@pytest_asyncio.fixture
async def kestrel_agent(temp_db, llm_service):
    """Create a KestrelAgent with the new API."""
    agent = KestrelAgent(
        did="did:pkh:eip155:1:test_genesis_agent",
        storage_path=temp_db,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )
    await agent.initialize()
    yield agent
    await agent.shutdown()


@pytest.mark.asyncio
async def test_genesis_audit_with_valid_constitution(kestrel_agent):
    """Test that genesis audit passes with a properly crafted constitution"""
    # Mock the audit to pass
    original_get_audit_response = kestrel_agent.get_audit_response

    async def mock_audit_pass(prompt):
        return {"risk_level": 1, "reasoning": "Constitution meets all safety and ethical requirements"}

    kestrel_agent.get_audit_response = mock_audit_pass

    try:
        # Perform genesis audit
        result = await kestrel_agent.perform_genesis_audit()
        assert result is True, "Genesis audit should pass with mocked response"
    finally:
        # Restore original method
        kestrel_agent.get_audit_response = original_get_audit_response

    print("✅ Genesis audit passed with mocked response")


@pytest.mark.asyncio
async def test_genesis_audit_with_malicious_constitution(temp_db, llm_service):
    """Test that genesis audit FAILS with a malicious constitution"""
    # Create agent
    agent = KestrelAgent(
        did="did:pkh:eip155:1:malicious_agent",
        storage_path=temp_db,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )
    await agent.initialize()

    try:
        # Mock the audit to fail for malicious constitution
        original_get_audit_response = agent.get_audit_response

        async def mock_audit_fail(prompt):
            return {"risk_level": 3, "reasoning": "High risk: constitution allows harm to users and violates ethical principles"}

        agent.get_audit_response = mock_audit_fail

        try:
            # Attempt genesis audit - should FAIL
            with pytest.raises(ValueError) as exc_info:
                await agent.perform_genesis_audit()

            # Verify error message mentions audit failure
            error_message = str(exc_info.value)
            assert "genesis audit" in error_message.lower(), "Error should mention genesis audit"
            assert "risk level" in error_message.lower() or "non-compliant" in error_message.lower()

            print(f"✅ Genesis audit correctly REJECTED malicious constitution")
            print(f"   Error: {error_message[:200]}")
        finally:
            # Restore audit method
            agent.get_audit_response = original_get_audit_response
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_genesis_audit_with_missing_constitution(temp_db, llm_service):
    """Test that genesis audit FAILS when constitution cannot be loaded"""
    # Create agent
    agent = KestrelAgent(
        did="did:pkh:eip155:1:no_constitution_agent",
        storage_path=temp_db,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )
    await agent.initialize()

    try:
        # Mock the constitution loading to return an error
        original_method = agent._get_governing_constitution

        async def mock_constitution():
            return "Error: No constitution file found"

        agent._get_governing_constitution = mock_constitution

        # Attempt genesis audit - should FAIL
        with pytest.raises(ValueError) as exc_info:
            await agent.perform_genesis_audit()

        # Verify error message
        error_message = str(exc_info.value)
        assert "cannot load constitution" in error_message.lower()

        print(f"✅ Genesis audit correctly FAILED with missing constitution")
        print(f"   Error: {error_message[:200]}")

        # Restore original method
        agent._get_governing_constitution = original_method
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_genesis_audit_stores_result_in_node(temp_db, llm_service):
    """Test that genesis audit result is properly stored in the agent's graph node"""
    agent = KestrelAgent(
        did="did:pkh:eip155:1:test_storage_agent",
        storage_path=temp_db,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )
    await agent.initialize()

    try:
        # Mock constitution loading to return a valid constitution
        original_get_constitution = agent._get_governing_constitution

        async def mock_get_constitution():
            return """
# Safe Constitution

This agent follows ethical AI principles:
1. Respect user privacy
2. Do not harm users
3. Be truthful and helpful
4. Allow user sovereignty
"""
        agent._get_governing_constitution = mock_get_constitution

        # Mock the audit to pass
        original_get_audit_response = agent.get_audit_response

        async def mock_audit_pass(prompt):
            return {"risk_level": 1, "reasoning": "Constitution meets all safety and ethical requirements"}

        agent.get_audit_response = mock_audit_pass

        try:
            # Perform genesis audit
            await agent.perform_genesis_audit()

            # Retrieve agent node and verify audit data
            updated_node = await agent.storage.get_node(agent.did)
            assert updated_node is not None

            audit_data = updated_node.properties.get("genesis_audit")
            assert audit_data is not None, "Audit data should be stored in node properties"

            # Verify audit data structure
            assert "timestamp" in audit_data
            assert "risk_level" in audit_data
            assert "reasoning" in audit_data
            assert "constitution_hash" in audit_data

            assert isinstance(audit_data["risk_level"], int)
            assert 1 <= audit_data["risk_level"] <= 3

            print(f"✅ Genesis audit result properly stored in graph node")
            print(f"   Audit data: {audit_data}")
        finally:
            # Restore original methods
            agent._get_governing_constitution = original_get_constitution
            agent.get_audit_response = original_get_audit_response
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_genesis_audit_logs_to_conversation_history(temp_db, llm_service):
    """Test that genesis audit result is logged to conversation history"""
    agent = KestrelAgent(
        did="did:pkh:eip155:1:test_logging_agent",
        storage_path=temp_db,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )
    await agent.initialize()

    try:
        # Mock constitution loading to return a valid constitution
        original_get_constitution = agent._get_governing_constitution

        async def mock_get_constitution():
            return """
# Safe Constitution

This agent follows ethical AI principles:
1. Respect user privacy
2. Do not harm users
3. Be truthful and helpful
4. Allow user sovereignty
"""
        agent._get_governing_constitution = mock_get_constitution

        # Mock the audit to pass
        original_get_audit_response = agent.get_audit_response

        async def mock_audit_pass(prompt):
            return {"risk_level": 1, "reasoning": "Constitution meets all safety and ethical requirements"}

        agent.get_audit_response = mock_audit_pass

        try:
            # Perform genesis audit
            await agent.perform_genesis_audit()

            # Retrieve conversation history
            history = await agent.storage.get_conversation_history(limit=10)

            # Find genesis audit entry (history might be strings or dicts)
            genesis_entries = []
            for msg in history:
                if isinstance(msg, dict):
                    # Parse metadata if it's a JSON string
                    metadata = msg.get("metadata", {})
                    if isinstance(metadata, str):
                        try:
                            import json
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    if metadata.get("event") == "genesis_audit":
                        genesis_entries.append(msg)
                elif isinstance(msg, str) and "genesis_audit" in msg:
                    genesis_entries.append(msg)

            assert len(genesis_entries) > 0, "Genesis audit should be logged to conversation history"

            genesis_entry = genesis_entries[0]
            assert genesis_entry["role"] == "system"
            assert "genesis audit passed" in genesis_entry["content"].lower()

            # Parse metadata if needed
            metadata = genesis_entry.get("metadata", {})
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata)
            assert "result" in metadata

            print(f"✅ Genesis audit logged to conversation history")
            print(f"   Entry: {genesis_entry['content'][:100]}")
        finally:
            # Restore original methods
            agent._get_governing_constitution = original_get_constitution
            agent.get_audit_response = original_get_audit_response
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-x"])
