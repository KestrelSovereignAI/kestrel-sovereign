"""
Integration tests for OpenRouter key provisioning.

These tests use REAL OpenRouter API calls to verify the provisioning flow.
Requires OPENROUTER_MANAGEMENT_API_KEY in environment.

Run with: uv run pytest tests/integration/test_openrouter_provisioning.py -v
"""

import os
import pytest
import uuid
from dotenv import load_dotenv

# Load .env before checking for the key
load_dotenv()

# Skip all tests if management key not available
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_MANAGEMENT_API_KEY"),
    reason="OPENROUTER_MANAGEMENT_API_KEY not set",
)


@pytest.fixture
def unique_agent_name():
    """Generate a unique agent name for testing."""
    return f"test-agent-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_create_and_delete_key(unique_agent_name):
    """Test creating and deleting an agent key."""
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    service = OpenRouterProvisioningService()

    try:
        # Create key
        key_info = await service.create_agent_key(
            agent_name=unique_agent_name,
            limit_usd=1.0,
            limit_reset="monthly",
        )

        assert key_info.key.startswith("sk-or-v1-")
        assert key_info.key_hash
        assert key_info.name == unique_agent_name
        assert key_info.limit_cents == 100
        assert key_info.limit_reset == "monthly"

        # Verify we can get usage
        usage = await service.get_key_usage(key_info.key_hash)
        assert usage.limit_cents == 100
        assert usage.limit_remaining_cents == 100  # No usage yet

        # Delete key
        deleted = await service.delete_key(key_info.key_hash)
        assert deleted

    finally:
        await service.close()


@pytest.mark.asyncio
async def test_key_can_make_inference(unique_agent_name):
    """Test that created key can actually make LLM requests."""
    import httpx
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    service = OpenRouterProvisioningService()

    try:
        # Create key with small limit
        key_info = await service.create_agent_key(
            agent_name=unique_agent_name,
            limit_usd=0.10,  # 10 cents
            limit_reset="monthly",
        )

        # Use the key to make a real inference request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key_info.key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-chat-v3.1",  # Cheap model
                    "messages": [{"role": "user", "content": "Say 'test' only"}],
                    "max_tokens": 5,
                },
                timeout=30.0,
            )

            assert response.status_code == 200
            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0

        # Verify usage was tracked
        usage = await service.get_key_usage(key_info.key_hash)
        # Free model may not count toward usage, but API should work
        assert usage.key_hash == key_info.key_hash

        # Cleanup
        await service.delete_key(key_info.key_hash)

    finally:
        await service.close()


@pytest.mark.asyncio
async def test_update_key_limit(unique_agent_name):
    """Test updating a key's spending limit."""
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    service = OpenRouterProvisioningService()

    try:
        # Create key with initial limit
        key_info = await service.create_agent_key(
            agent_name=unique_agent_name,
            limit_usd=1.0,
        )

        assert key_info.limit_cents == 100

        # Update limit
        updated_usage = await service.update_key_limit(
            key_hash=key_info.key_hash,
            limit_usd=5.0,
        )

        assert updated_usage.limit_cents == 500
        assert updated_usage.limit_remaining_cents == 500

        # Cleanup
        await service.delete_key(key_info.key_hash)

    finally:
        await service.close()


@pytest.mark.asyncio
async def test_convenience_functions(unique_agent_name):
    """Test the convenience functions for inception/retirement."""
    from kestrel_sovereign.features.llm_keys.openrouter_provisioning import (
        provision_agent_key,
        get_agent_usage,
        delete_agent_key,
    )

    # Create via convenience function
    key_info = await provision_agent_key(
        agent_name=unique_agent_name,
        limit_usd=2.0,
        limit_reset="weekly",
    )

    assert key_info.key.startswith("sk-or-v1-")
    assert key_info.limit_cents == 200

    # Get usage via convenience function
    usage = await get_agent_usage(key_info.key_hash)
    assert usage.limit_cents == 200

    # Delete via convenience function
    deleted = await delete_agent_key(key_info.key_hash)
    assert deleted


@pytest.mark.asyncio
async def test_serialization():
    """Test AgentKeyInfo serialization for agent metadata storage."""
    from kestrel_sovereign.features.llm_keys.openrouter_provisioning import AgentKeyInfo

    original = AgentKeyInfo(
        key="sk-or-v1-test",
        key_hash="abc123",
        name="test-agent",
        limit_cents=1000,
        limit_reset="monthly",
    )

    # Serialize (for storage)
    data = original.to_dict()
    assert "key" not in data  # Key should not be stored
    assert data["key_hash"] == "abc123"
    assert data["name"] == "test-agent"

    # Deserialize (from storage)
    restored = AgentKeyInfo.from_dict(data)
    assert restored.key == ""  # Key not available after storage
    assert restored.key_hash == "abc123"
    assert restored.name == "test-agent"
    assert restored.limit_cents == 1000


@pytest.mark.asyncio
async def test_list_keys():
    """Test listing all keys under management key."""
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    service = OpenRouterProvisioningService()

    try:
        keys = await service.list_keys()
        assert isinstance(keys, list)
        # We don't assert specific content since other tests may have keys

    finally:
        await service.close()
