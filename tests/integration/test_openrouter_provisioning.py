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


@pytest.fixture
async def provisioning_service():
    """Create and cleanup a provisioning service."""
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    service = OpenRouterProvisioningService()
    created_hashes = []

    class TrackedService:
        """Wrapper that tracks created keys for guaranteed cleanup."""

        def __getattr__(self, name):
            return getattr(service, name)

        async def create_agent_key(self, **kwargs):
            key_info = await service.create_agent_key(**kwargs)
            created_hashes.append(key_info.key_hash)
            return key_info

    yield TrackedService()

    # Always clean up ALL keys created during this test
    for key_hash in created_hashes:
        try:
            await service.delete_key(key_hash)
        except Exception:
            pass
    await service.close()


@pytest.mark.asyncio
async def test_create_and_delete_key(unique_agent_name, provisioning_service):
    """Test creating and deleting an agent key."""
    key_info = await provisioning_service.create_agent_key(
        agent_name=unique_agent_name,
        limit_usd=0.10,
        limit_reset="monthly",
    )

    assert key_info.key.startswith("sk-or-v1-")
    assert key_info.key_hash
    assert key_info.name == unique_agent_name
    assert key_info.limit_usd == 0.10
    assert key_info.limit_reset == "monthly"

    # Verify we can get usage
    usage = await provisioning_service.get_key_usage(key_info.key_hash)
    assert usage.limit_usd == 0.10
    assert usage.limit_remaining_usd == 0.10  # No usage yet

    # Explicit delete (fixture will also clean up if this fails)
    deleted = await provisioning_service.delete_key(key_info.key_hash)
    assert deleted


@pytest.mark.asyncio
async def test_key_can_make_inference(unique_agent_name, provisioning_service):
    """Test that created key can actually make LLM requests."""
    import httpx

    key_info = await provisioning_service.create_agent_key(
        agent_name=unique_agent_name,
        limit_usd=0.10,
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
    usage = await provisioning_service.get_key_usage(key_info.key_hash)
    assert usage.key_hash == key_info.key_hash


@pytest.mark.asyncio
async def test_update_key_limit(unique_agent_name, provisioning_service):
    """Test updating a key's spending limit."""
    key_info = await provisioning_service.create_agent_key(
        agent_name=unique_agent_name,
        limit_usd=0.10,
    )

    assert key_info.limit_usd == 0.10

    # Update limit
    updated_usage = await provisioning_service.update_key_limit(
        key_hash=key_info.key_hash,
        limit_usd=5.0,
    )

    assert updated_usage.limit_usd == 5.0
    assert updated_usage.limit_remaining_usd == 5.0


@pytest.mark.asyncio
async def test_convenience_functions(unique_agent_name):
    """Test the convenience functions for inception/retirement."""
    from kestrel_sovereign.features.llm_keys.openrouter_provisioning import (
        provision_agent_key,
        get_agent_usage,
        delete_agent_key,
    )

    key_info = await provision_agent_key(
        agent_name=unique_agent_name,
        limit_usd=0.10,
        limit_reset="weekly",
    )

    try:
        assert key_info.key.startswith("sk-or-v1-")
        assert key_info.limit_usd == 0.10

        # Get usage via convenience function
        usage = await get_agent_usage(key_info.key_hash)
        assert usage.limit_usd == 0.10
    finally:
        # Always clean up
        await delete_agent_key(key_info.key_hash)


@pytest.mark.asyncio
async def test_serialization():
    """Test AgentKeyInfo serialization for agent metadata storage."""
    from kestrel_sovereign.features.llm_keys.openrouter_provisioning import AgentKeyInfo

    original = AgentKeyInfo(
        key="sk-or-v1-test",
        key_hash="abc123",
        name="test-agent",
        limit_usd=10.0,
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
    assert restored.limit_usd == 10.0


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
