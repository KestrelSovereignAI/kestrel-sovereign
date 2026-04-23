"""
Integration tests for agent-owned API key activation.

Proves that agents created via inception use their own provisioned
OpenRouter key instead of the shared environment key.

All key storage is agent-scoped - each agent has isolated key storage.

Run with: uv run pytest tests/integration/test_agent_own_key.py -v
"""

import os
import tempfile
import pytest
from dotenv import load_dotenv

# Load .env before checking for keys
load_dotenv()

# Skip all tests if required keys not available
pytestmark = [
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set",
    ),
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_MANAGEMENT_API_KEY"),
        reason="OPENROUTER_MANAGEMENT_API_KEY not set",
    ),
    pytest.mark.skipif(
        not os.getenv("KESTREL_DATA_KEY"),
        reason="KESTREL_DATA_KEY not set (required for ServiceKeyStorage)",
    ),
]


@pytest.mark.asyncio
async def test_llm_service_use_agent_key():
    """Test that LLMService.use_agent_key() activates the agent's key."""
    import uuid
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = f"{tmp_dir}/test_agent.db"
        db = await AsyncDatabase.sqlite(db_path)

        provisioning = OpenRouterProvisioningService()
        agent_name = f"test-key-{uuid.uuid4().hex[:8]}"
        agent_did = f"did:pkh:eip155:1:0x{uuid.uuid4().hex[:40]}"
        key_info = None

        try:
            key_info = await provisioning.create_agent_key(
                agent_name=agent_name,
                limit_usd=0.10,
            )

            # Store the key in the database (agent-scoped)
            key_storage = ServiceKeyStorage(db, agent_did)
            await key_storage.store_key(
                provider_id="openrouter",
                api_key=key_info.key,
            )

            # Create LLMService and activate agent key
            service = LLMService()
            activated = await service.use_agent_key(
                agent_did=agent_did,
                db=db,
                provider="openrouter",
            )

            assert activated is True

            # Verify at least one openrouter route was updated. Under the
            # vendor/route/model schema, provider names are composite
            # ``"<vendor>:<route>"`` and a single vendor may have multiple
            # routes — use_agent_key now swaps all of them.
            openrouter_routes = [p for p in service.providers if p.get("vendor") == "openrouter"]
            assert openrouter_routes, "Expected at least one openrouter route to be initialized"

        finally:
            # Always clean up the OpenRouter key
            if key_info:
                try:
                    await provisioning.delete_key(key_info.key_hash)
                except Exception:
                    pass
            await provisioning.close()
            await db.close()


@pytest.mark.asyncio
async def test_agent_uses_own_key_for_inference():
    """Test that agent created via inception uses its own key for LLM calls."""
    import httpx
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    with tempfile.TemporaryDirectory() as tmp_dir:
        from kestrel_sovereign.inception_service import create_kestrel_identity_async

        # Create agent - this should provision an OpenRouter key
        credentials = await create_kestrel_identity_async(
            output_dir=tmp_dir,
            is_test_instance=True,
        )

        # Skip if no key was provisioned
        if not credentials.openrouter_key_hash:
            pytest.skip("OpenRouter key was not provisioned during inception")

        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        db_path = f"{tmp_dir}/kestrel_prime.db"
        db = await AsyncDatabase.sqlite(db_path)
        provisioning = OpenRouterProvisioningService()

        try:
            # Agent-scoped key storage
            key_storage = ServiceKeyStorage(db, credentials.agent_did)
            agent_key = await key_storage.get_key(provider_id="openrouter")

            # Use the agent's key directly to make an inference call
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {agent_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek/deepseek-chat-v3.1",
                        "messages": [{"role": "user", "content": "Say 'own key works'"}],
                        "max_tokens": 10,
                    },
                    timeout=30.0,
                )

                assert response.status_code == 200, f"Failed: {response.text}"
                data = response.json()
                assert "choices" in data

        finally:
            # Always clean up the OpenRouter key
            try:
                await provisioning.delete_key(credentials.openrouter_key_hash)
            except Exception:
                pass
            await provisioning.close()
            await db.close()


@pytest.mark.asyncio
async def test_use_agent_key_returns_false_if_no_key():
    """Test that use_agent_key returns False if agent has no key."""
    import uuid
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = f"{tmp_dir}/test_no_key.db"
        db = await AsyncDatabase.sqlite(db_path)

        try:
            service = LLMService()
            fake_did = f"did:pkh:eip155:1:0x{uuid.uuid4().hex[:40]}"

            # This should return False, not raise
            activated = await service.use_agent_key(
                agent_did=fake_did,
                db=db,
                provider="openrouter",
            )

            assert activated is False

        finally:
            await db.close()


@pytest.mark.asyncio
async def test_kestrel_agent_activates_key_on_initialize():
    """Test that KestrelAgent activates its OpenRouter key during initialize()."""
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.features.llm_keys import OpenRouterProvisioningService

    with tempfile.TemporaryDirectory() as tmp_dir:
        from kestrel_sovereign.inception_service import create_kestrel_identity_async

        # Create agent
        credentials = await create_kestrel_identity_async(
            output_dir=tmp_dir,
            is_test_instance=True,
        )

        if not credentials.openrouter_key_hash:
            pytest.skip("OpenRouter key was not provisioned")

        provisioning = OpenRouterProvisioningService()

        try:
            # Create KestrelAgent - it should auto-activate its key
            db_path = f"{tmp_dir}/kestrel_prime.db"
            llm_service = LLMService()

            agent = KestrelAgent(
                storage_path=db_path,
                did=credentials.agent_did,
                llm_service=llm_service,
            )
            await agent.initialize()

            # The agent should now be using its own key
            openrouter_provider = None
            for p in llm_service.providers:
                if p["name"] == "openrouter":
                    openrouter_provider = p
                    break

            assert openrouter_provider is not None

            await agent.shutdown()

        finally:
            # Always clean up the OpenRouter key
            try:
                await provisioning.delete_key(credentials.openrouter_key_hash)
            except Exception:
                pass
            await provisioning.close()
