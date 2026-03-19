"""
Tests for LLM model discovery functionality.
Tests REAL API calls to verify model discovery works.
"""
import pytest
import pytest_asyncio
import asyncio
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.model_metadata import ModelInfo, ModelCategory


class TestModelDiscovery:
    """Test model discovery from all providers"""

    @pytest_asyncio.fixture
    async def llm_service(self):
        """Create an LLM service instance"""
        service = LLMService()
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_discover_all_models_returns_model_info(self, llm_service):
        """Test that discover_all_models returns List[ModelInfo]"""
        models = await llm_service.discover_all_models(use_cache=False)

        assert len(models) > 0, "No models discovered"
        assert all(isinstance(m, ModelInfo) for m in models), "Not all items are ModelInfo"

        print(f"✅ Discovered {len(models)} models")

    @pytest.mark.asyncio
    async def test_discover_all_models_has_providers(self, llm_service):
        """Test that models are discovered from multiple providers"""
        models = await llm_service.discover_all_models(use_cache=False)

        providers = set(m.provider for m in models)

        # Should have at least OpenAI (API key usually available)
        # Ollama may not be running
        assert len(providers) > 0, "No providers discovered"

        print(f"✅ Providers discovered: {', '.join(providers)}")

        # Print breakdown
        for provider in providers:
            count = len([m for m in models if m.provider == provider])
            print(f"   - {provider}: {count} models")

    @pytest.mark.asyncio
    async def test_discover_all_models_with_featured_filter(self, llm_service):
        """Test featured_only filter"""
        all_models = await llm_service.discover_all_models(
            use_cache=False,
            featured_only=False
        )
        featured_models = await llm_service.discover_all_models(
            use_cache=False,
            featured_only=True
        )

        # Featured should be subset of all (or equal if all are featured)
        assert len(featured_models) <= len(all_models)

        # All featured models should have is_featured=True
        assert all(m.is_featured for m in featured_models), "Not all featured models have is_featured=True"

        print(f"✅ All models: {len(all_models)}, Featured: {len(featured_models)}")

    @pytest.mark.asyncio
    async def test_discover_all_models_with_category_filter(self, llm_service):
        """Test category filter"""
        chat_models = await llm_service.discover_all_models(
            use_cache=False,
            featured_only=False,
            category=ModelCategory.CHAT
        )

        if chat_models:
            assert all(m.category == ModelCategory.CHAT for m in chat_models)
            print(f"✅ Found {len(chat_models)} chat models")
        else:
            print("⚠️  No chat models found")

    @pytest.mark.asyncio
    async def test_discover_all_models_with_provider_filter(self, llm_service):
        """Test provider filter"""
        all_models = await llm_service.discover_all_models(
            use_cache=False,
            featured_only=False
        )

        providers = list(set(m.provider for m in all_models))
        if providers:
            # Filter to first provider only
            filtered = await llm_service.discover_all_models(
                use_cache=False,
                featured_only=False,
                providers=[providers[0]]
            )

            assert all(m.provider == providers[0] for m in filtered)
            print(f"✅ Filtered to {providers[0]}: {len(filtered)} models")

    @pytest.mark.asyncio
    async def test_discover_all_models_with_cache(self, llm_service):
        """Test model discovery with caching"""
        from kestrel_sovereign.llm.model_cache import get_shared_model_cache

        # First call - should fetch fresh
        models1 = await llm_service.discover_all_models(use_cache=True)
        shared_cache = get_shared_model_cache()
        assert shared_cache.has_data()

        # Second call - should use cache
        models2 = await llm_service.discover_all_models(use_cache=True)

        # Should return same models
        assert len(models1) == len(models2)

        print(f"✅ Model caching working, cached {len(models1)} models")

    def test_set_default_model(self, llm_service):
        """Test setting default model"""
        original = llm_service.default_model

        # Set new default
        llm_service.set_default_model("gpt-5-mini")
        assert llm_service.default_model == "gpt-5-mini"

        # Reset
        llm_service.set_default_model(original)
        assert llm_service.default_model == original

        print(f"✅ Default model setting works")


class TestModelInfoStructure:
    """Test ModelInfo structure from discovered models"""

    @pytest_asyncio.fixture
    async def llm_service(self):
        """Create an LLM service instance"""
        service = LLMService()
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_model_has_required_fields(self, llm_service):
        """Test that discovered models have required fields"""
        models = await llm_service.discover_all_models(use_cache=False)

        if models:
            model = models[0]
            assert model.id is not None
            assert model.provider is not None
            assert model.display_name is not None
            assert model.category is not None

            print(f"✅ Model structure valid: {model.id}")

    @pytest.mark.asyncio
    async def test_model_serialization(self, llm_service):
        """Test that models can be serialized to dict"""
        models = await llm_service.discover_all_models(use_cache=False)

        if models:
            model = models[0]
            d = model.to_dict()

            assert "id" in d
            assert "provider" in d
            assert "display_name" in d
            assert "category" in d

            print(f"✅ Model serialization works: {d['id']}")


class TestLegacyCompatibility:
    """Test backward compatibility with old API"""

    @pytest_asyncio.fixture
    async def llm_service(self):
        """Create an LLM service instance"""
        service = LLMService()
        yield service
        await service.close()

    def test_list_available_models_still_works(self, llm_service):
        """Test that old list_available_models method still works"""
        models = llm_service.list_available_models()

        # Should return a list of configured models
        assert isinstance(models, list)

        print(f"✅ Legacy list_available_models works: {len(models)} models")


if __name__ == "__main__":
    # Run tests directly
    print("Running model discovery tests...")
    print("=" * 60)

    async def run_all():
        service = LLMService()
        try:
            test = TestModelDiscovery()

            print("\n1. Testing model discovery returns ModelInfo...")
            await test.test_discover_all_models_returns_model_info(service)

            print("\n2. Testing provider discovery...")
            await test.test_discover_all_models_has_providers(service)

            print("\n3. Testing featured filter...")
            await test.test_discover_all_models_with_featured_filter(service)

            print("\n4. Testing caching...")
            await test.test_discover_all_models_with_cache(service)

            print("\n5. Testing default model setting...")
            test.test_set_default_model(service)

            print("\n" + "=" * 60)
        finally:
            await service.close()
        print("✅ All tests completed!")

    asyncio.run(run_all())
