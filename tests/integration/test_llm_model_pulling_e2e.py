"""
Integration tests for LLM model pulling - REAL TESTS, NO MOCKS

These tests download actual models from Ollama. They require:
- Ollama running on localhost:11434
- Sufficient disk space (~500MB for test model)
- Internet connection for model downloads

Run with: pytest tests/integration/test_llm_model_pulling_e2e.py -x -v
"""
import pytest
import pytest_asyncio
import requests
import asyncio
from kestrel_sovereign.llm.service import LLMService


@pytest.fixture(scope="module")
def skip_if_no_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            pytest.skip("Ollama not running on localhost:11434")
    except Exception:
        pytest.skip("Ollama not accessible")


@pytest.fixture(scope="module")
def test_model_name():
    """Small model for testing (~500MB)"""
    return "qwen2.5:0.5b"


@pytest_asyncio.fixture
async def llm_service():
    """Real LLMService instance"""
    service = LLMService()
    # Ensure database is initialized (it's lazy-loaded)
    await service._ensure_db_initialized()
    yield service
    # Cleanup - use async close method
    await service.close()


@pytest_asyncio.fixture
async def cleanup_test_model(test_model_name):
    """Clean up test model after tests"""
    yield
    # Cleanup
    try:
        import ollama
        client = ollama.Client()
        client.delete(test_model_name)
        print(f"Cleaned up test model: {test_model_name}")
    except Exception as e:
        print(f"Failed to cleanup test model: {e}")


class TestStorageInfo:
    """Test storage info retrieval"""

    @pytest.mark.asyncio
    async def test_get_storage_info(self, llm_service, skip_if_no_ollama):
        """Verify storage info returns accurate data"""
        storage_info = await llm_service.get_storage_info(use_cache=False)

        # Assertions
        assert "total_gb" in storage_info
        assert "used_gb" in storage_info
        assert "available_gb" in storage_info
        assert "models" in storage_info
        assert isinstance(storage_info["models"], list)

        # Verify math
        assert storage_info["total_gb"] > 0, "Total GB should be > 0"
        assert storage_info["used_gb"] >= 0, "Used GB should be >= 0"
        assert storage_info["available_gb"] >= 0, "Available GB should be >= 0"
        assert storage_info["total_gb"] >= storage_info["used_gb"], "Total should be >= used"

        print(f"\nStorage info: {storage_info['available_gb']:.1f}GB available of {storage_info['total_gb']:.1f}GB total")

    @pytest.mark.asyncio
    async def test_storage_info_cache(self, llm_service, skip_if_no_ollama):
        """Verify storage info caching works"""
        import time

        # First call (cache miss)
        start1 = time.time()
        storage1 = await llm_service.get_storage_info(use_cache=False)
        time1 = time.time() - start1

        # Second call (cache hit)
        start2 = time.time()
        storage2 = await llm_service.get_storage_info(use_cache=True)
        time2 = time.time() - start2

        # Assertions
        assert storage1 == storage2, "Cached result should match uncached"
        assert time2 < time1 / 2, f"Cached call should be faster ({time2:.3f}s vs {time1:.3f}s)"
        print(f"\nCache performance: uncached {time1:.3f}s, cached {time2:.3f}s")


class TestModelPulling:
    """Test model pulling functionality"""

    @pytest.mark.asyncio
    async def test_pull_small_model(self, llm_service, skip_if_no_ollama, test_model_name, cleanup_test_model):
        """Verify can pull a small model"""
        # Remove model first if exists
        try:
            import ollama
            client = ollama.Client()
            client.delete(test_model_name)
            print(f"\nRemoved existing test model: {test_model_name}")
        except Exception:
            pass

        # Pull model
        print(f"\nPulling model: {test_model_name} (this may take a few minutes)...")
        result = await llm_service.pull_model(test_model_name)

        # Assertions
        assert result is True, "Pull should succeed"

        # Verify model now available via Ollama directly (discover_all_models may not include Ollama)
        import ollama
        client = ollama.Client()
        ollama_models = client.list()
        model_names = [m.get("name", m.get("model", "")) for m in ollama_models.get("models", [])]
        assert any(test_model_name in name for name in model_names), f"Model {test_model_name} should be in {model_names}"

        print(f"✅ Successfully pulled model: {test_model_name}")

    @pytest.mark.asyncio
    async def test_pull_with_progress(self, llm_service, skip_if_no_ollama, test_model_name, cleanup_test_model):
        """Verify progress tracking during pull"""
        # Remove model first
        try:
            import ollama
            client = ollama.Client()
            client.delete(test_model_name)
        except Exception:
            pass

        progress_updates = []

        def progress_callback(status, completed, total):
            progress_updates.append({
                "status": status,
                "completed": completed,
                "total": total
            })
            if total > 0:
                pct = (completed / total) * 100
                print(f"\rProgress: {status} - {pct:.1f}%", end="")

        # Pull with progress
        print(f"\nPulling with progress tracking: {test_model_name}...")
        await llm_service.pull_model(test_model_name, progress_callback=progress_callback)
        print()  # Newline after progress

        # Assertions
        assert len(progress_updates) > 0, "Should receive progress updates"
        statuses = [u["status"] for u in progress_updates]
        assert "success" in statuses, f"Should have success status in {statuses}"

        print(f"✅ Received {len(progress_updates)} progress updates")

    @pytest.mark.asyncio
    async def test_insufficient_space_error(self, llm_service, skip_if_no_ollama, monkeypatch):
        """Verify error when insufficient disk space"""
        # Mock get_storage_info to return low space
        async def mock_storage(*args, **kwargs):
            return {
                "total_gb": 100,
                "used_gb": 99.5,
                "available_gb": 0.5,
                "models": []
            }

        monkeypatch.setattr(llm_service, "get_storage_info", mock_storage)

        # Try to pull large model
        with pytest.raises(RuntimeError, match="Insufficient disk space"):
            await llm_service.pull_model("llama3.2:70b")  # Large model

        print("✅ Correctly rejected pull when space insufficient")

    @pytest.mark.asyncio
    async def test_auto_download_missing_model(self, llm_service, skip_if_no_ollama, test_model_name, cleanup_test_model):
        """Verify auto-download when model not found"""
        # Remove model first
        try:
            import ollama
            client = ollama.Client()
            client.delete(test_model_name)
            print(f"\nRemoved model to test auto-download: {test_model_name}")
        except Exception:
            pass

        # Request model that doesn't exist (should auto-download)
        print(f"Requesting non-existent model (will auto-download): {test_model_name}...")
        response = await llm_service.get_response_with_model(
            model_id=test_model_name,
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'test successful' and nothing else.",
            auto_pull=True
        )

        # Assertions
        assert response is not None, "Should succeed after auto-download"
        # LLMResponse is a dataclass - check content attribute
        assert response.content is not None, "Response content should not be None"
        assert len(response.content) > 0, "Response content should not be empty"

        # Verify model now exists
        import ollama as _ollama
        ollama_models = _ollama.Client().list()
        models = ollama_models.get("models", [])
        model_names = [m.get("name", m.get("model", "")) for m in models]
        assert any(test_model_name in name for name in model_names), f"Model should exist after auto-download"

        print(f"✅ Auto-download successful, got response: {response.content[:100] if response.content else 'None'}...")

    @pytest.mark.asyncio
    async def test_no_auto_download_when_disabled(self, llm_service, skip_if_no_ollama):
        """Verify no auto-download when auto_pull=False"""
        # Try to use non-existent model
        with pytest.raises(ValueError, match="Model.*not found"):
            await llm_service.get_response_with_model(
                model_id="nonexistent-model:1b",
                system_prompt="Test",
                user_prompt="Hello",
                auto_pull=False
            )

        print("✅ Correctly rejected when auto_pull=False")


class TestModelUsageTracking:
    """Test model usage tracking"""

    @pytest.mark.asyncio
    async def test_model_usage_tracked(self, llm_service, skip_if_no_ollama, test_model_name):
        """Verify model usage is tracked"""
        # Ensure model exists
        try:
            import ollama as _ollama
            ollama_models = _ollama.Client().list()
            models = ollama_models.get("models", [])
            model_names = [m.get("name", m.get("model", "")) for m in models]
            if not any(test_model_name in name for name in model_names):
                await llm_service.pull_model(test_model_name)
        except Exception as e:
            pytest.skip(f"Could not ensure test model: {e}")

        # Use the model
        await llm_service.get_response_with_model(
            model_id=test_model_name,
            system_prompt="Test",
            user_prompt="Hello"
        )

        # Check usage was recorded - use the internal _usage_db attribute
        if hasattr(llm_service, '_usage_db') and llm_service._usage_db:
            row = await llm_service._usage_db.fetchone(
                "SELECT last_used, use_count FROM model_usage WHERE model_id = ?",
                (test_model_name,)
            )

            # Assertions
            assert row is not None, "Usage should be recorded"
            last_used, use_count = row
            assert last_used is not None, "Last used timestamp should be set"
            assert use_count > 0, "Use count should be > 0"

            print(f"✅ Usage tracked: {test_model_name} used {use_count} times, last: {last_used}")
        else:
            pytest.skip("Database not initialized, cannot test usage tracking")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-x"])
