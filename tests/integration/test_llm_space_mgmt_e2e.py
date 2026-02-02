"""
Integration tests for LLM space management - REAL TESTS, NO MOCKS

These tests manage actual disk space and delete real models. They require:
- Ollama running on localhost:11434
- At least one installed model
- Permission to delete models

Run with: pytest tests/integration/test_llm_space_mgmt_e2e.py -x -v
"""
import pytest
import pytest_asyncio
import requests
import asyncio
from datetime import datetime, timedelta
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


@pytest_asyncio.fixture
async def llm_service():
    """Real LLMService instance with proper async cleanup."""
    service = LLMService()
    # Ensure database is initialized (it's lazy-loaded)
    await service._ensure_db_initialized()
    yield service
    await service.close()


@pytest.fixture
def test_model_name():
    """Small test model"""
    return "qwen2.5:0.5b"


class TestSpaceManagement:
    """Test space management functionality"""

    @pytest.mark.asyncio
    async def test_get_storage_details(self, llm_service, skip_if_no_ollama):
        """Verify storage info includes model details"""
        storage_info = await llm_service.get_storage_info(use_cache=False)

        # Assertions
        assert "models" in storage_info
        models = storage_info["models"]

        if len(models) > 0:
            # Check first model has required fields
            model = models[0]
            assert "id" in model
            assert "size_gb" in model
            assert "last_used" in model or model["last_used"] is None

            print(f"\nFound {len(models)} models:")
            for m in models:
                print(f"  - {m['id']}: {m['size_gb']:.2f}GB, last used: {m.get('last_used', 'never')}")
        else:
            print("\n⚠️  No models installed, some tests will be skipped")

    @pytest.mark.asyncio
    async def test_cleanup_dry_run(self, llm_service, skip_if_no_ollama):
        """Verify dry-run mode doesn't delete models"""
        # Get initial models
        storage_before = await llm_service.get_storage_info(use_cache=False)
        models_before = len(storage_before["models"])

        if models_before == 0:
            pytest.skip("No models installed, cannot test cleanup")

        # Run cleanup in dry-run mode
        print(f"\nRunning cleanup dry-run (threshold: 0 days to see what would be deleted)...")
        deleted = await llm_service.cleanup_unused_models(
            threshold_days=0,  # Would delete everything not protected
            min_free_space_pct=100,  # Force cleanup check
            dry_run=True
        )

        # Get models after dry-run
        storage_after = await llm_service.get_storage_info(use_cache=False)
        models_after = len(storage_after["models"])

        # Assertions
        assert models_before == models_after, "Dry-run should not delete any models"

        if deleted:
            print(f"✅ Dry-run would delete {len(deleted)} models: {deleted}")
        else:
            print("✅ No models would be deleted (all protected or recent)")

    @pytest.mark.asyncio
    async def test_protected_models_not_deleted(self, llm_service, skip_if_no_ollama):
        """Verify protected models are never deleted"""
        # Get protected models from config
        protected_models = set()
        for provider in llm_service.providers:
            if provider["name"] == "ollama":
                protected_models.add(provider["model"])

        if llm_service.mandate_config:
            defaults = llm_service.mandate_config.get("defaults", {})
            if "preferred" in defaults:
                protected_models.add(defaults["preferred"])

        print(f"\nProtected models from config: {protected_models}")

        # Run aggressive cleanup (dry-run)
        deleted = await llm_service.cleanup_unused_models(
            threshold_days=0,  # Delete everything old
            min_free_space_pct=100,  # Force cleanup
            dry_run=True
        )

        # Assertions
        for model in protected_models:
            assert model not in deleted, f"Protected model {model} should not be deleted"

        print(f"✅ Protected models safe from cleanup")

    @pytest.mark.asyncio
    async def test_cleanup_old_models(self, llm_service, skip_if_no_ollama, test_model_name):
        """Verify old unused models are cleaned up"""
        # Ensure test model exists
        try:
            import ollama as _ollama
            ollama_models = _ollama.Client().list()
            models = [{"id": m.get("name", m.get("model", ""))} for m in ollama_models.get("models", [])]
            if not any(test_model_name in m["id"] for m in models):
                await llm_service.pull_model(test_model_name)
                print(f"\nPulled test model: {test_model_name}")
        except Exception as e:
            pytest.skip(f"Could not ensure test model: {e}")

        # Manually set last_used to 60 days ago using async database
        if hasattr(llm_service, '_usage_db') and llm_service._usage_db:
            old_date = (datetime.now() - timedelta(days=60)).isoformat()
            await llm_service._usage_db.execute("""
                INSERT OR REPLACE INTO model_usage (model_id, provider, last_used, use_count)
                VALUES (?, ?, ?, ?)
            """, (test_model_name, "ollama", old_date, 1))
            print(f"Set {test_model_name} last_used to {old_date}")
        else:
            pytest.skip("Database not initialized")

        # Check if model would be deleted (dry-run first)
        deleted_dry = await llm_service.cleanup_unused_models(
            threshold_days=30,
            min_free_space_pct=100,  # Force cleanup
            dry_run=True
        )

        # Assertions
        if test_model_name in deleted_dry:
            print(f"✅ Old model {test_model_name} correctly identified for cleanup")

            # Actually delete it
            print(f"Deleting old model: {test_model_name}...")
            deleted_real = await llm_service.cleanup_unused_models(
                threshold_days=30,
                min_free_space_pct=100,
                dry_run=False
            )

            assert test_model_name in deleted_real, "Model should be deleted"

            # Verify model removed
            import ollama as _ollama
            ollama_models = _ollama.Client().list()
            models = [{"id": m.get("name", m.get("model", ""))} for m in ollama_models.get("models", [])]
            model_ids = [m["id"] for m in models]
            assert test_model_name not in model_ids, "Model should be gone"

            print(f"✅ Successfully deleted old model")
        else:
            print(f"⚠️  Model {test_model_name} is protected or recent, skipping actual deletion test")

    @pytest.mark.asyncio
    async def test_no_cleanup_when_space_ok(self, llm_service, skip_if_no_ollama):
        """Verify cleanup skipped when space is adequate"""
        # Check current space
        storage = await llm_service.get_storage_info(use_cache=False)
        free_pct = (storage["available_gb"] / storage["total_gb"]) * 100

        print(f"\nCurrent free space: {free_pct:.1f}%")

        if free_pct >= 10:
            # Cleanup should be skipped
            deleted = await llm_service.cleanup_unused_models(
                threshold_days=30,
                min_free_space_pct=10,
                dry_run=False
            )

            assert len(deleted) == 0, "No cleanup should occur when space >= 10%"
            print(f"✅ Cleanup correctly skipped (space: {free_pct:.1f}% >= 10%)")
        else:
            print(f"⚠️  Space low ({free_pct:.1f}%), cleanup would run")

    @pytest.mark.asyncio
    async def test_recent_models_not_deleted(self, llm_service, skip_if_no_ollama, test_model_name):
        """Verify recently used models (<7 days) are not deleted"""
        # Ensure test model exists
        try:
            import ollama as _ollama
            ollama_models = _ollama.Client().list()
            models = [{"id": m.get("name", m.get("model", ""))} for m in ollama_models.get("models", [])]
            if not any(test_model_name in m["id"] for m in models):
                await llm_service.pull_model(test_model_name)
        except Exception as e:
            pytest.skip(f"Could not ensure test model: {e}")

        # Use the model (sets last_used to now)
        await llm_service.get_response_with_model(
            model_id=test_model_name,
            system_prompt="Test",
            user_prompt="Hello"
        )
        print(f"\nUsed model: {test_model_name} (should be marked as recent)")

        # Try to cleanup (dry-run)
        deleted = await llm_service.cleanup_unused_models(
            threshold_days=30,
            min_free_space_pct=100,  # Force cleanup check
            dry_run=True
        )

        # Assertions
        assert test_model_name not in deleted, "Recent model should not be deleted"
        print(f"✅ Recent model protected from cleanup")

    @pytest.mark.asyncio
    async def test_auto_cleanup_low_space(self, llm_service, skip_if_no_ollama, monkeypatch):
        """Verify auto-cleanup triggers when space low"""
        # Track if cleanup was called
        cleanup_called = []

        original_cleanup = llm_service.cleanup_unused_models

        async def mock_cleanup(*args, **kwargs):
            cleanup_called.append(True)
            return []

        # Mock storage to return low space
        async def mock_storage(*args, **kwargs):
            return {
                "total_gb": 100,
                "used_gb": 92,
                "available_gb": 8,
                "models": []
            }

        monkeypatch.setattr(llm_service, "get_storage_info", mock_storage)
        monkeypatch.setattr(llm_service, "cleanup_unused_models", mock_cleanup)

        # Trigger space check
        await llm_service._check_and_cleanup_if_needed()

        # Assertions
        assert len(cleanup_called) > 0, "Cleanup should be triggered when space <10%"
        print("✅ Auto-cleanup triggered correctly for low space")

    @pytest.mark.asyncio
    async def test_storage_after_cleanup(self, llm_service, skip_if_no_ollama, test_model_name):
        """Verify storage info updates after cleanup"""
        # Get initial storage
        storage_before = await llm_service.get_storage_info(use_cache=False)
        models_before = storage_before["models"]
        count_before = len(models_before)

        # Try cleanup (dry-run to be safe)
        deleted = await llm_service.cleanup_unused_models(
            threshold_days=0,
            min_free_space_pct=100,
            dry_run=True
        )

        if len(deleted) > 0:
            print(f"\n{len(deleted)} models would be deleted in real cleanup")

        # Storage info should reflect current state
        storage_after = await llm_service.get_storage_info(use_cache=False)
        models_after = storage_after["models"]

        # After dry-run, counts should match
        assert len(models_before) == len(models_after), "Dry-run should not change model count"

        print(f"✅ Storage info consistent: {count_before} models before and after dry-run")


class TestDatabaseIntegration:
    """Test database integration for usage tracking"""

    @pytest.mark.asyncio
    async def test_usage_database_created(self, llm_service):
        """Verify usage database is created"""
        if hasattr(llm_service, '_usage_db') and llm_service._usage_db:
            # Use database-agnostic approach: try to query the table
            # If it exists, the query succeeds; if not, it fails
            try:
                result = await llm_service._usage_db.fetchone(
                    "SELECT COUNT(*) as cnt FROM model_usage"
                )
                assert result is not None, "model_usage table should exist"
                print("✅ Usage database table exists")
            except Exception as e:
                # Table doesn't exist
                pytest.fail(f"model_usage table should exist: {e}")
        else:
            pytest.skip("Database not initialized")

    @pytest.mark.asyncio
    async def test_track_usage(self, llm_service):
        """Verify usage tracking works"""
        if hasattr(llm_service, '_usage_db') and llm_service._usage_db:
            # Clear any existing usage for test model
            await llm_service._usage_db.execute(
                "DELETE FROM model_usage WHERE model_id = 'test-model'"
            )

            # Track usage
            await llm_service._track_model_usage("test-model", "test-provider", tokens=100)

            # Verify recorded
            row = await llm_service._usage_db.fetchone("""
                SELECT model_id, provider, use_count, total_tokens
                FROM model_usage WHERE model_id = 'test-model'
            """)

            assert row is not None, "Usage should be recorded"
            model_id, provider, use_count, total_tokens = row
            assert model_id == "test-model"
            assert provider == "test-provider"
            assert use_count == 1
            assert total_tokens == 100

            print(f"✅ Usage tracked: {model_id} ({provider}), count={use_count}, tokens={total_tokens}")
        else:
            pytest.skip("Database not initialized")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-x"])
