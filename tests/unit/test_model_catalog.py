"""
Unit tests for ModelCatalogService in model_catalog.py
"""
import json
import pytest
import tempfile
from pathlib import Path

from kestrel_sovereign.llm.model_catalog import ModelCatalogService, get_catalog_service
from kestrel_sovereign.llm.model_metadata import ModelInfo, ModelCategory


class TestModelCatalogServiceInit:
    """Test ModelCatalogService initialization"""

    def test_init_with_default_path(self):
        """Test initialization with default config path"""
        service = ModelCatalogService()
        assert service.config_path.name == "model_catalog.toml"
        assert not service._loaded

    def test_init_with_custom_path(self):
        """Test initialization with custom config path"""
        custom_path = Path("/tmp/custom_catalog.toml")
        service = ModelCatalogService(config_path=custom_path)
        assert service.config_path == custom_path

    def test_load_missing_file_doesnt_crash(self):
        """Test that loading a missing file doesn't crash"""
        service = ModelCatalogService(config_path=Path("/nonexistent/file.toml"))
        service.load()  # Should not raise
        assert service._loaded


class TestModelCatalogServiceLoad:
    """Test ModelCatalogService loading from TOML"""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config file with the new override-only format"""
        content = '''
[hidden]
openai = ["gpt-4-internal"]

[categories.embedding]
openai = ["text-embedding-3-large", "text-embedding-3-small"]

[categories.image]
openai = ["dall-e-3"]

[context_limits_override]
"gpt-5.1" = 256000
"gpt-4" = 8192

[display_name_overrides]
"gpt-5.1" = "GPT-5.1 (Latest)"
"claude-sonnet-4-5-20250929" = "Claude Sonnet 4.5"
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            yield Path(f.name)

    def test_load_hidden_models(self, sample_config):
        """Test loading hidden models"""
        service = ModelCatalogService(config_path=sample_config)
        service.load()

        assert service.is_hidden("openai", "gpt-4-internal")
        assert not service.is_hidden("openai", "gpt-5.1")

    def test_load_display_name_overrides(self, sample_config):
        """Test loading display name overrides"""
        service = ModelCatalogService(config_path=sample_config)
        service.load()

        assert service.get_display_name("gpt-5.1") == "GPT-5.1 (Latest)"
        assert service.get_display_name("unknown-model") == "unknown-model"
        assert service.get_display_name("unknown-model", "Default") == "Default"

    def test_load_categories(self, sample_config):
        """Test loading model categories"""
        service = ModelCatalogService(config_path=sample_config)
        service.load()

        assert service.get_category("openai", "text-embedding-3-large") == ModelCategory.EMBEDDING
        assert service.get_category("openai", "dall-e-3") == ModelCategory.IMAGE
        assert service.get_category("openai", "gpt-5.1") == ModelCategory.CHAT  # default

    def test_category_explicit_matching_only(self, sample_config):
        """Test that get_category only matches explicitly listed models."""
        service = ModelCatalogService(config_path=sample_config)
        service.load()

        assert service.get_category("openai", "text-embedding-3-large") == ModelCategory.EMBEDDING
        assert service.get_category("openai", "text-embedding-3-large:latest") == ModelCategory.CHAT
        assert service.get_category("openai", "gpt-5.1:turbo") == ModelCategory.CHAT

    def test_load_context_limits_override(self, sample_config):
        """Test loading context limits from override section"""
        service = ModelCatalogService(config_path=sample_config)
        service.load()

        assert service.get_context_limit("gpt-5.1") == 256000
        assert service.get_context_limit("gpt-4") == 8192

    def test_backward_compat_context_limits_key(self):
        """Test backward compatibility with old [context_limits] key name"""
        content = '''
[context_limits]
"gpt-4" = 8192
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            service = ModelCatalogService(config_path=Path(f.name))
            service.load()
            assert service.get_context_limit("gpt-4") == 8192

    def test_backward_compat_display_names_key(self):
        """Test backward compatibility with old [display_names] key name"""
        content = '''
[display_names]
"gpt-5.1" = "GPT-5.1 Legacy"
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            service = ModelCatalogService(config_path=Path(f.name))
            service.load()
            assert service.get_display_name("gpt-5.1") == "GPT-5.1 Legacy"

    def test_backward_compat_featured_section(self):
        """Test that legacy [featured] section still works (OR behavior)"""
        content = '''
[featured]
openai = ["gpt-5.1", "gpt-5-mini"]
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            service = ModelCatalogService(config_path=Path(f.name))
            service.load()
            assert service.is_featured("openai", "gpt-5.1")
            assert not service.is_featured("openai", "gpt-4")


class TestModelCatalogServiceEnrich:
    """Test model enrichment functionality"""

    @pytest.fixture
    def service(self):
        """Create a service with sample config"""
        content = '''
[hidden]
openai = ["gpt-4-internal"]

[categories.embedding]
openai = ["text-embedding-3-large"]

[display_name_overrides]
"gpt-5.1" = "GPT-5.1 (Latest)"
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()
            yield svc

    def test_enrich_preserves_featured_status(self, service):
        """Test that enrichment preserves existing is_featured (OR behavior)"""
        model = ModelInfo(id="gpt-5.1", provider="openai", display_name="GPT-5.1", is_featured=True)
        enriched = service.enrich_model(model)

        # Featured status preserved even though gpt-5.1 is not in [featured] section
        assert enriched.is_featured is True
        assert enriched.is_hidden is False
        assert enriched.display_name == "GPT-5.1 (Latest)"

    def test_enrich_unfeatured_stays_unfeatured(self, service):
        """Test that unfeatured models stay unfeatured when no [featured] section"""
        model = ModelInfo(id="gpt-3.5-turbo", provider="openai", display_name="GPT-3.5", is_featured=False)
        enriched = service.enrich_model(model)
        assert enriched.is_featured is False

    def test_enrich_hidden_model(self, service):
        """Test enriching a hidden model"""
        model = ModelInfo(id="gpt-4-internal", provider="openai", display_name="Internal")
        enriched = service.enrich_model(model)

        assert enriched.is_hidden is True

    def test_enrich_embedding_model(self, service):
        """Test enriching an embedding model explicitly listed in catalog"""
        model = ModelInfo(id="text-embedding-3-large", provider="openai", display_name="Embedding")
        enriched = service.enrich_model(model)

        assert enriched.category == ModelCategory.EMBEDDING

    def test_enrich_preserves_adapter_detected_category(self, service):
        """Test that enrich_model preserves category for models not in catalog."""
        model = ModelInfo(
            id="nomic-embed-text:latest",
            provider="ollama",
            display_name="Nomic Embed",
            category=ModelCategory.EMBEDDING
        )
        enriched = service.enrich_model(model)
        assert enriched.category == ModelCategory.EMBEDDING

    def test_enrich_multiple_models(self, service):
        """Test enriching multiple models at once"""
        models = [
            ModelInfo(id="gpt-5.1", provider="openai", display_name="GPT", is_featured=True),
            ModelInfo(id="text-embedding-3-large", provider="openai", display_name="Embed"),
        ]
        enriched = service.enrich_models(models)

        assert len(enriched) == 2
        assert enriched[0].is_featured is True
        assert enriched[1].category == ModelCategory.EMBEDDING

    def test_enrich_featured_or_behavior_with_legacy_section(self):
        """Test that legacy [featured] section adds featured but doesn't remove it."""
        content = '''
[featured]
openai = ["gpt-5.1"]
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()

        # Model in [featured]: gets featured
        model_in = ModelInfo(id="gpt-5.1", provider="openai", display_name="GPT-5.1", is_featured=False)
        enriched = svc.enrich_model(model_in)
        assert enriched.is_featured is True

        # Model NOT in [featured] but already featured: stays featured (OR)
        model_already = ModelInfo(id="gpt-5-mini", provider="openai", display_name="Mini", is_featured=True)
        enriched2 = svc.enrich_model(model_already)
        assert enriched2.is_featured is True

        # Model NOT in [featured] and NOT already featured: stays unfeatured
        model_out = ModelInfo(id="gpt-3.5", provider="openai", display_name="Old", is_featured=False)
        enriched3 = svc.enrich_model(model_out)
        assert enriched3.is_featured is False


class TestModelCatalogServiceHelpers:
    """Test helper methods"""

    @pytest.fixture
    def service(self):
        """Create a service with sample config"""
        content = '''
[hidden]
google = ["gemini-internal"]
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()
            yield svc

    def test_get_all_providers(self, service):
        """Test getting all configured providers"""
        providers = service.get_all_providers()
        assert "google" in providers


class TestCatalogServiceSingleton:
    """Test the global singleton pattern"""

    def test_get_catalog_service_returns_same_instance(self):
        """Test that get_catalog_service returns singleton"""
        service1 = get_catalog_service()
        service2 = get_catalog_service()
        assert service1 is service2


class TestCatalogWithRealConfig:
    """Test with the real model_catalog.toml file"""

    def test_load_real_config(self):
        """Test loading the actual config file"""
        service = ModelCatalogService()
        service.load()

        assert service._loaded

    def test_real_config_has_hidden_models(self):
        """Test that real config has hidden models configured"""
        service = ModelCatalogService()
        service.load()

        assert service.is_hidden("openai", "babbage-002")

    def test_real_config_embedding_models_explicit_only(self):
        """Test that catalog only matches explicitly listed embedding models."""
        service = ModelCatalogService()
        service.load()

        assert service.get_category("ollama", "nomic-embed-text") == ModelCategory.EMBEDDING
        assert service.get_category("ollama", "mxbai-embed-large") == ModelCategory.EMBEDDING
        assert service.get_category("ollama", "all-minilm") == ModelCategory.EMBEDDING

        # Variants NOT in catalog return CHAT
        assert service.get_category("ollama", "nomic-embed-text:latest") == ModelCategory.CHAT

        # Unknown models default to CHAT
        assert service.get_category("ollama", "llama3.2:3b") == ModelCategory.CHAT


class TestContextLimits:
    """Test context limit functionality."""

    @pytest.fixture
    def service_with_limits(self):
        """Create a service with context limits config."""
        content = '''
[context_limits_override]
"gpt-4" = 8192
"gpt-4-turbo" = 128000
"gpt-5" = 128000
"gpt-5.1" = 256000
"claude-opus-4-5-20251101" = 1000000
"llama3.2" = 128000
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()
            yield svc

    def test_get_context_limit_exact_match(self, service_with_limits):
        """Test getting context limit with exact model match."""
        assert service_with_limits.get_context_limit("gpt-4") == 8192
        assert service_with_limits.get_context_limit("gpt-5.1") == 256000
        assert service_with_limits.get_context_limit("claude-opus-4-5-20251101") == 1000000

    def test_get_context_limit_base_model(self, service_with_limits):
        """Test getting context limit by base model name (before colon)."""
        assert service_with_limits.get_context_limit("llama3.2:3b") == 128000
        assert service_with_limits.get_context_limit("llama3.2:70b") == 128000

    def test_get_context_limit_partial_match(self, service_with_limits):
        """Test getting context limit by partial match."""
        limit = service_with_limits.get_context_limit("gpt-5-preview")
        assert limit == 128000

    def test_get_context_limit_unknown_model(self, service_with_limits):
        """Test that unknown model returns None."""
        assert service_with_limits.get_context_limit("totally-unknown-model") is None

    def test_enrich_model_sets_context_limit(self, service_with_limits):
        """Test that enrich_model sets context_limit on ModelInfo."""
        model = ModelInfo(id="gpt-5.1", provider="openai", display_name="GPT-5.1")
        enriched = service_with_limits.enrich_model(model)
        assert enriched.context_limit == 256000

    def test_real_config_has_context_limits(self):
        """Test that real config file has context limit overrides."""
        service = ModelCatalogService()
        service.load()

        assert service.get_context_limit("claude-opus-4-5-20251101") == 1000000
        assert service.get_context_limit("gemma2:9b") == 8192
        assert service.get_context_limit("gpt-4") == 8192
        assert service.get_context_limit("gpt-5-mini") == 128000
        assert service.get_context_limit("totally-unknown-model-xyz") is None


class TestTokenCounterCatalogIntegration:
    """Test integration between TokenCounter and ModelCatalogService."""

    def test_token_counter_uses_catalog(self):
        """Test that TokenCounter fetches limits from catalog."""
        from kestrel_sovereign.agent.token_counter import get_token_counter

        counter = get_token_counter("gpt-5.1")
        limit = counter.get_context_limit()
        assert limit == 256000

    def test_token_counter_uses_catalog_override(self):
        """Test that TokenCounter uses catalog TOML overrides."""
        from kestrel_sovereign.agent.token_counter import get_token_counter

        counter = get_token_counter("phi3:3.8b")
        limit = counter.get_context_limit()
        assert limit == 4096

    def test_token_counter_default_for_unknown(self):
        """Test that unknown models get default limit."""
        from kestrel_sovereign.agent.token_counter import get_token_counter, DEFAULT_CONTEXT_LIMIT

        counter = get_token_counter("completely-made-up-model-xyz")
        limit = counter.get_context_limit()
        assert limit == DEFAULT_CONTEXT_LIMIT


class TestDiscoveryCache:
    """Test discovery cache read/write."""

    def test_write_and_load_cache(self, tmp_path):
        """Test writing and loading discovery cache."""
        cache_path = tmp_path / "test_cache.json"
        service = ModelCatalogService(
            config_path=Path("/nonexistent/file.toml"),
            cache_path=cache_path,
        )

        models = [
            ModelInfo(id="gpt-5.1", provider="openai", display_name="GPT-5.1",
                      is_featured=True, context_limit=256000),
            ModelInfo(id="claude-sonnet-4-6", provider="anthropic", display_name="Sonnet 4.6",
                      supports_tools=True, supports_vision=True),
        ]

        service.write_cache(models)
        assert cache_path.exists()

        loaded = service.load_cache()
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].id == "gpt-5.1"
        assert loaded[0].is_featured is True
        assert loaded[0].context_limit == 256000
        assert loaded[1].id == "claude-sonnet-4-6"
        assert loaded[1].supports_tools is True

    def test_load_cache_missing_file(self, tmp_path):
        """Test loading cache when file doesn't exist."""
        service = ModelCatalogService(
            config_path=Path("/nonexistent/file.toml"),
            cache_path=tmp_path / "nonexistent.json",
        )
        result = service.load_cache()
        assert result is None

    def test_load_cache_corrupted_file(self, tmp_path):
        """Test loading cache when file is corrupted."""
        cache_path = tmp_path / "bad_cache.json"
        cache_path.write_text("not valid json{{{")

        service = ModelCatalogService(
            config_path=Path("/nonexistent/file.toml"),
            cache_path=cache_path,
        )
        result = service.load_cache()
        assert result is None

    def test_cache_roundtrip_preserves_category(self, tmp_path):
        """Test that category enum survives cache roundtrip."""
        cache_path = tmp_path / "cat_cache.json"
        service = ModelCatalogService(
            config_path=Path("/nonexistent/file.toml"),
            cache_path=cache_path,
        )

        models = [
            ModelInfo(id="embed-v3", provider="openai", display_name="Embed",
                      category=ModelCategory.EMBEDDING),
        ]
        service.write_cache(models)
        loaded = service.load_cache()

        assert loaded[0].category == ModelCategory.EMBEDDING


class TestFeaturedComputed:
    """Test that featured status is computed, not from TOML.

    With the new system, featured models are:
    - Models configured in llm_config.toml (marked by discovery)
    - Models in legacy [featured] section (OR behavior, backward compat)
    Featured is NOT removed by enrichment — only added.
    """

    def test_configured_models_stay_featured_without_toml_section(self):
        """Configured models keep featured status even without [featured] in TOML."""
        content = '''
[hidden]
openai = ["babbage-002"]
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()

        # Model marked featured by discovery (configured provider)
        model = ModelInfo(id="gpt-5.1", provider="openai", display_name="GPT", is_featured=True)
        enriched = svc.enrich_model(model)
        assert enriched.is_featured is True  # Preserved, not overwritten to False

    def test_non_configured_models_stay_unfeatured(self):
        """Non-configured models don't magically become featured."""
        content = '''
[hidden]
openai = ["babbage-002"]
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()

        model = ModelInfo(id="gpt-3.5-turbo", provider="openai", display_name="Old", is_featured=False)
        enriched = svc.enrich_model(model)
        assert enriched.is_featured is False
