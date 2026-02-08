"""
Unit tests for ModelCatalogService in model_catalog.py
"""
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
        """Create a sample config file"""
        content = '''
[featured]
openai = ["gpt-5.1", "gpt-5-mini"]
anthropic = ["claude-sonnet-4-5-20250929"]
ollama = ["llama3.2:3b"]

[display_names]
"gpt-5.1" = "GPT-5.1 (Latest)"
"claude-sonnet-4-5-20250929" = "Claude Sonnet 4.5"

[categories.embedding]
openai = ["text-embedding-3-large", "text-embedding-3-small"]

[categories.image]
openai = ["dall-e-3"]

[hidden]
openai = ["gpt-4-internal"]
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            yield Path(f.name)
        # Cleanup happens after test

    def test_load_featured_models(self, sample_config):
        """Test loading featured models from config"""
        service = ModelCatalogService(config_path=sample_config)
        service.load()

        assert service.is_featured("openai", "gpt-5.1")
        assert service.is_featured("openai", "gpt-5-mini")
        assert service.is_featured("anthropic", "claude-sonnet-4-5-20250929")
        assert not service.is_featured("openai", "gpt-3.5-turbo")

    def test_load_display_names(self, sample_config):
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
        """Test that get_category only matches explicitly listed models.

        The catalog does exact matching only. Variant detection (like :tag suffix)
        is handled by each adapter's list_models() method, not the catalog.
        """
        service = ModelCatalogService(config_path=sample_config)
        service.load()

        # Exact match should work
        assert service.get_category("openai", "text-embedding-3-large") == ModelCategory.EMBEDDING
        # Variants NOT in catalog return CHAT (adapter handles variant detection)
        assert service.get_category("openai", "text-embedding-3-large:latest") == ModelCategory.CHAT
        assert service.get_category("openai", "text-embedding-3-large:v2") == ModelCategory.CHAT
        # Unknown model defaults to CHAT
        assert service.get_category("openai", "gpt-5.1:turbo") == ModelCategory.CHAT

    def test_load_hidden_models(self, sample_config):
        """Test loading hidden models"""
        service = ModelCatalogService(config_path=sample_config)
        service.load()

        assert service.is_hidden("openai", "gpt-4-internal")
        assert not service.is_hidden("openai", "gpt-5.1")


class TestModelCatalogServiceEnrich:
    """Test model enrichment functionality"""

    @pytest.fixture
    def service(self):
        """Create a service with sample config"""
        content = '''
[featured]
openai = ["gpt-5.1"]

[display_names]
"gpt-5.1" = "GPT-5.1 (Latest)"

[categories.embedding]
openai = ["text-embedding-3-large"]

[hidden]
openai = ["gpt-4-internal"]
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()
            yield svc

    def test_enrich_featured_model(self, service):
        """Test enriching a featured model"""
        model = ModelInfo(id="gpt-5.1", provider="openai", display_name="GPT-5.1")
        enriched = service.enrich_model(model)

        assert enriched.is_featured is True
        assert enriched.is_hidden is False
        assert enriched.display_name == "GPT-5.1 (Latest)"
        assert enriched.category == ModelCategory.CHAT

    def test_enrich_hidden_model(self, service):
        """Test enriching a hidden model"""
        model = ModelInfo(id="gpt-4-internal", provider="openai", display_name="Internal")
        enriched = service.enrich_model(model)

        assert enriched.is_hidden is True
        assert enriched.is_featured is False

    def test_enrich_embedding_model(self, service):
        """Test enriching an embedding model explicitly listed in catalog"""
        model = ModelInfo(id="text-embedding-3-large", provider="openai", display_name="Embedding")
        enriched = service.enrich_model(model)

        assert enriched.category == ModelCategory.EMBEDDING

    def test_enrich_preserves_adapter_detected_category(self, service):
        """Test that enrich_model preserves category for models not in catalog.

        When an adapter sets category=EMBEDDING (e.g., Ollama detecting 'nomic-embed-text:latest'),
        enrich_model should NOT overwrite it with CHAT just because the variant isn't in catalog.
        """
        # Model with adapter-detected embedding category, not explicitly in catalog
        model = ModelInfo(
            id="nomic-embed-text:latest",
            provider="ollama",
            display_name="Nomic Embed",
            category=ModelCategory.EMBEDDING  # Set by adapter
        )
        enriched = service.enrich_model(model)

        # Category should be preserved, not overwritten to CHAT
        assert enriched.category == ModelCategory.EMBEDDING

    def test_enrich_multiple_models(self, service):
        """Test enriching multiple models at once"""
        models = [
            ModelInfo(id="gpt-5.1", provider="openai", display_name="GPT"),
            ModelInfo(id="text-embedding-3-large", provider="openai", display_name="Embed"),
        ]
        enriched = service.enrich_models(models)

        assert len(enriched) == 2
        assert enriched[0].is_featured is True
        assert enriched[1].category == ModelCategory.EMBEDDING


class TestModelCatalogServiceHelpers:
    """Test helper methods"""

    @pytest.fixture
    def service(self):
        """Create a service with sample config"""
        content = '''
[featured]
openai = ["gpt-5.1", "gpt-5-mini"]
anthropic = ["claude-sonnet-4-5"]

[hidden]
google = ["gemini-internal"]
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()
            yield svc

    def test_get_featured_models(self, service):
        """Test getting featured models for a provider"""
        openai_featured = service.get_featured_models("openai")
        assert "gpt-5.1" in openai_featured
        assert "gpt-5-mini" in openai_featured
        assert len(openai_featured) == 2

    def test_get_all_providers(self, service):
        """Test getting all configured providers"""
        providers = service.get_all_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers  # from hidden section


class TestCatalogServiceSingleton:
    """Test the global singleton pattern"""

    def test_get_catalog_service_returns_same_instance(self):
        """Test that get_catalog_service returns singleton"""
        # Note: This may affect other tests, use with care
        service1 = get_catalog_service()
        service2 = get_catalog_service()
        assert service1 is service2


class TestCatalogWithRealConfig:
    """Test with the real model_catalog.toml file"""

    def test_load_real_config(self):
        """Test loading the actual config file"""
        service = ModelCatalogService()
        service.load()

        # Should have loaded without errors
        assert service._loaded

        # Should have some featured models
        providers = service.get_all_providers()
        assert len(providers) > 0

    def test_real_config_has_openai_featured(self):
        """Test that real config has OpenAI featured models"""
        service = ModelCatalogService()
        service.load()

        openai_featured = service.get_featured_models("openai")
        # Should have at least one GPT-5 variant
        assert any("gpt-5" in model for model in openai_featured)

    def test_real_config_embedding_models_explicit_only(self):
        """Test that catalog only matches explicitly listed embedding models.

        The catalog does exact matching. Ollama adapter's list_models() handles
        variant detection (e.g., 'nomic-embed-text:latest' vs 'nomic-embed-text').
        enrich_model() preserves adapter-detected categories for unlisted models.
        """
        service = ModelCatalogService()
        service.load()

        # Ollama embedding models - base names in catalog are matched
        assert service.get_category("ollama", "nomic-embed-text") == ModelCategory.EMBEDDING
        assert service.get_category("ollama", "mxbai-embed-large") == ModelCategory.EMBEDDING
        assert service.get_category("ollama", "all-minilm") == ModelCategory.EMBEDDING

        # Variants NOT in catalog return CHAT (adapter handles detection)
        assert service.get_category("ollama", "nomic-embed-text:latest") == ModelCategory.CHAT
        assert service.get_category("ollama", "mxbai-embed-large:latest") == ModelCategory.CHAT

        # Unknown models default to CHAT
        assert service.get_category("ollama", "llama3.2:3b") == ModelCategory.CHAT


class TestContextLimits:
    """Test context limit functionality."""

    @pytest.fixture
    def service_with_limits(self):
        """Create a service with context limits config."""
        content = '''
[featured]
openai = ["gpt-5.1"]

[context_limits]
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
        # llama3.2:3b should match llama3.2
        assert service_with_limits.get_context_limit("llama3.2:3b") == 128000
        assert service_with_limits.get_context_limit("llama3.2:70b") == 128000

    def test_get_context_limit_partial_match(self, service_with_limits):
        """Test getting context limit by partial match."""
        # gpt-5-preview should match gpt-5
        limit = service_with_limits.get_context_limit("gpt-5-preview")
        assert limit == 128000  # Matches gpt-5

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

        # Check models that are genuine overrides in model_catalog.toml
        assert service.get_context_limit("claude-opus-4-5-20251101") == 1000000
        assert service.get_context_limit("gemini-3-pro") == 2000000
        assert service.get_context_limit("gpt-4-32k") == 32768
        # Models removed from TOML (covered by family patterns) return None
        assert service.get_context_limit("gpt-4") is None


class TestTokenCounterCatalogIntegration:
    """Test integration between TokenCounter and ModelCatalogService."""

    def test_token_counter_uses_catalog(self):
        """Test that TokenCounter fetches limits from catalog."""
        from kestrel_sovereign.agent.token_counter import get_token_counter

        # These limits are in model_catalog.toml, not the hardcoded dict
        counter = get_token_counter("gemini-3-pro")
        limit = counter.get_context_limit()
        # Gemini-3-pro has 2M context in catalog
        assert limit == 2000000

    def test_token_counter_fallback_to_hardcoded(self):
        """Test that TokenCounter falls back to hardcoded limits."""
        from kestrel_sovereign.agent.token_counter import get_token_counter

        # This model might not be in catalog but is in hardcoded dict
        counter = get_token_counter("phi3:3.8b")
        limit = counter.get_context_limit()
        # Should get either from catalog or hardcoded (4096)
        assert limit > 0

    def test_token_counter_default_for_unknown(self):
        """Test that unknown models get default limit."""
        from kestrel_sovereign.agent.token_counter import get_token_counter, DEFAULT_CONTEXT_LIMIT

        counter = get_token_counter("completely-made-up-model-xyz")
        limit = counter.get_context_limit()
        assert limit == DEFAULT_CONTEXT_LIMIT
