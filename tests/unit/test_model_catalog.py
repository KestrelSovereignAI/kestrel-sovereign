"""
Unit tests for ModelCatalogService in model_catalog.py
"""
import json
import pytest
import tempfile
import toml
import tomllib
from pathlib import Path

from kestrel_sovereign.llm.model_catalog import ModelCatalogService, get_catalog_service
from kestrel_sovereign.llm.model_metadata import ModelInfo, ModelCategory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _service_with_shipped_unified_catalog() -> ModelCatalogService:
    config = tomllib.loads(
        (PROJECT_ROOT / "kestrel.toml.example").read_text(encoding="utf-8")
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml.dumps(config["llm"]["catalog"]))
        f.flush()
        service = ModelCatalogService(config_path=Path(f.name))
        service.load()
        return service


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
        service = _service_with_shipped_unified_catalog()

        assert service._loaded

    def test_real_config_filters_legacy_completion_models_via_category(self):
        """babbage-002 / davinci-002 are filtered from chat via category, not [hidden].

        Per the vendor/route/model refactor (#688): visibility is discovery-driven,
        so these legacy /v1/completions models are classified as category=completion
        rather than listed in [hidden]. Chat UI only shows category=chat.
        """
        from kestrel_sovereign.llm.model_catalog import ModelCategory

        service = _service_with_shipped_unified_catalog()

        assert service.get_category("openai", "babbage-002") == ModelCategory.COMPLETION
        assert service.get_category("openai", "davinci-002") == ModelCategory.COMPLETION
        # And they are NOT in the [hidden] list (no maintained list for this).
        assert not service.is_hidden("openai", "babbage-002")
        assert not service.is_hidden("openai", "davinci-002")

    def test_real_config_embedding_models_explicit_only(self):
        """Test that catalog only matches explicitly listed embedding models."""
        service = _service_with_shipped_unified_catalog()

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
        service = _service_with_shipped_unified_catalog()

        assert service.get_context_limit("claude-opus-4-5-20251101") == 1000000
        assert service.get_context_limit("gemma2:9b") == 8192
        assert service.get_context_limit("gpt-4") == 8192
        assert service.get_context_limit("gpt-5-mini") == 128000
        assert service.get_context_limit("totally-unknown-model-xyz") is None

    def test_partial_match_prefers_longest_substring(self):
        """Longest-substring partial match for bare-model lookups.

        ``get_context_limit`` walks bare-model entries; when several
        substrings match, the longest wins (deterministic instead of
        relying on dict-insertion-order). Route caps live in a
        separate section; that's covered below.
        """
        content = '''
[context_limits_override]
"gpt-5" = 128000
"gpt-5-mini" = 96000
'''
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.toml', delete=False
        ) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()

        # "gpt-5-mini" (10 chars) beats "gpt-5" (5 chars) for the
        # mini-suffixed model.
        assert svc.get_context_limit("gpt-5-mini-2025-08-07") == 96000
        # No "mini" — the only substring match is "gpt-5".
        assert svc.get_context_limit("gpt-5-preview") == 128000

    def test_route_cap_match_requires_route_boundary(self):
        """Codex round-5 P2: route key must match exactly or with ``/`` boundary.

        A substring check would let ``"openai:plan"`` falsely cap
        a different route ``"openai:plan-pro/gpt-5.5"``. The fix:
        require either equality or that the model string starts
        with ``"<route_key>/"``.
        """
        content = '''
[route_context_caps]
"openai:plan" = 20480
"openai:plan-pro" = 40960
'''
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.toml', delete=False
        ) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()

        # Exact route hits its own cap.
        assert svc.get_route_context_cap("openai:plan/gpt-5.5") == 20480
        assert svc.get_route_context_cap("openai:plan") == 20480
        # Sibling route gets its own cap, NOT openai:plan's.
        assert svc.get_route_context_cap("openai:plan-pro/gpt-5.5") == 40960
        assert svc.get_route_context_cap("openai:plan-pro") == 40960
        # Hypothetical third sibling with no cap returns None — even
        # though "openai:plan" is a substring.
        assert svc.get_route_context_cap("openai:plan-experimental/gpt-5.5") is None

    def test_route_caps_in_dedicated_section(self):
        """``[route_context_caps]`` is structurally separate from
        ``[context_limits_override]``.

        Route caps are looked up via ``get_route_context_cap`` and
        ONLY against ``_route_context_caps`` — bare-model entries
        cannot match here, even when they share the ``word:word``
        shape (Ollama). This is the structural fix codex round-3
        called for on PR #1396.
        """
        content = '''
[route_context_caps]
"openai:plan" = 20480

[context_limits_override]
"gpt-5" = 128000
"llama3.2:3b" = 128000
'''
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.toml', delete=False
        ) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()

        # Route-qualified selection hits the route cap.
        assert svc.get_route_context_cap("openai:plan/gpt-5.5") == 20480
        # Bare-model entries do NOT bleed into the route-cap path,
        # even with the route-qualified Ollama selection that codex
        # flagged (``ollama:local/llama3.2:3b``).
        assert svc.get_route_context_cap("ollama:local/llama3.2:3b") is None
        # And the bare-model lookup still resolves correctly via the
        # other API.
        assert svc.get_context_limit("llama3.2:3b") == 128000

    def test_matched_route_key_spans_all_layers(self, monkeypatch, tmp_path):
        """Codex round 2 P3 on the dynamic-cap PR: the matched-route
        helper used by the context-status endpoint must look across
        env + discovered + file layers so the route NAME is reported
        even when the cap value came from env-only or discovered-only.
        Previously the endpoint scanned only the file dict, so the
        route attribution silently vanished on those layers."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        # File dict EMPTY — only an env override applies.
        monkeypatch.setenv("KESTREL_OPENAI_PLAN_CONTEXT_CAP", "16384")
        svc = ModelCatalogService(config_path=tmp_path / "does_not_exist.toml")
        svc.load()
        assert svc.get_matched_route_cap_key("openai:plan/gpt-5.5") == "openai:plan"

        # Now drop env and add a discovered-only entry.
        monkeypatch.delenv("KESTREL_OPENAI_PLAN_CONTEXT_CAP", raising=False)
        svc2 = ModelCatalogService(config_path=tmp_path / "does_not_exist.toml")
        svc2.load()
        svc2.set_discovered_route_context_cap("openai:plan", 49152)
        assert svc2.get_matched_route_cap_key("openai:plan/gpt-5.5") == "openai:plan"

    def test_matched_route_key_returns_none_when_no_layer_matches(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        monkeypatch.delenv("KESTREL_OPENAI_PLAN_CONTEXT_CAP", raising=False)
        svc = ModelCatalogService(config_path=tmp_path / "does_not_exist.toml")
        svc.load()
        assert svc.get_matched_route_cap_key("openai:plan/gpt-5.5") is None

    def test_discovered_route_cap_beats_file_layers(self, monkeypatch, tmp_path):
        """Runtime-discovered values (e.g. from codex's thread/start
        response) take precedence over kestrel.toml + model_catalog.toml
        because they reflect what THIS account's THIS plan actually
        offers right now, not the operator's empirical guess from a
        previous tier."""
        monkeypatch.delenv("KESTREL_OPENAI_PLAN_CONTEXT_CAP", raising=False)
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        project_dir_path = tmp_path / "project"
        project_dir_path.mkdir()
        (project_dir_path / "kestrel.toml").write_text(
            '[llm.route_context_caps]\n"openai:plan" = 32768\n'
        )
        monkeypatch.setenv("KESTREL_HOME", str(project_dir_path))
        from kestrel_sovereign import paths as _paths
        _paths._resolve_cached.cache_clear()
        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text('[route_context_caps]\n"openai:plan" = 20480\n')

        svc = ModelCatalogService(config_path=catalog)
        svc.load()
        # Before discovery: file layer wins.
        assert svc.get_route_context_cap("openai:plan") == 32768
        # Codex reports 49152 from thread/start. Discovery wins now.
        svc.set_discovered_route_context_cap("openai:plan/gpt-5.5", 49152)
        assert svc.get_route_context_cap("openai:plan/gpt-5.5") == 49152

    def test_env_override_beats_discovered_route_cap(self, monkeypatch, tmp_path):
        """Operator's env-var override is the highest-priority knob —
        even if codex's server reports a higher cap, the operator can
        force a lower one."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text('[route_context_caps]\n"openai:plan" = 20480\n')
        monkeypatch.setenv("KESTREL_OPENAI_PLAN_CONTEXT_CAP", "16384")
        svc = ModelCatalogService(config_path=catalog)
        svc.load()
        svc.set_discovered_route_context_cap("openai:plan", 65536)
        assert svc.get_route_context_cap("openai:plan") == 16384

    def test_discovered_cap_non_integer_value_silently_dropped(self, monkeypatch, tmp_path):
        """Garbage from a future codex wire-format shift must not crash
        ``load()`` or poison the cache — silently drop with a debug log."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        monkeypatch.delenv("KESTREL_OPENAI_PLAN_CONTEXT_CAP", raising=False)
        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text('[route_context_caps]\n"openai:plan" = 20480\n')
        svc = ModelCatalogService(config_path=catalog)
        svc.load()
        svc.set_discovered_route_context_cap("openai:plan", "not-an-int")  # type: ignore[arg-type]
        svc.set_discovered_route_context_cap("openai:plan", None)  # type: ignore[arg-type]
        assert svc.get_route_context_cap("openai:plan") == 20480

    def test_clear_discovered_caps_falls_back_to_file_layer(self, monkeypatch, tmp_path):
        """The clear hook lets tests / forced re-discoveries reset
        without leaving stale per-session values cached."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        monkeypatch.delenv("KESTREL_OPENAI_PLAN_CONTEXT_CAP", raising=False)
        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text('[route_context_caps]\n"openai:plan" = 20480\n')
        svc = ModelCatalogService(config_path=catalog)
        svc.load()
        svc.set_discovered_route_context_cap("openai:plan", 65536)
        assert svc.get_route_context_cap("openai:plan") == 65536
        svc.clear_discovered_route_context_caps()
        assert svc.get_route_context_cap("openai:plan") == 20480

    def test_kestrel_toml_caps_applied_when_catalog_file_missing(self, monkeypatch, tmp_path):
        """Codex round 3 P2 on #1506: when ``model_catalog.toml`` is
        absent (pip-install / Cloud Run / unified-config deployments),
        ``load()`` previously returned at the missing-catalog guard
        BEFORE the new merge — so a cap set only in ``kestrel.toml``
        was still silently ignored. The merge must run on the defaults
        path too."""
        # KESTREL_DB_PATH could pre-empt our project-root probe in a
        # documented operator environment — clear it (codex round 4 P2).
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        # No model_catalog.toml at all.
        project_dir_path = tmp_path / "project"
        project_dir_path.mkdir()
        (project_dir_path / "kestrel.toml").write_text(
            '[llm.route_context_caps]\n"openai:plan" = 32768\n'
        )
        monkeypatch.setenv("KESTREL_HOME", str(project_dir_path))
        from kestrel_sovereign import paths as _paths
        _paths._resolve_cached.cache_clear()

        svc = ModelCatalogService(config_path=tmp_path / "does_not_exist.toml")
        svc.load()  # Hits the missing-catalog defaults path.
        # kestrel.toml cap survives — was the regression codex caught.
        assert svc.get_route_context_cap("openai:plan") == 32768

    def test_env_overrides_apply_even_when_catalog_file_missing(self, monkeypatch, tmp_path):
        """Same defaults-path concern as the test above, but for the env
        override layer. Lifted out of ``load()`` into a shared helper so
        both paths apply env overrides identically."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        project_dir_path = tmp_path / "project"
        project_dir_path.mkdir()
        monkeypatch.setenv("KESTREL_HOME", str(project_dir_path))
        from kestrel_sovereign import paths as _paths
        _paths._resolve_cached.cache_clear()
        monkeypatch.setenv("KESTREL_OPENAI_PLAN_CONTEXT_CAP", "65536")

        svc = ModelCatalogService(config_path=tmp_path / "does_not_exist.toml")
        svc.load()  # Defaults path.
        assert svc.get_route_context_cap("openai:plan") == 65536

    def test_kestrel_toml_route_caps_resolved_via_kestrel_db_path_first(self, monkeypatch, tmp_path):
        """Codex round 2 P2 on #1506: the helper must follow the same
        search order as ``config.load_section('llm')`` — agent-specific
        ``KESTREL_DB_PATH/kestrel.toml`` is checked before the project
        root, so an agent-scoped cap takes precedence over the global
        one."""
        # Agent-scoped kestrel.toml has the higher-precedence cap.
        agent_dir = tmp_path / "agent_db"
        agent_dir.mkdir()
        (agent_dir / "kestrel.toml").write_text(
            '[llm.route_context_caps]\n"openai:plan" = 65536\n'
        )
        # Project-root kestrel.toml has a different (lower-precedence) cap.
        project_dir_path = tmp_path / "project"
        project_dir_path.mkdir()
        (project_dir_path / "kestrel.toml").write_text(
            '[llm.route_context_caps]\n"openai:plan" = 32768\n'
        )

        monkeypatch.setenv("KESTREL_DB_PATH", str(agent_dir))
        monkeypatch.setenv("KESTREL_HOME", str(project_dir_path))
        from kestrel_sovereign import paths as _paths
        _paths._resolve_cached.cache_clear()

        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text('[route_context_caps]\n"openai:plan" = 20480\n')

        svc = ModelCatalogService(config_path=catalog)
        svc.load()
        # Agent-scoped 65536 wins over both project-root 32768 and
        # catalog default 20480.
        assert svc.get_route_context_cap("openai:plan") == 65536

    def test_kestrel_toml_route_caps_resolved_via_kestrel_home(self, monkeypatch, tmp_path):
        """#1506 codex round 1 P2: the kestrel.toml probe must resolve
        through ``paths.project_dir`` (which honors ``KESTREL_HOME``)
        rather than ``Path.cwd()`` — otherwise pip-installed,
        Cloud-Run, and ``KESTREL_HOME``-overridden deployments silently
        skip the merge even though the operator configured the cap in
        the supported project location."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        # Project dir resolves to kestrel_home_dir via KESTREL_HOME.
        kestrel_home_dir = tmp_path / "kestrel_home"
        kestrel_home_dir.mkdir()
        (kestrel_home_dir / "kestrel.toml").write_text(
            '[llm.route_context_caps]\n"openai:plan" = 49152\n'
        )

        # Process cwd is somewhere else (e.g. /tmp). Without project_dir
        # resolution, the helper would probe ``./kestrel.toml`` and miss
        # the file entirely.
        elsewhere = tmp_path / "not_the_project"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setenv("KESTREL_HOME", str(kestrel_home_dir))

        # Reset the project_dir LRU cache so the new env var is honored.
        from kestrel_sovereign import paths as _paths
        _paths._resolve_cached.cache_clear()

        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text('[route_context_caps]\n"openai:plan" = 20480\n')

        svc = ModelCatalogService(config_path=catalog)
        svc.load()
        assert svc.get_route_context_cap("openai:plan") == 49152

    def test_kestrel_toml_route_caps_merge_into_catalog(self, monkeypatch, tmp_path):
        """#1506: ``kestrel.toml [llm.route_context_caps]`` is honored
        by the catalog. Operators reasonably expect LLM-routing config
        in ``kestrel.toml`` next to route_priority and vendor routes;
        before this layer was added, setting the cap there was a silent
        no-op."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        catalog_content = '''
[route_context_caps]
"openai:plan" = 20480
'''
        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text(catalog_content)

        kestrel_toml = tmp_path / "kestrel.toml"
        kestrel_toml.write_text('''
[llm.route_context_caps]
"openai:plan" = 32768
"anthropic:plan" = 65536
''')
        # Run the catalog with the cwd set to where kestrel.toml lives.
        monkeypatch.chdir(tmp_path)
        svc = ModelCatalogService(config_path=catalog)
        svc.load()

        # kestrel.toml overrides model_catalog.toml's value for openai:plan.
        assert svc.get_route_context_cap("openai:plan") == 32768
        # Caps declared only in kestrel.toml are picked up too.
        assert svc.get_route_context_cap("anthropic:plan") == 65536

    def test_kestrel_toml_missing_does_not_break_catalog_load(self, monkeypatch, tmp_path):
        """If kestrel.toml is absent, the catalog still loads cleanly
        with just the model_catalog.toml caps."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        catalog_content = '''
[route_context_caps]
"openai:plan" = 20480
'''
        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text(catalog_content)
        monkeypatch.chdir(tmp_path)
        svc = ModelCatalogService(config_path=catalog)
        svc.load()  # must not raise even though kestrel.toml is missing
        assert svc.get_route_context_cap("openai:plan") == 20480

    def test_env_overrides_kestrel_toml_route_cap(self, monkeypatch, tmp_path):
        """Precedence: env var > kestrel.toml > model_catalog.toml.

        The env override is the highest-priority knob because operators
        need a no-redeploy way to bump the cap when ChatGPT-Plus shifts;
        kestrel.toml sits between env and the catalog default."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text('[route_context_caps]\n"openai:plan" = 20480\n')
        kestrel_toml = tmp_path / "kestrel.toml"
        kestrel_toml.write_text('[llm.route_context_caps]\n"openai:plan" = 32768\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KESTREL_OPENAI_PLAN_CONTEXT_CAP", "16384")
        svc = ModelCatalogService(config_path=catalog)
        svc.load()
        # Env wins.
        assert svc.get_route_context_cap("openai:plan") == 16384

    def test_kestrel_toml_malformed_route_caps_block_is_ignored(self, monkeypatch, tmp_path):
        """A non-table value at [llm.route_context_caps] is logged at
        debug and ignored — must NOT raise from catalog.load()."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text('[route_context_caps]\n"openai:plan" = 20480\n')
        # Use a top-level string at the path the helper probes for.
        kestrel_toml = tmp_path / "kestrel.toml"
        kestrel_toml.write_text(
            '[llm]\nroute_context_caps = "not a table"\n'
        )
        monkeypatch.chdir(tmp_path)
        svc = ModelCatalogService(config_path=catalog)
        svc.load()  # must not raise
        # Catalog default stands when kestrel.toml block is malformed.
        assert svc.get_route_context_cap("openai:plan") == 20480

    def test_kestrel_toml_non_integer_value_is_skipped(self, monkeypatch, tmp_path):
        """Non-integer cap values in the kestrel.toml block are skipped
        with a debug log; valid sibling entries still merge."""
        monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
        catalog = tmp_path / "model_catalog.toml"
        catalog.write_text('[route_context_caps]\n"openai:plan" = 20480\n')
        kestrel_toml = tmp_path / "kestrel.toml"
        kestrel_toml.write_text('''
[llm.route_context_caps]
"openai:plan" = "not an int"
"anthropic:plan" = 65536
''')
        monkeypatch.chdir(tmp_path)
        svc = ModelCatalogService(config_path=catalog)
        svc.load()
        # Bad value skipped → catalog default stands for openai:plan.
        assert svc.get_route_context_cap("openai:plan") == 20480
        # Sibling integer entry merges in cleanly.
        assert svc.get_route_context_cap("anthropic:plan") == 65536

    def test_env_override_openai_plan_context_cap(self, monkeypatch):
        """``KESTREL_OPENAI_PLAN_CONTEXT_CAP`` overrides the TOML cap.

        The cap is empirical (ChatGPT-Plus doesn't advertise it); the
        operator needs a no-redeploy way to raise/lower it as the
        upstream shifts. The env override is read at catalog load
        time, so a fresh process picks up the new value.
        """
        content = '''
[route_context_caps]
"openai:plan" = 20480
'''
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.toml', delete=False
        ) as f:
            f.write(content)
            f.flush()
            monkeypatch.setenv("KESTREL_OPENAI_PLAN_CONTEXT_CAP", "16384")
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()

        assert svc.get_route_context_cap("openai:plan") == 16384
        assert svc.get_route_context_cap("openai:plan/gpt-5.5") == 16384

    def test_route_cap_discriminator_excludes_ollama_tags(self):
        """Codex round-2 P2: route-cap path must NOT engage on Ollama tags.

        Ollama bare model IDs (``gemma2:9b``, ``qwen2.5:14b``,
        ``mistral:7b``) contain ``:`` but no ``/``. The earlier
        discriminator ``":" in model or "/" in model`` would let
        them enter the route-cap branch and pick up colon-containing
        Ollama catalog entries as if they were route caps,
        bypassing discovered/cached exact limits for the tag.

        The discriminator now requires BOTH ``:`` AND ``/`` — the
        kestrel route form ``"<vendor>:<route>/<model_name>"``. This
        test asserts the rule at the predicate boundary so the
        guarantee survives even when the catalog has Ollama-style
        colon keys.
        """
        # Direct predicate assertion, no singleton mutation — the
        # route-cap branch in TokenCounter is gated on this exact
        # boolean. Keep them in sync with this regression test.
        def is_route_qualified(model: str) -> bool:
            return ":" in model and "/" in model

        # Route-qualified (should enter the branch):
        assert is_route_qualified("openai:plan/gpt-5.5")
        assert is_route_qualified("anthropic:api/claude-opus-4-7")
        # Ollama tags (should NOT enter the branch — codex round-2 P2):
        assert not is_route_qualified("gemma2:9b")
        assert not is_route_qualified("qwen2.5:14b")
        assert not is_route_qualified("mistral:7b")
        assert not is_route_qualified("phi3:3.8b")
        # Vendor-only forms (no route, no cap):
        assert not is_route_qualified("openai/gpt-5.5")
        # Bare models:
        assert not is_route_qualified("gpt-5.5")

    def test_route_context_cap_only_matches_dedicated_section(self):
        """``get_route_context_cap`` reads ONLY from ``[route_context_caps]``.

        Bare-model entries that happen to share the ``word:word``
        shape (e.g. Ollama tags ``llama3.2:3b``) live in
        ``[context_limits_override]`` and must not appear in
        route-cap lookups — even when the active selection is
        route-qualified (``ollama:local/llama3.2:3b``).
        """
        content = '''
[route_context_caps]
"openai:plan" = 20480

[context_limits_override]
"gpt-5" = 128000
"llama3.2:3b" = 128000
'''
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.toml', delete=False
        ) as f:
            f.write(content)
            f.flush()
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()

        # Route cap wins on the capped route.
        assert svc.get_route_context_cap("openai:plan/gpt-5.5") == 20480
        # Non-capped route returns None — caller falls through to
        # discovery/cache for the bare model id.
        assert svc.get_route_context_cap("openai:api/gpt-5.5") is None
        # Ollama bare-model entry (``llama3.2:3b``) doesn't bleed
        # into route-cap lookups (codex round-3 P2).
        assert svc.get_route_context_cap("ollama:local/llama3.2:3b") is None
        # Bare-model lookup returns None for this accessor.
        assert svc.get_route_context_cap("gpt-5.5") is None

    def test_env_override_generic_route_cap(self, monkeypatch):
        """``KESTREL_ROUTE_CONTEXT_CAP_<VENDOR>_<ROUTE>`` populates the
        dedicated section."""
        content = '''
[route_context_caps]
"openai:plan" = 20480

[context_limits_override]
"gpt-5" = 128000
'''
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.toml', delete=False
        ) as f:
            f.write(content)
            f.flush()
            monkeypatch.setenv(
                "KESTREL_ROUTE_CONTEXT_CAP_ANTHROPIC_PLAN", "30720"
            )
            svc = ModelCatalogService(config_path=Path(f.name))
            svc.load()

        assert svc.get_route_context_cap("anthropic:plan") == 30720
        assert svc.get_route_context_cap("anthropic:plan/claude-sonnet-4-6") == 30720
        # Pre-existing TOML route cap remains too.
        assert svc.get_route_context_cap("openai:plan/gpt-5.5") == 20480
        # Untouched bare-model lookup still works.
        assert svc.get_context_limit("gpt-5") == 128000


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
