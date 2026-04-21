"""Contracts for auto model selection from config, cache, and discovery."""

from pathlib import Path

from kestrel_sovereign.llm.model_discovery import ModelDiscoveryMixin
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _DiscoveryHarness(ModelDiscoveryMixin):
    def __init__(self, config, providers):
        self.config = config
        self.providers = providers


def test_auto_resolution_uses_selection_hints_over_discovery_order():
    harness = _DiscoveryHarness(
        config={"anthropic": {"selection_hints": ["sonnet", "haiku"]}},
        providers=[{"name": "anthropic", "model": "auto"}],
    )
    models = [
        ModelInfo(
            id="claude-haiku-4-5-20251001",
            provider="anthropic",
            display_name="Claude Haiku 4.5",
            category=ModelCategory.CHAT,
            supports_tools=True,
        ),
        ModelInfo(
            id="claude-sonnet-4-6",
            provider="anthropic",
            display_name="Claude Sonnet 4.6",
            category=ModelCategory.CHAT,
            supports_tools=True,
        ),
    ]

    harness._resolve_auto_providers(models)

    assert harness.providers[0]["model"] == "claude-sonnet-4-6"


def test_auto_resolution_prefers_featured_when_no_selection_hints_exist():
    harness = _DiscoveryHarness(
        config={"openai": {}},
        providers=[{"name": "openai", "model": "auto"}],
    )
    models = [
        ModelInfo(
            id="gpt-5-mini",
            provider="openai",
            display_name="GPT-5 Mini",
            category=ModelCategory.CHAT,
            supports_tools=True,
            is_featured=False,
        ),
        ModelInfo(
            id="gpt-5.1",
            provider="openai",
            display_name="GPT-5.1",
            category=ModelCategory.CHAT,
            supports_tools=True,
            is_featured=True,
        ),
    ]

    harness._resolve_auto_providers(models)

    assert harness.providers[0]["model"] == "gpt-5.1"


def test_auto_resolution_avoids_preview_models_when_choosing_fallback():
    harness = _DiscoveryHarness(
        config={"vertex_ai": {}},
        providers=[{"name": "vertex_ai", "model": "auto"}],
    )
    models = [
        ModelInfo(
            id="gemini-3-pro-preview",
            provider="vertex_ai",
            display_name="Gemini 3 Pro Preview",
            category=ModelCategory.CHAT,
            supports_tools=True,
        ),
        ModelInfo(
            id="gemini-3-pro",
            provider="vertex_ai",
            display_name="Gemini 3 Pro",
            category=ModelCategory.CHAT,
            supports_tools=True,
        ),
    ]

    harness._resolve_auto_providers(models)

    assert harness.providers[0]["model"] == "gemini-3-pro"


def test_auto_resolution_uses_canonical_discovery_for_plan_provider():
    harness = _DiscoveryHarness(
        config={"claude_plan": {"selection_hints": ["sonnet"]}},
        providers=[{"name": "claude_plan", "model": "auto"}],
    )
    models = [
        ModelInfo(
            id="claude-sonnet-4-6",
            provider="anthropic",
            display_name="Claude Sonnet 4.6",
            category=ModelCategory.CHAT,
            supports_tools=True,
        ),
        ModelInfo(
            id="claude-opus-4-6",
            provider="anthropic",
            display_name="Claude Opus 4.6",
            category=ModelCategory.CHAT,
            supports_tools=True,
        ),
    ]

    harness._resolve_auto_providers(models)

    assert harness.providers[0]["model"] == "claude-sonnet-4-6"


def test_discovery_alias_projection_builds_plan_provider_models():
    harness = _DiscoveryHarness(
        config={"openai_plan": {"selection_hints": ["gpt-5.4"]}},
        providers=[{"name": "openai_plan", "model": "auto"}],
    )
    models = [
        ModelInfo(
            id="gpt-5.4",
            provider="openai",
            display_name="GPT-5.4",
            category=ModelCategory.CHAT,
            supports_tools=True,
        ),
    ]

    projected = harness._build_alias_discovery_models(models)

    assert len(projected) == 1
    assert projected[0].provider == "openai_plan"
    assert projected[0].id == "gpt-5.4"


def test_shipped_llm_config_uses_auto_models_for_primary_providers():
    import tomllib

    with open(PROJECT_ROOT / "llm_config.toml", "rb") as handle:
        config = tomllib.load(handle)

    for provider_name in [
        "claude_plan",
        "openai_plan",
        "openrouter",
        "openai",
        "openai_mini",
        "anthropic",
        "vertex_ai",
        "runpod",
        "llama_cpp",
        "ollama",
        "xai",
        "groq",
    ]:
        assert config[provider_name]["model"] == "auto"
        assert "selection_hints" in config[provider_name] or provider_name == "llama_cpp"
