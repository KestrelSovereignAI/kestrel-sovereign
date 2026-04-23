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
        config={},
        providers=[{"name": "anthropic", "vendor": "anthropic", "route": "api", "model": "auto", "selection_hints": ["sonnet", "haiku"]}],
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
        providers=[{"name": "openai", "vendor": "openai", "route": "api", "model": "auto"}],
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
        providers=[{"name": "vertex_ai", "vendor": "vertex_ai", "route": "api", "model": "auto"}],
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


def test_auto_resolution_for_subscription_route_shares_vendor_catalog():
    """A subscription-style route (``anthropic:plan``) shares the vendor's catalog.

    Under the vendor/route architecture, the route is ``plan`` but the vendor
    is ``anthropic`` — and the model catalog is keyed on vendor. The resolver
    reads ``provider["vendor"]`` to find candidate models.
    """
    harness = _DiscoveryHarness(
        config={},
        providers=[{
            "name": "anthropic:plan",
            "vendor": "anthropic",
            "route": "plan",
            "model": "auto",
            "selection_hints": ["sonnet"],
        }],
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


def test_shipped_llm_config_uses_auto_models_for_primary_routes():
    """Every shipped route in llm_config.toml has ``model = "auto"``.

    Concrete IDs are never hardcoded in config; they resolve from discovery.
    """
    import tomllib

    with open(PROJECT_ROOT / "llm_config.toml", "rb") as handle:
        config = tomllib.load(handle)

    vendors = config.get("vendors") or {}
    # Walk every (vendor, route) pair and assert it asks discovery to pick.
    for vendor_name, vendor_cfg in vendors.items():
        for route_name, route_cfg in (vendor_cfg.get("routes") or {}).items():
            assert route_cfg.get("model") == "auto", (
                f"{vendor_name}:{route_name} must use model='auto' — "
                "no hardcoded model IDs in config."
            )
