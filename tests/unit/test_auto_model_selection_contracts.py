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


def test_openai_plan_resolves_against_codex_catalog_not_openai_api():
    """openai:plan must resolve ``auto`` against codex's serveable subset.

    The two openai routes have DIFFERENT serveable sets: ``openai:api`` sees
    the full OpenAI catalog (incl. ``gpt-5.5-pro``, which codex/ChatGPT
    rejects); ``openai:plan`` (CodexAdapter) sees only what codex serves. With
    hint ``"gpt-5"`` the plan route must pick ``gpt-5.5`` (top list-visible
    codex model), NEVER ``gpt-5.5-pro`` — while ``openai:api`` still resolves
    against its full catalog.
    """
    from unittest.mock import AsyncMock

    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    codex_catalog = [
        ModelInfo(id="gpt-5.5", provider="openai", display_name="GPT-5.5",
                  category=ModelCategory.CHAT, supports_tools=True, is_featured=True),
        ModelInfo(id="gpt-5.4", provider="openai", display_name="GPT-5.4",
                  category=ModelCategory.CHAT, supports_tools=True),
        ModelInfo(id="gpt-5.4-mini", provider="openai", display_name="GPT-5.4 mini",
                  category=ModelCategory.CHAT, supports_tools=True),
    ]
    codex_adapter = CodexAdapter()
    # Stub the route-specific catalog read (would hit models_cache.json).
    codex_adapter.list_models = AsyncMock(return_value=codex_catalog)

    harness = _DiscoveryHarness(
        config={},
        providers=[
            {"name": "openai:api", "vendor": "openai", "route": "api",
             "model": "auto", "selection_hints": ["gpt-5"], "adapter": object()},
            {"name": "openai:plan", "vendor": "openai", "route": "plan",
             "model": "auto", "selection_hints": ["gpt-5"], "adapter": codex_adapter},
        ],
    )
    # Shared vendor discovery (openai:api's full catalog) includes gpt-5.5-pro.
    models = [
        ModelInfo(id="gpt-5.5-pro", provider="openai", display_name="GPT-5.5 Pro",
                  category=ModelCategory.CHAT, supports_tools=True, is_featured=True),
        ModelInfo(id="gpt-5.1", provider="openai", display_name="GPT-5.1",
                  category=ModelCategory.CHAT, supports_tools=True, is_featured=True),
    ]

    harness._resolve_auto_providers(models)

    by_name = {p["name"]: p["model"] for p in harness.providers}
    # plan resolves against codex's subset -> gpt-5.5 (NOT gpt-5.5-pro).
    assert by_name["openai:plan"] == "gpt-5.5"
    # api still resolves against the full OpenAI catalog.
    assert by_name["openai:api"] == "gpt-5.5-pro"


def test_shipped_llm_config_uses_auto_models_for_primary_routes():
    """Every shipped route in kestrel.toml [llm] has ``model = "auto"``.

    Concrete IDs are never hardcoded in config; they resolve from discovery.
    Repointed from llm_config.toml in #940 — the standalone file no longer
    exists at the repo root; the seed config now lives in kestrel.toml.example.
    """
    import tomllib

    with open(PROJECT_ROOT / "kestrel.toml.example", "rb") as handle:
        config = tomllib.load(handle)

    llm = config.get("llm", {})
    vendors = llm.get("vendors") or {}
    # Walk every (vendor, route) pair and assert it asks discovery to pick.
    for vendor_name, vendor_cfg in vendors.items():
        for route_name, route_cfg in (vendor_cfg.get("routes") or {}).items():
            assert route_cfg.get("model") == "auto", (
                f"{vendor_name}:{route_name} must use model='auto' — "
                "no hardcoded model IDs in config."
            )
