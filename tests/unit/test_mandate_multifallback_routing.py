"""Mandate fallback routing must use each fallback's OWN model (#1685).

When the mandated vendor is unavailable, ``resolve_provider_routing`` builds a
fallback chain from ``_mandate_fallbacks``. Each fallback entry may name a
different vendor AND a different model. The chain returns a single
``target_model``, so pinning it to ``fallbacks[0].model`` made every later
fallback (whose model differed) get rejected downstream by
``_model_available_for_route`` — effectively unreachable. The fix pins each
fallback's model onto its own provider dict and clears the global target so
the downstream loop resolves per-provider.
"""
from __future__ import annotations

import pytest

from kestrel_sovereign.llm.service import LLMService


def _provider(name: str, vendor: str, model: str) -> dict:
    return {"name": name, "vendor": vendor, "model": model, "adapter": object(), "client": object()}


def _routing_service(providers, *, mandate, fallbacks) -> LLMService:
    svc = LLMService.__new__(LLMService)
    svc._disabled_routes = {}
    svc.providers = providers
    svc._mandate_preference = mandate
    svc._mandate_fallbacks = fallbacks
    return svc


def test_each_fallback_keeps_its_own_model():
    """Two fallbacks with DIFFERENT models — each provider must carry its own."""
    providers = [
        _provider("openai:api", "openai", "gpt-default"),
        _provider("openrouter:api", "openrouter", "or-default"),
    ]
    svc = _routing_service(
        providers,
        # Mandated vendor "anthropic" is not in the provider set -> fallbacks.
        mandate={"vendor": "anthropic", "model": "claude-x", "route": None},
        fallbacks=[
            {"vendor": "openai", "model": "gpt-4o"},
            {"vendor": "openrouter", "model": "meta/llama-3.1-70b"},
        ],
    )

    routed, target_model = svc.resolve_provider_routing()

    # The global target is cleared; each provider carries its own model.
    assert target_model is None
    by_vendor = {p["vendor"]: p["model"] for p in routed}
    assert by_vendor == {"openai": "gpt-4o", "openrouter": "meta/llama-3.1-70b"}


def test_fallback_without_model_uses_route_default():
    """A fallback entry that names only a vendor keeps the route's own model."""
    providers = [_provider("openai:api", "openai", "gpt-default")]
    svc = _routing_service(
        providers,
        mandate={"vendor": "anthropic", "model": "claude-x", "route": None},
        fallbacks=[{"vendor": "openai"}],  # no model -> use route default
    )

    routed, target_model = svc.resolve_provider_routing()

    assert target_model is None
    assert [p["model"] for p in routed] == ["gpt-default"]


def test_fallback_does_not_mutate_shared_provider_dicts():
    """Pinning a fallback model must not corrupt the shared self.providers."""
    providers = [_provider("openai:api", "openai", "gpt-default")]
    svc = _routing_service(
        providers,
        mandate={"vendor": "anthropic", "model": "claude-x", "route": None},
        fallbacks=[{"vendor": "openai", "model": "gpt-4o"}],
    )

    routed, _ = svc.resolve_provider_routing()

    assert routed[0]["model"] == "gpt-4o"
    # The original provider dict is untouched.
    assert svc.providers[0]["model"] == "gpt-default"


def test_primary_mandate_path_unchanged():
    """When the mandated vendor IS available, routing is unaffected."""
    providers = [
        _provider("anthropic:api", "anthropic", "claude-default"),
        _provider("openai:api", "openai", "gpt-default"),
    ]
    svc = _routing_service(
        providers,
        mandate={"vendor": "anthropic", "model": "claude-x", "route": None},
        fallbacks=[{"vendor": "openai", "model": "gpt-4o"}],
    )

    routed, target_model = svc.resolve_provider_routing()

    assert [p["vendor"] for p in routed] == ["anthropic"]
    assert target_model == "claude-x"
