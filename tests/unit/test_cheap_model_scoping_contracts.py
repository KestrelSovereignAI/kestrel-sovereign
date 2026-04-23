"""Contract tests for vendor-scoped cheap-model selection.

The old ``get_cheap_model()`` returned a bare model id which upstream callers
forced as ``model_override`` onto every provider in the fallback chain —
producing four garbage attempts on purpose (the "cheap-model broadcast" bug).

The new ``get_cheap_model_selector()`` returns a ``"<vendor>/<model>"``
selector. ``resolve_provider_routing`` then filters to that vendor's routes
only, so the bogus-id cascade is structurally impossible.
"""

from types import SimpleNamespace

from kestrel_sovereign.llm.service import LLMService


def _fake_registry(providers):
    """Minimal registry mock — only ``get_providers_with_pattern`` is used."""

    def get_providers_with_pattern(patterns):
        if not patterns:
            return []
        pl = [p.lower() for p in patterns]
        out = []
        for p in providers:
            model = (p.model or "").lower()
            if any(pat in model for pat in pl):
                out.append(p)
        return out

    reg = SimpleNamespace(get_providers_with_pattern=get_providers_with_pattern)
    return reg


def _route(vendor, route, model):
    return SimpleNamespace(
        name=f"{vendor}:{route}",
        vendor=vendor,
        route=route,
        model=model,
    )


def _make_service(providers, mandate_defaults):
    """Build a LLMService stub with just what get_cheap_model_selector needs."""
    svc = SimpleNamespace()
    svc.mandate_config = {"defaults": dict(mandate_defaults)}
    svc.provider_registry = _fake_registry(providers)
    svc.providers = [
        {"name": p.name, "vendor": p.vendor, "route": p.route, "model": p.model}
        for p in providers
    ]
    # Bind the unbound methods onto the stub so we test the real implementation.
    svc.get_cheap_model_selector = LLMService.get_cheap_model_selector.__get__(svc)
    svc.get_cheap_model = LLMService.get_cheap_model.__get__(svc)
    svc._resolve_model_selector = LLMService._resolve_model_selector.__get__(svc)
    return svc


def test_no_config_no_hints_returns_none():
    """Without config, caller uses default routing — no override."""
    svc = _make_service(
        providers=[_route("openai", "api", "gpt-5")],
        mandate_defaults={},
    )
    assert svc.get_cheap_model_selector() is None


def test_hints_match_returns_vendor_scoped_selector():
    """A hint that matches a route's model returns ``vendor/model``.

    This is the core scoping fix: without the vendor prefix, upstream code
    would inject the bare model id into every provider's fallback chain.
    """
    svc = _make_service(
        providers=[
            _route("openai", "api", "gpt-5"),
            _route("openai", "mini", "gpt-5-mini"),  # matches "mini" hint
            _route("anthropic", "api", "claude-sonnet-4-6"),
        ],
        mandate_defaults={"cheap_model_hints": ["mini"]},
    )
    sel = svc.get_cheap_model_selector()
    assert sel == "openai/gpt-5-mini", f"expected vendor-scoped selector, got {sel}"


def test_no_matching_hint_returns_none():
    """Hints that match nothing yield no override (caller uses default)."""
    svc = _make_service(
        providers=[
            _route("anthropic", "api", "claude-sonnet-4-6"),
            _route("openai", "api", "gpt-5"),
        ],
        mandate_defaults={"cheap_model_hints": ["haiku", "flash", "nano"]},
    )
    assert svc.get_cheap_model_selector() is None


def test_explicit_cheap_model_selector_honored():
    """``cheap_model`` in config, if set explicitly, takes precedence over hints."""
    svc = _make_service(
        providers=[
            _route("anthropic", "api", "claude-haiku-4-5"),
        ],
        mandate_defaults={"cheap_model": "anthropic/claude-haiku-4-5"},
    )
    sel = svc.get_cheap_model_selector()
    assert sel == "anthropic/claude-haiku-4-5"


def test_cheap_model_auto_falls_through_to_hints():
    """``cheap_model = 'auto'`` means "use hints" — not a literal override."""
    svc = _make_service(
        providers=[
            _route("openai", "mini", "gpt-5-mini"),
        ],
        mandate_defaults={
            "cheap_model": "auto",
            "cheap_model_hints": ["mini"],
        },
    )
    assert svc.get_cheap_model_selector() == "openai/gpt-5-mini"


def test_get_cheap_model_backcompat_returns_bare_model():
    """The deprecated ``get_cheap_model()`` still returns a bare id for
    legacy callers, but the structural broadcast bug is only fully gone
    when callers migrate to ``get_cheap_model_selector()``."""
    svc = _make_service(
        providers=[_route("openai", "mini", "gpt-5-mini")],
        mandate_defaults={"cheap_model_hints": ["mini"]},
    )
    assert svc.get_cheap_model() == "gpt-5-mini"
