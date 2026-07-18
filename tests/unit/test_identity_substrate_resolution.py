"""Canonical active-substrate resolution contracts (#2603)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kestrel_sovereign.identity import (
    IdentityExporter,
    SubstrateType,
    resolve_active_substrate,
)


class _Adapter:
    def __init__(self, family):
        self.family = family

    def substrate_type(self):
        return self.family


def _service(*, vendor, route, model, family, preference=None):
    pref = preference or {"vendor": vendor, "route": route, "model": model}
    return SimpleNamespace(
        providers=[{
            "name": f"{vendor}:{route}",
            "vendor": vendor,
            "route": route,
            "model": model,
            "adapter": _Adapter(family),
        }],
        get_model_preference=lambda: dict(pref),
    )


@pytest.mark.parametrize(
    ("vendor", "route", "model", "family", "expected"),
    [
        ("anthropic", "api", "claude-opus-4", "claude", SubstrateType.ANTHROPIC_CLAUDE.value),
        ("openai", "api", "gpt-5", "gpt", SubstrateType.OPENAI_GPT.value),
        ("openai", "plan", "gpt-5.5", "gpt", SubstrateType.OPENAI_GPT.value),
        ("google", "api", "gemini-2.5-pro", "gemini", SubstrateType.GOOGLE_GEMINI.value),
        ("vertex_ai", "adc", "gemini-2.5-pro", "gemini", SubstrateType.GOOGLE_GEMINI.value),
        ("local", "api", "llama-3.3", "llama", SubstrateType.META_LLAMA.value),
        ("mistral", "api", "mistral-large", "mistral", SubstrateType.MISTRAL.value),
    ],
)
def test_adapter_family_maps_to_identity_substrate(
    vendor, route, model, family, expected
):
    resolution = resolve_active_substrate(
        _service(vendor=vendor, route=route, model=model, family=family)
    )

    assert resolution.substrate == expected
    assert resolution.provider_selector == f"{vendor}:{route}"
    assert resolution.model == model


def test_composite_preference_selects_matching_route():
    service = SimpleNamespace(
        providers=[
            {
                "name": "openai:api",
                "vendor": "openai",
                "route": "api",
                "model": "gpt-5-mini",
                "adapter": _Adapter("gpt"),
            },
            {
                "name": "anthropic:plan",
                "vendor": "anthropic",
                "route": "plan",
                "model": "claude-opus-4",
                "adapter": _Adapter("claude"),
            },
        ],
        get_model_preference=lambda: {
            "vendor": "anthropic",
            "route": "plan",
            "model": "claude-opus-4",
        },
    )

    resolution = resolve_active_substrate(service)

    assert resolution.substrate == SubstrateType.ANTHROPIC_CLAUDE.value
    assert resolution.provider_selector == "anthropic:plan"


@pytest.mark.parametrize(
    ("vendor", "model", "expected"),
    [
        ("openrouter", "anthropic/claude-3.7-sonnet", SubstrateType.ANTHROPIC_CLAUDE.value),
        ("openrouter", "openai/gpt-5", SubstrateType.OPENAI_GPT.value),
        ("openrouter", "google/gemini-2.5-pro", SubstrateType.GOOGLE_GEMINI.value),
        ("ollama", "llama3.3:70b", SubstrateType.META_LLAMA.value),
        ("ollama", "mistral-small", SubstrateType.MISTRAL.value),
    ],
)
def test_heterogeneous_route_uses_active_model_family(vendor, model, expected):
    resolution = resolve_active_substrate(
        _service(vendor=vendor, route="api", model=model, family=None)
    )

    assert resolution.substrate == expected


@pytest.mark.parametrize(
    ("vendor", "expected"),
    [
        ("openrouter", SubstrateType.OPENROUTER.value),
        ("ollama", SubstrateType.OLLAMA_LOCAL.value),
    ],
)
def test_heterogeneous_route_keeps_explicit_fallback_when_model_is_auto(
    vendor, expected
):
    resolution = resolve_active_substrate(
        _service(vendor=vendor, route="api", model="auto", family=None)
    )

    assert resolution.substrate == expected
    assert resolution.reason


def test_plugin_family_is_preserved_without_framework_vendor_branch():
    resolution = resolve_active_substrate(
        _service(vendor="moonshot", route="api", model="kimi-k2", family="kimi")
    )

    assert resolution.substrate == "kimi"
    assert resolution.adapter_family == "kimi"
    assert resolution.capability_profile_known is False
    assert resolution.reason == "capability_profile_unavailable"


def test_missing_runtime_is_explicit_unknown():
    resolution = resolve_active_substrate(None)

    assert resolution.substrate == SubstrateType.UNKNOWN.value
    assert resolution.reason == "llm_service_unavailable"


@pytest.mark.asyncio
async def test_exporter_uses_its_agent_llm_service_not_process_global_config():
    service = _service(
        vendor="anthropic",
        route="plan",
        model="claude-opus-4",
        family="claude",
    )
    agent = SimpleNamespace(llm_service=service)
    exporter = IdentityExporter(object(), "did:test:substrate", agent=agent)

    assert (
        await exporter._detect_substrate()
        == SubstrateType.ANTHROPIC_CLAUDE.value
    )
