"""Contracts for mandate selector resolution and shipped config shape."""

from pathlib import Path
from unittest.mock import MagicMock

import tomllib

from kestrel_sovereign.llm.service import LLMService


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bind_service_method(service, method_name):
    method = getattr(LLMService, method_name)
    return method.__get__(service)


def test_resolve_model_selector_maps_provider_name_to_current_model():
    service = MagicMock(spec=LLMService)
    service.providers = [
        {"name": "openai", "model": "gpt-5.1"},
        {"name": "anthropic", "model": "claude-sonnet-4-6"},
    ]
    service._resolve_model_selector = _bind_service_method(service, "_resolve_model_selector")

    resolved = service._resolve_model_selector("anthropic")

    assert resolved == {
        "selector": "anthropic/claude-sonnet-4-6",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
    }


def test_resolve_model_selector_maps_cheap_alias_via_cheap_model_policy():
    service = MagicMock(spec=LLMService)
    service.providers = [{"name": "anthropic", "model": "claude-sonnet-4-6"}]
    service.get_cheap_model = MagicMock(return_value="claude-haiku-4-5")
    service._resolve_model_selector = _bind_service_method(service, "_resolve_model_selector")

    resolved = service._resolve_model_selector("cheap")

    assert resolved == {
        "selector": "claude-haiku-4-5",
        "provider": None,
        "model": "claude-haiku-4-5",
    }


def test_get_model_for_prompt_normalizes_provider_selectors():
    service = MagicMock(spec=LLMService)
    service.providers = [
        {"name": "openai", "model": "gpt-5.1"},
        {"name": "anthropic", "model": "claude-sonnet-4-6"},
    ]
    service.mandate_config = {
        "defaults": {"preferred": "openai", "banned": []},
        "mandates": {"code": "anthropic"},
    }
    service._resolve_model_selector = _bind_service_method(service, "_resolve_model_selector")
    service._get_default_mandate_selector = _bind_service_method(service, "_get_default_mandate_selector")
    service._is_banned_selector = _bind_service_method(service, "_is_banned_selector")
    service._get_model_for_prompt = _bind_service_method(service, "_get_model_for_prompt")

    assert service._get_model_for_prompt("please help with code review") == "anthropic/claude-sonnet-4-6"
    assert service._get_model_for_prompt("hello there") == "openai/gpt-5.1"


def test_shipped_mandate_configs_use_discovery_backed_cheap_policy():
    example_config = tomllib.loads((PROJECT_ROOT / "model_mandate.toml.example").read_text(encoding="utf-8"))
    unified_config = tomllib.loads((PROJECT_ROOT / "kestrel.toml.example").read_text(encoding="utf-8"))

    for config in [
        example_config["defaults"],
        unified_config["llm"]["mandate"]["defaults"],
    ]:
        assert "feedback_audit_model" not in config
        assert config["cheap_model"] == "auto"
        assert config["cheap_model_hints"]
