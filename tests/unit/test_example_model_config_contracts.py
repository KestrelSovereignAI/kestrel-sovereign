"""Contracts for example model config files and catalog source-of-truth shape."""

from pathlib import Path

import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_llm_config_example_uses_auto_models_and_selection_hints():
    config = tomllib.loads((PROJECT_ROOT / "llm_config.toml.example").read_text(encoding="utf-8"))

    assert config["openai"]["model"] == "auto"
    assert config["openai"]["selection_hints"]
    assert config["ollama"]["model"] == "auto"
    assert config["ollama"]["selection_hints"]


def test_unified_example_uses_auto_models_for_primary_llm_providers():
    config = tomllib.loads((PROJECT_ROOT / "kestrel.toml.example").read_text(encoding="utf-8"))

    assert config["llm"]["openai"]["model"] == "auto"
    assert config["llm"]["openai"]["selection_hints"]
    assert config["llm"]["ollama"]["model"] == "auto"
    assert config["llm"]["ollama"]["selection_hints"]


def test_root_model_catalog_is_manual_overrides_only():
    catalog_text = (PROJECT_ROOT / "model_catalog.toml").read_text(encoding="utf-8")
    catalog = tomllib.loads(catalog_text)

    assert "Manual Overrides Only" in catalog_text
    assert "featured" not in catalog
    assert "display_name_overrides" in catalog
    assert "context_limits_override" in catalog


def test_package_local_model_catalog_duplicate_is_removed():
    assert not (PROJECT_ROOT / "kestrel_sovereign/model_catalog.toml").exists()
