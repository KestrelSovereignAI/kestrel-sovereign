from __future__ import annotations

import logging

import pytest
import toml

from kestrel_sovereign import paths
from kestrel_sovereign.config import load_config
from kestrel_sovereign.setup.migrate_config import migrate_config


MODEL_MANDATE = {
    "defaults": {
        "preferred": "",
        "cheap_model": "auto",
        "cheap_model_hints": ["haiku", "mini"],
        "banned": [],
    },
    "mandates": {
        "summarize": "cheap",
        "creative": "anthropic",
    },
}

MODEL_CATALOG = {
    "hidden": {"openai": ["bad-model"]},
    "categories": {
        "embedding": {"openai": ["text-embedding-3-small"]},
        "completion": {"openai": ["babbage-002"]},
    },
    "route_context_caps": {"openai:plan": 20480},
    "context_limits_override": {"gpt-5": 128000},
    "display_name_overrides": {"gpt-5": "GPT-5"},
    "size_tiers": {"openai": {"small": "gpt-5-mini", "large": "gpt-5"}},
}


@pytest.fixture(autouse=True)
def _reset_paths_cache():
    paths.reset_cache()
    yield
    paths.reset_cache()


def _pin_project_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
    paths.reset_cache()


def test_model_mandate_legacy_and_migrated_unified_load_identically(
    tmp_path, monkeypatch
):
    _pin_project_dir(tmp_path, monkeypatch)
    (tmp_path / "kestrel.toml").write_text("[agent]\nname = 'Test'\n")
    (tmp_path / "model_mandate.toml").write_text(toml.dumps(MODEL_MANDATE))

    legacy = load_config("model_mandate.toml")

    result = migrate_config(tmp_path)
    assert result.action == "migrated"
    (tmp_path / "model_mandate.toml").unlink()
    migrated = load_config("model_mandate.toml")

    assert migrated == legacy


def test_model_catalog_legacy_and_migrated_unified_load_identically(
    tmp_path, monkeypatch
):
    _pin_project_dir(tmp_path, monkeypatch)
    (tmp_path / "kestrel.toml").write_text("[agent]\nname = 'Test'\n")
    (tmp_path / "model_catalog.toml").write_text(toml.dumps(MODEL_CATALOG))

    legacy = load_config("model_catalog.toml")

    result = migrate_config(tmp_path)
    assert result.action == "migrated"
    (tmp_path / "model_catalog.toml").unlink()
    migrated = load_config("model_catalog.toml")

    assert migrated == legacy


def test_unified_model_config_emits_no_deprecation_and_does_not_recreate(
    tmp_path, monkeypatch, caplog
):
    _pin_project_dir(tmp_path, monkeypatch)
    (tmp_path / "kestrel.toml").write_text(toml.dumps({
        "llm": {
            "mandate": MODEL_MANDATE,
            "catalog": MODEL_CATALOG,
        },
    }))
    (tmp_path / "model_mandate.toml.example").write_text(
        toml.dumps({"defaults": {"preferred": "example-only"}})
    )
    (tmp_path / "model_catalog.toml.example").write_text(
        toml.dumps({"hidden": {"openai": ["example-only"]}})
    )

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.config")

    assert load_config("model_mandate.toml") == MODEL_MANDATE
    assert load_config("model_catalog.toml") == MODEL_CATALOG

    assert "DEPRECATION" not in caplog.text
    assert not (tmp_path / "model_mandate.toml").exists()
    assert not (tmp_path / "model_catalog.toml").exists()


def test_missing_mapped_legacy_file_is_not_recreated_when_kestrel_toml_exists(
    tmp_path, monkeypatch
):
    _pin_project_dir(tmp_path, monkeypatch)
    (tmp_path / "kestrel.toml").write_text("[agent]\nname = 'Test'\n")
    (tmp_path / "model_mandate.toml.example").write_text(
        toml.dumps(MODEL_MANDATE)
    )

    assert load_config("model_mandate.toml") == {}
    assert not (tmp_path / "model_mandate.toml").exists()


def test_migrate_config_is_idempotent_and_preserves_existing_kestrel_content(
    tmp_path,
):
    (tmp_path / "kestrel.toml").write_text(toml.dumps({
        "agent": {"name": "Existing"},
        "llm": {"route_priority": ["ollama:local"]},
    }))
    (tmp_path / "model_mandate.toml").write_text(toml.dumps(MODEL_MANDATE))
    (tmp_path / "model_catalog.toml").write_text(toml.dumps(MODEL_CATALOG))

    first = migrate_config(tmp_path)
    second = migrate_config(tmp_path)
    parsed = toml.loads((tmp_path / "kestrel.toml").read_text())

    assert first.action == "migrated"
    assert first.backup_path is not None
    assert second.action == "already_clean"
    assert parsed["agent"]["name"] == "Existing"
    assert parsed["llm"]["route_priority"] == ["ollama:local"]
    assert parsed["llm"]["mandate"] == MODEL_MANDATE
    assert parsed["llm"]["catalog"] == MODEL_CATALOG
