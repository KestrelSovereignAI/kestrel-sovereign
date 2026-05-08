"""Regression tests for the empty-[llm] startup warning.

Filed after a real-world incident on 2026-05-04: PR #944 (close of epic
#938) deleted ``llm_config.toml`` on the assumption that users had run
``kestrel migrate-llm-config`` first. A user who hadn't came back from
the merge to find their UI showing empty Provider and "loading" Model
with no clear log signal — ``LLMService`` had silently initialised with
zero providers. The warning here exists so that case is loud."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kestrel_sovereign.llm.service import _warn_no_llm_config_found


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    """Chdir into an empty tmp dir AND pin ``KESTREL_HOME`` to it.

    Without ``KESTREL_HOME`` the project-dir resolver
    (:func:`kestrel_sovereign.paths.project_dir`) walks up from CWD looking
    for a ``kestrel.toml`` marker — which it finds in the dev repo above the
    ``tmp_path`` pytest gives us. That makes ``load_section('llm')`` read the
    dev config instead of returning empty, and the empty-[llm] warning the
    tests in this file are checking for never fires."""
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    from kestrel_sovereign import paths
    paths.reset_cache()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    paths.reset_cache()


def test_warn_includes_no_section_message(cwd, caplog):
    with caplog.at_level(logging.WARNING):
        _warn_no_llm_config_found()
    assert "no [llm] section" in caplog.text
    assert "model selector will appear empty" in caplog.text


def test_warn_points_at_migrate_command_when_legacy_file_exists(cwd, caplog):
    """Most informative hint: legacy llm_config.toml is sitting right there
    and the user just hasn't migrated."""
    (cwd / "llm_config.toml").write_text('route_priority = ["ollama:local"]\n')
    with caplog.at_level(logging.WARNING):
        _warn_no_llm_config_found()
    assert "kestrel migrate-llm-config" in caplog.text
    assert "Legacy llm_config.toml detected" in caplog.text


def test_warn_handles_post_migration_bak_case(cwd, caplog):
    """Subtler case: migration ran but produced an empty [llm] (e.g. the
    source was empty or got corrupted). Hint should point at the .bak."""
    (cwd / "llm_config.toml.bak").write_text('route_priority = []\n')
    with caplog.at_level(logging.WARNING):
        _warn_no_llm_config_found()
    assert "llm_config.toml.bak detected" in caplog.text
    assert "kestrel migrate-llm-config --force" in caplog.text


def test_warn_recommends_setup_or_example_when_neither_exists(cwd, caplog):
    """Fresh-checkout case: no legacy file, but the example is present."""
    (cwd / "kestrel.toml.example").write_text("# example\n")
    with caplog.at_level(logging.WARNING):
        _warn_no_llm_config_found()
    assert "kestrel setup llm" in caplog.text
    assert "cp kestrel.toml.example kestrel.toml" in caplog.text


def test_warn_minimal_hint_when_nothing_present(cwd, caplog):
    """No legacy, no .bak, no example — just say to run setup."""
    with caplog.at_level(logging.WARNING):
        _warn_no_llm_config_found()
    assert "kestrel setup llm" in caplog.text
    # Specifically should NOT mention migration or example copy in this branch.
    assert "migrate-llm-config" not in caplog.text
    assert "kestrel.toml.example" not in caplog.text


def test_llmservice_init_emits_warning_when_config_section_empty(cwd, caplog):
    """End-to-end: an LLMService constructed in a directory with no
    kestrel.toml (and no legacy file) must emit the warning during init.
    Otherwise the silent-failure regression returns."""
    from unittest.mock import patch
    from kestrel_sovereign.llm import service as svc_mod

    # Skip the heavy mixin init paths — we only care that the warning fires
    # before the registry tries to initialize zero providers.
    with caplog.at_level(logging.WARNING):
        with patch.object(svc_mod, "ProviderRegistry") as mock_registry, \
             patch.object(svc_mod.LLMService, "_load_from_disk_cache"), \
             patch.object(svc_mod.LLMService, "_init_usage_tracking"), \
             patch.object(svc_mod.LLMService, "_init_constitutional_profiles"):
            mock_registry.return_value.initialize_providers.return_value = []
            svc_mod.LLMService()

    assert "no [llm] section" in caplog.text


def test_llmservice_init_does_not_warn_when_config_present(tmp_path, monkeypatch, caplog):
    """Negative case: a populated [llm] section must NOT trigger the warning."""
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    from kestrel_sovereign import paths
    paths.reset_cache()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kestrel.toml").write_text(
        '[llm]\n'
        'route_priority = ["ollama:local"]\n'
        '[llm.vendors.ollama]\n'
        'is_cloud = false\n'
        '[llm.vendors.ollama.routes.local]\n'
        'adapter = "OllamaAdapter"\n'
        'host = "http://localhost:11434"\n'
        'model = "auto"\n'
    )

    from unittest.mock import patch
    from kestrel_sovereign.llm import service as svc_mod

    with caplog.at_level(logging.WARNING):
        with patch.object(svc_mod, "ProviderRegistry") as mock_registry, \
             patch.object(svc_mod.LLMService, "_load_from_disk_cache"), \
             patch.object(svc_mod.LLMService, "_init_usage_tracking"), \
             patch.object(svc_mod.LLMService, "_init_constitutional_profiles"):
            mock_registry.return_value.initialize_providers.return_value = []
            svc_mod.LLMService()

    assert "no [llm] section" not in caplog.text
