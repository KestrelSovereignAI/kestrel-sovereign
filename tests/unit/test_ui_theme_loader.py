"""Unit tests for the UI theme loader (epic #986, sub-issue #989)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kestrel_sovereign.ui import theme_loader
from kestrel_sovereign.ui.theme_loader import (
    DEFAULT_LOCALE,
    DEFAULT_THEME,
    ThemeNotFoundError,
    list_available_themes,
    load_theme,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    theme_loader.clear_cache()
    yield
    theme_loader.clear_cache()


@pytest.fixture
def isolated_themes(tmp_path, monkeypatch):
    """Redirect THEMES_DIR to a temp dir so tests can build deliberately
    malformed or partial theme sets without touching the real files."""
    themes = tmp_path / "themes"
    themes.mkdir()
    monkeypatch.setattr(theme_loader, "THEMES_DIR", themes)
    theme_loader.clear_cache()
    return themes


def _write_theme(root: Path, theme: str, locale: str, labels: dict[str, str]) -> None:
    d = root / theme
    d.mkdir(parents=True, exist_ok=True)
    body = [f'schema_version = "1"', f'theme = "{theme}"', f'locale = "{locale}"', "", "[labels]"]
    body.extend(f'{k} = {v!r}' for k, v in labels.items())
    (d / f"{locale}.toml").write_text("\n".join(body) + "\n", encoding="utf-8")


# ---- Real-file tests (legacy/falconry/plain × en) --------------------------


def test_load_legacy_returns_full_label_map():
    bundle = load_theme("legacy", "en")
    assert bundle.theme == "legacy"
    assert bundle.locale == "en"
    assert bundle.fallback_keys == ()
    assert bundle.labels["tab_identity"] == "Identity"
    assert bundle.labels["sidebar_agents"] == "Agents"
    assert bundle.labels["spawn_title"] == "Spawn Manager"


def test_load_falconry_overrides_legacy_on_diverging_keys():
    bundle = load_theme("falconry", "en")
    assert bundle.theme == "falconry"
    assert bundle.labels["sidebar_agents"] == "Mews"
    assert bundle.labels["spawn_title"] == "Hatchery"
    assert bundle.labels["tab_resources"] == "Equipage"
    # Identical-to-legacy keys still present
    assert bundle.labels["tab_identity"] == "Identity"
    # All three theme files have identical key sets, so no fallback
    assert bundle.fallback_keys == ()


def test_load_plain_overrides_legacy_on_diverging_keys():
    bundle = load_theme("plain", "en")
    assert bundle.theme == "plain"
    assert bundle.labels["spawn_title"] == "Multi-Agent"
    assert bundle.labels["memories_title"] == "Memories"
    assert bundle.labels["sidebar_agents"] == "Agents"
    assert bundle.fallback_keys == ()


def test_unknown_theme_raises():
    with pytest.raises(ThemeNotFoundError):
        load_theme("nonexistent", "en")


def test_list_available_themes_includes_shipped_themes():
    available = list_available_themes()
    assert "legacy" in available
    assert "falconry" in available
    assert "plain" in available


def test_default_theme_and_locale_constants():
    assert DEFAULT_THEME == "legacy"
    assert DEFAULT_LOCALE == "en"


def test_load_theme_caches_results():
    a = load_theme("legacy", "en")
    b = load_theme("legacy", "en")
    assert a is b  # cached identity


# ---- Isolated-fixture tests for fallback semantics --------------------------


def test_legacy_locale_fallback_to_en(isolated_themes, caplog):
    _write_theme(isolated_themes, "legacy", "en", {"foo": "Foo", "bar": "Bar"})
    # No legacy/es.toml → falls back to legacy/en.toml
    with caplog.at_level(logging.WARNING):
        bundle = load_theme("legacy", "es")
    assert bundle.locale == "en"
    assert bundle.labels == {"foo": "Foo", "bar": "Bar"}
    assert bundle.fallback_keys == ()
    assert any("locale_fallback" in rec.message for rec in caplog.records)


def test_partial_theme_falls_back_to_legacy_per_key(isolated_themes, caplog):
    _write_theme(isolated_themes, "legacy", "en", {
        "foo": "Foo", "bar": "Bar", "baz": "Baz",
    })
    # Falconry only overrides one key
    _write_theme(isolated_themes, "falconry", "en", {"foo": "Mews"})

    with caplog.at_level(logging.WARNING):
        bundle = load_theme("falconry", "en")

    assert bundle.theme == "falconry"
    assert bundle.labels["foo"] == "Mews"
    assert bundle.labels["bar"] == "Bar"  # filled from legacy
    assert bundle.labels["baz"] == "Baz"  # filled from legacy
    assert bundle.fallback_keys == ("bar", "baz")
    assert any("key_fallback" in rec.message for rec in caplog.records)


def test_unknown_theme_in_isolated_dir_raises(isolated_themes):
    _write_theme(isolated_themes, "legacy", "en", {"x": "X"})
    with pytest.raises(ThemeNotFoundError):
        load_theme("doesnotexist", "en")


def test_theme_with_no_locale_files_raises(isolated_themes):
    # Create theme directory but no toml file inside
    (isolated_themes / "broken").mkdir()
    with pytest.raises(ThemeNotFoundError):
        load_theme("broken", "en")


def test_clear_cache_drops_results(isolated_themes):
    _write_theme(isolated_themes, "legacy", "en", {"x": "X"})
    a = load_theme("legacy", "en")
    theme_loader.clear_cache()
    b = load_theme("legacy", "en")
    assert a == b
    assert a is not b  # cache was cleared


# ---- Path traversal hardening ----------------------------------------------


@pytest.mark.parametrize("bad_theme", [
    "..",
    "../etc",
    "../../tmp",
    "legacy/../legacy",
    "legacy\x00",
    "../",
    "/etc",
    "legacy.",
    ".legacy",
    "legacy ",
    "",
])
def test_path_traversal_in_theme_rejected(bad_theme):
    with pytest.raises(ThemeNotFoundError):
        load_theme(bad_theme, "en")


@pytest.mark.parametrize("bad_locale", [
    "..",
    "../passwd",
    "../../etc/passwd",
    "en/../en",
    "en\x00",
    "/en",
    "en.",
    ".en",
    "en ",
    "",
])
def test_path_traversal_in_locale_rejected(bad_locale):
    with pytest.raises(ThemeNotFoundError):
        load_theme("legacy", bad_locale)


def test_safe_locale_with_region_accepted():
    """ISO 639-1 + region tag (en-US) should pass the validator,
    even though there's no en-US theme file (it'll fall back to en)."""
    bundle = load_theme("legacy", "en-US")
    # Falls back to en because en-US.toml doesn't ship
    assert bundle.locale == "en"
