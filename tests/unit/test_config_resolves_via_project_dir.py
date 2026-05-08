"""Regression tests: ``config.load_section`` and ``config.load_config``
must consult :func:`kestrel_sovereign.paths.project_dir` when resolving
``kestrel.toml`` and the legacy individual config files.

Pre-fix, both helpers used ``Path("kestrel.toml")`` (CWD-relative). When
the server subprocess inherited ``cwd=site-packages-parent`` from the
pip-installed CLI's old ``_get_project_dir()``, ``load_section('llm')``
returned an empty dict and ``LLMService`` came up with zero providers
even when the operator's ``kestrel.toml`` was perfectly valid in their
real project directory.

These tests cover the two angles:

  1. ``KESTREL_HOME`` lets us pin the resolver at an arbitrary tmp dir
     and verify writes there are visible to the loaders.
  2. CWD on its own is no longer enough to win — that's the whole point
     of the bug — but a CWD with marker files still wins via the
     resolver's ancestor-walk.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign import paths
from kestrel_sovereign.config import load_config, load_section


@pytest.fixture(autouse=True)
def _reset_paths_cache():
    paths.reset_cache()
    yield
    paths.reset_cache()


def test_load_section_reads_kestrel_toml_under_kestrel_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
    (tmp_path / "kestrel.toml").write_text(
        "[llm]\nroute_priority = ['ollama:local']\n"
    )

    out = load_section("llm")

    assert out == {"route_priority": ["ollama:local"]}


def test_load_section_returns_empty_when_no_kestrel_toml(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)

    assert load_section("llm") == {}


def test_load_section_walks_up_from_cwd_to_find_marker(tmp_path, monkeypatch):
    """No ``KESTREL_HOME``: a marker in an ancestor of CWD wins."""
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
    (tmp_path / "kestrel.toml").write_text(
        "[llm]\nroute_priority = ['anthropic:cloud']\n"
    )
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    paths.reset_cache()  # honor the new CWD

    out = load_section("llm")

    assert out["route_priority"] == ["anthropic:cloud"]


def test_load_config_legacy_file_lookup_uses_project_dir(tmp_path, monkeypatch):
    """``load_config`` falls back to a legacy individual TOML file. That
    fallback also needs to be anchored at the resolved project dir, not
    the operator's incidental CWD."""
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
    legacy = tmp_path / "model_mandate.toml"
    legacy.write_text(
        '[mandate]\ndefault_model = "test-model"\n'
    )

    cfg = load_config("model_mandate.toml", section="mandate")

    assert cfg.get("default_model") == "test-model"


def test_load_config_does_not_read_cwd_when_no_marker_in_path(
    tmp_path, monkeypatch
):
    """A bare CWD with a stray ``model_mandate.toml`` and no project
    markers must NOT be picked up — the operator hasn't asked us to
    treat that directory as a project."""
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
    # Pretend we're a pip install so the resolver falls through past
    # the source-clone branch to ~/.kestrel.
    monkeypatch.setattr(paths, "_is_inside_site_packages", lambda _p: True)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    work = tmp_path / "work"
    work.mkdir()
    (work / "model_mandate.toml").write_text(  # accidental file in CWD
        '[mandate]\ndefault_model = "should-not-load"\n'
    )
    monkeypatch.chdir(work)
    paths.reset_cache()

    cfg = load_config("model_mandate.toml", section="mandate")

    # Resolver lands on ~/.kestrel; no model_mandate.toml there → empty.
    assert cfg == {}
