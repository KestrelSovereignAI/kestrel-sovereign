"""Unit tests for :mod:`kestrel_sovereign.paths`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kestrel_sovereign import paths


@pytest.fixture(autouse=True)
def _reset_paths_cache():
    paths.reset_cache()
    yield
    paths.reset_cache()


def test_explicit_kestrel_home_overrides_everything(tmp_path, monkeypatch):
    """``KESTREL_HOME`` is the highest-priority signal."""
    home = tmp_path / "explicit"
    home.mkdir()
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    # Even with a project marker in CWD, the explicit env wins.
    (tmp_path / "multi_agent.toml").write_text("")

    assert paths.project_dir() == home.resolve()


def test_kestrel_home_expands_user_and_resolves(tmp_path, monkeypatch):
    """``~`` and relative paths in ``KESTREL_HOME`` are normalised."""
    home = tmp_path / "abs"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KESTREL_HOME", "~/abs")
    monkeypatch.chdir(tmp_path)

    assert paths.project_dir() == home.resolve()


def test_marker_in_cwd_wins(tmp_path, monkeypatch):
    """``multi_agent.toml`` in CWD is recognised as a project root."""
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    (tmp_path / "multi_agent.toml").write_text("")
    monkeypatch.chdir(tmp_path)

    assert paths.project_dir() == tmp_path.resolve()


def test_marker_in_ancestor_wins(tmp_path, monkeypatch):
    """Walks up the tree until it finds a project marker."""
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    (tmp_path / "kestrel.toml").write_text("")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    assert paths.project_dir() == tmp_path.resolve()


def test_kestrel_sovereign_source_dir_is_a_marker(tmp_path, monkeypatch):
    """A directory containing the package source counts as a project root,
    even before the wizard has generated any of the marker files."""
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    pkg = tmp_path / "kestrel_sovereign"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    monkeypatch.chdir(tmp_path)

    assert paths.project_dir() == tmp_path.resolve()


def test_pip_install_with_no_markers_falls_back_to_home_kestrel(
    tmp_path, monkeypatch
):
    """When the package lives in ``site-packages`` and no marker is
    found anywhere in the CWD ancestry, the resolver lands on
    ``~/.kestrel`` and creates it."""
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    # Make ``_is_inside_site_packages`` think we're pip-installed.
    monkeypatch.setattr(paths, "_is_inside_site_packages", lambda _p: True)

    resolved = paths.project_dir()

    assert resolved == (fake_home / ".kestrel").resolve()
    assert resolved.is_dir()


def test_source_clone_uses_package_parent_when_no_markers(tmp_path, monkeypatch):
    """Editable installs (package not in site-packages) keep the historical
    behaviour of using ``package_dir().parent`` so ``cd kestrel-sovereign &&
    kestrel start`` still works on a freshly-cloned repo."""
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    fake_pkg = tmp_path / "repo" / "kestrel_sovereign"
    fake_pkg.mkdir(parents=True)
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    monkeypatch.setattr(paths, "package_dir", lambda: fake_pkg)
    monkeypatch.setattr(paths, "_is_inside_site_packages", lambda _p: False)

    assert paths.project_dir() == fake_pkg.parent.resolve()


def test_resolver_is_cached_per_inputs(tmp_path, monkeypatch):
    """Two consecutive calls with identical inputs hit the cache."""
    home = tmp_path / "h"
    home.mkdir()
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    paths.project_dir()
    info_before = paths._resolve_cached.cache_info()
    paths.project_dir()
    info_after = paths._resolve_cached.cache_info()

    assert info_after.hits == info_before.hits + 1


def test_chdir_invalidates_cache_via_input_change(tmp_path, monkeypatch):
    """``project_dir`` after a ``chdir`` re-resolves because CWD is part of
    the cache key."""
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    a = tmp_path / "a"
    a.mkdir()
    (a / "kestrel.toml").write_text("")
    b = tmp_path / "b"
    b.mkdir()
    (b / "kestrel.toml").write_text("")

    monkeypatch.chdir(a)
    assert paths.project_dir() == a.resolve()
    monkeypatch.chdir(b)
    assert paths.project_dir() == b.resolve()


def test_is_inside_site_packages_detects_real_install_layouts():
    """Black-box check on the segment-name heuristic."""
    assert paths._is_inside_site_packages(
        Path("/Users/foo/.venv/lib/python3.13/site-packages/kestrel_sovereign")
    )
    assert paths._is_inside_site_packages(
        Path("/usr/lib/python3/dist-packages/foo")  # noqa: ERA001
    ) is False  # dist-packages is Debian's dir; we accept that it's *not* caught.
    assert not paths._is_inside_site_packages(
        Path("/Users/foo/projects/kestrel-sovereign/kestrel_sovereign")
    )
    assert paths._is_inside_site_packages(
        Path("/opt/homebrew/lib/python3.13/site-packages/kestrel_sovereign")
    )


def test_package_dir_points_at_real_package():
    """``package_dir`` is the directory containing this very file."""
    assert paths.package_dir() == Path(paths.__file__).resolve().parent
    assert (paths.package_dir() / "paths.py").exists()
