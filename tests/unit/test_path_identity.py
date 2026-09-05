"""Physical and spelling aliases cannot cross host custody boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel_sovereign.features.compute import python_delete_runtime
from kestrel_sovereign.security import path_identity


@pytest.mark.parametrize(
    ("module", "function_name"),
    (
        (path_identity, "paths_overlap_by_filesystem_identity"),
        (python_delete_runtime, "_paths_overlap_by_filesystem_identity"),
    ),
)
def test_future_descendants_below_aliased_existing_ancestors_overlap(
    tmp_path,
    monkeypatch,
    module,
    function_name,
):
    """Bind/mount aliases retain one identity below not-yet-created leaves."""

    real = (tmp_path / "real-parent").resolve()
    alias = (tmp_path / "mount-alias").resolve()
    real.mkdir()
    alias.mkdir()
    original_samefile = Path.samefile

    def samefile(candidate, other):
        first = Path(candidate)
        second = Path(other)
        if {first, second} == {real, alias}:
            return True
        return original_samefile(first, second)

    monkeypatch.setattr(Path, "samefile", samefile)

    overlap = getattr(module, function_name)
    assert overlap(real / "future" / "control", alias / "future")
    assert not overlap(real / "first", alias / "second")


@pytest.mark.parametrize(
    ("module", "function_name"),
    (
        (path_identity, "paths_overlap_by_filesystem_identity"),
        (python_delete_runtime, "_paths_overlap_by_filesystem_identity"),
    ),
)
def test_unicode_normalization_aliases_do_not_depend_on_case_behavior(
    tmp_path,
    monkeypatch,
    module,
    function_name,
):
    """NFC/NFD aliases remain protected on a case-sensitive volume."""

    monkeypatch.setattr(module, "_filesystem_is_case_insensitive", lambda _path: False)
    overlap = getattr(module, function_name)

    assert overlap(
        tmp_path / "\N{LATIN SMALL LETTER E WITH ACUTE}" / "control",
        tmp_path / "e\N{COMBINING ACUTE ACCENT}" / "control" / "child",
    )


@pytest.mark.parametrize(
    ("module", "function_name"),
    (
        (path_identity, "paths_overlap_by_filesystem_identity"),
        (python_delete_runtime, "_paths_overlap_by_filesystem_identity"),
    ),
)
def test_unknown_case_semantics_fail_closed_for_future_aliases(
    tmp_path,
    monkeypatch,
    module,
    function_name,
):
    """A volume with no case-bearing probe component grants no write path."""

    monkeypatch.setattr(module, "_alternate_case", lambda _name: None)

    assert module._filesystem_is_case_insensitive(tmp_path) is None
    overlap = getattr(module, function_name)
    assert overlap(
        tmp_path / "host-data" / "control.db",
        tmp_path / "HOST-DATA",
    )
    assert not overlap(
        tmp_path / "host-data" / "control.db",
        tmp_path / "agent-data",
    )


def test_equal_identity_fails_closed_when_case_semantics_are_unknown(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(path_identity, "_alternate_case", lambda _name: None)

    assert path_identity.paths_equal_by_filesystem_identity(
        tmp_path / "host-data" / "control.db",
        tmp_path / "HOST-DATA" / "control.db",
    )


def test_equal_identity_uses_aliased_ancestor_suffixes(tmp_path, monkeypatch):
    """Exact leaf identity gets the same mount-alias treatment as overlap."""

    real = (tmp_path / "real-parent").resolve()
    alias = (tmp_path / "mount-alias").resolve()
    real.mkdir()
    alias.mkdir()
    original_samefile = Path.samefile

    def samefile(candidate, other):
        first = Path(candidate)
        second = Path(other)
        if {first, second} == {real, alias}:
            return True
        return original_samefile(first, second)

    monkeypatch.setattr(Path, "samefile", samefile)

    assert path_identity.paths_equal_by_filesystem_identity(
        real / "future" / "control",
        alias / "future" / "control",
    )
    assert not path_identity.paths_equal_by_filesystem_identity(
        real / "future",
        alias / "future" / "control",
    )
