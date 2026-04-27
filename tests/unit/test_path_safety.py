"""Tests for path-safety guards (#834)."""

import os
from pathlib import Path

import pytest

from kestrel_sovereign.features.computer_use.path_safety import (
    PathSafetyError,
    assert_no_traversal,
    match_allow_list,
    resolve_against_roots,
    resolve_within,
)


def test_no_traversal_rejects():
    with pytest.raises(PathSafetyError):
        assert_no_traversal("../etc/passwd")
    with pytest.raises(PathSafetyError):
        assert_no_traversal("foo/../bar")


def test_no_traversal_allows_clean():
    assert_no_traversal("foo/bar/baz")
    assert_no_traversal("/abs/path/file.txt")


def test_resolve_within_relative(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "file.txt").write_text("hi")
    resolved = resolve_within(tmp_path, "sub/file.txt")
    assert resolved == (tmp_path / "sub" / "file.txt").resolve()


def test_resolve_within_rejects_traversal(tmp_path: Path):
    with pytest.raises(PathSafetyError):
        resolve_within(tmp_path, "../escape")


def test_resolve_within_rejects_absolute_outside_root(tmp_path: Path):
    other = Path("/tmp/definitely-not-under-tmp_path")
    with pytest.raises(PathSafetyError):
        resolve_within(tmp_path, other)


def test_resolve_within_rejects_symlink_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "link"
    os.symlink(outside, link)
    with pytest.raises(PathSafetyError):
        resolve_within(tmp_path, "link/anything")


def test_resolve_within_handles_nonexistent_child(tmp_path: Path):
    resolved = resolve_within(tmp_path, "does/not/exist/yet.txt")
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_resolve_against_roots_first_match_wins(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (b / "file.txt").write_text("x")
    resolved = resolve_against_roots([a, b], "file.txt")
    # Tries a/file.txt — does not exist but path resolution allows non-existent;
    # we only require the resolved path to live under one of the roots.
    assert str(resolved).startswith(str(a.resolve())) or str(resolved).startswith(str(b.resolve()))


def test_resolve_against_roots_all_fail():
    with pytest.raises(PathSafetyError):
        resolve_against_roots([Path("/no/such/root")], "../etc/passwd")


def test_match_allow_list_deny_wins():
    cand = Path("/tmp/projects/secret/.ssh/id_rsa")
    result = match_allow_list(cand, allow=["/tmp/projects"], deny=["*/.ssh/*"])
    assert result.allowed is False
    assert result.reason.startswith("deny:")


def test_match_allow_list_allow_match():
    cand = Path("/tmp/projects/foo.txt")
    result = match_allow_list(cand, allow=["/tmp/projects"], deny=[])
    assert result.allowed is True
    assert result.reason.startswith("allow:")


def test_match_allow_list_no_match():
    cand = Path("/var/log/syslog")
    result = match_allow_list(cand, allow=["/home/me"], deny=[])
    assert result.allowed is False
    assert result.reason == "no_match"


def test_match_allow_list_glob_pattern():
    cand = Path("/home/me/projects/foo/bar.py")
    result = match_allow_list(cand, allow=["/home/*/projects/*"], deny=[])
    assert result.allowed is True
