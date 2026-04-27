"""Tests for path-safety primitives (#834).

Path safety is now narrowly scoped: traversal segments and NUL bytes
are intrinsic violations; symlinks are *resolved*, not rejected.
Allow-list / deny-list membership is a policy decision, not a safety
decision — see test_computer_use_policy.
"""

import os
from pathlib import Path

import pytest

from kestrel_sovereign.features.computer_use.path_safety import (
    PathSafetyError,
    assert_no_traversal,
    match_allow_list,
    resolve_realpath,
)


def test_no_traversal_rejects_dotdot():
    with pytest.raises(PathSafetyError):
        assert_no_traversal("../etc/passwd")
    with pytest.raises(PathSafetyError):
        assert_no_traversal("foo/../bar")


def test_no_traversal_rejects_nul():
    with pytest.raises(PathSafetyError):
        assert_no_traversal("foo\x00.txt")


def test_no_traversal_allows_clean_paths():
    assert_no_traversal("foo/bar/baz")
    assert_no_traversal("/abs/path/file.txt")
    assert_no_traversal("./relative/path")  # leading "./" normalizes away


def test_resolve_realpath_canonicalizes(tmp_path: Path):
    target = tmp_path / "target.txt"
    target.write_text("hi")
    resolved = resolve_realpath(str(target))
    assert resolved == Path(os.path.realpath(target))


def test_resolve_realpath_follows_symlink(tmp_path: Path):
    """A symlink resolves to its target — the policy layer decides what to do with it."""
    real = tmp_path / "real.txt"
    real.write_text("payload")
    link = tmp_path / "link"
    os.symlink(real, link)
    resolved = resolve_realpath(str(link))
    assert resolved == Path(os.path.realpath(real))


def test_resolve_realpath_rejects_traversal(tmp_path: Path):
    with pytest.raises(PathSafetyError):
        resolve_realpath(f"{tmp_path}/../escape.txt")


def test_resolve_realpath_handles_nonexistent_leaf(tmp_path: Path):
    """Non-existent leaves still get a sensible realpath (parent realpath + leaf)."""
    resolved = resolve_realpath(str(tmp_path / "does_not_exist_yet.txt"))
    assert str(resolved).startswith(os.path.realpath(tmp_path))


def test_resolve_realpath_expands_user():
    home = Path.home()
    resolved = resolve_realpath("~/some-nonexistent-marker-file")
    assert str(resolved).startswith(str(home.resolve()))


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
