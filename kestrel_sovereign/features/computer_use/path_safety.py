"""Path-safety guards for the computer-use feature.

Pure ``pathlib`` + ``os.path.realpath``. No I/O beyond resolution. The job
is to make every path the agent supplies safe to hand to a backend by:

- forbidding ``..`` traversal segments,
- resolving symlinks and asserting the realpath stays under a configured
  root,
- matching the resolved path against an allow/deny list with deny-wins
  glob semantics.

Callers should always go through :func:`resolve_within` (or
:func:`resolve_against_roots`) before any open/stat/exec, and then through
:func:`match_allow_list` to decide whether the operation is permitted.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


class PathSafetyError(Exception):
    """Raised when a candidate path violates a safety guard."""


def assert_no_traversal(candidate: str | os.PathLike) -> None:
    """Reject any path containing ``..`` segments after normalization."""
    parts = PurePosixPath(str(candidate).replace(os.sep, "/")).parts
    if any(p == ".." for p in parts):
        raise PathSafetyError(f"path contains traversal segment: {candidate!r}")


def assert_no_symlink_escape(root: Path, candidate: Path) -> None:
    """Reject if the realpath of ``candidate`` is not under ``root``.

    ``candidate`` does not need to exist; we resolve as much as we can and
    walk up to the nearest existing ancestor for the symlink check.
    """
    real_root = Path(os.path.realpath(root))
    cur = candidate
    while True:
        if cur.exists() or cur.is_symlink():
            real = Path(os.path.realpath(cur))
            try:
                real.relative_to(real_root)
            except ValueError:
                raise PathSafetyError(
                    f"symlink escape: {candidate} resolves to {real} "
                    f"outside root {real_root}"
                )
            return
        if cur.parent == cur:
            return
        cur = cur.parent


def resolve_within(root: Path | str, candidate: str | os.PathLike) -> Path:
    """Resolve ``candidate`` and assert it lives under ``root``.

    Returns the absolute, symlink-resolved path. Raises
    :class:`PathSafetyError` on any violation.
    """
    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = root_path.resolve()
    assert_no_traversal(candidate)

    cand = Path(candidate).expanduser()
    if not cand.is_absolute():
        cand = (root_path / cand)
    cand = cand.resolve(strict=False) if cand.exists() else _best_effort_resolve(cand)

    real_root = Path(os.path.realpath(root_path))
    try:
        cand.relative_to(real_root)
    except ValueError:
        raise PathSafetyError(
            f"path escapes root: {cand} not under {real_root}"
        )
    assert_no_symlink_escape(root_path, cand)
    return cand


def resolve_against_roots(roots: Iterable[Path | str], candidate: str | os.PathLike) -> Path:
    """Resolve ``candidate`` and assert it lives under at least one root.

    Returns the resolved absolute path. The first root that contains the
    resolved candidate wins. Raises :class:`PathSafetyError` if the
    candidate escapes every root.
    """
    last_err: PathSafetyError | None = None
    for root in roots:
        try:
            return resolve_within(root, candidate)
        except PathSafetyError as e:
            last_err = e
    if last_err is None:
        raise PathSafetyError("no roots configured")
    raise last_err


def _best_effort_resolve(path: Path) -> Path:
    """Resolve a non-existent path by walking up to the deepest existing parent."""
    parts: list[str] = []
    cur = path
    while not cur.exists() and cur.parent != cur:
        parts.append(cur.name)
        cur = cur.parent
    base = Path(os.path.realpath(cur)) if cur.exists() else cur
    for name in reversed(parts):
        base = base / name
    return base


@dataclass(frozen=True)
class MatchResult:
    """Outcome of an allow/deny match."""

    allowed: bool
    reason: str  # "deny:<pat>", "allow:<pat>", or "no_match"


def match_allow_list(
    candidate: Path,
    allow: Iterable[str | Path],
    deny: Iterable[str | Path],
) -> MatchResult:
    """Match ``candidate`` against allow + deny lists. Deny wins.

    Patterns may be:
    - absolute paths (must be a prefix of the candidate)
    - tilde-prefixed paths (expanded)
    - glob patterns (``fnmatch`` semantics)

    Returns :class:`MatchResult`. ``allowed=False`` on no_match — callers
    decide whether to require approval or hard-reject.
    """
    cand_str = str(candidate)
    for pat in deny:
        if _matches(cand_str, pat):
            return MatchResult(False, f"deny:{pat}")
    for pat in allow:
        if _matches(cand_str, pat):
            return MatchResult(True, f"allow:{pat}")
    return MatchResult(False, "no_match")


def _matches(candidate: str, pattern: str | Path) -> bool:
    pat_str = str(pattern)
    expanded = os.path.expanduser(pat_str)
    if any(c in expanded for c in "*?["):
        return fnmatch.fnmatchcase(candidate, expanded) or fnmatch.fnmatchcase(
            candidate, expanded.rstrip("/") + "/*"
        )
    expanded = os.path.realpath(expanded) if os.path.exists(expanded) else expanded
    expanded = expanded.rstrip("/")
    return candidate == expanded or candidate.startswith(expanded + "/")
