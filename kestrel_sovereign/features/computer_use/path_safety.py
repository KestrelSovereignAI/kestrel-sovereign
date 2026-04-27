"""Path-safety guards for the computer-use feature.

Pure ``pathlib`` + ``os.path.realpath``. No I/O beyond resolution.

This module's only job is to canonicalize a candidate path:

- reject ``..`` traversal segments and embedded NULs (always unsafe),
- expand ``~`` and resolve symlinks,
- return the absolute realpath.

Whether the resulting realpath is **allowed** is a policy decision, not a
safety decision — see :mod:`policy`. Conflating the two (the v0 design)
made it impossible for a path "outside the allow-list" to ever reach the
approval queue, because the path-safety guard rejected it first. That
is a real threat-model bug — the realpath is exactly what the human
approver needs to see in the prompt, even (especially) for unusual paths.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


class PathSafetyError(Exception):
    """Raised when a candidate path is *intrinsically* unsafe.

    Reserved for unsafe-by-construction inputs: traversal segments and
    embedded NUL bytes. Allow-list / deny-list decisions belong to
    policy, not here.
    """


def assert_no_traversal(candidate: str | os.PathLike) -> None:
    """Reject any path containing ``..`` segments after normalization."""
    text = str(candidate)
    if "\x00" in text:
        raise PathSafetyError(f"path contains NUL byte: {candidate!r}")
    parts = PurePosixPath(text.replace(os.sep, "/")).parts
    if any(p == ".." for p in parts):
        raise PathSafetyError(f"path contains traversal segment: {candidate!r}")


def resolve_realpath(candidate: str | os.PathLike) -> Path:
    """Canonicalize ``candidate`` to an absolute realpath.

    Steps:

    1. :func:`assert_no_traversal` — reject ``..`` segments and NUL bytes.
    2. Expand ``~`` to the user's home directory.
    3. If relative, treat as relative to the current working directory
       (the feature does not assume a workspace root here — that is a
       policy concern).
    4. Walk symlinks with ``os.path.realpath`` so the returned path is
       what would actually be opened.

    The returned path is what the policy layer should match against the
    allow/deny lists, and what the approval payload should display to the
    human. Symlink targets are resolved before the human sees them, so
    a "safe-looking" path inside an allowed root that points to
    ``/etc/passwd`` will appear as ``/etc/passwd`` in the approval prompt.
    """
    assert_no_traversal(candidate)
    cand = Path(candidate).expanduser()
    if not cand.is_absolute():
        cand = Path.cwd() / cand
    return Path(_best_effort_realpath(cand))


def _best_effort_realpath(path: Path) -> str:
    """Return the realpath even when the leaf doesn't exist yet.

    ``os.path.realpath`` handles non-existent leaves correctly on modern
    Python by walking up to the deepest existing ancestor. We rely on
    that and return the result as a plain string.
    """
    return os.path.realpath(str(path))


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
