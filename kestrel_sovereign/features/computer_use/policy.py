"""Allow/deny policy resolution for paths and binaries.

The feature uses two independent policies:

- :class:`PathPolicy` — decides whether a filesystem operation on a given
  path is allowed, must seek approval, or must be hard-rejected.
- :class:`BinaryPolicy` — same shape for the executable that a shell
  command resolves to (``argv[0]``).

Both policies share deny-wins semantics: any deny match short-circuits to
``deny`` regardless of the allow list. Glob patterns are honored via
``fnmatch`` (delegated to :func:`path_safety.match_allow_list`).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .path_safety import match_allow_list


class Decision(str, Enum):
    """Outcome of a policy evaluation."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyResult:
    """Carries the decision plus the rule that triggered it (for audit)."""

    decision: Decision
    rule: str  # e.g. "deny:~/.ssh", "allow:~/projects", "no_match"


@dataclass
class PathPolicy:
    """Allow/deny policy for filesystem paths.

    ``auto_approve_read`` only applies inside the allow-list; reads
    outside the allow-list always go through approval (or get denied if
    they hit the deny-list).
    """

    allow: list[str] = None  # type: ignore[assignment]
    deny: list[str] = None  # type: ignore[assignment]
    auto_approve_read: bool = True

    def __post_init__(self) -> None:
        if self.allow is None:
            self.allow = []
        if self.deny is None:
            self.deny = []

    def evaluate(self, path: Path, *, write: bool) -> PolicyResult:
        """Evaluate a path for read or write.

        - Deny match → ``DENY`` (always).
        - Allow match + read + ``auto_approve_read`` → ``ALLOW``.
        - Allow match + write → ``REQUIRE_APPROVAL`` (writes always gated).
        - No match → ``REQUIRE_APPROVAL`` (human can vouch for an unusual path).
        """
        match = match_allow_list(path, self.allow, self.deny)
        if match.reason.startswith("deny:"):
            return PolicyResult(Decision.DENY, match.reason)
        if match.allowed:
            if write:
                return PolicyResult(Decision.REQUIRE_APPROVAL, match.reason)
            if self.auto_approve_read:
                return PolicyResult(Decision.ALLOW, match.reason)
            return PolicyResult(Decision.REQUIRE_APPROVAL, match.reason)
        return PolicyResult(Decision.REQUIRE_APPROVAL, match.reason)


@dataclass
class BinaryPolicy:
    """Allow/deny policy for the executable in a shell command."""

    allow: list[str] = None  # type: ignore[assignment]
    deny: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.allow is None:
            self.allow = []
        if self.deny is None:
            self.deny = []

    def evaluate(self, argv: list[str] | str) -> PolicyResult:
        """Evaluate ``argv[0]`` against the allow/deny lists.

        Strings are tokenized with ``shlex``. Match is on the basename of
        ``argv[0]`` so callers can list ``git`` rather than ``/usr/bin/git``.
        """
        if isinstance(argv, str):
            tokens = shlex.split(argv)
        else:
            tokens = list(argv)
        if not tokens:
            return PolicyResult(Decision.DENY, "no_argv")

        binary = Path(tokens[0]).name
        if binary in self.deny:
            return PolicyResult(Decision.DENY, f"deny:{binary}")
        if binary in self.allow:
            return PolicyResult(Decision.REQUIRE_APPROVAL, f"allow:{binary}")
        return PolicyResult(Decision.DENY, f"no_match:{binary}")


def split_command(cmd: str | Iterable[str]) -> list[str]:
    """Tokenize a shell command using POSIX rules. Used by tools and tests."""
    if isinstance(cmd, str):
        return shlex.split(cmd)
    return list(cmd)
