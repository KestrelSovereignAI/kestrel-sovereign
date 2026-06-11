"""Allow/deny policy resolution for paths and binaries.

The feature uses two independent policies:

- :class:`PathPolicy` — decides whether a filesystem operation on a given
  path is allowed, must seek approval, or must be hard-rejected.
- :class:`BinaryPolicy` — same shape for the executable that a shell
  command resolves to (``argv[0]``).

Both policies share three-state, deny-wins semantics (#1694):

- ``DENY`` — deny-list match. Hard refuse; never raises a prompt.
- ``ALLOW`` — allow-list match (binaries; reads inside the path
  allow-list when ``auto_approve_read`` is on). Bypass the queue.
- ``REQUIRE_APPROVAL`` — no allow-list match, or an allow-list write
  for paths. Route through the ApprovalQueue.

Glob patterns are honored via ``fnmatch`` (delegated to
:func:`path_safety.match_allow_list`).
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

        Three-state semantics (#1694), mirroring :class:`PathPolicy`:

        - Deny match → ``DENY`` (hard refuse; never raises a prompt).
        - Allow match → ``ALLOW`` (pre-approved; bypass the queue).
        - No match → ``REQUIRE_APPROVAL`` (raise a prompt — operator
          can vouch for an unusual binary).

        The previous shape mapped no-match to ``DENY``, which collapsed
        branches 1 and 3 to the same outcome and made the deny-list a
        no-op for binaries not on the allow-list. The new shape gives
        the deny-list the distinct meaning "never, even with operator
        approval" and routes everything unfamiliar through the
        ApprovalQueue.

        Strings are tokenized with ``shlex``. Match is on the basename of
        ``argv[0]`` so callers can list ``git`` rather than ``/usr/bin/git``.
        """
        raw_string = argv if isinstance(argv, str) else None
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
            # Compound-command guard (#1694 codex review P1): the
            # allow-list match is on the FIRST token. A raw string
            # like ``echo hi; rm -rf /`` would otherwise auto-approve
            # because ``echo`` is allow-listed. Downgrade to
            # REQUIRE_APPROVAL so the queue (and the operator) see
            # the full compound. Argv-list inputs are trusted — the
            # caller already vouched for the exact executable vector.
            if raw_string is not None and command_contains_unquoted_shell_control(raw_string):
                return PolicyResult(
                    Decision.REQUIRE_APPROVAL,
                    f"compound_command:allow:{binary}",
                )
            return PolicyResult(Decision.ALLOW, f"allow:{binary}")
        return PolicyResult(Decision.REQUIRE_APPROVAL, f"no_match:{binary}")


def split_command(cmd: str | Iterable[str]) -> list[str]:
    """Tokenize a shell command using POSIX rules. Used by tools and tests."""
    if isinstance(cmd, str):
        return shlex.split(cmd)
    return list(cmd)


# Shell metacharacters that compose a separate command (or substitute
# command output) regardless of the first token. If any unquoted form
# appears in a raw ``command`` string, an allow-listed first token
# cannot stand in for "the whole compound is safe" — the bridge and
# direct shell path downgrade ``ALLOW`` to ``REQUIRE_APPROVAL`` so the
# queue (and the operator) sees the full intent.
#
# Quoting matters: ``echo "; rm -rf /"`` is safe — the semicolon is
# inside a quoted argument and the shell never interprets it. So we
# check the post-tokenize residue: any tokens that contain a raw shell
# control char in an unquoted position carry the risk. ``shlex.split``
# strips quotes, so we instead scan the original string with a tiny
# state machine that respects single- and double-quoted regions.
_SHELL_CONTROL_CHARS = frozenset(";&|`$()<>\n\r")


def command_contains_unquoted_shell_control(cmd: str) -> bool:
    """Return True iff ``cmd`` has a shell control character outside
    a quoted region.

    Catches: ``;``, ``&``/``&&``, ``|``/``||``, backticks, ``$(...)``,
    ``$VAR`` substitution, redirects ``<``/``>``, newlines.
    Misses: process substitution ``<(...)`` (covered by ``<``/``(``),
    here-docs (covered by ``<``). Anything inside ``'...'``/``"..."``
    is ignored — quoted control characters are inert to the shell.

    Defense in depth, not exhaustive parsing. The QUEUE remains the
    real authoritative gate for anything we downgrade.
    """
    if not isinstance(cmd, str):
        return False
    # Inside double quotes the shell still expands ``$VAR`` and runs
    # ``$(...)`` / backticks. Only single-quoted regions truly disable
    # those — so we treat ``$`` and backticks as risky regardless of
    # double-quote context.
    DQ_ACTIVE_CHARS = frozenset("$`")
    in_single = False
    in_double = False
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if in_single:
            if c == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if c == "\\" and i + 1 < len(cmd):
                # Skip an escaped char inside double quotes.
                i += 2
                continue
            if c == '"':
                in_double = False
                i += 1
                continue
            if c in DQ_ACTIVE_CHARS:
                return True
            i += 1
            continue
        if c == "'":
            in_single = True
            i += 1
            continue
        if c == '"':
            in_double = True
            i += 1
            continue
        if c == "\\" and i + 1 < len(cmd):
            # Outside quotes, a backslash escapes the next character —
            # which still neutralizes its shell-control meaning. Skip.
            i += 2
            continue
        if c in _SHELL_CONTROL_CHARS:
            return True
        i += 1
    return False
