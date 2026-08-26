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

from .path_safety import PathSafetyError, match_allow_list, resolve_realpath


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


@dataclass(frozen=True)
class ArgvPathResult:
    """Aggregate outcome of vetting a command's argument paths (F137).

    ``decision`` is the strictest across all argument tokens; ``rule``
    and ``path`` identify the token that drove it (empty/``None`` when
    no argument path changed the decision).
    """

    decision: Decision
    rule: str
    path: str | None = None


def evaluate_argv_paths(
    argv: list[str] | str,
    path_policy: PathPolicy,
    *,
    cwd: str | Path | None = None,
) -> ArgvPathResult:
    """Vet a shell command's *arguments* against a :class:`PathPolicy` (F137).

    :class:`BinaryPolicy` only inspects ``argv[0]``. Without this, an
    auto-approved reader (``cat``/``rg``/``ls``) short-circuits to ALLOW
    and its file argument is never checked against ``deny_paths`` — so
    ``cat ~/.aws/credentials`` would read a host secret with no approval,
    even though the ``fs_*`` tools honor the same deny-list. This unifies
    the guarantee across ``fs_*``, the ``shell`` tool, and the codex
    native-command bridge.

    Every non-flag argument token (``argv[1:]``, skipping ``-``/``--``
    flags) is resolved to its realpath (``~`` expanded; relatives joined
    to ``cwd`` — process cwd when unset) and evaluated with read
    semantics:

    - a ``deny`` hit → ``DENY``, returned immediately (deny-wins; honored
      even for a not-yet-existing path since deny patterns match by
      prefix);
    - an *existing*, non-allow-listed path → ``REQUIRE_APPROVAL``
      (aggregated, so an auto-approved binary can't bypass the queue for
      an unusual path);
    - anything else has no effect.

    Tokens that can't be resolved (traversal/NUL) are skipped. Returns
    the strictest aggregate — ``ALLOW`` means no argument path changed
    the binary's decision.
    """
    tokens = split_command(argv) if isinstance(argv, str) else list(argv)
    aggregate = ArgvPathResult(Decision.ALLOW, "no_path_args")
    for token in tokens[1:]:
        if not token or token.startswith("-"):
            continue
        try:
            candidate = resolve_realpath(token, base=cwd)
        except PathSafetyError:
            continue
        result = path_policy.evaluate(candidate, write=False)
        if result.decision is Decision.DENY:
            return ArgvPathResult(Decision.DENY, result.rule, str(candidate))
        if result.decision is Decision.REQUIRE_APPROVAL and candidate.exists():
            aggregate = ArgvPathResult(
                Decision.REQUIRE_APPROVAL, result.rule, str(candidate)
            )
    return aggregate


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


# A ``$`` only begins an expansion when the next character can start a
# parameter, command or arithmetic substitution. Measured against bash,
# not assumed: ``foo$:bar``, ``foo$,bar``, ``foo$.bar``, ``foo$/bar``,
# ``price$`` and ``echo "$"`` all reach the program with the ``$``
# intact, so ``shlex`` and a real shell build the same argument vector
# for them.
#
# Quotes are in the set but only apply outside a quoted region: bare
# ``$"..."`` is bash's localization form and ``$'...'`` is ANSI-C
# quoting, both of which change the word — while inside double quotes a
# ``$`` before the closing ``"`` is the literal in ``echo "$"``. The
# caller passes ``dollar_may_open_quote`` to say which it is looking at.
_DOLLAR_EXPANSION_STARTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_{[(@*#?-$!"
)

# Pathname expansion. Whether these change the word depends on what is
# on disk, so the string alone does not determine the argv — which is
# reason enough to refuse rather than guess (``echo *`` prints the
# directory or a literal ``*``, both with status 0).
_GLOB_CHARS = frozenset("*?[")


@dataclass(frozen=True)
class ShellSyntax:
    r"""A construct a shell would act on that ``exec`` will not.

    ``kind`` exists because the character alone does not explain the
    divergence. A bare ``$`` expands; a ``\$`` inside double quotes does
    the opposite — bash removes the backslash and ``shlex`` keeps it —
    and a refusal that said "cannot expand a variable" for the second
    would be telling the caller something untrue about their own line.
    """

    index: int
    char: str
    kind: str  # control | expansion | comment | tilde | glob | escape


def _quote_context(cmd: str):
    """Yield ``(index, char, context, escaped)`` for each character.

    ``context`` is ``"bare"``, ``"single"`` or ``"double"``; ``escaped``
    says a backslash the shell honours precedes this character. Quote
    marks that open or close a region are not yielded — they are
    structure, not content.

    The two questions this module asks both need to know where they are
    in a string, and they disagree about what to do when they get
    there, so the walk is shared and the judgement is not.
    """
    if not isinstance(cmd, str):
        return
    in_single = in_double = False
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if in_single:
            if c == "'":
                in_single = False
            else:
                yield (i, c, "single", False)
            i += 1
            continue
        if in_double:
            if c == "\\" and i + 1 < len(cmd):
                yield (i + 1, cmd[i + 1], "double", True)
                i += 2
                continue
            if c == '"':
                in_double = False
            else:
                yield (i, c, "double", False)
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
            yield (i + 1, cmd[i + 1], "bare", True)
            i += 2
            continue
        yield (i, c, "bare", False)
        i += 1


def _dollar_expands(cmd: str, index: int, *, may_open_quote: bool) -> bool:
    """True iff the ``$`` at *index* begins a shell expansion."""
    nxt = cmd[index + 1 : index + 2]
    if not nxt:
        return False
    if nxt in "\"'":
        return may_open_quote
    return nxt in _DOLLAR_EXPANSION_STARTERS


def _starts_a_word(cmd: str, index: int) -> bool:
    """True iff *index* begins a word — nothing before it, or blank."""
    return index == 0 or cmd[index - 1] in " \t\n\r"


def first_unquoted_shell_control(cmd: str) -> tuple[int, str] | None:
    """First unquoted shell control character, counting every ``$``.

    This is the compound-command guard's question (#1694): an
    allow-listed first token cannot vouch for a string that might
    compose a second command, so a suspicious ``$`` is worth an
    approval prompt even when it would have been literal. Reading too
    much here is free — the answer only routes a command to the queue.

    Deliberately NOT widened alongside :func:`first_shell_syntax_exec_ignores`.
    Globs and tilde compose no second command, and adding them would
    put the codex bridge in front of the operator for ``ls *.py``.
    """
    for index, char, context, escaped in _quote_context(cmd):
        if escaped or context == "single":
            continue
        if context == "double" and char not in "$`":
            continue
        if char in _SHELL_CONTROL_CHARS:
            return (index, char)
    return None


def command_contains_unquoted_shell_control(cmd: str) -> bool:
    """Return True iff ``cmd`` has a shell control character outside a
    quoted region — see :func:`first_unquoted_shell_control`.
    """
    return first_unquoted_shell_control(cmd) is not None


def first_shell_syntax_exec_ignores(cmd: str) -> ShellSyntax | None:
    r"""First construct a shell would act on that ``exec`` will not.

    The tool tokenizes with ``shlex`` and executes the argv vector, so
    this is the question that decides whether the command that runs is
    the command that was written (#3129).

    Every rule below was measured against bash rather than reasoned
    about, by sweeping each punctuation character through ten positions
    and comparing ``shlex.split`` to the word vector bash builds. That
    sweep is a test — the first three rounds of review on this ticket
    each found a construct an enumeration had missed, so the boundary
    is checked rather than asserted.

    Refused, because the argv differs:

    - the control characters that compose or redirect commands;
    - a ``$`` that expands, including bash's ``$[...]`` and the
      quote-opening ``$"..."`` / ``$'...'`` forms;
    - a ``#`` beginning a word — the rest of the line is a comment;
    - a ``~`` beginning a word — home-directory expansion;
    - ``*``, ``?`` and ``[`` — pathname expansion, where what the argv
      becomes depends on the directory rather than the string;
    - a backslash that bash removes and ``shlex`` keeps: ``\$`` and
      ``\```` inside double quotes, and a line continuation anywhere;
    - a trailing carriage return, which bash keeps inside the last word
      and ``shlex`` discards as whitespace.

    Trailing spaces, tabs and newlines are not constructs: bash and
    ``shlex`` both discard them.
    """
    if not isinstance(cmd, str):
        return None
    # Only the trailing blanks bash also discards. A trailing carriage
    # return is NOT one of them: bash keeps it inside the last word,
    # ``shlex`` treats it as whitespace and drops it. Measured.
    trimmed = cmd.rstrip(" \t\n")
    for index, char, context, escaped in _quote_context(trimmed):
        if context == "single":
            continue
        if escaped:
            # bash drops the backslash; shlex keeps it for these, so the
            # word that reaches the program differs. A backslash before
            # a carriage return is not a continuation — both keep it.
            if (context == "double" and char in "$`") or char == "\n":
                return ShellSyntax(index, char, "escape")
            continue
        if char == "$":
            if _dollar_expands(trimmed, index, may_open_quote=context == "bare"):
                return ShellSyntax(index, char, "expansion")
            continue
        if context == "double":
            if char == "`":
                return ShellSyntax(index, char, "expansion")
            continue
        if char == "`":
            return ShellSyntax(index, char, "expansion")
        if char in _SHELL_CONTROL_CHARS:
            return ShellSyntax(index, char, "control")
        if char in _GLOB_CHARS:
            return ShellSyntax(index, char, "glob")
        if char in "#~" and _starts_a_word(trimmed, index):
            return ShellSyntax(
                index, char, "comment" if char == "#" else "tilde"
            )
    return None
