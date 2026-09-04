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

import re
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


# Characters that reach a program unchanged: a shell gives them no
# meaning in a bare word, and ``shlex`` treats them as ordinary too. The
# rule below is an ALLOW-list of these rather than a deny-list of shell
# constructs, and that direction is the whole point.
#
# Five rounds of review on #3129 each found another construct a
# deny-list had missed — ``$[1+1]``, a comment, ``$"..."``, brace
# expansion, a tilde after an assignment, a nested brace — or another
# command it refused that bash and ``shlex`` agreed on. Enumerating what
# a shell does is writing a shell; the list never closes. Enumerating
# what a shell ignores closes immediately, and errs toward refusing.
_INERT_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_./:=,+@%^"
)


# Words a shell reads as GRAMMAR rather than as the name of a program.
#
# The character allow-list above is not enough on its own, because a
# grammar word is spelled entirely in inert characters. ``eval``,
# ``FOO=x`` and ``trap`` are requests to run some *other* program, and
# ``BinaryPolicy`` vets only the first token — so what the operator
# approves is not what the caller asked for.
#
# This was originally a containment for a backend that did not exec the
# vector it was handed: ``DockerSandboxBackend`` rebuilt a bash script
# from it, where those three words ran a binary the policy never saw
# (#3187). That backend now execs argv, so such a word fails as a
# missing executable on both backends instead. The rule stays because
# the tool's contract is stated here, not in a backend: a refusal that
# names what is wrong beats a container that starts and exits 127, and
# a future backend cannot re-open the hole by being written the old
# way.
#
# Listing the builtins that dispatch a command is the enumeration this
# ticket has been burned by six times: round 7 named `eval`, `exec`,
# `source`, `.`, `command`, `builtin`, and round 8 answered with
# `trap` — with `mapfile -C`, `enable -f` and `complete -C` behind it.
# So the rule is inverted, like the character rule before it: a BUILTIN
# IS NOT A PROGRAM. Every builtin is refused except the few that are
# also real binaries behaving identically, which is a short list that
# closes.
_BUILTINS_THAT_ARE_ALSO_PROGRAMS = frozenset(
    {"echo", "printf", "test", "[", "true", "false", "pwd", "kill"}
)

# Snapshot of bash 5.2, NOT of whatever bash is on this machine. macOS
# ships bash 3.2, whose `compgen -k` omits `coproc` and whose
# `compgen -b` omits `mapfile`/`readarray`/`compopt` — so a differential
# that only asked the local shell passed here and would have failed on
# the Linux CI runner. The live check still runs, and catches anything a
# newer bash adds; this snapshot is what makes the check version-proof
# where the developer's shell is older than the runner's.
#
# It is a union of bash and BusyBox ash, because "which shell" was not
# a constant while a backend still ran one: DockerExecutor wrote a
# script it called bash and ran it with `sh` inside `alpine:3.19`,
# where /bin/sh is BusyBox. Measured in that image rather than read
# from documentation — BusyBox ash has `chdir`, which bash does not,
# and `shell("chdir /tmp")` therefore succeeded under the default
# backend while failing "command not found" under the local one (codex
# round 9).
#
# #3187 removed the shell from that path, and with it the open-ended
# obligation this list could not meet: no image's `/bin/sh` decides
# what runs any more, so an operator who repoints `DEFAULT_IMAGES`
# cannot introduce grammar the list has never heard of. What remains is
# a closed question about the two shells a caller may plausibly have in
# mind when they write a command by hand.
_SHELL_RESERVED_WORDS = frozenset(
    "if then else elif fi case esac for select while until do done in "
    "function time coproc { } ! [[ ]]".split()
)
_SHELL_BUILTINS = frozenset(
    ". : [ alias bg bind break builtin caller cd command compgen compopt "
    "complete continue declare dirs disown echo enable eval exec exit "
    "export false fc fg getopts hash help history jobs kill let local "
    "logout mapfile popd printf pushd pwd read readarray readonly return "
    "set shift shopt source suspend test times trap true type typeset "
    "ulimit umask unalias unset wait "
    # BusyBox ash, measured in alpine:3.19 — `chdir` is the only one it
    # has that bash does not.
    "chdir".split()
)

# An assignment needs a valid shell NAME before the ``=``. bash reports
# "command not found" for ``--foo=bar`` and ``a-b=x``, so those are
# ordinary command names and refusing them would be a false refusal —
# the class codex flagged in round 4. The append form ``FOO+=x`` is an
# assignment too, and the allow-list lets ``+`` through (round 8).
_ASSIGNMENT_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])?\+?=")


@dataclass(frozen=True)
class ShellSyntax:
    """A character whose shell meaning this tool cannot honour."""

    index: int
    char: str


def command_word_is_shell_grammar(word: str) -> str | None:
    """Describe how *word* is grammar rather than a program, or None.

    Only meaningful in command position — ``FOO=x`` is an ordinary
    argument anywhere else, and ``if`` is just a word.
    """
    if not word:
        return None
    if _ASSIGNMENT_PREFIX.match(word):
        return "an assignment, which sets a variable for another command"
    if word in _SHELL_RESERVED_WORDS:
        return "a shell keyword, which introduces a compound command"
    if word in _SHELL_BUILTINS and word not in _BUILTINS_THAT_ARE_ALSO_PROGRAMS:
        return "a shell builtin, not a program this tool can run"
    return None


def _quote_context(cmd: str):
    r"""Yield ``(index, char, context, escaped)`` for each character.

    ``context`` is ``"bare"``, ``"single"`` or ``"double"``; ``escaped``
    says a backslash the shell honours precedes this character. Quote
    marks that open or close a region are not yielded — they are
    structure, not content — but an ESCAPED quote is, because it is
    content: bash reads ``foo\\'`` as a literal apostrophe and keeps
    reading the line unquoted.

    That distinction is not decoration. Dropping it opened a bogus
    single-quoted region at the escaped quote in
    ``echo foo\\'; sudo -n true``, which hid the ``;`` from
    :func:`first_unquoted_shell_control` — so ``BinaryPolicy`` saw an
    allow-listed ``echo``, the codex bridge auto-accepted, and a real
    shell ran the deny-listed second command with no approval. The
    backslash itself is still yielded, so a caller that treats it as
    significant sees it.

    Single quotes take no escapes in a shell, so a backslash inside one
    is an ordinary character.
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
        context = "double" if in_double else "bare"
        if c == "\\" and i + 1 < len(cmd):
            yield (i, c, context, False)
            yield (i + 1, cmd[i + 1], context, True)
            i += 2
            continue
        if in_double:
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
        yield (i, c, "bare", False)
        i += 1


def first_unquoted_shell_control(cmd: str) -> tuple[int, str] | None:
    """First unquoted shell control character, counting every ``$``.

    This is the compound-command guard's question (#1694): an
    allow-listed first token cannot vouch for a string that might
    compose a second command, so a suspicious ``$`` is worth an
    approval prompt even when it would have been literal. Reading too
    much here is free — the answer only routes a command to the queue.

    Deliberately NOT the same reading as
    :func:`first_shell_significant_character`. This one gates the codex
    bridge, where a real shell runs the line; widening it to every
    character a shell could interpret would put the operator in front
    of ``ls -la`` for its hyphen.
    """
    for index, char, context, escaped in _quote_context(cmd):
        if escaped or context == "single":
            # A shell reads an escaped character literally, so it
            # composes nothing — and the backslash that escaped it was
            # yielded separately, for callers that care.
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


def first_shell_significant_character(cmd: str) -> ShellSyntax | None:
    r"""First character a shell would read that ``exec`` will not.

    The tool tokenizes with ``shlex`` and executes the argv vector, so
    the question is whether the command that runs is the command that
    was written (#3129). This answers it soundly rather than exactly:
    a command it allows provably reaches the program as bash would have
    built it, and some commands it refuses would in fact have been
    identical.

    That trade is deliberate. The exact answer requires modelling every
    expansion bash performs, which five rounds of review demonstrated is
    a shell rather than a predicate. The sound answer needs only the set
    of characters a shell leaves alone, which is short and closed.

    - Single-quoted text is inert to both, so anything may appear in it.
    - Double-quoted text still expands ``$`` and backticks, and bash
      drops a backslash where ``shlex`` keeps it, so those three are
      refused there while ordinary characters are not.
    - Bare text may hold only inert characters. Spaces and tabs
      separate words for both; a newline does not — it separates
      commands for bash.

    A refused character is not a verdict on the caller's intent: if
    they meant it literally, quoting it makes bash and ``exec`` agree by
    construction, which is what the refusal offers back.
    """
    if not isinstance(cmd, str):
        return None
    for index, char, context, _escaped in _quote_context(cmd):
        if context == "single":
            continue
        if context == "double":
            # bash expands these inside double quotes, and removes a
            # backslash that shlex keeps.
            if char in "$`\\":
                return ShellSyntax(index, char)
            continue
        if char in " \t":
            continue
        if char not in _INERT_CHARACTERS:
            return ShellSyntax(index, char)
    return None


def quote_words_containing_shell_syntax(cmd: str) -> str | None:
    """Rewrite *cmd* so every shell-significant character is literal.

    The refusal can offer this because it is bounded by construction:
    single-quoted text is inert to bash and to ``shlex`` alike, so the
    rewrite's argv is the words themselves — and those words are what
    the path and binary policies already vetted. That is the property
    the ``bash -lc`` wrapper this once suggested could not have (#3130).

    Returns ``None`` when the command cannot be tokenized, which is the
    one case where there are no words to quote.
    """
    try:
        words = shlex.split(cmd)
    except ValueError:
        return None
    if not words:
        return None
    return " ".join(shlex.quote(word) for word in words)
