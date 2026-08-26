"""Tests for the path & binary policy resolvers (#835)."""

from pathlib import Path

import itertools
import shlex
import shutil
import string
import subprocess
import tempfile

import pytest

from kestrel_sovereign.features.computer_use.policy import (
    BinaryPolicy,
    Decision,
    PathPolicy,
    command_contains_unquoted_shell_control,
    first_shell_syntax_exec_ignores,
    first_unquoted_shell_control,
    split_command,
)


def test_path_policy_deny_wins():
    pol = PathPolicy(allow=["/home/me"], deny=["/home/me/.ssh"])
    result = pol.evaluate(Path("/home/me/.ssh/id_rsa"), write=False)
    assert result.decision is Decision.DENY


def test_path_policy_allow_read_auto_approve():
    pol = PathPolicy(allow=["/home/me"], deny=[], auto_approve_read=True)
    result = pol.evaluate(Path("/home/me/foo.txt"), write=False)
    assert result.decision is Decision.ALLOW


def test_path_policy_allow_write_requires_approval():
    pol = PathPolicy(allow=["/home/me"], deny=[])
    result = pol.evaluate(Path("/home/me/foo.txt"), write=True)
    assert result.decision is Decision.REQUIRE_APPROVAL


def test_path_policy_no_match_requires_approval():
    pol = PathPolicy(allow=["/home/me"], deny=[])
    result = pol.evaluate(Path("/var/log/syslog"), write=False)
    assert result.decision is Decision.REQUIRE_APPROVAL


def test_path_policy_auto_approve_off():
    pol = PathPolicy(allow=["/home/me"], deny=[], auto_approve_read=False)
    result = pol.evaluate(Path("/home/me/foo.txt"), write=False)
    assert result.decision is Decision.REQUIRE_APPROVAL


def test_binary_policy_denied():
    pol = BinaryPolicy(allow=["git"], deny=["rm"])
    assert pol.evaluate(["rm", "-rf", "/"]).decision is Decision.DENY


def test_binary_policy_allow_listed_short_circuits_to_allow():
    # #1694: allow-listed binaries are pre-approved (Decision.ALLOW)
    # so the queue is bypassed, matching auto_approve_read for paths.
    pol = BinaryPolicy(allow=["git"], deny=[])
    result = pol.evaluate(["git", "status"])
    assert result.decision is Decision.ALLOW
    assert result.rule == "allow:git"


def test_binary_policy_unknown_requires_approval():
    # #1694: no_match no longer hard-denies; it routes through the
    # ApprovalQueue so the operator can vouch for an unusual binary.
    pol = BinaryPolicy(allow=["git"], deny=[])
    result = pol.evaluate(["touch", "/tmp/x"])
    assert result.decision is Decision.REQUIRE_APPROVAL
    assert result.rule == "no_match:touch"


def test_binary_policy_deny_wins_over_allow():
    # An entry on both lists must still hard-deny (deny-wins).
    pol = BinaryPolicy(allow=["sudo"], deny=["sudo"])
    assert pol.evaluate(["sudo", "-i"]).decision is Decision.DENY


def test_binary_policy_basename_match():
    pol = BinaryPolicy(allow=["git"], deny=[])
    assert pol.evaluate(["/usr/bin/git", "status"]).decision is Decision.ALLOW


def test_binary_policy_string_input():
    pol = BinaryPolicy(allow=["git"], deny=[])
    assert pol.evaluate("git status").decision is Decision.ALLOW


def test_binary_policy_empty_argv():
    pol = BinaryPolicy(allow=["git"], deny=[])
    assert pol.evaluate([]).decision is Decision.DENY


def test_split_command():
    assert split_command("git commit -m 'hello world'") == ["git", "commit", "-m", "hello world"]
    assert split_command(["already", "split"]) == ["already", "split"]


# ---------------------------------------------------------------------------
# Compound-command guard (#1694 codex review P1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "echo hi",
        "ls -la /tmp",
        "git status",
        # Quoted control chars are inert.
        'echo "; rm -rf /"',
        "echo '; rm -rf /'",
        # Quoted dollar sign inside single quotes is inert.
        "echo 'cost $5'",
        # Single hyphen in arg name is fine.
        "rg -i pattern",
    ],
)
def test_compound_guard_clean_commands(cmd):
    assert command_contains_unquoted_shell_control(cmd) is False


@pytest.mark.parametrize(
    "cmd,trigger",
    [
        ("git status; rm -rf /tmp/x", ";"),
        ("git status && rm -rf /tmp/x", "&"),
        ("git status || true", "|"),
        ("ls | grep secret", "|"),
        ("echo `whoami`", "backtick"),
        ("echo $(whoami)", "$ + ("),
        ("ls > /etc/foo", ">"),
        ("cat < /etc/passwd", "<"),
        ("git status\nrm -rf /tmp/x", "newline"),
        # Double-quoted but with a $ inside — still active (variable
        # expansion is enabled in "..."), so we treat $ as risky.
        ('echo "$HOME"', "$"),
    ],
)
def test_compound_guard_flags_unquoted_metacharacters(cmd, trigger):
    assert command_contains_unquoted_shell_control(cmd) is True, (
        f"expected {trigger!r} to flag: {cmd!r}"
    )


@pytest.mark.parametrize(
    "cmd,expected",
    [
        ("ls | grep secret", (3, "|")),
        ("git status; rm -rf /tmp/x", (10, ";")),
        # The FIRST one, not any one: an inert quoted ``;`` sits before
        # the live ``|``, and reporting the quoted one would name a
        # character the caller is allowed to keep.
        ("""echo '; x' | wc -l""", (11, "|")),
        ('echo "$HOME"', (6, "$")),
        ("echo hi", None),
    ],
)
def test_first_unquoted_shell_control_reports_where_and_which(cmd, expected):
    """#3129 needs the character, not just its existence.

    ``ComputerUseFeature.shell`` refuses a command it cannot honour and
    has to say which character it could not honour — a caller told only
    "no" cannot tell which part of what they wrote was the problem.
    """
    assert first_unquoted_shell_control(cmd) == expected


@pytest.mark.parametrize(
    "cmd",
    [
        "echo hi",
        'echo "; rm -rf /"',
        "ls | grep secret",
        "git status && rm -rf /tmp/x",
        "echo `whoami`",
        None,
        ["ls", "-la"],
    ],
)
def test_the_boolean_guard_is_exactly_the_scanner(cmd):
    """The compound-command guard is now a wrapper over the scanner.

    #1694's downgrade and #3129's refusal must never disagree about
    what counts as a shell control character: two answers to one
    question is how a command gets refused on one path and queued on
    another.
    """
    assert command_contains_unquoted_shell_control(cmd) is (
        first_unquoted_shell_control(cmd) is not None
    )


@pytest.mark.parametrize(
    "cmd,diverges",
    [
        # shlex and bash build the SAME vector for these — measured, not
        # assumed: `printf "%s\\0" <cmd>` against shlex.split.
        ("rg foo$ file", False),
        ('echo "$"', False),
        ("echo price$", False),
        ("echo $:x", False),
        ("grep -E \'a|b\' f", False),
        # ...and different vectors for these.
        ("cat a.txt | tr a-z A-Z", True),
        ("echo $HOME", True),
        ('echo "$HOME"', True),
        ("echo `whoami`", True),
        ("cat ${X}", True),
        ("echo hi; true", True),
        ("ls > /tmp/x", True),
        # shlex splits on a bare CR; bash keeps it inside the word.
        ("echo a\rb", True),
        # bash's legacy arithmetic form: `echo $[1+1]` prints 2. Found by
        # codex review round 2 — the starter set had `{` and `(` but not
        # `[`, so this divergence was silent, which is the exact defect.
        ("echo $[1+1]", True),
        # ...and inside double quotes, where the glob rule does not
        # apply and only the dollar starter set can catch it. Without
        # this case, dropping `[` from that set changes nothing any
        # test can see.
        ('echo "$[1+1]"', True),
        # Codex review round 3, named rather than left to the sweep:
        # each of these ran and reported success with a different argv.
        ("echo hi # ignored", True),          # the rest is a comment
        ("echo a#b", False),                  # ...but only at word start
        ('echo "a\\$HOME"', True),            # bash drops the backslash
        ('echo "a\\`x"', True),
        ('echo $"hello"', True),              # localization opens a quote
        ("echo a\\\nb", True),                # line continuation
        ("echo a\\\rb", False),               # ...but a CR is not one
        ("echo hi\n", False),                 # trailing newline is nothing
        ("echo hi\r", True),                  # a trailing CR is a word char
        ("echo ~", True),                      # home expansion
        ("echo a~b", False),                   # ...only at word start
        ("ls *.py", True),                     # pathname expansion
        ("echo hi", False),
        # Codex review round 4. Every one of these needs two characters
        # to exist, so the single-character sweep could not produce them
        # — which is why the corpus now sweeps pairs too.
        ("echo {a,b}", True),                  # brace expansion
        ("echo {1..3}", True),                 # ...and its range form
        ("echo {a}", False),                   # ...but a brace alone is literal
        ("echo HOME=~", True),                 # tilde after an assignment
        ("echo PATH=foo:~", True),             # ...and after its colon
        ("echo foo:~", False),                 # ...but not without the `=`
        ("echo [a]", True),                    # a bracket glob that closes
        ("echo a[b", False),                   # ...but an unclosed `[` is literal
        ("echo []", False),                    # ...and an empty one never globs
        ("echo \\ #x", False),                 # an escaped blank keeps the word
        ("echo hi\\\n", True),                # a trailing line continuation
        # A newline INSIDE the line is a command separator: bash runs
        # `b` as its own command, shlex hands `b` to echo as an
        # argument. The bash differential cannot see this one — bash
        # exits non-zero running `b`, so the case is skipped there — and
        # a mutant that stopped refusing it survived until this case
        # existed.
        ("echo a\nb", True),
    ],
)
def test_exec_ignores_exactly_what_a_shell_would_have_acted_on(cmd, diverges):
    """#3129 asks one question: will exec build a different argument
    vector than a shell would?

    Answering it with #1694's guard refused ``rg foo$ file`` and
    ``echo "$"``, which were never broken (codex review round 1, P2).
    A ``$`` only counts when the next character can begin an expansion.
    """
    assert (first_shell_syntax_exec_ignores(cmd) is not None) is diverges


@pytest.mark.parametrize(
    "cmd,refused,flagged",
    [
        # Both: a control character composes a command AND changes the argv.
        ("echo hi; true", True, True),
        ("cat a.txt | tr a-z A-Z", True, True),
        ("echo $HOME", True, True),
        # Refusal only: these change the argv but compose nothing. The
        # guard is deliberately not widened to them — it gates the codex
        # bridge, where a real shell runs the line, and flagging `ls
        # *.py` there would put the operator in front of every glob.
        ("ls *.py", True, False),
        ("echo hi # note", True, False),
        ("echo ~", True, False),
        # Guard only: a literal `$` composes nothing and changes nothing,
        # but reading it as suspicious costs only an approval prompt,
        # while refusing it would break a command that worked.
        ("rg foo$ file", False, True),
        ('echo "$"', False, True),
        # Neither.
        ("echo hi", False, False),
        ('echo "; rm -rf /"', False, False),
    ],
)
def test_the_two_predicates_answer_two_questions(cmd, refused, flagged):
    """They overlap on control characters and diverge on purpose.

    An earlier version of this test asserted containment — anything
    refused is also flagged — and it passed only because its cases
    happened to contain no counterexample. Globs, comments and tilde
    broke it the moment they were added, and a mutation run is what
    surfaced the stale claim. The real relationship is that each
    predicate reads what its own consequence justifies: over-reading
    costs an approval prompt on one side and a broken command on the
    other.
    """
    assert (first_shell_syntax_exec_ignores(cmd) is not None) is refused
    assert command_contains_unquoted_shell_control(cmd) is flagged
# Pathname and tilde expansion turn on what is on disk and who the user
# is, so a string alone does not determine the argv. The refusal covers
# them deliberately, and the differential below cannot judge them: in an
# empty directory bash leaves them literal and agrees with shlex, which
# would read as a false positive. Named here rather than silently
# skipped, so the exclusion is a decision someone can argue with.
_EXPANSION_DEPENDS_ON_ENVIRONMENT = set("*?[~")

_SWEEP_TEMPLATES = [
    "echo a{c}b",
    "echo {c}",
    "echo a {c} b",
    "echo {c}b",
    "echo 'a{c}b'",
    'echo "a{c}b"',
    "echo a\\{c}b",
    'echo "a\\{c}b"',
    "echo ${c}x",
    'echo "${c}x"',
]

# Single characters are not enough. Round 4 of review found five gaps
# the single-character sweep structurally could not produce — brace
# expansion, tilde after an assignment, a bracket glob that closes, an
# escaped blank before a `#`, a trailing backslash-newline — because
# every one of them needs two characters to exist. Pairs cost about
# three seconds of bash and would have produced all five.
_SWEEP_PAIR_TEMPLATES = ["echo x{a}y{b}z", "echo {a}{b}", "echo {a}x {b}y"]

# Constructs worth naming even though the sweeps generate them, so a
# reader can see what the boundary covers without running it.
_NAMED_CONSTRUCTS = [
    "echo {a,b}", "echo {1..3}", "echo x{a,b}y", "echo {a}", "echo {}",
    "echo {a,}", "echo a{b}c", "echo HOME=~", "echo PATH=foo:~",
    "echo a=~/x", "echo ~/x", "echo x~", "echo foo:~", "echo ~x",
    "echo \\ #x", "echo \\ ~", "echo a\\ #b", "echo hi\\\n",
    "echo [", "echo a[b", "echo [ab", "echo [a]", "echo a[bc]d",
    "echo []", "echo a]b", "echo hi \\\n there",
]


def _sweep_corpus():
    """Every command the differential compares against bash."""
    for char in string.punctuation + " \t\n\r":
        for template in _SWEEP_TEMPLATES:
            yield template.format(c=char)
    for first, second in itertools.product(string.punctuation, repeat=2):
        for template in _SWEEP_PAIR_TEMPLATES:
            yield template.format(a=first, b=second)
    yield from _NAMED_CONSTRUCTS


@pytest.fixture(scope="module")
def bash_differential():
    """(cmd, bash_argv, shlex_argv) for the whole corpus, computed once.

    Both the differential and its positive control need it, and running
    bash over the corpus twice doubled the file's runtime for no extra
    coverage.
    """
    rows = []
    with tempfile.TemporaryDirectory() as empty:
        for cmd in _sweep_corpus():
            if set(cmd) & _EXPANSION_DEPENDS_ON_ENVIRONMENT:
                continue
            expected = _bash_word_vector(cmd, empty)
            if expected is None:
                continue
            try:
                actual = shlex.split(cmd)
            except ValueError:
                continue
            rows.append((cmd, expected, actual))
    return rows


def _bash_word_vector(cmd: str, cwd: str):
    """The argv bash would build for *cmd*, or None if bash refuses it."""
    result = subprocess.run(
        ["bash", "-c", 'printf "%s\\0" ' + cmd],
        capture_output=True,
        cwd=cwd,
    )
    if result.returncode:
        return None
    return result.stdout.decode(errors="replace").split("\0")[:-1]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_refusal_agrees_with_bash_across_every_punctuation_character(
    bash_differential,
):
    """#3129: the boundary is measured, not enumerated.

    Three consecutive review rounds each found a construct an
    enumeration had missed — `$[1+1]`, a shell comment, `$"..."`, an
    escaped `\$` inside double quotes, a line continuation — and every
    one of them was the ticket's own defect in a narrower spelling: a
    command that runs as a different argv and reports success.

    So the predicate is checked against the thing it models. Every
    ASCII punctuation character is swept through ten positions, and
    `shlex.split` is compared to the word vector bash actually builds.
    The predicate must say "this diverges" exactly when it does.

    Commands bash itself rejects are skipped: there is no argv to
    compare, and a caller who writes an unparseable line is not the
    silent-divergence case this guards.
    """
    disagreements = [
        f"{cmd!r}: bash={expected!r} shlex={actual!r} "
        f"{'refused' if first_shell_syntax_exec_ignores(cmd) else 'allowed'}"
        for cmd, expected, actual in bash_differential
        if (actual != expected) != (first_shell_syntax_exec_ignores(cmd) is not None)
    ]
    assert disagreements == [], (
        "the refusal disagrees with bash:\n  " + "\n  ".join(disagreements)
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_sweep_would_notice_a_gap(bash_differential):
    """A positive control for the differential above.

    A sweep that compares two things can pass by comparing nothing —
    if the templates stopped producing divergent commands, or bash
    rejected all of them, the assertion would be empty and green. This
    pins that the corpus does contain both answers.
    """
    outcomes = {actual != expected for _cmd, expected, actual in bash_differential}
    assert outcomes == {True, False}, (
        f"the sweep no longer exercises both answers: {outcomes}"
    )
    # A collapse detector, not an exact count: the corpus is filtered
    # (bash rejects some generated lines, and glob/tilde cases are
    # excluded by name), so the number moves when templates change. It
    # should never fall to a handful.
    assert len(bash_differential) > 1000, (
        f"the corpus shrank to {len(bash_differential)} commands"
    )


@pytest.mark.parametrize(
    "cmd",
    ["ls *.py", "cat ~/notes.txt", "ls a?b", "ls [ab]c"],
)
def test_expansion_that_depends_on_the_directory_is_refused(cmd):
    """The differential cannot judge these, so they are pinned directly.

    Whether ``ls *.py`` reaches ``ls`` as one word or forty depends on
    the directory, not the string. "It might be the same" is not a
    reason to run it — the whole defect is a command whose argv is not
    the one that was written.
    """
    assert first_shell_syntax_exec_ignores(cmd) is not None


def test_compound_guard_handles_non_string():
    # ``argv`` lists go straight through BinaryPolicy.evaluate; the
    # helper is for raw command strings and gracefully no-ops on
    # anything else.
    assert command_contains_unquoted_shell_control(None) is False  # type: ignore[arg-type]
    assert command_contains_unquoted_shell_control(["ls", "-la"]) is False  # type: ignore[arg-type]
