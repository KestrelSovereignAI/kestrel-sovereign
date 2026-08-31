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
    first_shell_significant_character,
    first_unquoted_shell_control,
    quote_words_containing_shell_syntax,
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
    "cmd",
    [
        "echo hi",
        "git diff -U2 -- kestrel_sovereign/policy.py",
        "ls -la /tmp",
        "python -m pytest -q",
        "grep -E 'a|b' file.txt",
        'echo "plain text"',
        "curl -sS https://example.com/a,b+c@d",
        "echo 'anything at all: | ; $x `y` *'",
    ],
)
def test_an_inert_command_runs(cmd):
    """The allow-list has to leave ordinary commands alone.

    A sound rule that refused everything would also never run a command
    whose argv differed, so this half is what makes the other half mean
    something.
    """
    assert first_shell_significant_character(cmd) is None


@pytest.mark.parametrize(
    "cmd,char",
    [
        # Composition and redirection.
        ("cat a.txt | tr a-z A-Z", "|"),
        ("echo hi; true", ";"),
        ("ls > /tmp/x", ">"),
        ("echo a && b", "&"),
        # Expansion, in all the spellings five review rounds turned up.
        ("echo $HOME", "$"),
        ('echo "$HOME"', "$"),
        ("echo `whoami`", "`"),
        ("echo $[1+1]", "$"),
        ('echo $"hello"', "$"),
        ("ls *.py", "*"),
        ("echo {a,b}", "{"),
        ("echo {a{b}c,d}", "{"),
        ("echo ~", "~"),
        ("echo HOME=~", "~"),
        ("echo hi # note", "#"),
        (r'echo "a\$HOME"', "\\"),
        # Word structure the tokenizer and the shell disagree about.
        ("echo a\nb", "\n"),
        ("echo hi\r", "\r"),
        ("echo a\\\nb", "\\"),
    ],
)
def test_a_shell_significant_character_is_refused(cmd, char):
    """Each of these ran and reported success with a different argv.

    The list is the accumulated output of five review rounds. Under the
    deny-list it took five rounds to cover them; under the allow-list
    every one falls out of the same rule, and so does the next spelling
    nobody has thought of.
    """
    found = first_shell_significant_character(cmd)
    assert found is not None, cmd
    assert found.char == char, (cmd, found)


@pytest.mark.parametrize(
    "cmd",
    [
        "echo price$",
        "rg foo$ file",
        'echo "$"',
        "echo a[b",
        "echo {a}",
        "echo --foo=~",
        'echo ""#x',
    ],
)
def test_the_rule_is_sound_rather_than_exact(cmd):
    """These reach the program identically either way, and are refused.

    Stated as a test rather than hidden in a docstring, because it is
    the cost of the design and someone may want to argue with it. The
    exact answer means modelling every expansion bash performs — five
    rounds of review each found another one, and each near-miss was a
    command running with the wrong argv. The refusal offers quoting,
    which makes these run unchanged.
    """
    assert first_shell_significant_character(cmd) is not None
    rewritten = quote_words_containing_shell_syntax(cmd)
    assert first_shell_significant_character(rewritten) is None, rewritten


@pytest.mark.parametrize(
    "cmd,refused,flagged",
    [
        # Both: a control character composes a command AND changes the argv.
        ("echo hi; true", True, True),
        ("cat a.txt | tr a-z A-Z", True, True),
        ("echo $HOME", True, True),
        # Refusal only: these change the argv but compose nothing. The
        # guard gates the codex bridge, where a real shell runs the
        # line, and widening it to every interpretable character would
        # put the operator in front of `ls -la` for its hyphen.
        ("ls *.py", True, False),
        ("echo hi # note", True, False),
        ("echo ~", True, False),
        # Guard only: a literal `$` composes nothing, but reading it as
        # suspicious costs only an approval prompt.
        ('echo "$"', True, True),
        ("echo hi", False, False),
        ('echo "; rm -rf /"', False, False),
    ],
)
def test_the_two_predicates_answer_two_questions(cmd, refused, flagged):
    """They overlap and diverge on purpose.

    An earlier version asserted containment — anything refused is also
    flagged — and it passed only because its cases happened to hold no
    counterexample. Each predicate reads what its own consequence
    justifies: over-reading costs an approval prompt on one side and a
    refused command on the other.
    """
    assert (first_shell_significant_character(cmd) is not None) is refused
    assert command_contains_unquoted_shell_control(cmd) is flagged


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

# Single characters are not enough: brace expansion, a tilde after an
# assignment, a bracket glob that closes and an escaped blank before a
# `#` all need two characters to exist, and round 4 of review found
# every one of them in the gap that left.
_SWEEP_PAIR_TEMPLATES = ["echo x{a}y{b}z", "echo {a}{b}", "echo {a}x {b}y"]

# Constructs worth naming even though the sweeps generate most of them.
_NAMED_CONSTRUCTS = [
    "echo {a,b}", "echo {1..3}", "echo {a{b}c,d}", "echo x{a,b}y", "echo {a}",
    "echo {}", "echo {a,}", "echo a{b}c", "echo HOME=~", "echo PATH=foo:~",
    "echo a=~/x", "echo ~/x", "echo x~", "echo foo:~", "echo ~x",
    "echo --foo=~", "echo a-b=~", "echo \\ #x", "echo \\ ~", "echo a\\ #b",
    "echo hi\\\n", "echo [", "echo a[b", "echo [ab", "echo [a]", "echo a[bc]d",
    "echo []", "echo a]b", "echo [a\\]", "echo {a\\,b}", 'echo ""#x',
    'echo ""~', "echo hi \\\n there", "git diff -U2 -- a.py", "ls -la /tmp",
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


@pytest.fixture(scope="module")
def bash_differential():
    """(cmd, bash_argv, shlex_argv) for the whole corpus, computed once."""
    rows = []
    with tempfile.TemporaryDirectory() as empty:
        for cmd in _sweep_corpus():
            expected = _bash_word_vector(cmd, empty)
            if expected is None:
                continue
            try:
                actual = shlex.split(cmd)
            except ValueError:
                continue
            rows.append((cmd, expected, actual))
    return rows


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_nothing_the_rule_allows_runs_differently_under_bash(bash_differential):
    """The guarantee, checked against the thing it is about.

    One-directional on purpose. "Refuses exactly when bash differs"
    would need a model of every expansion bash performs, and five
    rounds of review showed that model is a shell. "Never allows a
    command bash would build differently" is what actually protects the
    caller, and an allow-list of inert characters can satisfy it.

    The corpus sweeps every punctuation character through ten
    positions, every ordered pair through three more, and the named
    constructs the review rounds turned up.
    """
    unsound = [
        f"{cmd!r}: bash={expected!r} shlex={actual!r}"
        for cmd, expected, actual in bash_differential
        if first_shell_significant_character(cmd) is None and actual != expected
    ]
    assert unsound == [], (
        "these were allowed but bash builds a different argv:\n  "
        + "\n  ".join(unsound)
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_sweep_would_notice_a_gap(bash_differential):
    """A positive control for the differential above.

    A one-directional assertion passes trivially if the rule allows
    nothing, or if the corpus holds no divergent command. Both halves
    have to be present for the check above to mean anything.
    """
    allowed = [c for c, _e, _a in bash_differential
               if first_shell_significant_character(c) is None]
    divergent = [c for c, e, a in bash_differential if e != a]
    assert allowed, "the rule allows nothing, so soundness is vacuous"
    assert divergent, "the corpus holds no divergent command to catch"
    # A collapse detector, not an exact count: the number moves when
    # templates change, but should never fall to a handful.
    assert len(bash_differential) > 1000, (
        f"the corpus shrank to {len(bash_differential)} commands"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_quoting_remedy_is_bounded_by_construction(bash_differential):
    """What the refusal hands back must be safe AND must work.

    Round 2 removed a `bash -lc` suggestion because no check could
    bound what it would run. Quoting is different in kind: the rewrite
    is inert to bash and to shlex alike, so its argv is its words —
    and this asserts exactly that, over every refused command in the
    corpus, rather than trusting the argument.
    """
    with tempfile.TemporaryDirectory() as empty:
        broken = []
        for cmd, _expected, _actual in bash_differential:
            if first_shell_significant_character(cmd) is None:
                continue
            rewritten = quote_words_containing_shell_syntax(cmd)
            if rewritten is None:
                continue
            if first_shell_significant_character(rewritten) is not None:
                broken.append(f"{cmd!r} -> {rewritten!r} is still refused")
                continue
            bash_argv = _bash_word_vector(rewritten, empty)
            if bash_argv is not None and bash_argv != shlex.split(rewritten):
                broken.append(f"{cmd!r} -> {rewritten!r} still diverges")
        assert broken == [], "\n  ".join(broken)



def test_compound_guard_handles_non_string():
    # ``argv`` lists go straight through BinaryPolicy.evaluate; the
    # helper is for raw command strings and gracefully no-ops on
    # anything else.
    assert command_contains_unquoted_shell_control(None) is False  # type: ignore[arg-type]
    assert command_contains_unquoted_shell_control(["ls", "-la"]) is False  # type: ignore[arg-type]
