"""Tests for the path & binary policy resolvers (#835)."""

from pathlib import Path

import pytest

from kestrel_sovereign.features.computer_use.policy import (
    BinaryPolicy,
    Decision,
    PathPolicy,
    command_contains_unquoted_shell_control,
    first_shell_syntax_exec_ignores,
    first_unquoted_expansion,
    first_unquoted_shell_control,
    token_binary_name,
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
    "cmd",
    [
        "rg foo$ file",
        'echo "$"',
        "echo price$",
        "cat a.txt | tr a-z A-Z",
        "echo $HOME",
        "echo hi; true",
        "echo hi",
        'echo "; rm -rf /"',
    ],
)
def test_the_refusal_never_reaches_further_than_the_compound_guard(cmd):
    """Two predicates, one containment: anything the refusal refuses,
    the compound guard also flags.

    They answer different questions — "would this run differently?"
    versus "could this be more than it looks?" — and the second must
    stay the wider one. If the refusal ever exceeded it, a command
    would be refused outright that the guard was content to send to
    the operator, and the two would be arguing.
    """
    if first_shell_syntax_exec_ignores(cmd) is not None:
        assert command_contains_unquoted_shell_control(cmd) is True


@pytest.mark.parametrize(
    "cmd,expected",
    [
        ("echo $HOME", (5, "$")),
        ("echo `whoami`", (5, "`")),
        ("echo $(whoami)", (5, "$")),
        # A literal dollar names nothing, so there is nothing a gate
        # could fail to see.
        ("echo price$", None),
        ("cat a.txt | tr a-z A-Z", None),
    ],
)
def test_first_unquoted_expansion_finds_what_no_gate_can_read(cmd, expected):
    """An expansion is the reason a refusal may not hand back a wrapper.

    ``cat $SECRET`` and ``echo `sudo -n true`` name a path and a binary
    that only exist once a shell has run, so the path and binary
    policies vetted something else entirely.
    """
    assert first_unquoted_expansion(cmd) == expected


@pytest.mark.parametrize(
    "token,expected",
    [
        ("rm", "rm"),
        ("/usr/bin/rm", "rm"),
        ("`rm", "rm"),
        ("(sudo", "sudo"),
        ("rm`", "rm"),
        (";rm;", "rm"),
        ("harmless", "harmless"),
    ],
)
def test_token_binary_name_sees_through_welded_punctuation(token, expected):
    """``shlex`` leaves a control character welded to its neighbour.

    ``echo `rm -rf /`` tokenizes to ``['echo', '`rm', '-rf', '/`']``,
    so a deny-list lookup on the raw token misses the one word that
    decides whether a shell wrapper may be offered.
    """
    assert token_binary_name(token) == expected


def test_compound_guard_handles_non_string():
    # ``argv`` lists go straight through BinaryPolicy.evaluate; the
    # helper is for raw command strings and gracefully no-ops on
    # anything else.
    assert command_contains_unquoted_shell_control(None) is False  # type: ignore[arg-type]
    assert command_contains_unquoted_shell_control(["ls", "-la"]) is False  # type: ignore[arg-type]
