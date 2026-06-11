"""Tests for the path & binary policy resolvers (#835)."""

from pathlib import Path

import pytest

from kestrel_sovereign.features.computer_use.policy import (
    BinaryPolicy,
    Decision,
    PathPolicy,
    command_contains_unquoted_shell_control,
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


def test_compound_guard_handles_non_string():
    # ``argv`` lists go straight through BinaryPolicy.evaluate; the
    # helper is for raw command strings and gracefully no-ops on
    # anything else.
    assert command_contains_unquoted_shell_control(None) is False  # type: ignore[arg-type]
    assert command_contains_unquoted_shell_control(["ls", "-la"]) is False  # type: ignore[arg-type]
