"""Tests for ComputerUseFeature gate ordering and lifecycle (#838)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.computer_use.feature import ComputerUseFeature
from kestrel_sovereign.privacy import PrivacyConfig


class FakeApprovalQueue:
    """Stand-in for SecurityFeature.approval_queue.

    ``decision`` controls what every ``request_approval`` call returns.
    """

    def __init__(self, decision: tuple[bool, str] = (True, "once")):
        self.decision = decision
        self.calls: list[dict] = []

    async def request_approval(self, feature_name, tool_name, tool_args, timeout):
        self.calls.append({"feature": feature_name, "tool": tool_name, "args": tool_args})
        return self.decision


class FakeSecurityFeature:
    def __init__(self, queue: FakeApprovalQueue):
        self.approval_queue = queue


class FakeAgent:
    """Minimal agent stub for feature unit tests."""

    def __init__(self, *, privacy: PrivacyConfig, grants: set[str], queue: FakeApprovalQueue):
        self.privacy_config = privacy
        self.granted_capabilities = frozenset(grants)
        self._security = FakeSecurityFeature(queue)
        self.did = "did:test:agent"
        self.features = {"security": self._security}

    def get_feature(self, name):
        return self.features.get(name)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "ok.txt").write_text("hello")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "leak.txt").write_text("ssh keys")
    return tmp_path


def _config(workspace: Path, *, enabled: bool = True, backend: str = "local") -> dict[str, Any]:
    return {
        "enabled": enabled,
        "backend": backend,
        "allowed_paths": [str(workspace)],
        "deny_paths": [str(workspace / "secret")],
        "allowed_binaries": ["echo"],
        "denied_binaries": ["rm"],
        "auto_approve_read": True,
        "audit_log_path": str(workspace / "audit.jsonl"),
    }


async def _make_feature(workspace: Path, *, agent: FakeAgent, backend: str = "local") -> ComputerUseFeature:
    feature = ComputerUseFeature(agent)
    feature._cfg = _config(workspace, backend=backend)
    # Skip the toml lookup; populate state by re-using initialize body.
    # Avoid LocalSandboxBackend unless the agent has both shell grants.
    if backend == "local":
        # Ensure both grants present for local backend construction
        agent.granted_capabilities = agent.granted_capabilities | {
            "shell_execution_sandboxed",
            "shell_execution_host",
        }
    feature._cfg = _config(workspace, backend=backend)
    return feature


@pytest.mark.asyncio
async def test_disabled_feature_returns_error(tmp_path: Path):
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants=set(),
        queue=FakeApprovalQueue(),
    )
    feature = ComputerUseFeature(agent)
    feature._cfg = _config(tmp_path, enabled=False)
    envelope = await feature.fs_read(path=str(tmp_path / "ok.txt"))
    assert envelope.status is not ToolResultStatus.OK
    assert envelope.error.startswith("readiness:")


@pytest.mark.asyncio
async def test_privacy_gate_blocks_when_flag_off(workspace: Path):
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=False),
        grants={"filesystem_read"},
        queue=FakeApprovalQueue(),
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert envelope.status is not ToolResultStatus.OK
    assert envelope.error.startswith("privacy")


@pytest.mark.asyncio
async def test_constitution_gate_blocks_without_grant(workspace: Path):
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants=set(),  # no Amendment IX grants
        queue=FakeApprovalQueue(),
    )
    feature = await _make_feature(workspace, agent=agent)
    # local backend init requires shell grants, so re-init with docker capability check skipped
    feature._cfg["backend"] = "local"
    agent.granted_capabilities = frozenset({"shell_execution_sandboxed", "shell_execution_host"})
    await feature.initialize()
    # Now strip the filesystem grant to test the gate at call time.
    agent.granted_capabilities = frozenset({"shell_execution_sandboxed", "shell_execution_host"})
    envelope = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert envelope.status is not ToolResultStatus.OK
    assert envelope.error.startswith("constitution")


@pytest.mark.asyncio
async def test_deny_path_hard_rejects_before_approval(workspace: Path):
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"filesystem_read", "shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.fs_read(path=str(workspace / "secret" / "leak.txt"))
    assert envelope.status is not ToolResultStatus.OK
    assert envelope.error.startswith("policy:deny")
    assert queue.calls == []  # never reached the approval queue


@pytest.mark.asyncio
async def test_allowed_read_succeeds_without_approval(workspace: Path):
    queue = FakeApprovalQueue()
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"filesystem_read", "shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert envelope.status is ToolResultStatus.OK
    assert envelope.data["content"] == "hello"
    assert queue.calls == []  # auto-approved


@pytest.mark.asyncio
async def test_write_requires_approval(workspace: Path):
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "filesystem_write",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.fs_write(path=str(workspace / "new.txt"), content="data")
    assert envelope.status is ToolResultStatus.OK
    assert len(queue.calls) == 1
    assert queue.calls[0]["tool"] == "fs-write"
    assert "diff_preview" in queue.calls[0]["args"]
    assert (workspace / "new.txt").read_text() == "data"


@pytest.mark.asyncio
async def test_write_denied_when_user_refuses(workspace: Path):
    queue = FakeApprovalQueue(decision=(False, "denied"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "filesystem_write",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.fs_write(path=str(workspace / "new.txt"), content="data")
    assert envelope.status is not ToolResultStatus.OK
    assert envelope.error.startswith("approval")
    assert not (workspace / "new.txt").exists()


@pytest.mark.asyncio
async def test_shell_denied_binary(workspace: Path):
    queue = FakeApprovalQueue()
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.shell(command="rm -rf /tmp", timeout=5)
    assert envelope.status is not ToolResultStatus.OK
    assert envelope.error.startswith("policy:deny")
    assert queue.calls == []


@pytest.mark.asyncio
async def test_shell_allow_listed_binary_bypasses_queue(workspace: Path):
    """#1694: allow-listed (auto-approved) binaries run without
    queueing. Mirrors auto_approve_read for PathPolicy.

    Also pins the legacy ``allowed_binaries`` config key (still accepted
    as a one-release deprecation synonym for ``auto_approved_binaries``;
    see [features.computer_use] in kestrel.toml).
    """
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.shell(command="echo hi", timeout=5)
    assert envelope.status is ToolResultStatus.OK
    assert envelope.data["returncode"] == 0
    assert "hi" in envelope.data["stdout"]
    assert queue.calls == [], (
        "allow-listed binary must bypass the ApprovalQueue (#1694)"
    )


def test_default_auto_approved_binaries_exclude_interpreters():
    """#1694 codex review P1: interpreters (``python``, ``node``) and
    rich CLIs (``uv``, ``gh``) MUST NOT be in the default
    auto-approve list, because under ALLOW they bypass the queue and
    ``python -c "..."`` becomes arbitrary host execution without a
    prompt. Operators can opt back in via kestrel.toml; the default
    has to err on the side of "ask"."""
    from kestrel_sovereign.features.computer_use.feature import (
        _DEFAULT_AUTO_APPROVED_BINS,
    )
    risky = {"python", "node", "uv", "gh", "bash", "sh", "zsh", "git"}
    overlap = risky & set(_DEFAULT_AUTO_APPROVED_BINS)
    assert overlap == set(), (
        f"default auto-approve list must not include {overlap} — "
        "they execute arbitrary code on behalf of the agent"
    )


def test_default_denied_binaries_reserved_for_unrecoverable():
    """#1739: the deny-list is "hard refuse without ever asking the
    operator." It is reserved for binaries whose blast radius is so
    wide that even an operator yes-click could be catastrophic.

    Pins:

    - Unrecoverable binaries MUST stay denied (host shutdown,
      privilege escalation, raw disk writes, remote access, filesystem
      creation).
    - ``rm`` MUST NOT be on the default deny-list. The workspace-write
      sandbox (#1737) handles in-workspace ``rm`` silently;
      out-of-workspace ``rm`` routes through the queue so the operator
      authorizes specific cleanups. Hard-deny was hostile UX in an
      operator-in-the-loop system (Emma's #1737 dogfood: couldn't
      clean up an approved-but-no-longer-needed probe file).
    """
    from kestrel_sovereign.features.computer_use.feature import (
        _DEFAULT_DENIED_BINS,
    )
    must_be_denied = {"dd", "mkfs", "shutdown", "sudo", "ssh"}
    missing = must_be_denied - set(_DEFAULT_DENIED_BINS)
    assert missing == set(), (
        f"default deny-list dropped binaries that must stay denied: "
        f"{missing}"
    )
    assert "rm" not in _DEFAULT_DENIED_BINS, (
        "rm must NOT be on the default deny-list (#1739) — under "
        "workspace-write the queue routes it for operator approval"
    )


@pytest.mark.asyncio
async def test_shell_compound_command_with_allow_listed_head_is_refused(workspace: Path):
    """#1694 asked that an allow-listed first token not bless a
    piggy-backed second command, and answered it by routing the
    compound to the queue. #3129 answers it harder: the second command
    was never going to run — ``shlex`` hands ``echo`` the tokens
    ``hi;`` and ``true`` — so queueing it asked the operator to approve
    a line that does not exist. The compound is refused instead, and
    the queue is not consulted at all.

    ``BinaryPolicy``'s ALLOW → REQUIRE_APPROVAL downgrade is unchanged
    and still covered by test_computer_use_policy.py; it remains the
    live guard for the codex bridge, where a flat ``command`` string is
    run by a real shell.
    """
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.shell(command="echo hi; true", timeout=5)
    assert envelope.status is not ToolResultStatus.OK
    assert "\';\'" in envelope.error
    assert queue.calls == [], (
        "the operator must not be asked to approve a command that would "
        "not be the command that runs"
    )


@pytest.mark.asyncio
async def test_shell_refuses_a_pipe_rather_than_running_it_unfiltered(workspace: Path):
    """#3129: the bound the caller asked for must not be silently dropped.

    ``cat ok.txt | tr a-z A-Z`` tokenizes to ``cat`` with four extra
    arguments. Before this, that ran, printed the unfiltered file and
    exited 0 — 128 live calls did exactly that. The refusal has to name
    the character, because a caller told only "no" cannot tell which
    part of what they wrote was not going to be honoured.
    """
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    target = workspace / "ok.txt"

    # Positive control FIRST: without the pipe this exact command runs.
    # Otherwise the refusal below could be any unrelated failure and the
    # test would still pass.
    ran = await feature.shell(command=f"cat {target}", timeout=5)
    assert ran.status is ToolResultStatus.OK
    assert ran.data["stdout"] == "hello"

    envelope = await feature.shell(command=f"cat {target} | tr a-z A-Z", timeout=5)
    assert envelope.status is not ToolResultStatus.OK
    assert "\'|\'" in envelope.error, envelope.error
    # Assert against the explanation, not the whole message: the message
    # ends with the suggested command, which quotes the caller's own line
    # back — and pytest's tmp_path carries this test's own name, so a
    # loose ``"cat" in error`` passes on the path alone. Both of these
    # survived a mutant that deleted what they were meant to pin.
    explanation, _, _ = envelope.error.partition(" Nothing ran.")
    assert "reaches \'cat\' as a literal argument" in explanation, envelope.error
    assert "cannot pipe one command\'s output into the next" in explanation, (
        "naming the character is not enough — the refusal has to say what "
        "the caller was counting on it to do"
    )
    assert len(queue.calls) == 1, "only the positive control should have reached the queue"


@pytest.mark.asyncio
async def test_the_refusal_offers_only_an_inert_rewrite(workspace: Path):
    """What may be handed back, and what may not.

    Rounds 1 and 2 were spent trying to bound a ``bash -lc`` wrapper by
    inspecting shlex tokens. Each round found another spelling that
    walked through the bound — ``cat </deny_paths/file`` hands the path
    policy the literal token ``</deny_paths/file``, which resolves to
    nothing — so the wrapper was removed. That is still true: no shell
    invocation is ever composed.

    Quoting is different in kind, not in degree. A single-quoted word
    is inert to bash and to shlex alike, so the rewrite's argv is its
    own words — the words the gates already vetted — and it cannot
    invoke anything the original did not name.
    """
    import shlex

    from kestrel_sovereign.features.computer_use.policy import (
        first_shell_significant_character,
    )

    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    secret = workspace / "secret" / "leak.txt"

    for command in (
        f"cat {workspace / 'ok.txt'} | tr a-z A-Z",
        f"cat <{secret} | head",
        "echo hi;rm -rf /tmp/x",
        "cat $HOME/.ssh/id_rsa | head",
    ):
        envelope = await feature.shell(command=command, timeout=5)
        assert envelope.status is not ToolResultStatus.OK, command
        error = envelope.error
        assert "bash" not in error and "-lc" not in error, error
        assert command not in error, (
            "the caller's line must not come back in a form that would run"
        )
        _, _, suggested = error.partition("quote it: ")
        if suggested:
            # Whatever is offered must be inert: every word literal, so
            # the argv is the words themselves.
            assert shlex.split(suggested) == shlex.split(command), (
                "the rewrite must carry the same words, only literally"
            )
            assert (
                first_shell_significant_character(suggested) is None
            ), f"the rewrite is itself refused: {suggested!r}"
    assert queue.calls == []



@pytest.mark.asyncio
async def test_a_denied_path_in_a_compound_is_still_reported_as_the_denial(workspace: Path):
    """codex review round 1, P1 — a regression this fix introduced.

    The refusal originally ran at the tool boundary, before any gate.
    So ``cat <deny_paths file> | tr A-Z a-z`` stopped being a deny-list
    hit and became a shape complaint that handed back a runnable
    ``bash -lc`` line — and running that line printed the file. A hard
    deny turned into "denied, and here is how", with no audit row for
    the denial.

    The check now sits inside the gate sequence, after both deny-lists
    and before the queue.
    """
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    secret = workspace / "secret" / "leak.txt"

    envelope = await feature.shell(command=f"cat {secret} | tr a-z A-Z", timeout=5)

    assert envelope.status is not ToolResultStatus.OK
    assert envelope.error.startswith("path_policy:deny"), envelope.error
    assert "bash -lc" not in envelope.error, (
        "a denied command must not be handed a form that would run it"
    )
    assert queue.calls == []


@pytest.mark.asyncio
async def test_a_literal_dollar_is_refused_and_the_remedy_runs_it(workspace: Path):
    """The rule is sound, not exact — and this is what that costs.

    ``echo price$`` reaches the program identically under bash and
    under direct exec, so refusing it is a false refusal. It is the
    price of a rule that cannot silently miss: five review rounds of
    modelling bash's expansions each left another spelling running with
    the wrong argv, and each near-miss was the defect this ticket
    exists to close.

    The cost is bounded because the refusal converges. Quoting makes
    bash and exec agree by construction, and the quoted form runs and
    prints exactly what the caller wanted.
    """
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()

    refusal = await feature.shell(command="echo price$", timeout=5)
    assert refusal.status is not ToolResultStatus.OK
    _, _, suggested = refusal.error.partition("quote it: ")
    assert suggested == "echo 'price$'", refusal.error

    envelope = await feature.shell(command=suggested, timeout=5)
    assert envelope.status is ToolResultStatus.OK, envelope.error
    assert envelope.data["stdout"].strip() == "price$"

    from kestrel_sovereign.features.computer_use.policy import (
        command_contains_unquoted_shell_control,
    )

    assert command_contains_unquoted_shell_control("echo price$") is True, (
        "the compound guard keeps its own wider reading"
    )


@pytest.mark.asyncio
async def test_the_refusal_names_what_the_character_would_have_done(workspace: Path):
    """A caller told only "no" cannot tell which part was the problem.

    The audit rule carries the character too, so refusals can be
    counted by cause on the log that found this defect in the first
    place.
    """
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()

    cases = {
        "echo hi | wc -l": "pipe one command's output into the next",
        "echo hi # note": "start a comment hiding the rest of the line",
        "echo ~": "expand to a home directory",
        "echo *.py": "expand to the filenames it matches",
        "echo {a,b}": "expand into several arguments",
    }
    for command, phrase in cases.items():
        envelope = await feature.shell(command=command, timeout=5)
        assert envelope.status is not ToolResultStatus.OK, command
        assert phrase in envelope.error, (command, envelope.error)

    rows = [
        json.loads(line)
        for line in (workspace / "audit.jsonl").read_text().splitlines()
    ]
    rules = [r["args"]["rule"] for r in rows if r["outcome"] == "denied"]
    assert rules == [
        "shell_syntax:|",
        "shell_syntax:#",
        "shell_syntax:~",
        "shell_syntax:*",
        "shell_syntax:{",
    ], rules



@pytest.mark.asyncio
async def test_command_position_grammar_is_refused(workspace: Path):
    """codex review round 7, P1 — the allow-list cannot see this.

    Every character in ``eval`` and ``FOO=x`` is inert, so the
    character rule passes them. They matter because the DEFAULT backend
    does not exec the argv it is handed: DockerSandboxBackend rebuilds
    a bash script from it, where a bare word in command position is
    grammar. ``eval 'dd ...'`` then runs ``dd`` with only ``eval``
    vetted — and ``eval`` is not a binary at all.

    Measured before fixing: the script the backend builds for
    ``eval 'printf HACKED'`` prints HACKED under bash, while the local
    backend raises FileNotFoundError. The backend's own claim to exec
    argv is #3187.
    """
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()

    for command, fragment in (
        ("eval 'printf HACKED'", "runs a command of its own"),
        ("exec printf HACKED", "runs a command of its own"),
        ("FOO=x printf pwned", "sets a variable for another command"),
        ("if true", "introduces a compound command"),
    ):
        envelope = await feature.shell(command=command, timeout=5)
        assert envelope.status is not ToolResultStatus.OK, command
        assert fragment in envelope.error, (command, envelope.error)
        assert "Nothing ran." in envelope.error

    assert queue.calls == [], (
        "grammar is refused before the operator is asked to approve it"
    )

    # The positive control: an ordinary program with the same shape of
    # arguments still runs, so the guard is not simply refusing
    # everything.
    ran = await feature.shell(command="printf ok", timeout=5)
    assert ran.status is ToolResultStatus.OK, ran.error
    assert ran.data["stdout"] == "ok"


@pytest.mark.asyncio
async def test_a_refused_command_is_audited(workspace: Path):
    """Every refusal in this gate sequence writes an audit row.

    That log is how the defect was found at all — 128 calls counted on
    the live surface — and it is how anyone checks whether the refusal
    is firing in production. A refusal that leaves no row is invisible.
    """
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()

    await feature.shell(command=f"cat {workspace / 'ok.txt'} | tr a-z A-Z", timeout=5)

    rows = [
        json.loads(line)
        for line in (workspace / "audit.jsonl").read_text().splitlines()
    ]
    refusals = [r for r in rows if r["outcome"] == "denied"]
    assert len(refusals) == 1, rows
    assert refusals[0]["args"]["rule"] == "shell_syntax:|"
    assert refusals[0]["allowed_by"][-1] == "denied:shell_syntax"


@pytest.mark.asyncio
async def test_shell_runs_a_quoted_metacharacter(workspace: Path):
    """Over-refusal is its own defect: a quoted ``|`` is inert to a
    shell, so it is an ordinary argument here too and must still run.

    ``echo`` is the fixture's allow-listed binary, so this also pins
    that the refusal check does not disturb the ALLOW fast path.
    """
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.shell(command="echo \'a|b\'", timeout=5)
    assert envelope.status is ToolResultStatus.OK, envelope.error
    assert envelope.data["stdout"].strip() == "a|b"
    assert queue.calls == [], "allow-listed binary must still bypass the queue"


@pytest.mark.asyncio
async def test_shell_unlisted_binary_routes_through_queue(workspace: Path):
    """#1694: an unlisted binary (not on allow OR deny list) now
    returns REQUIRE_APPROVAL and reaches the ApprovalQueue, where the
    operator (or auto-mode) decides. Previously the policy gate would
    hard-deny before the queue was even consulted."""
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    # ``true`` is neither allow- nor deny-listed in the fixture (only
    # ``echo`` is allow, ``rm`` is deny). Under the new contract it
    # routes through the queue, the queue says yes, the shell runs.
    envelope = await feature.shell(command="true", timeout=5)
    assert envelope.status is ToolResultStatus.OK
    assert envelope.data["returncode"] == 0
    assert len(queue.calls) == 1, (
        "unlisted binary must reach the ApprovalQueue (#1694)"
    )


@pytest.mark.asyncio
async def test_shell_coerces_string_timeout(workspace: Path):
    # The LLM may pass timeout as a string ("60"); the backend does a
    # numeric ``<= 0`` comparison, so the boundary must coerce to int
    # rather than raising TypeError deep in the exec path (issue #1302).
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.shell(command="echo hi", timeout="5")
    assert envelope.status is ToolResultStatus.OK
    assert envelope.data["returncode"] == 0


@pytest.mark.asyncio
async def test_shell_rejects_non_numeric_timeout(workspace: Path):
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.shell(command="echo hi", timeout="soon")
    assert envelope.status is not ToolResultStatus.OK
    assert "timeout must be an integer" in envelope.error
    # Rejected before reaching the approval gate.
    assert queue.calls == []


@pytest.mark.asyncio
async def test_shell_rejects_non_positive_timeout(workspace: Path):
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    envelope = await feature.shell(command="echo hi", timeout="0")
    assert envelope.status is not ToolResultStatus.OK
    assert "timeout must be a positive" in envelope.error
    assert queue.calls == []


@pytest.mark.asyncio
async def test_audit_log_records_denied_calls(workspace: Path):
    queue = FakeApprovalQueue()
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=False),  # privacy gate will refuse
        grants={"filesystem_read", "shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    # initialize requires shell grants on local backend; we keep them.
    await feature.initialize()
    await feature.fs_read(path=str(workspace / "ok.txt"))
    log_path = workspace / "audit.jsonl"
    assert log_path.exists()
    line = log_path.read_text().strip()
    assert line  # at least one record
    import json

    parsed = json.loads(line.splitlines()[0])
    assert parsed["outcome"] == "denied"
    assert parsed["allowed_by"] == ["denied:privacy"]


def _f137_config(root: Path) -> dict[str, Any]:
    """Config for the F137 argv path-policy tests.

    ``allow_root`` is a *subdirectory* of ``root`` so a file can exist
    inside ``root`` but outside the allow-list (to exercise the
    non-allow-listed REQUIRE_APPROVAL branch). ``cat`` is auto-approved
    so the reader short-circuits to ALLOW on ``argv[0]`` — the argv path
    policy is the only thing that can still gate the file argument.
    """
    return {
        "enabled": True,
        "backend": "local",
        "allowed_paths": [str(root / "proj")],
        "deny_paths": [str(root / "secret")],
        "allowed_binaries": ["echo", "cat"],
        "denied_binaries": ["rm"],
        "auto_approve_read": True,
        "audit_log_path": str(root / "audit.jsonl"),
    }


async def _make_f137_feature(root: Path, agent: FakeAgent) -> ComputerUseFeature:
    feature = ComputerUseFeature(agent)
    feature._cfg = _f137_config(root)
    await feature.initialize()
    return feature


@pytest.mark.asyncio
async def test_shell_deny_path_arg_hard_rejects_auto_approved_reader(tmp_path: Path):
    """F137: an auto-approved reader (``cat``) reading a ``deny_paths``
    file is DENIED even though ``argv[0]`` short-circuits to ALLOW."""
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "creds").write_text("aws keys")
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_f137_feature(tmp_path, agent)
    envelope = await feature.shell(
        command=f"cat {tmp_path / 'secret' / 'creds'}", timeout=5
    )
    assert envelope.status is not ToolResultStatus.OK
    assert envelope.error.startswith("path_policy:deny")
    assert queue.calls == [], "deny-path arg must not reach the queue"


@pytest.mark.asyncio
async def test_shell_deny_path_arg_universal_even_for_approval_routed_binary(
    tmp_path: Path,
):
    """F137/Q2: the deny_paths hit is universal — an approval-routed
    (unlisted) binary reading a deny file is DENIED before the queue,
    matching the ``fs_*`` 'never, even with approval' guarantee."""
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "creds").write_text("aws keys")
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_f137_feature(tmp_path, agent)
    # ``grep`` is neither allow- nor deny-listed → would route to the
    # queue on argv[0], but the deny-path arg must deny first.
    envelope = await feature.shell(
        command=f"grep secret {tmp_path / 'secret' / 'creds'}", timeout=5
    )
    assert envelope.status is not ToolResultStatus.OK
    assert envelope.error.startswith("path_policy:deny")
    assert queue.calls == []


@pytest.mark.asyncio
async def test_shell_non_allow_listed_path_arg_requires_approval(tmp_path: Path):
    """F137: an auto-approved reader reading an existing, non-allow-listed
    path routes through the queue instead of silently running."""
    (tmp_path / "proj").mkdir()
    (tmp_path / "outside.txt").write_text("data")
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_f137_feature(tmp_path, agent)
    envelope = await feature.shell(
        command=f"cat {tmp_path / 'outside.txt'}", timeout=5
    )
    assert envelope.status is ToolResultStatus.OK
    assert len(queue.calls) == 1, (
        "non-allow-listed path arg must reach the queue even for an "
        "auto-approved reader"
    )


@pytest.mark.asyncio
async def test_shell_allow_listed_path_arg_still_runs(tmp_path: Path):
    """F137: an auto-approved reader reading an allow-listed path still
    bypasses the queue and runs."""
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "ok.txt").write_text("hello world")
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_f137_feature(tmp_path, agent)
    envelope = await feature.shell(
        command=f"cat {tmp_path / 'proj' / 'ok.txt'}", timeout=5
    )
    assert envelope.status is ToolResultStatus.OK
    assert envelope.data["returncode"] == 0
    assert "hello world" in envelope.data["stdout"]
    assert queue.calls == [], "allow-listed path arg must bypass the queue"


@pytest.mark.asyncio
async def test_shell_non_path_argv_token_ignored(tmp_path: Path):
    """F137: a non-flag token that doesn't resolve to an existing path
    (e.g. an ``rg``/``grep`` search pattern) is ignored — an
    auto-approved binary with only such tokens still bypasses the queue."""
    (tmp_path / "proj").mkdir()
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_f137_feature(tmp_path, agent)
    # ``echo`` is auto-approved; ``some-nonexistent-pattern`` doesn't
    # resolve to a file, so it must not force an approval prompt.
    envelope = await feature.shell(
        command="echo some-nonexistent-pattern", timeout=5
    )
    assert envelope.status is ToolResultStatus.OK
    assert queue.calls == []


@pytest.mark.asyncio
async def test_local_backend_requires_both_shell_grants(workspace: Path):
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed"},  # missing host grant
        queue=FakeApprovalQueue(),
    )
    feature = ComputerUseFeature(agent)
    feature._cfg = _config(workspace, backend="local")
    # initialize swallows CapabilityBlocked and leaves backend=None
    await feature.initialize()
    assert feature._backend is None
    envelope = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert envelope.status is not ToolResultStatus.OK
    assert envelope.error.startswith("readiness:")
