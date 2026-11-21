"""Tests for ComputerUseFeature gate ordering and lifecycle (#838)."""

from __future__ import annotations

import json
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
        "capture_dir": str(workspace / "captures"),
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
    character rule passes them. They mattered because the DEFAULT
    backend did not exec the argv it was handed: DockerSandboxBackend
    rebuilt a bash script from it, where a bare word in command
    position is grammar, so ``eval 'dd ...'`` ran ``dd`` with only
    ``eval`` vetted — and ``eval`` is not a binary at all.

    Measured before that was fixed: the script the backend built for
    ``eval 'printf HACKED'`` printed HACKED, while the local backend
    raised FileNotFoundError. #3187 closed the divergence at the
    backend; this refusal stays in front of it, because the answer a
    caller needs is "that is not a program", not exit 127 from a
    container that started to find out.
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
        ("eval 'printf HACKED'", "a shell builtin, not a program"),
        ("exec printf HACKED", "a shell builtin, not a program"),
        # codex round 8: `trap` dispatches its command later. Naming
        # the dispatching builtins one at a time is what round 7 did
        # and round 8 answered; the rule is now "a builtin is not a
        # program", so this falls out rather than being listed.
        ("trap 'printf HACKED' EXIT", "a shell builtin, not a program"),
        ("FOO=x printf pwned", "sets a variable for another command"),
        ("FOO+=x printf pwned", "sets a variable for another command"),
        ("if true", "introduces a compound command"),
        ("coproc printf x", "introduces a compound command"),
    ):
        envelope = await feature.shell(command=command, timeout=5)
        assert envelope.status is not ToolResultStatus.OK, command
        assert fragment in envelope.error, (command, envelope.error)
        assert "Nothing ran." in envelope.error

    assert queue.calls == [], (
        "grammar is refused before the operator is asked to approve it"
    )

    # The positive control: ordinary programs still run, including the
    # builtins that are ALSO real binaries — refusing those would cost
    # without buying anything, since they behave the same either way.
    for command, expected in (("printf ok", "ok"), ("echo hi", "hi\n")):
        ran = await feature.shell(command=command, timeout=5)
        assert ran.status is ToolResultStatus.OK, (command, ran.error)
        assert ran.data["stdout"] == expected, command


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
        "capture_dir": str(root / "captures"),
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


@pytest.mark.asyncio
async def test_shell_backend_refusal_names_backend_none_for_a_read_only_grant(tmp_path: Path):
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"filesystem_read"},
        queue=FakeApprovalQueue(),
    )
    feature = ComputerUseFeature(agent)
    feature._cfg = _config(tmp_path, backend="docker")
    await feature.initialize()

    envelope = await feature.fs_read(path=str(tmp_path / "ok.txt"))
    assert envelope.error.startswith("readiness:backend not initialized")
    assert 'backend = "none"' in envelope.error


@pytest.mark.asyncio
async def test_filesystem_only_backend_cannot_execute():
    from kestrel_sovereign.features.computer_use.backends import (
        CapabilityBlocked,
        FilesystemOnlyBackend,
    )

    backend = FilesystemOnlyBackend()
    assert backend.supports_exec is False
    with pytest.raises(CapabilityBlocked):
        await backend.exec(["true"], cwd=None, env=None, timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["None", "nnone", "filesystem", "Docker", "", None, 0, ["none"]])
async def test_an_unknown_backend_is_refused_not_read_as_docker(tmp_path: Path, value):
    """A mistyped "none" must not build a backend that can execute (#3760)."""
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"filesystem_read", "shell_execution_sandboxed", "shell_execution_host"},
        queue=FakeApprovalQueue(),
    )
    feature = ComputerUseFeature(agent)
    feature._cfg = {**_config(tmp_path), "backend": value}
    await feature.initialize()

    assert feature._backend is None
    envelope = await feature.shell(command="echo hi")
    assert envelope.error.startswith("readiness:backend not initialized")
    assert f"unknown backend {value!r}" in envelope.error


# --- auto_approved_subcommands: wired into the shell tool's gate ---


def _git_repo(root: Path) -> Path:
    import subprocess

    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def _subcommand_config(root: Path, subcommands: Any) -> dict[str, Any]:
    return {
        **_config(root),
        "allowed_binaries": ["git"],
        "auto_approved_subcommands": subcommands,
    }


async def _make_subcommand_feature(root: Path, queue: FakeApprovalQueue, subcommands: Any):
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = ComputerUseFeature(agent)
    feature._cfg = _subcommand_config(root, subcommands)
    await feature.initialize()
    return feature


@pytest.mark.asyncio
async def test_a_listed_git_read_runs_without_the_queue(tmp_path: Path):
    repo = _git_repo(tmp_path)
    queue = FakeApprovalQueue(decision=(False, "once"))
    feature = await _make_subcommand_feature(
        tmp_path, queue, {"git": {"allow": ["status"], "value_options": ["-C"]}}
    )
    envelope = await feature.shell(command=f"git -C {repo} status --short", timeout=10)
    assert envelope.status is ToolResultStatus.OK, envelope.error
    assert queue.calls == []


@pytest.mark.asyncio
async def test_an_unlisted_git_write_queues_inside_an_allowed_path(tmp_path: Path):
    """The path is allow-listed for reading, which used to be enough for
    any git subcommand to skip the queue. ``reset`` is not listed."""
    repo = _git_repo(tmp_path)
    queue = FakeApprovalQueue(decision=(False, "once"))
    feature = await _make_subcommand_feature(
        tmp_path, queue, {"git": {"allow": ["status"], "value_options": ["-C"]}}
    )
    envelope = await feature.shell(command=f"git -C {repo} reset --hard", timeout=10)
    assert envelope.status is not ToolResultStatus.OK
    assert len(queue.calls) == 1
    assert queue.calls[0]["args"]["rule"] == "subcommand:git:reset"


@pytest.mark.asyncio
async def test_a_malformed_subcommand_policy_refuses_every_call(tmp_path: Path):
    queue = FakeApprovalQueue()
    feature = await _make_subcommand_feature(tmp_path, queue, {"git": {"allow": []}})
    assert feature._backend is None
    envelope = await feature.shell(command="git status", timeout=10)
    assert envelope.error.startswith("readiness:backend not initialized")
    assert "must name at least one subcommand" in envelope.error
    assert queue.calls == []


# ---------------------------------------------------------------------------
# #4069: fs_read returns the file once, in data.content
# ---------------------------------------------------------------------------


def _fs_read_tool(feature: ComputerUseFeature):
    return next(t for t in feature.get_tools() if t.name == "fs_read")


@pytest.mark.asyncio
async def test_fs_read_model_facing_result_carries_the_file_once(workspace: Path):
    """The envelope every model-facing path serializes (the ``DynamicTool``
    wrapper) holds an N-character file once. With the old copy in
    ``confirmation`` a 20,000-character file serialized to over 40,000."""
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"filesystem_read"},
        queue=FakeApprovalQueue(),
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    # No character JSON escapes, so the content serializes to its own length.
    text = "kestrel " * 2_500
    assert len(text) == 20_000
    target = workspace / "big.txt"
    target.write_text(text)

    response = await _fs_read_tool(feature).execute(path=str(target))

    assert response["status"] == "ok", response.get("error")
    assert len(json.dumps(response)) <= len(text) + 1_000
    assert response["data"]["content"] == text
    assert response["confirmation"] == f"Read 20000 bytes from {target.resolve()}."


@pytest.mark.parametrize("content", ["first line\nsecond line\n", ""], ids=["text", "empty"])
@pytest.mark.asyncio
async def test_fs_read_command_shows_the_file_content(
    native_feature_case, workspace: Path, content: str
):
    """``!fs-read`` through the real command path (``CommandHandler`` →
    ``TaskManager.execute_command`` → the tool) prints the summary line and
    then the file, once."""
    from kestrel_sovereign.command_handler import CommandHandler
    from tests.utils.native_tool_effects import drive_native

    agent, manager = native_feature_case.agent, native_feature_case.manager
    agent.privacy_agent._privacy_config.computer_access = True
    agent.granted_capabilities = frozenset(
        {"filesystem_read", "shell_execution_sandboxed", "shell_execution_host"}
    )
    feature = ComputerUseFeature(agent)
    feature._cfg = _config(workspace)
    await feature.initialize()
    manager.register_provider(
        feature.name, feature.get_agent_skills(), feature, {"!fs-read": "fs_read"}
    )
    target = workspace / "notes.txt"
    target.write_text(content)
    handler = CommandHandler(agent, task_manager=manager)

    text = await drive_native(agent, lambda: handler.handle(f"!fs-read {target}"))

    summary = f"Read {len(content.encode())} bytes from {target.resolve()}."
    assert text == (f"{summary}\n{content}" if content else summary)


@pytest.mark.parametrize("name", ["missing.txt", "secret/leak.txt"], ids=["missing", "refused"])
@pytest.mark.asyncio
async def test_fs_read_missing_or_refused_path_is_unchanged(workspace: Path, name: str):
    """A read that fails returns only its error: no confirmation, no data."""
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"filesystem_read"},
        queue=FakeApprovalQueue(),
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()

    envelope = await feature.fs_read(path=str(workspace / name))

    assert envelope.status is ToolResultStatus.ERROR
    assert (envelope.confirmation, envelope.data) == (None, None)
    if name == "missing.txt":
        assert "No such file or directory" in envelope.error
    else:
        assert envelope.error.startswith("policy:deny")


# --- #4104: a gate exception is an audited refusal ------------------------------

import asyncio  # noqa: E402
import errno  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402

from kestrel_sovereign.features.computer_use import feature as feature_module  # noqa: E402
from tests.unit.test_computer_use_policy import python_c_program  # noqa: E402

_PYTHON = Path(sys.executable).name


def _audit_rows(root: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / "audit.jsonl").read_text().splitlines()
    ]


def _raise_name_too_long(argv, path_policy, *, cwd=None):
    """What ``evaluate_argv_paths`` raised before #4104, on a 372-byte token."""
    raise OSError(errno.ENAMETOOLONG, "File name too long", f"{cwd}/{'x' * 372}")


@pytest.mark.parametrize("cwd_name, approvals", [("proj", 0), ("elsewhere", 1)])
@pytest.mark.asyncio
async def test_shell_long_python_c_token_gets_the_short_token_decision(
    tmp_path: Path, cwd_name: str, approvals: int
):
    """C2 through the ``shell`` door: the over-long token is not a path, so
    the call is decided by its binary and cwd alone, as a short one is."""
    (tmp_path / "proj").mkdir()
    (tmp_path / "elsewhere").mkdir()
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = ComputerUseFeature(agent)
    feature._cfg = _f137_config(tmp_path)
    feature._cfg["allowed_binaries"] = ["echo", "cat", _PYTHON]
    await feature.initialize()

    for size in (372, 60):
        queue.calls.clear()
        envelope = await feature.shell(
            command=f"{sys.executable} -c '{python_c_program(size)}'",
            cwd=str(tmp_path / cwd_name),
            timeout=30,
        )
        assert envelope.status is ToolResultStatus.OK, (size, envelope.error)
        assert envelope.data["stdout"] == "ran"
        assert len(queue.calls) == approvals, size


@pytest.mark.asyncio
async def test_shell_long_token_outside_the_roots_does_not_ask(tmp_path: Path):
    """C2 where it decides the call: an allow-listed binary in an allowed cwd
    is not sent to the queue by a token that cannot name a file, wherever
    the token points."""
    (tmp_path / "proj").mkdir()
    (tmp_path / "elsewhere").mkdir()
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_f137_feature(tmp_path, agent)
    token = f"{tmp_path / 'elsewhere'}/{'x' * 372}"

    envelope = await feature.shell(
        command=f"echo {token}", cwd=str(tmp_path / "proj"), timeout=5
    )

    assert envelope.status is ToolResultStatus.OK, envelope.error
    assert envelope.data["stdout"] == f"{token}\n"
    assert queue.calls == []


@pytest.mark.asyncio
async def test_shell_gate_exception_is_an_audited_refusal(tmp_path: Path, monkeypatch):
    """C4: an exception inside the gates leaves an audit row and a failure."""
    (tmp_path / "proj").mkdir()
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_f137_feature(tmp_path, agent)
    monkeypatch.setattr(feature_module, "evaluate_argv_paths", _raise_name_too_long)

    envelope = await feature.shell(command="echo hi", timeout=5)

    assert envelope.status is ToolResultStatus.ERROR
    assert envelope.error.startswith("gate_error:OSError: [Errno ")
    assert "File name too long" in envelope.error
    assert queue.calls == []
    [row] = _audit_rows(tmp_path)
    assert row["tool"] == "shell"
    assert row["outcome"] == "denied"
    assert row["allowed_by"] == ["privacy", "constitution", "policy", "denied:gate_error"]
    assert row["error"].startswith("OSError: [Errno ")
    assert "File name too long" in row["error"]
    assert row["args"]["argv"] == ["echo", "hi"]


@pytest.mark.asyncio
async def test_fs_read_unresolvable_home_is_an_audited_refusal(workspace: Path):
    """C4 without an injection: ``~user`` naming no user makes path safety
    raise ``RuntimeError``, which used to escape the gates unaudited."""
    queue = FakeApprovalQueue()
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"filesystem_read"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()

    envelope = await feature.fs_read(path="~kestrel-no-such-user-4104/notes.txt")

    assert envelope.status is ToolResultStatus.ERROR
    assert envelope.error == "gate_error:RuntimeError: Could not determine home directory."
    [row] = _audit_rows(workspace)
    assert row["allowed_by"] == ["privacy", "constitution", "denied:gate_error"]
    assert row["error"] == "RuntimeError: Could not determine home directory."
    assert queue.calls == []


async def _fail_audit_writes(record):
    raise OSError(errno.ENOSPC, "No space left on device")


@pytest.mark.asyncio
async def test_a_gate_error_whose_audit_also_fails_still_refuses(
    tmp_path: Path, monkeypatch, caplog
):
    """The refusal stands without its row, says the row is missing, and the
    failed audit write is logged at ERROR."""
    (tmp_path / "proj").mkdir()
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=FakeApprovalQueue(),
    )
    feature = await _make_f137_feature(tmp_path, agent)
    monkeypatch.setattr(feature_module, "evaluate_argv_paths", _raise_name_too_long)
    monkeypatch.setattr(feature._audit, "write", _fail_audit_writes)
    caplog.set_level(logging.ERROR, logger=feature_module.__name__)

    envelope = await feature.shell(command="echo hi", timeout=5)

    assert envelope.status is ToolResultStatus.ERROR
    assert envelope.error.startswith("gate_error:OSError: [Errno ")
    assert envelope.error.endswith(
        "; the refusal was not audited: OSError: [Errno "
        f"{errno.ENOSPC}] No space left on device"
    )
    assert any(
        record.levelno == logging.ERROR
        and record.getMessage().startswith("computer_use: could not audit the shell gate error")
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


@pytest.mark.asyncio
async def test_a_cancellation_inside_a_gate_propagates_unchanged(
    tmp_path: Path, monkeypatch
):
    """C4 stops at ``Exception``: a cancellation is not a refusal, so it is
    neither audited nor turned into a result."""
    (tmp_path / "proj").mkdir()
    queue = FakeApprovalQueue()
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_f137_feature(tmp_path, agent)
    cancellation = asyncio.CancelledError("stop requested")

    def evaluate(argv, path_policy, *, cwd=None):
        raise cancellation

    monkeypatch.setattr(feature_module, "evaluate_argv_paths", evaluate)

    with pytest.raises(asyncio.CancelledError) as raised:
        await feature.shell(command="echo hi", timeout=5)

    assert raised.value is cancellation
    assert queue.calls == []
    assert not (tmp_path / "audit.jsonl").exists() or _audit_rows(tmp_path) == []


@pytest.mark.asyncio
async def test_gate_execution_control_is_audited_and_propagates(
    tmp_path: Path, monkeypatch
):
    """Execution control is the host's to act on, so it is not a result."""
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError

    (tmp_path / "proj").mkdir()
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=FakeApprovalQueue(),
    )
    feature = await _make_f137_feature(tmp_path, agent)
    denial = ExecutionAuthorityError("original admission retired")

    def evaluate(argv, path_policy, *, cwd=None):
        raise denial

    monkeypatch.setattr(feature_module, "evaluate_argv_paths", evaluate)

    with pytest.raises(ExecutionAuthorityError) as raised:
        await feature.shell(command="echo hi", timeout=5)

    assert raised.value is denial
    [row] = _audit_rows(tmp_path)
    assert row["allowed_by"][-1] == "denied:gate_error"
    assert row["error"] == "ExecutionAuthorityError: original admission retired"


@pytest.mark.parametrize("audit_fails", [False, True])
@pytest.mark.asyncio
async def test_a_gate_exception_on_the_direct_tool_door_does_not_fence_the_turn(
    tmp_path: Path, monkeypatch, native_tool_effect_agent, audit_fails: bool
):
    """The 2026-11-14 shape, reproduced: the model's ``shell`` call reaches
    the gate through the direct-tool door. Its claim settles as a returned
    error, not ``UNCONFIRMED``, so no uncertain effect fences the turn, and
    that holds when the refusal's audit write fails too."""
    from kestrel_sdk.llm.response import ToolCall

    from tests.utils.native_tool_effects import drive_native

    (tmp_path / "proj").mkdir()
    agent = await native_tool_effect_agent(did="did:test:computer-use-gate-error")
    agent.privacy_agent._privacy_config.computer_access = True
    agent.granted_capabilities = frozenset(
        {"shell_execution_sandboxed", "shell_execution_host"}
    )
    feature = ComputerUseFeature(agent)
    feature._cfg = _f137_config(tmp_path)
    await feature.initialize()
    agent.features[type(feature).__name__] = feature
    agent._register_explored_feature_tools(feature)
    monkeypatch.setattr(feature_module, "evaluate_argv_paths", _raise_name_too_long)
    if audit_fails:
        monkeypatch.setattr(feature._audit, "write", _fail_audit_writes)

    result = await drive_native(
        agent,
        lambda: agent._dispatch_tool_call(
            ToolCall(id="tc-4104", name="shell", arguments={"command": "echo hi"}),
            features_by_tool_name={},
            known_tools={"shell"},
            messages=[],
            iteration=0,
            user_message="",
            session_id="sess-4104",
        ),
    )

    assert "gate_error:OSError" in json.dumps(result)
    assert await agent.tool_effect_store.database.fetchone(
        "SELECT state FROM tool_effect_claims ORDER BY rowid DESC LIMIT 1"
    ) == ("RETURNED_ERROR",)
    if audit_fails:
        assert "the refusal was not audited" in json.dumps(result)
        assert not (tmp_path / "audit.jsonl").exists()
    else:
        [row] = _audit_rows(tmp_path)
        assert row["allowed_by"][-1] == "denied:gate_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edition, outside_grant, expected",
    [(1, False, True), (2, False, False), (2, True, True)],
)
async def test_file_workspace_scope_keeps_independent_legacy_and_v2_decisions(
    workspace, edition, outside_grant, expected,
):
    from kestrel_sovereign.constitution.grants import GrantResolution

    grants = {"filesystem_read"}
    if outside_grant:
        grants.add("filesystem_outside_workspace")
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(privacy=PrivacyConfig(computer_access=True), grants=grants, queue=queue)
    agent.storage_path = str(workspace / "state.db")
    if edition == 2:
        agent._constitution_host_grant_baseline = GrantResolution(
            frozenset(grants), "test authority", "test", "absent",
        )
    inside = workspace / "workspace"
    inside.mkdir()
    (inside / "note.txt").write_text("ordinary fixture content")
    feature = await _make_feature(workspace, agent=agent, backend="none")
    await feature.initialize()

    in_workspace = await feature.fs_read(path=str(inside / "note.txt"))
    assert in_workspace.status is ToolResultStatus.OK
    out_of_workspace = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert (out_of_workspace.status is ToolResultStatus.OK) is expected
    assert not queue.calls
    if not expected:
        assert "filesystem_outside_workspace" in out_of_workspace.error
