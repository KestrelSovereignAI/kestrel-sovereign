"""Unit tests for the Emma closed-loop machinery (epic #1290).

Covers:
  D1 - scoped auto-approve matcher, two-phase audit, ApprovalQueue seam
  D3 - pre-turn state block gating + assembly + token cap
  D4 - talon_file_and_claim file → parse → claim flow
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign.security.auto_approve import (
    AutoApprovePolicy,
    AutoApproveRule,
    derive_command,
    suggest_rule_from_command,
)
from kestrel_sovereign.features.security.permissions import (
    PermissionLevel,
    PermissionStore,
)
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue


# --------------------------------------------------------------------------
# D1 - matcher
# --------------------------------------------------------------------------

def test_derive_command_shell_and_compute():
    assert derive_command(
        "computer_use", "shell", {"argv": ["gh", "issue", "create"]}
    ) == "gh issue create"
    # compute.run_script is intentionally NOT auto-approvable: a regex
    # allowlist can't bind a signed script's content (codex P1, #1290).
    assert derive_command(
        "ComputeFeature", "run_script",
        {"script_name": "x.py", "purpose": "do thing"},
    ) is None
    assert derive_command("computer_use", "fs_read", {"path": "/x"}) is None
    # codex P2: the PRE_TOOL_USE hook passes raw {"command": ...} (no
    # argv) and cannot finalize the audit row — must NOT match, so only
    # the internal gate (argv) is the auto-approve+audit point.
    assert derive_command(
        "computer_use", "shell", {"command": "gh issue create -R o/r"}
    ) is None
    assert derive_command(
        "computer_use", "shell", {"argv": ["gh", "issue", "create"]}
    ) == "gh issue create"


@pytest.mark.asyncio
async def test_policy_match_positive_and_scopes():
    rule = AutoApproveRule(
        pattern=r"^gh issue (create|comment) -R KestrelSovereignAI/kestrel-sovereign",
        repo_scope="KestrelSovereignAI/kestrel-sovereign",
        agent="Emma",
    )
    pol = AutoApprovePolicy([rule])
    args = {
        "argv": [
            "gh", "issue", "create",
            "-R", "KestrelSovereignAI/kestrel-sovereign",
            "--title", "t",
        ]
    }
    # Positive
    m = await pol.evaluate(
        agent_name="Emma", feature_name="computer_use",
        tool_name="shell", tool_args=args,
    )
    assert m is not None and m.rule is rule
    # Wrong agent → no match
    assert await pol.evaluate(
        agent_name="Claw", feature_name="computer_use",
        tool_name="shell", tool_args=args,
    ) is None
    # Wrong repo (repo_scope guard) → no match
    other = {"argv": ["gh", "issue", "create", "-R", "evil/repo"]}
    assert await pol.evaluate(
        agent_name="Emma", feature_name="computer_use",
        tool_name="shell", tool_args=other,
    ) is None
    # Non-allowlisted command → no match
    rm = {"argv": ["rm", "-rf", "/"]}
    assert await pol.evaluate(
        agent_name="Emma", feature_name="computer_use",
        tool_name="shell", tool_args=rm,
    ) is None


@pytest.mark.asyncio
async def test_repo_scope_is_exact_not_substring():
    """codex P1: owner/repo must never authorise owner/repo-fork."""
    rule = AutoApproveRule(
        pattern=r"^gh issue create -R o/r",
        repo_scope="o/r", agent="Emma",
    )
    pol = AutoApprovePolicy([rule])
    forked = {"argv": ["gh", "issue", "create", "-R", "o/r-fork",
                        "--title", "x"]}
    assert await pol.evaluate(
        agent_name="Emma", feature_name="computer_use",
        tool_name="shell", tool_args=forked,
    ) is None
    exact = {"argv": ["gh", "issue", "create", "-R", "o/r", "--title", "x"]}
    assert await pol.evaluate(
        agent_name="Emma", feature_name="computer_use",
        tool_name="shell", tool_args=exact,
    ) is not None


@pytest.mark.asyncio
async def test_rule_without_repo_scope_never_auto_approves():
    """codex P2: a scoped allowlist must be scoped — an unscoped rule
    must never auto-approve in any repo context."""
    rule = AutoApproveRule(pattern=r"^echo hi", repo_scope="", agent="Emma")
    pol = AutoApprovePolicy([rule])
    args = {"argv": ["echo", "hi"]}
    assert await pol.evaluate(
        agent_name="Emma", feature_name="computer_use",
        tool_name="shell", tool_args=args,
    ) is None


def test_suggest_rule_has_trailing_boundary():
    pattern, _ = suggest_rule_from_command("gh issue create -R o/r --title x")
    import re as _re

    rx = _re.compile(pattern)
    assert rx.search("gh issue create -R o/r --title y")
    # Must NOT match a longer repo token.
    assert not rx.search("gh issue create -R o/r-fork --title y")


def test_suggest_rule_from_command_is_conservative():
    pattern, repo = suggest_rule_from_command(
        'gh issue create -R KestrelSovereignAI/kestrel-sovereign '
        '--title "hi" --body "there"'
    )
    assert repo == "KestrelSovereignAI/kestrel-sovereign"
    # Prefix anchored, free-form tail (title/body) excluded.
    assert pattern.startswith("^gh\\ issue\\ create")
    assert "title" not in pattern and "there" not in pattern


# --------------------------------------------------------------------------
# D1 - two-phase audit + dynamic rules (real sqlite)
# --------------------------------------------------------------------------

@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as d:
        s = PermissionStore(str(Path(d) / "k.db"))
        await s.initialize()
        yield s


@pytest.mark.asyncio
async def test_two_phase_audit(store):
    aid = await store.log_auto_approve(
        agent_did="did:pkh:emma", agent_name="Emma",
        feature_name="computer_use", tool_name="shell",
        command="gh issue create -R o/r", pattern="^gh issue create",
        repo_scope="o/r", rule_source="seed",
    )
    rows = await store.get_auto_approve_audit()
    assert rows[0]["id"] == aid
    assert rows[0]["exit_code"] is None
    assert rows[0]["agent_did"] == "did:pkh:emma"
    await store.finalize_auto_approve(aid, 0)
    rows = await store.get_auto_approve_audit()
    assert rows[0]["exit_code"] == 0
    assert rows[0]["completed_at"] is not None


@pytest.mark.asyncio
async def test_dynamic_rules_add_list_revoke(store):
    await store.add_auto_approve_rule(
        pattern="^gh issue create", repo_scope="o/r",
        agent="Emma", added_by="mews_approval",
    )
    rules = await store.list_auto_approve_rules()
    assert len(rules) == 1 and rules[0]["agent"] == "Emma"
    assert await store.remove_auto_approve_rule(rules[0]["id"]) is True
    assert await store.list_auto_approve_rules() == []


@pytest.mark.asyncio
async def test_approval_queue_auto_approves_and_deny_still_wins(store):
    agent = SimpleNamespace(_agent_name="Emma", did="did:pkh:emma")
    pol = AutoApprovePolicy(
        [AutoApproveRule(
            pattern=r"^gh issue create -R o/r",
            repo_scope="o/r", agent="Emma",
        )],
        store,
    )
    q = ApprovalQueue(permission_store=store, auto_approve_policy=pol,
                       agent=agent)
    args = {"argv": ["gh", "issue", "create", "-R", "o/r", "--title", "t"]}

    approved, scope = await q.request_approval(
        "computer_use", "shell", args, timeout=1,
    )
    assert approved is True
    assert scope.startswith("auto_approve:")
    audit = await store.get_auto_approve_audit()
    assert audit and audit[0]["command"].startswith("gh issue create")

    # Operator DENY must hard-stop even when the allowlist would match.
    await store.set_permission(
        "computer_use", "shell", PermissionLevel.DENY, scope="always",
    )
    approved2, scope2 = await q.request_approval(
        "computer_use", "shell", args, timeout=1,
    )
    assert approved2 is False and scope2 == "denied"


@pytest.mark.asyncio
async def test_canonical_deny_blocks_auto_approve(store):
    """codex P1: DENY registered under the canonical class name
    (ComputerUseFeature.shell) must block the internal-gate
    (computer_use.shell) auto-approve path — revocation must work."""
    agent = SimpleNamespace(_agent_name="Emma", did="did:pkh:emma")
    pol = AutoApprovePolicy(
        [AutoApproveRule(pattern=r"^gh issue create -R o/r",
                         repo_scope="o/r", agent="Emma")],
        store,
    )
    q = ApprovalQueue(permission_store=store, auto_approve_policy=pol,
                      agent=agent)
    args = {"argv": ["gh", "issue", "create", "-R", "o/r", "--title", "t"]}
    # Deny under the canonical permissions-UI key, NOT the gate key.
    await store.set_permission(
        "ComputerUseFeature", "shell", PermissionLevel.DENY, scope="always",
    )
    approved, scope = await q.request_approval(
        "computer_use", "shell", args, timeout=1,
    )
    assert approved is False and scope == "denied"


# --------------------------------------------------------------------------
# D3 - pre-turn state block
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_preturn_block_disabled_by_default(monkeypatch):
    from kestrel_sovereign.agent import preturn_state

    monkeypatch.setattr(preturn_state, "load_section", lambda s: {},
                         raising=False)
    agent = SimpleNamespace(_agent_name="Emma", storage_path=None)
    assert await preturn_state.build_preturn_state_block(agent) is None


@pytest.mark.asyncio
async def test_preturn_block_enabled_for_emma(monkeypatch, tmp_path):
    import kestrel_sovereign.config as cfg

    monkeypatch.setattr(
        cfg, "load_section",
        lambda s: {"enabled": True, "agents": ["Emma"], "max_tokens": 500}
        if s == "preturn_state" else {},
    )
    db = tmp_path / "k.db"
    db.write_text("")
    (tmp_path / "strategy.yaml").write_text(
        "vision: Close the loop\nblockers: [keyboard]\n"
    )
    sec = SimpleNamespace(approval_queue=SimpleNamespace(pending_count=3))
    agent = SimpleNamespace(
        _agent_name="Emma",
        storage_path=str(db),
        get_feature=lambda name: sec if name == "SecurityFeature" else None,
    )
    from kestrel_sovereign.agent.preturn_state import (
        build_preturn_state_block,
    )

    block = await build_preturn_state_block(agent)
    assert block is not None
    assert "AGENT STATE" in block
    assert "Close the loop" in block
    assert "Pending approvals: 3" in block
    # Other agents must not get it.
    agent._agent_name = "Claw"
    assert await build_preturn_state_block(agent) is None


# --------------------------------------------------------------------------
# D4 - talon_file_and_claim
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_talon_file_and_claim_files_then_claims():
    from kestrel_sdk.tools.result import ToolResult
    from kestrel_sovereign.features.talon.coordinator import (
        TalonCoordinatorFeature,
    )

    feat = TalonCoordinatorFeature.__new__(TalonCoordinatorFeature)

    class FakeCU:
        async def shell(self, command, timeout=60):
            assert command.startswith("gh issue create -R ")
            return ToolResult.ok(
                "created",
                data={
                    "returncode": 0,
                    "stdout": "https://github.com/o/r/issues/4242\n",
                },
            )

    feat.agent = SimpleNamespace(
        get_feature=lambda n: FakeCU() if n == "ComputerUseFeature" else None
    )

    async def fake_claim(repo, issue):
        assert issue == 4242
        return ToolResult.ok("dispatched",
                             data={"dispatched": True, "job_id": "job-1"})

    feat.talon_claim = fake_claim

    res = await feat.talon_file_and_claim(
        title="Mesh reliability", body="fix it", labels="bug,agent-claimed",
        repo="o/r",
    )
    assert res.data["filed"] is True
    assert res.data["issue_number"] == 4242
    assert res.data["job_id"] == "job-1"
    assert res.data["dispatched"] is True


@pytest.mark.asyncio
async def test_talon_file_and_claim_strips_talon_reserved_labels():
    """Filing with agent-claimed makes Talon's claim abort as 'already
    claimed' (#1299/#1301/#1303). The primitive must drop Talon's
    reserved lifecycle labels; Talon stamps agent-claimed itself."""
    from kestrel_sdk.tools.result import ToolResult
    from kestrel_sovereign.features.talon.coordinator import (
        TalonCoordinatorFeature,
    )

    seen = {}

    class FakeCU:
        async def shell(self, command, timeout=60):
            seen["command"] = command
            return ToolResult.ok(
                "created",
                data={"returncode": 0,
                      "stdout": "https://github.com/o/r/issues/77\n"},
            )

    feat = TalonCoordinatorFeature.__new__(TalonCoordinatorFeature)
    feat.agent = SimpleNamespace(
        get_feature=lambda n: FakeCU() if n == "ComputerUseFeature" else None
    )

    async def fake_claim(repo, issue):
        return ToolResult.ok("d", data={"dispatched": True, "job_id": "j"})

    feat.talon_claim = fake_claim

    res = await feat.talon_file_and_claim(
        title="t", body="b",
        # mixed case + multiple reserved + a real one
        labels="agent-claimed, reliability, Agent-Complete,bug",
        repo="o/r",
    )
    cmd = seen["command"]
    # No Talon-reserved label reaches gh issue create...
    assert "agent-claimed" not in cmd.lower()
    assert "agent-complete" not in cmd.lower()
    # ...but real labels do.
    assert "--label reliability" in cmd and "--label bug" in cmd
    assert set(res.data["applied_labels"]) == {"reliability", "bug"}
    assert "agent-claimed" in [s.lower() for s in res.data["stripped_labels"]]
    assert "agent-complete" in [s.lower() for s in res.data["stripped_labels"]]
    assert res.data["dispatched"] is True


@pytest.mark.asyncio
async def test_talon_file_and_claim_retries_without_unknown_label():
    """A typo'd/nonexistent label must not sink the whole loop: gh
    create hard-fails on unknown labels, so file+claim retries with no
    labels, still closes, and reports the dropped label."""
    from kestrel_sdk.tools.result import ToolResult
    from kestrel_sovereign.features.talon.coordinator import (
        TalonCoordinatorFeature,
    )

    calls = []

    class FakeCU:
        async def shell(self, command, timeout=60):
            calls.append(command)
            if "--label reliability" in command:
                return ToolResult.failed(
                    "exit 1",
                    data={"returncode": 1, "stdout": "",
                          "stderr": "could not add label: 'reliability' "
                                    "not found"},
                )
            return ToolResult.ok(
                "created",
                data={"returncode": 0,
                      "stdout": "https://github.com/o/r/issues/99\n"},
            )

    feat = TalonCoordinatorFeature.__new__(TalonCoordinatorFeature)
    feat.agent = SimpleNamespace(
        get_feature=lambda n: FakeCU() if n == "ComputerUseFeature" else None
    )

    async def fake_claim(repo, issue):
        assert issue == 99
        return ToolResult.ok("d", data={"dispatched": True, "job_id": "j"})

    feat.talon_claim = fake_claim

    res = await feat.talon_file_and_claim(
        title="t", body="b", labels="reliability", repo="o/r",
    )
    assert len(calls) == 2
    assert "--label" not in calls[1]
    assert res.data["filed"] is True
    assert res.data["dispatched"] is True
    assert res.data["label_retry"] is True
    assert res.data["dropped_unknown_labels"] == ["reliability"]
    assert res.data["issue_number"] == 99


@pytest.mark.asyncio
async def test_talon_file_and_claim_does_not_retry_on_nonlabel_failure():
    """A non-label create failure (e.g. auth) must NOT be masked by the
    no-label retry — it returns failed with diagnostics."""
    from kestrel_sdk.tools.result import ToolResult
    from kestrel_sovereign.features.talon.coordinator import (
        TalonCoordinatorFeature,
    )

    calls = []

    class FakeCU:
        async def shell(self, command, timeout=60):
            calls.append(command)
            return ToolResult.failed(
                "exit 1",
                data={"returncode": 1, "stdout": "",
                      "stderr": "HTTP 401: Bad credentials"},
            )

    feat = TalonCoordinatorFeature.__new__(TalonCoordinatorFeature)
    feat.agent = SimpleNamespace(
        get_feature=lambda n: FakeCU() if n == "ComputerUseFeature" else None
    )
    res = await feat.talon_file_and_claim(
        title="t", body="b", labels="bug", repo="o/r",
    )
    assert len(calls) == 1
    assert res.data["filed"] is False
    assert res.data["label_retry"] is False


@pytest.mark.asyncio
async def test_talon_file_and_claim_refuses_without_computer_use():
    from kestrel_sovereign.features.talon.coordinator import (
        TalonCoordinatorFeature,
    )

    feat = TalonCoordinatorFeature.__new__(TalonCoordinatorFeature)
    feat.agent = SimpleNamespace(get_feature=lambda n: None)
    res = await feat.talon_file_and_claim(title="x", body="y")
    assert res.data["filed"] is False
    assert "ComputerUseFeature unavailable" in (res.error or "")
    # #1383: reason_code must be set even on the no-CU short-circuit
    # so the audit trail and the model's narration can carry it.
    assert res.data["reason_code"] == "MISSING_COMPUTER_USE"


# --------------------------------------------------------------------------
# #1383 - talon_file_and_claim discriminated reason codes + outcome audit
# --------------------------------------------------------------------------


def _file_and_claim_feat_with(shell_factory, claim_data=None, security=None):
    """Build a TalonCoordinatorFeature wired with a fake CU.shell and
    optionally a fake SecurityFeature for outcome-audit assertions.

    ``shell_factory`` is a callable receiving (command, timeout) that
    returns a ToolResult — used to simulate the various failure modes
    the classifier must distinguish (#1383)."""
    from types import SimpleNamespace

    from kestrel_sovereign.features.talon.coordinator import (
        TalonCoordinatorFeature,
    )

    feat = TalonCoordinatorFeature.__new__(TalonCoordinatorFeature)

    class FakeCU:
        async def shell(self, command, timeout=60):
            return await shell_factory(command, timeout)

    def _get_feature(name):
        if name == "ComputerUseFeature":
            return FakeCU()
        if name == "SecurityFeature":
            return security
        return None

    feat.agent = SimpleNamespace(get_feature=_get_feature)

    async def fake_claim(repo, issue):
        from kestrel_sdk.tools.result import ToolResult
        if claim_data is None:
            return ToolResult.failed("not dispatched", data={"dispatched": False})
        return ToolResult.ok("dispatched", data=claim_data)

    feat.talon_claim = fake_claim
    return feat


@pytest.mark.asyncio
async def test_file_and_claim_classifies_missing_gh_auth():
    """gh exiting with 'Bad credentials' must surface MISSING_GH_AUTH —
    not the historical catch-all "may have been denied at the approval
    gate, gh is not authenticated, or it failed for a non-label
    reason."""
    from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

    async def _shell(cmd, timeout):
        return ToolResult.partial(
            "ran but failed",
            "exit 1",
            data={
                "returncode": 1,
                "stdout": "",
                "stderr": "HTTP 401: Bad credentials",
            },
        )

    feat = _file_and_claim_feat_with(_shell)
    res = await feat.talon_file_and_claim(
        title="t", body="b", labels="bug", repo="o/r",
    )
    assert res.status is ToolResultStatus.ERROR
    assert res.data["reason_code"] == "MISSING_GH_AUTH"
    assert "MISSING_GH_AUTH" in (res.error or "")
    assert "GH_TOKEN" in (res.error or "") or "auth" in (res.error or "").lower()


@pytest.mark.asyncio
async def test_file_and_claim_classifies_repo_not_found():
    from kestrel_sdk.tools.result import ToolResult

    async def _shell(cmd, timeout):
        return ToolResult.partial(
            "ran but failed",
            "exit 1",
            data={
                "returncode": 1,
                "stdout": "",
                "stderr": (
                    "GraphQL: Could not resolve to a Repository with the "
                    "name 'o/r'."
                ),
            },
        )

    feat = _file_and_claim_feat_with(_shell)
    res = await feat.talon_file_and_claim(title="t", body="b", repo="o/r")
    assert res.data["reason_code"] == "REPO_NOT_FOUND"


@pytest.mark.asyncio
async def test_file_and_claim_classifies_shell_timeout():
    from kestrel_sdk.tools.result import ToolResult

    async def _shell(cmd, timeout):
        return ToolResult.partial(
            "timed out",
            "exit -1",
            data={
                "returncode": -1,
                "stdout": "",
                "stderr": "",
                "timed_out": True,
            },
        )

    feat = _file_and_claim_feat_with(_shell)
    res = await feat.talon_file_and_claim(title="t", body="b", repo="o/r")
    assert res.data["reason_code"] == "SHELL_TIMEOUT"
    assert res.data["shell_timed_out"] is True


@pytest.mark.asyncio
async def test_file_and_claim_classifies_gate_denied():
    from kestrel_sdk.tools.result import ToolResult

    async def _shell(cmd, timeout):
        # cu.shell returns failed (no data) when the approval gate
        # refuses the command.
        return ToolResult.failed(
            "approval-gate denied shell_execution_host: not allowlisted",
        )

    feat = _file_and_claim_feat_with(_shell)
    res = await feat.talon_file_and_claim(title="t", body="b", repo="o/r")
    assert res.data["reason_code"] == "GATE_DENIED"


@pytest.mark.asyncio
async def test_file_and_claim_url_parse_failed_when_gh_ok_but_no_url():
    """gh exited 0 but stdout was empty — URL_PARSE_FAILED, NOT a
    generic UNKNOWN_FAILURE."""
    from kestrel_sdk.tools.result import ToolResult

    async def _shell(cmd, timeout):
        return ToolResult.ok(
            "created",
            data={"returncode": 0, "stdout": "(nothing useful)\n"},
        )

    feat = _file_and_claim_feat_with(_shell)
    res = await feat.talon_file_and_claim(title="t", body="b", repo="o/r")
    assert res.data["reason_code"] == "URL_PARSE_FAILED"


@pytest.mark.asyncio
async def test_file_and_claim_writes_outcome_audit_row_on_failure():
    """#1383: the gate's pre-execution row logs the DECISION; the tool
    must ALSO append an OUTCOME row so audits show what happened, not
    just what was allowed."""
    from kestrel_sdk.tools.result import ToolResult
    from types import SimpleNamespace

    rows = []

    class FakeStore:
        async def log_decision(self, **kwargs):
            rows.append(kwargs)

    fake_security = SimpleNamespace(permission_store=FakeStore())

    async def _shell(cmd, timeout):
        return ToolResult.partial(
            "ran but failed", "exit 1",
            data={
                "returncode": 1, "stdout": "",
                "stderr": "HTTP 401: Bad credentials",
            },
        )

    feat = _file_and_claim_feat_with(_shell, security=fake_security)
    res = await feat.talon_file_and_claim(title="t", body="b", repo="o/r")
    assert res.data["reason_code"] == "MISSING_GH_AUTH"
    assert len(rows) == 1
    row = rows[0]
    assert row["action"] == "tool_outcome"
    assert row["tool_name"] == "talon_file_and_claim.outcome"
    assert row["feature_name"] == "talon_feature"
    # The summary carries enough for triage without parsing the data.
    summary = row["args_summary"]
    assert "MISSING_GH_AUTH" in summary
    assert "o/r" in summary


@pytest.mark.asyncio
async def test_file_and_claim_writes_outcome_audit_row_on_success():
    from kestrel_sdk.tools.result import ToolResult
    from types import SimpleNamespace

    rows = []

    class FakeStore:
        async def log_decision(self, **kwargs):
            rows.append(kwargs)

    fake_security = SimpleNamespace(permission_store=FakeStore())

    async def _shell(cmd, timeout):
        return ToolResult.ok(
            "created",
            data={
                "returncode": 0,
                "stdout": "https://github.com/o/r/issues/1234\n",
            },
        )

    feat = _file_and_claim_feat_with(
        _shell,
        claim_data={"dispatched": True, "job_id": "j-1"},
        security=fake_security,
    )
    res = await feat.talon_file_and_claim(title="t", body="b", repo="o/r")
    assert res.data["dispatched"] is True
    # Success path also writes an outcome row — the audit log is the
    # ledger of WHAT HAPPENED, not just of failures.
    assert len(rows) == 1
    assert rows[0]["decision"] == "filed_and_dispatched"
    assert "1234" in rows[0]["args_summary"]


@pytest.mark.asyncio
async def test_file_and_claim_outcome_audit_is_best_effort():
    """A failure inside the audit-row write must NOT crash the loop-
    closing primitive — the tool still returns its real result."""
    from kestrel_sdk.tools.result import ToolResult
    from types import SimpleNamespace

    class BrokenStore:
        async def log_decision(self, **kwargs):
            raise RuntimeError("disk full")

    fake_security = SimpleNamespace(permission_store=BrokenStore())

    async def _shell(cmd, timeout):
        return ToolResult.ok(
            "created",
            data={
                "returncode": 0,
                "stdout": "https://github.com/o/r/issues/42\n",
            },
        )

    feat = _file_and_claim_feat_with(
        _shell,
        claim_data={"dispatched": True, "job_id": "j"},
        security=fake_security,
    )
    res = await feat.talon_file_and_claim(title="t", body="b", repo="o/r")
    assert res.data["dispatched"] is True
    assert res.data["issue_number"] == 42
