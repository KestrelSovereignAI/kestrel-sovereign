"""Unit tests for the Emma closed-loop machinery (epic #1290).

Covers:
  D1 - scoped auto-approve matcher, two-phase audit, ApprovalQueue seam
  D3 - pre-turn state block gating + assembly + token cap
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


@pytest.mark.asyncio
async def test_pre_approval_read_error_fails_closed(store):
    """#2056: a permission-store read error on the pre-approval check must
    fail CLOSED. A tool that has a matching auto-approve rule must NOT be
    auto-approved when the store can't confirm the tool is not ALWAYS_ASK —
    it must be queued (here: ``no_approver`` on a headless test instance),
    never ``(True, "auto_approve:*")``."""
    agent = SimpleNamespace(
        _agent_name="Emma", did="did:pkh:emma", is_test_instance=True,
    )
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

    # The very first (pre-approval) read raises; everything else is healthy.
    real_get_permission = store.get_permission
    calls = {"n": 0}

    async def flaky_get_permission(feature_name, tool_name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("permission store read failed")
        return await real_get_permission(feature_name, tool_name)

    store.get_permission = flaky_get_permission

    approved, scope = await q.request_approval(
        "computer_use", "shell", args, timeout=1,
    )
    assert approved is False
    assert not scope.startswith("auto_approve:")
    assert scope == "no_approver"


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
