"""Tests for the Talon test-evidence verification layer (#1542)."""

import asyncio
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.talon.verification import (
    CIStatus,
    CommandExecution,
    TalonVerifier,
    TestCommandResult,
    VerificationEvidence,
    VerificationState,
    classify_denial,
    classify_execution,
    is_allowlisted,
)


class TestAllowlist:
    def test_exact_match(self):
        assert is_allowlisted("uv run pytest")

    def test_prefix_with_args(self):
        assert is_allowlisted("uv run pytest tests/unit -q")

    def test_whitespace_normalized(self):
        assert is_allowlisted("uv  run   pytest    tests/unit")

    def test_non_test_command_rejected(self):
        assert not is_allowlisted("uv run python deploy.py")

    def test_substring_is_not_a_match(self):
        # Must match a whole prefix token boundary, not a substring.
        assert not is_allowlisted("echo uv run pytest")

    def test_playwright_allowed(self):
        assert is_allowlisted("npx playwright test e2e/")


class TestClassifyExecution:
    def test_passed(self):
        ex = CommandExecution(ran=True, returncode=0, stdout="ok")
        r = classify_execution("uv run pytest", ex, allowlisted=True)
        assert r.state is VerificationState.PASSED
        assert r.exit_code == 0
        assert r.is_pass

    def test_failed(self):
        ex = CommandExecution(ran=True, returncode=1, stderr="boom")
        r = classify_execution("uv run pytest", ex, allowlisted=True)
        assert r.state is VerificationState.FAILED
        assert r.exit_code == 1
        assert not r.is_pass

    def test_tooling_error_when_not_run(self):
        ex = CommandExecution(ran=False, error="command not found: pytest")
        r = classify_execution("pytest", ex, allowlisted=True)
        assert r.state is VerificationState.TOOLING_ERROR
        assert "not found" in r.summary

    def test_sandbox_denied(self):
        ex = CommandExecution(ran=False, sandbox_denied=True, error="sandbox refused")
        r = classify_execution("pytest", ex, allowlisted=True)
        assert r.state is VerificationState.BLOCKED_BY_SANDBOX

    def test_tail_truncation(self):
        big = "x" * 10_000
        ex = CommandExecution(ran=True, returncode=0, stdout=big)
        r = classify_execution("uv run pytest", ex, allowlisted=True)
        assert len(r.stdout_tail) < len(big)


class TestClassifyDenial:
    """The #1542 attribution rule lives here."""

    @pytest.mark.parametrize("scope", ["once", "session", "always"])
    def test_user_denial_only_for_explicit_user_scope(self, scope):
        r = classify_denial("rm -rf /", scope)
        assert r.state is VerificationState.BLOCKED_BY_USER
        assert "user" in r.summary.lower()

    def test_user_denied_scope_is_user_denial(self):
        # The real ApprovalQueue contract: an explicit deny via the deny
        # tool / !security-deny returns scope "user_denied" (#1542).
        r = classify_denial("rm -rf /", "user_denied")
        assert r.state is VerificationState.BLOCKED_BY_USER
        assert "user" in r.summary.lower()

    def test_policy_deny_is_not_user_denial(self):
        r = classify_denial("rm -rf /", "denied")
        assert r.state is VerificationState.BLOCKED_BY_POLICY
        assert "not a user denial" in r.summary

    def test_timeout_is_not_user_denial(self):
        r = classify_denial("rm -rf /", "timeout")
        assert r.state is VerificationState.BLOCKED_BY_POLICY
        assert "not a user denial" in r.summary

    @pytest.mark.parametrize("scope", ["cancelled", "cancelled_all"])
    def test_cancel_is_not_user_denial(self, scope):
        r = classify_denial("rm -rf /", scope)
        assert r.state is VerificationState.BLOCKED_BY_POLICY
        assert "not a user denial" in r.summary

    def test_unknown_scope_defaults_to_policy(self):
        r = classify_denial("rm -rf /", "")
        assert r.state is VerificationState.BLOCKED_BY_POLICY


async def _exec_ok(command, *, timeout=600):
    return CommandExecution(ran=True, returncode=0, stdout="passed")


async def _exec_fail(command, *, timeout=600):
    return CommandExecution(ran=True, returncode=2, stderr="failed")


class TestVerifierFlow:
    @pytest.mark.asyncio
    async def test_allowlisted_runs_without_approval(self):
        approve_called = False

        async def approve(cmd):
            nonlocal approve_called
            approve_called = True
            return (True, "once")

        v = TalonVerifier(execute=_exec_ok, approve=approve)
        r = await v.verify_command("uv run pytest tests/unit")
        assert r.state is VerificationState.PASSED
        assert approve_called is False  # allowlisted skips the gate

    @pytest.mark.asyncio
    async def test_non_allowlisted_requires_approval_and_runs_when_approved(self):
        async def approve(cmd):
            return (True, "once")

        v = TalonVerifier(execute=_exec_ok, approve=approve)
        r = await v.verify_command("make custom-tests")
        assert r.state is VerificationState.PASSED
        assert r.allowlisted is False

    @pytest.mark.asyncio
    async def test_non_allowlisted_user_denied(self):
        async def approve(cmd):
            return (False, "once")  # user submitted a denial

        v = TalonVerifier(execute=_exec_ok, approve=approve)
        r = await v.verify_command("make custom-tests")
        assert r.state is VerificationState.BLOCKED_BY_USER

    @pytest.mark.asyncio
    async def test_non_allowlisted_policy_denied(self):
        async def approve(cmd):
            return (False, "denied")  # operator/auto policy DENY

        v = TalonVerifier(execute=_exec_ok, approve=approve)
        r = await v.verify_command("make custom-tests")
        assert r.state is VerificationState.BLOCKED_BY_POLICY

    @pytest.mark.asyncio
    async def test_non_allowlisted_no_approver_fails_closed(self):
        v = TalonVerifier(execute=_exec_ok, approve=None)
        r = await v.verify_command("make custom-tests")
        assert r.state is VerificationState.BLOCKED_BY_POLICY
        assert "fail-closed" in r.summary

    @pytest.mark.asyncio
    async def test_approver_returns_none_fails_closed(self):
        async def approve(cmd):
            return None  # approval mechanism unavailable

        v = TalonVerifier(execute=_exec_ok, approve=approve)
        r = await v.verify_command("make custom-tests")
        assert r.state is VerificationState.BLOCKED_BY_POLICY

    @pytest.mark.asyncio
    async def test_empty_command_is_tooling_error(self):
        v = TalonVerifier(execute=_exec_ok)
        r = await v.verify_command("   ")
        assert r.state is VerificationState.TOOLING_ERROR


class TestEvidenceAggregation:
    @pytest.mark.asyncio
    async def test_overall_state_failed_dominates(self):
        async def execute(cmd, *, timeout=600):
            if "fail" in cmd:
                return CommandExecution(ran=True, returncode=1)
            return CommandExecution(ran=True, returncode=0)

        v = TalonVerifier(execute=execute)
        ev = await v.verify_commands(
            ["uv run pytest pass", "uv run pytest fail"]
        )
        assert ev.overall_state is VerificationState.FAILED
        assert ev.all_passed is False

    @pytest.mark.asyncio
    async def test_all_passed(self):
        v = TalonVerifier(execute=_exec_ok)
        ev = await v.verify_commands(["uv run pytest a", "uv run pytest b"])
        assert ev.overall_state is VerificationState.PASSED
        assert ev.all_passed is True

    def test_empty_evidence_is_not_run(self):
        ev = VerificationEvidence()
        assert ev.overall_state is VerificationState.NOT_RUN
        assert ev.all_passed is False

    def test_to_dict_round_trips_states(self):
        ev = VerificationEvidence(
            results=[
                TestCommandResult(
                    command="uv run pytest",
                    state=VerificationState.PASSED,
                    exit_code=0,
                )
            ],
            ci_status=CIStatus(state="passed", url="https://ci/123"),
            note="local ok",
        )
        d = ev.to_dict()
        assert d["overall_state"] == "passed"
        assert d["results"][0]["state"] == "passed"
        assert d["ci_status"]["state"] == "passed"
        assert d["note"] == "local ok"

    def test_markdown_includes_evidence_and_ci(self):
        ev = VerificationEvidence(
            results=[
                TestCommandResult(
                    command="uv run pytest tests/unit",
                    state=VerificationState.FAILED,
                    exit_code=1,
                    summary="exited 1",
                )
            ],
            ci_status=CIStatus(
                state="failed",
                summary="3 checks failed",
                url="https://ci/run/1",
                checks=({"name": "lint", "conclusion": "failure"},),
            ),
            note="CI is the remaining hard gate.",
        )
        md = ev.to_markdown()
        assert "## Test Evidence" in md
        assert "failed" in md
        assert "uv run pytest tests/unit" in md
        assert "CI" in md
        assert "https://ci/run/1" in md
        assert "remaining hard gate" in md

    def test_ci_status_from_mapping(self):
        ci = CIStatus.from_mapping(
            {"state": "passed", "checks": [{"name": "test", "conclusion": "success"}]}
        )
        assert ci is not None
        assert ci.state == "passed"
        assert ci.checks[0]["name"] == "test"
        assert CIStatus.from_mapping(None) is None


def _queue_approver(queue):
    """Approver that drives the *real* ApprovalQueue, mirroring the
    coordinator's ``_make_verify_approver`` adapter (returns the queue's
    own ``(approved, scope)`` tuple)."""

    async def _approve(command):
        approved, scope = await queue.request_approval(
            feature_name="talon",
            tool_name="verify_command",
            tool_args={"command": command},
        )
        return bool(approved), str(scope)

    return _approve


class TestRealApprovalQueueProvenance:
    """Regression: drive the real ApprovalQueue / SecurityFeature deny
    paths, not mocked approver tuples (#1542 review follow-up).

    Both an explicit user denial and an operator/auto policy DENY resolve
    through ``ApprovalQueue.request_approval`` — historically *both* as
    ``(False, "denied")``, which made ``blocked_by_user`` unreachable for
    the real UI deny path. These tests pin the corrected provenance.
    """

    @pytest.mark.asyncio
    async def test_user_deny_via_security_feature_is_blocked_by_user(self):
        from kestrel_sovereign.features.security.approval_queue import (
            ApprovalQueue,
        )
        from kestrel_sovereign.features.security.feature import SecurityFeature

        queue = ApprovalQueue()
        feature = SecurityFeature(MagicMock())
        feature.approval_queue = queue

        verifier = TalonVerifier(execute=_exec_ok, approve=_queue_approver(queue))
        verify_task = asyncio.create_task(
            verifier.verify_command("make custom-tests")
        )

        # Wait for the approval request to be queued, then deny it exactly
        # the way the UI / !security-deny tool does.
        for _ in range(400):
            if queue.pending_count == 1:
                break
            await asyncio.sleep(0.005)
        assert queue.pending_count == 1, "approval request was never queued"

        pending = queue.pending_requests[0]
        deny_result = await feature.deny_request(pending.id)
        assert deny_result.data["decision"] == "user_denied"

        result = await verify_task
        assert result.state is VerificationState.BLOCKED_BY_USER

    @pytest.mark.asyncio
    async def test_operator_policy_deny_is_blocked_by_policy(self, tmp_path):
        from kestrel_sovereign.features.security.approval_queue import (
            ApprovalQueue,
        )
        from kestrel_sovereign.features.security.permissions import (
            PermissionLevel,
            PermissionStore,
        )

        store = PermissionStore(str(tmp_path / "perms.db"))
        await store.initialize()
        await store.register_tool(
            "talon", "verify_command", PermissionLevel.DENY
        )

        queue = ApprovalQueue(permission_store=store)
        verifier = TalonVerifier(execute=_exec_ok, approve=_queue_approver(queue))

        result = await verifier.verify_command("make custom-tests")
        assert result.state is VerificationState.BLOCKED_BY_POLICY
        assert "not a user denial" in result.summary
        # The operator DENY must never be reported as a queued user prompt.
        assert queue.pending_count == 0
