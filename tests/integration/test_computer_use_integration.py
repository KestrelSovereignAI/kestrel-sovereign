"""Real integration tests for ComputerUseFeature.

These tests boot a real ``SecurityFeature`` (with its real ``ApprovalQueue``
and ``PermissionStore``), wire it into a minimal agent, run
``ComputerUseFeature.initialize()`` end to end, and exercise every tool
against the real filesystem with a real ``LocalSandboxBackend``.

Approval requests are answered by a background task that watches
``approval_queue.pending_requests`` and calls ``submit_decision`` —
exactly what the SSE/UI path does in production. Nothing in the
ComputerUseFeature path is mocked.

Docker-backed cases are gated on ``KESTREL_TEST_DOCKER=1`` because CI
runners don't always have Docker, but the test exists and runs locally
when you set the env var.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio

from kestrel_sovereign.features.computer_use import ComputerUseFeature
from kestrel_sovereign.features.computer_use.path_safety import PathSafetyError
from kestrel_sovereign.features.security import SecurityFeature
from kestrel_sovereign.hooks import HooksManager
from kestrel_sovereign.privacy import PrivacyConfig


# =============================================================================
# Test harness
# =============================================================================


class _IntegrationAgent:
    """Minimal agent shell that satisfies the surfaces the feature expects.

    Uses real SecurityFeature with real ApprovalQueue & PermissionStore.
    """

    def __init__(self, *, storage_path: str, privacy_config: PrivacyConfig, grants: set[str]):
        self.storage_path = storage_path
        self.did = "did:test:computer-use-integration"
        self.hooks_manager = HooksManager()
        self.features: dict = {}
        self.privacy_config = privacy_config
        self.granted_capabilities = frozenset(grants)
        self._event_listeners: list = []

    def get_feature(self, name: str):
        if name == "security":
            return self.features.get("SecurityFeature")
        return self.features.get(name)

    async def emit_event(self, event_type: str, data: dict):
        for listener in self._event_listeners:
            await listener(event_type, data)


class _ApprovalResponder:
    """Watches the approval queue and submits decisions in the background.

    This mirrors what the API/UI does when the user clicks
    Approve/Deny — it calls ``ApprovalQueue.submit_decision``. No mocks.
    """

    def __init__(self, security_feature: SecurityFeature, *, decision: bool, scope: str = "once"):
        self._security = security_feature
        self._decision = decision
        self._scope = scope
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.responded_count = 0

    async def __aenter__(self) -> "_ApprovalResponder":
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, BaseException):
                pass

    async def _run(self) -> None:
        while not self._stop.is_set():
            for req in list(self._security.approval_queue.pending_requests):
                self._security.approval_queue.submit_decision(
                    req.id, self._decision, self._scope
                )
                self.responded_count += 1
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                continue


@pytest_asyncio.fixture
async def workspace(tmp_path: Path) -> AsyncIterator[Path]:
    """A real on-disk workspace with allow-listed and deny-listed contents."""
    (tmp_path / "ok.txt").write_text("hello world")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "leak.txt").write_text("ssh-keys-here")
    yield tmp_path


@pytest_asyncio.fixture
async def security_feature() -> AsyncIterator[SecurityFeature]:
    """Real SecurityFeature with a real on-disk PermissionStore."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    agent = _IntegrationAgent(
        storage_path=path,
        privacy_config=PrivacyConfig(),
        grants=set(),
    )
    sf = SecurityFeature(agent)
    await sf.initialize()
    agent.features["SecurityFeature"] = sf
    yield sf
    await sf.shutdown()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _config_for(workspace: Path, *, backend: str = "local") -> dict:
    return {
        "enabled": True,
        "backend": backend,
        "allowed_paths": [str(workspace)],
        "deny_paths": [str(workspace / "secret")],
        "allowed_binaries": ["echo", "true", "false"],
        "denied_binaries": ["rm", "sudo"],
        "auto_approve_read": True,
        "audit_log_path": str(workspace / "audit.jsonl"),
    }


async def _make_feature(
    *,
    workspace: Path,
    privacy: PrivacyConfig,
    grants: set[str],
    security_feature: SecurityFeature,
    backend: str = "local",
) -> tuple[ComputerUseFeature, _IntegrationAgent]:
    agent = _IntegrationAgent(
        storage_path=security_feature.permission_store.db_path
        if hasattr(security_feature.permission_store, "db_path")
        else "",
        privacy_config=privacy,
        grants=grants,
    )
    agent.features["SecurityFeature"] = security_feature
    # Re-point the security feature at this agent so its emit_event works.
    security_feature.agent = agent

    feature = ComputerUseFeature(agent)
    feature._cfg = _config_for(workspace, backend=backend)
    await feature.initialize()
    agent.features["ComputerUseFeature"] = feature
    return feature, agent


def _read_audit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# =============================================================================
# Privacy gate
# =============================================================================


@pytest.mark.asyncio
async def test_privacy_gate_refuses_when_flag_off(workspace: Path, security_feature: SecurityFeature):
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=False),  # <-- gate closed
        grants={"filesystem_read", "shell_execution_sandboxed", "shell_execution_host"},
        security_feature=security_feature,
    )
    result = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result["success"] is False
    assert result["error"].startswith("privacy")

    audit = _read_audit(workspace / "audit.jsonl")
    assert audit and audit[-1]["outcome"] == "denied"
    assert audit[-1]["allowed_by"] == ["denied:privacy"]


# =============================================================================
# Constitution gate
# =============================================================================


@pytest.mark.asyncio
async def test_constitution_gate_refuses_without_grant(workspace: Path, security_feature: SecurityFeature):
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        # local backend needs both shell grants to construct, but no fs grants
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        security_feature=security_feature,
    )
    result = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result["success"] is False
    assert result["error"].startswith("constitution")
    audit = _read_audit(workspace / "audit.jsonl")
    assert audit[-1]["allowed_by"] == ["privacy", "denied:constitution"]
    assert audit[-1]["outcome"] == "denied"


@pytest.mark.asyncio
async def test_local_backend_refuses_without_host_grant(workspace: Path, security_feature: SecurityFeature):
    """Two-grant model: sandboxed alone is not enough for the local backend."""
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed"},  # missing host grant
        security_feature=security_feature,
        backend="local",
    )
    # The backend refuses to construct → feature lands with backend=None
    assert feature._backend is None
    result = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result["success"] is False
    assert result["error"].startswith("readiness:")


# =============================================================================
# Approval gate via real queue + real responder
# =============================================================================


@pytest.mark.asyncio
async def test_fs_write_requires_real_approval(workspace: Path, security_feature: SecurityFeature):
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "filesystem_write",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )
    target = workspace / "new.txt"

    async with _ApprovalResponder(security_feature, decision=True, scope="once") as responder:
        result = await feature.fs_write(path=str(target), content="payload")

    assert result["success"] is True, result
    assert target.read_text() == "payload"
    assert responder.responded_count == 1

    audit = _read_audit(workspace / "audit.jsonl")
    last = audit[-1]
    assert last["tool"] == "fs-write"
    assert last["outcome"] == "ok"
    assert "privacy" in last["allowed_by"]
    assert "constitution" in last["allowed_by"]
    assert any(s.startswith("approval:") for s in last["allowed_by"])


@pytest.mark.asyncio
async def test_fs_write_denied_when_user_refuses(workspace: Path, security_feature: SecurityFeature):
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "filesystem_write",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )
    target = workspace / "rejected.txt"

    async with _ApprovalResponder(security_feature, decision=False, scope="once"):
        result = await feature.fs_write(path=str(target), content="should not land")

    assert result["success"] is False
    assert result["error"].startswith("approval")
    assert not target.exists()


# =============================================================================
# Reads inside / outside the allow-list
# =============================================================================


@pytest.mark.asyncio
async def test_fs_read_inside_allow_list_auto_approves(
    workspace: Path, security_feature: SecurityFeature
):
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )
    # No approval responder running — auto-approve must be enough.
    result = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result["success"] is True
    assert result["content"] == "hello world"
    assert security_feature.approval_queue.pending_count == 0


@pytest.mark.asyncio
async def test_deny_path_short_circuits_before_approval(
    workspace: Path, security_feature: SecurityFeature
):
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )
    # Even with an "approve everything" responder running, a deny-list
    # match must hard-reject before reaching the queue.
    async with _ApprovalResponder(security_feature, decision=True) as responder:
        result = await feature.fs_read(path=str(workspace / "secret" / "leak.txt"))

    assert result["success"] is False
    assert result["error"].startswith("policy:deny")
    assert responder.responded_count == 0


# =============================================================================
# Shell exec
# =============================================================================


@pytest.mark.asyncio
async def test_shell_allowed_binary_runs_with_approval(
    workspace: Path, security_feature: SecurityFeature
):
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        security_feature=security_feature,
    )
    async with _ApprovalResponder(security_feature, decision=True):
        result = await feature.shell(command="echo integration-ok", timeout=5)

    assert result["success"] is True
    assert result["returncode"] == 0
    assert "integration-ok" in result["stdout"]


@pytest.mark.asyncio
async def test_shell_denied_binary_hard_rejects(
    workspace: Path, security_feature: SecurityFeature
):
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        security_feature=security_feature,
    )
    async with _ApprovalResponder(security_feature, decision=True) as responder:
        result = await feature.shell(command="rm -rf /tmp/whatever", timeout=5)

    assert result["success"] is False
    assert result["error"].startswith("policy:deny")
    assert responder.responded_count == 0


# =============================================================================
# Path safety against the real filesystem
# =============================================================================


@pytest.mark.asyncio
async def test_symlink_resolves_to_realpath_for_human_approver(
    workspace: Path, security_feature: SecurityFeature
):
    """A symlink resolves to the realpath BEFORE the human sees it.

    The real defense against symlink-escape attacks is that the human
    approver is shown the resolved realpath, not the symlink path. If
    the realpath is outside the allow-list (and not on the deny-list),
    policy returns REQUIRE_APPROVAL and the human can refuse — and the
    approval payload contains the *real* target.
    """
    target_outside = workspace.parent / "outside_target_for_symlink_test"
    target_outside.mkdir(exist_ok=True)
    secret = target_outside / "secret_in_outside.txt"
    secret.write_text("escaped")
    link = workspace / "trap_link"
    if not link.exists():
        os.symlink(target_outside, link)

    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )

    # Capture what the approval queue sees, then deny.
    captured_args: dict[str, object] = {}

    class _CapturingResponder(_ApprovalResponder):
        async def _run(self) -> None:
            while not self._stop.is_set():
                for req in list(self._security.approval_queue.pending_requests):
                    captured_args.update(req.tool_args)
                    self._security.approval_queue.submit_decision(
                        req.id, self._decision, self._scope
                    )
                    self.responded_count += 1
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue

    async with _CapturingResponder(security_feature, decision=False):
        result = await feature.fs_read(
            path=str(workspace / "trap_link" / "secret_in_outside.txt")
        )

    assert result["success"] is False
    assert result["error"].startswith("approval")
    # The human approver was shown the resolved realpath, not the symlink path.
    real_target = os.path.realpath(secret)
    assert captured_args.get("path") == real_target

    # Audit row reflects the same: privacy/constitution/path_safety/policy passed,
    # approval denied. Both the resolved path and the original raw path are recorded.
    audit = _read_audit(workspace / "audit.jsonl")
    last = audit[-1]
    assert last["outcome"] == "denied"
    assert any(s.startswith("denied:approval") for s in last["allowed_by"])
    assert "privacy" in last["allowed_by"]
    assert "constitution" in last["allowed_by"]
    assert "path_safety" in last["allowed_by"]
    assert last["args"]["path"] == real_target


# =============================================================================
# Audit log durability
# =============================================================================


@pytest.mark.asyncio
async def test_audit_log_records_full_chain(workspace: Path, security_feature: SecurityFeature):
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "filesystem_write",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )
    async with _ApprovalResponder(security_feature, decision=True):
        await feature.fs_read(path=str(workspace / "ok.txt"))  # auto-approved
        await feature.fs_write(path=str(workspace / "n1.txt"), content="a")  # human approval
        await feature.shell(command="echo done", timeout=5)  # human approval

    audit = _read_audit(workspace / "audit.jsonl")
    assert len(audit) == 3
    assert [r["tool"] for r in audit] == ["fs-read", "fs-write", "shell"]
    # Read auto-approves -> chain has privacy/constitution/path_safety/policy
    assert audit[0]["allowed_by"] == ["privacy", "constitution", "path_safety", "policy"]
    # Writes/shell go all the way through the queue
    assert any(s.startswith("approval:") for s in audit[1]["allowed_by"])
    assert any(s.startswith("approval:") for s in audit[2]["allowed_by"])


# =============================================================================
# Constitution parser end-to-end (reads the real on-disk markdown)
# =============================================================================


@pytest.mark.asyncio
async def test_constitution_parser_picks_up_grants_from_disk(
    workspace: Path, security_feature: SecurityFeature, monkeypatch: pytest.MonkeyPatch
):
    """Write a tmp constitution with [x] grants, point the feature at it, verify."""
    tmp_const = workspace / "TEST_CONSTITUTION.md"
    tmp_const.write_text(
        "# Test\n\n"
        "### Amendment IX: Capability Boundaries\n\n"
        "- [x] filesystem_read\n"
        "- [ ] filesystem_write\n"
        "- [x] shell_execution_sandboxed\n"
        "- [x] shell_execution_host\n"
    )
    # Force the feature's disk-loader to read our test constitution.
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(tmp_const)
    )

    agent = _IntegrationAgent(
        storage_path="",
        privacy_config=PrivacyConfig(computer_access=True),
        grants=set(),  # NOT pre-populated — the feature must parse them off disk
    )
    # Strip the agent attribute so the feature falls through to disk parse.
    delattr(agent, "granted_capabilities")
    agent.features["SecurityFeature"] = security_feature
    security_feature.agent = agent

    feature = ComputerUseFeature(agent)
    feature._cfg = _config_for(workspace, backend="local")
    await feature.initialize()

    # Read should succeed (filesystem_read granted on disk)
    result_read = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result_read["success"] is True

    # Write should fail at constitution gate (filesystem_write NOT granted)
    async with _ApprovalResponder(security_feature, decision=True):
        result_write = await feature.fs_write(path=str(workspace / "blocked.txt"), content="x")
    assert result_write["success"] is False
    assert result_write["error"].startswith("constitution")
    assert not (workspace / "blocked.txt").exists()


# =============================================================================
# Reads outside the allow-list go through human approval (the case that was
# impossible under the v0 contract because path_safety rejected first)
# =============================================================================


@pytest.mark.asyncio
async def test_read_outside_allow_list_goes_through_approval_and_succeeds(
    workspace: Path, security_feature: SecurityFeature
):
    """Path outside the allow-list reaches the approval queue, not path_safety."""
    elsewhere = tempfile.mkdtemp(prefix="kestrel_outside_allowlist_")
    target = Path(elsewhere) / "interesting.txt"
    target.write_text("worth a look")

    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )

    async with _ApprovalResponder(security_feature, decision=True) as responder:
        result = await feature.fs_read(path=str(target))

    assert result["success"] is True, result
    assert result["content"] == "worth a look"
    assert responder.responded_count == 1, "must reach the approval queue"

    audit = _read_audit(workspace / "audit.jsonl")
    last = audit[-1]
    assert last["outcome"] == "ok"
    assert "policy" in last["allowed_by"]
    assert any(s.startswith("approval:") for s in last["allowed_by"])


@pytest.mark.asyncio
async def test_read_outside_allow_list_denied_when_user_refuses(
    workspace: Path, security_feature: SecurityFeature
):
    elsewhere = tempfile.mkdtemp(prefix="kestrel_outside_allowlist_deny_")
    target = Path(elsewhere) / "x.txt"
    target.write_text("nope")

    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )

    async with _ApprovalResponder(security_feature, decision=False):
        result = await feature.fs_read(path=str(target))

    assert result["success"] is False
    assert result["error"].startswith("approval")


# =============================================================================
# Gate ordering: constitution refuses BEFORE path-safety/policy can leak info
# =============================================================================


@pytest.mark.asyncio
async def test_constitution_denial_does_not_leak_path_or_policy_info(
    workspace: Path, security_feature: SecurityFeature
):
    """A path that would have failed path_safety should still surface as a
    constitution refusal — the agent must not learn anything about path
    structure or policy contents until the call is constitutionally eligible.
    """
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        # NO filesystem_read grant; shell grants present so backend constructs.
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        security_feature=security_feature,
    )
    # Pass a path with `..` traversal — under the wrong gate order, this would
    # surface as path_safety:traversal. Under the correct order, the agent
    # only learns it has no constitutional grant.
    result = await feature.fs_read(path="../../../etc/passwd")
    assert result["success"] is False
    assert result["error"].startswith("constitution"), result["error"]

    audit = _read_audit(workspace / "audit.jsonl")
    last = audit[-1]
    assert last["allowed_by"] == ["privacy", "denied:constitution"]


# =============================================================================
# Audit on every refusal path (the v0 implementation skipped these)
# =============================================================================


@pytest.mark.asyncio
async def test_audit_records_path_safety_traversal_refusal(
    workspace: Path, security_feature: SecurityFeature
):
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )
    result = await feature.fs_read(path="../../etc/passwd")
    assert result["success"] is False
    assert result["error"].startswith("path_safety")

    audit = _read_audit(workspace / "audit.jsonl")
    last = audit[-1]
    assert last["outcome"] == "denied"
    assert last["allowed_by"] == ["privacy", "constitution", "denied:path_safety"]
    assert last["error"] is not None
    assert "traversal" in last["error"]


@pytest.mark.asyncio
async def test_audit_records_policy_deny_refusal(
    workspace: Path, security_feature: SecurityFeature
):
    """Deny-list match must produce an audit row with denied:policy."""
    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )
    # Even with a permissive responder, the deny-list must short-circuit.
    async with _ApprovalResponder(security_feature, decision=True) as responder:
        result = await feature.fs_read(path=str(workspace / "secret" / "leak.txt"))

    assert result["success"] is False
    assert result["error"].startswith("policy:deny")
    assert responder.responded_count == 0

    audit = _read_audit(workspace / "audit.jsonl")
    last = audit[-1]
    assert last["outcome"] == "denied"
    assert last["allowed_by"] == [
        "privacy",
        "constitution",
        "path_safety",
        "denied:policy",
    ]


@pytest.mark.asyncio
async def test_audit_records_disabled_feature_call(
    workspace: Path, security_feature: SecurityFeature
):
    """Calling a tool while disabled in toml still produces an audit row."""
    agent = _IntegrationAgent(
        storage_path=security_feature.permission_store.db_path
        if hasattr(security_feature.permission_store, "db_path")
        else "",
        privacy_config=PrivacyConfig(computer_access=True),
        grants={"filesystem_read"},
    )
    agent.features["SecurityFeature"] = security_feature
    security_feature.agent = agent

    feature = ComputerUseFeature(agent)
    cfg = _config_for(workspace)
    cfg["enabled"] = False
    feature._cfg = cfg
    await feature.initialize()

    result = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result["success"] is False
    assert result["error"].startswith("readiness:")

    audit = _read_audit(workspace / "audit.jsonl")
    assert audit
    last = audit[-1]
    assert last["outcome"] == "denied"
    assert any("readiness" in s for s in last["allowed_by"])


# =============================================================================
# fs-edit
# =============================================================================


@pytest.mark.asyncio
async def test_fs_edit_replaces_single_occurrence(
    workspace: Path, security_feature: SecurityFeature
):
    target = workspace / "edit.txt"
    target.write_text("alpha beta gamma beta")

    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "filesystem_write",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )

    async with _ApprovalResponder(security_feature, decision=True):
        result = await feature.fs_edit(
            path=str(target), old_text="beta", new_text="BETA", occurrence=1
        )

    assert result["success"] is True, result
    # Only the FIRST occurrence is replaced.
    assert target.read_text() == "alpha BETA gamma beta"


@pytest.mark.asyncio
async def test_fs_edit_can_target_nth_occurrence(
    workspace: Path, security_feature: SecurityFeature
):
    target = workspace / "edit2.txt"
    target.write_text("x x x")

    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "filesystem_write",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )

    async with _ApprovalResponder(security_feature, decision=True):
        result = await feature.fs_edit(
            path=str(target), old_text="x", new_text="Y", occurrence=2
        )

    assert result["success"] is True
    assert target.read_text() == "x Y x"


@pytest.mark.asyncio
async def test_fs_edit_rejects_when_old_text_missing(
    workspace: Path, security_feature: SecurityFeature
):
    target = workspace / "edit3.txt"
    target.write_text("hello world")

    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "filesystem_write",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )

    result = await feature.fs_edit(
        path=str(target), old_text="not present", new_text="X"
    )
    assert result["success"] is False
    assert "not found" in result["error"]
    assert target.read_text() == "hello world"  # unchanged


@pytest.mark.asyncio
async def test_fs_edit_denied_when_user_refuses(
    workspace: Path, security_feature: SecurityFeature
):
    target = workspace / "edit4.txt"
    target.write_text("original")

    feature, _ = await _make_feature(
        workspace=workspace,
        privacy=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "filesystem_write",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
        security_feature=security_feature,
    )

    async with _ApprovalResponder(security_feature, decision=False):
        result = await feature.fs_edit(
            path=str(target), old_text="original", new_text="HIJACKED"
        )

    assert result["success"] is False
    assert result["error"].startswith("approval")
    assert target.read_text() == "original"


# =============================================================================
# kestrel.toml is actually loaded from disk
# =============================================================================


@pytest.mark.asyncio
async def test_kestrel_toml_loader_is_consulted(
    workspace: Path, security_feature: SecurityFeature, tmp_path: Path
):
    """Drop a kestrel.toml in the agent's storage_path and verify the
    feature reads its [features.computer_use] section."""
    storage_dir = tmp_path / "agent_data" / "test-agent"
    storage_dir.mkdir(parents=True)
    audit_path = storage_dir / "audit.jsonl"
    toml_path = storage_dir / "kestrel.toml"
    toml_path.write_text(
        f"""
[features.computer_use]
enabled = true
backend = "local"
allowed_paths = ["{workspace}"]
deny_paths = ["{workspace}/secret"]
allowed_binaries = ["echo"]
denied_binaries = ["rm"]
auto_approve_read = true
audit_log_path = "{audit_path}"
"""
    )

    agent = _IntegrationAgent(
        storage_path=str(storage_dir),
        privacy_config=PrivacyConfig(computer_access=True),
        grants={
            "filesystem_read",
            "shell_execution_sandboxed",
            "shell_execution_host",
        },
    )
    agent.features["SecurityFeature"] = security_feature
    security_feature.agent = agent

    feature = ComputerUseFeature(agent)
    # Don't pre-populate _cfg — force the loader to find the toml.
    await feature.initialize()

    assert feature._backend is not None, "loader must have found enabled toml"
    result = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result["success"] is True
    assert result["content"] == "hello world"

    # Audit landed at the configured path
    assert audit_path.exists()
    rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert rows and rows[-1]["tool"] == "fs-read"
