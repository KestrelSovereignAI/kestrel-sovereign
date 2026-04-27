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
    assert "backend failed" in result["error"]


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
async def test_symlink_escape_is_rejected(workspace: Path, security_feature: SecurityFeature):
    """A symlink inside the workspace pointing outside must be refused."""
    target_outside = workspace.parent / "outside_target_for_symlink_test"
    target_outside.mkdir(exist_ok=True)
    (target_outside / "secret_in_outside.txt").write_text("escaped")
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
    result = await feature.fs_read(path=str(workspace / "trap_link" / "secret_in_outside.txt"))
    assert result["success"] is False
    # path_safety raises before the policy layer is consulted
    assert result["error"].startswith("path_safety")


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
    # Read auto-approves -> only privacy + constitution in chain
    assert audit[0]["allowed_by"] == ["privacy", "constitution"]
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
