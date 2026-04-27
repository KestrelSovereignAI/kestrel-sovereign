"""Tests for ComputerUseFeature gate ordering and lifecycle (#838)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

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
    result = await feature.fs_read(path=str(tmp_path / "ok.txt"))
    assert result["success"] is False
    assert "not enabled" in result["error"]


@pytest.mark.asyncio
async def test_privacy_gate_blocks_when_flag_off(workspace: Path):
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=False),
        grants={"filesystem_read"},
        queue=FakeApprovalQueue(),
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    result = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result["success"] is False
    assert result["error"].startswith("privacy")


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
    result = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result["success"] is False
    assert result["error"].startswith("constitution")


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
    result = await feature.fs_read(path=str(workspace / "secret" / "leak.txt"))
    assert result["success"] is False
    assert result["error"].startswith("policy:deny")
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
    result = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result["success"] is True
    assert result["content"] == "hello"
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
    result = await feature.fs_write(path=str(workspace / "new.txt"), content="data")
    assert result["success"] is True
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
    result = await feature.fs_write(path=str(workspace / "new.txt"), content="data")
    assert result["success"] is False
    assert result["error"].startswith("approval")
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
    result = await feature.shell(command="rm -rf /tmp", timeout=5)
    assert result["success"] is False
    assert result["error"].startswith("policy:deny")
    assert queue.calls == []


@pytest.mark.asyncio
async def test_shell_allowed_runs(workspace: Path):
    queue = FakeApprovalQueue(decision=(True, "once"))
    agent = FakeAgent(
        privacy=PrivacyConfig(computer_access=True),
        grants={"shell_execution_sandboxed", "shell_execution_host"},
        queue=queue,
    )
    feature = await _make_feature(workspace, agent=agent)
    await feature.initialize()
    result = await feature.shell(command="echo hi", timeout=5)
    assert result["success"] is True
    assert result["returncode"] == 0
    assert "hi" in result["stdout"]
    assert len(queue.calls) == 1


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
    result = await feature.fs_read(path=str(workspace / "ok.txt"))
    assert result["success"] is False
    assert "backend failed" in result["error"]
