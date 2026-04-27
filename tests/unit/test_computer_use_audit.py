"""Tests for the JSONL audit log (#836)."""

import asyncio
import json
from pathlib import Path

import pytest

from kestrel_sovereign.features.computer_use.audit import AuditLog, AuditRecord


@pytest.mark.asyncio
async def test_writes_one_record(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl")
    await log.write(
        AuditRecord(
            tool="fs-read", backend="local", args={"path": "/x"}, allowed_by=["privacy"]
        )
    )
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["tool"] == "fs-read"
    assert parsed["backend"] == "local"
    assert parsed["allowed_by"] == ["privacy"]


@pytest.mark.asyncio
async def test_records_in_order(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(20):
        await log.write(
            AuditRecord(tool=f"fs-read", backend="local", args={"i": i}, allowed_by=["x"])
        )
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 20
    for i, line in enumerate(lines):
        assert json.loads(line)["args"]["i"] == i


@pytest.mark.asyncio
async def test_concurrent_writes_serialized(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl")

    async def writer(label: int) -> None:
        for j in range(10):
            await log.write(
                AuditRecord(
                    tool="t", backend="local", args={"w": label, "j": j}, allowed_by=["x"]
                )
            )

    await asyncio.gather(writer(0), writer(1), writer(2))
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    assert len(lines) == 30
    # Every line is valid JSON — no torn writes
    for line in lines:
        parsed = json.loads(line)
        assert parsed["tool"] == "t"


@pytest.mark.asyncio
async def test_creates_parent_dir(tmp_path: Path):
    log = AuditLog(tmp_path / "nested" / "deeper" / "audit.jsonl")
    await log.write(
        AuditRecord(tool="t", backend="local", args={}, allowed_by=["x"])
    )
    assert (tmp_path / "nested" / "deeper" / "audit.jsonl").exists()


@pytest.mark.asyncio
async def test_forwards_to_feedback_hook(tmp_path: Path):
    captured: list[tuple[str, str, dict]] = []

    class FakeAgent:
        async def record_tool_usage(self, feature, tool, payload):
            captured.append((feature, tool, payload))

    log = AuditLog(tmp_path / "audit.jsonl", agent=FakeAgent())
    await log.write(AuditRecord(tool="fs-write", backend="docker", args={}, allowed_by=["x"]))
    assert captured and captured[0][0] == "computer_use"
    assert captured[0][1] == "fs-write"


@pytest.mark.asyncio
async def test_record_includes_outcome_and_error(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl")
    await log.write(
        AuditRecord(
            tool="shell",
            backend="docker",
            args={"argv": ["git", "status"]},
            allowed_by=["privacy", "constitution", "approval:once:human"],
            outcome="ok",
            duration_ms=42,
        )
    )
    parsed = json.loads((tmp_path / "audit.jsonl").read_text().strip())
    assert parsed["outcome"] == "ok"
    assert parsed["duration_ms"] == 42
    assert parsed["allowed_by"] == ["privacy", "constitution", "approval:once:human"]
