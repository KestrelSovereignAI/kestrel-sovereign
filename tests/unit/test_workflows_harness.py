"""Tests for the reusable WorkflowHarness fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel_sdk.signals import SignalMode

from kestrel_sovereign.features.workflows import RunStatus, Stage, WorkflowSpec
from tests.fixtures.workflow_harness import (
    CASSETTE_KEY_ENV,
    CASSETTE_SECRET_STORE_ENV,
    EncryptedWorkflowCassetteStore,
    WorkflowCassetteError,
    WorkflowHarness,
    assert_no_plaintext_workflow_cassettes,
)


@pytest.mark.asyncio
async def test_workflow_harness_runs_signed_action_workflow(tmp_path):
    calls: list[dict] = []

    async def handler(payload):
        calls.append(payload)
        return {"ok": True}

    async with WorkflowHarness(tmp_path) as harness:
        harness.register_action("ci.lint", handler)
        await harness.put_signed(
            WorkflowSpec(
                name="release",
                version=1,
                stages=[
                    Stage(
                        name="lint",
                        signal_source="ci.lint",
                        signal_mode=SignalMode.ACTION,
                        read_only=True,
                    )
                ],
            )
        )

        result = await harness.runner.run_to_completion(name="release")

        assert result.status == RunStatus.COMPLETED
        assert calls == [{}]
        links = await harness.store.list_stage_links(result.run_id)
        assert links[0].gate_outcome.value == "pass"


def test_workflow_cassette_store_encrypts_payload_and_binds_owner(tmp_path):
    store = EncryptedWorkflowCassetteStore(
        tmp_path / ".workflow-cassettes",
        key="test cassette root key",
        retention_seconds=60,
    )
    payload = {
        "request": {
            "pr_diff": "SECRET_PR_DIFF",
            "canary": "a" * 64,
        },
        "response": {
            "blockers": [
                {
                    "severity": "high",
                    "rationale": "SECRET_BREAK_THIS_PR",
                }
            ]
        },
    }

    path = store.record(
        owner_did="did:web:owner.example",
        cassette_id="red-team/review-1",
        payload=payload,
        now=100,
    )

    on_disk = path.read_text(encoding="utf-8")
    assert "SECRET_PR_DIFF" not in on_disk
    assert "SECRET_BREAK_THIS_PR" not in on_disk
    assert "red-team/review-1" not in path.name
    assert (
        store.replay(
            owner_did="did:web:owner.example",
            cassette_id="red-team/review-1",
            now=120,
        )
        == payload
    )
    with pytest.raises(WorkflowCassetteError, match="not found|mismatch"):
        store.replay(
            owner_did="did:web:other.example",
            cassette_id="red-team/review-1",
            now=120,
        )


def test_workflow_cassette_store_purges_expired(tmp_path):
    store = EncryptedWorkflowCassetteStore(
        tmp_path / ".workflow-cassettes",
        key=b"test cassette root key",
        retention_seconds=10,
    )
    expired = store.record(
        owner_did="did:web:owner.example",
        cassette_id="expired",
        payload={"ok": False},
        now=100,
    )
    active = store.record(
        owner_did="did:web:owner.example",
        cassette_id="active",
        payload={"ok": True},
        now=120,
    )

    purged = store.purge_expired(now=111)

    assert purged == [expired]
    assert not expired.exists()
    assert active.exists()


def test_workflow_harness_requires_manual_secret_store_for_env_cassettes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(CASSETTE_KEY_ENV, "test cassette root key")
    monkeypatch.delenv(CASSETTE_SECRET_STORE_ENV, raising=False)

    with pytest.raises(WorkflowCassetteError, match="manual approval"):
        WorkflowHarness(tmp_path, require_cassette_secret_store=True)

    monkeypatch.setenv(CASSETTE_SECRET_STORE_ENV, "approved")
    harness = WorkflowHarness(tmp_path, require_cassette_secret_store=True)

    assert harness.cassette_store is not None


def test_workflow_cassette_gitignore_and_plaintext_lint(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    gitignore = (repo_root / ".gitignore").read_text(
        encoding="utf-8"
    )
    assert ".workflow-cassettes/" in gitignore
    assert "tests/cassettes/" in gitignore
    assert "*.workflow-cassette.enc" in gitignore
    assert "*.workflow-cassette.json" in gitignore

    assert_no_plaintext_workflow_cassettes(tmp_path)
    offender = tmp_path / "bad.workflow-cassette.json"
    offender.write_text('{"pr_diff":"SECRET"}', encoding="utf-8")

    with pytest.raises(WorkflowCassetteError, match="plaintext"):
        assert_no_plaintext_workflow_cassettes(tmp_path)
