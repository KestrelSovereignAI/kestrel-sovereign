"""Tests for the reusable WorkflowHarness fixture."""

from __future__ import annotations

import pytest

from kestrel_sdk.signals import SignalMode

from kestrel_sovereign.features.workflows import RunStatus, Stage, WorkflowSpec
from tests.fixtures.workflow_harness import WorkflowHarness


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
