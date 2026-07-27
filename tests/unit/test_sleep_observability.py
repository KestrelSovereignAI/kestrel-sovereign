"""Contracts for privacy-safe semantic maintenance sleep observability."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign.agent.sleep import SleepMixin, SleepReport


def _maintenance(
    *,
    status: str = "complete",
    reason: str | None = None,
    source_generation: int = 17,
    checkpoint_generation: int = 17,
    changes_consumed: int = 3,
    assertions_validated: int = 3,
    assertions_inferred: int = 2,
    assertions_retracted: int = 1,
    contradictions: int = 1,
    reports_created: int = 2,
    backlog_assertions: int = 0,
    backlog_reports: int = 0,
    duration_ms: int = 19,
    capability_versions: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "reason": reason,
        "source_generation": source_generation,
        "checkpoint_generation": checkpoint_generation,
        "changes_consumed": changes_consumed,
        "assertions_validated": assertions_validated,
        "assertions_inferred": assertions_inferred,
        "assertions_retracted": assertions_retracted,
        "contradictions": contradictions,
        "reports_created": reports_created,
        "backlog_assertions": backlog_assertions,
        "backlog_reports": backlog_reports,
        "duration_ms": duration_ms,
        "capability_versions": capability_versions
        or {
            "semantic_maintenance": "v2",
            "maintenance_budget": "budget-digest",
            "shape_set": "default@1",
            "validation_capability": "shacl",
            "validation_profile_version": "1",
            "validation_artifact_pins": "artifact-digest",
            "inference_profile": "rdfs",
            "rule_profile": "rdfs-1",
            "ontology": "core@1",
        },
    }


class _CommandSleepAgent(SleepMixin):
    """Small command seam: the command must render the real report text."""

    def __init__(self, report: SleepReport) -> None:
        self._report = report
        self.sleep_calls: list[dict[str, object]] = []

    async def sleep(self, **kwargs) -> SleepReport:
        self.sleep_calls.append(kwargs)
        return self._report


def _prepare_app(agent):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original) -> None:
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


@pytest.mark.parametrize(
    ("status", "success", "reason", "expected_heading"),
    (
        pytest.param("complete", True, None, "Sleep cycle complete:", id="complete"),
        pytest.param("no_op", True, None, "Sleep cycle complete:", id="no-op"),
        pytest.param(
            "partial",
            False,
            "assertion_budget",
            "Sleep incomplete: semantic maintenance is partial.",
            id="partial",
        ),
    ),
)
def test_sleep_report_renders_distinct_bounded_maintenance_statuses(
    status: str,
    success: bool,
    reason: str | None,
    expected_heading: str,
) -> None:
    report = SleepReport(
        success=success,
        semantic_maintenance=_maintenance(
            status=status,
            reason=reason,
            backlog_assertions=4 if status == "partial" else 0,
            backlog_reports=1 if status == "partial" else 0,
        ),
    )

    rendered = str(report)

    assert rendered.startswith(expected_heading)
    assert f"status: {status}" in rendered
    assert f"reason: {reason or 'none'}" in rendered
    assert "generations: source=17 checkpoint=17" in rendered
    assert "changes: consumed=3 validated=3 inferred=2 retracted=1" in rendered
    assert "contradictions: 1" in rendered
    assert "reports: created=2" in rendered
    assert (
        f"backlog: assertions={4 if status == 'partial' else 0} "
        f"reports={1 if status == 'partial' else 0}"
    ) in rendered
    assert "duration: 19ms" in rendered
    assert "capabilities: versions=9 digest=" in rendered
    assert "Sleep failed: None" not in rendered


def test_sleep_report_maintenance_renderer_redacts_content_ids_and_raw_errors() -> None:
    sensitive_assertion = "ASSERTION_TERM_DO_NOT_RENDER"
    sensitive_id = "assertion-id-DO-NOT-RENDER"
    sensitive_tenant = "tenant-DO-NOT-RENDER"
    sensitive_sql = "SELECT private_fact FROM assertions"
    sensitive_provenance = "source-provenance-DO-NOT-RENDER"
    report = SleepReport(
        success=False,
        error=f"{sensitive_sql}: provider error",
        semantic_maintenance={
            **_maintenance(
                status="partial",
                reason=f"{sensitive_sql} {sensitive_assertion}",
                capability_versions={
                    "semantic_maintenance": "v2",
                    "shape_set": sensitive_provenance,
                    "validation_capability": "shacl",
                    "validation_profile_version": "1",
                    "tenant_id": sensitive_tenant,
                },
            ),
            "run_id": sensitive_id,
            "assertion_id": sensitive_id,
            "source": sensitive_provenance,
            "tenant_id": sensitive_tenant,
            "raw_error": sensitive_sql,
        },
    )

    rendered = str(report)
    summary = report.semantic_maintenance_summary()

    assert rendered == str(report), "rendering must be deterministic"
    assert summary is not None
    assert len(summary) <= 1_024
    assert "reason: unavailable" in rendered
    assert "capabilities: versions=4 digest=" in rendered
    for secret in (
        sensitive_assertion,
        sensitive_id,
        sensitive_tenant,
        sensitive_sql,
        sensitive_provenance,
        "run_id",
        "assertion_id",
        "raw_error",
    ):
        assert secret not in rendered


@pytest.mark.asyncio
async def test_consolidate_only_command_includes_safe_maintenance_summary() -> None:
    report = SleepReport(
        success=True,
        semantic_maintenance=_maintenance(status="complete", reason=None),
    )
    agent = _CommandSleepAgent(report)

    rendered = await agent._command_sleep("!sleep --consolidate-only")

    assert agent.sleep_calls == [
        {
            "tier": "ipfs",
            "skip_consolidation": False,
            "skip_export": True,
        }
    ]
    assert "Semantic maintenance:" in rendered
    assert "status: complete" in rendered
    assert "generations: source=17 checkpoint=17" in rendered
    assert "capabilities: versions=9 digest=" in rendered


def test_authenticated_invoke_preserves_consolidate_only_maintenance_summary() -> None:
    """The actual HTTP invoke envelope must not drop the command's summary."""
    command_agent = _CommandSleepAgent(
        SleepReport(
            success=False,
            semantic_maintenance=_maintenance(
                status="partial",
                reason="assertion_budget",
                backlog_assertions=1,
            ),
        )
    )

    async def process_input(user_input: str, **_kwargs) -> str:
        return await command_agent._command_sleep(user_input)

    agent = MagicMock()
    agent.agent_id = "did:test:sleep-observability"
    agent.storage.resolve_session_id = AsyncMock(return_value=None)
    agent.register_active_request = MagicMock()
    agent._cleanup_cancelled_request = MagicMock()
    agent.process_input = AsyncMock(side_effect=process_input)
    agent._conversation_response_identity = MagicMock(
        return_value={"model": None, "provider": None}
    )
    app, original = _prepare_app(agent)

    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agent/invoke",
                    headers={"X-API-Key": "test-key"},
                    json={"input": "!sleep --consolidate-only"},
                )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response"].startswith(
            "Sleep incomplete: semantic maintenance is partial."
        )
        assert "status: partial" in body["response"]
        assert "backlog: assertions=1 reports=0" in body["response"]
        assert "capabilities: versions=9 digest=" in body["response"]
        agent.process_input.assert_awaited_once()
        assert command_agent.sleep_calls == [
            {
                "tier": "ipfs",
                "skip_consolidation": False,
                "skip_export": True,
            }
        ]
    finally:
        _restore_app(app, original)
