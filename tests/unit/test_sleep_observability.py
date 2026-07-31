"""Contracts for privacy-safe semantic maintenance sleep observability."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign.agent.sleep import SleepMixin, SleepReport
from kestrel_sovereign.knowledge.assertion import OntologyRef
from kestrel_sovereign.knowledge.inference import InferenceProfile
from kestrel_sovereign.knowledge.registry import ResourceKind, get_knowledge_registry
from kestrel_sovereign.knowledge.release_evidence import release_gate_specs
from kestrel_sovereign.knowledge.release_evidence_models import (
    ArtifactReference,
    EvidenceRecord,
    GateResult,
)


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


def _known_inference_profile_values() -> tuple[dict[str, str], InferenceProfile]:
    registry = get_knowledge_registry()
    ontology = next(
        resource
        for resource in registry.resources
        if resource.identifier == "kestrel-vocab"
        and str(resource.version) == "1.0.0"
        and resource.kind is ResourceKind.ONTOLOGY
    )
    profile = InferenceProfile(
        OntologyRef(
            ontology.namespace,
            str(ontology.version),
            ontology.sha256,
            registry.contract_version,
        ),
        "1.0.0",
    )
    return {
        "inference_profile": profile.key,
        "rule_profile": profile.rule_profile_version,
        "ontology": f"{ontology.namespace}@{ontology.version}",
    }, profile


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
    diagnostics = report.semantic_maintenance_diagnostics()

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
        assert secret not in repr(diagnostics)


def test_sleep_diagnostics_expose_verified_capabilities_and_repair_guidance() -> None:
    profile_values, profile = _known_inference_profile_values()
    report = SleepReport(
        success=False,
        semantic_maintenance=_maintenance(
            status="partial",
            reason="assertion_budget",
            backlog_assertions=3,
            backlog_reports=1,
            capability_versions={
                "semantic_maintenance": "v3",
                "shape_set": "kestrel-assertion-shapes@1.0.0",
                "validation_capability": "validation-profile:shacl-core-20170720",
                "validation_profile_version": "registry-selected",
                **profile_values,
            },
        ),
    )

    diagnostics = report.semantic_maintenance_diagnostics()
    assert diagnostics == {
        "status": "partial",
        "reason": "assertion_budget",
        "checkpoint": {
            "source_generation": "17",
            "checkpoint_generation": "17",
        },
        "backlog": {"assertions": "3", "reports": "1"},
        "partial": True,
        "repair_guidance": "rerun_bounded_maintenance",
        "active_capabilities": [
            "semantic_maintenance=v3",
            "shape_set=kestrel-assertion-shapes@1.0.0",
            "validation_capability=validation-profile:shacl-core-20170720",
            "validation_profile_version=registry-selected",
            f"rule_profile={profile.rule_profile_version}",
            "ontology=kestrel-vocab@1.0.0",
            f"inference_profile={profile.key}",
        ],
    }
    assert "active capabilities: semantic_maintenance=v3" in str(report)
    assert "repair guidance: rerun_bounded_maintenance" in str(report)
    assert report.to_dict()["semantic_maintenance_diagnostics"] == diagnostics


def test_sleep_diagnostics_reject_version_shaped_unregistered_capability_labels() -> None:
    report = SleepReport(
        success=False,
        semantic_maintenance=_maintenance(
            status="partial",
            reason="assertion_budget",
            capability_versions={
                "semantic_maintenance": "v999",
                "shape_set": "kestrel-assertion-shapes@999.999.999",
                "validation_capability": "validation-profile:unknown",
                "validation_profile_version": "999.999.999",
                "inference_profile": "sha256:" + "f" * 64,
                "rule_profile": "rdfs-v1@999.999.999",
                "ontology": "https://kestrel.ai/vocab/@999.999.999",
            },
        ),
    )

    active = report.semantic_maintenance_diagnostics()["active_capabilities"]
    rendered = " ".join(active)

    assert "v999" not in rendered
    assert "999.999.999" not in rendered
    assert "validation-profile:unknown" not in rendered
    assert "inference_profile=unavailable" in active
    assert "ontology=unavailable" in active


def test_sleep_diagnostics_marks_absent_producer_profile_and_ontology_as_omitted() -> None:
    report = SleepReport(
        success=True,
        semantic_maintenance=_maintenance(
            capability_versions={"semantic_maintenance": "v3"},
        ),
    )

    assert report.semantic_maintenance_diagnostics()["active_capabilities"] == [
        "semantic_maintenance=v3",
        "ontology=omitted",
        "inference_profile=omitted",
    ]


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
    replay_assertion_id = "replay-assertion-id-DO-NOT-RENDER"
    command_agent = _CommandSleepAgent(
        SleepReport(
            success=False,
            semantic_maintenance={
                **_maintenance(
                    status="partial",
                    reason="change_replay",
                    backlog_assertions=1,
                ),
                "run_id": replay_assertion_id,
                "assertion_id": replay_assertion_id,
            },
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
        assert "reason: change_replay" in body["response"]
        assert "backlog: assertions=1 reports=0" in body["response"]
        assert "capabilities: versions=9 digest=" in body["response"]
        assert replay_assertion_id not in body["response"]
        diagnostics_gate = next(
            spec
            for spec in release_gate_specs()
            if spec.gate_id == "semantic_maintenance_diagnostics_contract"
        )
        assert diagnostics_gate.required_for_ready is True
        assert {
            field.field_id: field.kind
            for field in diagnostics_gate.observation_schema.fields
        } == {
            "diagnostic_count": "positive_count",
            "redaction_violation_count": "zero_count",
        }
        live_observation = {
            "diagnostic_count": sum(
                marker in body["response"]
                for marker in (
                    "status:",
                    "reason:",
                    "generations:",
                    "backlog:",
                    "active capabilities:",
                    "repair guidance:",
                )
            ),
            "redaction_violation_count": sum(
                marker in body["response"]
                for marker in (
                    replay_assertion_id,
                    "tenant_id",
                    "raw_error",
                )
            ),
        }
        GateResult(
            diagnostics_gate,
            EvidenceRecord.attest(
                diagnostics_gate,
                live_observation,
                ArtifactReference(
                    "ci://semantic-release/kite-http-diagnostics",
                    "a" * 64,
                ),
            ),
        )
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
