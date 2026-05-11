"""Tests for the reflection-cycle workflow pilot contract (#1145)."""

from __future__ import annotations

from jsonschema import Draft202012Validator
from kestrel_sdk.signals import SignalMode

from kestrel_sovereign.features.workflows.models import (
    TriggerKind,
    WorkflowSpec,
)
from kestrel_sovereign.features.workflows.reflection_cycle import (
    REFLECTION_CYCLE_COMPENSATION_SOURCES,
    REFLECTION_CYCLE_CRON,
    REFLECTION_CYCLE_STAGE_SOURCES,
    REFLECTION_CYCLE_WORKFLOW_NAME,
    reflection_cycle_spec_payload,
    reflection_outputs_match,
    reflection_shadow_report,
)
from kestrel_sovereign.features.workflows.schema import validate_spec_payload


def test_reflection_cycle_spec_payload_is_valid_unsigned_workflow():
    payload = reflection_cycle_spec_payload()
    validate_spec_payload(payload)
    spec = WorkflowSpec.from_dict(payload)

    assert spec.name == REFLECTION_CYCLE_WORKFLOW_NAME
    assert spec.version == 1
    assert spec.triggers[0].kind == TriggerKind.CRON
    assert spec.triggers[0].cron_expression == REFLECTION_CYCLE_CRON
    assert spec.triggers[0].params == {"scope": "all", "depth": "normal"}
    assert [stage.name for stage in spec.stages] == [
        "gather_observations",
        "analyze_observations",
        "propose_improvement",
        "constitutional_review",
        "council_review",
        "apply_or_defer",
    ]


def test_reflection_cycle_params_schema_accepts_raw_shadow_outputs():
    payload = reflection_cycle_spec_payload()
    schema = payload["params_schema"]
    validator = Draft202012Validator(schema)

    for shadow in (
        '{"status":"deferred"}',
        "plain text reflection",
        {"status": "deferred"},
        ["insight-a", "insight-b"],
        None,
    ):
        validator.validate(
            {"scope": "all", "depth": "normal", "shadow_legacy_result": shadow}
        )


def test_reflection_cycle_stages_map_existing_internal_flow():
    spec = WorkflowSpec.from_dict(reflection_cycle_spec_payload())
    by_name = {stage.name: stage for stage in spec.stages}

    assert by_name["gather_observations"].signal_source == (
        REFLECTION_CYCLE_STAGE_SOURCES["gather_observations"]
    )
    assert by_name["analyze_observations"].non_deterministic is True
    assert by_name["propose_improvement"].non_deterministic is True
    assert by_name["constitutional_review"].signal_mode == SignalMode.COGNITION
    assert by_name["constitutional_review"].gate.type == (
        "constitution_echo_verified"
    )
    assert by_name["council_review"].gate.type == "council_approve"
    assert by_name["apply_or_defer"].compensate == (
        REFLECTION_CYCLE_COMPENSATION_SOURCES["apply_or_defer"]
    )


def test_reflection_cycle_edges_are_strictly_sequential():
    spec = WorkflowSpec.from_dict(reflection_cycle_spec_payload())
    assert [(edge.from_stage, edge.to_stage) for edge in spec.edges] == [
        ("gather_observations", "analyze_observations"),
        ("analyze_observations", "propose_improvement"),
        ("propose_improvement", "constitutional_review"),
        ("constitutional_review", "council_review"),
        ("council_review", "apply_or_defer"),
    ]


def test_reflection_shadow_report_normalizes_json_outputs():
    legacy = '{"insights":["a"],"status":"deferred"}'
    workflow = {"insights": ["a"], "status": "deferred"}
    report = reflection_shadow_report(
        legacy_output=legacy,
        workflow_output=workflow,
        workflow_run={"run_id": "run-1", "status": "completed"},
    )

    assert reflection_outputs_match(legacy, workflow) is True
    assert report["matched"] is True
    assert report["legacy_output"] == workflow
    assert report["workflow_run"] == {"run_id": "run-1", "status": "completed"}


def test_reflection_shadow_report_detects_mismatch():
    assert reflection_outputs_match(
        {"status": "applied"},
        {"status": "deferred"},
    ) is False
