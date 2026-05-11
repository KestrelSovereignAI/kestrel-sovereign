"""Reflection-cycle workflow pilot contract (#1145).

The reflection feature is an optional package, so core cannot migrate
its internals directly. This module defines the workflow-shaped contract
that package can register against: stage names, signal sources, gates,
compensation sources, and the cron trigger used during the shadow-run
pilot before cutover.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from kestrel_sdk.signals import SignalMode

from kestrel_sovereign.features.workflows.models import (
    Edge,
    EdgeKind,
    Gate,
    Stage,
    Trigger,
    TriggerKind,
    WorkflowSpec,
)

REFLECTION_CYCLE_WORKFLOW_NAME = "reflection_cycle"
REFLECTION_CYCLE_CRON = "0 */4 * * *"

REFLECTION_CYCLE_STAGE_SOURCES = {
    "gather_observations": "reflection.gather_observations",
    "analyze_observations": "reflection.analyze_observations",
    "propose_improvement": "reflection.propose_improvement",
    "constitutional_review": "reflection.constitutional_review",
    "council_review": "reflection.council_review",
    "apply_or_defer": "reflection.apply_or_defer",
}

REFLECTION_CYCLE_COMPENSATION_SOURCES = {
    "constitutional_review": "reflection.compensate_constitutional_review",
    "council_review": "reflection.compensate_council_review",
    "apply_or_defer": "reflection.compensate_apply_or_defer",
}


def reflection_cycle_spec_payload(
    *,
    version: int = 1,
    cron_expression: str = REFLECTION_CYCLE_CRON,
    retention_days: int = 30,
) -> dict[str, Any]:
    """Return the unsigned workflow definition payload for the pilot.

    The optional reflection package owns the SourceRegistrations named
    here. Core keeps this payload unsigned so each agent signs it with
    its own DID at ``workflow_define`` time.
    """

    spec = WorkflowSpec(
        name=REFLECTION_CYCLE_WORKFLOW_NAME,
        version=version,
        stages=[
            Stage(
                name="gather_observations",
                signal_source=REFLECTION_CYCLE_STAGE_SOURCES[
                    "gather_observations"
                ],
                signal_mode=SignalMode.ACTION,
                params={"scope_param": "scope", "depth_param": "depth"},
                gate=Gate(type="signal_status_ok"),
                compensate="noop_idempotent",
                read_only=True,
            ),
            Stage(
                name="analyze_observations",
                signal_source=REFLECTION_CYCLE_STAGE_SOURCES[
                    "analyze_observations"
                ],
                signal_mode=SignalMode.ACTION,
                gate=Gate(type="signal_status_ok"),
                compensate="noop_idempotent",
                read_only=True,
                non_deterministic=True,
            ),
            Stage(
                name="propose_improvement",
                signal_source=REFLECTION_CYCLE_STAGE_SOURCES[
                    "propose_improvement"
                ],
                signal_mode=SignalMode.ACTION,
                gate=Gate(type="signal_status_ok"),
                compensate="noop_idempotent",
                read_only=True,
                non_deterministic=True,
            ),
            Stage(
                name="constitutional_review",
                signal_source=REFLECTION_CYCLE_STAGE_SOURCES[
                    "constitutional_review"
                ],
                signal_mode=SignalMode.COGNITION,
                gate=Gate(type="constitution_echo_verified"),
                compensate=REFLECTION_CYCLE_COMPENSATION_SOURCES[
                    "constitutional_review"
                ],
                non_deterministic=True,
            ),
            Stage(
                name="council_review",
                signal_source=REFLECTION_CYCLE_STAGE_SOURCES["council_review"],
                signal_mode=SignalMode.ACTION,
                gate=Gate(
                    type="council_approve",
                    params={"quorum": 2, "timeout": 3600},
                ),
                compensate=REFLECTION_CYCLE_COMPENSATION_SOURCES[
                    "council_review"
                ],
                non_deterministic=True,
            ),
            Stage(
                name="apply_or_defer",
                signal_source=REFLECTION_CYCLE_STAGE_SOURCES["apply_or_defer"],
                signal_mode=SignalMode.ACTION,
                gate=Gate(type="signal_status_ok"),
                compensate=REFLECTION_CYCLE_COMPENSATION_SOURCES[
                    "apply_or_defer"
                ],
            ),
        ],
        edges=[
            Edge(
                kind=EdgeKind.SEQUENTIAL,
                from_stage="gather_observations",
                to_stage="analyze_observations",
            ),
            Edge(
                kind=EdgeKind.SEQUENTIAL,
                from_stage="analyze_observations",
                to_stage="propose_improvement",
            ),
            Edge(
                kind=EdgeKind.SEQUENTIAL,
                from_stage="propose_improvement",
                to_stage="constitutional_review",
            ),
            Edge(
                kind=EdgeKind.SEQUENTIAL,
                from_stage="constitutional_review",
                to_stage="council_review",
            ),
            Edge(
                kind=EdgeKind.SEQUENTIAL,
                from_stage="council_review",
                to_stage="apply_or_defer",
            ),
        ],
        triggers=[
            Trigger(
                kind=TriggerKind.CRON,
                cron_expression=cron_expression,
                params={"scope": "all", "depth": "normal"},
            )
        ],
        params_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scope": {"type": "string", "default": "all"},
                "depth": {"type": "string", "default": "normal"},
                "shadow_legacy_result": {},
            },
        },
        retention_days=retention_days,
    )
    return spec.to_dict()


def normalize_reflection_output(value: Any) -> Any:
    """Normalize legacy/workflow reflection outputs for shadow comparison."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
    return value


def reflection_outputs_match(
    legacy_output: Any,
    workflow_output: Any,
) -> bool:
    """Return whether old ARTIFACT output and workflow output match."""

    return normalize_reflection_output(legacy_output) == normalize_reflection_output(
        workflow_output
    )


def reflection_shadow_report(
    *,
    legacy_output: Any,
    workflow_output: Any,
    workflow_run: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the side-by-side pilot report recorded before cutover."""

    return {
        "workflow": REFLECTION_CYCLE_WORKFLOW_NAME,
        "matched": reflection_outputs_match(legacy_output, workflow_output),
        "legacy_output": normalize_reflection_output(legacy_output),
        "workflow_output": normalize_reflection_output(workflow_output),
        "workflow_run": dict(workflow_run),
    }


__all__ = [
    "REFLECTION_CYCLE_COMPENSATION_SOURCES",
    "REFLECTION_CYCLE_CRON",
    "REFLECTION_CYCLE_STAGE_SOURCES",
    "REFLECTION_CYCLE_WORKFLOW_NAME",
    "normalize_reflection_output",
    "reflection_cycle_spec_payload",
    "reflection_outputs_match",
    "reflection_shadow_report",
]
