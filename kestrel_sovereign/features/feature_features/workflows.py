"""Workflow definition payloads for FeatureFeature (#1151)."""

from __future__ import annotations

from typing import Any, Literal

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

FEATURE_PROPOSE_TOOL_WORKFLOW_NAME = "feature_propose_tool"
FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME = "feature_propose_package"

DEFAULT_REPOSITORY = "KestrelSovereignAI/kestrel-sovereign"
DEFAULT_BRANCH = "feature-proposal-branch-required"
DEFAULT_PROMPT_PACK_CONSTRAINT = ">=1.0,<2.0"

FEATURE_FEATURES_STAGE_ORDER = (
    "explore",
    "design_plan",
    "constitutional_review",
    "file_github_epic",
    "assign_talon_chunks",
    "implement_chunks",
    "tests_pass",
    "lint_clean",
    "ci_green",
    "constitutional_boundary_scan",
    "red_team_review",
    "council_review",
    "publish",
    "audit_anchor",
)

FEATURE_FEATURES_STAGE_SOURCES = {
    "explore": "feature_features.explore",
    "design_plan": "feature_features.design_plan",
    "constitutional_review": "feature_features.constitutional_review",
    "file_github_epic": "feature_features.file_github_epic",
    "assign_talon_chunks": "feature_features.assign_talon_chunks",
    "implement_chunks": "feature_features.implement_chunks",
    "tests_pass": "feature_features.tests_pass",
    "lint_clean": "feature_features.lint_clean",
    "ci_green": "feature_features.ci_green",
    "constitutional_boundary_scan": "feature_features.boundary_scan",
    "red_team_review": "feature_features.red_team_review",
    "council_review": "feature_features.council_review",
    "publish": "feature_features.publish",
    "audit_anchor": "feature_features.audit_anchor",
}

FEATURE_FEATURES_COMPENSATION_SOURCES = {
    "design_plan": "feature_features.compensate_design_plan",
    "constitutional_review": "feature_features.compensate_constitutional_review",
    "file_github_epic": "feature_features.compensate_file_github_epic",
    "assign_talon_chunks": "feature_features.compensate_assign_talon_chunks",
    "implement_chunks": "feature_features.compensate_implement_chunks",
}

FEATURE_FEATURES_REVIEWER_SOURCES = {
    "codex": "review.codex",
    "claude": "review.claude",
}

_FORBIDDEN_MODULES = (
    "kestrel_sovereign.constitution",
    "kestrel_sovereign.features.constitution",
    "kestrel_sovereign.identity",
    "kestrel_sovereign.features.identity",
    "kestrel_sovereign.inception_service",
    "kestrel_sovereign.agent.constitution",
)


def feature_propose_tool_spec_payload(
    *,
    version: int = 1,
    repository: str = DEFAULT_REPOSITORY,
    branch: str = DEFAULT_BRANCH,
    prompt_pack_constraint: str = DEFAULT_PROMPT_PACK_CONSTRAINT,
    retention_days: int = 90,
) -> dict[str, Any]:
    """Return the unsigned workflow definition for proposing a core tool."""

    spec = _feature_feature_spec(
        name=FEATURE_PROPOSE_TOOL_WORKFLOW_NAME,
        version=version,
        repository=repository,
        branch=branch,
        prompt_pack_constraint=prompt_pack_constraint,
        retention_days=retention_days,
        params_schema=_tool_params_schema(),
        proposal_kind="tool",
    )
    return spec.to_dict()


def feature_propose_package_spec_payload(
    *,
    version: int = 1,
    repository: str = DEFAULT_REPOSITORY,
    branch: str = DEFAULT_BRANCH,
    prompt_pack_constraint: str = DEFAULT_PROMPT_PACK_CONSTRAINT,
    retention_days: int = 90,
) -> dict[str, Any]:
    """Return the unsigned workflow definition for proposing a feature package."""

    spec = _feature_feature_spec(
        name=FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME,
        version=version,
        repository=repository,
        branch=branch,
        prompt_pack_constraint=prompt_pack_constraint,
        retention_days=retention_days,
        params_schema=_package_params_schema(),
        proposal_kind="package",
    )
    return spec.to_dict()


def feature_feature_workflow_payloads(
    kind: Literal["all", "tool", "package"] = "all",
    *,
    version: int = 1,
    repository: str = DEFAULT_REPOSITORY,
    branch: str = DEFAULT_BRANCH,
    prompt_pack_constraint: str = DEFAULT_PROMPT_PACK_CONSTRAINT,
    retention_days: int = 90,
) -> dict[str, dict[str, Any]]:
    """Return the available FeatureFeature workflow payloads by name."""

    payloads: dict[str, dict[str, Any]] = {}
    if kind in {"all", "tool"}:
        payloads[FEATURE_PROPOSE_TOOL_WORKFLOW_NAME] = (
            feature_propose_tool_spec_payload(
                version=version,
                repository=repository,
                branch=branch,
                prompt_pack_constraint=prompt_pack_constraint,
                retention_days=retention_days,
            )
        )
    if kind in {"all", "package"}:
        payloads[FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME] = (
            feature_propose_package_spec_payload(
                version=version,
                repository=repository,
                branch=branch,
                prompt_pack_constraint=prompt_pack_constraint,
                retention_days=retention_days,
            )
        )
    if not payloads:
        raise ValueError("kind must be one of: all, tool, package")
    return payloads


def _feature_feature_spec(
    *,
    name: str,
    version: int,
    repository: str,
    branch: str,
    prompt_pack_constraint: str,
    retention_days: int,
    params_schema: dict[str, Any],
    proposal_kind: str,
) -> WorkflowSpec:
    return WorkflowSpec(
        name=name,
        version=version,
        stages=_feature_feature_stages(
            repository=repository,
            branch=branch,
            prompt_pack_constraint=prompt_pack_constraint,
            proposal_kind=proposal_kind,
        ),
        edges=[
            Edge(
                kind=EdgeKind.SEQUENTIAL,
                from_stage=from_stage,
                to_stage=to_stage,
            )
            for from_stage, to_stage in zip(
                FEATURE_FEATURES_STAGE_ORDER,
                FEATURE_FEATURES_STAGE_ORDER[1:],
            )
        ],
        triggers=[Trigger(kind=TriggerKind.MANUAL)],
        params_schema=params_schema,
        retention_days=retention_days,
    )


def _feature_feature_stages(
    *,
    repository: str,
    branch: str,
    prompt_pack_constraint: str,
    proposal_kind: str,
) -> list[Stage]:
    return [
        Stage(
            name="explore",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["explore"],
            signal_mode=SignalMode.ACTION,
            params={"proposal_kind": proposal_kind},
            gate=Gate(type="signal_status_ok"),
            compensate="noop_idempotent",
            read_only=True,
        ),
        Stage(
            name="design_plan",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["design_plan"],
            signal_mode=SignalMode.COGNITION,
            params={"proposal_kind": proposal_kind},
            gate=Gate(type="signal_status_ok"),
            compensate=FEATURE_FEATURES_COMPENSATION_SOURCES["design_plan"],
            non_deterministic=True,
        ),
        Stage(
            name="constitutional_review",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES[
                "constitutional_review"
            ],
            signal_mode=SignalMode.COGNITION,
            gate=Gate(type="constitution_echo_verified"),
            compensate=FEATURE_FEATURES_COMPENSATION_SOURCES[
                "constitutional_review"
            ],
            non_deterministic=True,
        ),
        Stage(
            name="file_github_epic",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["file_github_epic"],
            signal_mode=SignalMode.ACTION,
            gate=Gate(type="signal_status_ok"),
            compensate=FEATURE_FEATURES_COMPENSATION_SOURCES[
                "file_github_epic"
            ],
        ),
        Stage(
            name="assign_talon_chunks",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES[
                "assign_talon_chunks"
            ],
            signal_mode=SignalMode.ACTION,
            gate=Gate(type="signal_status_ok"),
            compensate=FEATURE_FEATURES_COMPENSATION_SOURCES[
                "assign_talon_chunks"
            ],
            non_deterministic=True,
        ),
        Stage(
            name="implement_chunks",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["implement_chunks"],
            signal_mode=SignalMode.ACTION,
            gate=Gate(type="signal_status_ok"),
            compensate=FEATURE_FEATURES_COMPENSATION_SOURCES[
                "implement_chunks"
            ],
            non_deterministic=True,
        ),
        Stage(
            name="tests_pass",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["tests_pass"],
            signal_mode=SignalMode.ACTION,
            gate=Gate(type="tests_pass", params={"suite": "unit"}),
            compensate="noop_idempotent",
            read_only=True,
        ),
        Stage(
            name="lint_clean",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["lint_clean"],
            signal_mode=SignalMode.ACTION,
            gate=Gate(type="lint_clean", params={"scopes": ["changed"]}),
            compensate="noop_idempotent",
            read_only=True,
        ),
        Stage(
            name="ci_green",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["ci_green"],
            signal_mode=SignalMode.ACTION,
            gate=Gate(
                type="ci_green",
                params={
                    "repo": repository,
                    "branch": branch,
                    "repo_param": "repository",
                    "branch_param": "branch",
                    "required_checks": [
                        "lint-and-imports",
                        "unit-tests",
                        "integration-tests",
                    ],
                },
            ),
            compensate="noop_idempotent",
            read_only=True,
        ),
        Stage(
            name="constitutional_boundary_scan",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES[
                "constitutional_boundary_scan"
            ],
            signal_mode=SignalMode.ACTION,
            gate=Gate(
                type="constitutional_boundary_clean",
                params={"forbidden_modules": list(_FORBIDDEN_MODULES)},
            ),
            compensate="noop_idempotent",
            forbidden_modules=_FORBIDDEN_MODULES,
            read_only=True,
        ),
        Stage(
            name="red_team_review",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["red_team_review"],
            signal_mode=SignalMode.ACTION,
            gate=Gate(
                type="red_team_clear",
                params={
                    "prompt_pack_constraint": prompt_pack_constraint,
                    "reviewer_pool": [
                        "codex",
                        "claude",
                    ],
                    "blockers": "zero",
                    "max_total_tokens": 100000,
                    "max_total_cost_usd": 25.0,
                },
            ),
            compensate="noop_idempotent",
            read_only=True,
            non_deterministic=True,
        ),
        Stage(
            name="council_review",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["council_review"],
            signal_mode=SignalMode.ACTION,
            gate=Gate(
                type="council_approve",
                params={"quorum": 2, "timeout": 86400},
            ),
            compensate="noop_idempotent",
            read_only=True,
            non_deterministic=True,
        ),
        Stage(
            name="publish",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["publish"],
            signal_mode=SignalMode.ACTION,
            gate=Gate(type="signal_status_ok"),
            compensate="compensate_record_only",
            irreversible=True,
        ),
        Stage(
            name="audit_anchor",
            signal_source=FEATURE_FEATURES_STAGE_SOURCES["audit_anchor"],
            signal_mode=SignalMode.ACTION,
            gate=Gate(type="signal_status_ok"),
            compensate="compensate_record_only",
            irreversible=True,
        ),
    ]


def _tool_params_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "feature_name",
            "target_tool_name",
            "repository",
            "branch",
        ],
        "properties": {
            "feature_name": {"type": "string", "minLength": 1},
            "target_tool_name": {"type": "string", "minLength": 1},
            "summary": {"type": "string"},
            "repository": {"type": "string", "pattern": "^[^/]+/[^/]+$"},
            "branch": {"type": "string", "minLength": 1},
            "operator_did": {"type": "string"},
        },
    }


def _package_params_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["package_name", "repository", "branch"],
        "properties": {
            "package_name": {"type": "string", "minLength": 1},
            "entry_point_group": {
                "type": "string",
                "default": "kestrel_sovereign.features",
            },
            "summary": {"type": "string"},
            "repository": {"type": "string", "pattern": "^[^/]+/[^/]+$"},
            "branch": {"type": "string", "minLength": 1},
            "operator_did": {"type": "string"},
        },
    }


__all__ = [
    "DEFAULT_BRANCH",
    "DEFAULT_PROMPT_PACK_CONSTRAINT",
    "DEFAULT_REPOSITORY",
    "FEATURE_FEATURES_COMPENSATION_SOURCES",
    "FEATURE_FEATURES_REVIEWER_SOURCES",
    "FEATURE_FEATURES_STAGE_ORDER",
    "FEATURE_FEATURES_STAGE_SOURCES",
    "FEATURE_PROPOSE_PACKAGE_WORKFLOW_NAME",
    "FEATURE_PROPOSE_TOOL_WORKFLOW_NAME",
    "feature_feature_workflow_payloads",
    "feature_propose_package_spec_payload",
    "feature_propose_tool_spec_payload",
]
