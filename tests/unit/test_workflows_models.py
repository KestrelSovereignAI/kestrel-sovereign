"""Phase 0 chunk A — dataclass + schema invariants for kestrel-feature-workflows.

Pins the closed vocabularies and round-trip behavior. Anything that
loosens these enums is a load-bearing design change and should land
behind a separate spec update first.
"""

from __future__ import annotations

import json

import pytest

try:
    import jsonschema  # noqa: F401
except ImportError:  # pragma: no cover
    jsonschema = None

from kestrel_sdk.signals import SignalMode

from kestrel_sovereign.features.workflows import (
    BUILT_IN_GATE_TYPES,
    BUILT_IN_GATE_TYPES_NEEDING_REGISTRATION,
    Edge,
    EdgeKind,
    Gate,
    GateOutcome,
    RunStatus,
    Stage,
    StageLink,
    Trigger,
    TriggerKind,
    WorkflowDefinitionError,
    WorkflowRun,
    WorkflowSpec,
)
from kestrel_sovereign.features.workflows.schema import (
    WORKFLOW_RUN_SCHEMA,
    WORKFLOW_SPEC_SCHEMA,
    WORKFLOW_STAGE_LINK_SCHEMA,
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


def test_built_in_gate_types_match_design_doc():
    """Gates listed in design §3.3 — the schema validator and Phase 1
    runner depend on this set. Adding a gate type without updating the
    design and the schema is a synchronization bug."""
    assert BUILT_IN_GATE_TYPES == frozenset(
        {
            "signal_status_ok",
            "tests_pass",
            "ci_green",
            "lint_clean",
            "red_team_clear",
            "council_approve",
            "consent_collect",
            "signature_collected",
            "script",
            "constitution_echo_verified",
            "constitutional_boundary_clean",
        }
    )


def test_gate_types_needing_registration_subset_of_built_ins():
    assert BUILT_IN_GATE_TYPES_NEEDING_REGISTRATION.issubset(BUILT_IN_GATE_TYPES)


def test_run_status_enum_matches_design_doc_section_5():
    assert {s.value for s in RunStatus} == {
        "pending",
        "running",
        "paused",
        "waiting",
        "compensating",
        "completed",
        "failed",
        "cancelled",
        "cancelled_with_irreversible_residue",
    }


def test_gate_outcome_vocabulary():
    assert {o.value for o in GateOutcome} == {"pass", "fail", "pending"}


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_gate_rejects_unknown_type():
    with pytest.raises(WorkflowDefinitionError):
        Gate(type="custom_callable")  # removed in v4


def test_gate_signal_status_ok_default():
    gate = Gate(type="signal_status_ok")
    assert gate.params == {}


def test_gate_constitutional_boundary_clean_requires_forbidden_modules():
    with pytest.raises(WorkflowDefinitionError):
        Gate(type="constitutional_boundary_clean")
    with pytest.raises(WorkflowDefinitionError):
        Gate(type="constitutional_boundary_clean", params={"forbidden_modules": "features.security"})
    Gate(
        type="constitutional_boundary_clean",
        params={"forbidden_modules": ["features.security", "features.identity"]},
    )


def test_gate_red_team_clear_requires_prompt_pack_constraint():
    with pytest.raises(WorkflowDefinitionError):
        Gate(type="red_team_clear", params={"reviewer_pool": ["did:web:r1"]})
    Gate(
        type="red_team_clear",
        params={"prompt_pack_constraint": "==1.2.0"},
    )


def test_gate_round_trip_dict():
    gate = Gate(type="red_team_clear", params={"prompt_pack_constraint": ">=1,<2"})
    assert Gate.from_dict(gate.to_dict()) == gate


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def _action_stage(**overrides):
    base = dict(
        name="emit_artifact",
        signal_source="emit.docs",
        signal_mode=SignalMode.ACTION,
        read_only=True,
    )
    base.update(overrides)
    return Stage(**base)


def test_stage_minimal_action_read_only_uses_noop_idempotent():
    stage = _action_stage()
    assert stage.compensate == "noop_idempotent"


def test_stage_noop_idempotent_rejected_for_writeful_action():
    with pytest.raises(WorkflowDefinitionError):
        _action_stage(read_only=False)


def test_stage_noop_idempotent_allowed_when_gate_is_consent_collect():
    Stage(
        name="ask_user",
        signal_source="hooks.consent",
        signal_mode=SignalMode.ACTION,
        read_only=False,
        gate=Gate(type="consent_collect", params={"scope": "publish_pr"}),
    )


def test_stage_compensate_can_be_named_source():
    stage = Stage(
        name="publish",
        signal_source="ci.publish",
        signal_mode=SignalMode.ACTION,
        compensate="ci.publish.compensate",
    )
    assert stage.compensate == "ci.publish.compensate"


def test_stage_signal_mode_string_coercion():
    stage = Stage(
        name="cognition",
        signal_source="agent.write_pr",
        signal_mode="cognition",
        compensate="agent.write_pr.compensate",
    )
    assert stage.signal_mode == SignalMode.COGNITION


def test_stage_invalid_signal_mode_raises():
    with pytest.raises(WorkflowDefinitionError):
        Stage(
            name="bad",
            signal_source="x",
            signal_mode="reflexive",
            compensate="x.compensate",
        )


def test_stage_from_dict_rejects_string_booleans():
    """Codex round 1 P2: ``bool("false")`` is True. ``Stage.from_dict``
    must NOT silently coerce strings to booleans, or a writeful ACTION
    stage with ``read_only="false"`` would slip past the
    noop_idempotent eligibility check."""
    with pytest.raises(WorkflowDefinitionError):
        Stage.from_dict(
            {
                "name": "publish",
                "signal_source": "ci.publish",
                "signal_mode": "action",
                "compensate": "noop_idempotent",
                "read_only": "false",  # the bug — string, not bool
            }
        )


def test_stage_round_trip():
    stage = _action_stage(
        gate=Gate(
            type="red_team_clear",
            params={"prompt_pack_constraint": "==1.0.0"},
        ),
    )
    assert Stage.from_dict(stage.to_dict()) == stage


def test_stage_invalid_name():
    with pytest.raises(WorkflowDefinitionError):
        Stage(
            name="9bad-start",
            signal_source="x",
            signal_mode=SignalMode.ACTION,
            read_only=True,
        )


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


def test_edge_sequential_minimal():
    edge = Edge(kind=EdgeKind.SEQUENTIAL, from_stage="a", to_stage="b")
    assert Edge.from_dict(edge.to_dict()) == edge


def test_edge_sequential_rejects_cross_kind_fields():
    with pytest.raises(WorkflowDefinitionError):
        Edge(
            kind=EdgeKind.SEQUENTIAL,
            from_stage="a",
            to_stage="b",
            condition="x == 1",
        )


def test_edge_branch_requires_branches():
    with pytest.raises(WorkflowDefinitionError):
        Edge(
            kind=EdgeKind.BRANCH,
            from_stage="a",
            condition="x == 1",
            true_stage="b",
            # false_stage missing
        )
    Edge(
        kind=EdgeKind.BRANCH,
        from_stage="a",
        condition="x == 1",
        true_stage="b",
        false_stage="c",
    )


def test_edge_parallel_requires_min_two_stages():
    with pytest.raises(WorkflowDefinitionError):
        Edge(
            kind=EdgeKind.PARALLEL,
            from_stage="a",
            stages=("only_one",),
            join_strategy="all",
        )
    Edge(
        kind=EdgeKind.PARALLEL,
        from_stage="a",
        stages=("b", "c"),
        join_strategy="all",
    )


def test_edge_parallel_rejects_unknown_join_strategy():
    with pytest.raises(WorkflowDefinitionError):
        Edge(
            kind=EdgeKind.PARALLEL,
            from_stage="a",
            stages=("b", "c"),
            join_strategy="quorum",
        )


def test_edge_subworkflow_requires_version():
    with pytest.raises(WorkflowDefinitionError):
        Edge(
            kind=EdgeKind.SUBWORKFLOW,
            from_stage="a",
            subworkflow_name="child",
        )
    Edge(
        kind=EdgeKind.SUBWORKFLOW,
        from_stage="a",
        subworkflow_name="child",
        subworkflow_version=2,
    )


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


def test_trigger_manual_default():
    Trigger(kind=TriggerKind.MANUAL)


def test_trigger_cron_requires_expression():
    with pytest.raises(WorkflowDefinitionError):
        Trigger(kind=TriggerKind.CRON)
    Trigger(kind=TriggerKind.CRON, cron_expression="0 4 * * *")


def test_trigger_signal_source_requires_name():
    with pytest.raises(WorkflowDefinitionError):
        Trigger(kind=TriggerKind.SIGNAL_SOURCE)
    Trigger(kind=TriggerKind.SIGNAL_SOURCE, signal_source="github.pr_opened")


# ---------------------------------------------------------------------------
# WorkflowSpec
# ---------------------------------------------------------------------------


def _minimal_spec(**overrides):
    base = dict(
        name="release",
        version=1,
        stages=[
            _action_stage(name="lint"),
            _action_stage(name="test"),
        ],
        edges=[Edge(kind=EdgeKind.SEQUENTIAL, from_stage="lint", to_stage="test")],
    )
    base.update(overrides)
    return WorkflowSpec(**base)


def test_workflow_spec_roundtrip_dict():
    spec = _minimal_spec()
    assert WorkflowSpec.from_dict(spec.to_dict()) == spec


def test_workflow_spec_rejects_duplicate_stage_names():
    with pytest.raises(WorkflowDefinitionError):
        WorkflowSpec(
            name="dup",
            version=1,
            stages=[_action_stage(name="x"), _action_stage(name="x")],
        )


def test_workflow_spec_edge_must_reference_declared_stages():
    with pytest.raises(WorkflowDefinitionError):
        WorkflowSpec(
            name="r",
            version=1,
            stages=[_action_stage(name="lint")],
            edges=[
                Edge(
                    kind=EdgeKind.SEQUENTIAL,
                    from_stage="lint",
                    to_stage="test",  # not declared
                )
            ],
        )


def test_workflow_spec_compute_spec_hash_excludes_signature_and_hash():
    spec = _minimal_spec()
    h1 = spec.compute_spec_hash()

    signed = _minimal_spec().__class__(**{**spec.__dict__, "author_sig": "deadbeef", "spec_hash": "stale"})
    h2 = signed.compute_spec_hash()
    assert h1 == h2  # signature and hash fields excluded from canonical payload
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_workflow_spec_canonical_payload_is_deterministic():
    spec = _minimal_spec()
    payload = spec.canonical_payload()
    j1 = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    j2 = json.dumps(spec.canonical_payload(), sort_keys=True, separators=(",", ":"))
    assert j1 == j2


def test_workflow_spec_default_trigger_is_manual():
    spec = _minimal_spec()
    assert len(spec.triggers) == 1
    assert spec.triggers[0].kind == TriggerKind.MANUAL


def test_workflow_spec_retention_days_must_be_positive_or_none():
    _minimal_spec(retention_days=None)
    _minimal_spec(retention_days=30)
    with pytest.raises(WorkflowDefinitionError):
        _minimal_spec(retention_days=0)
    with pytest.raises(WorkflowDefinitionError):
        _minimal_spec(retention_days=-1)


# ---------------------------------------------------------------------------
# WorkflowRun + StageLink
# ---------------------------------------------------------------------------


def test_workflow_run_status_string_coercion():
    run = WorkflowRun(
        run_id="r-1",
        workflow_name="release",
        workflow_ver=1,
        params={},
        status="running",
        started_by_did="did:web:k.example",
    )
    assert run.status == RunStatus.RUNNING


def test_workflow_run_rejects_bad_status():
    with pytest.raises(WorkflowDefinitionError):
        WorkflowRun(
            run_id="r-2",
            workflow_name="release",
            workflow_ver=1,
            params={},
            status="exploded",
            started_by_did="did:web:k.example",
        )


def test_stage_link_rejects_bad_compensate_state():
    with pytest.raises(WorkflowDefinitionError):
        StageLink(
            link_id="l-1",
            run_id="r-1",
            stage_name="lint",
            attempt_number=1,
            idempotency_key="0" * 64,
            actor_did="did:web:k.example",
            actor_sig="deadbeef",
            compensate_state="bogus",
        )


def test_stage_link_attempt_number_one_indexed():
    with pytest.raises(WorkflowDefinitionError):
        StageLink(
            link_id="l-1",
            run_id="r-1",
            stage_name="lint",
            attempt_number=0,
            idempotency_key="0" * 64,
            actor_did="did:web:k.example",
            actor_sig="deadbeef",
        )


# ---------------------------------------------------------------------------
# JSON Schema parity (tolerant if jsonschema isn't installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_workflow_spec_to_dict_validates_against_schema():
    spec = _minimal_spec()
    jsonschema.validate(instance=spec.to_dict(), schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_workflow_run_to_dict_validates_against_schema():
    run = WorkflowRun(
        run_id="r-1",
        workflow_name="release",
        workflow_ver=1,
        params={"branch": "main"},
        status=RunStatus.RUNNING,
        current_stages=("lint",),
        started_by_did="did:web:k.example",
    )
    jsonschema.validate(instance=run.to_dict(), schema=WORKFLOW_RUN_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_workflow_stage_link_to_dict_validates_against_schema():
    link = StageLink(
        link_id="l-1",
        run_id="r-1",
        stage_name="lint",
        attempt_number=1,
        idempotency_key="0" * 64,
        actor_did="did:web:k.example",
        actor_sig="deadbeef",
        gate_outcome=GateOutcome.PASS,
        compensate_state="not_required",
    )
    jsonschema.validate(
        instance=link.to_dict(), schema=WORKFLOW_STAGE_LINK_SCHEMA
    )


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_requires_compensate_so_dataclass_can_construct():
    """Codex round 1 P2: schema MUST require ``compensate`` at the wire
    boundary. Otherwise a writeful ACTION stage (omitting compensate,
    omitting read_only) validates against the schema but fails the
    dataclass eligibility check — split-brain wire contract."""
    spec = _minimal_spec().to_dict()
    minimal_stage = {
        "name": "publish",
        "signal_source": "ci.publish",
        "signal_mode": "action",
        # compensate omitted — schema must reject this
    }
    spec["stages"].append(minimal_stage)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_enforces_noop_idempotent_eligibility():
    """Codex round 2 P2: schema must reject ``compensate=noop_idempotent``
    on writeful ACTION stages (no read_only=True, no consent_collect
    gate) — same rule as Stage.__post_init__."""
    spec = _minimal_spec().to_dict()
    spec["stages"].append(
        {
            "name": "publish",
            "signal_source": "ci.publish",
            "signal_mode": "action",
            "compensate": "noop_idempotent",
            "read_only": False,
            "gate": {"type": "signal_status_ok"},
        }
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_accepts_noop_idempotent_with_consent_gate():
    """The escape hatch: noop_idempotent is also valid when the gate is
    consent_collect, regardless of read_only — design §3.5."""
    spec = _minimal_spec().to_dict()
    spec["stages"].append(
        {
            "name": "ask_user",
            "signal_source": "hooks.consent",
            "signal_mode": "action",
            "compensate": "noop_idempotent",
            "read_only": False,
            "gate": {"type": "consent_collect", "params": {"scope": "publish"}},
        }
    )
    jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_requires_red_team_clear_prompt_pack_constraint():
    spec = _minimal_spec().to_dict()
    spec["stages"][0]["gate"] = {"type": "red_team_clear", "params": {}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_requires_constitutional_boundary_clean_forbidden_modules():
    spec = _minimal_spec().to_dict()
    spec["stages"][0]["gate"] = {
        "type": "constitutional_boundary_clean",
        "params": {},
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_rejects_unknown_gate_type():
    spec = _minimal_spec().to_dict()
    spec["stages"][0]["gate"] = {"type": "custom_callable"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_rejects_cross_kind_edge_fields():
    spec = _minimal_spec().to_dict()
    spec["edges"] = [
        {
            "kind": "sequential",
            "from_stage": "lint",
            "to_stage": "test",
            "condition": "x",  # belongs to BRANCH
        }
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)
