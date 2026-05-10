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
    validate_spec_payload,
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


def test_gate_tests_pass_requires_suite():
    with pytest.raises(WorkflowDefinitionError, match="params.suite"):
        Gate(type="tests_pass")
    with pytest.raises(WorkflowDefinitionError, match="params.suite"):
        Gate(type="tests_pass", params={"suite": "   "})
    Gate(type="tests_pass", params={"suite": "unit"})


def test_gate_ci_green_requires_repo_and_branch():
    with pytest.raises(WorkflowDefinitionError, match="params.repo"):
        Gate(type="ci_green", params={"branch": "main"})
    with pytest.raises(WorkflowDefinitionError, match="params.branch"):
        Gate(type="ci_green", params={"repo": "owner/repo"})
    with pytest.raises(WorkflowDefinitionError, match="owner/repo"):
        Gate(type="ci_green", params={"repo": "repo", "branch": "main"})
    with pytest.raises(WorkflowDefinitionError, match="owner/repo"):
        Gate(type="ci_green", params={"repo": "owner/repo/extra", "branch": "main"})
    with pytest.raises(WorkflowDefinitionError, match="required_checks"):
        Gate(
            type="ci_green",
            params={"repo": "owner/repo", "branch": "main", "required_checks": []},
        )
    with pytest.raises(WorkflowDefinitionError, match="poll_interval_seconds"):
        Gate(
            type="ci_green",
            params={
                "repo": "owner/repo",
                "branch": "main",
                "poll_interval_seconds": True,
            },
        )
    Gate(type="ci_green", params={"repo": "owner/repo", "branch": "main"})


def test_gate_lint_clean_requires_scopes():
    with pytest.raises(WorkflowDefinitionError, match="params.scopes"):
        Gate(type="lint_clean")
    with pytest.raises(WorkflowDefinitionError, match="params.scopes"):
        Gate(type="lint_clean", params={"scopes": []})
    with pytest.raises(WorkflowDefinitionError, match="params.scopes"):
        Gate(type="lint_clean", params={"scopes": [""]})
    Gate(type="lint_clean", params={"scopes": ["kestrel_sovereign/features/workflows"]})


def test_gate_red_team_clear_requires_prompt_pack_constraint():
    with pytest.raises(WorkflowDefinitionError):
        Gate(type="red_team_clear", params={"reviewer_pool": ["did:web:r1"]})
    Gate(
        type="red_team_clear",
        params={"prompt_pack_constraint": "==1.2.0"},
    )


def test_gate_script_requires_all_security_fields():
    """Round-6 P2 (chunk B): script gate is the sandboxed custom
    predicate; all five fields (language, src_hash, signature,
    signing_did, sandbox) are security-load-bearing per design §3.3."""
    full_params = {
        "language": "python",
        "src_hash": "sha256:abcd",
        "signature": "deadbeef",
        "signing_did": "did:web:k.example",
        "sandbox": "compute:firecracker",
    }
    Gate(type="script", params=full_params)  # full set OK
    for missing in full_params:
        partial = {k: v for k, v in full_params.items() if k != missing}
        with pytest.raises(WorkflowDefinitionError):
            Gate(type="script", params=partial)


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


def test_stage_compensate_must_be_valid_source_name():
    """Round 18 P2: a real compensate value is a SourceRegistration
    name; whitespace/invalid chars in it would crash dispatch at cancel
    time, leaving side effects uncompensated. Reserved sentinels
    (noop_idempotent / compensate_record_only) bypass the check."""
    with pytest.raises(WorkflowDefinitionError):
        Stage(
            name="x",
            signal_source="x",
            signal_mode=SignalMode.ACTION,
            compensate="bad name",  # whitespace forbidden
        )
    # Reserved sentinels still accepted on appropriate stages:
    Stage(
        name="x",
        signal_source="x",
        signal_mode=SignalMode.ACTION,
        read_only=True,
        compensate="noop_idempotent",
    )


def test_stage_record_only_only_for_irreversible():
    """Round 13 P2: compensate_record_only is reserved for irreversible
    stages — a reversible stage that declares it would get a record-
    only rollback for a side effect that should have a real compensation
    source."""
    with pytest.raises(WorkflowDefinitionError):
        Stage(
            name="x",
            signal_source="x",
            signal_mode=SignalMode.ACTION,
            irreversible=False,
            compensate="compensate_record_only",
        )


def test_workflow_run_rejects_non_string_optional_ids():
    """Round 17 P2: parent_run_id/scheduler_task_id are Optional[str].
    Non-string values led to schema-invalid or unserializable to_dict
    output."""
    with pytest.raises(WorkflowDefinitionError):
        WorkflowRun(
            run_id="r-1",
            workflow_name="release",
            workflow_ver=1,
            params={},
            status=RunStatus.RUNNING,
            engine_nonce="0" * 32,
            started_by_did="did:web:k.example",
            parent_run_id=42,  # int, not str-or-None
        )
    with pytest.raises(WorkflowDefinitionError):
        WorkflowRun(
            run_id="r-1",
            workflow_name="release",
            workflow_ver=1,
            params={},
            status=RunStatus.RUNNING,
            engine_nonce="0" * 32,
            started_by_did="did:web:k.example",
            scheduler_task_id=object(),  # arbitrary, not str-or-None
        )


def test_stage_link_rejects_non_string_optional_fields():
    """Round 17 P2: signal_id/gate_reason are Optional[str]."""
    with pytest.raises(WorkflowDefinitionError):
        StageLink(
            link_id="l-1",
            run_id="r-1",
            stage_name="lint",
            attempt_number=1,
            idempotency_key="0" * 64,
            actor_did="did:web:k.example",
            actor_sig="deadbeef",
            signal_id=123,
        )
    with pytest.raises(WorkflowDefinitionError):
        StageLink(
            link_id="l-1",
            run_id="r-1",
            stage_name="lint",
            attempt_number=1,
            idempotency_key="0" * 64,
            actor_did="did:web:k.example",
            actor_sig="deadbeef",
            gate_reason=object(),
        )


def test_workflow_run_validates_current_stage_names():
    """Round 13 P2: each name must satisfy _NAME_RE; otherwise to_dict
    emits a value violating WORKFLOW_RUN_SCHEMA's pattern."""
    with pytest.raises(WorkflowDefinitionError):
        WorkflowRun(
            run_id="r-1",
            workflow_name="release",
            workflow_ver=1,
            params={},
            status=RunStatus.RUNNING,
            engine_nonce="0" * 32,
            current_stages=("bad name",),  # whitespace forbidden
            started_by_did="did:web:k.example",
        )


def test_stage_signal_source_accepts_did_bearing_names():
    """Round 10 P2: design's ``agent.<did>`` pattern embeds DIDs
    containing ``:``. _NAME_RE rejected these; signal_source uses the
    wider _SOURCE_NAME_RE."""
    Stage(
        name="ask_agent",
        signal_source="agent.did:web:k.example",
        signal_mode=SignalMode.COGNITION,
        compensate="agent.did:web:k.example.compensate",
    )
    Stage(
        name="ask_pkh",
        signal_source="agent.did:pkh:eip155:1:0xabcdef",
        signal_mode=SignalMode.COGNITION,
        compensate="agent.did:pkh:eip155:1:0xabcdef.compensate",
    )


def test_stage_irreversible_requires_record_only_compensate():
    """Round 9 P2: design §3.5 — irreversible stages must use
    compensate_record_only. Any other compensate gives the runner
    conflicting cancellation instructions."""
    with pytest.raises(WorkflowDefinitionError):
        Stage(
            name="ship",
            signal_source="ci.publish",
            signal_mode=SignalMode.ACTION,
            irreversible=True,
            compensate="ci.publish.compensate",  # reversible compensate
        )
    # OK with the right compensate:
    Stage(
        name="ship",
        signal_source="ci.publish",
        signal_mode=SignalMode.ACTION,
        irreversible=True,
        compensate="compensate_record_only",
    )


def test_workflow_run_rejects_iso_string_for_timestamp():
    """Round 9 P2: schema permits string timestamps but the dataclass
    is the canonical Python view (datetime). Storage adapters MUST
    parse before constructing — otherwise to_dict() crashes calling
    .isoformat() on a string."""
    with pytest.raises(WorkflowDefinitionError):
        WorkflowRun(
            run_id="r-1",
            workflow_name="release",
            workflow_ver=1,
            params={},
            status=RunStatus.RUNNING,
            engine_nonce="0" * 32,
            started_by_did="did:web:k.example",
            cancel_barrier_at="2026-05-09T12:00:00+00:00",
        )


def test_stage_link_rejects_iso_string_for_occurred_at():
    with pytest.raises(WorkflowDefinitionError):
        StageLink(
            link_id="l-1",
            run_id="r-1",
            stage_name="lint",
            attempt_number=1,
            idempotency_key="0" * 64,
            actor_did="did:web:k.example",
            actor_sig="deadbeef",
            occurred_at="2026-05-09T12:00:00+00:00",
        )


def test_stage_from_dict_requires_compensate():
    """Round 8 P2: signed-spec integrity. Defaulting ``compensate`` in
    from_dict lets a wire form omit it and re-hash to a canonical
    signed form. Schema requires it; from_dict must too."""
    with pytest.raises(WorkflowDefinitionError):
        Stage.from_dict(
            {
                "name": "x",
                "signal_source": "x",
                "signal_mode": "action",
                "read_only": True,
                # compensate omitted
            }
        )


def test_stage_from_dict_rejects_string_for_forbidden_modules():
    """Codex round 3 P2: ``tuple('features.security')`` expands a
    string into single chars, each a non-empty str; the all-isinstance
    check then passes and the constitutional-boundary scan would scope
    to nothing meaningful. ``Stage.from_dict`` must pass the raw value
    through so __post_init__'s isinstance(list/tuple) guard rejects it."""
    with pytest.raises(WorkflowDefinitionError):
        Stage.from_dict(
            {
                "name": "publish",
                "signal_source": "ci.publish",
                "signal_mode": "action",
                "compensate": "noop_idempotent",
                "read_only": True,
                "forbidden_modules": "features.security",  # string, not list
            }
        )


def test_edge_from_dict_rejects_string_for_parallel_stages():
    """Same trap on ``Edge(kind=parallel)`` — ``tuple('abc')`` would
    parse 'abc' as three single-char stages."""
    with pytest.raises(WorkflowDefinitionError):
        Edge.from_dict(
            {
                "kind": "parallel",
                "from_stage": "fanout",
                "stages": "ab",  # string, not list
                "join_strategy": "all",
            }
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
            engine_nonce="0" * 32,
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
            engine_nonce="0" * 32,
            started_by_did="did:web:k.example",
        )


def test_workflow_run_requires_hex_engine_nonce():
    with pytest.raises(WorkflowDefinitionError):
        WorkflowRun(
            run_id="r-1",
            workflow_name="release",
            workflow_ver=1,
            params={},
            status=RunStatus.RUNNING,
            engine_nonce="not-a-hex-nonce",
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


def test_stage_link_rejects_malformed_idempotency_key():
    """Codex round 4 P2: dedupe invariant requires hex sha256 digest."""
    for bad in ("retry", "abcd", "x" * 64, "0" * 63, "0" * 65, 12345):
        with pytest.raises(WorkflowDefinitionError):
            StageLink(
                link_id="l-1",
                run_id="r-1",
                stage_name="lint",
                attempt_number=1,
                idempotency_key=bad,
                actor_did="did:web:k.example",
                actor_sig="deadbeef",
            )


def test_workflow_spec_from_dict_rejects_string_version():
    """Codex round 4 P2 (signed-spec integrity): a wire ``version: "1"``
    must NOT be silently coerced to int 1, or its compute_spec_hash
    would match a signature whose canonical signed form was the int."""
    spec = _minimal_spec().to_dict()
    spec["version"] = "1"  # string, not int
    with pytest.raises(WorkflowDefinitionError):
        WorkflowSpec.from_dict(spec)


def test_workflow_spec_from_dict_rejects_int_name():
    spec = _minimal_spec().to_dict()
    spec["name"] = 42
    with pytest.raises(WorkflowDefinitionError):
        WorkflowSpec.from_dict(spec)


def test_workflow_spec_from_dict_rejects_falsy_wrong_type_params_schema():
    """Codex round 5 P2: ``data.get('params_schema') or {}`` rewrote a
    present-but-falsy ``params_schema: []`` into the canonical empty
    dict, masking malformed wire forms. ``_present_or`` keeps the value
    so __post_init__'s isinstance(Mapping) guard rejects it."""
    spec = _minimal_spec().to_dict()
    spec["params_schema"] = []  # list, not object
    with pytest.raises(WorkflowDefinitionError):
        WorkflowSpec.from_dict(spec)


def test_stage_from_dict_rejects_falsy_wrong_type_params():
    with pytest.raises(WorkflowDefinitionError):
        Stage.from_dict(
            {
                "name": "x",
                "signal_source": "x",
                "signal_mode": "action",
                "compensate": "noop_idempotent",
                "read_only": True,
                "params": [],  # list, not object
            }
        )


def test_stage_from_dict_rejects_empty_string_for_forbidden_modules():
    with pytest.raises(WorkflowDefinitionError):
        Stage.from_dict(
            {
                "name": "x",
                "signal_source": "x",
                "signal_mode": "action",
                "compensate": "noop_idempotent",
                "read_only": True,
                "forbidden_modules": "",  # falsy non-list
            }
        )


def test_trigger_rejects_non_string_non_enum_kind():
    """Round 5 P2: a JSON number for ``kind`` would fall through silently
    and crash later in ``to_dict()`` at ``.value``."""
    with pytest.raises(WorkflowDefinitionError):
        Trigger(kind=42)


def test_workflow_run_rejects_non_string_non_enum_status():
    with pytest.raises(WorkflowDefinitionError):
        WorkflowRun(
            run_id="r-1",
            workflow_name="release",
            workflow_ver=1,
            params={},
            status=42,
            engine_nonce="0" * 32,
            started_by_did="did:web:k.example",
        )


def test_stage_link_rejects_non_string_non_enum_gate_outcome():
    with pytest.raises(WorkflowDefinitionError):
        StageLink(
            link_id="l-1",
            run_id="r-1",
            stage_name="lint",
            attempt_number=1,
            idempotency_key="0" * 64,
            actor_did="did:web:k.example",
            actor_sig="deadbeef",
            gate_outcome=42,
        )


def test_stage_from_dict_rejects_falsy_wrong_type_gate():
    """Round 6 P2: ``data.get('gate') or default`` rewrote a wire
    ``gate: []`` (list, not object) into the canonical default-gate
    form, masking malformed wire payloads."""
    with pytest.raises(WorkflowDefinitionError):
        Stage.from_dict(
            {
                "name": "x",
                "signal_source": "x",
                "signal_mode": "action",
                "compensate": "noop_idempotent",
                "read_only": True,
                "gate": [],  # list, not object
            }
        )


def test_gate_constitutional_boundary_clean_requires_non_empty_forbidden():
    """Round 6 P2: ``all([]) is True``; an explicit empty list silently
    passed dataclass validation while the schema rejects via
    ``minItems: 1``. A boundary-clean gate that scopes to nothing scans
    nothing — worse than no gate."""
    with pytest.raises(WorkflowDefinitionError):
        Gate(
            type="constitutional_boundary_clean",
            params={"forbidden_modules": []},
        )


def test_trigger_manual_rejects_cron_expression():
    """Round 6 P2: ``Trigger(kind=manual, cron_expression='...')`` was
    accepted by the dataclass but rejected by the schema oneOf —
    schema/model drift on signed wire forms."""
    with pytest.raises(WorkflowDefinitionError):
        Trigger(kind=TriggerKind.MANUAL, cron_expression="0 4 * * *")


def test_trigger_cron_rejects_signal_source():
    with pytest.raises(WorkflowDefinitionError):
        Trigger(
            kind=TriggerKind.CRON,
            cron_expression="0 4 * * *",
            signal_source="github.pr_opened",
        )


def test_trigger_signal_source_rejects_cron_expression():
    with pytest.raises(WorkflowDefinitionError):
        Trigger(
            kind=TriggerKind.SIGNAL_SOURCE,
            signal_source="github.pr_opened",
            cron_expression="0 4 * * *",
        )


def test_workflow_spec_rejects_bool_for_version():
    """Round 7 P2: bool is subclass of int. ``isinstance(True, int)`` is
    True, so ``version=True`` slipped through and ``to_dict()`` emitted
    a JSON boolean where the schema requires integer."""
    with pytest.raises(WorkflowDefinitionError):
        WorkflowSpec(
            name="r",
            version=True,
            stages=[_action_stage(name="lint")],
        )


def test_workflow_spec_rejects_bool_for_retention_days():
    with pytest.raises(WorkflowDefinitionError):
        _minimal_spec(retention_days=True)


def test_workflow_run_rejects_bool_for_workflow_ver():
    with pytest.raises(WorkflowDefinitionError):
        WorkflowRun(
            run_id="r-1",
            workflow_name="release",
            workflow_ver=True,
            params={},
            status=RunStatus.RUNNING,
            engine_nonce="0" * 32,
            started_by_did="did:web:k.example",
        )


def test_stage_link_rejects_bool_for_attempt_number():
    with pytest.raises(WorkflowDefinitionError):
        StageLink(
            link_id="l-1",
            run_id="r-1",
            stage_name="lint",
            attempt_number=True,
            idempotency_key="0" * 64,
            actor_did="did:web:k.example",
            actor_sig="deadbeef",
        )


def test_edge_subworkflow_rejects_bool_for_version():
    with pytest.raises(WorkflowDefinitionError):
        Edge(
            kind=EdgeKind.SUBWORKFLOW,
            from_stage="a",
            subworkflow_name="child",
            subworkflow_version=True,
        )


def test_stage_params_are_immutable():
    """Round 15 P2: Stage holds a frozen view of params so post-sign
    mutation can't drift the live payload from the signed canonical
    hash."""
    stage = _action_stage(params={"branch": "main", "tags": ["a", "b"]})
    with pytest.raises(TypeError):
        stage.params["branch"] = "stolen"  # type: ignore[index]
    with pytest.raises(TypeError):
        stage.params["new"] = 1  # type: ignore[index]
    # Nested list became tuple
    assert isinstance(stage.params["tags"], tuple)


def test_workflow_spec_params_schema_immutable():
    spec = _minimal_spec(params_schema={"type": "object", "required": ["x"]})
    with pytest.raises(TypeError):
        spec.params_schema["type"] = "array"  # type: ignore[index]


def test_workflow_spec_compute_spec_hash_with_nested_params():
    """Round 16 P2: shallow dict() left MappingProxyType in output,
    breaking json.dumps in compute_spec_hash. Recursive _thaw_value
    fixes it for nested params."""
    spec = WorkflowSpec(
        name="r",
        version=1,
        stages=[
            Stage(
                name="lint",
                signal_source="x",
                signal_mode=SignalMode.ACTION,
                read_only=True,
                params={"branch": {"name": "main", "tags": ["a", "b"]}},
            )
        ],
    )
    # Must not raise — exercises json.dumps over the canonical payload.
    h = spec.compute_spec_hash()
    assert len(h) == 64
    # And to_dict's JSON form is a plain dict tree.
    payload = spec.to_dict()
    json.dumps(payload)


def test_workflow_spec_post_construction_mutation_does_not_change_hash():
    """Constructing then attempting mutation must NOT change
    compute_spec_hash — the freeze prevents it from succeeding at all,
    but the hash test pins the broader invariant."""
    spec = _minimal_spec(params_schema={"x": 1})
    h1 = spec.compute_spec_hash()
    with pytest.raises(TypeError):
        spec.params_schema["x"] = 2  # type: ignore[index]
    h2 = spec.compute_spec_hash()
    assert h1 == h2


def test_workflow_spec_empty_triggers_round_trip_stable():
    """Round 12 P2: an empty triggers list constructs a spec whose
    hash matches a from_dict-loaded spec of the same wire form.
    Otherwise persisting a signed spec with empty triggers and
    reloading would change the canonical hash, breaking signature
    verification."""
    direct = WorkflowSpec(
        name="r",
        version=1,
        stages=[_action_stage(name="lint")],
        triggers=[],  # explicit empty
    )
    loaded = WorkflowSpec.from_dict(direct.to_dict())
    assert direct.compute_spec_hash() == loaded.compute_spec_hash()
    assert direct.triggers == loaded.triggers
    assert direct.triggers[0].kind == TriggerKind.MANUAL


def test_workflow_spec_from_dict_rejects_explicit_null_triggers():
    """Round 7 P2: an explicit ``triggers: null`` was silently rewritten
    to the manual-default trigger, letting a schema-invalid wire form
    construct the same canonical payload as a real manual-trigger spec."""
    spec = _minimal_spec().to_dict()
    spec["triggers"] = None
    with pytest.raises(WorkflowDefinitionError):
        WorkflowSpec.from_dict(spec)


def test_gate_from_dict_rejects_int_type():
    """Mirror of WorkflowSpec/Stage round-4 hardening at the gate layer."""
    with pytest.raises(WorkflowDefinitionError):
        Gate.from_dict({"type": 42})


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
            engine_nonce="0" * 32,
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
def test_schema_requires_tests_pass_suite():
    spec = _minimal_spec().to_dict()
    spec["stages"][0]["gate"] = {"type": "tests_pass", "params": {}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_requires_ci_green_repo_and_branch():
    spec = _minimal_spec().to_dict()
    spec["stages"][0]["gate"] = {"type": "ci_green", "params": {"repo": "owner/repo"}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)

    spec["stages"][0]["gate"] = {
        "type": "ci_green",
        "params": {"repo": "owner/repo/extra", "branch": "main"},
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)

    spec["stages"][0]["gate"] = {
        "type": "ci_green",
        "params": {"repo": "owner/repo", "branch": "main", "required_checks": []},
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_requires_lint_clean_scopes():
    spec = _minimal_spec().to_dict()
    spec["stages"][0]["gate"] = {"type": "lint_clean", "params": {"scopes": []}}
    with pytest.raises(jsonschema.ValidationError):
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
def test_schema_requires_all_script_gate_fields():
    """Round-6 P2 (chunk B): schema mirror — wire form must reject a
    script gate missing any of language/src_hash/signature/signing_did/sandbox."""
    spec = _minimal_spec().to_dict()
    spec["stages"][0]["gate"] = {
        "type": "script",
        "params": {
            "language": "python",
            "src_hash": "sha256:abcd",
            # signature, signing_did, sandbox missing
        },
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_validate_spec_payload_catches_duplicate_stage_names():
    """Round-4 P2: graph invariants (duplicate names, dangling edge
    references) aren't expressible in pure draft-2020-12. The
    ``validate_spec_payload`` helper layers them on top so schema-
    valid payloads also construct successfully."""
    spec = _minimal_spec().to_dict()
    spec["stages"].append(spec["stages"][0])  # duplicate by deep equality
    with pytest.raises(ValueError):
        validate_spec_payload(spec)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_validate_spec_payload_catches_dangling_edge_reference():
    spec = _minimal_spec().to_dict()
    spec["edges"] = [
        {"kind": "sequential", "from_stage": "lint", "to_stage": "ghost"}
    ]
    with pytest.raises(ValueError):
        validate_spec_payload(spec)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_validate_spec_payload_passes_canonical_form():
    spec = _minimal_spec().to_dict()
    validate_spec_payload(spec)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_rejects_whitespace_only_branch_condition():
    """Round 11 P2: same pattern as prompt_pack_constraint —
    Edge.__post_init__ rejects whitespace-only via .strip(); schema
    must too."""
    spec = _minimal_spec().to_dict()
    # Need three stages so the branch has both targets declared.
    spec["stages"].extend(
        [
            {
                "name": "branch_t",
                "signal_source": "x.t",
                "signal_mode": "action",
                "compensate": "noop_idempotent",
                "read_only": True,
            },
            {
                "name": "branch_f",
                "signal_source": "x.f",
                "signal_mode": "action",
                "compensate": "noop_idempotent",
                "read_only": True,
            },
        ]
    )
    spec["edges"] = [
        {
            "kind": "branch",
            "from_stage": "lint",
            "condition": "   ",
            "true_stage": "branch_t",
            "false_stage": "branch_f",
        }
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_rejects_whitespace_only_prompt_pack_constraint():
    """Round 10 P2: schema/dataclass parity. Schema's minLength counts
    whitespace; the dataclass's .strip() check rejects "   "."""
    spec = _minimal_spec().to_dict()
    spec["stages"][0]["gate"] = {
        "type": "red_team_clear",
        "params": {"prompt_pack_constraint": "   "},
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_accepts_did_bearing_signal_source():
    spec = _minimal_spec().to_dict()
    spec["stages"][0]["signal_source"] = "agent.did:web:k.example"
    jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_record_only_only_for_irreversible():
    """Round 14 P2: schema mirror of Stage's bidirectional invariant.
    A reversible stage with compensate=compensate_record_only must be
    rejected at the wire boundary."""
    spec = _minimal_spec().to_dict()
    spec["stages"].append(
        {
            "name": "ship",
            "signal_source": "ci.publish",
            "signal_mode": "action",
            "compensate": "compensate_record_only",
            "irreversible": False,
        }
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=spec, schema=WORKFLOW_SPEC_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_stage_link_rejects_empty_actor_fields():
    """Round 14 P2: identity fields must be non-empty at the wire too."""
    link = {
        "link_id": "l-1",
        "run_id": "r-1",
        "stage_name": "lint",
        "attempt_number": 1,
        "idempotency_key": "0" * 64,
        "actor_did": "",
        "actor_sig": "deadbeef",
        "post_cancel": False,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=link, schema=WORKFLOW_STAGE_LINK_SCHEMA)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_schema_irreversible_stage_requires_record_only_compensate():
    spec = _minimal_spec().to_dict()
    spec["stages"].append(
        {
            "name": "ship",
            "signal_source": "ci.publish",
            "signal_mode": "action",
            "compensate": "ci.publish.compensate",
            "irreversible": True,
        }
    )
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
