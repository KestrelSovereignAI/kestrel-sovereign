"""JSON Schema for the workflow domain model.

The schema is the wire-format contract for ``workflow_define(spec)`` and
the runtime ``params_schema`` validator. It mirrors :mod:`models` exactly
— a drift between the two is a bug; the round-trip test in
``tests/unit/features/workflows/test_models.py`` enforces parity.

We use draft-2020-12 because that's what every other Kestrel surface
uses (signal source registrations, hooks, identity packages). No
``$schema`` URI is hard-coded into the *value* of WorkflowSpec — it's
only declared at the top level of the SCHEMA itself. Validators (Phase
1's ``WorkflowDefinitionValidator``) pin to draft-2020-12 explicitly.
"""

from __future__ import annotations

from typing import Any

from kestrel_sdk.signals import SignalMode

from kestrel_sovereign.features.workflows.models import (
    BUILT_IN_GATE_TYPES,
    EdgeKind,
    GateOutcome,
    RunStatus,
    TriggerKind,
)


# Hash form for spec_hash / actor_sig payloads (lower-hex sha256, 64 chars).
_HASH_PATTERN = r"^[0-9a-f]{64}$"

# Lenient identifier pattern matching ``models._NAME_RE``.
_NAME_PATTERN = r"^[a-zA-Z_][a-zA-Z0-9_.\-]*$"

# Source-reference pattern matching ``models._SOURCE_NAME_RE`` — wider
# than ``_NAME_PATTERN`` because the design's ``agent.<did>`` source
# names embed DIDs containing ``:``, ``@``, ``%``, ``+``.
_SOURCE_NAME_PATTERN = r"^[a-zA-Z0-9_][a-zA-Z0-9_.\-:@~+%/=]*$"
_GITHUB_REPO_PATTERN = r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
_DID_PATTERN = r"^did:[a-z0-9]+:\S+$"
_SCRIPT_HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"

# Pattern requiring at least one non-whitespace char anywhere in the
# value. Schema's ``minLength: 1`` counts whitespace, but the dataclass
# rejects whitespace-only via ``.strip()`` — round-10 P2 schema/model
# parity. Use this wherever the dataclass strips before checking length.
_NON_WHITESPACE_PATTERN = r"\S"
_TRIMMED_NON_EMPTY_PATTERN = r"^(?!\s)(?!.*\s$)(?!.*[\r\n])[\s\S]+$"


def _gate_schema() -> dict[str, Any]:
    # Codex round 2 P2: mirror the dataclass's per-type required-params
    # rules into the schema so the wire contract doesn't accept gates
    # that ``Gate.__post_init__`` would reject. Gate-specific fields
    # are load-bearing because signed specs must validate the same way
    # whether they enter through dataclasses or JSON Schema.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["type"],
        "properties": {
            "type": {"type": "string", "enum": sorted(BUILT_IN_GATE_TYPES)},
            "params": {"type": "object"},
        },
        "allOf": [
            {
                "if": {
                    "properties": {"type": {"const": "tests_pass"}},
                    "required": ["type"],
                },
                "then": {
                    "required": ["params"],
                    "properties": {
                        "params": {
                            "type": "object",
                            "required": ["suite"],
                            "properties": {
                                "suite": {
                                    "type": "string",
                                    "pattern": _NON_WHITESPACE_PATTERN,
                                },
                            },
                        },
                    },
                },
            },
            {
                "if": {
                    "properties": {"type": {"const": "lint_clean"}},
                    "required": ["type"],
                },
                "then": {
                    "required": ["params"],
                    "properties": {
                        "params": {
                            "type": "object",
                            "required": ["scopes"],
                            "properties": {
                                "scopes": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "pattern": _NON_WHITESPACE_PATTERN,
                                    },
                                    "minItems": 1,
                                },
                            },
                        },
                    },
                },
            },
            {
                "if": {
                    "properties": {"type": {"const": "ci_green"}},
                    "required": ["type"],
                },
                "then": {
                    "required": ["params"],
                    "properties": {
                        "params": {
                            "type": "object",
                            "required": ["repo", "branch"],
                            "properties": {
                                "repo": {
                                    "type": "string",
                                    "pattern": _GITHUB_REPO_PATTERN,
                                },
                                "branch": {
                                    "type": "string",
                                    "pattern": _NON_WHITESPACE_PATTERN,
                                },
                                "required_checks": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "pattern": _NON_WHITESPACE_PATTERN,
                                    },
                                    "minItems": 1,
                                },
                                "poll_interval_seconds": {
                                    "type": "integer",
                                    "minimum": 1,
                                },
                                "max_wait_seconds": {
                                    "type": "integer",
                                    "minimum": 1,
                                },
                            },
                        },
                    },
                },
            },
            {
                "if": {
                    "properties": {"type": {"const": "consent_collect"}},
                    "required": ["type"],
                },
                "then": {
                    "required": ["params"],
                    "properties": {
                        "params": {
                            "type": "object",
                            "required": ["scope"],
                            "properties": {
                                "scope": {
                                    "type": "string",
                                    "pattern": _TRIMMED_NON_EMPTY_PATTERN,
                                },
                            },
                        },
                    },
                },
            },
            {
                "if": {
                    "properties": {"type": {"const": "signature_collected"}},
                    "required": ["type"],
                },
                "then": {
                    "required": ["params"],
                    "properties": {
                        "params": {
                            "type": "object",
                            "required": ["did"],
                            "properties": {
                                "did": {
                                    "type": "string",
                                    "pattern": _DID_PATTERN,
                                },
                            },
                        },
                    },
                },
            },
            {
                "if": {
                    "properties": {"type": {"const": "council_approve"}},
                    "required": ["type"],
                },
                "then": {
                    "required": ["params"],
                    "properties": {
                        "params": {
                            "type": "object",
                            "required": ["quorum", "timeout"],
                            "properties": {
                                "quorum": {
                                    "type": "integer",
                                    "minimum": 1,
                                },
                                "timeout": {
                                    "type": "integer",
                                    "minimum": 1,
                                },
                            },
                        },
                    },
                },
            },
            {
                "if": {
                    "properties": {"type": {"const": "red_team_clear"}},
                    "required": ["type"],
                },
                "then": {
                    "required": ["params"],
                    "properties": {
                        "params": {
                            "type": "object",
                            "required": ["prompt_pack_constraint"],
                            "properties": {
                                "prompt_pack_constraint": {
                                    "type": "string",
                                    "minLength": 1,
                                    # Reject whitespace-only values so
                                    # the schema matches Gate's
                                    # ``not constraint.strip()`` check.
                                    "pattern": _NON_WHITESPACE_PATTERN,
                                },
                            },
                        },
                    },
                },
            },
            {
                "if": {
                    "properties": {
                        "type": {"const": "constitutional_boundary_clean"}
                    },
                    "required": ["type"],
                },
                "then": {
                    "required": ["params"],
                    "properties": {
                        "params": {
                            "type": "object",
                            "required": ["forbidden_modules"],
                            "properties": {
                                "forbidden_modules": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                            },
                        },
                    },
                },
            },
            # Codex round 6 P2 (chunk B): script gate is the sandboxed
            # custom predicate; all five identity/integrity fields are
            # security-load-bearing per design §3.3.
            {
                "if": {
                    "properties": {"type": {"const": "script"}},
                    "required": ["type"],
                },
                "then": {
                    "required": ["params"],
                    "properties": {
                        "params": {
                            "type": "object",
                            "required": [
                                "language",
                                "src_hash",
                                "signature",
                                "signing_did",
                                "sandbox",
                            ],
                            "properties": {
                                "language": {
                                    "type": "string",
                                    "enum": ["bash", "python"],
                                },
                                "src_hash": {
                                    "type": "string",
                                    "pattern": _SCRIPT_HASH_PATTERN,
                                },
                                "signature": {
                                    "type": "string",
                                    "pattern": _NON_WHITESPACE_PATTERN,
                                },
                                "signing_did": {
                                    "type": "string",
                                    "pattern": _DID_PATTERN,
                                },
                                "sandbox": {
                                    "type": "string",
                                    "pattern": _NON_WHITESPACE_PATTERN,
                                },
                            },
                        },
                    },
                },
            },
        ],
    }


def _stage_schema() -> dict[str, Any]:
    # ``compensate`` is REQUIRED at the wire-format boundary even though
    # ``Stage`` itself has a Python-side default of ``noop_idempotent``.
    # Codex round 1 P2 catch: if the schema let it default, a client
    # could submit a writeful ACTION stage (no ``read_only=True``,
    # default ``compensate=noop_idempotent``) which validates against
    # the schema but the dataclass rejects in __post_init__. Forcing
    # the field at the wire boundary keeps schema parity with the
    # dataclass — what validates here also constructs in Python.
    #
    # Codex round 2 P2: ``noop_idempotent`` eligibility (design §3.5)
    # is also enforced here via if/then/anyOf so the schema rejects the
    # same ineligible payloads that the dataclass would reject.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "signal_source", "signal_mode", "compensate"],
        "properties": {
            "name": {"type": "string", "pattern": _NAME_PATTERN},
            "signal_source": {
                "type": "string",
                # Source references use the wider source-name vocabulary
                # so DID-bearing patterns like ``agent.did:web:k.example``
                # validate at the wire boundary too (round 10 P2).
                "pattern": _SOURCE_NAME_PATTERN,
            },
            "signal_mode": {
                "type": "string",
                "enum": [m.value for m in SignalMode],
            },
            "params": {"type": "object"},
            "gate": _gate_schema(),
            "compensate": {
                # Codex round 18 P2: a real compensation must satisfy
                # the source-name vocabulary so the runner can dispatch
                # to it at cancellation. Two reserved sentinels
                # (``noop_idempotent``, ``compensate_record_only``)
                # bypass that — they aren't dispatched as signals;
                # the runner handles them inline. ``anyOf`` (not
                # ``oneOf``) so the sentinels still validate even
                # though they also match the source-name pattern.
                "type": "string",
                "anyOf": [
                    {"const": "noop_idempotent"},
                    {"const": "compensate_record_only"},
                    {"pattern": _SOURCE_NAME_PATTERN},
                ],
            },
            "forbidden_modules": {
                "type": "array",
                "items": {"type": "string"},
            },
            "irreversible": {"type": "boolean"},
            "non_deterministic": {"type": "boolean"},
            "read_only": {"type": "boolean"},
        },
        "allOf": [
            {
                "if": {
                    "properties": {
                        "compensate": {"const": "noop_idempotent"},
                    },
                    "required": ["compensate"],
                },
                "then": {
                    "anyOf": [
                        {
                            "properties": {
                                "signal_mode": {"const": "action"},
                                "read_only": {"const": True},
                            },
                            "required": ["signal_mode", "read_only"],
                        },
                        {
                            "properties": {
                                "gate": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "consent_collect"},
                                    },
                                    "required": ["type"],
                                },
                            },
                            "required": ["gate"],
                        },
                    ],
                },
            },
            # Codex round 9 P2: irreversible stages must use
            # compensate_record_only (design §3.5).
            {
                "if": {
                    "properties": {"irreversible": {"const": True}},
                    "required": ["irreversible"],
                },
                "then": {
                    "properties": {
                        "compensate": {"const": "compensate_record_only"},
                    },
                    "required": ["compensate"],
                },
            },
            # Codex round 14 P2: the converse — compensate_record_only
            # is reserved for irreversible stages. Mirror Stage's
            # bidirectional irreversible↔record_only invariant.
            {
                "if": {
                    "properties": {
                        "compensate": {"const": "compensate_record_only"},
                    },
                    "required": ["compensate"],
                },
                "then": {
                    "properties": {"irreversible": {"const": True}},
                    "required": ["irreversible"],
                },
            },
        ],
    }


def _edge_schema() -> dict[str, Any]:
    """Edges use ``oneOf`` so only the relevant fields are populated per
    kind — mirrors ``Edge._reject_unrelated_fields``."""

    base = {
        "from_stage": {"type": "string", "pattern": _NAME_PATTERN},
        "params": {"type": "object"},
    }
    return {
        "type": "object",
        "oneOf": [
            {
                "additionalProperties": False,
                "required": ["kind", "from_stage", "to_stage"],
                "properties": {
                    **base,
                    "kind": {"const": EdgeKind.SEQUENTIAL.value},
                    "to_stage": {"type": "string", "pattern": _NAME_PATTERN},
                },
            },
            {
                "additionalProperties": False,
                "required": [
                    "kind",
                    "from_stage",
                    "condition",
                    "true_stage",
                    "false_stage",
                ],
                "properties": {
                    **base,
                    "kind": {"const": EdgeKind.BRANCH.value},
                    "condition": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": _NON_WHITESPACE_PATTERN,
                    },
                    "true_stage": {
                        "type": "string",
                        "pattern": _NAME_PATTERN,
                    },
                    "false_stage": {
                        "type": "string",
                        "pattern": _NAME_PATTERN,
                    },
                },
            },
            {
                "additionalProperties": False,
                "required": [
                    "kind",
                    "from_stage",
                    "stages",
                    "join_strategy",
                ],
                "properties": {
                    **base,
                    "kind": {"const": EdgeKind.PARALLEL.value},
                    "stages": {
                        "type": "array",
                        "minItems": 2,
                        "items": {
                            "type": "string",
                            "pattern": _NAME_PATTERN,
                        },
                    },
                    "join_strategy": {
                        "type": "string",
                        "enum": ["all", "any", "first_success"],
                    },
                },
            },
            {
                "additionalProperties": False,
                "required": [
                    "kind",
                    "from_stage",
                    "subworkflow_name",
                    "subworkflow_version",
                ],
                "properties": {
                    **base,
                    "kind": {"const": EdgeKind.SUBWORKFLOW.value},
                    "subworkflow_name": {
                        "type": "string",
                        "pattern": _NAME_PATTERN,
                    },
                    "subworkflow_version": {
                        "type": "integer",
                        "minimum": 1,
                    },
                },
            },
        ],
    }


def _trigger_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "oneOf": [
            {
                "additionalProperties": False,
                "required": ["kind"],
                "properties": {
                    "kind": {"const": TriggerKind.MANUAL.value},
                    "params": {"type": "object"},
                },
            },
            {
                "additionalProperties": False,
                "required": ["kind", "cron_expression"],
                "properties": {
                    "kind": {"const": TriggerKind.CRON.value},
                    "cron_expression": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": _NON_WHITESPACE_PATTERN,
                    },
                    "params": {"type": "object"},
                },
            },
            {
                "additionalProperties": False,
                "required": ["kind", "signal_source"],
                "properties": {
                    "kind": {"const": TriggerKind.SIGNAL_SOURCE.value},
                    "signal_source": {
                        "type": "string",
                        "pattern": _SOURCE_NAME_PATTERN,
                    },
                    "params": {"type": "object"},
                },
            },
        ],
    }


WORKFLOW_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WorkflowSpec",
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "version", "stages"],
    "properties": {
        "name": {"type": "string", "pattern": _NAME_PATTERN},
        "version": {"type": "integer", "minimum": 1},
        "stages": {
            "type": "array",
            "minItems": 1,
            "items": _stage_schema(),
        },
        "edges": {"type": "array", "items": _edge_schema()},
        "triggers": {"type": "array", "items": _trigger_schema()},
        "params_schema": {"type": "object"},
        "retention_days": {
            "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}],
        },
        "author_did": {"type": "string"},
        "author_sig": {"type": "string"},
        "spec_hash": {"type": "string"},
    },
}


WORKFLOW_RUN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WorkflowRun",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "run_id",
        "workflow_name",
        "workflow_ver",
        "params",
        "status",
        "engine_nonce",
    ],
    "properties": {
        "run_id": {"type": "string", "minLength": 1},
        "workflow_name": {"type": "string", "pattern": _NAME_PATTERN},
        "workflow_ver": {"type": "integer", "minimum": 1},
        "params": {"type": "object"},
        "status": {"type": "string", "enum": [s.value for s in RunStatus]},
        "engine_nonce": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
        "current_stages": {
            "type": "array",
            "items": {"type": "string", "pattern": _NAME_PATTERN},
        },
        "parent_run_id": {"type": ["string", "null"]},
        "cancel_barrier_at": {"type": ["string", "null"]},
        "started_by_did": {"type": "string"},
        "scheduler_task_id": {"type": ["string", "null"]},
        "signature_post_revocation": {"type": "boolean"},
        "started_at": {"type": ["string", "null"]},
        "finished_at": {"type": ["string", "null"]},
        "deleted_at": {"type": ["string", "null"]},
    },
}


WORKFLOW_STAGE_LINK_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WorkflowStageLink",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "link_id",
        "run_id",
        "stage_name",
        "attempt_number",
        "idempotency_key",
        "actor_did",
        "actor_sig",
    ],
    "properties": {
        "link_id": {"type": "string", "minLength": 1},
        "run_id": {"type": "string", "minLength": 1},
        "stage_name": {"type": "string", "pattern": _NAME_PATTERN},
        "attempt_number": {"type": "integer", "minimum": 1},
        "signal_id": {"type": ["string", "null"]},
        "idempotency_key": {"type": "string", "pattern": _HASH_PATTERN},
        "gate_outcome": {
            "anyOf": [
                {"type": "string", "enum": [o.value for o in GateOutcome]},
                {"type": "null"},
            ],
        },
        "gate_reason": {"type": ["string", "null"]},
        "compensate_state": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "not_required",
                        "pending",
                        "complete",
                        "record_only",
                        "failed",
                    ],
                },
                {"type": "null"},
            ],
        },
        "post_cancel": {"type": "boolean"},
        # Codex round 14 P2: dataclass requires non-empty actor identity.
        # An unsigned/identity-less transition would pass schema before
        # failing later or persisting in a schema-only path.
        "actor_did": {"type": "string", "minLength": 1},
        "actor_sig": {"type": "string", "minLength": 1},
        "occurred_at": {"type": ["string", "null"]},
    },
}


def validate_spec_payload(payload: Any) -> None:
    """Run draft-2020-12 schema validation AND the graph invariants
    that JSON Schema alone can't express.

    Codex round-4 P2: stage-name uniqueness and edge→stage references
    are correlation rules across separate arrays — draft-2020-12 has
    no native syntax for them. ``WORKFLOW_SPEC_SCHEMA`` checks
    everything that's expressible declaratively; this helper layers
    on the same graph-level checks ``WorkflowSpec.__post_init__``
    enforces, so schema-valid payloads also construct successfully.

    Callers SHOULD use this rather than ``jsonschema.validate(payload,
    WORKFLOW_SPEC_SCHEMA)`` directly. Raises:

    - ``jsonschema.ValidationError`` from the schema layer.
    - ``ValueError`` from the graph layer (so callers can disambiguate).

    ``jsonschema`` is a runtime dependency of kestrel-sovereign (added
    for this surface specifically), so the import is unconditional.
    """
    import jsonschema

    jsonschema.validate(instance=payload, schema=WORKFLOW_SPEC_SCHEMA)

    if not isinstance(payload, dict):  # pragma: no cover — schema rejects
        return

    stages = payload.get("stages") or []
    declared: set[str] = set()
    for stage in stages:
        name = stage.get("name") if isinstance(stage, dict) else None
        if not isinstance(name, str):
            continue
        if name in declared:
            raise ValueError(f"duplicate stage name: {name!r}")
        declared.add(name)

    edges = payload.get("edges") or []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        from_stage = edge.get("from_stage")
        if from_stage not in declared:
            raise ValueError(
                f"edge.from_stage {from_stage!r} not in declared stages"
            )
        kind = edge.get("kind")
        if kind == "sequential":
            target = edge.get("to_stage")
            if target not in declared:
                raise ValueError(
                    f"edge.to_stage {target!r} not in declared stages"
                )
        elif kind == "branch":
            for branch_field in ("true_stage", "false_stage"):
                target = edge.get(branch_field)
                if target not in declared:
                    raise ValueError(
                        f"edge.{branch_field} {target!r} not in declared stages"
                    )
        elif kind == "parallel":
            missing = [s for s in (edge.get("stages") or []) if s not in declared]
            if missing:
                raise ValueError(
                    f"edge.parallel.stages reference undeclared: {missing}"
                )
        # SUBWORKFLOW edges reference an external workflow name, not
        # a stage in this spec, so no in-graph reference check applies.


__all__ = [
    "WORKFLOW_RUN_SCHEMA",
    "WORKFLOW_SPEC_SCHEMA",
    "WORKFLOW_STAGE_LINK_SCHEMA",
    "validate_spec_payload",
]
