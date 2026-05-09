"""Workflow domain model — Phase 0.

Implements the dataclasses called out in §3 of
``docs/architecture/WORKFLOWS_FEATURE_DESIGN.md`` (v4.1):

- :class:`WorkflowSpec`        — versioned, signed definition.
- :class:`Stage`               — node in the workflow graph.
- :class:`Edge`                — typed connection between stages.
- :class:`Gate`                — pass/fail predicate evaluated after a stage.
- :class:`Trigger`             — how a workflow run is started.
- :class:`WorkflowRun`         — one execution.
- :class:`StageLink`           — join from (run_id, stage, attempt) to the
                                 dispatched signal_id, with workflow-only
                                 fields (gate outcome, compensate state,
                                 actor signature).

These are pure-Python dataclasses with structural validation. Storage
shape and JSON Schema are defined in ``store.py`` and ``schema.py``
respectively; this module is the single source of truth for the values
those layers serialize.

Validation philosophy: every constructor enforces the closed vocabularies
the design doc declares (gate types, edge kinds, run statuses, trigger
kinds). Invalid inputs raise :class:`WorkflowDefinitionError` at the
dataclass boundary so callers (the registrar tool, the runner, the
schema validator) all get a single, consistent failure mode.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from kestrel_sdk.signals import SignalMode

# ---------------------------------------------------------------------------
# Closed vocabularies (mirror docs/architecture/WORKFLOWS_FEATURE_DESIGN.md)
# ---------------------------------------------------------------------------

# Gate types are a closed set; arbitrary callables were intentionally
# removed in v4 (replaced by the sandboxed ``script(...)`` form).
BUILT_IN_GATE_TYPES: frozenset[str] = frozenset(
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

# Subset that requires a SourceRegistration to exist for the gate's
# back-end actor (e.g. ``red_team_clear`` dispatches to a reviewer pool;
# ``council_approve`` to the council source). The registry validator in
# Phase 1 will check the named source exists at run-start.
BUILT_IN_GATE_TYPES_NEEDING_REGISTRATION: frozenset[str] = frozenset(
    {
        "red_team_clear",
        "council_approve",
        "consent_collect",
        "tests_pass",
        "ci_green",
        "lint_clean",
        "script",
    }
)

# Run statuses, lifted verbatim from §5 of the design doc.
_RUN_STATUSES: frozenset[str] = frozenset(
    {
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
)

# Gate outcome vocabulary used in workflow_stage_links.gate_outcome.
_GATE_OUTCOMES: frozenset[str] = frozenset({"pass", "fail", "pending"})

# Compensate states, design §3.5.
_COMPENSATE_STATES: frozenset[str] = frozenset(
    {"not_required", "pending", "complete", "record_only", "failed"}
)


class EdgeKind(str, Enum):
    """Closed vocabulary for stage-to-stage connections (design §3.1)."""

    SEQUENTIAL = "sequential"
    BRANCH = "branch"
    PARALLEL = "parallel"
    SUBWORKFLOW = "subworkflow"


class TriggerKind(str, Enum):
    """Trigger vocabulary — Phase 3 ships ``manual`` and ``cron``; the
    Signal-source-as-trigger generalization is a follow-up (design §6
    Phase 3)."""

    MANUAL = "manual"
    CRON = "cron"
    SIGNAL_SOURCE = "signal_source"


class GateOutcome(str, Enum):
    """Mirror of ``workflow_stage_links.gate_outcome`` values."""

    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"


class RunStatus(str, Enum):
    """Mirror of ``workflow_runs.status`` values."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING = "waiting"
    COMPENSATING = "compensating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCELLED_WITH_IRREVERSIBLE_RESIDUE = "cancelled_with_irreversible_residue"


# Sanity check: enum values must match the lower-level frozensets so
# storage parsing never disagrees with dataclass validation. A drift
# here is the kind of bug only an end-to-end test catches.
assert {s.value for s in RunStatus} == _RUN_STATUSES
assert {s.value for s in GateOutcome} == _GATE_OUTCOMES


_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.\-]*$")
# Lowercase hex sha256 digest, 64 chars; mirrors signing-side helpers and the
# JSON Schema `_HASH_PATTERN`.
_IDEMPOTENCY_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


class WorkflowDefinitionError(ValueError):
    """Raised when a dataclass constructor receives input that violates
    the design's closed vocabularies or structural invariants."""


def _validate_name(label: str, name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise WorkflowDefinitionError(
            f"{label} must be a non-empty identifier matching "
            f"{_NAME_RE.pattern!r}, got {name!r}"
        )


def _coerce_signal_mode(value: Any) -> SignalMode:
    """Accept either a :class:`SignalMode` instance or its string form
    (``"action"``/``"artifact"``/``"cognition"``). Raise on anything else.

    Phase 1's ``WorkflowRunner`` only knows :class:`SignalMode`; storing
    the string form makes Phase 0 round-trips through JSON natural and
    keeps the constructor permissive for callers building Stages from
    parsed JSON.
    """
    if isinstance(value, SignalMode):
        return value
    if isinstance(value, str):
        try:
            return SignalMode(value)
        except ValueError as exc:  # noqa: BLE001 — we re-raise with context
            valid = ", ".join(sorted(m.value for m in SignalMode))
            raise WorkflowDefinitionError(
                f"signal_mode {value!r} not in closed vocabulary "
                f"({valid})"
            ) from exc
    raise WorkflowDefinitionError(
        f"signal_mode must be SignalMode or str, got {type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """Closed-vocabulary pass/fail predicate evaluated after a stage's
    SignalResult returns. Design §3.3."""

    type: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, str):
            raise WorkflowDefinitionError(
                f"gate.type must be str, got {type(self.type).__name__}"
            )
        if self.type not in BUILT_IN_GATE_TYPES:
            valid = ", ".join(sorted(BUILT_IN_GATE_TYPES))
            raise WorkflowDefinitionError(
                f"gate type {self.type!r} not in closed vocabulary ({valid})"
            )
        if not isinstance(self.params, Mapping):
            raise WorkflowDefinitionError(
                f"gate params must be a mapping, got "
                f"{type(self.params).__name__}"
            )
        # Frozen mapping for hashability of Gate (subclassed dicts would
        # leak mutability into the dataclass). dict() is fine — JSON
        # round-trip materializes a plain dict on the way back.
        object.__setattr__(self, "params", dict(self.params))

        # Light per-type schema validation; the heavy structural rules
        # (e.g. red_team_clear's reviewer pool size) belong in Phase 2's
        # gate-specific validators. Phase 0 catches the cheap, type-
        # required-field mistakes that would otherwise corrupt storage.
        if self.type == "constitutional_boundary_clean":
            forbidden = self.params.get("forbidden_modules")
            if not isinstance(forbidden, (list, tuple)) or not all(
                isinstance(m, str) for m in forbidden
            ):
                raise WorkflowDefinitionError(
                    "gate constitutional_boundary_clean requires "
                    "params.forbidden_modules: list[str]"
                )
        if self.type == "red_team_clear":
            constraint = self.params.get("prompt_pack_constraint")
            if not isinstance(constraint, str) or not constraint.strip():
                raise WorkflowDefinitionError(
                    "gate red_team_clear requires "
                    "params.prompt_pack_constraint (PEP 440 spec) per §3.4"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Gate":
        # Round 4 P2 (signed-spec integrity): pass raw values through so
        # type-coercion can't normalize a malformed wire form into a
        # canonical one. ``str(non_str)`` would silently accept a wire
        # ``type: 42`` (int) as ``"42"`` — the dataclass would then
        # reject it for the unrelated reason of "not in built-in gate
        # types," but with a different malformed value (e.g. integer
        # number that happens to stringify to a known type) the
        # signature/hash verification path would round-trip a value that
        # was never in the signed canonical form.
        if not isinstance(data, Mapping):
            raise WorkflowDefinitionError(
                f"Gate.from_dict expected mapping, got {type(data).__name__}"
            )
        return cls(type=data["type"], params=data.get("params") or {})


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """Node in the workflow graph (design §3.1)."""

    name: str
    signal_source: str
    signal_mode: SignalMode
    params: Mapping[str, Any] = field(default_factory=dict)
    gate: Gate = field(default_factory=lambda: Gate(type="signal_status_ok"))
    compensate: str = "noop_idempotent"
    forbidden_modules: Sequence[str] = ()
    irreversible: bool = False
    non_deterministic: bool = False
    read_only: bool = False

    def __post_init__(self) -> None:
        _validate_name("stage.name", self.name)
        _validate_name("stage.signal_source", self.signal_source)
        object.__setattr__(self, "signal_mode", _coerce_signal_mode(self.signal_mode))

        if not isinstance(self.params, Mapping):
            raise WorkflowDefinitionError(
                f"stage.params must be a mapping, got "
                f"{type(self.params).__name__}"
            )
        object.__setattr__(self, "params", dict(self.params))

        if not isinstance(self.gate, Gate):
            raise WorkflowDefinitionError(
                "stage.gate must be a Gate instance (use Gate.from_dict for "
                "JSON-loaded inputs)"
            )

        if not isinstance(self.compensate, str) or not self.compensate.strip():
            raise WorkflowDefinitionError(
                "stage.compensate must be a non-empty string ("
                "'noop_idempotent' is the default; otherwise the name of a "
                "registered SourceRegistration that runs the compensation)"
            )

        if not isinstance(self.forbidden_modules, (list, tuple)) or not all(
            isinstance(m, str) for m in self.forbidden_modules
        ):
            raise WorkflowDefinitionError(
                "stage.forbidden_modules must be a list[str]"
            )
        object.__setattr__(self, "forbidden_modules", tuple(self.forbidden_modules))

        for flag_name in ("irreversible", "non_deterministic", "read_only"):
            value = getattr(self, flag_name)
            if not isinstance(value, bool):
                raise WorkflowDefinitionError(
                    f"stage.{flag_name} must be bool, got {type(value).__name__}"
                )

        # noop_idempotent eligibility (design §3.5). Phase 0 only checks
        # the *Stage-side* preconditions; Phase 1's runner additionally
        # verifies "no DB writes outside workflow_* tables" at runtime.
        # Stages whose gate is consent_collect are also eligible (the
        # rejection naturally compensates the request).
        if self.compensate == "noop_idempotent":
            ok = (
                (self.signal_mode == SignalMode.ACTION and self.read_only)
                or self.gate.type == "consent_collect"
            )
            if not ok:
                raise WorkflowDefinitionError(
                    f"stage {self.name!r}: noop_idempotent is only valid "
                    "when signal_mode=ACTION and read_only=True, OR when "
                    "gate.type=consent_collect (design §3.5). Otherwise "
                    "declare a real compensate (a registered SourceRegistration)."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signal_source": self.signal_source,
            "signal_mode": self.signal_mode.value,
            "params": dict(self.params),
            "gate": self.gate.to_dict(),
            "compensate": self.compensate,
            "forbidden_modules": list(self.forbidden_modules),
            "irreversible": self.irreversible,
            "non_deterministic": self.non_deterministic,
            "read_only": self.read_only,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Stage":
        if not isinstance(data, Mapping):
            raise WorkflowDefinitionError(
                f"Stage.from_dict expected mapping, got {type(data).__name__}"
            )
        gate_data = data.get("gate") or {"type": "signal_status_ok"}
        # Boolean fields pass through verbatim. We deliberately do NOT
        # call ``bool(...)`` here because ``bool("false")`` is True —
        # which would silently flip read_only=False into True for any
        # client that emitted strings instead of JSON booleans, letting
        # writeful ACTION stages slip past the noop_idempotent
        # eligibility check in __post_init__. Pass-through keeps the
        # type guard in __post_init__ load-bearing.
        #
        # ``forbidden_modules`` similarly passes through raw (no
        # ``tuple(...)`` wrap) — round-3 codex P2: ``tuple("features")``
        # is ``('f', 'e', ..., 's')``, an iterable of single-char
        # strings that would silently pass __post_init__'s
        # ``all(isinstance(m, str) ...)`` check while completely
        # mangling the constitutional-boundary scope. Passing the raw
        # value lets the ``isinstance(value, (list, tuple))`` guard in
        # __post_init__ reject strings as the user intended.
        #
        # Round 4 P2 (signed-spec integrity): every other field also
        # passes raw — no ``str(...)``/``dict(...)`` coercion. A wire
        # value of the wrong type would otherwise be normalized into
        # the canonical form silently, and ``WorkflowSpec.compute_spec_hash``
        # would compute the same digest as the actually-signed
        # canonical bytes, letting a malformed wire form claim a valid
        # signature.
        return cls(
            name=data["name"],
            signal_source=data["signal_source"],
            signal_mode=data["signal_mode"],
            params=data.get("params") or {},
            gate=Gate.from_dict(gate_data),
            compensate=data.get("compensate", "noop_idempotent"),
            forbidden_modules=data.get("forbidden_modules") or (),
            irreversible=data.get("irreversible", False),
            non_deterministic=data.get("non_deterministic", False),
            read_only=data.get("read_only", False),
        )


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """Typed connection between stages (design §3.1)."""

    kind: EdgeKind
    from_stage: str
    # SEQUENTIAL: ``to_stage`` populated.
    # BRANCH:     ``condition``, ``true_stage``, ``false_stage`` populated.
    # PARALLEL:   ``stages`` (>= 2) and ``join_strategy`` populated.
    # SUBWORKFLOW: ``subworkflow_name``, ``subworkflow_version`` populated;
    #              ``params`` carry the sub-call payload.
    to_stage: Optional[str] = None
    condition: Optional[str] = None
    true_stage: Optional[str] = None
    false_stage: Optional[str] = None
    stages: Sequence[str] = ()
    join_strategy: Optional[str] = None
    subworkflow_name: Optional[str] = None
    subworkflow_version: Optional[int] = None
    params: Mapping[str, Any] = field(default_factory=dict)

    _VALID_JOIN_STRATEGIES: frozenset[str] = frozenset(
        {"all", "any", "first_success"}
    )

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            try:
                object.__setattr__(self, "kind", EdgeKind(self.kind))
            except ValueError as exc:
                valid = ", ".join(sorted(k.value for k in EdgeKind))
                raise WorkflowDefinitionError(
                    f"edge.kind {self.kind!r} not in vocabulary ({valid})"
                ) from exc
        elif not isinstance(self.kind, EdgeKind):
            raise WorkflowDefinitionError(
                f"edge.kind must be EdgeKind or str, got "
                f"{type(self.kind).__name__}"
            )

        _validate_name("edge.from_stage", self.from_stage)

        if self.kind == EdgeKind.SEQUENTIAL:
            _validate_name("edge.to_stage", self.to_stage or "")
            self._reject_unrelated_fields(
                {"to_stage", "from_stage", "kind", "params"}
            )
        elif self.kind == EdgeKind.BRANCH:
            if not isinstance(self.condition, str) or not self.condition.strip():
                raise WorkflowDefinitionError(
                    "edge.kind=branch requires non-empty condition (string "
                    "expression evaluated against stage outputs in Phase 1)"
                )
            _validate_name("edge.true_stage", self.true_stage or "")
            _validate_name("edge.false_stage", self.false_stage or "")
            self._reject_unrelated_fields(
                {
                    "from_stage",
                    "kind",
                    "condition",
                    "true_stage",
                    "false_stage",
                    "params",
                }
            )
        elif self.kind == EdgeKind.PARALLEL:
            if (
                not isinstance(self.stages, (list, tuple))
                or len(self.stages) < 2
                or not all(isinstance(s, str) and s for s in self.stages)
            ):
                raise WorkflowDefinitionError(
                    "edge.kind=parallel requires stages: list[str] with >= 2 "
                    "non-empty entries"
                )
            object.__setattr__(self, "stages", tuple(self.stages))
            if self.join_strategy not in self._VALID_JOIN_STRATEGIES:
                valid = ", ".join(sorted(self._VALID_JOIN_STRATEGIES))
                raise WorkflowDefinitionError(
                    f"edge.kind=parallel requires join_strategy in "
                    f"({valid}); got {self.join_strategy!r}"
                )
            self._reject_unrelated_fields(
                {"from_stage", "kind", "stages", "join_strategy", "params"}
            )
        else:  # SUBWORKFLOW
            _validate_name("edge.subworkflow_name", self.subworkflow_name or "")
            if (
                not isinstance(self.subworkflow_version, int)
                or self.subworkflow_version < 1
            ):
                raise WorkflowDefinitionError(
                    "edge.kind=subworkflow requires subworkflow_version: "
                    "positive int"
                )
            self._reject_unrelated_fields(
                {
                    "from_stage",
                    "kind",
                    "subworkflow_name",
                    "subworkflow_version",
                    "params",
                }
            )

        if not isinstance(self.params, Mapping):
            raise WorkflowDefinitionError(
                "edge.params must be a mapping (SUBWORKFLOW uses it as the "
                "sub-call payload; other kinds keep it empty)"
            )
        object.__setattr__(self, "params", dict(self.params))

    def _reject_unrelated_fields(self, allowed: set[str]) -> None:
        """Frozen-dataclass guard: every kind populates only its own
        fields. Mixing branch+parallel inputs is a definition bug; we
        catch it at construction so storage doesn't store it."""
        for forbidden in (
            "to_stage",
            "condition",
            "true_stage",
            "false_stage",
            "stages",
            "join_strategy",
            "subworkflow_name",
            "subworkflow_version",
        ):
            if forbidden in allowed:
                continue
            value = getattr(self, forbidden)
            empty = (
                value is None
                or value == ""
                or (isinstance(value, (list, tuple)) and len(value) == 0)
            )
            if not empty:
                raise WorkflowDefinitionError(
                    f"edge.kind={self.kind.value} cannot set "
                    f"{forbidden!r}={value!r}; field belongs to a "
                    "different edge kind"
                )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind.value,
            "from_stage": self.from_stage,
        }
        if self.kind == EdgeKind.SEQUENTIAL:
            out["to_stage"] = self.to_stage
        elif self.kind == EdgeKind.BRANCH:
            out["condition"] = self.condition
            out["true_stage"] = self.true_stage
            out["false_stage"] = self.false_stage
        elif self.kind == EdgeKind.PARALLEL:
            out["stages"] = list(self.stages)
            out["join_strategy"] = self.join_strategy
        else:
            out["subworkflow_name"] = self.subworkflow_name
            out["subworkflow_version"] = self.subworkflow_version
        if self.params:
            out["params"] = dict(self.params)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Edge":
        if not isinstance(data, Mapping):
            raise WorkflowDefinitionError(
                f"Edge.from_dict expected mapping, got {type(data).__name__}"
            )
        # ``stages`` (parallel-edge fan-out list) passes through raw —
        # ``tuple("a,b,c")`` would expand a string into single chars,
        # each of which is itself a non-empty str, silently producing 5
        # "stages" from one mistyped value. Same trap as Stage's
        # forbidden_modules (round-3 codex P2). __post_init__ rejects
        # non-list/non-tuple inputs.
        #
        # Round 4 P2 (signed-spec integrity): pass everything raw so
        # a malformed wire value can't be normalized into the canonical
        # signed form. ``__post_init__`` enforces the closed vocabulary
        # and per-kind required fields.
        return cls(
            kind=data["kind"],
            from_stage=data["from_stage"],
            to_stage=data.get("to_stage"),
            condition=data.get("condition"),
            true_stage=data.get("true_stage"),
            false_stage=data.get("false_stage"),
            stages=data.get("stages") or (),
            join_strategy=data.get("join_strategy"),
            subworkflow_name=data.get("subworkflow_name"),
            subworkflow_version=data.get("subworkflow_version"),
            params=data.get("params") or {},
        )


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trigger:
    """How a workflow run is started (design §6 Phase 3)."""

    kind: TriggerKind
    cron_expression: Optional[str] = None
    signal_source: Optional[str] = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            try:
                object.__setattr__(self, "kind", TriggerKind(self.kind))
            except ValueError as exc:
                valid = ", ".join(sorted(k.value for k in TriggerKind))
                raise WorkflowDefinitionError(
                    f"trigger.kind {self.kind!r} not in vocabulary ({valid})"
                ) from exc

        if self.kind == TriggerKind.CRON:
            if not isinstance(self.cron_expression, str) or not self.cron_expression.strip():
                raise WorkflowDefinitionError(
                    "trigger.kind=cron requires non-empty cron_expression"
                )
        elif self.kind == TriggerKind.SIGNAL_SOURCE:
            _validate_name("trigger.signal_source", self.signal_source or "")
        # MANUAL needs no extra fields.

        if not isinstance(self.params, Mapping):
            raise WorkflowDefinitionError(
                "trigger.params must be a mapping"
            )
        object.__setattr__(self, "params", dict(self.params))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind.value}
        if self.cron_expression is not None:
            out["cron_expression"] = self.cron_expression
        if self.signal_source is not None:
            out["signal_source"] = self.signal_source
        if self.params:
            out["params"] = dict(self.params)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trigger":
        # Round 4 P2 (signed-spec integrity): raw pass-through; let
        # __post_init__ enforce types so coercion can't normalize a
        # malformed wire form.
        if not isinstance(data, Mapping):
            raise WorkflowDefinitionError(
                f"Trigger.from_dict expected mapping, got "
                f"{type(data).__name__}"
            )
        return cls(
            kind=data["kind"],
            cron_expression=data.get("cron_expression"),
            signal_source=data.get("signal_source"),
            params=data.get("params") or {},
        )


# ---------------------------------------------------------------------------
# WorkflowSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowSpec:
    """Versioned, signed workflow definition (design §3.1, §3.6).

    The ``spec_hash`` and ``author_sig`` fields are populated by the
    signing helpers in :mod:`kestrel_sovereign.features.workflows.signing`
    (Phase 0 chunk C); ``WorkflowSpec`` itself only validates structure.
    A spec with empty author_did/author_sig is a "draft" and must NOT be
    accepted by the runner — Phase 1's run-start path enforces that.
    """

    name: str
    version: int
    stages: Sequence[Stage]
    edges: Sequence[Edge] = ()
    triggers: Sequence[Trigger] = field(
        default_factory=lambda: (Trigger(kind=TriggerKind.MANUAL),)
    )
    params_schema: Mapping[str, Any] = field(default_factory=dict)
    retention_days: Optional[int] = None
    author_did: str = ""
    author_sig: str = ""
    spec_hash: str = ""

    def __post_init__(self) -> None:
        _validate_name("workflow.name", self.name)
        if not isinstance(self.version, int) or self.version < 1:
            raise WorkflowDefinitionError(
                "workflow.version must be a positive int"
            )

        if not isinstance(self.stages, (list, tuple)) or len(self.stages) < 1:
            raise WorkflowDefinitionError(
                "workflow.stages must be a non-empty list of Stage instances"
            )
        if not all(isinstance(s, Stage) for s in self.stages):
            raise WorkflowDefinitionError(
                "every entry in workflow.stages must be a Stage"
            )
        # Stage names unique within a workflow.
        names = [s.name for s in self.stages]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise WorkflowDefinitionError(
                f"duplicate stage names: {dupes}"
            )
        object.__setattr__(self, "stages", tuple(self.stages))

        # Edges must reference declared stages only.
        if not isinstance(self.edges, (list, tuple)) or not all(
            isinstance(e, Edge) for e in self.edges
        ):
            raise WorkflowDefinitionError(
                "workflow.edges must be a list of Edge instances"
            )
        declared = set(names)
        for edge in self.edges:
            if edge.from_stage not in declared:
                raise WorkflowDefinitionError(
                    f"edge.from_stage {edge.from_stage!r} not in declared "
                    f"stages {sorted(declared)}"
                )
            if edge.kind == EdgeKind.SEQUENTIAL and edge.to_stage not in declared:
                raise WorkflowDefinitionError(
                    f"edge.to_stage {edge.to_stage!r} not in declared stages"
                )
            if edge.kind == EdgeKind.BRANCH:
                for branch_field in ("true_stage", "false_stage"):
                    target = getattr(edge, branch_field)
                    if target not in declared:
                        raise WorkflowDefinitionError(
                            f"edge.{branch_field} {target!r} not in declared "
                            "stages"
                        )
            if edge.kind == EdgeKind.PARALLEL:
                missing = [s for s in edge.stages if s not in declared]
                if missing:
                    raise WorkflowDefinitionError(
                        f"edge.parallel.stages reference undeclared: {missing}"
                    )
        object.__setattr__(self, "edges", tuple(self.edges))

        if not isinstance(self.triggers, (list, tuple)) or not all(
            isinstance(t, Trigger) for t in self.triggers
        ):
            raise WorkflowDefinitionError(
                "workflow.triggers must be a list of Trigger instances"
            )
        object.__setattr__(self, "triggers", tuple(self.triggers))

        if not isinstance(self.params_schema, Mapping):
            raise WorkflowDefinitionError(
                "workflow.params_schema must be a mapping (JSON Schema fragment)"
            )
        object.__setattr__(self, "params_schema", dict(self.params_schema))

        if self.retention_days is not None and (
            not isinstance(self.retention_days, int) or self.retention_days < 1
        ):
            raise WorkflowDefinitionError(
                "workflow.retention_days must be None (retain forever) or a "
                "positive int"
            )

        for str_field in ("author_did", "author_sig", "spec_hash"):
            value = getattr(self, str_field)
            if not isinstance(value, str):
                raise WorkflowDefinitionError(
                    f"workflow.{str_field} must be a string (empty for unsigned drafts)"
                )

    def canonical_payload(self) -> dict[str, Any]:
        """The hashable, signable payload — excludes signature and hash
        fields. Stable JSON serialization (sort_keys, no whitespace) is
        the responsibility of the signing helpers; this returns the
        plain dict so callers can compose it with their own framing.
        """
        return {
            "name": self.name,
            "version": self.version,
            "stages": [s.to_dict() for s in self.stages],
            "edges": [e.to_dict() for e in self.edges],
            "triggers": [t.to_dict() for t in self.triggers],
            "params_schema": dict(self.params_schema),
            "retention_days": self.retention_days,
            "author_did": self.author_did,
        }

    def compute_spec_hash(self) -> str:
        """SHA-256 over the JSON-canonicalized signable payload.

        Excludes ``author_sig`` and ``spec_hash`` so the hash is the
        thing the author signs and what verification recomputes. Output
        is lowercase hex (64 chars), matching the rest of Kestrel's hash
        plumbing (signal_log digests, constitution_hash, doctrine
        bundle hash).
        """
        canonical = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        out = self.canonical_payload()
        out["author_sig"] = self.author_sig
        out["spec_hash"] = self.spec_hash
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowSpec":
        # Round 4 P2 (signed-spec integrity): WorkflowSpec is a DID-
        # signed artifact. ``from_dict`` MUST NOT normalize wire types —
        # an attacker who emits ``"version": "1"`` (string) instead of
        # the signed canonical ``"version": 1`` (int) would otherwise
        # produce a WorkflowSpec whose ``compute_spec_hash`` matches the
        # signed hash, letting the malformed wire form claim a valid
        # signature. Raw pass-through plus ``__post_init__`` isinstance
        # checks reject the wrong type at the boundary, before any
        # signature verification reads the spec.
        if not isinstance(data, Mapping):
            raise WorkflowDefinitionError(
                f"WorkflowSpec.from_dict expected mapping, got "
                f"{type(data).__name__}"
            )
        triggers_data = data.get("triggers")
        triggers: Sequence[Trigger]
        if triggers_data is None:
            triggers = (Trigger(kind=TriggerKind.MANUAL),)
        else:
            triggers = tuple(Trigger.from_dict(t) for t in triggers_data)
            if not triggers:
                triggers = (Trigger(kind=TriggerKind.MANUAL),)
        # author_did / author_sig / spec_hash are stored as strings on
        # the dataclass; ``__post_init__`` rejects non-string values.
        # ``data.get(... or "")`` would coerce missing into ""; we use
        # an explicit-default form so a present-but-non-string value
        # propagates and is rejected.
        return cls(
            name=data["name"],
            version=data["version"],
            stages=tuple(Stage.from_dict(s) for s in data.get("stages") or ()),
            edges=tuple(Edge.from_dict(e) for e in data.get("edges") or ()),
            triggers=triggers,
            params_schema=data.get("params_schema") or {},
            retention_days=data.get("retention_days"),
            author_did=data.get("author_did", ""),
            author_sig=data.get("author_sig", ""),
            spec_hash=data.get("spec_hash", ""),
        )


# ---------------------------------------------------------------------------
# WorkflowRun
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowRun:
    """One execution (design §5 ``workflow_runs`` row)."""

    run_id: str
    workflow_name: str
    workflow_ver: int
    params: Mapping[str, Any]
    status: RunStatus
    current_stages: Sequence[str] = ()
    parent_run_id: Optional[str] = None
    cancel_barrier_at: Optional[datetime] = None
    started_by_did: str = ""
    scheduler_task_id: Optional[str] = None
    signature_post_revocation: bool = False
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise WorkflowDefinitionError("run.run_id must be a non-empty string")
        _validate_name("run.workflow_name", self.workflow_name)
        if not isinstance(self.workflow_ver, int) or self.workflow_ver < 1:
            raise WorkflowDefinitionError(
                "run.workflow_ver must be a positive int"
            )

        if isinstance(self.status, str):
            try:
                object.__setattr__(self, "status", RunStatus(self.status))
            except ValueError as exc:
                raise WorkflowDefinitionError(
                    f"run.status {self.status!r} not in vocabulary"
                ) from exc

        if not isinstance(self.params, Mapping):
            raise WorkflowDefinitionError("run.params must be a mapping")
        object.__setattr__(self, "params", dict(self.params))

        if not isinstance(self.current_stages, (list, tuple)) or not all(
            isinstance(s, str) for s in self.current_stages
        ):
            raise WorkflowDefinitionError(
                "run.current_stages must be a list[str] of stage names"
            )
        object.__setattr__(self, "current_stages", tuple(self.current_stages))

        if not isinstance(self.started_by_did, str):
            raise WorkflowDefinitionError(
                "run.started_by_did must be a string"
            )
        if not isinstance(self.signature_post_revocation, bool):
            raise WorkflowDefinitionError(
                "run.signature_post_revocation must be bool"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "workflow_ver": self.workflow_ver,
            "params": dict(self.params),
            "status": self.status.value,
            "current_stages": list(self.current_stages),
            "parent_run_id": self.parent_run_id,
            "cancel_barrier_at": _iso(self.cancel_barrier_at),
            "started_by_did": self.started_by_did,
            "scheduler_task_id": self.scheduler_task_id,
            "signature_post_revocation": self.signature_post_revocation,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "deleted_at": _iso(self.deleted_at),
        }


# ---------------------------------------------------------------------------
# StageLink
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageLink:
    """Join row from (run_id, stage, attempt) to the dispatched signal_id
    (design §3.1, §5 ``workflow_stage_links``).

    ``actor_sig`` is the DID signature over the canonical transition
    payload (run_id, stage_name, attempt_number, signal_id, gate_outcome).
    Phase 0 doesn't compute signatures here — that's the signing helper's
    job. The dataclass just enforces structural validity so storage gets
    a single, well-shaped row.
    """

    link_id: str
    run_id: str
    stage_name: str
    attempt_number: int
    idempotency_key: str
    actor_did: str
    actor_sig: str
    signal_id: Optional[str] = None
    gate_outcome: Optional[GateOutcome] = None
    gate_reason: Optional[str] = None
    compensate_state: Optional[str] = None
    post_cancel: bool = False
    occurred_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        for str_field in ("link_id", "run_id", "actor_did", "actor_sig"):
            value = getattr(self, str_field)
            if not isinstance(value, str) or not value:
                raise WorkflowDefinitionError(
                    f"stage_link.{str_field} must be a non-empty string"
                )
        # Round 4 P2: idempotency_key is the dedupe/attempt invariant
        # the runner relies on (design §3.5: ``sha256(run_id||stage||
        # sha256(canonical_input||attempt||nonce))``). Any non-hex value
        # passing the dataclass would corrupt that invariant — runner
        # callers (Phase 1) read the field and would log mismatches as
        # if it were a digest. Enforce the shape at the dataclass
        # boundary so direct constructors get the same guarantee the
        # JSON schema gives wire callers.
        if not isinstance(self.idempotency_key, str) or not _IDEMPOTENCY_KEY_RE.match(
            self.idempotency_key
        ):
            raise WorkflowDefinitionError(
                "stage_link.idempotency_key must be a lowercase hex sha256 "
                "digest (64 chars matching ^[0-9a-f]{64}$); got "
                f"{self.idempotency_key!r}"
            )
        _validate_name("stage_link.stage_name", self.stage_name)
        if not isinstance(self.attempt_number, int) or self.attempt_number < 1:
            raise WorkflowDefinitionError(
                "stage_link.attempt_number must be a positive int (1-indexed)"
            )

        if self.gate_outcome is not None and isinstance(self.gate_outcome, str):
            try:
                object.__setattr__(self, "gate_outcome", GateOutcome(self.gate_outcome))
            except ValueError as exc:
                raise WorkflowDefinitionError(
                    f"stage_link.gate_outcome {self.gate_outcome!r} invalid"
                ) from exc

        if self.compensate_state is not None and self.compensate_state not in _COMPENSATE_STATES:
            valid = ", ".join(sorted(_COMPENSATE_STATES))
            raise WorkflowDefinitionError(
                f"stage_link.compensate_state {self.compensate_state!r} not "
                f"in vocabulary ({valid})"
            )

        if not isinstance(self.post_cancel, bool):
            raise WorkflowDefinitionError(
                "stage_link.post_cancel must be bool"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "attempt_number": self.attempt_number,
            "signal_id": self.signal_id,
            "idempotency_key": self.idempotency_key,
            "gate_outcome": (
                self.gate_outcome.value if self.gate_outcome is not None else None
            ),
            "gate_reason": self.gate_reason,
            "compensate_state": self.compensate_state,
            "post_cancel": self.post_cancel,
            "actor_did": self.actor_did,
            "actor_sig": self.actor_sig,
            "occurred_at": _iso(self.occurred_at),
        }


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None
