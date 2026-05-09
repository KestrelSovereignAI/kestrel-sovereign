"""kestrel-feature-workflows — first-class agent workflow primitive.

Phase 0 (this commit): dataclasses, JSON schemas, storage migrations, DID
signing helpers, stage-to-signal mapping spec. No tools, no runner yet —
those come in Phase 1.

See ``docs/architecture/WORKFLOWS_FEATURE_DESIGN.md`` (v4.1) for the full
design and ``docs/architecture/WORKFLOWS_STAGE_TO_SIGNAL_MAPPING.md`` for
how stages reduce onto SignalDispatcher source registrations.
"""

from kestrel_sovereign.features.workflows.models import (
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

__all__ = [
    "BUILT_IN_GATE_TYPES",
    "BUILT_IN_GATE_TYPES_NEEDING_REGISTRATION",
    "Edge",
    "EdgeKind",
    "Gate",
    "GateOutcome",
    "RunStatus",
    "Stage",
    "StageLink",
    "Trigger",
    "TriggerKind",
    "WorkflowDefinitionError",
    "WorkflowRun",
    "WorkflowSpec",
]
