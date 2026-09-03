"""
Sleep functionality for Kestrel Agent.

Implements human-like "sleep" cycle that:
1. Consolidates memories (creates episodes, archives decayed)
2. Exports state to sovereignty storage (IPFS/Lighthouse)
3. Returns a CID that can be used for restoration

This is inspired by how human memory consolidation occurs during sleep:
- Short-term memories are organized into long-term storage
- Unimportant details fade (forgetting curve)
- Patterns are detected and strengthened
- A "checkpoint" is created for disaster recovery
"""

from collections.abc import Mapping
import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import heapq
from typing import Optional, Dict, Any, Callable, List, Tuple

logger = logging.getLogger(__name__)


# ``!sleep`` is an authenticated operational surface, but its response still
# travels through the normal chat/invoke formatter.  Keep semantic-maintenance
# observability deliberately aggregate-only so a run report can never turn that
# surface into an assertion, provenance, or raw-error disclosure.
_SEMANTIC_MAINTENANCE_SUMMARY_MAX_CHARS = 1_024
_SEMANTIC_MAINTENANCE_MAX_RENDERED_NUMBER = 1_000_000_000
_SEMANTIC_MAINTENANCE_CAPABILITY_VALUE_MAX_CHARS = 256
_SEMANTIC_MAINTENANCE_CONTRACT_VERSION = "v3"
_SEMANTIC_MAINTENANCE_CAPABILITY_KEYS = (
    "semantic_maintenance",
    "maintenance_budget",
    "shape_set",
    "validation_capability",
    "validation_profile_version",
    "validation_artifact_pins",
    "inference_profile",
    "rule_profile",
    "ontology",
    "semantic_capability_mode",
    "rdf12_capability",
    "rdf12_version",
    "sparql12_capability",
    "sparql12_version",
)
_SEMANTIC_MAINTENANCE_STATUSES = frozenset(
    {"complete", "partial", "failed", "no_op", "disabled"}
)
_SEMANTIC_MAINTENANCE_REASONS = frozenset(
    {
        "assertion_budget",
        "change_replay",
        "consolidation_failed",
        "consolidation_skipped",
        "context_assertions",
        "contradiction_context_budget",
        "derivation_budget",
        "generated_assertions",
        "inference_incomplete",
        "iterations",
        "memory",
        "repair_change_replay",
        "report_budget",
        "semantic_inference_revocation_failed",
        "semantic_maintenance_busy",
        "semantic_maintenance_capability_unavailable",
        "semantic_maintenance_capability_mismatch",
        "semantic_maintenance_checkpoint_behind",
        "semantic_maintenance_failed",
        "semantic_maintenance_lease_lost",
        "semantic_maintenance_state_missing",
        "semantic_storage_unavailable",
        "semantic_maintenance_validation_capability_unavailable",
        "source_assertions",
        "source_changed_during_closure",
        "validation_incomplete",
        "wall_time",
    }
)
_SLEEP_FAILURE_REASONS = frozenset(
    {
        "consolidation_failed",
        "consolidation_skipped",
        "export_failed",
        "semantic_artifact_expiry_sweep_failed",
        "semantic_inference_revocation_failed",
        "semantic_maintenance_failed",
        "semantic_storage_unavailable",
    }
)


def _bounded_summary_number(value: Any) -> str:
    """Render a finite aggregate count without trusting a report payload."""
    if type(value) is not int or value < 0:
        return "0"
    if value > _SEMANTIC_MAINTENANCE_MAX_RENDERED_NUMBER:
        return f">={_SEMANTIC_MAINTENANCE_MAX_RENDERED_NUMBER}"
    return str(value)


def _semantic_maintenance_status(value: Any) -> str:
    """Return one of the public status tokens, never report-controlled text."""
    if isinstance(value, str) and value in _SEMANTIC_MAINTENANCE_STATUSES:
        return value
    return "unknown"


def _semantic_maintenance_reason(value: Any) -> str:
    """Return a known content-free reason code, redacting all other values."""
    if value is None:
        return "none"
    if isinstance(value, str) and value in _SEMANTIC_MAINTENANCE_REASONS:
        return value
    return "unavailable"


def _sleep_failure_reason(value: Any) -> Optional[str]:
    """Return an established content-free sleep failure code, if available.

    Legacy fallback for a report that carries no ``failure_code`` (the field
    is set at every failing phase; only the skip-only cycle leaves it None).
    Reads the FIRST known token of the composed ``error`` string.
    """
    if not isinstance(value, str):
        return None
    for candidate in value.split(";"):
        code = candidate.strip()
        if code in _SLEEP_FAILURE_REASONS:
            return code
    return None


#: The skip is a deliberate no-op, not a failure: it is never recorded as
#: the cycle's failure code, so a later real failure is the one named.
_SLEEP_NON_TERMINAL_REASON = "consolidation_skipped"


def _record_failure_code(report: "SleepReport", code: Any) -> None:
    """Record the cycle's FIRST terminal failure code, structurally.

    ``error`` stays the composed human string it always was ("; "-joined,
    with interpolated exception text from the export phase). Consumers that
    need a cause — the scheduler's failed-outcome door, the ``!sleep``
    summary — read this field instead of guessing from that string: three
    review rounds of string extraction each named the wrong phase. Only
    codes from the closed vocabulary are recorded, so prose never lands here.
    """
    if code not in _SLEEP_FAILURE_REASONS or code == _SLEEP_NON_TERMINAL_REASON:
        return
    if report.failure_code is None:
        report.failure_code = code


def _semantic_maintenance_capability_summary(value: Any) -> Tuple[int, str]:
    """Return a bounded capability-version count and deterministic digest.

    The maintenance service owns the raw capability map.  Its values can
    contain deployment-specific identifiers, so this renderer considers only
    the fixed contract keys and publishes a digest rather than the map itself.
    """
    if not isinstance(value, Mapping):
        return 0, "none"

    canonical: List[Tuple[str, str]] = []
    for key in _SEMANTIC_MAINTENANCE_CAPABILITY_KEYS:
        if key not in value:
            continue
        raw = value[key]
        if not isinstance(raw, str):
            canonical.append((key, "invalid"))
        elif len(raw) > _SEMANTIC_MAINTENANCE_CAPABILITY_VALUE_MAX_CHARS:
            canonical.append((key, "oversize"))
        else:
            canonical.append((key, raw))

    if not canonical:
        return 0, "none"
    encoded = json.dumps(
        canonical,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return len(canonical), hashlib.sha256(encoded).hexdigest()[:16]


def _semantic_maintenance_active_capabilities(value: Any) -> Tuple[str, ...]:
    """Return registry-verified active capability/version labels.

    The raw maintenance map is intentionally not trusted as a presentation
    surface: a feature or a malformed storage result must not inject source
    text through an operational diagnostic. Values are shown only when they
    are the exact current maintenance contract value or resolve to an exact,
    locally verified registry resource/capability. A version-shaped string is
    never enough: ``v999`` and ``999.999.999`` are not evidence of an active
    contract. Everything else remains represented by the existing digest line,
    which is useful for comparison without disclosure.
    """
    if not isinstance(value, Mapping):
        return ()
    raw = value.get("capability_versions")
    if not isinstance(raw, Mapping):
        return ()

    active: List[str] = []
    maintenance = raw.get("semantic_maintenance")
    if maintenance == _SEMANTIC_MAINTENANCE_CONTRACT_VERSION:
        active.append(f"semantic_maintenance={maintenance}")

    try:
        from kestrel_sovereign.knowledge.registry import (
            KnowledgeRegistryError,
            ResourceKind,
            get_knowledge_registry,
        )

        registry = get_knowledge_registry()
    except (KnowledgeRegistryError, OSError):
        active.extend(
            (
                "inference_profile=unavailable"
                if "inference_profile" in raw
                else "inference_profile=omitted",
                "ontology=unavailable" if "ontology" in raw else "ontology=omitted",
            )
        )
        return tuple(active)

    resource_by_key = {resource.key: resource for resource in registry.resources}
    capability_mode = raw.get("semantic_capability_mode")
    allow_experimental = capability_mode == "experimental"
    if capability_mode in {"stable", "experimental"}:
        active.append(f"semantic_capability_mode={capability_mode}")
    shape_set = raw.get("shape_set")
    if (
        isinstance(shape_set, str)
        and (shape := resource_by_key.get(shape_set)) is not None
        and shape.kind is ResourceKind.SHAPE_SET
    ):
        active.append(f"shape_set={shape_set}")

    validation_profile = None
    validation_capability = raw.get("validation_capability")
    if isinstance(validation_capability, str):
        try:
            selected_validation = registry.select_capability(
                validation_capability,
                allow_experimental=allow_experimental,
            )
        except KnowledgeRegistryError:
            selected_validation = None
        if (
            selected_validation is not None
            and selected_validation.resource.kind is ResourceKind.VALIDATION_PROFILE
        ):
            validation_profile = selected_validation.resource
    if validation_profile is not None:
        active.append(f"validation_capability={validation_capability}")
    validation_version = raw.get("validation_profile_version")
    if validation_profile is not None and (
        validation_version == "registry-selected"
        or validation_version == str(validation_profile.version)
    ):
        active.append(f"validation_profile_version={validation_version}")

    # Draft RDF/SPARQL names are only diagnostic evidence when the producer
    # declared experimental mode *and* each exact registry pin still resolves.
    # A report-shaped payload cannot make a disabled agent look experimental.
    if allow_experimental:
        for prefix, capability_key, version_key in (
            ("rdf-profile:rdf12", "rdf12_capability", "rdf12_version"),
            ("query-profile:sparql12", "sparql12_capability", "sparql12_version"),
        ):
            capability = raw.get(capability_key)
            version = raw.get(version_key)
            try:
                selected = (
                    registry.select_capability(capability, allow_experimental=True)
                    if isinstance(capability, str) and capability.startswith(prefix)
                    else None
                )
            except KnowledgeRegistryError:
                selected = None
            if selected is not None and version == str(selected.resource.version):
                active.extend((f"{capability_key}={capability}", f"{version_key}={version}"))

    rule_profile = raw.get("rule_profile")
    rule_versions = _registered_rule_profile_versions(
        rule_profile,
        resource_by_key,
        ResourceKind,
    )
    if rule_versions is not None:
        active.append(f"rule_profile={rule_profile}")

    ontology = _registered_ontology(raw.get("ontology"), registry.resources, ResourceKind)
    if ontology is not None:
        active.append(f"ontology={ontology.key}")
    elif "ontology" in raw:
        active.append("ontology=unavailable")
    else:
        active.append("ontology=omitted")

    profile_value = raw.get("inference_profile")
    if "inference_profile" not in raw:
        active.append("inference_profile=omitted")
    elif _is_registered_inference_profile(
        profile_value,
        ontology=ontology,
        rule_versions=rule_versions,
        registry_contract=registry.contract_version,
    ):
        active.append(f"inference_profile={profile_value}")
    else:
        active.append("inference_profile=unavailable")
    return tuple(active)


def _registered_rule_profile_versions(
    value: Any,
    resource_by_key: Mapping[str, Any],
    resource_kind: Any,
) -> Tuple[str, Optional[str]] | None:
    """Resolve the exact compact rule-profile label emitted by the producer."""
    if not isinstance(value, str):
        return None
    parts = value.split("+")
    if len(parts) not in {1, 2}:
        return None
    expected_identifiers = ("rdfs-v1", "owl2rl-kestrel-v1")
    versions: List[str] = []
    for index, part in enumerate(parts):
        identifier, separator, version = part.partition("@")
        if not separator or identifier != expected_identifiers[index] or not version:
            return None
        resource = resource_by_key.get(f"{identifier}@{version}")
        if (
            resource is None
            or resource.identifier != identifier
            or resource.kind is not resource_kind.RULE_PROFILE
        ):
            return None
        versions.append(str(resource.version))
    canonical = f"rdfs-v1@{versions[0]}" + (
        f"+owl2rl-kestrel-v1@{versions[1]}" if len(versions) == 2 else ""
    )
    if value != canonical:
        return None
    return versions[0], versions[1] if len(versions) == 2 else None


def _registered_ontology(value: Any, resources: Any, resource_kind: Any):
    """Return the local ontology behind the producer's namespace/version label."""
    if not isinstance(value, str):
        return None
    matches = [
        resource
        for resource in resources
        if resource.kind is resource_kind.ONTOLOGY
        and value == f"{resource.namespace}@{resource.version}"
    ]
    return matches[0] if len(matches) == 1 else None


def _is_registered_inference_profile(
    value: Any,
    *,
    ontology: Any,
    rule_versions: Tuple[str, Optional[str]] | None,
    registry_contract: str,
) -> bool:
    """Check the producer hash against exact local ontology/rule pins."""
    if not isinstance(value, str) or ontology is None or rule_versions is None:
        return False
    try:
        from kestrel_sovereign.knowledge.assertion import OntologyRef
        from kestrel_sovereign.knowledge.inference import InferenceProfile

        expected = InferenceProfile(
            OntologyRef(
                ontology.namespace,
                str(ontology.version),
                ontology.sha256,
                registry_contract,
            ),
            rule_versions[0],
            rule_versions[1],
        ).key
    except (TypeError, ValueError):
        return False
    return value == expected


def _semantic_maintenance_repair_guidance(value: Any) -> str:
    """Return a fixed next action for partial or unavailable maintenance.

    These are action *codes*, not exception echoes.  The operator runbook maps
    them to the bounded retry/repair commands, so a failure cannot expose an
    assertion, SQL fragment, tenant, or source locator in an HTTP invoke reply.
    """
    if not isinstance(value, Mapping):
        return "inspect_semantic_configuration"
    status = _semantic_maintenance_status(value.get("status"))
    reason = _semantic_maintenance_reason(value.get("reason"))
    if status in {"complete", "no_op", "disabled"}:
        return "none"
    if reason == "semantic_maintenance_busy":
        return "wait_for_active_maintenance_lease"
    if reason in {
        "semantic_maintenance_capability_unavailable",
        "semantic_maintenance_capability_mismatch",
        "semantic_maintenance_validation_capability_unavailable",
    }:
        return "check_semantic_profile_pins"
    if status == "partial":
        return "rerun_bounded_maintenance"
    if status == "failed":
        return "inspect_semantic_configuration"
    return "rerun_bounded_maintenance"


def _semantic_maintenance_diagnostics(value: Any) -> Optional[Dict[str, Any]]:
    """Create the one content-free structured diagnostic for live maintenance."""
    if not isinstance(value, Mapping):
        return None
    status = _semantic_maintenance_status(value.get("status"))
    return {
        "status": status,
        "reason": _semantic_maintenance_reason(value.get("reason")),
        "checkpoint": {
            "source_generation": _bounded_summary_number(value.get("source_generation")),
            "checkpoint_generation": _bounded_summary_number(
                value.get("checkpoint_generation")
            ),
        },
        "backlog": {
            "assertions": _bounded_summary_number(value.get("backlog_assertions")),
            "reports": _bounded_summary_number(value.get("backlog_reports")),
        },
        "partial": status == "partial",
        "repair_guidance": _semantic_maintenance_repair_guidance(value),
        "active_capabilities": list(_semantic_maintenance_active_capabilities(value)),
    }


def _render_semantic_maintenance_summary(value: Any) -> Optional[str]:
    """Render the fixed, content-free semantic-maintenance text block.

    This is intentionally the only human renderer for the maintenance map.
    It is used by :class:`SleepReport` and therefore reaches authenticated
    ``!sleep --consolidate-only`` calls through both the command and HTTP
    invoke paths.  The field allowlist and fixed output shape keep it
    deterministic, bounded, and safe to show to an operator.
    """
    if not isinstance(value, Mapping):
        return None

    diagnostics = _semantic_maintenance_diagnostics(value)
    if diagnostics is None:  # pragma: no cover - guarded immediately above.
        return None
    status = str(diagnostics["status"])
    reason = str(diagnostics["reason"])
    capability_count, capability_digest = _semantic_maintenance_capability_summary(
        value.get("capability_versions")
    )
    lines = [
        "  Semantic maintenance:",
        f"    status: {status}",
        f"    reason: {reason}",
        "    generations: "
        f"source={_bounded_summary_number(value.get('source_generation'))} "
        f"checkpoint={_bounded_summary_number(value.get('checkpoint_generation'))}",
        "    changes: "
        f"consumed={_bounded_summary_number(value.get('changes_consumed'))} "
        f"validated={_bounded_summary_number(value.get('assertions_validated'))} "
        f"inferred={_bounded_summary_number(value.get('assertions_inferred'))} "
        f"retracted={_bounded_summary_number(value.get('assertions_retracted'))}",
        f"    contradictions: {_bounded_summary_number(value.get('contradictions'))}",
        f"    reports: created={_bounded_summary_number(value.get('reports_created'))}",
        "    backlog: "
        f"assertions={_bounded_summary_number(value.get('backlog_assertions'))} "
        f"reports={_bounded_summary_number(value.get('backlog_reports'))}",
        f"    duration: {_bounded_summary_number(value.get('duration_ms'))}ms",
        "    capabilities: "
        f"versions={capability_count} digest={capability_digest}",
        "    active capabilities: "
        + (
            ", ".join(diagnostics["active_capabilities"])
            if diagnostics["active_capabilities"]
            else "unavailable"
        ),
        f"    repair guidance: {diagnostics['repair_guidance']}",
    ]
    # Every interpolated value above is independently bounded.  Retain this
    # final cap as a defense-in-depth contract for future edits.
    return "\n".join(lines)[:_SEMANTIC_MAINTENANCE_SUMMARY_MAX_CHARS]


class SleepHookPhase(str, Enum):
    """Ordered post-consolidation phases for declarative sleep hooks.

    The phases deliberately describe data boundaries rather than feature
    classes.  A feature can therefore participate without core knowing which
    package provides it.  ``REFLECTION`` is an alias retained for hooks whose
    work is more naturally described as reflection than knowledge extraction.
    """

    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    REFLECTION = "knowledge_extraction"
    SEMANTIC_MAINTENANCE = "semantic_maintenance"
    TRAINING_PREPARATION = "training_preparation"
    TRAINING = "training"

    @classmethod
    def coerce(cls, value: "SleepHookPhase | str") -> "SleepHookPhase":
        """Return a phase while accepting the documented string spellings."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        if normalized == "reflection":
            normalized = cls.KNOWLEDGE_EXTRACTION.value
        return cls(normalized)

    @classmethod
    def ordered(cls) -> Tuple["SleepHookPhase", ...]:
        """Return the stable execution order, without enum aliases."""
        return (
            cls.KNOWLEDGE_EXTRACTION,
            cls.SEMANTIC_MAINTENANCE,
            cls.TRAINING_PREPARATION,
            cls.TRAINING,
        )


class PrerequisiteFailurePolicy(str, Enum):
    """How a declared ``after`` prerequisite treats an unsuccessful result.

    ``BLOCK`` is the safe default: a consumer cannot run against stale input.
    ``SKIP`` expresses the same safety boundary as an expected no-op. A missing
    ``after`` dependency is always a configuration error; use
    ``optional_after`` when the provider itself is optional.
    """

    BLOCK = "block"
    SKIP = "skip"

    @classmethod
    def coerce(
        cls, value: "PrerequisiteFailurePolicy | str"
    ) -> "PrerequisiteFailurePolicy":
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class SleepHookStatus(str, Enum):
    """Content-free terminal statuses for a hook invocation."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class SleepHookContract:
    """Declarative identity and ordering for a post-consolidation sleep hook.

    Attach one of these as ``sleep_hook_contract`` on a hook instance or class.
    ``after`` dependencies must be present and complete successfully by
    default. ``prerequisite_failure_policy`` chooses whether a failed required
    prerequisite is reported as blocked or skipped; neither policy runs stale
    downstream work.
    ``optional_after`` dependencies only constrain order when they are present.
    ``before`` is the inverse spelling and must name a registered hook: its
    target becomes a required consumer of this hook. ``optional_before``
    creates an optional consumer edge. A legacy hook can expose a stable
    ``sleep_hook_id`` string without adopting this contract; an annotated hook
    may name that ID in ``after`` while the legacy hook otherwise retains its
    registration-order behavior.
    """

    hook_id: str
    phase: SleepHookPhase
    before: Tuple[str, ...] = ()
    after: Tuple[str, ...] = ()
    optional_before: Tuple[str, ...] = ()
    optional_after: Tuple[str, ...] = ()
    prerequisite_failure_policy: PrerequisiteFailurePolicy = (
        PrerequisiteFailurePolicy.BLOCK
    )

    def __post_init__(self) -> None:
        hook_id = self.hook_id.strip()
        if not hook_id:
            raise ValueError("Sleep hook contract hook_id must be non-empty")
        object.__setattr__(self, "hook_id", hook_id)
        object.__setattr__(self, "phase", SleepHookPhase.coerce(self.phase))
        object.__setattr__(
            self,
            "prerequisite_failure_policy",
            PrerequisiteFailurePolicy.coerce(self.prerequisite_failure_policy),
        )
        for attribute in (
            "before",
            "after",
            "optional_before",
            "optional_after",
        ):
            raw_values = getattr(self, attribute)
            # A singleton string is a common declarative spelling. Treat it as
            # one dependency rather than accidentally iterating its characters.
            values = (raw_values,) if isinstance(raw_values, str) else tuple(raw_values)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(
                    f"Sleep hook contract {attribute} entries must be non-empty strings"
                )
            object.__setattr__(self, attribute, tuple(value.strip() for value in values))


@dataclass(frozen=True)
class SleepHookExecution:
    """Structured, content-free record of one sleep hook's terminal state."""

    hook_id: str
    phase: str
    status: SleepHookStatus
    duration_ms: int
    stage: str
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the JSON-safe shape included in :class:`SleepReport`."""
        return {
            "hook_id": self.hook_id,
            "phase": self.phase,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "stage": self.stage,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _PostConsolidationHook:
    """Internal binding between a registered post-consolidation hook and ID."""

    hook: Any
    hook_id: str
    contract: Optional[SleepHookContract]
    registration_index: int


@dataclass
class _PostConsolidationPlan:
    """Preflight result for deterministic post-consolidation hooks."""

    ordered_ids: List[str]
    hooks_by_id: Dict[str, _PostConsolidationHook]
    prerequisites: Dict[str, Dict[str, bool]]
    blocked_reasons: Dict[str, str]


class _CoreSemanticMaintenanceHook:
    """Expose core semantic work through the same dependency graph as features."""

    sleep_hook_contract = SleepHookContract(
        hook_id="kestrel_sovereign.semantic_maintenance",
        phase=SleepHookPhase.SEMANTIC_MAINTENANCE,
    )

    def __init__(self, report: "SleepReport") -> None:
        self._report = report

    async def on_post_consolidation(
        self, agent: Any, consolidation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        del consolidation_result
        success = await agent._run_semantic_maintenance(self._report)
        return {"success": success}


@dataclass
class SleepReport:
    """Result of a sleep cycle."""
    success: bool
    cid: Optional[str] = None

    # Consolidation stats
    episodes_created: int = 0
    patterns_found: int = 0
    messages_archived: int = 0
    episodes_deleted: int = 0  # forgetting deletion tier (#1674)
    total_messages: int = 0

    # Export stats
    shards_exported: int = 0
    total_size_bytes: int = 0
    storage_tier: str = "local"

    # Sleep-hook stats: one result dict per registered sleep hook (reflection,
    # parametric-self, ...). Lists rather than a single dict now that the sleep
    # cycle dispatches a list of hooks.
    pre_reflection: List[Dict[str, Any]] = field(default_factory=list)
    post_reflection: List[Dict[str, Any]] = field(default_factory=list)
    insights_generated: int = 0

    # Content-free execution diagnostics. ``pre_reflection`` and
    # ``post_reflection`` retain their established raw hook-result shapes for
    # existing JSON consumers; this adds stable identity, phase, status, and
    # timing without changing either legacy field.
    hook_results: List[SleepHookExecution] = field(default_factory=list)

    # Semantic materialization is core maintenance rather than a feature hook:
    # it must still advance on an idle cycle where reflection is skipped.
    # The shape intentionally contains only operational metadata, never an
    # assertion term or source text.
    semantic_inference: Optional[Dict[str, Any]] = None

    # Incremental validation, audit, and inference is reported separately from
    # the legacy inference-only field.  Both mappings contain only aggregate
    # counts, status, and capability versions.
    semantic_maintenance: Optional[Dict[str, Any]] = None

    # Timing
    consolidation_ms: int = 0
    export_ms: int = 0
    reflection_ms: int = 0

    # Incorporation info (cryostasis may trigger incorporation)
    incorporation_attempted: bool = False
    incorporation_success: bool = False
    incorporation_package_hash: Optional[str] = None

    # Error info
    error: Optional[str] = None
    #: The first terminal failure's content-free code (see _record_failure_code).
    failure_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "success": self.success,
            "cid": self.cid,
            "consolidation": {
                "episodes_created": self.episodes_created,
                "patterns_found": self.patterns_found,
                "messages_archived": self.messages_archived,
                "episodes_deleted": self.episodes_deleted,
                "total_messages": self.total_messages,
                "duration_ms": self.consolidation_ms,
            },
            "reflection": {
                "pre_reflection": self.pre_reflection,
                "post_reflection": self.post_reflection,
                "insights_generated": self.insights_generated,
                "duration_ms": self.reflection_ms,
            },
            "hook_results": [result.to_dict() for result in self.hook_results],
            "semantic_inference": self.semantic_inference,
            "semantic_maintenance": self.semantic_maintenance,
            "semantic_maintenance_diagnostics": self.semantic_maintenance_diagnostics(),
            "export": {
                "shards_exported": self.shards_exported,
                "total_size_bytes": self.total_size_bytes,
                "storage_tier": self.storage_tier,
                "duration_ms": self.export_ms,
            },
            "incorporation": {
                "attempted": self.incorporation_attempted,
                "success": self.incorporation_success,
                "package_hash": self.incorporation_package_hash,
            },
            "error": self.error,
            "failure_code": self.failure_code,
        }

    def semantic_maintenance_summary(self) -> Optional[str]:
        """Return the bounded aggregate block for authenticated sleep output.

        The raw ``semantic_maintenance`` map remains available through
        :meth:`to_dict` for governed programmatic consumers.  Human-facing
        command and HTTP-invoke output must use this method instead: it emits
        only fixed aggregate fields and a capability digest, never assertion
        content, identifiers, tenant/provenance details, raw errors, or the
        capability map itself.
        """
        return _render_semantic_maintenance_summary(self.semantic_maintenance)

    def semantic_maintenance_diagnostics(self) -> Optional[Dict[str, Any]]:
        """Return the content-free operational view used by live invoke output.

        The raw maintenance result is retained for trusted programmatic callers,
        while this method is the safe presentation contract: active verified
        profiles, checkpoint/backlog state, partial status, and fixed repair
        guidance are visible without assertion content or identifiers.
        """
        return _semantic_maintenance_diagnostics(self.semantic_maintenance)

    def __str__(self) -> str:
        """Human-readable sleep summary with safe maintenance observability."""
        maintenance_summary = self.semantic_maintenance_summary()
        if not self.success:
            failure_reason = self.failure_code or _sleep_failure_reason(self.error)
            maintenance_status = _semantic_maintenance_status(
                self.semantic_maintenance.get("status")
                if isinstance(self.semantic_maintenance, Mapping)
                else None
            )
            if failure_reason is not None:
                lines = [f"Sleep failed: {failure_reason}"]
            elif maintenance_status == "partial":
                # A bounded maintenance unit can be intentionally incomplete.
                # Its report often has no separate legacy ``error`` field, so
                # do not degrade this operator-visible state to ``None``.
                lines = ["Sleep incomplete: semantic maintenance is partial."]
            elif maintenance_status == "failed":
                lines = ["Sleep failed: semantic_maintenance_failed"]
            else:
                lines = ["Sleep failed: unavailable"]
            if maintenance_summary:
                lines.append(maintenance_summary)
            return "\n".join(lines)

        lines = [
            "Sleep cycle complete:",
            f"  Episodes created: {self.episodes_created}",
            f"  Patterns found: {self.patterns_found}",
            f"  Memories archived: {self.messages_archived}",
            f"  Insights generated: {self.insights_generated}",
            f"  Shards exported: {self.shards_exported}",
            f"  Storage tier: {self.storage_tier}",
        ]
        if self.cid:
            lines.append(f"  CID: {self.cid}")
        if maintenance_summary:
            lines.append(maintenance_summary)
        return "\n".join(lines)


class SleepMixin:
    """
    Mixin class providing sleep/consolidation methods.

    Sleep combines:
    1. Memory consolidation (via MemoryConsolidator)
    2. Sovereignty export (via SovereignStorageAdapter)

    The result is a CID that represents the agent's complete state,
    which can be used for restoration or migration.

    Callback:
        Set `on_sleep_complete` to a coroutine that receives (cid: str, report: SleepReport).
        This allows platforms to update their database with the new CID.
    """

    # Optional callback for platforms to hook into sleep completion
    # Set this to an async function: async def callback(cid: str, report: SleepReport)
    on_sleep_complete: Optional[Callable] = None

    # Sleep hooks: features register ``*SleepHook`` instances here to run
    # pre-sleep / post-consolidation work during sleep (reflection,
    # parametric-self, ...). A hook may implement ``on_pre_sleep(agent)`` and/or
    # ``on_post_consolidation(agent, consolidation_result)``. Initialized
    # per-instance in KestrelAgent; iterated by ``sleep()``.
    sleep_hooks: Optional[List[Any]] = None

    async def sleep(
        self,
        tier: str = "ipfs",
        skip_consolidation: bool = False,
        skip_export: bool = False,
        skip_reflection: bool = False,
    ) -> SleepReport:
        """
        Execute a full sleep cycle.

        The sleep cycle now includes reflection for self-improvement:
        1. Pre-sleep reflection (analyze current session)
        2. Memory consolidation (create episodes, archive decayed)
        3. Post-consolidation reflection (deeper analysis with episodes)
        4. Sovereignty export (backup to IPFS/Filecoin)

        Args:
            tier: Storage tier for export ("local", "ipfs", "filecoin")
            skip_consolidation: Skip memory consolidation (just export)
            skip_export: Skip sovereignty export (just consolidate)
            skip_reflection: Skip reflection hooks

        Returns:
            SleepReport with details of operations performed
        """
        import time
        from kestrel_sovereign.filecoin_adapter import StorageTier
        from kestrel_sovereign.storage.memory_system import (
            MemoryConsolidationTimeoutError,
        )

        report = SleepReport(success=False)
        reflection_start = time.time()
        consolidation_succeeded = False
        export_succeeded = False
        artifact_sweep_succeeded = True

        # The nightly scheduler invokes this same sleep path.  Sweep governed
        # semantic artifacts even when inference maintenance is disabled so a
        # retention deadline never depends on a consumer attempting a read.
        artifact_sweep = getattr(
            getattr(self, "storage", None),
            "sweep_expired_governed_semantic_artifacts",
            None,
        )
        if callable(artifact_sweep):
            try:
                await artifact_sweep()
            except Exception:
                logger.warning("Governed semantic artifact expiry sweep failed")
                report.error = "semantic_artifact_expiry_sweep_failed"
                _record_failure_code(report, "semantic_artifact_expiry_sweep_failed")
                artifact_sweep_succeeded = False

        # Note: Privacy mode checks are handled by the storage layer.
        # - EPHEMERAL/ISOLATED: Storage will raise PrivacyViolationError on export
        # - Consolidation is always allowed (reorganizes existing data)
        # We don't block sleep here - let it try and handle errors gracefully.

        tier_map = {
            "local": StorageTier.LOCAL_ONLY,
            "ipfs": StorageTier.IPFS,
            "filecoin": StorageTier.FILECOIN,
            "lighthouse": StorageTier.IPFS,  # Lighthouse uses IPFS protocol
        }
        storage_tier = tier_map.get(tier.lower(), StorageTier.LOCAL_ONLY)
        report.storage_tier = tier

        # 0. Pre-sleep hooks (analyze current session before consolidation).
        # Pre-sleep remains registration ordered: the declarative phase contract
        # applies only to the post-consolidation data boundary.
        if not skip_reflection and self.sleep_hooks:
            await self._run_pre_sleep_hooks(report)

        # 1. Memory Consolidation
        if not skip_consolidation:
            start = time.time()
            consolidation_result: Optional[Dict[str, Any]] = None
            try:
                consolidation_result = await self._consolidate_memories()
                unavailability_reason = self._post_consolidation_unavailability_reason(
                    consolidation_result
                )
                if unavailability_reason is None:
                    report.episodes_created = consolidation_result.get("episodes_created", 0)
                    report.patterns_found = consolidation_result.get("patterns_found", 0)
                    report.messages_archived = consolidation_result.get("messages_archived", 0)
                    report.episodes_deleted = consolidation_result.get("episodes_deleted", 0)
                    report.total_messages = consolidation_result.get("total_messages_processed", 0)
                    consolidation_succeeded = True
                    logger.info(
                        f"Consolidation complete: {report.episodes_created} episodes, "
                        f"{report.messages_archived} archived, "
                        f"{report.episodes_deleted} forgotten"
                    )
                else:
                    # MemorySystem.consolidate() reports some failures and
                    # privacy-gated passes as dictionaries.  Do not let their
                    # zero-valued counters masquerade as a completed sleep
                    # cycle, and do not surface provider or memory content in
                    # this operator-facing report.
                    if report.error:
                        report.error += f"; {unavailability_reason}"
                    else:
                        report.error = unavailability_reason
                    _record_failure_code(report, unavailability_reason)
            except MemoryConsolidationTimeoutError:
                logger.error("Consolidation timed out", exc_info=True)
                if report.error:
                    report.error += "; consolidation_failed"
                else:
                    report.error = "consolidation_failed"
                _record_failure_code(report, "consolidation_failed")
                report.consolidation_ms = int((time.time() - start) * 1000)
                report.reflection_ms = (
                    int((time.time() - reflection_start) * 1000)
                    - report.consolidation_ms
                )
                # A cancelled aiosqlite statement can still be draining in its
                # worker thread. Every later database access is fenced behind
                # that cleanup, so continuing into hooks or export would put
                # the dispatcher-owned MEMORY lock back on an unbounded wait.
                return report
            except Exception:
                logger.error("Consolidation failed", exc_info=True)
                if report.error:
                    report.error += "; consolidation_failed"
                else:
                    report.error = "consolidation_failed"
                _record_failure_code(report, "consolidation_failed")
                # Continue to export anyway - partial sleep is better than none
            report.consolidation_ms = int((time.time() - start) * 1000)

            # 1.5 Post-consolidation hooks use the new episodes. Annotated hooks
            # are phase- and dependency-ordered; legacy hooks retain their
            # registration-order, continue-on-error behavior.
            if (
                (not skip_reflection and self.sleep_hooks)
                or self._semantic_maintenance_required()
            ):
                unavailability_reason = self._post_consolidation_unavailability_reason(
                    consolidation_result
                )
                if unavailability_reason is not None:
                    self._record_unavailable_post_consolidation_hooks(
                        report,
                        reason=unavailability_reason,
                        include_semantic_maintenance=self._semantic_maintenance_required(),
                    )
                else:
                    await self._run_post_consolidation_hooks(
                        consolidation_result,
                        report,
                        include_feature_hooks=not skip_reflection,
                    )

        # The core semantic service is a declared post-consolidation hook, so
        # phase-ordered training consumers see a failed/incomplete maintenance
        # prerequisite before they can consume a corpus.  An explicitly
        # requested skip still runs the service for compatibility with manual
        # repair callers, but has no post-consolidation feature consumers.
        if skip_consolidation and self._semantic_maintenance_required():
            semantic_maintenance_succeeded = await self._run_semantic_maintenance(report)
        elif self._semantic_maintenance_required():
            semantic_maintenance_succeeded = self._semantic_maintenance_successful(report)
        else:
            semantic_maintenance_succeeded = True

        # Record total reflection time
        report.reflection_ms = int((time.time() - reflection_start) * 1000) - report.consolidation_ms

        # 2. Sovereignty Export
        if not skip_export:
            start = time.time()
            try:
                export_result = await self._export_sovereignty(storage_tier)
                report.cid = export_result.get("cid")
                report.shards_exported = export_result.get("shards_exported", 0)
                report.total_size_bytes = export_result.get("total_size_bytes", 0)
                export_succeeded = report.cid is not None
                logger.info(f"Export complete: CID={report.cid}")
            except Exception as e:
                logger.error(f"Export failed: {e}")
                if report.error:
                    report.error += f"; Export failed: {e}"
                else:
                    report.error = f"Export failed: {e}"
                _record_failure_code(report, "export_failed")
            report.export_ms = int((time.time() - start) * 1000)

        # Success is an actual completed consolidation or export, never merely
        # a requested operation with zero counters.  In particular,
        # MemorySystem.consolidate() reports some failures as ``{"error": ...}``
        # and the scheduler normally skips export.
        report.success = (
            consolidation_succeeded or export_succeeded
        ) and semantic_maintenance_succeeded and artifact_sweep_succeeded

        # Invoke callback if set (allows platform to update latest_cid)
        if report.success and report.cid and self.on_sleep_complete:
            try:
                await self.on_sleep_complete(report.cid, report)
            except Exception as e:
                logger.warning(f"on_sleep_complete callback failed: {e}")
                # Don't fail the sleep for callback errors

        return report

    def _semantic_maintenance_required(self) -> bool:
        return bool(
            getattr(self, "semantic_inference_configured", False)
            or getattr(self, "semantic_maintenance_configured", False)
            or getattr(self, "semantic_capabilities_configured", False)
            or getattr(self, "semantic_inference_profile", None) is not None
        )

    @staticmethod
    def _semantic_maintenance_successful(report: SleepReport) -> bool:
        """Return the core hook's declared outcome, never infer one from a map."""
        core_results = [
            item
            for item in report.hook_results
            if item.hook_id == "kestrel_sovereign.semantic_maintenance"
        ]
        return bool(core_results) and core_results[-1].status is SleepHookStatus.SUCCESS

    async def _run_semantic_maintenance(
        self,
        report: SleepReport,
    ) -> bool:
        """Run one governed semantic-maintenance unit, if configured."""
        profile = getattr(self, "semantic_inference_profile", None)
        inference_configured = bool(
            getattr(self, "semantic_inference_configured", False)
        )
        maintenance_configured = bool(
            getattr(self, "semantic_maintenance_configured", False)
        )
        capabilities_configured = bool(
            getattr(self, "semantic_capabilities_configured", False)
        )
        if (
            profile is None
            and not inference_configured
            and not maintenance_configured
            and not capabilities_configured
        ):
            return True
        storage = getattr(self, "storage", None)
        if profile is None and inference_configured:
            revoke = getattr(storage, "revoke_semantic_inference", None)
            if not callable(revoke):
                report.semantic_inference = {
                    "status": "failed",
                    "reason": "semantic_storage_unavailable",
                }
                report.semantic_maintenance = dict(report.semantic_inference)
                logger.warning(
                    "Semantic inference revocation skipped: governed storage is unavailable"
                )
                return False
            try:
                result = await revoke()
            except Exception:
                report.semantic_inference = {
                    "status": "failed",
                    "reason": "semantic_inference_revocation_failed",
                }
                report.semantic_maintenance = dict(report.semantic_inference)
                report.error = (
                    f"{report.error}; semantic_inference_revocation_failed"
                    if report.error
                    else "semantic_inference_revocation_failed"
                )
                _record_failure_code(report, "semantic_inference_revocation_failed")
                logger.warning("Semantic inference revocation failed")
                return False
            report.semantic_inference = {
                "status": "disabled",
                "retracted_assertions": result.retracted_assertions,
                "deactivated_derivations": result.deactivated_derivations,
                "generation": result.generation,
            }
            if not maintenance_configured:
                report.semantic_maintenance = dict(report.semantic_inference)
                return True
        maintain = getattr(storage, "run_semantic_maintenance", None)
        if callable(maintain):
            try:
                maintenance_kwargs = {
                    "inference_limits": getattr(self, "semantic_inference_limits", None),
                    "maintenance_limits": getattr(self, "semantic_maintenance_limits", None),
                }
                semantic_capabilities = getattr(self, "semantic_capabilities", None)
                if semantic_capabilities is not None:
                    maintenance_kwargs["semantic_capabilities"] = semantic_capabilities
                    # Exercise the storage-owned RDF runtime on the real
                    # ``!sleep`` path.  This prevents an agent config from
                    # being represented only by maintenance diagnostics while
                    # a different codec silently handles later graph reads.
                    runtime_report = getattr(storage, "semantic_rdf_capability_report", None)
                    if callable(runtime_report):
                        active_runtime = runtime_report()
                        if not semantic_capabilities.rdf_runtime_matches(active_runtime):
                            raise RuntimeError("semantic_rdf_runtime_capability_mismatch")
                result = await maintain(
                    profile,
                    **maintenance_kwargs,
                )
            except Exception:
                report.semantic_maintenance = {
                    "status": "failed",
                    "reason": "semantic_maintenance_failed",
                }
                report.semantic_inference = dict(report.semantic_maintenance)
                report.error = (
                    f"{report.error}; semantic_maintenance_failed"
                    if report.error
                    else "semantic_maintenance_failed"
                )
                _record_failure_code(report, "semantic_maintenance_failed")
                logger.warning("Semantic maintenance failed")
                return False
            report.semantic_maintenance = result.to_mapping()
            # Keep the established field available to API clients while the
            # richer maintenance report is adopted.  A validation-only run
            # must not masquerade as inference, and a preceding explicit
            # revocation remains observable as ``disabled``.
            if profile is not None:
                report.semantic_inference = {
                    "status": result.status.value,
                    "incomplete_reason": result.reason,
                    "source_generation": result.source_generation,
                    "checkpoint_generation": result.checkpoint_generation,
                    "generated_assertions": result.assertions_inferred,
                    "retracted_assertions": result.assertions_retracted,
                }
            return result.status.value in {"complete", "no_op"}

        report.semantic_maintenance = {
            "status": "failed",
            "reason": "semantic_storage_unavailable",
        }
        report.semantic_inference = dict(report.semantic_maintenance)
        report.error = (
            f"{report.error}; semantic_storage_unavailable"
            if report.error
            else "semantic_storage_unavailable"
        )
        _record_failure_code(report, "semantic_storage_unavailable")
        logger.warning("Semantic maintenance skipped: governed storage is unavailable")
        return False

    async def _run_semantic_inference_maintenance(
        self,
        report: SleepReport,
    ) -> bool:
        """Compatibility alias for callers of the prior private method name."""
        return await self._run_semantic_maintenance(report)

    @staticmethod
    def _sleep_hook_contract(hook: Any) -> Optional[SleepHookContract]:
        """Return an opt-in contract without imposing a base class on features."""
        contract = getattr(hook, "sleep_hook_contract", None)
        return contract if isinstance(contract, SleepHookContract) else None

    @staticmethod
    def _legacy_sleep_hook_id(hook: Any, registration_index: int) -> str:
        """Return a legacy hook's stable opt-in ID or run-local fallback.

        Legacy hooks remain unannotated by default.  A hook that an annotated
        consumer must depend on can set ``sleep_hook_id = "package.hook"``;
        that public ID is stable across registrations and is valid in a
        :class:`SleepHookContract` dependency.  Hooks without it preserve the
        existing diagnostic-only, registration-indexed identity.
        """
        declared_id = getattr(hook, "sleep_hook_id", None)
        if declared_id is not None:
            if not isinstance(declared_id, str) or not declared_id.strip():
                raise ValueError(
                    "Legacy sleep hook sleep_hook_id must be a non-empty string"
                )
            return declared_id.strip()
        hook_type = type(hook)
        return (
            f"legacy:{registration_index}:"
            f"{hook_type.__module__}.{hook_type.__qualname__}"
        )

    @staticmethod
    def _hook_outcome(
        result: Any,
    ) -> Tuple[SleepHookStatus, Optional[str], Optional[int]]:
        """Classify and validate a hook result without exposing its contents.

        The legacy hook result dictionaries remain part of ``SleepReport`` for
        compatibility, but their control fields are a small protocol boundary:
        a malformed value must not escape a hook's error-isolation boundary or
        be aggregated into the report.  ``None`` means that the hook did not
        explicitly report successful insight generation.
        """
        if not isinstance(result, dict):
            return SleepHookStatus.FAILED, "invalid_hook_result", None

        has_success = "success" in result
        has_skipped = "skipped" in result
        success = result.get("success")
        skipped = result.get("skipped")
        insights_generated = result.get("insights_generated", 0)
        if (
            (has_success and not isinstance(success, bool))
            or (has_skipped and not isinstance(skipped, bool))
            or isinstance(insights_generated, bool)
            or not isinstance(insights_generated, int)
        ):
            return SleepHookStatus.FAILED, "invalid_hook_result", None
        if skipped:
            return SleepHookStatus.SKIPPED, "hook_reported_skipped", None
        if success is False:
            return SleepHookStatus.FAILED, "hook_reported_failure", None
        if success is True:
            return SleepHookStatus.SUCCESS, None, insights_generated
        return SleepHookStatus.SUCCESS, None, None

    @staticmethod
    def _post_consolidation_unavailability_reason(
        consolidation_result: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Return why post hooks cannot consume this consolidation result.

        ``MemorySystem.consolidate()`` deliberately reports failed and
        privacy-skipped work as dictionaries so callers can finish their
        surrounding sleep/export cycle.  Those dictionaries are not a fresh
        corpus boundary: no post-consolidation hook may acknowledge or consume
        them.  Keep the reason content-free because the original error can
        contain memory or provider details.
        """
        if consolidation_result is None:
            return "consolidation_failed"
        if "error" in consolidation_result:
            return "consolidation_failed"
        if consolidation_result.get("skipped"):
            return "consolidation_skipped"
        return None

    @staticmethod
    def _record_hook_execution(
        report: SleepReport,
        *,
        hook_id: str,
        phase: str,
        status: SleepHookStatus,
        duration_ms: int,
        stage: str,
        reason: Optional[str] = None,
    ) -> None:
        """Add a content-free, JSON-serializable hook execution record."""
        report.hook_results.append(
            SleepHookExecution(
                hook_id=hook_id,
                phase=phase,
                status=status,
                duration_ms=duration_ms,
                stage=stage,
                reason=reason,
            )
        )

    async def _run_pre_sleep_hooks(self, report: SleepReport) -> None:
        """Run pre-sleep hooks in their established registration order."""
        import time

        for registration_index, hook in enumerate(self.sleep_hooks or []):
            handler = getattr(hook, "on_pre_sleep", None)
            if not callable(handler):
                continue
            contract = self._sleep_hook_contract(hook)
            hook_id = (
                contract.hook_id
                if contract is not None
                else self._legacy_sleep_hook_id(hook, registration_index)
            )
            phase = contract.phase.value if contract is not None else "legacy"
            start = time.perf_counter()
            try:
                pre_result = await handler(self)
                duration_ms = int((time.perf_counter() - start) * 1000)
                status, reason, insights_generated = self._hook_outcome(pre_result)
                report.pre_reflection.append(pre_result)
                self._record_hook_execution(
                    report,
                    hook_id=hook_id,
                    phase=phase,
                    status=status,
                    duration_ms=duration_ms,
                    stage="pre_sleep",
                    reason=reason,
                )
                if insights_generated is not None:
                    report.insights_generated += insights_generated
                    logger.info(
                        "Pre-sleep hook completed: insights=%d",
                        insights_generated,
                    )
            except Exception:  # Hook errors are isolated from the sleep cycle.
                duration_ms = int((time.perf_counter() - start) * 1000)
                self._record_hook_execution(
                    report,
                    hook_id=hook_id,
                    phase=phase,
                    status=SleepHookStatus.FAILED,
                    duration_ms=duration_ms,
                    stage="pre_sleep",
                    reason="hook_exception",
                )
                # Exception text can contain user content. Keep diagnostics
                # content-free while preserving the historical continuation.
                logger.warning("Pre-sleep hook failed")
                continue

    def _record_unavailable_post_consolidation_hooks(
        self,
        report: SleepReport,
        *,
        reason: str,
        include_semantic_maintenance: bool = False,
    ) -> None:
        """Report hooks skipped because consolidation produced no safe input.

        Previously an unsuccessful consolidation left no local result for the
        hook call, so every post hook was effectively skipped.  Preserve that
        safety boundary while making it observable in the structured report.
        """
        if include_semantic_maintenance:
            report.semantic_maintenance = {
                "status": "partial",
                "reason": reason,
            }
        hooks = list(self.sleep_hooks or [])
        if include_semantic_maintenance:
            hooks.append(_CoreSemanticMaintenanceHook(report))
        for registration_index, hook in enumerate(hooks):
            if not callable(getattr(hook, "on_post_consolidation", None)):
                continue
            contract = self._sleep_hook_contract(hook)
            hook_id = (
                contract.hook_id
                if contract is not None
                else self._legacy_sleep_hook_id(hook, registration_index)
            )
            phase = contract.phase.value if contract is not None else "legacy"
            self._record_hook_execution(
                report,
                hook_id=hook_id,
                phase=phase,
                status=SleepHookStatus.SKIPPED,
                duration_ms=0,
                stage="post_consolidation",
                reason=reason,
            )

    def _build_post_consolidation_plan(
        self, hooks: List[_PostConsolidationHook]
    ) -> _PostConsolidationPlan:
        """Validate and topologically order post-consolidation hooks before run.

        Annotated hooks are deterministic by phase and ID. Legacy hooks are
        graph vertices too, but registration order is preserved *only among
        legacy hooks*. Cross-group order comes exclusively from declared
        dependencies so a legacy registration position cannot form a false
        cycle with phase ordering. Invalid nodes are reported as blocked,
        while independent nodes still receive their normal chance to run.
        """
        phase_order = {
            phase: index for index, phase in enumerate(SleepHookPhase.ordered())
        }

        def entry_sort_key(entry: _PostConsolidationHook) -> Tuple[int, int, str]:
            if entry.contract is None:
                return (1, entry.registration_index, entry.hook_id)
            return (0, phase_order[entry.contract.phase], entry.hook_id)

        by_id_groups: Dict[str, List[_PostConsolidationHook]] = {}
        for hook in hooks:
            by_id_groups.setdefault(hook.hook_id, []).append(hook)

        hooks_by_id: Dict[str, _PostConsolidationHook] = {
            hook_id: grouped[0]
            for hook_id, grouped in by_id_groups.items()
            if len(grouped) == 1
        }
        duplicate_ids = {
            hook_id for hook_id, grouped in by_id_groups.items() if len(grouped) > 1
        }
        blocked_reasons: Dict[str, str] = {
            hook_id: f"duplicate_hook_id:{hook_id}" for hook_id in duplicate_ids
        }
        prerequisites: Dict[str, Dict[str, bool]] = {
            hook_id: {} for hook_id in hooks_by_id
        }
        edges: Dict[str, set[str]] = {hook_id: set() for hook_id in hooks_by_id}

        def add_edge(
            prerequisite_id: str,
            dependent_id: str,
            *,
            require_success: bool,
        ) -> None:
            edges[prerequisite_id].add(dependent_id)
            prior = prerequisites[dependent_id].get(prerequisite_id, False)
            prerequisites[dependent_id][prerequisite_id] = (
                prior or require_success
            )

        # Declared phases order annotated work independently of feature
        # registration. A semantic-maintenance failure is a hard data boundary
        # for training preparation and training, so those edges require
        # successful completion rather than merely execution order.
        for source_id, source in hooks_by_id.items():
            if source.contract is None:
                continue
            for target_id, target in hooks_by_id.items():
                if target.contract is None:
                    continue
                if phase_order[source.contract.phase] < phase_order[target.contract.phase]:
                    semantic_to_training = (
                        source.contract.phase is SleepHookPhase.SEMANTIC_MAINTENANCE
                        and target.contract.phase
                        in (
                            SleepHookPhase.TRAINING_PREPARATION,
                            SleepHookPhase.TRAINING,
                        )
                    )
                    add_edge(
                        source_id,
                        target_id,
                        require_success=semantic_to_training,
                    )

        # A duplicate cannot enter ``hooks_by_id`` because it has no
        # unambiguous graph vertex.  That must not erase the semantic
        # maintenance -> training safety boundary, though: the duplicate is
        # recorded as blocked before execution, and every later training
        # consumer treats that blocked semantic prerequisite as unsuccessful.
        # This mirrors the successful-completion edge above without inventing
        # an arbitrary duplicate node in the topological graph.
        for duplicate_id in duplicate_ids:
            duplicate_entries = by_id_groups[duplicate_id]
            if not any(
                entry.contract is not None
                and entry.contract.phase is SleepHookPhase.SEMANTIC_MAINTENANCE
                for entry in duplicate_entries
            ):
                continue
            for target_id, target in hooks_by_id.items():
                if (
                    target.contract is not None
                    and target.contract.phase
                    in (
                        SleepHookPhase.TRAINING_PREPARATION,
                        SleepHookPhase.TRAINING,
                    )
                ):
                    prerequisites[target_id][duplicate_id] = True

        # Preserve registration order within the legacy group.  Never create
        # registration-placement edges across annotated and legacy hooks:
        # phases and declared dependencies own that relationship.
        legacy_entries = sorted(
            (
                entry
                for entry in hooks_by_id.values()
                if entry.contract is None
            ),
            key=lambda entry: entry.registration_index,
        )
        for predecessor, successor in zip(legacy_entries, legacy_entries[1:]):
            add_edge(
                predecessor.hook_id,
                successor.hook_id,
                require_success=False,
            )

        for hook_id, entry in hooks_by_id.items():
            if entry.contract is None:
                continue
            contract = entry.contract
            for dependency_id in contract.after:
                if dependency_id in duplicate_ids:
                    blocked_reasons[hook_id] = (
                        f"duplicate_required_dependency:{dependency_id}"
                    )
                elif dependency_id not in hooks_by_id:
                    blocked_reasons[hook_id] = (
                        f"missing_required_dependency:{dependency_id}"
                    )
                else:
                    add_edge(
                        dependency_id,
                        hook_id,
                        require_success=True,
                    )
            for dependency_id in contract.optional_after:
                if dependency_id in hooks_by_id:
                    add_edge(dependency_id, hook_id, require_success=False)
            for dependent_id in contract.before:
                if dependent_id in duplicate_ids:
                    blocked_reasons[hook_id] = (
                        f"duplicate_required_dependency:{dependent_id}"
                    )
                elif dependent_id not in hooks_by_id:
                    blocked_reasons[hook_id] = (
                        f"missing_required_dependency:{dependent_id}"
                    )
                else:
                    add_edge(hook_id, dependent_id, require_success=True)
            for dependent_id in contract.optional_before:
                if dependent_id in hooks_by_id:
                    add_edge(hook_id, dependent_id, require_success=False)

        indegree = {hook_id: 0 for hook_id in hooks_by_id}
        for targets in edges.values():
            for target_id in targets:
                indegree[target_id] += 1
        ready = [
            (entry_sort_key(entry), hook_id)
            for hook_id, entry in hooks_by_id.items()
            if indegree[hook_id] == 0
        ]
        heapq.heapify(ready)
        ordered_topologically: List[str] = []
        while ready:
            _, hook_id = heapq.heappop(ready)
            ordered_topologically.append(hook_id)
            for target_id in sorted(edges[hook_id]):
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    target = hooks_by_id[target_id]
                    heapq.heappush(
                        ready,
                        (entry_sort_key(target), target_id),
                    )

        cycle_affected = set(hooks_by_id).difference(ordered_topologically)
        for hook_id in cycle_affected:
            blocked_reasons.setdefault(hook_id, "dependency_cycle")

        ordered_ids = ordered_topologically + sorted(
            cycle_affected,
            key=lambda hook_id: entry_sort_key(hooks_by_id[hook_id]),
        )
        return _PostConsolidationPlan(
            ordered_ids=ordered_ids,
            hooks_by_id=hooks_by_id,
            prerequisites=prerequisites,
            blocked_reasons=blocked_reasons,
        )

    async def _run_post_consolidation_hooks(
        self,
        consolidation_result: Dict[str, Any],
        report: SleepReport,
        *,
        include_feature_hooks: bool = True,
    ) -> None:
        """Run dependency-aware hooks while preserving legacy-group order."""
        import time

        registered_hooks = list(self.sleep_hooks or []) if include_feature_hooks else []
        if self._semantic_maintenance_required():
            registered_hooks.append(_CoreSemanticMaintenanceHook(report))
        post_consolidation_hooks: List[_PostConsolidationHook] = []
        for registration_index, hook in enumerate(registered_hooks):
            contract = self._sleep_hook_contract(hook)
            handler = getattr(hook, "on_post_consolidation", None)
            declared_legacy_id = getattr(hook, "sleep_hook_id", None)
            if (
                contract is None
                and not callable(handler)
                and declared_legacy_id is None
            ):
                continue
            hook_id = (
                contract.hook_id
                if contract is not None
                else self._legacy_sleep_hook_id(hook, registration_index)
            )
            post_consolidation_hooks.append(
                _PostConsolidationHook(
                    hook=hook,
                    hook_id=hook_id,
                    contract=contract,
                    registration_index=registration_index,
                )
            )

        plan = self._build_post_consolidation_plan(post_consolidation_hooks)
        status_by_id: Dict[str, SleepHookStatus] = {}

        # Duplicates have no unambiguous identity to enter the graph. Record a
        # blocked result for every registration and never invoke either hook.
        duplicate_groups: Dict[str, List[_PostConsolidationHook]] = {}
        for entry in post_consolidation_hooks:
            duplicate_groups.setdefault(entry.hook_id, []).append(entry)
        phase_order = {
            phase: index for index, phase in enumerate(SleepHookPhase.ordered())
        }
        for hook_id, entries in sorted(duplicate_groups.items()):
            if len(entries) < 2:
                continue
            for entry in sorted(
                entries,
                key=lambda item: (
                    phase_order[item.contract.phase]
                    if item.contract is not None
                    else len(phase_order),
                    item.registration_index,
                ),
            ):
                self._record_hook_execution(
                    report,
                    hook_id=hook_id,
                    phase=(
                        entry.contract.phase.value
                        if entry.contract is not None
                        else "legacy"
                    ),
                    status=SleepHookStatus.BLOCKED,
                    duration_ms=0,
                    stage="post_consolidation",
                    reason=f"duplicate_hook_id:{hook_id}",
                )
            status_by_id[hook_id] = SleepHookStatus.BLOCKED

        for hook_id in plan.ordered_ids:
            entry = plan.hooks_by_id[hook_id]
            phase = (
                entry.contract.phase.value if entry.contract is not None else "legacy"
            )
            block_reason = plan.blocked_reasons.get(hook_id)
            prerequisite_failed = False
            if block_reason is None:
                for prerequisite_id in sorted(plan.prerequisites[hook_id]):
                    require_success = plan.prerequisites[hook_id][prerequisite_id]
                    prerequisite_status = status_by_id.get(prerequisite_id)
                    if require_success and prerequisite_status is not SleepHookStatus.SUCCESS:
                        prerequisite_failed = True
                        observed = (
                            prerequisite_status.value
                            if prerequisite_status is not None
                            else "not_run"
                        )
                        block_reason = (
                            "required_prerequisite_not_successful:"
                            f"{prerequisite_id}:{observed}"
                        )
                        break
            if block_reason is not None:
                status = (
                    SleepHookStatus.SKIPPED
                    if (
                        prerequisite_failed
                        and entry.contract is not None
                        and entry.contract.prerequisite_failure_policy
                        is PrerequisiteFailurePolicy.SKIP
                    )
                    else SleepHookStatus.BLOCKED
                )
                status_by_id[hook_id] = status
                self._record_hook_execution(
                    report,
                    hook_id=hook_id,
                    phase=phase,
                    status=status,
                    duration_ms=0,
                    stage="post_consolidation",
                    reason=block_reason,
                )
                logger.warning(
                    "Post-consolidation hook not run: status=%s",
                    status.value,
                )
                continue

            handler = getattr(entry.hook, "on_post_consolidation", None)
            if not callable(handler):
                status_by_id[hook_id] = SleepHookStatus.SKIPPED
                self._record_hook_execution(
                    report,
                    hook_id=hook_id,
                    phase=phase,
                    status=SleepHookStatus.SKIPPED,
                    duration_ms=0,
                    stage="post_consolidation",
                    reason="missing_post_consolidation_handler",
                )
                continue

            start = time.perf_counter()
            try:
                post_result = await handler(self, consolidation_result)
                duration_ms = int((time.perf_counter() - start) * 1000)
                status, reason, insights_generated = self._hook_outcome(post_result)
                report.post_reflection.append(post_result)
                status_by_id[hook_id] = status
                self._record_hook_execution(
                    report,
                    hook_id=hook_id,
                    phase=phase,
                    status=status,
                    duration_ms=duration_ms,
                    stage="post_consolidation",
                    reason=reason,
                )
                if insights_generated is not None:
                    report.insights_generated += insights_generated
                    logger.info(
                        "Post-consolidation hook completed: insights=%d",
                        insights_generated,
                    )
            except Exception:  # Individual hooks must not abort the sleep cycle.
                duration_ms = int((time.perf_counter() - start) * 1000)
                status_by_id[hook_id] = SleepHookStatus.FAILED
                self._record_hook_execution(
                    report,
                    hook_id=hook_id,
                    phase=phase,
                    status=SleepHookStatus.FAILED,
                    duration_ms=duration_ms,
                    stage="post_consolidation",
                    reason="hook_exception",
                )
                logger.warning("Post-consolidation hook failed")
                continue

    async def _consolidate_memories(self) -> Dict[str, Any]:
        """
        Run memory consolidation through the single MemorySystem chokepoint.

        Routes through ``MemorySystem.consolidate()`` (not the lower-level
        ``MemoryConsolidator.run_consolidation()``) so the sleep cycle inherits
        the SAME flow the manual tool uses: episode creation, pattern
        detection, decay-archival, AND the forgetting deletion tier (#1674).
        This is the point of P3 — one consolidation path, not a per-cron copy.
        Falls back to the raw consolidator only when no MemorySystem is wired.
        """
        memory_system = getattr(self, "memory_system", None)
        if memory_system is not None:
            return await memory_system.consolidate()

        if not hasattr(self, 'memory_consolidator') or not self.memory_consolidator:
            logger.warning("MemoryConsolidator not available, skipping consolidation")
            return {"error": "MemoryConsolidator not initialized"}

        return await self.memory_consolidator.run_consolidation()

    async def _export_sovereignty(self, storage_tier) -> Dict[str, Any]:
        """
        Export agent state to sovereignty storage.

        Uses SovereignStorageAdapter for sharded, encrypted export.
        """
        from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter
        from kestrel_sovereign.filecoin_adapter import FilecoinAdapter
        import os

        # Get user secret for convergent encryption
        user_secret = os.getenv("KESTREL_USER_SECRET", "default-secret")
        if hasattr(self, 'did') and self.did:
            # Include DID for isolation
            user_secret = f"{user_secret}:{self.did}"

        # Get the raw database for direct access
        db = None
        if hasattr(self, '_raw_storage') and self._raw_storage:
            db = self._raw_storage.db
        elif hasattr(self, 'storage') and self.storage:
            if hasattr(self.storage, 'database'):
                db = self.storage.database
            elif hasattr(self.storage, 'db'):
                db = self.storage.db

        if not db:
            raise RuntimeError("No database available for sovereignty export")

        # Create adapter
        filecoin_adapter = FilecoinAdapter()
        adapter = SovereignStorageAdapter(
            db=db,
            user_secret=user_secret,
            filecoin_adapter=filecoin_adapter,
            agent_id=getattr(self, 'did', '') or ''
        )

        # Export
        agent_did = getattr(self, 'did', None) or 'unknown'
        cid = await adapter.export_agent(agent_did, storage_tier=storage_tier)

        # Get export stats (from manifest if available)
        return {
            "cid": cid,
            "shards_exported": len(adapter._last_manifest.shards) if hasattr(adapter, '_last_manifest') else 1,
            "total_size_bytes": sum(s.size_bytes for s in adapter._last_manifest.shards) if hasattr(adapter, '_last_manifest') else 0,
        }

    async def _command_sleep(self, user_input: str) -> str:
        """
        Handle !sleep command.

        Usage:
            !sleep                    - Full sleep (consolidate + export to IPFS)
            !sleep --tier local       - Export to local only
            !sleep --tier filecoin    - Export to Filecoin for permanent storage
            !sleep --consolidate-only - Only run memory consolidation
            !sleep --export-only      - Only run sovereignty export

        When semantic maintenance is configured, the response includes a
        bounded, content-free aggregate summary (status, generations, counters,
        backlog, duration, and a capability digest).  It intentionally omits
        assertion content, identifiers, provenance, tenant data, and raw errors.
        """
        parts = user_input.split()

        tier = "ipfs"
        skip_consolidation = False
        skip_export = False

        for i, part in enumerate(parts):
            if part == "--tier" and i + 1 < len(parts):
                tier = parts[i + 1]
            elif part.startswith("--tier="):
                tier = part.split("=", 1)[1]
            elif part == "--consolidate-only":
                skip_export = True
            elif part == "--export-only":
                skip_consolidation = True

        report = await self.sleep(
            tier=tier,
            skip_consolidation=skip_consolidation,
            skip_export=skip_export,
        )

        return str(report)

    async def cryostasis_sleep(
        self,
        incorporation_params: Optional[Dict[str, Any]] = None,
    ) -> SleepReport:
        """Execute a cryostasis sleep cycle with optional incorporation.

        This is called when the wallet balance drops below the cryostasis
        threshold. Unlike a regular sleep, cryostasis:
        1. Attempts to incorporate the agent as a Wyoming DAO LLC (if affordable)
        2. Runs full memory consolidation
        3. Exports everything to permanent storage (Filecoin)
        4. The legal entity persists even while the agent sleeps

        Args:
            incorporation_params: If provided, attempts incorporation before cryo.
                Expected keys: entity_name, organizer_name, organizer_address,
                registered_agent_name, registered_agent_address.
                If None, generates draft documents bundled in the sovereignty export.

        Returns:
            SleepReport with incorporation details.
        """
        report = SleepReport(success=False)

        # 1. Attempt incorporation if params provided.
        # Resolves LegalFeature via the agent's feature registry rather than
        # importing kestrel_feature_legal directly: feature packages are
        # extensions, not dependencies of sovereign.
        if incorporation_params:
            feature = self.get_feature("legal")
            if feature is None:
                logger.warning(
                    "Pre-cryostasis incorporation requested but no 'legal' "
                    "feature is registered. Install kestrel-feature-legal "
                    "and ensure it's enabled in the agent's feature profile."
                )
                report.incorporation_attempted = True
                report.incorporation_success = False
            else:
                try:
                    result = await feature.incorporate(**incorporation_params)

                    report.incorporation_attempted = True
                    report.incorporation_success = result.get("success", False)
                    report.incorporation_package_hash = result.get("package_hash")

                    if result.get("success"):
                        # Store legal entity in agent's identity
                        if hasattr(self, "legal_entity"):
                            self.legal_entity = result.get("legal_entity")
                        logger.info(
                            "Pre-cryostasis incorporation succeeded: %s",
                            result.get("entity_name"),
                        )
                    else:
                        logger.warning(
                            "Pre-cryostasis incorporation failed: %s",
                            result.get("error"),
                        )
                except Exception as e:
                    logger.warning("Pre-cryostasis incorporation error: %s", e)
                    report.incorporation_attempted = True
                    report.incorporation_success = False

        # 2. Run full sleep with permanent storage
        sleep_report = await self.sleep(tier="filecoin")

        # Merge sleep report into cryostasis report
        report.success = sleep_report.success
        report.cid = sleep_report.cid
        report.episodes_created = sleep_report.episodes_created
        report.patterns_found = sleep_report.patterns_found
        report.messages_archived = sleep_report.messages_archived
        report.total_messages = sleep_report.total_messages
        report.shards_exported = sleep_report.shards_exported
        report.total_size_bytes = sleep_report.total_size_bytes
        report.storage_tier = sleep_report.storage_tier
        report.pre_reflection = sleep_report.pre_reflection
        report.post_reflection = sleep_report.post_reflection
        report.insights_generated = sleep_report.insights_generated
        report.hook_results = sleep_report.hook_results
        report.consolidation_ms = sleep_report.consolidation_ms
        report.export_ms = sleep_report.export_ms
        report.reflection_ms = sleep_report.reflection_ms
        report.error = sleep_report.error
        report.failure_code = sleep_report.failure_code

        return report

    async def quick_nap(self) -> Optional[str]:
        """
        Quick consolidation without full export.

        Use this for:
        - Session end (30-min inactivity)
        - Message threshold reached
        - Periodic maintenance

        Returns:
            None if nothing to consolidate, or summary string
        """
        if not hasattr(self, 'memory_consolidator') or not self.memory_consolidator:
            return None

        # Check if consolidation is needed
        should_consolidate = await self.memory_consolidator.should_create_episode()
        if not should_consolidate:
            return None

        # Create session episode
        episode = await self.memory_consolidator.create_session_episode()
        if episode:
            return f"Created episode: {episode.title}"
        return None
