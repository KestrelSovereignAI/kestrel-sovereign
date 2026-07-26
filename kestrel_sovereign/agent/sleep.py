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

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import heapq
from typing import Optional, Dict, Any, Callable, List, Tuple

logger = logging.getLogger(__name__)


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
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        if not self.success:
            return f"Sleep failed: {self.error}"

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

        report = SleepReport(success=False)
        reflection_start = time.time()
        consolidation_succeeded = False
        export_succeeded = False

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
                    report.error = unavailability_reason
            except Exception:
                logger.error("Consolidation failed")
                report.error = "consolidation_failed"
                # Continue to export anyway - partial sleep is better than none
            report.consolidation_ms = int((time.time() - start) * 1000)

            # 1.5 Post-consolidation hooks use the new episodes. Annotated hooks
            # are phase- and dependency-ordered; legacy hooks retain their
            # registration-order, continue-on-error behavior.
            if not skip_reflection and self.sleep_hooks:
                unavailability_reason = self._post_consolidation_unavailability_reason(
                    consolidation_result
                )
                if unavailability_reason is not None:
                    self._record_unavailable_post_consolidation_hooks(
                        report,
                        reason=unavailability_reason,
                    )
                else:
                    await self._run_post_consolidation_hooks(
                        consolidation_result, report
                    )

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
            report.export_ms = int((time.time() - start) * 1000)

        # Success is an actual completed consolidation or export, never merely
        # a requested operation with zero counters.  In particular,
        # MemorySystem.consolidate() reports some failures as ``{"error": ...}``
        # and the scheduler normally skips export.
        report.success = consolidation_succeeded or export_succeeded

        # Invoke callback if set (allows platform to update latest_cid)
        if report.success and report.cid and self.on_sleep_complete:
            try:
                await self.on_sleep_complete(report.cid, report)
            except Exception as e:
                logger.warning(f"on_sleep_complete callback failed: {e}")
                # Don't fail the sleep for callback errors

        return report

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
    ) -> None:
        """Report hooks skipped because consolidation produced no safe input.

        Previously an unsuccessful consolidation left no local result for the
        hook call, so every post hook was effectively skipped.  Preserve that
        safety boundary while making it observable in the structured report.
        """
        for registration_index, hook in enumerate(self.sleep_hooks or []):
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
        self, consolidation_result: Dict[str, Any], report: SleepReport
    ) -> None:
        """Run dependency-aware hooks while preserving legacy-group order."""
        import time

        registered_hooks = list(self.sleep_hooks or [])
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
