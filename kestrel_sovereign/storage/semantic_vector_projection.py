"""Derived, tenant-bound vector projection for canonical semantic assertions.

This module deliberately does *not* own semantic truth.  Every candidate is
bound to the exact current canonical revision which produced it, and recall
callers must hydrate that revision from :class:`AsyncAssertionStore` before it
can be shown to an agent.  Vectors are therefore disposable acceleration data,
not an alternate assertion or a generic RAG index.
"""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
import hashlib
import json
import math
import re
from collections.abc import Awaitable, Callable, Sequence

from kestrel_sovereign.knowledge import Assertion, AssertionStatus

from .async_assertion_store import AssertionCheckpoint, AsyncAssertionStore


_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_RENDERER_VERSION = "semantic-recall-claim-v1"
_MAX_VECTOR_DIMENSION = 8192
_MAX_VECTOR_BYTES = 262_144
_MAX_RECALL_SCAN = 2_000


class SemanticVectorProjectionError(RuntimeError):
    """A projection cannot safely process or serve the requested operation."""


@dataclass(frozen=True, slots=True)
class SemanticVectorProfile:
    """Exact capability pin for one derived vector coordinate space."""

    profile_id: str
    capability_digest: str
    provider: str
    model: str
    dimension: int
    visibility_ceiling: str = "private"
    privacy_ceiling: str = "normal"
    renderer_version: str = _RENDERER_VERSION
    max_claim_characters: int = 1_200
    embed_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not _PROFILE_RE.fullmatch(self.profile_id):
            raise ValueError("semantic vector profile_id is invalid")
        if not isinstance(self.capability_digest, str) or not _DIGEST_RE.fullmatch(self.capability_digest):
            raise ValueError("semantic vector capability_digest must be lowercase sha256 hex")
        if not isinstance(self.provider, str) or not _PROFILE_RE.fullmatch(self.provider):
            raise ValueError("semantic vector provider is invalid")
        if not isinstance(self.model, str) or not _PROFILE_RE.fullmatch(self.model):
            raise ValueError("semantic vector model is invalid")
        if type(self.dimension) is not int or not 1 <= self.dimension <= _MAX_VECTOR_DIMENSION:
            raise ValueError("semantic vector dimension is outside the supported bound")
        if self.visibility_ceiling not in {"private", "tenant", "delegated", "public"}:
            raise ValueError("semantic vector visibility_ceiling is invalid")
        if not isinstance(self.privacy_ceiling, str) or not _PROFILE_RE.fullmatch(self.privacy_ceiling):
            raise ValueError("semantic vector privacy_ceiling is invalid")
        if self.renderer_version != _RENDERER_VERSION:
            raise ValueError("semantic vector renderer_version is not approved")
        if type(self.max_claim_characters) is not int or not 1 <= self.max_claim_characters <= 8_192:
            raise ValueError("semantic vector max_claim_characters is outside the supported bound")
        if (
            not isinstance(self.embed_timeout_seconds, (int, float))
            or isinstance(self.embed_timeout_seconds, bool)
            or not 0 < float(self.embed_timeout_seconds) <= 300
        ):
            raise ValueError("semantic vector embed_timeout_seconds is outside the supported bound")


@dataclass(frozen=True, slots=True)
class SemanticVectorCandidate:
    """Content-free vector hit; canonical hydration remains mandatory."""

    assertion_id: str
    revision_id: str
    score: float
    generation: int


@dataclass(frozen=True, slots=True)
class SemanticVectorCheckpoint:
    """Durable projection cursor and its canonical source fence."""

    generation: int
    event_id: str | None


@dataclass(frozen=True, slots=True)
class SemanticVectorErasureObservation:
    """Content-free erasure observation for one tenant-bound projection."""

    generation: int
    candidate_count: int
    checkpoint_generation: int


Embedder = Callable[[str], Awaitable[Sequence[float] | None]]


def _revision_digest(assertion: Assertion) -> str:
    payload = json.dumps(
        assertion.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _approved_claim_text(assertion: Assertion, max_characters: int) -> str:
    """Use the prompt-reviewed, bounded renderer rather than raw RDF terms."""
    from kestrel_sovereign.agent.semantic_recall import _claim_text

    return _claim_text(assertion, max_characters)


def _normalise_vector(value: Sequence[float], *, dimension: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise SemanticVectorProjectionError("semantic vector embedder returned no vector")
    vector = tuple(float(item) for item in value)
    if len(vector) != dimension:
        raise SemanticVectorProjectionError("semantic vector dimension does not match profile")
    if any(not math.isfinite(item) for item in vector):
        raise SemanticVectorProjectionError("semantic vector contains non-finite values")
    magnitude = math.sqrt(sum(item * item for item in vector))
    if magnitude == 0:
        raise SemanticVectorProjectionError("semantic vector must have non-zero magnitude")
    normalised = tuple(item / magnitude for item in vector)
    if len(json.dumps(normalised, separators=(",", ":")).encode("utf-8")) > _MAX_VECTOR_BYTES:
        raise SemanticVectorProjectionError("semantic vector exceeds serialized-size bound")
    return normalised


class SemanticAssertionVectorProjection:
    """Durable vector owner consuming the canonical assertion outbox.

    The projection only advances its durable cursor after the matching vector
    mutation commits.  A retry therefore replays safely.  A physical-erasure
    outbox event contains no target IDs; it intentionally causes a full
    tenant/profile wipe, preventing erased vector resurrection after restart.
    """

    def __init__(self, store: AsyncAssertionStore, profile: SemanticVectorProfile, embedder: Embedder) -> None:
        if not isinstance(store, AsyncAssertionStore):
            raise TypeError("semantic vector projection requires AsyncAssertionStore")
        if not isinstance(profile, SemanticVectorProfile):
            raise TypeError("semantic vector projection requires SemanticVectorProfile")
        if not callable(embedder):
            raise TypeError("semantic vector projection embedder must be callable")
        self._store = store
        self._profile = profile
        self._embedder = embedder

    async def _embed(self, assertion: Assertion) -> tuple[float, ...]:
        try:
            embedded = await asyncio.wait_for(
                self._embedder(
                    _approved_claim_text(assertion, self._profile.max_claim_characters)
                ),
                timeout=float(self._profile.embed_timeout_seconds),
            )
        except TimeoutError as error:
            raise SemanticVectorProjectionError("semantic vector embedder timed out") from error
        if embedded is None:
            raise SemanticVectorProjectionError("semantic vector embedder unavailable")
        return _normalise_vector(embedded, dimension=self._profile.dimension)

    def _entry_values(
        self, assertion: Assertion, generation: int, vector: tuple[float, ...],
    ) -> tuple[object, ...]:
        return (
            self._store.tenant_id,
            self._profile.profile_id,
            self._profile.capability_digest,
            self._profile.provider,
            self._profile.model,
            self._profile.dimension,
            self._profile.renderer_version,
            self._profile.visibility_ceiling,
            self._profile.privacy_ceiling,
            assertion.visibility.value,
            assertion.privacy_classification,
            assertion.assertion_id,
            assertion.revision_id,
            _revision_digest(assertion),
            generation,
            json.dumps(vector, separators=(",", ":")),
        )

    async def _canonical_terminal(self) -> SemanticVectorCheckpoint:
        checkpoint = await self._store.event_checkpoint()
        return SemanticVectorCheckpoint(checkpoint.generation, checkpoint.latest_event_id)

    def _assert_privacy_scope(self, assertion: Assertion) -> None:
        order = {"private": 0, "tenant": 1, "delegated": 2, "public": 3}
        if order[assertion.visibility.value] > order[self._profile.visibility_ceiling]:
            raise SemanticVectorProjectionError("semantic vector visibility exceeds profile ceiling")
        if assertion.privacy_classification != self._profile.privacy_ceiling:
            raise SemanticVectorProjectionError("semantic vector privacy classification exceeds profile scope")

    async def checkpoint(self) -> SemanticVectorCheckpoint:
        row = await self._store._database.fetchone(  # noqa: SLF001 - sibling projection persistence
            "SELECT checkpoint_generation, checkpoint_event_id "
            "FROM semantic_assertion_vector_projection_state "
            "WHERE tenant_id = ? AND profile_id = ? AND capability_digest = ?",
            (self._store.tenant_id, self._profile.profile_id, self._profile.capability_digest),
        )
        return SemanticVectorCheckpoint(0, None) if row is None else SemanticVectorCheckpoint(int(row[0]), row[1])

    async def sync(self, *, limit: int = 100) -> SemanticVectorCheckpoint:
        """Drain a bounded outbox page with lineage and generation fencing."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("semantic vector sync limit must be an integer in [1, 1000]")
        cursor = await self.checkpoint()
        changes = await self._store.changes_after(
            AssertionCheckpoint(self._store.tenant_id, cursor.generation, cursor.event_id), limit=limit,
        )
        for change in changes:
            await self._apply_change(
                change.event_id, change.assertion_id, change.revision_id, change.generation,
                change.eligible, expected=cursor,
            )
            cursor = SemanticVectorCheckpoint(change.generation, change.event_id)
        return cursor

    async def _apply_change(
        self, event_id: str, assertion_id: str | None, revision_id: str | None, generation: int,
        eligible: bool, *, expected: SemanticVectorCheckpoint,
    ) -> None:
        # An opaque erasure signal has no target identities by design.  Wiping
        # the tenant/profile projection is the only safe replayable response.
        if assertion_id is None or revision_id is None:
            await self._wipe_for_erasure(event_id, generation, expected=expected)
            return

        assertion = await self._store.get_assertion(assertion_id)
        vector: tuple[float, ...] | None = None
        if eligible and assertion is not None and assertion.revision_id == revision_id and assertion.status is AssertionStatus.ACTIVE:
            self._assert_privacy_scope(assertion)
            vector = await self._embed(assertion)

        # Recheck exactly under the canonical lifecycle lock after embedding.
        # A supersession/retraction/erase during provider I/O cannot publish a
        # stale vector; it leaves this event unacknowledged for retry.
        async with self._store._mutation():  # noqa: SLF001 - canonical fence is the projection contract
            current = await self._store._current(assertion_id)  # noqa: SLF001
            current_eligible = (
                current is not None
                and current.revision_id == revision_id
                and current.status is AssertionStatus.ACTIVE
                and await self._store._is_current_active_eligible_revision(revision_id)  # noqa: SLF001
            )
            await self._store._database.execute(  # noqa: SLF001
                "DELETE FROM semantic_assertion_vector_projection_entries "
                "WHERE tenant_id = ? AND profile_id = ? AND capability_digest = ? AND assertion_id = ?",
                (self._store.tenant_id, self._profile.profile_id, self._profile.capability_digest, assertion_id),
            )
            if vector is not None and current_eligible:
                await self._store._database.execute(  # noqa: SLF001
                    "INSERT INTO semantic_assertion_vector_projection_entries "
                    "(tenant_id, profile_id, capability_digest, embedding_provider, embedding_model, "
                    "embedding_dimension, renderer_version, visibility_ceiling, privacy_ceiling, visibility, privacy_classification, "
                    "assertion_id, revision_id, revision_digest, source_generation, vector_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    self._entry_values(current, generation, vector),
                )
            await self._advance(event_id, generation, expected=expected)

    async def _wipe_for_erasure(
        self, event_id: str, generation: int, *, expected: SemanticVectorCheckpoint,
    ) -> None:
        # Erasure events deliberately retain no deleted target identifiers.
        # A wipe is therefore mandatory, but a wipe alone would strand
        # unrelated surviving assertions.  Snapshot those survivors from the
        # canonical owner, embed outside its lock, and publish the complete
        # replacement projection atomically.  Until that succeeds the cursor
        # stays behind and recall fails closed rather than serving a partial
        # post-erasure index.
        survivors = await self._eligible_current_assertions()
        projected: list[tuple[Assertion, tuple[float, ...]]] = []
        for assertion in survivors:
            self._assert_privacy_scope(assertion)
            projected.append((assertion, await self._embed(assertion)))
        async with self._store._mutation():  # noqa: SLF001 - same lifecycle serialization as physical erase
            await self._store._database.execute(  # noqa: SLF001
                "DELETE FROM semantic_assertion_vector_projection_entries "
                "WHERE tenant_id = ? AND profile_id = ? AND capability_digest = ?",
                (self._store.tenant_id, self._profile.profile_id, self._profile.capability_digest),
            )
            for assertion, vector in projected:
                current = await self._store._current(assertion.assertion_id)  # noqa: SLF001
                if (
                    current is None
                    or current.revision_id != assertion.revision_id
                    or current.status is not AssertionStatus.ACTIVE
                    or not await self._store._is_current_active_eligible_revision(assertion.revision_id)  # noqa: SLF001
                ):
                    # A concurrent lifecycle transition has an outbox event
                    # after this erasure event.  Do not revive its stale
                    # vector; the later event will project its new state.
                    continue
                await self._store._database.execute(  # noqa: SLF001
                    "INSERT INTO semantic_assertion_vector_projection_entries "
                    "(tenant_id, profile_id, capability_digest, embedding_provider, embedding_model, "
                    "embedding_dimension, renderer_version, visibility_ceiling, privacy_ceiling, visibility, privacy_classification, "
                    "assertion_id, revision_id, revision_digest, source_generation, vector_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    self._entry_values(current, generation, vector),
                )
            await self._advance(event_id, generation, expected=expected)

    async def _eligible_current_assertions(self) -> tuple[Assertion, ...]:
        """Read only canonical current rows eligible for a full erase rebuild."""
        rows = await self._store._database.fetchall(  # noqa: SLF001 - canonical projection read
            "SELECT r.assertion_mapping FROM semantic_assertions a "
            "JOIN semantic_assertion_revisions r ON r.tenant_id = a.tenant_id "
            " AND r.revision_id = a.current_revision_id "
            "JOIN semantic_projection_eligibility e ON e.tenant_id = r.tenant_id "
            " AND e.revision_id = r.revision_id "
            "WHERE a.tenant_id = ? AND r.status = ? AND r.eligible = 1 AND e.eligible = 1 "
            "ORDER BY r.accepted_order ASC, r.revision_id ASC",
            (self._store.tenant_id, AssertionStatus.ACTIVE.value),
        )
        return tuple(Assertion.from_mapping(json.loads(row[0])) for row in rows)

    async def _advance(
        self, event_id: str, generation: int, *, expected: SemanticVectorCheckpoint,
    ) -> None:
        """CAS the exact event cursor; never let a stale worker regress it."""
        row = await self._store._database.fetchone(  # noqa: SLF001
            "SELECT checkpoint_generation, checkpoint_event_id "
            "FROM semantic_assertion_vector_projection_state "
            "WHERE tenant_id = ? AND profile_id = ? AND capability_digest = ?",
            (self._store.tenant_id, self._profile.profile_id, self._profile.capability_digest),
        )
        current = SemanticVectorCheckpoint(0, None) if row is None else SemanticVectorCheckpoint(int(row[0]), row[1])
        if current != expected:
            raise SemanticVectorProjectionError("semantic_vector_projection_concurrent_progress")
        if row is None:
            await self._store._database.execute(  # noqa: SLF001
                "INSERT INTO semantic_assertion_vector_projection_state "
                "(tenant_id, profile_id, capability_digest, checkpoint_generation, checkpoint_event_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (self._store.tenant_id, self._profile.profile_id, self._profile.capability_digest, generation, event_id),
            )
            return
        changed = await self._store._database.execute(  # noqa: SLF001
            "UPDATE semantic_assertion_vector_projection_state "
            "SET checkpoint_generation = ?, checkpoint_event_id = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE tenant_id = ? AND profile_id = ? AND capability_digest = ? "
            "AND checkpoint_generation = ? AND "
            "(checkpoint_event_id = ? OR (checkpoint_event_id IS NULL AND ? IS NULL))",
            (
                generation, event_id, self._store.tenant_id, self._profile.profile_id,
                self._profile.capability_digest, expected.generation, expected.event_id,
                expected.event_id,
            ),
        )
        if changed != 1:
            raise SemanticVectorProjectionError("semantic_vector_projection_concurrent_progress")

    async def recall(
        self, query_vector: Sequence[float], *, limit: int = 8,
        candidate_scan_limit: int = _MAX_RECALL_SCAN,
    ) -> tuple[SemanticVectorCandidate, ...]:
        """Return only lineage tokens, after a current projection fence.

        This is deliberately not a content-returning retrieval API.  Callers
        must use the canonical semantic-recall hydration fence with these IDs.
        """
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("semantic vector recall limit must be an integer in [1, 100]")
        if type(candidate_scan_limit) is not int or not 1 <= candidate_scan_limit <= _MAX_RECALL_SCAN:
            raise ValueError("semantic vector candidate_scan_limit is outside the supported bound")
        query = _normalise_vector(query_vector, dimension=self._profile.dimension)
        state = await self.checkpoint()
        canonical = await self._canonical_terminal()
        if state != canonical:
            raise SemanticVectorProjectionError("semantic_vector_projection_checkpoint_stale")
        rows = await self._store._database.fetchall(  # noqa: SLF001
            "SELECT v.assertion_id, v.revision_id, v.source_generation, v.vector_json, "
            "v.embedding_provider, v.embedding_model, v.embedding_dimension, v.renderer_version, "
            "v.visibility_ceiling, v.privacy_ceiling, v.visibility, v.privacy_classification, v.revision_digest, "
            "r.assertion_mapping FROM semantic_assertion_vector_projection_entries v "
            "JOIN semantic_assertions a ON a.tenant_id = v.tenant_id "
            " AND a.assertion_id = v.assertion_id AND a.current_revision_id = v.revision_id "
            "JOIN semantic_assertion_revisions r ON r.tenant_id = v.tenant_id "
            " AND r.revision_id = v.revision_id "
            "JOIN semantic_projection_eligibility e ON e.tenant_id = v.tenant_id "
            " AND e.revision_id = v.revision_id "
            "WHERE v.tenant_id = ? AND v.profile_id = ? AND v.capability_digest = ? "
            "AND r.status = ? AND r.eligible = 1 AND e.eligible = 1 "
            "ORDER BY v.assertion_id, v.revision_id LIMIT ?",
            (
                self._store.tenant_id, self._profile.profile_id, self._profile.capability_digest,
                AssertionStatus.ACTIVE.value, candidate_scan_limit + 1,
            ),
        )
        if len(rows) > candidate_scan_limit:
            raise SemanticVectorProjectionError("semantic_vector_projection_candidate_window_exceeded")
        candidates: list[SemanticVectorCandidate] = []
        for (
            assertion_id, revision_id, generation, raw_vector, provider, model, dimension,
            renderer_version, visibility_ceiling, privacy_ceiling, visibility, privacy_classification,
            revision_digest, assertion_mapping,
        ) in rows:
            assertion = Assertion.from_mapping(json.loads(assertion_mapping))
            if (
                provider != self._profile.provider
                or model != self._profile.model
                or int(dimension) != self._profile.dimension
                or renderer_version != self._profile.renderer_version
                or visibility_ceiling != self._profile.visibility_ceiling
                or privacy_ceiling != self._profile.privacy_ceiling
                or visibility != assertion.visibility.value
                or privacy_classification != assertion.privacy_classification
                or revision_digest != _revision_digest(assertion)
            ):
                raise SemanticVectorProjectionError("semantic_vector_projection_profile_drift")
            if len(str(raw_vector).encode("utf-8")) > _MAX_VECTOR_BYTES:
                raise SemanticVectorProjectionError("semantic vector exceeds serialized-size bound")
            vector = tuple(float(value) for value in json.loads(raw_vector))
            if len(vector) != len(query):
                raise SemanticVectorProjectionError("semantic_vector_projection_dimension_drift")
            score = sum(left * right for left, right in zip(query, vector))
            candidates.append(SemanticVectorCandidate(str(assertion_id), str(revision_id), score, int(generation)))
        candidates.sort(key=lambda item: (-item.score, item.assertion_id, item.revision_id))
        # A lifecycle mutation may have committed while vectors were read.
        if await self._canonical_terminal() != canonical:
            raise SemanticVectorProjectionError("semantic_vector_projection_checkpoint_changed")
        return tuple(candidates[:limit])

    async def recall_hydrated(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 8,
        inference_profile=None,
        inference_limits=None,
        maintenance_limits=None,
    ):
        """Bridge vector ranking into the existing canonical recall boundary.

        The vector layer supplies only IDs and revisions.  This method passes
        those IDs through the authoritative semantic-recall hydration fence,
        which rejects a lifecycle/maintenance checkpoint change and returns
        current provenance rather than vector-owned content.
        """
        hits = await self.recall(query_vector, limit=limit)
        checkpoint = await self._canonical_terminal()
        hydrated = await self._store.hydrate_recall_candidates(
            [hit.assertion_id for hit in hits],
            expected_checkpoint_generation=checkpoint.generation,
            inference_profile=inference_profile,
            inference_limits=inference_limits,
            maintenance_limits=maintenance_limits,
        )
        by_id = {item.assertion.assertion_id: item for item in hydrated}
        # Require exact revision equality as defence in depth: a vector hit is
        # never allowed to select a later revision merely because it shares an
        # assertion identity.
        return tuple(
            by_id[hit.assertion_id]
            for hit in hits
            if hit.assertion_id in by_id
            and by_id[hit.assertion_id].assertion.revision_id == hit.revision_id
        )

    async def erasure_observation(self) -> SemanticVectorErasureObservation:
        """Return content-free candidate cardinality at one canonical fence."""
        checkpoint = await self._store.checkpoint()
        state = await self.checkpoint()
        terminal = await self._canonical_terminal()
        if state != terminal:
            raise SemanticVectorProjectionError("semantic_vector_projection_checkpoint_stale")
        count = await self._store._database.fetchval(  # noqa: SLF001
            "SELECT COUNT(*) FROM semantic_assertion_vector_projection_entries "
            "WHERE tenant_id = ? AND profile_id = ? AND capability_digest = ?",
            (self._store.tenant_id, self._profile.profile_id, self._profile.capability_digest),
        )
        if await self._canonical_terminal() != terminal:
            raise SemanticVectorProjectionError("semantic_vector_projection_checkpoint_changed")
        return SemanticVectorErasureObservation(checkpoint.generation, int(count or 0), state.generation)


__all__ = [
    "SemanticAssertionVectorProjection", "SemanticVectorCandidate", "SemanticVectorCheckpoint",
    "SemanticVectorErasureObservation", "SemanticVectorProfile", "SemanticVectorProjectionError",
]
