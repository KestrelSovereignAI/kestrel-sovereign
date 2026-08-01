"""
Memory consolidation and forgetting curve.

Implements human-like memory consolidation:
- Creates narrative episodes from related messages
- Detects temporal patterns
- Implements forgetting curve for unimportant memories
- Runs as nightly maintenance (or on-demand)

This is inspired by how human memory works during sleep:
memories are consolidated, patterns are detected, and
unimportant details fade.
"""
import logging
import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

from .memory_models import MemoryEpisode, TemporalPattern
from .async_database import AsyncDatabase
from .memory_retriever import calculate_decay
from .async_conversation_store import (
    _strip_search_wrappers,
    _token_match_score,
    _tokenize_for_search,
)
from .session_grouping import timestamp_predicate, timestamp_query_param
from kestrel_sovereign.security.input_guardrails import extract_raw_user_content

# ``SalvageState`` lives in ``kestrel_sovereign.agent.salvage``;
# importing it here would create a circular import via
# ``agent/__init__.py`` → ``context_builder`` → ``features.bootstrap``
# → ``features.base``. Inline the canonical state strings instead.
_SALVAGE_STATE_POINTER_ONLY = "pointer-only"
_SALVAGE_STATE_PENDING_SUMMARY = "pending-summary"
_SALVAGE_STATE_DURABLE_FOLDED = "durable-folded"

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Memory maintenance and episode creation.

    Responsibilities:
    1. Create narrative episodes from related messages
    2. Detect temporal patterns
    3. Archive fully decayed memories
    4. Update decay-related metadata

    Episode creation triggers:
    - Nightly consolidation (run_consolidation)
    - Session end (30-min gap detected)
    - Message threshold (> SESSION_EPISODE_THRESHOLD messages)
    - Manual trigger via !consolidate command
    """

    # Thresholds
    DECAY_ARCHIVE_THRESHOLD = 0.1  # Archive if strength < 10%
    MIN_EPISODE_MESSAGES = 3       # Minimum messages for an episode
    MAX_EPISODE_HOURS = 24         # Maximum episode time span
    SESSION_EPISODE_THRESHOLD = 20 # Create episode after N messages in session

    @property
    def SESSION_GAP_MINUTES(self) -> int:  # noqa: N802 (kept uppercase for back-compat)
        """Session boundary constant — see kestrel_sdk.config.constants."""
        from kestrel_sdk.config.constants import SESSION_GAP_MINUTES as _GAP
        return _GAP

    def __init__(self, db: AsyncDatabase, agent_id: str, graph_store=None,
                 llm_service=None, persist_policy=None, conversation_store=None):
        """
        Initialize consolidator.

        Args:
            db: AsyncDatabase instance
            agent_id: Agent ID to consolidate memories for
            graph_store: Optional AsyncGraphStore for writing episodes to the KG.
                When the memory system wires the privacy-governing graph proxy
                here, volatile-mode episode KG writes fail closed at the storage
                boundary (#2672).
            llm_service: Optional agent-scoped LLM service used to resolve the
                provider embedding service for episode embeddings + semantic
                recall (#1674 P2). When absent (or no embedding provider is
                configured), episode recall degrades to keyword search and no
                embeddings are written — both paths stay functional.
            conversation_store: The AsyncConversationStore that owns message
                decryption. This class reads ``conversation_history`` with its
                own SQL for clustering, so without it every encrypted row's
                at-rest envelope was treated as text and tokenized straight
                into episode titles, summaries and affect (#2850). ``None``
                (raw storage, tests) means rows are assumed plaintext, and any
                row still carrying an envelope is skipped rather than
                summarized.
            persist_policy: Optional privacy authority exposing
                ``allows_persistent_writes() -> bool``. Consulted before the
                direct ``memory_episodes`` write so manual / scheduled
                consolidation cannot persist a user-derived episode summary
                while the agent is in a volatile privacy mode (#2672). ``None``
                (raw storage, tests) imposes no gate — durable persistence
                proceeds, preserving prior behaviour.
        """
        self._db = db
        self.agent_id = agent_id
        self._graph_store = graph_store
        self._llm_service = llm_service
        self._persist_policy = persist_policy
        self._conversation_store = conversation_store
        # Lazily-built SQLAlchemy session factory for the shared vector
        # backend (mirrors SavedItemsStore). None when unavailable.
        self._sqla_factory = None
        self._sqla_factory_unavailable = False

    def _get_embedding_service(self):
        """Resolve the active provider's embedding service (reused infra).

        Returns None when no LLM service / embedding provider is available —
        callers then fall back to keyword recall and skip embedding writes.
        """
        if self._llm_service is None:
            return None
        try:
            from kestrel_sovereign.llm.embedding_service import (
                get_provider_embedding_service,
            )
            return get_provider_embedding_service(self._llm_service)
        except Exception as e:  # noqa: BLE001 - embeddings are best-effort
            logger.debug("episode embedding service unavailable: %s", e)
            return None

    def _get_vector_session_factory(self):
        """Lazy-build a SQLAlchemy session factory for the shared vector
        backend, pointed at the same DB as ``self._db`` (mirrors
        ``SavedItemsStore._get_vector_session_factory``). Returns None when
        construction fails (in-memory SQLite, bare pool, etc.) — callers then
        fall back to keyword recall."""
        if self._sqla_factory is not None:
            return self._sqla_factory
        if self._sqla_factory_unavailable:
            return None
        try:
            from kestrel_sovereign.storage.sqla import make_session_factory
            self._sqla_factory = make_session_factory(self._db)
            return self._sqla_factory
        except Exception as e:  # noqa: BLE001
            logger.debug("episode vector session factory unavailable: %s", e)
            self._sqla_factory_unavailable = True
            return None

    async def run_consolidation(self) -> Dict[str, Any]:
        """
        Run full consolidation cycle.

        Steps:
        1. Create narrative episodes from related messages
        2. Detect temporal patterns
        3. Archive fully decayed memories
        4. Update statistics

        Returns:
            Report dict with counts of each operation
        """
        report = {
            "episodes_created": 0,
            "patterns_found": 0,
            "messages_archived": 0,
            "total_messages_processed": 0,
            "clusters_skipped": 0,
            "skip_reasons": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Privacy boundary (#2672): gate the ENTIRE durable consolidation path,
        # not just the episode row. ``_create_episodes`` is already gated at
        # ``_save_episode``, but ``_detect_patterns`` writes durable
        # ``temporal_patterns`` (e.g. "User is most active late at night") through
        # the raw DB, ``_archive_decayed`` mutates ``conversation_history``, and
        # ``_backfill_episode_embeddings`` writes embeddings — all user-derived
        # and all outside the graph proxy. In a volatile privacy mode none of them
        # may persist, so fail closed here and return an explicit privacy-blocked
        # report so the manual ``memory_consolidate`` tool reports "skipped", not
        # "complete".
        if not self._durable_writes_allowed():
            report["skipped"] = True
            report["skipped_reason"] = "privacy_mode_forbids_persistence"
            report["privacy_blocked"] = True
            logger.info(
                "Consolidation skipped: current privacy mode forbids durable "
                "memory writes; no episodes, patterns, archival, or embeddings "
                "persisted (#2672)"
            )
            return report

        try:
            # 1. Create episodes from high-importance clusters
            episodes, skipped = await self._create_episodes()
            report["episodes_created"] = len(episodes)
            report["clusters_skipped"] = len(skipped)
            if skipped:
                report["skip_reasons"] = [
                    {"date": d, "messages": n, "reason": r}
                    for d, n, r in skipped
                ]

            # 2. Detect temporal patterns
            patterns = await self._detect_patterns()
            report["patterns_found"] = len(patterns)

            # 3. Archive decayed messages
            archived = await self._archive_decayed()
            report["messages_archived"] = archived

            # 3b. Backfill embeddings for a bounded batch of pre-P2 episodes
            # (#1674 P2) so older episodes become semantically recallable —
            # and thus eligible for access-heat protection — over successive
            # nights. Best-effort; no-op when no embedding provider is wired.
            report["episodes_embedded"] = await self._backfill_episode_embeddings()

            # Get total message count
            count = await self._db.fetchval(
                "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?",
                (self.agent_id,)
            )
            report["total_messages_processed"] = count or 0

            logger.info(
                f"Consolidation complete: {report['episodes_created']} episodes, "
                f"{report['patterns_found']} patterns, "
                f"{report['messages_archived']} archived"
            )

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            report["error"] = str(e)

        return report

    async def _create_episodes(self) -> Tuple[List[MemoryEpisode], List[Tuple[str, int, str]]]:
        """
        Group related messages into narrative episodes.

        Strategy:
        - Look at messages from last 30 days
        - Group by day
        - Within each day, find high-emotion clusters
        - Create episode if cluster has enough messages

        Returns:
            Tuple of (created episodes, skipped clusters as (date, count, reason))
        """
        episodes: List[MemoryEpisode] = []
        report_skipped: List[Tuple[str, int, str]] = []

        # Get messages from last 30 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        created_at_predicate = timestamp_predicate(
            self._db.backend_type, "created_at", ">"
        )

        rows = await self._db.fetchall(
            f"""SELECT id, content, metadata, created_at, role
               FROM conversation_history
               WHERE agent_id = ? AND {created_at_predicate}
                 AND deleted_at IS NULL AND archived_at IS NULL
               ORDER BY created_at""",
            (self.agent_id, timestamp_query_param(self._db.backend_type, cutoff))
        )

        if not rows:
            return episodes, report_skipped

        # Probe once: which message IDs are already covered by ANY existing
        # episode (consolidator or session)? Used per-cluster to skip / pare
        # down messages so nightly runs don't duplicate prior work (#1489 P2).
        covered_message_ids = await self._covered_message_ids()

        # Prefer explicit conversation sessions. Legacy rows without a session
        # ID retain the date bucket as a bounded fallback.
        clusters: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            msg_id, content, metadata, created_at, role = row

            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            # Excluded conversation artifacts are provenance-tainted
            # derivatives.  They must never seed a fresh visible episode after
            # a fact lifecycle action, even if the original episode had been
            # excluded before this nightly pass.
            if metadata.get("excluded_from_context"):
                continue

            # #2850: rows come from raw SQL, so `content` is still the at-rest
            # envelope. Decode before anything reads it as text.
            content = self._row_plaintext(content, metadata)
            if content is None:
                continue

            # Parse date from created_at
            try:
                if isinstance(created_at, str):
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    dt = created_at
                date_key = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            if metadata.get("type") in {"new_session", "compaction", "compression"}:
                continue
            cluster_key = str(metadata.get("session_id") or f"date:{date_key}")
            clusters[cluster_key].append({
                "id": msg_id,
                "content": content,
                "metadata": metadata,
                "created_at": created_at,
                "role": role,
            })

        # For each session/date fallback, check if there's a significant cluster
        for cluster_key, day_messages in clusters.items():
            first_created = day_messages[0].get("created_at")
            try:
                date_key = datetime.fromisoformat(
                    str(first_created).replace("Z", "+00:00")
                ).strftime("%Y-%m-%d")
            except (TypeError, ValueError):
                date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            original_count = len(day_messages)
            if original_count < self.MIN_EPISODE_MESSAGES:
                report_skipped.append(
                    (date_key, original_count, "below_min_messages")
                )
                continue

            # Message-level idempotency (#1489 P2). Dedup BEFORE the
            # emotional / pending-salvage gates so they don't average over
            # messages already locked into a prior episode (codex round 4):
            # otherwise a day with many already-covered low-importance
            # messages plus a few new high-importance ones can fail the
            # emotional-threshold gate against the *old* messages and
            # permanently shadow the new span.
            messages = [
                m for m in day_messages
                if str(m["id"]) not in covered_message_ids
            ]

            if not messages:
                report_skipped.append(
                    (date_key, original_count, "already_consolidated")
                )
                continue

            if len(messages) < self.MIN_EPISODE_MESSAGES:
                report_skipped.append(
                    (date_key, len(messages), "below_min_after_dedup")
                )
                continue

            # Calculate average emotional intensity (post-dedup messages only)
            intensities = []
            importances = []
            enriched_count = 0
            for msg in messages:
                meta = msg.get("metadata", {})
                intensities.append(meta.get("emotional_intensity", 0.0))
                importances.append(meta.get("importance", 0.5))
                # A message is "enriched" if the tagger has run on it —
                # detect by the presence of emotional_categories or an
                # explicit importance value.
                if meta.get("emotional_categories") or "importance" in meta:
                    enriched_count += 1

            avg_intensity = sum(intensities) / len(intensities) if intensities else 0
            avg_importance = sum(importances) / len(importances) if importances else 0.5

            # Only apply the emotional-significance gate when messages
            # actually carry emotional metadata.  Messages with default
            # metadata (intensity=0.0, importance=0.5) are "unenriched" —
            # the tagger never ran or the metadata was lost.  Gating on
            # emotional significance for unenriched clusters silently
            # drops every conversation that wasn't explicitly tagged,
            # which is the root cause of #1489 (scheduled consolidation
            # produces zero episodes for agents with hundreds of messages).
            has_enrichment = enriched_count > 0
            if has_enrichment and avg_intensity < 0.3 and avg_importance < 0.6:
                report_skipped.append(
                    (date_key, len(messages), "below_emotional_threshold")
                )
                continue

            # C / #1311 pending-state idempotency (Emma 2026-05-21):
            # when every message in the cluster has a linked salvage
            # that is still ``pointer-only`` or ``pending-summary``,
            # the salvage summariser is about to fold the same span.
            # Fabricating an episode from the raw rows now would race
            # the summariser and create two parallel records of the
            # same span with no causal link. Defer this cluster — the
            # next consolidator pass picks it up after the salvage
            # settles into ``durable-folded`` or ``failed-fold``.
            if await self._all_messages_have_pending_salvage(messages):
                logger.debug(
                    "consolidator: deferring cluster %s — every message "
                    "has a pending salvage; episode will fire after the "
                    "salvage settles",
                    date_key,
                )
                report_skipped.append(
                    (date_key, len(messages), "pending_salvage")
                )
                continue

            # Create episode
            episode = await self._create_episode_from_messages(
                date_key, messages, avg_intensity, avg_importance
            )
            if episode:
                episodes.append(episode)
                await self._save_episode(episode)

        return episodes, report_skipped

    async def _covered_message_ids(self) -> set:
        """Return the set of message IDs already covered by any existing
        episode for this agent.

        Used by ``_create_episodes`` to dedup against prior consolidator AND
        session-episode runs. Daily-consolidator episodes have IDs of the
        form ``episode:<agent>:YYYY-MM-DD:<suffix>``, while session episodes
        use ``episode:<agent>:YYYY-MM-DD-HHMM:<suffix>``. Querying by
        ``agent_id`` alone (no LIKE on the date) catches both, and avoids
        N-per-day queries (#1489 P2).
        """
        rows = await self._db.fetchall(
            """SELECT key_message_ids FROM memory_episodes
               WHERE agent_id = ? AND COALESCE(excluded_from_context, 0) = 0""",
            (self.agent_id,),
        )
        covered: set = set()
        for row in rows or []:
            kmi = row[0] if isinstance(row, (tuple, list)) else row
            if isinstance(kmi, str):
                try:
                    parsed = json.loads(kmi)
                    if isinstance(parsed, list):
                        covered.update(str(x) for x in parsed)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(kmi, list):
                covered.update(str(x) for x in kmi)
        return covered

    async def _all_messages_have_pending_salvage(
        self, messages: List[Dict[str, Any]]
    ) -> bool:
        """C / #1311 helper — return True when every message in the
        cluster has a ``summarized_into`` link AND the linked marker
        is still in a *pre-folded* state (``pointer-only`` or
        ``pending-summary``).

        Codex round 1 #5 caught a regression in the earlier sync
        version of this helper: it returned True for any cluster
        where every row had ``summarized_into`` set — but legacy
        ``compact_session`` markers ALSO set that field and are
        already ``durable-folded``. The consolidator would have
        skipped clusters whose narrative the salvage summariser is
        not about to write, with no recovery path. We now load each
        marker and check its actual ``salvage_state``.

        ``durable-folded`` and ``failed-fold`` (and the legacy
        ``compact_session`` markers, which carry ``type ==
        "compaction"`` and no ``salvage_state``) are treated as
        settled — the consolidator may run its emotional-cluster
        logic for those spans, using the summary marker as input on
        Emma's "episode-as-input" preference (deferred to a follow-up
        as long as the consolidator at least doesn't skip wrongly).
        """
        if not messages:
            return False
        pending_states = {
            _SALVAGE_STATE_POINTER_ONLY,
            _SALVAGE_STATE_PENDING_SUMMARY,
        }
        seen_marker_ids: set = set()
        for msg in messages:
            meta = msg.get("metadata") or {}
            marker_id = meta.get("summarized_into")
            if not marker_id:
                return False
            try:
                marker_id = int(marker_id)
            except (TypeError, ValueError):
                return False
            seen_marker_ids.add(marker_id)
        for mid in seen_marker_ids:
            state = await self._load_marker_state(mid)
            if state not in pending_states:
                return False
        return True

    async def _load_marker_state(self, marker_id: int) -> Optional[str]:
        """Return the linked marker's ``salvage_state``, or None when
        the row is missing or is a legacy ``compaction`` marker that
        has no salvage_state field (treated as ``durable-folded`` for
        the pending-check above)."""
        try:
            row = await self._db.fetchone(
                "SELECT metadata FROM conversation_history WHERE id = ?",
                (marker_id,),
            )
        except Exception as e:
            logger.debug("consolidator: marker %s state lookup failed: %s", marker_id, e)
            return None
        if not row:
            return None
        raw = row[0] if row else None
        if not raw:
            return None
        try:
            meta = json.loads(raw)
        except (TypeError, ValueError):
            return None
        # Pre-salvage ``compact_session`` markers don't have
        # ``salvage_state``; they are durable-folded by construction.
        # (Pre-rename rows carried ``type == "compression"`` — the
        # startup migration rewrites them to "compaction" before this
        # reader runs; an unmigrated row degrades to None, which the
        # pending-check treats the same as durable-folded.)
        if meta.get("type") == "compaction":
            return _SALVAGE_STATE_DURABLE_FOLDED
        return meta.get("salvage_state")

    async def _create_episode_from_messages(
        self,
        date_key: str,
        messages: List[Dict],
        avg_intensity: float,
        avg_importance: float = 0.5,
    ) -> Optional[MemoryEpisode]:
        """Create a MemoryEpisode from a cluster of messages.

        ``avg_importance`` is the mean message-importance over the cluster,
        stamped onto the episode so the forgetting curve (#1674) can decay it
        at an importance-scaled half-life rather than by raw age.
        """
        if not messages:
            return None

        # Get emotional arc (sequence of valences)
        valences = []
        for msg in messages:
            meta = msg.get("metadata", {})
            valence = meta.get("emotional_valence", 0.0)
            valences.append(valence)

        emotional_arc = self._describe_emotional_arc(valences)

        # Get first and last timestamps
        timestamps = []
        for msg in messages:
            try:
                ts = msg.get("created_at", "")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                timestamps.append(ts)
            except (ValueError, TypeError):
                pass

        timespan_start = min(timestamps) if timestamps else None
        timespan_end = max(timestamps) if timestamps else None

        # Generate title based on content themes
        title = self._generate_episode_title(messages, avg_intensity)

        # Get key message IDs
        key_message_ids = [str(msg["id"]) for msg in messages]

        # Generate summary (simplified - could use LLM for better summaries)
        summary = self._generate_episode_summary(messages, emotional_arc)

        return MemoryEpisode(
            id=f"episode:{self.agent_id}:{date_key}:{uuid.uuid4().hex[:8]}",
            agent_id=self.agent_id,
            title=title,
            summary=summary,
            timespan_start=timespan_start,
            timespan_end=timespan_end,
            key_message_ids=key_message_ids,
            emotional_arc=emotional_arc,
            created_at=datetime.now(timezone.utc),
            importance=avg_importance,
        )

    def _describe_emotional_arc(self, valences: List[float]) -> str:
        """Describe the emotional trajectory of an episode."""
        if not valences:
            return "neutral"

        start = valences[0]
        end = valences[-1]
        avg = sum(valences) / len(valences)

        # Describe based on trajectory
        if start < -0.3 and end > 0.3:
            return "difficult start → positive resolution"
        elif start > 0.3 and end < -0.3:
            return "started well → ended difficult"
        elif avg > 0.3:
            return "generally positive"
        elif avg < -0.3:
            return "challenging throughout"
        elif abs(end - start) > 0.5:
            return "emotional journey"
        else:
            return "emotionally steady"

    def _generate_episode_title(
        self,
        messages: List[Dict],
        avg_intensity: float
    ) -> str:
        """Generate a title for the episode."""
        # Extract themes from messages
        themes = set()
        for msg in messages:
            meta = msg.get("metadata", {})
            categories = meta.get("emotional_categories", [])
            themes.update(categories)
            reasons = meta.get("importance_reasons", [])
            themes.update(reasons)

        # Convert themes to readable title
        if "life_event" in themes:
            return "A significant day"
        elif "personal_disclosure" in themes:
            return "Opening up"
        elif "joy" in themes and avg_intensity > 0.5:
            return "A joyful moment"
        elif "sadness" in themes and avg_intensity > 0.5:
            return "Working through sadness"
        elif "anxiety" in themes:
            return "Processing worries"
        elif avg_intensity > 0.6:
            return "An emotional conversation"
        else:
            topics = self._extract_episode_topics(messages, limit=3)
            return f"Discussion of {', '.join(topics)}" if topics else "A memorable exchange"

    def _generate_episode_summary(
        self,
        messages: List[Dict],
        emotional_arc: str
    ) -> str:
        """Generate a summary of the episode."""
        user_messages = [m for m in messages if m.get("role") == "user"]
        message_count = len(messages)
        user_count = len(user_messages)

        topics = self._extract_episode_topics(messages, limit=8)
        topic_sentence = (
            f" Topics: {', '.join(topics)}." if topics else ""
        )
        return (
            f"A conversation with {message_count} messages "
            f"({user_count} from user).{topic_sentence} "
            f"Emotional trajectory: {emotional_arc}."
        )

    @staticmethod
    def _extract_episode_topics(messages: List[Dict], limit: int = 8) -> List[str]:
        """Extract bounded topic terms without persisting rendered context.

        Message IDs remain the provenance trail; summaries contain only
        normalized terms from canonical content, avoiding transport wrappers
        and verbatim prompt-injection text.
        """
        stop_words = {
            "about", "after", "again", "also", "and", "are", "been", "but",
            "can", "could", "did", "for", "from", "have", "her", "him", "his",
            "how", "into", "just", "not", "our", "she", "that", "the", "their",
            "them", "then", "there", "they", "this", "was", "were", "what",
            "when", "where", "which", "with", "would", "you", "your", "context",
            "conversation", "conversations", "document", "documents", "end",
            "memories", "memory", "relevant", "retrieved",
        }
        counts: Counter[str] = Counter()
        first_seen: Dict[str, int] = {}
        position = 0
        for message in messages:
            content = str(message.get("content") or "")
            if message.get("role") == "user":
                content = extract_raw_user_content(content)
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{2,}", content.lower()):
                if token in stop_words:
                    continue
                counts[token] += 1
                first_seen.setdefault(token, position)
                position += 1
        ranked = sorted(counts, key=lambda term: (-counts[term], first_seen[term], term))
        return ranked[:limit]

    # ------------------------------------------------------------------
    # Row content decoding (#2850)
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_ciphertext(content: str) -> bool:
        """Whether ``content`` still carries the at-rest AEAD envelope."""
        try:
            from kestrel_sdk.security.aead import KSA_V2_PREFIX

            prefix = KSA_V2_PREFIX.decode()
        except Exception:  # noqa: BLE001 - never let a probe break consolidation
            prefix = "KSAv2:"
        return content.startswith(prefix)

    def _row_plaintext(self, content: Any, metadata: Optional[Dict]) -> Optional[str]:
        """Decrypt one raw ``conversation_history`` row to plaintext.

        Returns ``None`` when the row cannot be trusted as text, and the
        caller must then SKIP it. Failing closed is the point: an episode
        title, summary, emotional arc or embedding derived from ciphertext is
        silently wrong forever, and it corrupts every downstream consumer with
        no error anywhere (#2850). A dropped message is visible in the message
        counts; a ciphertext topic is not.
        """
        text = "" if content is None else str(content)
        store = self._conversation_store
        if store is not None and hasattr(store, "decrypt_stored_content"):
            try:
                text = store.decrypt_stored_content(text, metadata)
            except Exception as e:  # noqa: BLE001 - DecryptionError and friends
                logger.warning(
                    "episode consolidation skipped a message it could not "
                    "decrypt: %s", e,
                )
                return None
        if text and self._looks_like_ciphertext(text):
            # Unreachable once the store is wired; loud rather than silent
            # because this is precisely the bug that produced episode titles
            # reading "Discussion of ksav2, <base64>".
            logger.error(
                "episode consolidation refused a message that is still "
                "encrypted after decode (agent=%s) — check that the "
                "conversation store is wired into MemoryConsolidator",
                self.agent_id,
            )
            return None
        return text

    def _durable_writes_allowed(self) -> bool:
        """Whether the current privacy mode permits persisting an episode.

        ``memory_episodes`` and the KG episode node both hold user-derived
        conversation summaries, so a durable episode write while volatile
        (EPHEMERAL / ISOLATED / DEIDENTIFIED) is a leak. Manual ``memory_consolidate``
        and the scheduled ``sleep`` flow call ``_save_episode`` directly, outside
        the #1760 post-response gate, so this is the boundary that stops them
        (#2672). ``None`` policy (raw storage, tests) → allowed, preserving prior
        behaviour.
        """
        policy = self._persist_policy
        if policy is None:
            return True
        try:
            return bool(policy.allows_persistent_writes())
        except Exception as e:  # noqa: BLE001 - never let a policy probe crash consolidation
            logger.debug("episode persist-policy probe failed, denying: %s", e)
            return False

    async def _save_episode(self, episode: MemoryEpisode) -> None:
        """Save episode to database and optionally to the Knowledge Graph."""
        if not self._durable_writes_allowed():
            # Volatile privacy mode: an episode summary is user-derived content
            # that must not reach durable storage. Skip BOTH the memory_episodes
            # row and the KG node/edge; the KG path is separately governed by the
            # privacy-graph proxy, but gating here also blocks the direct table
            # write the proxy cannot see (#2672).
            logger.debug(
                "consolidator: skipping durable episode persist for %s — "
                "persistent writes are disabled in the current privacy mode "
                "(#2672)",
                episode.id,
            )
            return
        await self._db.execute(
            """INSERT INTO memory_episodes
               (id, agent_id, title, summary, timespan_start, timespan_end,
                key_message_ids, emotional_arc, created_at, importance)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.id,
                episode.agent_id,
                episode.title,
                episode.summary,
                episode.timespan_start.isoformat() if episode.timespan_start else None,
                episode.timespan_end.isoformat() if episode.timespan_end else None,
                json.dumps(episode.key_message_ids),
                episode.emotional_arc,
                datetime.now(timezone.utc).isoformat(),
                episode.importance,
            )
        )

        # Embed the episode so it's discoverable by relevance later (#1674 P2).
        # Best-effort and reuses the shared embedding service + vector column
        # (same path as saved_items); a missing provider just leaves NULL and
        # recall falls back to keyword search.
        await self._embed_episode(episode.id, episode.title, episode.summary)

        # Write episode as a KG node so it appears in the Memories panel
        if self._graph_store:
            try:
                from .async_graph_store import GraphNode

                episode_node = GraphNode(
                    node_id=episode.id,
                    node_type="episode",
                    label=episode.title,
                    properties={
                        "source": "consolidator",
                        "agent_id": self.agent_id,
                        "summary": episode.summary,
                        "emotional_arc": episode.emotional_arc,
                        "message_count": len(episode.key_message_ids),
                        "timespan_start": (
                            episode.timespan_start.isoformat()
                            if episode.timespan_start else None
                        ),
                        "timespan_end": (
                            episode.timespan_end.isoformat()
                            if episode.timespan_end else None
                        ),
                    },
                )
                await self._graph_store.add_node(episode_node)
                await self._graph_store.add_edge(
                    self.agent_id, episode.id, "remembers"
                )
            except Exception as e:
                # KG write is best-effort — don't fail episode creation
                logger.warning("Failed to write episode to KG: %s", e)

    # ------------------------------------------------------------------
    # Relevance-based episode recall + access tracking (#1674 P2)
    #
    # Thin glue over the SHARED semantic infra — no new embedding/kNN
    # machinery. Reuses get_provider_embedding_service, _serialize_embedding,
    # the vector backend (build_episode_spec + get_vector_backend), and the
    # embedding_profile_id filter, exactly as SavedItemsStore does.
    # ------------------------------------------------------------------

    @staticmethod
    def _episode_embed_text(title: Optional[str], summary: Optional[str]) -> str:
        """The text embedded/searched for an episode: title + summary."""
        return "\n".join(p for p in (title, summary) if p).strip()

    async def _embed_episode(
        self, episode_id: str, title: Optional[str], summary: Optional[str]
    ) -> None:
        """Embed an episode's title+summary and store it in ``embedding_vec``
        (+ the active ``embedding_profile_id``). Best-effort: any failure
        (no provider, missing column) leaves the row NULL and recall falls
        back to keyword search."""
        text = self._episode_embed_text(title, summary)
        if not text:
            return
        service = self._get_embedding_service()
        if service is None:
            return
        try:
            embedding = await service.aembed(text)
        except Exception as e:  # noqa: BLE001
            logger.debug("episode embed failed for %s: %s", episode_id, e)
            return
        if not embedding:
            return
        profile_id = None
        if hasattr(service, "current_profile_id"):
            try:
                profile_id = service.current_profile_id()
            except Exception:  # noqa: BLE001 - defensive
                profile_id = None
        from .saved_items_store import _serialize_embedding
        # Store float32 bytes on BOTH backends — episode embedding_vec is BLOB
        # (SQLite) / BYTEA (PG), not a native pgvector column yet. The SQLite
        # PurePythonBackend reads these bytes for cosine; PG semantic recall
        # awaits a pgvector episode migration (keyword recall works meanwhile),
        # mirroring saved_items' staged #1447 rollout. Writing ``?::vector``
        # here would fail against a BYTEA column.
        try:
            await self._db.execute(
                "UPDATE memory_episodes SET embedding_vec = ?, "
                "embedding_profile_id = ? WHERE id = ? AND agent_id = ?",
                (_serialize_embedding(embedding), profile_id,
                 episode_id, self.agent_id),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("episode embedding_vec write failed for %s: %s",
                         episode_id, e)

    async def _backfill_episode_embeddings(self, limit: int = 50) -> int:
        """Embed up to ``limit`` episodes that still lack an embedding (pre-P2
        rows). Bounded per consolidation pass so the corpus drains over nights
        without a heavy one-shot migration. Best-effort: returns the count
        embedded, 0 when no provider is available."""
        if self._get_embedding_service() is None:
            return 0
        try:
            rows = await self._db.fetchall(
                """SELECT id, title, summary FROM memory_episodes
                   WHERE agent_id = ? AND embedding_vec IS NULL
                     AND COALESCE(excluded_from_context, 0) = 0
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (self.agent_id, limit),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("episode embedding backfill query failed: %s", e)
            return 0
        embedded = 0
        for ep_id, title, summary in rows or []:
            before = await self._db.fetchone(
                "SELECT embedding_vec FROM memory_episodes WHERE id = ?",
                (ep_id,),
            )
            await self._embed_episode(ep_id, title, summary)
            after = await self._db.fetchone(
                "SELECT embedding_vec FROM memory_episodes WHERE id = ?",
                (ep_id,),
            )
            if after and after[0] is not None and (not before or before[0] is None):
                embedded += 1
        return embedded

    async def _increment_episode_access(self, episode_ids: List[str]) -> None:
        """Bump ``access_count`` for genuinely-recalled episodes (rehearsal
        signal feeding the decay half-life). Best-effort — bookkeeping must
        never break recall."""
        if not episode_ids:
            return
        placeholders = ",".join("?" for _ in episode_ids)
        try:
            await self._db.execute(
                f"UPDATE memory_episodes SET access_count = access_count + 1 "
                f"WHERE agent_id = ? AND id IN ({placeholders})",
                (self.agent_id, *episode_ids),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("episode access increment failed: %s", e)

    async def search_episodes(
        self, query: str, limit: int = 5
    ) -> List[MemoryEpisode]:
        """Recall episodes RELEVANT to ``query`` (not just recent), and stamp
        the surfaced episodes as accessed so they resist the deletion tier.

        Reuses the shared vector backend (SQLite PurePythonBackend) via
        ``build_episode_spec``, MERGED with a keyword LIKE scan over
        title+summary. The merge matters: semantic kNN only sees embedded
        rows, so legacy / not-yet-backfilled episodes (``embedding_vec IS
        NULL``) would otherwise be unrecallable even on an exact title match —
        and would never accrue the access-heat that protects them from the
        deletion tier. Vector hits rank first (relevance), keyword hits fill
        in behind, deduped and capped at ``limit``.
        """
        query = (query or "").strip()
        if not query:
            return []

        # Always run BOTH and merge, so un-embedded episodes stay recallable by
        # keyword. _knn returns None when the vector path can't run at all (no
        # provider / non-SQLite / error) — treat as empty.
        vec_ids = await self._knn_episode_ids(query, limit) or []
        # Exclude vector hits in SQL so keyword returns up to `limit` NEW matches
        # (legacy / un-embedded rows), not ones the vector path already covered.
        kw_only = await self._keyword_episode_ids(
            query, limit, exclude_ids=vec_ids
        )

        # Interleave (vector-first) rather than concatenate: a pure
        # vector-then-keyword merge lets `limit` embedded hits fill every slot
        # and starve an exact keyword match from a legacy NULL-embedding
        # episode — which then also never gets its access-heat protection. Round-
        # robin guarantees keyword-only hits get slots while the top relevance
        # hit still ranks first.
        ordered: List[str] = []
        seen: set = set()
        vi = ki = 0
        while len(ordered) < limit and (vi < len(vec_ids) or ki < len(kw_only)):
            if vi < len(vec_ids):
                eid = vec_ids[vi]
                vi += 1
                if eid not in seen:
                    seen.add(eid)
                    ordered.append(eid)
                if len(ordered) >= limit:
                    break
            if ki < len(kw_only):
                eid = kw_only[ki]
                ki += 1
                if eid not in seen:
                    seen.add(eid)
                    ordered.append(eid)

        episodes = await self._episodes_by_ids(ordered)
        await self._increment_episode_access([ep.id for ep in episodes])
        return episodes

    async def _knn_episode_ids(
        self, query: str, limit: int
    ) -> Optional[List[str]]:
        """Vector recall → ordered episode ids, or None when the vector path
        can't run (no provider / non-SQLite backend / no session factory /
        backend error / no embedded rows); the caller then relies on keyword
        recall.

        Gated to SQLite: episode ``embedding_vec`` is BYTEA on Postgres (not a
        native pgvector column yet), so the pgvector backend can't operate on
        it. PG semantic episode recall awaits a pgvector migration — keyword
        recall covers PG meanwhile. This mirrors saved_items' staged rollout."""
        if getattr(self._db, "backend_type", None) == "postgres":
            return None
        service = self._get_embedding_service()
        if service is None:
            return None
        try:
            from kestrel_sovereign.llm.embedding_service import (
                EPISODE_RETRIEVAL_INSTRUCTION,
                aembed_retrieval_query,
            )

            query_embedding = await aembed_retrieval_query(
                service,
                query,
                instruction=EPISODE_RETRIEVAL_INSTRUCTION,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("episode query embed failed: %s", e)
            return None
        if not query_embedding:
            return None
        sf = self._get_vector_session_factory()
        if sf is None:
            return None
        try:
            from .saved_items_store import _serialize_embedding
            from kestrel_sovereign.storage.sqla import build_episode_spec
            from kestrel_sovereign.storage.vector import get_vector_backend

            spec = build_episode_spec(dimension=len(query_embedding))
            filter_kwargs: Dict[str, Any] = {
                "agent_id": self.agent_id,
                "excluded_from_context": 0,
            }
            if hasattr(service, "current_profile_id"):
                try:
                    pid = service.current_profile_id()
                    if pid is not None:
                        filter_kwargs["embedding_profile_id"] = pid
                except Exception:  # noqa: BLE001
                    pass
            backend = get_vector_backend(sf, spec)
            top_k = await backend.knn(
                _serialize_embedding(query_embedding), k=limit,
                filter=filter_kwargs,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("episode kNN failed (%s); using keyword fallback", e)
            return None
        if not top_k:
            # No embedded rows for this scope yet → let keyword fallback run.
            return None
        return [episode_id for episode_id, _score in top_k]

    async def _keyword_episode_ids(
        self, query: str, limit: int, exclude_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Token-aware keyword recall over title+summary.

        ``exclude_ids`` (the vector hits) are filtered out IN SQL so the LIMIT
        returns up to ``limit`` genuinely-NEW matches — otherwise a query that
        the vector path already satisfied would consume the whole LIMIT with
        rows already surfaced, starving keyword-only legacy episodes."""
        query_projection = _strip_search_wrappers(query)
        query_tokens = list(dict.fromkeys(_tokenize_for_search(query_projection)))
        if not query_tokens:
            return []
        exclude_ids = exclude_ids or []
        exclude_clause = ""
        token_clause = " OR ".join(
            "(LOWER(title) LIKE ? OR LOWER(COALESCE(summary, '')) LIKE ?)"
            for _ in query_tokens
        )
        score_clause = " + ".join(
            "CASE WHEN LOWER(title) LIKE ? "
            "OR LOWER(COALESCE(summary, '')) LIKE ? THEN 1 ELSE 0 END"
            for _ in query_tokens
        )
        score_params: list = []
        match_params: list = []
        for token in query_tokens:
            like = f"%{token}%"
            score_params.extend((like, like))
            match_params.extend((like, like))
        params: list = [*score_params, self.agent_id, *match_params]
        if exclude_ids:
            exclude_clause = (
                " AND id NOT IN (" + ",".join("?" for _ in exclude_ids) + ")"
            )
            params.extend(exclude_ids)
        # Fetch a bounded superset so overlap ranking, not raw recency, chooses
        # the final rows. Exact phrases receive a deterministic preference.
        params.append(max(limit * 5, limit))
        try:
            rows = await self._db.fetchall(
                f"""SELECT id, title, summary, created_at,
                           ({score_clause}) AS token_match_count
                    FROM memory_episodes
                    WHERE agent_id = ?
                      AND COALESCE(excluded_from_context, 0) = 0
                      AND ({token_clause})
                    {exclude_clause}
                    ORDER BY token_match_count DESC, created_at DESC
                    LIMIT ?""",
                tuple(params),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("episode keyword recall failed: %s", e)
            return []
        query_lower = " ".join(query_projection.lower().split())
        ranked: List[Tuple[float, int, str]] = []
        for position, row in enumerate(rows or []):
            episode_id, title, summary, _created_at, _match_count = row
            combined = f"{title or ''} {summary or ''}".lower()
            overlap = _token_match_score(query_tokens, combined)
            phrase_bonus = 1.0 if query_lower and query_lower in combined else 0.0
            ranked.append((overlap + phrase_bonus, -position, episode_id))
        ranked.sort(reverse=True)
        return [episode_id for _score, _position, episode_id in ranked[:limit]]

    async def _episodes_by_ids(self, ids: List[str]) -> List[MemoryEpisode]:
        """Materialize MemoryEpisode rows by id, preserving the given order."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = await self._db.fetchall(
            f"""SELECT id, agent_id, title, summary, timespan_start, timespan_end,
                       key_message_ids, emotional_arc, created_at, importance,
                       access_count
                FROM memory_episodes
                WHERE agent_id = ? AND id IN ({placeholders})
                  AND COALESCE(excluded_from_context, 0) = 0""",
            (self.agent_id, *ids),
        )
        by_id = {r[0]: MemoryEpisode.from_row(r) for r in rows or []}
        # Preserve relevance order from the recall step.
        return [by_id[i] for i in ids if i in by_id]

    async def _detect_patterns(self) -> List[TemporalPattern]:
        """
        Detect temporal patterns from recent messages.

        Delegates to TemporalAnalyzer.
        """
        from .temporal_analyzer import TemporalAnalyzer

        # Get messages from last 90 days for pattern detection
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        created_at_predicate = timestamp_predicate(
            self._db.backend_type, "created_at", ">"
        )

        rows = await self._db.fetchall(
            f"""SELECT content, metadata, created_at
               FROM conversation_history
               WHERE agent_id = ? AND {created_at_predicate}
                 AND deleted_at IS NULL AND archived_at IS NULL
               ORDER BY created_at""",
            (self.agent_id, timestamp_query_param(self._db.backend_type, cutoff))
        )

        messages = []
        for row in rows:
            content, metadata, created_at = row
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            content = self._row_plaintext(content, metadata)  # #2850
            if content is None:
                continue
            # Temporal patterns are durable derived memory too.  Do not allow
            # a hidden semantic-recall artifact to re-enter context through a
            # newly detected pattern on the next sleep cycle.
            if metadata.get("excluded_from_context"):
                continue
            messages.append({
                "content": content,
                "metadata": metadata,
                "created_at": created_at,
            })

        analyzer = TemporalAnalyzer(self._db)
        patterns = await analyzer.detect_patterns(messages, self.agent_id)

        # Save patterns
        if patterns:
            await analyzer.save_patterns(patterns)

        return patterns

    async def _archive_decayed(self) -> int:
        """
        Mark fully decayed messages as archived.

        Archived messages are not deleted; ``conversation_history.archived_at``
        is the sole archive-state field. Metadata retains decay evidence only.
        They won't appear in normal retrieval but can still be
        accessed if specifically requested.

        Returns:
            Number of messages archived
        """
        archived_count = 0

        # Get all messages (paginated for large histories)
        rows = await self._db.fetchall(
            """SELECT id, metadata, created_at
               FROM conversation_history
               WHERE agent_id = ?
                 AND deleted_at IS NULL AND archived_at IS NULL
               ORDER BY created_at""",
            (self.agent_id,)
        )

        for row in rows:
            msg_id, metadata, created_at = row

            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            # Keep the exclusion barrier sticky.  Decay processing is not a
            # context producer, but must not normalize an excluded row into a
            # legacy archive representation that other paths could restore.
            if metadata.get("excluded_from_context"):
                continue

            # Migrate the legacy dual-state representation. Older releases
            # wrote metadata.archived without the dedicated column, making
            # those rows visible to normal retrieval forever. Canonicalize
            # them in-place and remove the redundant state keys.
            legacy_archived = metadata.get("archived") in (True, 1, "true", "True")
            if legacy_archived:
                archived_at = metadata.pop("archived_at", None)
                metadata.pop("archived", None)
                if not archived_at:
                    archived_at = datetime.now(timezone.utc).isoformat()
                await self._db.execute(
                    """UPDATE conversation_history
                       SET metadata = ?, archived_at = ?
                       WHERE id = ? AND agent_id = ?
                         AND deleted_at IS NULL AND archived_at IS NULL""",
                    (json.dumps(metadata), archived_at, msg_id, self.agent_id),
                )
                archived_count += 1
                continue

            # decay_protected pins prevent ROUTINE archival only.
            # Sovereign deletion (privacy wipes, compliance erasure) overrides
            # pins unconditionally -- see MemoryAgencyFeature.sovereign_override_pins().
            if metadata.get("decay_protected"):
                continue

            # Calculate decay.  ``applied_count`` is the load-bearing
            # signal added in #1326 — a memory that's been demonstrably
            # applied decays slower than one that's merely been
            # retrieved at the same rate.  Default 0 keeps behavior
            # unchanged for pre-#1326 metadata rows.
            importance = metadata.get("importance", 0.5)
            access_count = metadata.get("access_count", 0)
            applied_count = metadata.get("applied_count", 0)

            strength = calculate_decay(
                created_at,
                importance=importance,
                access_count=access_count,
                applied_count=applied_count,
                decay_protected=False,
            )

            # Archive if below threshold
            if strength < self.DECAY_ARCHIVE_THRESHOLD:
                archived_at = datetime.now(timezone.utc).isoformat()
                metadata["archived_strength"] = strength

                await self._db.execute(
                    """UPDATE conversation_history
                       SET metadata = ?, archived_at = ?
                       WHERE id = ? AND agent_id = ?
                         AND deleted_at IS NULL AND archived_at IS NULL""",
                    (json.dumps(metadata), archived_at, msg_id, self.agent_id)
                )
                archived_count += 1

        return archived_count

    async def should_create_episode(self, session_messages: int = 0) -> bool:
        """
        Check if an episode should be created for the current session.

        Triggers:
        1. Message count exceeds threshold (SESSION_EPISODE_THRESHOLD)
        2. Session gap detected (SESSION_GAP_MINUTES of inactivity)

        Args:
            session_messages: Number of messages in current session

        Returns:
            True if episode should be created
        """
        if session_messages >= self.SESSION_EPISODE_THRESHOLD:
            return True

        # Check for session gap (inactivity)
        last_message = await self._db.fetchone(
            """SELECT created_at FROM conversation_history
               WHERE agent_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (self.agent_id,)
        )

        if last_message and last_message[0]:
            try:
                if isinstance(last_message[0], str):
                    last_time = datetime.fromisoformat(
                        last_message[0].replace("Z", "+00:00")
                    )
                else:
                    last_time = last_message[0]

                gap = datetime.now(timezone.utc) - last_time
                if gap.total_seconds() > self.SESSION_GAP_MINUTES * 60:
                    return True
            except (ValueError, TypeError):
                pass

        return False

    async def create_session_episode(
        self,
        force: bool = False
    ) -> Optional[MemoryEpisode]:
        """
        Create an episode from the current session's messages.

        Called when:
        - Session ends (30-min gap detected)
        - Message threshold exceeded
        - Manual trigger via !consolidate

        Args:
            force: Create episode even if threshold not met

        Returns:
            Created MemoryEpisode or None if not enough messages
        """
        # Get messages since last episode or session start
        last_episode = await self._db.fetchone(
            """SELECT timespan_end FROM memory_episodes
               WHERE agent_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (self.agent_id,)
        )

        cutoff = None
        if last_episode and last_episode[0]:
            cutoff = last_episode[0]

        # Build query for recent messages
        if cutoff:
            created_at_predicate = timestamp_predicate(
                self._db.backend_type, "created_at", ">"
            )
            rows = await self._db.fetchall(
                f"""SELECT id, content, metadata, created_at, role
                   FROM conversation_history
                   WHERE agent_id = ? AND {created_at_predicate}
                     AND deleted_at IS NULL AND archived_at IS NULL
                   ORDER BY created_at""",
                (
                    self.agent_id,
                    timestamp_query_param(self._db.backend_type, cutoff),
                )
            )
        else:
            # Get messages from last 24 hours if no previous episode
            cutoff_time = datetime.now(timezone.utc) - timedelta(
                hours=self.MAX_EPISODE_HOURS
            )
            created_at_predicate = timestamp_predicate(
                self._db.backend_type, "created_at", ">"
            )
            rows = await self._db.fetchall(
                f"""SELECT id, content, metadata, created_at, role
                   FROM conversation_history
                   WHERE agent_id = ? AND {created_at_predicate}
                     AND deleted_at IS NULL AND archived_at IS NULL
                   ORDER BY created_at""",
                (
                    self.agent_id,
                    timestamp_query_param(self._db.backend_type, cutoff_time),
                )
            )

        if not rows:
            return None

        # Convert rows to message dicts
        messages = []
        for row in rows:
            msg_id, content, metadata, created_at, role = row

            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            # Session episodes use a separate source query from nightly
            # consolidation.  Apply the same barrier here or sleep/restart can
            # recreate visible episode text from an excluded derivative.
            if metadata.get("excluded_from_context"):
                continue

            # #2850: same raw-SQL source, same envelope — decode before this
            # content reaches title/summary/affect synthesis.
            content = self._row_plaintext(content, metadata)
            if content is None:
                continue

            messages.append({
                "id": msg_id,
                "content": content,
                "metadata": metadata,
                "created_at": created_at,
                "role": role,
            })

        # The threshold is intentionally evaluated after exclusion.  Hidden
        # artifacts must neither produce an episode nor count toward forcing
        # a visible episode from the remaining unrelated messages.
        if len(messages) < self.MIN_EPISODE_MESSAGES and not force:
            return None
        if not messages:
            return None

        # Calculate emotional intensity for episode worthiness
        intensities = [
            msg.get("metadata", {}).get("emotional_intensity", 0.0)
            for msg in messages
        ]
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0
        # Stamp the same importance decay signal as nightly episodes (#1674) so
        # session episodes participate in the forgetting curve consistently —
        # without this they'd all default to 0.5 regardless of their content.
        importances = [
            msg.get("metadata", {}).get("importance", 0.5)
            for msg in messages
        ]
        avg_importance = sum(importances) / len(importances) if importances else 0.5

        # Create episode
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
        episode = await self._create_episode_from_messages(
            date_key, messages, avg_intensity, avg_importance
        )

        if episode:
            await self._save_episode(episode)
            logger.info(
                f"Session episode created: {episode.title} "
                f"({len(messages)} messages, intensity={avg_intensity:.2f})"
            )

        return episode

    async def get_recent_episodes_for_context(
        self,
        max_tokens: int = 2000,
        max_episodes: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recent episodes formatted for context inclusion.

        Returns episodes optimized for LLM context, with summaries
        and emotional arcs that help the agent understand the
        conversation's history.

        Args:
            max_tokens: Approximate token budget for episodes
            max_episodes: Maximum number of episodes to return

        Returns:
            List of episode dicts with title, summary, emotional_arc
        """
        episodes = await self.get_episodes(limit=max_episodes)

        if not episodes:
            return []

        # Format for context (estimate ~50 tokens per episode summary)
        result = []
        estimated_tokens = 0
        tokens_per_episode = 50  # Conservative estimate

        for ep in episodes:
            if estimated_tokens + tokens_per_episode > max_tokens:
                break

            result.append({
                "title": ep.title,
                "summary": ep.summary,
                "emotional_arc": ep.emotional_arc,
                "timespan": (
                    ep.timespan_start.strftime("%Y-%m-%d")
                    if ep.timespan_start else "unknown"
                ),
            })
            estimated_tokens += tokens_per_episode

        return result

    async def get_episodes(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[MemoryEpisode]:
        """
        Get stored episodes for this agent.

        Args:
            limit: Max episodes to return
            offset: Offset for pagination

        Returns:
            List of MemoryEpisode objects
        """
        rows = await self._db.fetchall(
            """SELECT id, agent_id, title, summary, timespan_start, timespan_end,
                      key_message_ids, emotional_arc, created_at, importance,
                      access_count
               FROM memory_episodes
               WHERE agent_id = ? AND COALESCE(excluded_from_context, 0) = 0
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (self.agent_id, limit, offset)
        )

        episodes = []
        for row in rows:
            (ep_id, agent_id, title, summary, timespan_start, timespan_end,
             key_message_ids, emotional_arc, created_at, importance,
             access_count) = row

            # Parse JSON fields
            if isinstance(key_message_ids, str):
                try:
                    key_message_ids = json.loads(key_message_ids)
                except (json.JSONDecodeError, TypeError):
                    key_message_ids = []

            # Parse timestamps
            try:
                timespan_start = datetime.fromisoformat(
                    timespan_start.replace("Z", "+00:00")
                ) if timespan_start else None
            except (ValueError, TypeError):
                timespan_start = None

            try:
                timespan_end = datetime.fromisoformat(
                    timespan_end.replace("Z", "+00:00")
                ) if timespan_end else None
            except (ValueError, TypeError):
                timespan_end = None

            try:
                created_at = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                ) if created_at else None
            except (ValueError, TypeError):
                created_at = None

            episodes.append(MemoryEpisode(
                id=ep_id,
                agent_id=agent_id,
                title=title,
                summary=summary,
                timespan_start=timespan_start,
                timespan_end=timespan_end,
                key_message_ids=key_message_ids,
                emotional_arc=emotional_arc,
                created_at=created_at,
                importance=importance if importance is not None else 0.5,
                access_count=access_count if access_count is not None else 0,
            ))

        return episodes
