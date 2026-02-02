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
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

from .memory_models import MemoryEpisode, TemporalPattern
from .async_database import AsyncDatabase
from .memory_retriever import calculate_decay

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
    SESSION_GAP_MINUTES = 30       # Minutes of inactivity = session end

    def __init__(self, db: AsyncDatabase, agent_id: str):
        """
        Initialize consolidator.

        Args:
            db: AsyncDatabase instance
            agent_id: Agent ID to consolidate memories for
        """
        self._db = db
        self.agent_id = agent_id

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # 1. Create episodes from high-importance clusters
            episodes = await self._create_episodes()
            report["episodes_created"] = len(episodes)

            # 2. Detect temporal patterns
            patterns = await self._detect_patterns()
            report["patterns_found"] = len(patterns)

            # 3. Archive decayed messages
            archived = await self._archive_decayed()
            report["messages_archived"] = archived

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

    async def _create_episodes(self) -> List[MemoryEpisode]:
        """
        Group related messages into narrative episodes.

        Strategy:
        - Look at messages from last 30 days
        - Group by day
        - Within each day, find high-emotion clusters
        - Create episode if cluster has enough messages
        """
        episodes: List[MemoryEpisode] = []

        # Get messages from last 30 days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        rows = await self._db.fetchall(
            """SELECT id, content, metadata, created_at, role
               FROM conversation_history
               WHERE agent_id = ? AND created_at > ?
               ORDER BY created_at""",
            (self.agent_id, cutoff)
        )

        if not rows:
            return episodes

        # Group messages by date
        by_date: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            msg_id, content, metadata, created_at, role = row

            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            # Parse date from created_at
            try:
                if isinstance(created_at, str):
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    dt = created_at
                date_key = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            by_date[date_key].append({
                "id": msg_id,
                "content": content,
                "metadata": metadata,
                "created_at": created_at,
                "role": role,
            })

        # For each day, check if there's a significant cluster
        for date_key, messages in by_date.items():
            if len(messages) < self.MIN_EPISODE_MESSAGES:
                continue

            # Calculate average emotional intensity
            intensities = []
            importances = []
            for msg in messages:
                meta = msg.get("metadata", {})
                intensities.append(meta.get("emotional_intensity", 0.0))
                importances.append(meta.get("importance", 0.5))

            avg_intensity = sum(intensities) / len(intensities) if intensities else 0
            avg_importance = sum(importances) / len(importances) if importances else 0.5

            # Only create episode if emotionally significant
            if avg_intensity < 0.3 and avg_importance < 0.6:
                continue

            # Create episode
            episode = await self._create_episode_from_messages(
                date_key, messages, avg_intensity
            )
            if episode:
                episodes.append(episode)
                await self._save_episode(episode)

        return episodes

    async def _create_episode_from_messages(
        self,
        date_key: str,
        messages: List[Dict],
        avg_intensity: float
    ) -> Optional[MemoryEpisode]:
        """Create a MemoryEpisode from a cluster of messages."""
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
            return "A memorable exchange"

    def _generate_episode_summary(
        self,
        messages: List[Dict],
        emotional_arc: str
    ) -> str:
        """Generate a summary of the episode."""
        user_messages = [m for m in messages if m.get("role") == "user"]
        message_count = len(messages)
        user_count = len(user_messages)

        # Simple summary (could be LLM-generated for richer summaries)
        return (
            f"A conversation with {message_count} messages "
            f"({user_count} from user). Emotional trajectory: {emotional_arc}."
        )

    async def _save_episode(self, episode: MemoryEpisode) -> None:
        """Save episode to database."""
        await self._db.execute(
            """INSERT INTO memory_episodes
               (id, agent_id, title, summary, timespan_start, timespan_end,
                key_message_ids, emotional_arc, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            )
        )

    async def _detect_patterns(self) -> List[TemporalPattern]:
        """
        Detect temporal patterns from recent messages.

        Delegates to TemporalAnalyzer.
        """
        from .temporal_analyzer import TemporalAnalyzer

        # Get messages from last 90 days for pattern detection
        cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()

        rows = await self._db.fetchall(
            """SELECT content, metadata, created_at
               FROM conversation_history
               WHERE agent_id = ? AND created_at > ?
               ORDER BY created_at""",
            (self.agent_id, cutoff)
        )

        messages = []
        for row in rows:
            content, metadata, created_at = row
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
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

        Archived messages are not deleted, just marked in metadata.
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

            # Skip already archived
            if metadata.get("archived"):
                continue

            # Skip protected
            if metadata.get("decay_protected"):
                continue

            # Calculate decay
            importance = metadata.get("importance", 0.5)
            access_count = metadata.get("access_count", 0)

            strength = calculate_decay(
                created_at,
                importance=importance,
                access_count=access_count,
                decay_protected=False,
            )

            # Archive if below threshold
            if strength < self.DECAY_ARCHIVE_THRESHOLD:
                metadata["archived"] = True
                metadata["archived_at"] = datetime.now(timezone.utc).isoformat()
                metadata["archived_strength"] = strength

                await self._db.execute(
                    """UPDATE conversation_history
                       SET metadata = ?
                       WHERE id = ?""",
                    (json.dumps(metadata), msg_id)
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
            rows = await self._db.fetchall(
                """SELECT id, content, metadata, created_at, role
                   FROM conversation_history
                   WHERE agent_id = ? AND created_at > ?
                   ORDER BY created_at""",
                (self.agent_id, cutoff)
            )
        else:
            # Get messages from last 24 hours if no previous episode
            cutoff_time = (
                datetime.now(timezone.utc) - timedelta(hours=self.MAX_EPISODE_HOURS)
            ).isoformat()
            rows = await self._db.fetchall(
                """SELECT id, content, metadata, created_at, role
                   FROM conversation_history
                   WHERE agent_id = ? AND created_at > ?
                   ORDER BY created_at""",
                (self.agent_id, cutoff_time)
            )

        if not rows:
            return None

        # Check minimum message count
        if len(rows) < self.MIN_EPISODE_MESSAGES and not force:
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

            messages.append({
                "id": msg_id,
                "content": content,
                "metadata": metadata,
                "created_at": created_at,
                "role": role,
            })

        # Calculate emotional intensity for episode worthiness
        intensities = [
            msg.get("metadata", {}).get("emotional_intensity", 0.0)
            for msg in messages
        ]
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0

        # Create episode
        date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
        episode = await self._create_episode_from_messages(
            date_key, messages, avg_intensity
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
                      key_message_ids, emotional_arc, created_at
               FROM memory_episodes
               WHERE agent_id = ?
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (self.agent_id, limit, offset)
        )

        episodes = []
        for row in rows:
            (ep_id, agent_id, title, summary, timespan_start, timespan_end,
             key_message_ids, emotional_arc, created_at) = row

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
            ))

        return episodes
