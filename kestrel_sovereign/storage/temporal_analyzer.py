"""
Temporal pattern detection for user behavior.

Analyzes conversation history to detect patterns in:
- When users communicate (time preferences)
- Emotional patterns by time (e.g., "deeper conversations late at night")
- Weekly rhythms (e.g., "reflective on Sundays")
- Topic patterns by time

These patterns enable proactive engagement and deeper understanding.
"""
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import uuid

from .memory_models import TemporalPattern, MemoryMetadata
from .async_database import AsyncDatabase

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Detects patterns in when and how users communicate.

    Analyzes message history to find:
    - Time preference patterns (most active time of day)
    - Day preference patterns (most active day of week)
    - Emotional × time correlations (e.g., sad on Sunday evenings)
    - Topic × time correlations (work topics on weekdays)
    """

    def __init__(self, db: AsyncDatabase):
        """
        Initialize with database connection.

        Args:
            db: AsyncDatabase instance for pattern storage
        """
        self._db = db

    async def detect_patterns(
        self,
        messages: List[Dict[str, Any]],
        agent_id: str,
        min_observations: int = 5
    ) -> List[TemporalPattern]:
        """
        Analyze message history for temporal patterns.

        Args:
            messages: List of message dicts with 'metadata' containing
                     time_of_day, day_of_week, emotional_valence, etc.
            agent_id: Agent ID for scoping patterns
            min_observations: Minimum observations to report a pattern

        Returns:
            List of detected TemporalPattern objects
        """
        if not messages:
            return []

        patterns = []

        # Detect time-of-day preference
        time_pattern = await self._detect_time_preference(
            messages, agent_id, min_observations
        )
        if time_pattern:
            patterns.append(time_pattern)

        # Detect day-of-week preference
        day_pattern = await self._detect_day_preference(
            messages, agent_id, min_observations
        )
        if day_pattern:
            patterns.append(day_pattern)

        # Detect emotional × time correlations
        emotion_time_patterns = await self._detect_emotion_time_correlation(
            messages, agent_id, min_observations
        )
        patterns.extend(emotion_time_patterns)

        return patterns

    async def _detect_time_preference(
        self,
        messages: List[Dict[str, Any]],
        agent_id: str,
        min_observations: int
    ) -> Optional[TemporalPattern]:
        """Detect user's preferred time of day for communication."""
        time_groups: Dict[str, int] = defaultdict(int)

        for msg in messages:
            metadata = msg.get("metadata", {})
            if isinstance(metadata, str):
                # Handle JSON string metadata
                import json
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            tod = metadata.get("time_of_day", "")
            if tod:
                time_groups[tod] += 1

        if not time_groups:
            return None

        # Find most active time
        most_active = max(time_groups, key=lambda k: time_groups[k])
        count = time_groups[most_active]
        total = sum(time_groups.values())

        if count < min_observations:
            return None

        confidence = count / total

        # Generate human-readable description
        time_descriptions = {
            "morning": "early in the morning",
            "afternoon": "during the afternoon",
            "evening": "in the evening",
            "late_night": "late at night",
        }
        desc = time_descriptions.get(most_active, most_active)

        return TemporalPattern(
            id=f"time_pref_{agent_id}_{most_active}",
            agent_id=agent_id,
            pattern_type="time_preference",
            description=f"User is most active {desc} ({count} of {total} messages, {confidence:.0%})",
            trigger_conditions={"time_of_day": most_active},
            confidence=confidence,
            observations=count,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    async def _detect_day_preference(
        self,
        messages: List[Dict[str, Any]],
        agent_id: str,
        min_observations: int
    ) -> Optional[TemporalPattern]:
        """Detect user's preferred day of week for communication."""
        day_groups: Dict[str, int] = defaultdict(int)

        for msg in messages:
            metadata = msg.get("metadata", {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            dow = metadata.get("day_of_week", "")
            if dow:
                day_groups[dow] += 1

        if not day_groups:
            return None

        most_active = max(day_groups, key=lambda k: day_groups[k])
        count = day_groups[most_active]
        total = sum(day_groups.values())

        if count < min_observations:
            return None

        confidence = count / total

        return TemporalPattern(
            id=f"day_pref_{agent_id}_{most_active}",
            agent_id=agent_id,
            pattern_type="day_preference",
            description=f"User is most active on {most_active.capitalize()}s ({count} of {total} messages, {confidence:.0%})",
            trigger_conditions={"day_of_week": most_active},
            confidence=confidence,
            observations=count,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    async def _detect_emotion_time_correlation(
        self,
        messages: List[Dict[str, Any]],
        agent_id: str,
        min_observations: int
    ) -> List[TemporalPattern]:
        """
        Detect emotional patterns correlated with time.

        Examples:
        - "User tends to be reflective late at night"
        - "User often expresses anxiety on Sunday evenings"
        """
        patterns = []

        # Group messages by time_of_day, track emotional valence
        emotion_by_time: Dict[str, List[float]] = defaultdict(list)

        for msg in messages:
            metadata = msg.get("metadata", {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            tod = metadata.get("time_of_day", "")
            valence = metadata.get("emotional_valence")

            if tod and valence is not None:
                emotion_by_time[tod].append(valence)

        # Check for significant emotional patterns by time
        for time_of_day, valences in emotion_by_time.items():
            if len(valences) < min_observations:
                continue

            avg_valence = sum(valences) / len(valences)

            # Significant negative pattern
            if avg_valence < -0.3:
                intensity_word = "often" if avg_valence < -0.5 else "sometimes"
                time_desc = self._time_description(time_of_day)

                patterns.append(TemporalPattern(
                    id=f"emotion_time_{agent_id}_{time_of_day}_negative",
                    agent_id=agent_id,
                    pattern_type="emotional_time_correlation",
                    description=f"User {intensity_word} feels down {time_desc} (avg valence: {avg_valence:.2f})",
                    trigger_conditions={
                        "time_of_day": time_of_day,
                        "emotional_tendency": "negative",
                        "avg_valence": avg_valence,
                    },
                    confidence=abs(avg_valence),
                    observations=len(valences),
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                ))

            # Significant positive pattern
            elif avg_valence > 0.3:
                intensity_word = "usually" if avg_valence > 0.5 else "often"
                time_desc = self._time_description(time_of_day)

                patterns.append(TemporalPattern(
                    id=f"emotion_time_{agent_id}_{time_of_day}_positive",
                    agent_id=agent_id,
                    pattern_type="emotional_time_correlation",
                    description=f"User is {intensity_word} in good spirits {time_desc} (avg valence: {avg_valence:.2f})",
                    trigger_conditions={
                        "time_of_day": time_of_day,
                        "emotional_tendency": "positive",
                        "avg_valence": avg_valence,
                    },
                    confidence=abs(avg_valence),
                    observations=len(valences),
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                ))

        # Check for day × emotion correlations
        emotion_by_day: Dict[str, List[float]] = defaultdict(list)

        for msg in messages:
            metadata = msg.get("metadata", {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            dow = metadata.get("day_of_week", "")
            valence = metadata.get("emotional_valence")

            if dow and valence is not None:
                emotion_by_day[dow].append(valence)

        for day_of_week, valences in emotion_by_day.items():
            if len(valences) < min_observations:
                continue

            avg_valence = sum(valences) / len(valences)

            # Significant negative pattern on specific day
            if avg_valence < -0.3:
                patterns.append(TemporalPattern(
                    id=f"emotion_day_{agent_id}_{day_of_week}_negative",
                    agent_id=agent_id,
                    pattern_type="emotional_day_correlation",
                    description=f"User tends to feel down on {day_of_week.capitalize()}s (avg valence: {avg_valence:.2f})",
                    trigger_conditions={
                        "day_of_week": day_of_week,
                        "emotional_tendency": "negative",
                        "avg_valence": avg_valence,
                    },
                    confidence=abs(avg_valence),
                    observations=len(valences),
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                ))

        return patterns

    def _time_description(self, time_of_day: str) -> str:
        """Convert time_of_day to human-readable phrase."""
        descriptions = {
            "morning": "in the morning",
            "afternoon": "in the afternoon",
            "evening": "in the evening",
            "late_night": "late at night",
        }
        return descriptions.get(time_of_day, time_of_day)

    async def save_patterns(
        self,
        patterns: List[TemporalPattern]
    ) -> int:
        """
        Save detected patterns to database.

        Args:
            patterns: List of TemporalPattern to save

        Returns:
            Number of patterns saved
        """
        import json

        saved = 0
        for pattern in patterns:
            # Upsert pattern (update if exists, insert if not)
            existing = await self._db.fetchone(
                "SELECT id FROM temporal_patterns WHERE id = ?",
                (pattern.id,)
            )

            if existing:
                # Update existing pattern
                await self._db.execute(
                    """UPDATE temporal_patterns
                       SET description = ?, trigger_conditions = ?,
                           confidence = ?, observations = ?, updated_at = ?
                       WHERE id = ?""",
                    (
                        pattern.description,
                        json.dumps(pattern.trigger_conditions),
                        pattern.confidence,
                        pattern.observations,
                        datetime.now(timezone.utc).isoformat(),
                        pattern.id,
                    )
                )
            else:
                # Insert new pattern
                await self._db.execute(
                    """INSERT INTO temporal_patterns
                       (id, agent_id, pattern_type, description,
                        trigger_conditions, confidence, observations,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pattern.id,
                        pattern.agent_id,
                        pattern.pattern_type,
                        pattern.description,
                        json.dumps(pattern.trigger_conditions),
                        pattern.confidence,
                        pattern.observations,
                        datetime.now(timezone.utc).isoformat(),
                        datetime.now(timezone.utc).isoformat(),
                    )
                )
            saved += 1

        return saved

    async def get_patterns(
        self,
        agent_id: str,
        pattern_type: Optional[str] = None
    ) -> List[TemporalPattern]:
        """
        Retrieve stored patterns for an agent.

        Args:
            agent_id: Agent ID to get patterns for
            pattern_type: Optional filter by pattern type

        Returns:
            List of TemporalPattern objects
        """
        import json

        if pattern_type:
            rows = await self._db.fetchall(
                """SELECT * FROM temporal_patterns
                   WHERE agent_id = ? AND pattern_type = ?
                   ORDER BY confidence DESC""",
                (agent_id, pattern_type)
            )
        else:
            rows = await self._db.fetchall(
                """SELECT * FROM temporal_patterns
                   WHERE agent_id = ?
                   ORDER BY confidence DESC""",
                (agent_id,)
            )

        patterns = []
        for row in rows:
            # row is a tuple: (id, agent_id, pattern_type, description,
            #                  trigger_conditions, confidence, observations,
            #                  created_at, updated_at)
            trigger_conditions = row[4]
            if isinstance(trigger_conditions, str):
                try:
                    trigger_conditions = json.loads(trigger_conditions)
                except (json.JSONDecodeError, TypeError):
                    trigger_conditions = {}

            patterns.append(TemporalPattern(
                id=row[0],
                agent_id=row[1],
                pattern_type=row[2],
                description=row[3],
                trigger_conditions=trigger_conditions,
                confidence=row[5],
                observations=row[6],
                created_at=datetime.fromisoformat(row[7]) if row[7] else None,
                updated_at=datetime.fromisoformat(row[8]) if row[8] else None,
            ))

        return patterns

    async def get_current_context(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Get patterns relevant to current time context.

        Args:
            agent_id: Agent ID

        Returns:
            Dict with current temporal context and matching patterns
        """
        now = datetime.now()
        current_time = self._get_time_of_day(now)
        current_day = now.strftime("%A").lower()

        # Get all patterns for this agent
        patterns = await self.get_patterns(agent_id)

        # Filter to patterns matching current context
        matching = []
        for p in patterns:
            conditions = p.trigger_conditions
            if conditions.get("time_of_day") == current_time:
                matching.append(p)
            elif conditions.get("day_of_week") == current_day:
                matching.append(p)

        return {
            "current_time_of_day": current_time,
            "current_day_of_week": current_day,
            "matching_patterns": matching,
            "pattern_insights": [p.description for p in matching],
        }

    @staticmethod
    def _get_time_of_day(dt: datetime) -> str:
        """Classify hour into time period."""
        hour = dt.hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "late_night"

    @staticmethod
    def get_time_of_day(dt: Optional[datetime] = None) -> str:
        """
        Classify time into period.

        Public static method for use by other modules.

        Args:
            dt: Datetime to classify (default: now)

        Returns:
            One of: 'morning', 'afternoon', 'evening', 'late_night'
        """
        dt = dt or datetime.now()
        return TemporalAnalyzer._get_time_of_day(dt)

    @staticmethod
    def get_day_of_week(dt: Optional[datetime] = None) -> str:
        """
        Get lowercase day name.

        Public static method for use by other modules.

        Args:
            dt: Datetime to get day for (default: now)

        Returns:
            Day name like 'monday', 'tuesday', etc.
        """
        dt = dt or datetime.now()
        return dt.strftime("%A").lower()
