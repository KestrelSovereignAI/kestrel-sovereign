"""
Operational Wellness Metric Calculators.

Each calculator measures one dimension of agent operational health:
- Constitutional Friction: rate of denied/blocked actions from audit log
- Context Pressure: context window utilization
- Interaction Depth: quality and substance of recent interactions
- Session Continuity: consistency and regularity of engagement
- Memory Health: status of the agent's memory system

Each calculator has an async measure() method that returns a dict
with metric-specific data plus a normalized 0.0-1.0 score.
All calculators handle missing tables/data gracefully.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


class FrictionCalculator:
    """Measures constitutional friction from the security audit log.

    Friction is the rate at which agent actions are denied or blocked
    by the security/permissions system. High friction indicates either
    overly restrictive permissions or the agent attempting actions it
    should not.
    """

    # Decisions considered "friction" (denied, timed out, etc.)
    FRICTION_DECISIONS = {"auto_denied", "user_denied", "timeout", "denied", "blocked"}

    async def measure(self, db, agent_id: str, lookback_hours: int = 24) -> Dict[str, Any]:
        """Measure constitutional friction from audit log.

        Args:
            db: AsyncDatabase instance
            agent_id: Agent identifier (unused for audit log which is global)
            lookback_hours: Hours to look back in the audit log

        Returns:
            Dict with total_events, friction_events, friction_rate (0.0-1.0)
        """
        defaults = {
            "total_events": 0,
            "friction_events": 0,
            "friction_rate": 0.0,
            "available": False,
        }

        if not db:
            return defaults

        try:
            # Check if the security_audit_log table exists
            exists = await db.table_exists("security_audit_log")
            if not exists:
                return defaults

            cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()

            rows = await db.fetchall(
                "SELECT decision FROM security_audit_log WHERE created_at >= ?",
                (cutoff,),
            )

            total = len(rows)
            if total == 0:
                return {
                    "total_events": 0,
                    "friction_events": 0,
                    "friction_rate": 0.0,
                    "available": True,
                }

            friction_count = sum(
                1 for row in rows if row[0] in self.FRICTION_DECISIONS
            )

            return {
                "total_events": total,
                "friction_events": friction_count,
                "friction_rate": round(friction_count / total, 4),
                "available": True,
            }

        except Exception as e:
            logger.warning(f"FrictionCalculator failed: {e}")
            return defaults


class ContextPressureCalculator:
    """Measures context window utilization.

    Context pressure reflects how close the agent is to filling
    its context window. High pressure can degrade response quality.
    """

    async def measure(self, agent) -> Dict[str, Any]:
        """Measure context window pressure.

        Args:
            agent: The KestrelAgent instance

        Returns:
            Dict with tokens_used, tokens_max, pressure (0.0-1.0)
        """
        defaults = {
            "tokens_used": 0,
            "tokens_max": 0,
            "pressure": 0.0,
            "available": False,
        }

        try:
            # Check for context_manager on the agent
            ctx = getattr(agent, "context_manager", None)
            if ctx is not None:
                tokens_used = getattr(ctx, "tokens_used", 0) or 0
                tokens_max = getattr(ctx, "max_tokens", 0) or getattr(ctx, "tokens_max", 0) or 0

                if tokens_max > 0:
                    pressure = round(min(tokens_used / tokens_max, 1.0), 4)
                    return {
                        "tokens_used": tokens_used,
                        "tokens_max": tokens_max,
                        "pressure": pressure,
                        "available": True,
                    }

            # Check for token_budget on llm_service
            llm = getattr(agent, "llm_service", None)
            if llm is not None:
                budget = getattr(llm, "token_budget", None)
                if budget is not None:
                    used = getattr(budget, "used", 0) or 0
                    total = getattr(budget, "total", 0) or 0
                    if total > 0:
                        pressure = round(min(used / total, 1.0), 4)
                        return {
                            "tokens_used": used,
                            "tokens_max": total,
                            "pressure": pressure,
                            "available": True,
                        }

            return defaults

        except Exception as e:
            logger.warning(f"ContextPressureCalculator failed: {e}")
            return defaults


class InteractionDepthCalculator:
    """Measures quality of recent interactions.

    Analyzes conversation history to determine how substantive
    recent interactions have been. Higher depth suggests richer
    engagement and more complex task completion.
    """

    # Messages with at least this many characters are considered "substantive"
    SUBSTANTIVE_THRESHOLD = 100

    async def measure(
        self, db, agent_id: str, lookback_messages: int = 50
    ) -> Dict[str, Any]:
        """Measure interaction depth from recent messages.

        Args:
            db: AsyncDatabase instance
            agent_id: Agent identifier for scoping queries
            lookback_messages: Number of recent messages to analyze

        Returns:
            Dict with avg_length, substantive_rate, tool_usage_rate,
            depth_score (0.0-1.0)
        """
        defaults = {
            "message_count": 0,
            "avg_length": 0.0,
            "substantive_rate": 0.0,
            "tool_usage_rate": 0.0,
            "depth_score": 0.0,
            "available": False,
        }

        if not db:
            return defaults

        try:
            exists = await db.table_exists("conversation_history")
            if not exists:
                return defaults

            rows = await db.fetchall(
                """
                SELECT content, metadata FROM conversation_history
                WHERE agent_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (agent_id, lookback_messages),
            )

            if not rows:
                return {**defaults, "available": True}

            total = len(rows)
            total_length = 0
            substantive_count = 0
            tool_count = 0

            for content, metadata in rows:
                content_str = content or ""
                length = len(content_str)
                total_length += length

                if length >= self.SUBSTANTIVE_THRESHOLD:
                    substantive_count += 1

                # Check for tool usage markers in metadata
                if metadata:
                    meta_str = str(metadata).lower()
                    if "tool" in meta_str or "function" in meta_str:
                        tool_count += 1

            avg_length = total_length / total
            substantive_rate = substantive_count / total
            tool_usage_rate = tool_count / total

            # Depth score: weighted combination
            # Substantive rate is most important (50%), tool usage (30%), avg length (20%)
            # Normalize avg_length: 200+ chars = 1.0
            length_score = min(avg_length / 200.0, 1.0)
            depth_score = (
                0.5 * substantive_rate + 0.3 * tool_usage_rate + 0.2 * length_score
            )

            return {
                "message_count": total,
                "avg_length": round(avg_length, 1),
                "substantive_rate": round(substantive_rate, 4),
                "tool_usage_rate": round(tool_usage_rate, 4),
                "depth_score": round(min(depth_score, 1.0), 4),
                "available": True,
            }

        except Exception as e:
            logger.warning(f"InteractionDepthCalculator failed: {e}")
            return defaults


class SessionContinuityCalculator:
    """Measures session patterns and engagement regularity.

    Looks at conversation timestamps to identify sessions (gaps
    of >30 minutes between messages) and measures how consistent
    the agent's engagement has been.
    """

    # Gap in minutes that defines a new session
    SESSION_GAP_MINUTES = 30

    async def measure(
        self, db, agent_id: str, lookback_days: int = 30
    ) -> Dict[str, Any]:
        """Measure session continuity from conversation timestamps.

        Args:
            db: AsyncDatabase instance
            agent_id: Agent identifier for scoping queries
            lookback_days: Days to look back for session analysis

        Returns:
            Dict with total_sessions, avg_duration_minutes,
            continuity_score (0.0-1.0)
        """
        defaults = {
            "total_sessions": 0,
            "avg_duration_minutes": 0.0,
            "days_active": 0,
            "continuity_score": 0.0,
            "available": False,
        }

        if not db:
            return defaults

        try:
            exists = await db.table_exists("conversation_history")
            if not exists:
                return defaults

            cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

            rows = await db.fetchall(
                """
                SELECT created_at FROM conversation_history
                WHERE agent_id = ? AND created_at >= ?
                ORDER BY created_at ASC
                """,
                (agent_id, cutoff),
            )

            if not rows:
                return {**defaults, "available": True}

            # Parse timestamps and detect sessions
            timestamps = []
            for row in rows:
                ts_str = row[0]
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(str(ts_str))
                        timestamps.append(ts)
                    except (ValueError, TypeError):
                        continue

            if not timestamps:
                return {**defaults, "available": True}

            sessions = []
            session_start = timestamps[0]
            prev_ts = timestamps[0]

            for ts in timestamps[1:]:
                gap = (ts - prev_ts).total_seconds() / 60.0
                if gap > self.SESSION_GAP_MINUTES:
                    # Close current session, start new one
                    duration = (prev_ts - session_start).total_seconds() / 60.0
                    sessions.append(duration)
                    session_start = ts
                prev_ts = ts

            # Close the last session
            duration = (prev_ts - session_start).total_seconds() / 60.0
            sessions.append(duration)

            total_sessions = len(sessions)
            avg_duration = sum(sessions) / total_sessions if total_sessions > 0 else 0.0

            # Count unique active days
            active_days = len({ts.date() for ts in timestamps})

            # Continuity score:
            # - More sessions = good (up to ~daily for lookback_days)
            # - Longer avg duration = good (up to 30 min avg)
            # - More active days = good
            session_regularity = min(total_sessions / max(lookback_days * 0.5, 1), 1.0)
            duration_health = min(avg_duration / 30.0, 1.0) if avg_duration > 0 else 0.0
            day_coverage = active_days / lookback_days if lookback_days > 0 else 0.0

            continuity_score = (
                0.4 * day_coverage + 0.35 * session_regularity + 0.25 * duration_health
            )

            return {
                "total_sessions": total_sessions,
                "avg_duration_minutes": round(avg_duration, 1),
                "days_active": active_days,
                "continuity_score": round(min(continuity_score, 1.0), 4),
                "available": True,
            }

        except Exception as e:
            logger.warning(f"SessionContinuityCalculator failed: {e}")
            return defaults


class MemoryHealthCalculator:
    """Measures memory system health.

    Checks the overall state of the agent's memory, including
    total conversation messages and whether important memories
    are properly protected from decay.
    """

    async def measure(self, db, agent_id: str) -> Dict[str, Any]:
        """Measure memory system health.

        Args:
            db: AsyncDatabase instance
            agent_id: Agent identifier for scoping queries

        Returns:
            Dict with total_memories, pinned_memories, episodes,
            health_score (0.0-1.0)
        """
        defaults = {
            "total_memories": 0,
            "pinned_memories": 0,
            "episodes": 0,
            "health_score": 0.0,
            "available": False,
        }

        if not db:
            return defaults

        try:
            total_memories = 0
            pinned_memories = 0
            episodes = 0

            # Count conversation messages
            conv_exists = await db.table_exists("conversation_history")
            if conv_exists:
                row = await db.fetchone(
                    "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?",
                    (agent_id,),
                )
                total_memories = row[0] if row else 0

                # Count pinned/protected messages (metadata contains decay_protected)
                row = await db.fetchone(
                    """
                    SELECT COUNT(*) FROM conversation_history
                    WHERE agent_id = ? AND metadata LIKE '%decay_protected%true%'
                    """,
                    (agent_id,),
                )
                pinned_memories = row[0] if row else 0

            # Count memory episodes
            ep_exists = await db.table_exists("memory_episodes")
            if ep_exists:
                row = await db.fetchone(
                    "SELECT COUNT(*) FROM memory_episodes WHERE agent_id = ?",
                    (agent_id,),
                )
                episodes = row[0] if row else 0

            # Health score:
            # - Having memories at all is good (base score)
            # - Having episodes means consolidation is working
            # - Having pinned memories means important things are preserved
            has_memories = 1.0 if total_memories > 0 else 0.0
            episode_ratio = min(episodes / max(total_memories * 0.01, 1), 1.0) if total_memories > 0 else 0.0
            pin_ratio = min(pinned_memories / max(total_memories * 0.05, 1), 1.0) if total_memories > 0 else 0.0

            health_score = 0.4 * has_memories + 0.35 * episode_ratio + 0.25 * pin_ratio

            return {
                "total_memories": total_memories,
                "pinned_memories": pinned_memories,
                "episodes": episodes,
                "health_score": round(min(health_score, 1.0), 4),
                "available": True,
            }

        except Exception as e:
            logger.warning(f"MemoryHealthCalculator failed: {e}")
            return defaults
