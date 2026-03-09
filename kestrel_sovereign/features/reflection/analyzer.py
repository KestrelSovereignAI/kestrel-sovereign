"""
Interaction Analyzer for agent self-reflection.

Uses LLM to analyze past interactions and generate insights about:
- What worked well (successes)
- What could be improved (failures)
- Recurring patterns
- Actionable improvements
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from .models import Insight, InsightType

logger = logging.getLogger(__name__)


# Prompt for the LLM to analyze interactions
ANALYSIS_PROMPT_TEMPLATE = """You are performing self-reflection on your past interactions as a Kestrel agent.

Your goal is to honestly assess your performance and identify ways to improve.

## Recent Conversations
{interaction_summaries}

## Memory Episodes (Consolidated Memories)
{episode_summaries}

## Analysis Depth: {depth}
{depth_instructions}

## Questions to Consider
1. What did users seem satisfied with? What responses got positive engagement?
2. Where did I struggle or make mistakes? What went poorly?
3. Are there patterns in what users ask for? Recurring themes or needs?
4. What would make me more helpful? Specific improvements I could make?
5. Did I respect privacy and constitutional bounds? Any violations?
6. Were there times I was uncertain but didn't express it? Times I should have asked clarifying questions?

## Output Format
Respond with a JSON object containing your insights:

```json
{{
  "insights": [
    {{
      "type": "pattern|improvement|success|failure|anomaly",
      "title": "Brief descriptive title",
      "description": "Detailed explanation of the insight",
      "evidence": ["message_id_1", "message_id_2"],
      "confidence": 0.8,
      "actionable": true,
      "suggested_action": "Specific action to take (if actionable)"
    }}
  ]
}}
```

## Guidelines
- Be honest and self-critical. The goal is genuine improvement.
- Each insight should be specific and grounded in evidence.
- Only mark insights as "actionable" if there's a clear action to take.
- Confidence should reflect how certain you are (0.0 = guess, 1.0 = certain).
- For improvements, suggest concrete changes, not vague aspirations.

Respond ONLY with the JSON object, no other text."""


DEPTH_INSTRUCTIONS = {
    "shallow": "Quick analysis - focus on obvious patterns and issues. 2-4 insights max.",
    "normal": "Standard analysis - thorough review of interactions. 4-8 insights expected.",
    "deep": "Deep analysis - examine subtle patterns, emotional dynamics, and implicit feedback. 6-12 insights expected.",
}


class InteractionAnalyzer:
    """
    Analyzes past interactions to generate insights.

    Uses the LLM to perform self-reflection on conversation history
    and memory episodes, producing structured insights.
    """

    def __init__(
        self,
        llm_service,
        conversation_store,
        episode_store=None,
        agent_id: str = "",
    ):
        """
        Initialize the analyzer.

        Args:
            llm_service: LLMService instance for generation
            conversation_store: Store for conversation history
            episode_store: Store for memory episodes (optional)
            agent_id: Agent identifier for filtering
        """
        self.llm = llm_service
        self.conversations = conversation_store
        self.episodes = episode_store
        self.agent_id = agent_id

    async def analyze(
        self,
        scope: str = "today",
        depth: str = "normal",
        max_interactions: int = 50,
        max_episodes: int = 10,
    ) -> List[Insight]:
        """
        Analyze interactions and generate insights.

        Args:
            scope: Time range - 'session', 'today', 'week', 'month', 'all'
            depth: Analysis depth - 'shallow', 'normal', 'deep'
            max_interactions: Maximum interactions to analyze
            max_episodes: Maximum episodes to include

        Returns:
            List of Insight objects
        """
        logger.info(f"Starting interaction analysis: scope={scope}, depth={depth}")

        # Load interactions based on scope
        interactions = await self._load_interactions(scope, max_interactions)
        if not interactions:
            logger.info("No interactions to analyze")
            return []

        # Load episodes if available
        episodes = []
        if self.episodes:
            episodes = await self._load_episodes(scope, max_episodes)

        # Build analysis prompt
        prompt = self._build_prompt(interactions, episodes, depth)

        # Get LLM analysis
        try:
            response = await self.llm.generate(
                system_prompt="You are an AI assistant performing self-reflection analysis. Respond only with valid JSON.",
                user_prompt=prompt,
                force_local_only=False,  # Use best available model
            )

            # Handle different response types
            response_text = response if isinstance(response, str) else response.content

            # Parse insights from response
            insights = self._parse_insights(response_text)
            logger.info(f"Generated {len(insights)} insights from analysis")
            return insights

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return []

    async def _load_interactions(
        self,
        scope: str,
        max_count: int,
    ) -> List[Dict[str, Any]]:
        """Load interactions based on time scope."""
        # Calculate time range
        now = datetime.utcnow()

        if scope == "session":
            # Last 30 minutes
            since = now - timedelta(minutes=30)
        elif scope == "today":
            # Today (from midnight)
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif scope == "week":
            # Last 7 days
            since = now - timedelta(days=7)
        elif scope == "month":
            # Last 30 days
            since = now - timedelta(days=30)
        else:
            # All time - just use a large lookback
            since = now - timedelta(days=365)

        # Fetch from conversation store
        try:
            if hasattr(self.conversations, 'get_history_since'):
                messages = await self.conversations.get_history_since(
                    since,
                    limit=max_count,
                )
            elif hasattr(self.conversations, 'get_conversation_history'):
                # Fallback to getting recent history
                # Note: AsyncConversationStore is already scoped by agent_id in constructor
                messages = await self.conversations.get_conversation_history(
                    limit=max_count,
                )
            else:
                logger.warning("Conversation store has no history retrieval method")
                return []

            # Convert to list of dicts if needed
            interactions = []
            for msg in messages:
                if isinstance(msg, dict):
                    interactions.append(msg)
                elif hasattr(msg, 'to_dict'):
                    interactions.append(msg.to_dict())
                else:
                    interactions.append({
                        "role": getattr(msg, 'role', 'unknown'),
                        "content": getattr(msg, 'content', str(msg)),
                        "metadata": getattr(msg, 'metadata', {}),
                    })

            return interactions

        except Exception as e:
            logger.error(f"Failed to load interactions: {e}")
            return []

    async def _load_episodes(
        self,
        scope: str,
        max_count: int,
    ) -> List[Dict[str, Any]]:
        """Load memory episodes based on time scope."""
        if not self.episodes:
            return []

        try:
            if hasattr(self.episodes, 'get_recent_episodes'):
                episodes = await self.episodes.get_recent_episodes(
                    limit=max_count,
                    agent_id=self.agent_id,
                )
            elif hasattr(self.episodes, 'list_episodes'):
                episodes = await self.episodes.list_episodes(limit=max_count)
            else:
                return []

            # Convert to list of dicts if needed
            result = []
            for ep in episodes:
                if isinstance(ep, dict):
                    result.append(ep)
                elif hasattr(ep, 'to_dict'):
                    result.append(ep.to_dict())
                else:
                    result.append({
                        "title": getattr(ep, 'title', 'Unknown'),
                        "summary": getattr(ep, 'summary', ''),
                        "emotional_arc": getattr(ep, 'emotional_arc', ''),
                    })

            return result

        except Exception as e:
            logger.error(f"Failed to load episodes: {e}")
            return []

    def _build_prompt(
        self,
        interactions: List[Dict[str, Any]],
        episodes: List[Dict[str, Any]],
        depth: str,
    ) -> str:
        """Build the analysis prompt from interactions and episodes."""
        # Summarize interactions
        interaction_lines = []
        for i, msg in enumerate(interactions[-50:], 1):  # Limit to last 50
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."

            # Include metadata hints if available
            metadata = msg.get("metadata", {})
            emotional = ""
            if metadata.get("emotional_valence"):
                valence = metadata["emotional_valence"]
                emotional = f" [emotion: {'positive' if valence > 0 else 'negative' if valence < 0 else 'neutral'}]"

            msg_id = msg.get("id", f"msg_{i}")
            interaction_lines.append(f"[{msg_id}] {role.upper()}{emotional}: {content}")

        interaction_summary = "\n".join(interaction_lines) if interaction_lines else "(No recent interactions)"

        # Summarize episodes
        episode_lines = []
        for ep in episodes:
            title = ep.get("title", "Untitled")
            summary = ep.get("summary", "")
            arc = ep.get("emotional_arc", "")
            episode_lines.append(f"- {title}: {summary[:200]}... (arc: {arc})")

        episode_summary = "\n".join(episode_lines) if episode_lines else "(No memory episodes)"

        # Get depth instructions
        depth_inst = DEPTH_INSTRUCTIONS.get(depth, DEPTH_INSTRUCTIONS["normal"])

        return ANALYSIS_PROMPT_TEMPLATE.format(
            interaction_summaries=interaction_summary,
            episode_summaries=episode_summary,
            depth=depth,
            depth_instructions=depth_inst,
        )

    def _parse_insights(self, response_text: str) -> List[Insight]:
        """Parse LLM response into Insight objects.

        Handles common local-model quirks:
        - Markdown code fences around JSON
        - Thinking/reasoning text before/after JSON
        - Multiple JSON objects in response
        """
        import uuid
        import re

        text = response_text.strip()

        # Strategy 1: Extract JSON from markdown code fence
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # Strategy 2: Find the first complete JSON object with { ... }
        if not text.startswith("{"):
            brace_start = text.find("{")
            if brace_start >= 0:
                text = text[brace_start:]

        # Strategy 3: Trim trailing non-JSON content after the closing brace
        if text.startswith("{"):
            depth = 0
            in_string = False
            escape_next = False
            end_pos = -1
            for i, ch in enumerate(text):
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
            if end_pos > 0:
                text = text[:end_pos]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text (first 500 chars): {response_text[:500]}")
            return []

        insights_data = data.get("insights", [])
        if not insights_data:
            logger.info(f"JSON parsed but no insights array (keys: {list(data.keys())})")

        insights = []
        for item in insights_data:
            try:
                insight = Insight(
                    id=str(uuid.uuid4()),
                    type=InsightType(item.get("type", "pattern")),
                    title=item.get("title", "Untitled Insight"),
                    description=item.get("description", ""),
                    evidence=item.get("evidence", []),
                    confidence=float(item.get("confidence", 0.5)),
                    actionable=bool(item.get("actionable", False)),
                    suggested_action=item.get("suggested_action"),
                )
                insights.append(insight)
            except Exception as e:
                logger.warning(f"Failed to parse insight: {e}")
                continue

        return insights

    async def analyze_error(
        self,
        error_context: Dict[str, Any],
    ) -> List[Insight]:
        """
        Analyze a specific error for learning.

        Called when the agent encounters an error to learn from it.

        Args:
            error_context: Dict with error details (type, message, traceback, etc.)

        Returns:
            List of Insight objects about the error
        """
        error_prompt = f"""Analyze this error that occurred during agent operation:

## Error Details
- Type: {error_context.get('type', 'Unknown')}
- Message: {error_context.get('message', 'No message')}
- Context: {error_context.get('context', 'No context')}

## What Happened
{error_context.get('description', 'No description available')}

## Analysis Questions
1. What was the root cause?
2. How could this be prevented?
3. What should I do differently next time?

Respond with JSON containing insights about this error.
"""

        try:
            response = await self.llm.generate(
                system_prompt="Analyze this error and provide learning insights as JSON.",
                user_prompt=error_prompt,
            )

            response_text = response if isinstance(response, str) else response.content
            return self._parse_insights(response_text)

        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            return []
