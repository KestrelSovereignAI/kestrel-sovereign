"""Layer 3: Mind - Cognitive checks."""
from typing import List, Optional, TYPE_CHECKING

from .base import HealthChecker
from ..models import HealthCheck, ReflectionLayer, CheckStatus, Severity, InsightType

if TYPE_CHECKING:
    from ..analyzer import InteractionAnalyzer


class MindChecker(HealthChecker):
    """Layer 3: Is my reasoning producing good outputs?"""

    def __init__(self, agent, analyzer: Optional["InteractionAnalyzer"] = None):
        super().__init__(agent)
        self.analyzer = analyzer

    async def run_all(self, depth: str = "normal") -> List[HealthCheck]:
        checks = [
            await self.check_response_coherence(),
        ]

        # Include interaction analysis if analyzer available
        if self.analyzer:
            checks.append(await self.check_interaction_patterns(depth))

        return checks

    async def check_response_coherence(self) -> HealthCheck:
        """Are recent responses coherent?"""
        check = HealthCheck(
            id="mind.coherence",
            layer=ReflectionLayer.MIND,
            name="Response Coherence",
            description="Evaluate quality of recent responses",
            status=CheckStatus.SKIP,
        )

        try:
            storage = getattr(self.agent, 'storage', None)
            conv_store = getattr(storage, 'conversation', None) if storage else None

            if not conv_store:
                check.status = CheckStatus.SKIP
                check.message = "Conversation store not available"
                return check

            history = await conv_store.get_conversation_history(limit=10)
            assistant_msgs = [m for m in history if m.get("role") == "assistant"]

            if not assistant_msgs:
                check.status = CheckStatus.SKIP
                check.message = "No recent responses to evaluate"
                return check

            # Check for basic issues
            issues = []
            for msg in assistant_msgs[-3:]:
                content = msg.get("content", "")
                if len(content) < 10:
                    issues.append("Empty/short response")
                if content.startswith("gAAAAA"):
                    issues.append("Encrypted content in response")

            if issues:
                check.status = CheckStatus.WARN
                check.severity = Severity.MEDIUM
                check.message = f"Issues found: {', '.join(set(issues))}"
            else:
                check.status = CheckStatus.PASS
                check.message = f"Evaluated {len(assistant_msgs)} responses"

        except Exception as e:
            check.status = CheckStatus.SKIP
            check.message = f"Coherence check error: {e}"

        return check

    async def check_interaction_patterns(self, depth: str) -> HealthCheck:
        """Analyze interaction patterns using existing analyzer."""
        check = HealthCheck(
            id="mind.patterns",
            layer=ReflectionLayer.MIND,
            name="Interaction Analysis",
            description="Patterns from recent interactions",
            status=CheckStatus.SKIP,
        )

        if not self.analyzer:
            check.status = CheckStatus.SKIP
            check.message = "Analyzer not available"
            return check

        try:
            insights = await self.analyzer.analyze(scope="today", depth=depth)

            if not insights:
                check.status = CheckStatus.PASS
                check.message = "No notable patterns found"
                return check

            failures = [i for i in insights if i.type == InsightType.FAILURE]
            improvements = [i for i in insights if i.type == InsightType.IMPROVEMENT]
            actionable = [i for i in insights if i.actionable]

            if failures:
                check.status = CheckStatus.WARN
                check.severity = Severity.MEDIUM
                check.message = f"{len(failures)} failures, {len(improvements)} improvements"
            else:
                check.status = CheckStatus.PASS
                check.message = f"{len(insights)} insights ({len(actionable)} actionable)"

            check.details = {"insights": [i.to_dict() for i in insights]}

        except Exception as e:
            check.status = CheckStatus.WARN
            check.message = f"Analysis error: {e}"

        return check
