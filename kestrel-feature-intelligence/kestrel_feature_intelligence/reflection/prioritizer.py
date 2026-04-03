"""Layer 4: Action Prioritizer - Convert checks to prioritized actions."""
from typing import List, Optional

from .models import (
    ActionItem,
    LayerResult,
    HealthCheck,
    CheckStatus,
    Severity,
)


class ActionPrioritizer:
    """Convert layer results into prioritized action list."""

    def prioritize(
        self,
        arms: Optional[LayerResult],
        memory: Optional[LayerResult],
        mind: Optional[LayerResult],
    ) -> List[ActionItem]:
        """Generate prioritized actions from all layer results."""
        actions = []

        # Collect all layers
        layers = [l for l in [arms, memory, mind] if l is not None]

        # P0: Critical failures (Arms and Memory are most critical)
        for layer in layers:
            for check in layer.checks:
                if check.status == CheckStatus.FAIL and check.severity == Severity.CRITICAL:
                    actions.append(self._check_to_action(check, Severity.CRITICAL))

        # P1: High severity failures
        for layer in layers:
            for check in layer.checks:
                if check.status == CheckStatus.FAIL and check.severity == Severity.HIGH:
                    actions.append(self._check_to_action(check, Severity.HIGH))

        # P2: Medium warnings
        for layer in layers:
            for check in layer.checks:
                if check.status == CheckStatus.WARN and check.severity == Severity.MEDIUM:
                    actions.append(self._check_to_action(check, Severity.MEDIUM))

        # P3: Actionable insights from mind layer
        if mind:
            for check in mind.checks:
                if check.id == "mind.patterns" and check.details.get("insights"):
                    for insight in check.details["insights"]:
                        if insight.get("actionable"):
                            actions.append(ActionItem(
                                id=f"action.insight.{insight['id'][:8]}",
                                priority=Severity.LOW,
                                title=insight["title"],
                                description=insight["description"],
                                source_insight=insight["id"],
                                fix_description=insight.get("suggested_action", ""),
                                effort_estimate="varies",
                            ))

        # Sort by priority
        priority_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }
        actions.sort(key=lambda a: priority_order.get(a.priority, 99))

        return actions

    def _check_to_action(self, check: HealthCheck, priority: Severity) -> ActionItem:
        """Convert a failed check to an action item."""
        prefix = "FIX" if check.status == CheckStatus.FAIL else "IMPROVE"
        return ActionItem(
            id=f"action.{check.id}",
            priority=priority,
            title=f"{prefix}: {check.name}",
            description=check.message,
            source_check=check.id,
            file_path=check.file_path,
            line_number=check.line_number,
            fix_description=check.suggested_fix or "Investigate and fix",
            effort_estimate="immediate" if priority == Severity.CRITICAL else "moderate",
        )
