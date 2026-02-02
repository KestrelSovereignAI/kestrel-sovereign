"""
Result formatting utilities for reflection feature.

Handles formatting of reflection results, summaries, and response structures.
"""

from typing import Dict, Any, List
from .models import ReflectionResult, LayerResult, HealthCheck


class ReflectionResultFormatter:
    """Formats reflection results for API responses."""

    @staticmethod
    def format_reflection_result(result: ReflectionResult) -> Dict[str, Any]:
        """
        Format a ReflectionResult for API response.

        Args:
            result: The reflection result to format

        Returns:
            Formatted dictionary suitable for API response
        """
        data = result.to_dict()

        # Add summary statistics
        summary = ReflectionResultFormatter._build_summary(result)
        data["summary"] = summary

        data["success"] = result.error is None
        return data

    @staticmethod
    def _build_summary(result: ReflectionResult) -> Dict[str, Any]:
        """
        Build summary statistics for a reflection result.

        Args:
            result: The reflection result

        Returns:
            Summary statistics dictionary
        """
        total_passed = 0
        total_failed = 0
        critical_count = 0

        for layer in [result.arms, result.memory, result.mind]:
            if layer:
                total_passed += layer.passed
                total_failed += layer.failed
                if layer.has_critical:
                    critical_count += 1

        return {
            "layers_completed": result.layers_completed,
            "stopped_early": result.stopped_at_layer is not None,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "critical_failures": critical_count,
            "action_count": len(result.actions),
        }

    @staticmethod
    def format_layer_result(layer: LayerResult) -> Dict[str, Any]:
        """
        Format a layer result with additional computed fields.

        Args:
            layer: The layer result to format

        Returns:
            Formatted layer result dictionary
        """
        data = layer.to_dict()

        # Add computed fields
        data["health_score"] = ReflectionResultFormatter._calculate_health_score(layer.checks)
        data["has_critical"] = layer.has_critical
        data["passed"] = layer.passed
        data["failed"] = layer.failed

        return data

    @staticmethod
    def _calculate_health_score(checks: List[HealthCheck]) -> float:
        """
        Calculate a health score (0.0 to 1.0) based on check results.

        Args:
            checks: List of health checks

        Returns:
            Health score between 0.0 and 1.0
        """
        if not checks:
            return 1.0

        total_checks = len(checks)
        passed_checks = sum(1 for check in checks if check.status.value == "PASS")

        # Weight critical failures more heavily
        critical_failures = sum(
            1 for check in checks
            if check.status.value == "FAIL" and check.severity.value == "CRITICAL"
        )

        # Base score from pass rate
        base_score = passed_checks / total_checks

        # Penalize critical failures heavily
        critical_penalty = (critical_failures / total_checks) * 0.5

        return max(0.0, base_score - critical_penalty)

    @staticmethod
    def format_training_result(
        iterations_completed: int,
        health_trend: List[Dict[str, Any]],
        tickets_created: List[str],
    ) -> Dict[str, Any]:
        """
        Format training cycle results.

        Args:
            iterations_completed: Number of iterations completed
            health_trend: List of health data per iteration
            tickets_created: List of ticket URLs created

        Returns:
            Formatted training result dictionary
        """
        final_health = health_trend[-1] if health_trend else {"healthy": False}

        return {
            "success": True,
            "iterations_completed": iterations_completed,
            "final_health": final_health,
            "health_trend": health_trend,
            "tickets_created": tickets_created,
            "improvement_trend": ReflectionResultFormatter._analyze_health_trend(health_trend),
            "message": ReflectionResultFormatter._format_training_message(
                iterations_completed, final_health
            ),
        }

    @staticmethod
    def _analyze_health_trend(health_trend: List[Dict[str, Any]]) -> str:
        """
        Analyze the health trend across iterations.

        Args:
            health_trend: List of health data per iteration

        Returns:
            Trend analysis string
        """
        if len(health_trend) < 2:
            return "insufficient_data"

        first = health_trend[0]
        last = health_trend[-1]

        first_score = first.get("passed", 0) - first.get("critical", 0) * 2
        last_score = last.get("passed", 0) - last.get("critical", 0) * 2

        if last_score > first_score:
            return "improving"
        elif last_score < first_score:
            return "declining"
        else:
            return "stable"

    @staticmethod
    def _format_training_message(
        iterations: int,
        final_health: Dict[str, Any]
    ) -> str:
        """
        Format a summary message for training completion.

        Args:
            iterations: Number of iterations completed
            final_health: Final health status

        Returns:
            Formatted message string
        """
        is_healthy = final_health.get("healthy", False)
        status = "HEALTHY" if is_healthy else "NEEDS ATTENTION"

        message = f"Training complete: {iterations} iterations, {status}"

        critical = final_health.get("critical", 0)
        warn = final_health.get("warn", 0)

        if critical > 0:
            message += f" ({critical} critical issues)"
        elif warn > 0:
            message += f" ({warn} warnings)"

        return message

    @staticmethod
    def format_insights_response(insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format a list of insights for API response.

        Args:
            insights: List of insight dictionaries

        Returns:
            Formatted insights response
        """
        return {
            "success": True,
            "count": len(insights),
            "insights": insights,
            "categories": ReflectionResultFormatter._categorize_insights(insights),
        }

    @staticmethod
    def _categorize_insights(insights: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Categorize insights by type.

        Args:
            insights: List of insight dictionaries

        Returns:
            Dictionary mapping insight types to counts
        """
        categories = {}
        for insight in insights:
            insight_type = insight.get("type", "unknown")
            categories[insight_type] = categories.get(insight_type, 0) + 1

        return categories

    @staticmethod
    def format_behavior_rules_response(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format behavior rules for API response.

        Args:
            rules: List of behavior rule dictionaries

        Returns:
            Formatted rules response
        """
        return {
            "success": True,
            "count": len(rules),
            "rules": rules,
            "by_type": ReflectionResultFormatter._categorize_rules(rules),
        }

    @staticmethod
    def _categorize_rules(rules: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Categorize rules by change type.

        Args:
            rules: List of rule dictionaries

        Returns:
            Dictionary mapping change types to counts
        """
        categories = {}
        for rule in rules:
            change_type = rule.get("change_type", "unknown")
            categories[change_type] = categories.get(change_type, 0) + 1

        return categories