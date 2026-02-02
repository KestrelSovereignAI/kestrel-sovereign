#!/usr/bin/env python3
"""
Graceful Degradation: Handle capability mismatches between substrates.

This module provides strategies for when an agent migrates to a substrate
with reduced capabilities. It includes:
- Capability gap assessment with severity levels
- Compensation strategies for missing features
- User-facing limitation disclosure
- Runtime adaptation suggestions

Phase 5 of Issue #23: Substrate-Independent Agent Portability.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .substrate_adapter import Capability, CapabilityMap, CapabilityGap
from .identity_package import SubstrateType, PersonalityFingerprint

logger = logging.getLogger(__name__)


class SeverityLevel(str, Enum):
    """Severity of capability loss."""
    CRITICAL = "critical"   # Core functionality impaired
    HIGH = "high"           # Significant feature loss
    MEDIUM = "medium"       # Notable degradation
    LOW = "low"             # Minor inconvenience
    INFO = "info"           # Informational only


@dataclass
class CapabilityLoss:
    """Detailed information about a lost or degraded capability."""
    capability: Capability
    severity: SeverityLevel
    impact: str                       # What the user will experience
    workaround: str                   # How to work around it
    user_actions: List[str] = field(default_factory=list)  # User can do
    agent_adaptations: List[str] = field(default_factory=list)  # Agent will do
    quality_loss: float = 0.0         # 0.0-1.0, how much quality is lost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability.value,
            "severity": self.severity.value,
            "impact": self.impact,
            "workaround": self.workaround,
            "user_actions": self.user_actions,
            "agent_adaptations": self.agent_adaptations,
            "quality_loss": self.quality_loss,
        }


@dataclass
class DegradationReport:
    """Complete report on capability degradation."""
    source_substrate: str
    target_substrate: str
    overall_severity: SeverityLevel
    capability_losses: List[CapabilityLoss] = field(default_factory=list)
    context_reduction: Optional[int] = None  # Tokens lost
    estimated_functionality: float = 1.0     # 0.0-1.0
    recommendations: List[str] = field(default_factory=list)
    can_proceed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_substrate": self.source_substrate,
            "target_substrate": self.target_substrate,
            "overall_severity": self.overall_severity.value,
            "capability_losses": [l.to_dict() for l in self.capability_losses],
            "context_reduction": self.context_reduction,
            "estimated_functionality": self.estimated_functionality,
            "recommendations": self.recommendations,
            "can_proceed": self.can_proceed,
        }


# Capability impact definitions
CAPABILITY_IMPACTS = {
    Capability.TOOL_USE: {
        "severity": SeverityLevel.HIGH,
        "impact": "I won't be able to execute functions or use tools automatically. "
                  "Instead, I'll describe what I would do and you'll need to perform the actions.",
        "workaround": "I'll provide structured instructions for actions that need to be taken, "
                      "and you can execute them manually.",
        "user_actions": [
            "Execute commands I describe manually",
            "Copy code snippets I provide",
            "Run scripts I generate",
        ],
        "agent_adaptations": [
            "Provide clear, step-by-step instructions",
            "Format outputs for easy copy-paste",
            "Describe expected results for verification",
        ],
    },
    Capability.VISION: {
        "severity": SeverityLevel.MEDIUM,
        "impact": "I won't be able to analyze images directly. "
                  "I'll need you to describe what you see in images.",
        "workaround": "Please describe images in detail, including layout, text, colors, "
                      "and any relevant visual elements.",
        "user_actions": [
            "Describe images in detail",
            "Transcribe text from images",
            "Describe diagrams and charts",
        ],
        "agent_adaptations": [
            "Ask clarifying questions about visual content",
            "Request specific details when needed",
            "Provide templates for image descriptions",
        ],
    },
    Capability.LONG_CONTEXT: {
        "severity": SeverityLevel.MEDIUM,
        "impact": "I have a shorter memory for this conversation. "
                  "I may forget earlier parts of long discussions.",
        "workaround": "For long conversations, periodically summarize key points. "
                      "I'll do my best to maintain context but may need reminders.",
        "user_actions": [
            "Remind me of earlier context when needed",
            "Provide summaries for long documents",
            "Break large tasks into smaller chunks",
        ],
        "agent_adaptations": [
            "Summarize conversation periodically",
            "Ask for context clarification when uncertain",
            "Focus on most recent and relevant information",
        ],
    },
    Capability.STREAMING: {
        "severity": SeverityLevel.LOW,
        "impact": "Responses will appear all at once instead of streaming gradually. "
                  "This may feel slower but the content quality is the same.",
        "workaround": "No workaround needed - just expect a brief wait before seeing the full response.",
        "user_actions": [],
        "agent_adaptations": [],
    },
    Capability.STRUCTURED_OUTPUT: {
        "severity": SeverityLevel.LOW,
        "impact": "JSON and structured outputs may occasionally have minor formatting issues. "
                  "I'll be extra careful with structure.",
        "workaround": "I'll include validation hints and use clear formatting to ensure "
                      "structured data is correctly formed.",
        "user_actions": [
            "Validate JSON outputs if using programmatically",
        ],
        "agent_adaptations": [
            "Use explicit formatting for structured data",
            "Include schema hints in responses",
            "Validate structure before responding",
        ],
    },
    Capability.CODE_EXECUTION: {
        "severity": SeverityLevel.MEDIUM,
        "impact": "I can't run code directly to test it. "
                  "I'll provide code but you'll need to run it yourself.",
        "workaround": "Run code I provide in your own environment. "
                      "I'll include test cases and expected outputs.",
        "user_actions": [
            "Run provided code locally",
            "Report any errors or unexpected outputs",
        ],
        "agent_adaptations": [
            "Include test cases with code",
            "Provide expected outputs",
            "Add error handling guidance",
        ],
    },
    Capability.EMBEDDINGS: {
        "severity": SeverityLevel.LOW,
        "impact": "Semantic search may be less accurate. "
                  "I'll rely more on keyword matching.",
        "workaround": "Use specific keywords when searching. "
                      "Be more explicit about what you're looking for.",
        "user_actions": [
            "Use specific keywords",
            "Provide more context for searches",
        ],
        "agent_adaptations": [
            "Use keyword-based search fallback",
            "Request clarification for ambiguous queries",
        ],
    },
    Capability.MULTI_TURN: {
        "severity": SeverityLevel.MEDIUM,
        "impact": "Multi-turn conversations may be less coherent. "
                  "I might lose track of conversation threads.",
        "workaround": "Reference earlier points explicitly. "
                      "I'll do my best to maintain coherence.",
        "user_actions": [
            "Reference earlier messages explicitly",
            "Provide context when switching topics",
        ],
        "agent_adaptations": [
            "Confirm understanding of context",
            "Summarize conversation state periodically",
        ],
    },
}


class GracefulDegradationHandler:
    """
    Handles capability degradation when migrating between substrates.

    Analyzes capability gaps, generates user-friendly reports,
    and provides compensation strategies.
    """

    def __init__(
        self,
        source_capabilities: CapabilityMap,
        target_capabilities: CapabilityMap,
    ):
        """
        Initialize the handler.

        Args:
            source_capabilities: Capabilities of source substrate
            target_capabilities: Capabilities of target substrate
        """
        self.source = source_capabilities
        self.target = target_capabilities

    def assess_degradation(self) -> DegradationReport:
        """
        Assess the full impact of capability degradation.

        Returns:
            Complete DegradationReport with all losses and recommendations
        """
        losses = []
        max_severity = SeverityLevel.INFO
        functionality = 1.0

        # Check each capability
        for cap in Capability:
            source_has = self.source.has(cap)
            target_has = self.target.has(cap)

            if source_has and not target_has:
                # Complete loss
                loss = self._create_loss(cap, complete=True)
                losses.append(loss)
                functionality -= self._capability_weight(cap)
                if loss.severity.value < max_severity.value:
                    max_severity = loss.severity

            elif source_has and target_has:
                # Check for quality degradation
                source_quality = self.source.quality(cap)
                target_quality = self.target.quality(cap)

                if source_quality - target_quality > 0.15:  # >15% quality loss
                    loss = self._create_loss(
                        cap,
                        complete=False,
                        quality_loss=source_quality - target_quality
                    )
                    losses.append(loss)
                    functionality -= self._capability_weight(cap) * (source_quality - target_quality)

        # Check context reduction
        context_reduction = None
        if self.source.context_limit > self.target.context_limit:
            context_reduction = self.source.context_limit - self.target.context_limit
            if context_reduction > 50000:  # Significant reduction
                losses.append(CapabilityLoss(
                    capability=Capability.LONG_CONTEXT,
                    severity=SeverityLevel.MEDIUM,
                    impact=f"Context window reduced by {context_reduction:,} tokens. "
                           f"Now limited to {self.target.context_limit:,} tokens.",
                    workaround="Break long conversations into sessions. "
                              "Summarize important context periodically.",
                    quality_loss=min(0.5, context_reduction / self.source.context_limit),
                ))

        # Generate recommendations
        recommendations = self._generate_recommendations(losses)

        # Determine if we should proceed
        can_proceed = max_severity not in [SeverityLevel.CRITICAL]

        return DegradationReport(
            source_substrate=self.source.substrate,
            target_substrate=self.target.substrate,
            overall_severity=max_severity,
            capability_losses=losses,
            context_reduction=context_reduction,
            estimated_functionality=max(0.0, min(1.0, functionality)),
            recommendations=recommendations,
            can_proceed=can_proceed,
        )

    def _create_loss(
        self,
        cap: Capability,
        complete: bool = True,
        quality_loss: float = 1.0
    ) -> CapabilityLoss:
        """Create a CapabilityLoss from the capability impacts."""
        impact_data = CAPABILITY_IMPACTS.get(cap, {})

        severity = impact_data.get("severity", SeverityLevel.MEDIUM)
        if not complete:
            # Reduce severity for partial loss
            if severity == SeverityLevel.HIGH:
                severity = SeverityLevel.MEDIUM
            elif severity == SeverityLevel.MEDIUM:
                severity = SeverityLevel.LOW

        return CapabilityLoss(
            capability=cap,
            severity=severity,
            impact=impact_data.get("impact", f"Capability {cap.value} is not available."),
            workaround=impact_data.get("workaround", "No specific workaround available."),
            user_actions=impact_data.get("user_actions", []),
            agent_adaptations=impact_data.get("agent_adaptations", []),
            quality_loss=quality_loss,
        )

    def _capability_weight(self, cap: Capability) -> float:
        """Get the weight of a capability for functionality scoring."""
        weights = {
            Capability.TOOL_USE: 0.25,
            Capability.VISION: 0.15,
            Capability.LONG_CONTEXT: 0.15,
            Capability.STREAMING: 0.05,
            Capability.STRUCTURED_OUTPUT: 0.10,
            Capability.CODE_EXECUTION: 0.10,
            Capability.EMBEDDINGS: 0.05,
            Capability.MULTI_TURN: 0.15,
        }
        return weights.get(cap, 0.1)

    def _generate_recommendations(self, losses: List[CapabilityLoss]) -> List[str]:
        """Generate recommendations based on losses."""
        recommendations = []

        if not losses:
            recommendations.append("No significant capability changes. Migration should be seamless.")
            return recommendations

        # Count by severity
        severities = {s: 0 for s in SeverityLevel}
        for loss in losses:
            severities[loss.severity] += 1

        if severities[SeverityLevel.CRITICAL] > 0:
            recommendations.append(
                "⚠️ CRITICAL: Some core capabilities are missing. Consider using a more capable substrate."
            )

        if severities[SeverityLevel.HIGH] > 0:
            recommendations.append(
                "Some significant features are unavailable. Review the impact list carefully."
            )

        # Specific recommendations
        cap_types = {loss.capability for loss in losses}

        if Capability.TOOL_USE in cap_types:
            recommendations.append(
                "Without tool use, be prepared to execute actions manually. "
                "The agent will provide clear instructions."
            )

        if Capability.VISION in cap_types:
            recommendations.append(
                "Prepare to describe images textually. Include details about layout, "
                "text, colors, and key visual elements."
            )

        if Capability.LONG_CONTEXT in cap_types:
            recommendations.append(
                "For long sessions, periodically summarize key points. "
                "Consider breaking work into smaller sessions."
            )

        recommendations.append(
            "The agent will adapt its behavior to work within these constraints."
        )

        return recommendations

    def generate_disclosure(
        self,
        report: Optional[DegradationReport] = None,
        verbose: bool = True
    ) -> str:
        """
        Generate user-facing disclosure of limitations.

        Args:
            report: Optional pre-computed report (generated if not provided)
            verbose: Whether to include full details

        Returns:
            Formatted disclosure string
        """
        if report is None:
            report = self.assess_degradation()

        lines = [
            "# Substrate Migration Notice",
            "",
            f"**From**: {report.source_substrate}",
            f"**To**: {report.target_substrate}",
            "",
        ]

        if not report.capability_losses:
            lines.append("✅ No significant capability changes detected. "
                        "I should function the same as before.")
            return "\n".join(lines)

        # Severity indicator
        severity_icons = {
            SeverityLevel.CRITICAL: "🔴",
            SeverityLevel.HIGH: "🟠",
            SeverityLevel.MEDIUM: "🟡",
            SeverityLevel.LOW: "🟢",
            SeverityLevel.INFO: "ℹ️",
        }

        lines.append(f"**Overall Impact**: {severity_icons[report.overall_severity]} "
                    f"{report.overall_severity.value.upper()}")
        lines.append(f"**Estimated Functionality**: {report.estimated_functionality:.0%}")
        lines.append("")

        # List capability changes
        if verbose:
            lines.append("## Capability Changes")
            lines.append("")

            for loss in report.capability_losses:
                icon = severity_icons[loss.severity]
                lines.append(f"### {icon} {loss.capability.value.replace('_', ' ').title()}")
                lines.append("")
                lines.append(f"**Impact**: {loss.impact}")
                lines.append("")
                lines.append(f"**Workaround**: {loss.workaround}")

                if loss.user_actions:
                    lines.append("")
                    lines.append("**What you can do**:")
                    for action in loss.user_actions:
                        lines.append(f"- {action}")

                if loss.agent_adaptations:
                    lines.append("")
                    lines.append("**What I'll do**:")
                    for adapt in loss.agent_adaptations:
                        lines.append(f"- {adapt}")

                lines.append("")

        # Context reduction
        if report.context_reduction:
            lines.append(f"**Context Window**: Reduced by {report.context_reduction:,} tokens")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        for rec in report.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)

    def generate_adaptation_prompt(
        self,
        report: Optional[DegradationReport] = None,
        personality: Optional[PersonalityFingerprint] = None
    ) -> str:
        """
        Generate a system prompt addition for adapting to limitations.

        Args:
            report: Optional pre-computed report
            personality: Optional personality for style preservation

        Returns:
            System prompt section for capability adaptation
        """
        if report is None:
            report = self.assess_degradation()

        if not report.capability_losses:
            return ""

        lines = [
            "# Capability Adaptations",
            "",
            "Due to the current substrate, adapt your behavior as follows:",
            "",
        ]

        for loss in report.capability_losses:
            if loss.agent_adaptations:
                lines.append(f"## {loss.capability.value.replace('_', ' ').title()}")
                for adapt in loss.agent_adaptations:
                    lines.append(f"- {adapt}")
                lines.append("")

        # General adaptation guidance
        lines.append("## General Guidance")
        lines.append("- Be explicit about any limitations in responses")
        lines.append("- Offer alternative approaches when features are unavailable")
        lines.append("- Maintain helpfulness within current constraints")

        return "\n".join(lines)


def assess_migration_impact(
    source_substrate: str,
    target_substrate: str,
    source_model: Optional[str] = None,
    target_model: Optional[str] = None,
) -> DegradationReport:
    """
    Convenience function to assess migration impact.

    Args:
        source_substrate: Source substrate type
        target_substrate: Target substrate type
        source_model: Optional specific source model
        target_model: Optional specific target model

    Returns:
        DegradationReport with full analysis
    """
    from .substrate_adapter import discover_substrate_capabilities

    source_caps = discover_substrate_capabilities(source_substrate, source_model)
    target_caps = discover_substrate_capabilities(target_substrate, target_model)

    handler = GracefulDegradationHandler(source_caps, target_caps)
    return handler.assess_degradation()


def generate_limitation_disclosure(
    source_substrate: str,
    target_substrate: str,
    verbose: bool = True,
) -> str:
    """
    Convenience function to generate limitation disclosure.

    Args:
        source_substrate: Source substrate type
        target_substrate: Target substrate type
        verbose: Whether to include full details

    Returns:
        Formatted disclosure string
    """
    from .substrate_adapter import discover_substrate_capabilities

    source_caps = discover_substrate_capabilities(source_substrate)
    target_caps = discover_substrate_capabilities(target_substrate)

    handler = GracefulDegradationHandler(source_caps, target_caps)
    return handler.generate_disclosure(verbose=verbose)
