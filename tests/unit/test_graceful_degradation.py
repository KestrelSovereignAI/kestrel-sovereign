#!/usr/bin/env pytest
"""
Unit tests for the Graceful Degradation module.

Tests capability loss assessment, compensation strategies, and disclosures.
"""
import pytest

from kestrel_sovereign.identity import (
    SubstrateType,
    Capability,
    CapabilityMap,
    GracefulDegradationHandler,
    SeverityLevel,
    CapabilityLoss,
    DegradationReport,
    assess_migration_impact,
    generate_limitation_disclosure,
)


class TestSeverityLevel:
    """Tests for SeverityLevel enum."""

    def test_severity_values(self):
        """Test severity level values."""
        assert SeverityLevel.CRITICAL.value == "critical"
        assert SeverityLevel.HIGH.value == "high"
        assert SeverityLevel.MEDIUM.value == "medium"
        assert SeverityLevel.LOW.value == "low"
        assert SeverityLevel.INFO.value == "info"


class TestCapabilityLoss:
    """Tests for CapabilityLoss dataclass."""

    def test_loss_creation(self):
        """Test creating a capability loss."""
        loss = CapabilityLoss(
            capability=Capability.TOOL_USE,
            severity=SeverityLevel.HIGH,
            impact="Cannot use tools",
            workaround="Manual execution",
        )
        assert loss.capability == Capability.TOOL_USE
        assert loss.severity == SeverityLevel.HIGH

    def test_loss_to_dict(self):
        """Test capability loss serialization."""
        loss = CapabilityLoss(
            capability=Capability.VISION,
            severity=SeverityLevel.MEDIUM,
            impact="Cannot see images",
            workaround="Describe images",
            user_actions=["Describe images in detail"],
            agent_adaptations=["Ask clarifying questions"],
            quality_loss=1.0,
        )
        d = loss.to_dict()
        assert d["capability"] == "vision"
        assert d["severity"] == "medium"
        assert len(d["user_actions"]) == 1
        assert d["quality_loss"] == 1.0


class TestDegradationReport:
    """Tests for DegradationReport dataclass."""

    def test_report_creation(self):
        """Test creating a degradation report."""
        report = DegradationReport(
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            target_substrate=SubstrateType.OLLAMA_LOCAL.value,
            overall_severity=SeverityLevel.MEDIUM,
            estimated_functionality=0.75,
        )
        assert report.source_substrate == "anthropic:claude"
        assert report.target_substrate == "ollama:local"
        assert report.can_proceed is True

    def test_report_to_dict(self):
        """Test report serialization."""
        loss = CapabilityLoss(
            capability=Capability.TOOL_USE,
            severity=SeverityLevel.HIGH,
            impact="Cannot use tools",
            workaround="Manual",
        )
        report = DegradationReport(
            source_substrate="anthropic:claude",
            target_substrate="ollama:local",
            overall_severity=SeverityLevel.HIGH,
            capability_losses=[loss],
            estimated_functionality=0.7,
            recommendations=["Be prepared for manual actions"],
        )
        d = report.to_dict()
        assert d["overall_severity"] == "high"
        assert len(d["capability_losses"]) == 1
        assert d["estimated_functionality"] == 0.7


class TestGracefulDegradationHandler:
    """Tests for GracefulDegradationHandler class."""

    @pytest.fixture
    def claude_caps(self):
        """Create Claude capability map."""
        return CapabilityMap(
            substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            model="claude-sonnet-4-5",
            capabilities={
                Capability.TOOL_USE: True,
                Capability.VISION: True,
                Capability.LONG_CONTEXT: True,
                Capability.STREAMING: True,
                Capability.MULTI_TURN: True,
            },
            quality_scores={
                Capability.TOOL_USE: 0.95,
                Capability.VISION: 0.9,
                Capability.MULTI_TURN: 0.95,
            },
            context_limit=200000,
        )

    @pytest.fixture
    def ollama_caps(self):
        """Create Ollama capability map (limited)."""
        return CapabilityMap(
            substrate=SubstrateType.OLLAMA_LOCAL.value,
            model="llama3",
            capabilities={
                Capability.STREAMING: True,
                Capability.MULTI_TURN: True,
            },
            quality_scores={
                Capability.MULTI_TURN: 0.7,
            },
            context_limit=8192,
        )

    @pytest.fixture
    def gpt_caps(self):
        """Create GPT capability map (similar to Claude)."""
        return CapabilityMap(
            substrate=SubstrateType.OPENAI_GPT.value,
            model="gpt-4o",
            capabilities={
                Capability.TOOL_USE: True,
                Capability.VISION: True,
                Capability.LONG_CONTEXT: True,
                Capability.STREAMING: True,
                Capability.MULTI_TURN: True,
                Capability.STRUCTURED_OUTPUT: True,
            },
            quality_scores={
                Capability.TOOL_USE: 0.9,
                Capability.VISION: 0.85,
                Capability.MULTI_TURN: 0.9,
            },
            context_limit=128000,
        )

    def test_assess_no_degradation(self, claude_caps, gpt_caps):
        """Test assessment when target is similar to source."""
        handler = GracefulDegradationHandler(claude_caps, gpt_caps)
        report = handler.assess_degradation()

        # GPT has similar capabilities, minimal degradation
        assert report.overall_severity in [SeverityLevel.INFO, SeverityLevel.LOW]
        assert report.estimated_functionality >= 0.9
        assert report.can_proceed is True

    def test_assess_significant_degradation(self, claude_caps, ollama_caps):
        """Test assessment with significant capability loss."""
        handler = GracefulDegradationHandler(claude_caps, ollama_caps)
        report = handler.assess_degradation()

        # Ollama missing tool_use, vision, long_context
        assert report.overall_severity in [SeverityLevel.HIGH, SeverityLevel.MEDIUM]
        assert report.estimated_functionality < 0.9
        assert len(report.capability_losses) >= 2  # At least tool_use and vision

    def test_detect_tool_use_loss(self, claude_caps, ollama_caps):
        """Test detection of tool use capability loss."""
        handler = GracefulDegradationHandler(claude_caps, ollama_caps)
        report = handler.assess_degradation()

        tool_losses = [l for l in report.capability_losses
                      if l.capability == Capability.TOOL_USE]
        assert len(tool_losses) == 1
        assert tool_losses[0].severity == SeverityLevel.HIGH

    def test_detect_vision_loss(self, claude_caps, ollama_caps):
        """Test detection of vision capability loss."""
        handler = GracefulDegradationHandler(claude_caps, ollama_caps)
        report = handler.assess_degradation()

        vision_losses = [l for l in report.capability_losses
                        if l.capability == Capability.VISION]
        assert len(vision_losses) == 1
        assert vision_losses[0].severity == SeverityLevel.MEDIUM

    def test_detect_context_reduction(self, claude_caps, ollama_caps):
        """Test detection of context window reduction."""
        handler = GracefulDegradationHandler(claude_caps, ollama_caps)
        report = handler.assess_degradation()

        assert report.context_reduction is not None
        assert report.context_reduction == 200000 - 8192

    def test_detect_quality_degradation(self, claude_caps, ollama_caps):
        """Test detection of quality degradation."""
        handler = GracefulDegradationHandler(claude_caps, ollama_caps)
        report = handler.assess_degradation()

        # Multi-turn quality drops from 0.95 to 0.7
        multi_turn_losses = [l for l in report.capability_losses
                           if l.capability == Capability.MULTI_TURN]
        # Either no loss (still available) or quality degradation
        if multi_turn_losses:
            assert multi_turn_losses[0].quality_loss > 0

    def test_generate_recommendations(self, claude_caps, ollama_caps):
        """Test recommendation generation."""
        handler = GracefulDegradationHandler(claude_caps, ollama_caps)
        report = handler.assess_degradation()

        assert len(report.recommendations) > 0
        # Should have recommendations about tool use and vision
        recs_text = " ".join(report.recommendations).lower()
        assert "tool" in recs_text or "manually" in recs_text

    def test_generate_disclosure_minimal(self, claude_caps, gpt_caps):
        """Test disclosure generation with minimal changes."""
        handler = GracefulDegradationHandler(claude_caps, gpt_caps)
        disclosure = handler.generate_disclosure()

        assert "Substrate Migration Notice" in disclosure
        # Should indicate minimal impact - 100% functionality, INFO level
        assert "100%" in disclosure or "INFO" in disclosure

    def test_generate_disclosure_significant(self, claude_caps, ollama_caps):
        """Test disclosure generation with significant changes."""
        handler = GracefulDegradationHandler(claude_caps, ollama_caps)
        disclosure = handler.generate_disclosure(verbose=True)

        assert "Substrate Migration Notice" in disclosure
        assert "Capability Changes" in disclosure
        # Should mention lost capabilities
        assert "Tool Use" in disclosure or "tool_use" in disclosure.lower()

    def test_generate_disclosure_non_verbose(self, claude_caps, ollama_caps):
        """Test non-verbose disclosure."""
        handler = GracefulDegradationHandler(claude_caps, ollama_caps)
        disclosure = handler.generate_disclosure(verbose=False)

        # Should have basic info but fewer details
        assert "Substrate Migration Notice" in disclosure

    def test_generate_adaptation_prompt(self, claude_caps, ollama_caps):
        """Test adaptation prompt generation."""
        handler = GracefulDegradationHandler(claude_caps, ollama_caps)
        prompt = handler.generate_adaptation_prompt()

        assert "Capability Adaptations" in prompt
        # Should include general guidance
        assert "General Guidance" in prompt

    def test_can_proceed_check(self, claude_caps, ollama_caps):
        """Test that can_proceed is correctly determined."""
        handler = GracefulDegradationHandler(claude_caps, ollama_caps)
        report = handler.assess_degradation()

        # Should still be able to proceed (no CRITICAL severity)
        assert report.can_proceed is True


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_assess_migration_impact(self):
        """Test the assess_migration_impact function."""
        report = assess_migration_impact(
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            target_substrate=SubstrateType.OLLAMA_LOCAL.value,
        )

        assert isinstance(report, DegradationReport)
        assert report.source_substrate == SubstrateType.ANTHROPIC_CLAUDE.value
        assert report.target_substrate == SubstrateType.OLLAMA_LOCAL.value

    def test_assess_migration_impact_same_substrate(self):
        """Test migration within same substrate type."""
        report = assess_migration_impact(
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            target_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
        )

        # Same substrate should have no capability losses
        assert len(report.capability_losses) == 0
        assert report.estimated_functionality == 1.0

    def test_generate_limitation_disclosure(self):
        """Test the generate_limitation_disclosure function."""
        disclosure = generate_limitation_disclosure(
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            target_substrate=SubstrateType.OLLAMA_LOCAL.value,
        )

        assert isinstance(disclosure, str)
        assert "Substrate Migration Notice" in disclosure

    def test_generate_limitation_disclosure_verbose(self):
        """Test verbose limitation disclosure."""
        disclosure = generate_limitation_disclosure(
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            target_substrate=SubstrateType.OLLAMA_LOCAL.value,
            verbose=True,
        )

        # Verbose should include capability details
        assert "Capability Changes" in disclosure


class TestCapabilityImpacts:
    """Tests for capability impact definitions."""

    def test_all_capabilities_have_impacts(self):
        """Test that all capabilities have defined impacts."""
        from kestrel_sovereign.identity.graceful_degradation import CAPABILITY_IMPACTS

        for cap in Capability:
            if cap in [Capability.STREAMING]:  # Some may be minor
                continue
            assert cap in CAPABILITY_IMPACTS or cap.value in str(CAPABILITY_IMPACTS)

    def test_tool_use_impact_complete(self):
        """Test tool use impact is complete."""
        from kestrel_sovereign.identity.graceful_degradation import CAPABILITY_IMPACTS

        tool_impact = CAPABILITY_IMPACTS.get(Capability.TOOL_USE, {})
        assert "severity" in tool_impact
        assert "impact" in tool_impact
        assert "workaround" in tool_impact
        assert "user_actions" in tool_impact
        assert "agent_adaptations" in tool_impact

    def test_severity_levels_make_sense(self):
        """Test that severity levels are logically assigned."""
        from kestrel_sovereign.identity.graceful_degradation import CAPABILITY_IMPACTS

        # Tool use should be high severity
        assert CAPABILITY_IMPACTS[Capability.TOOL_USE]["severity"] == SeverityLevel.HIGH

        # Streaming should be low severity
        assert CAPABILITY_IMPACTS[Capability.STREAMING]["severity"] == SeverityLevel.LOW

        # Vision should be medium
        assert CAPABILITY_IMPACTS[Capability.VISION]["severity"] == SeverityLevel.MEDIUM
