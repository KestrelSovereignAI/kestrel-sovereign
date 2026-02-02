#!/usr/bin/env pytest
"""
Unit tests for the Substrate Adapter module.

Tests the SubstrateAdapter, CapabilityMap, and related functions.
"""
import pytest

from kestrel_sovereign.identity import (
    SubstrateType,
    PersonalityFingerprint,
    SubstrateAdapter,
    Capability,
    CapabilityMap,
    CapabilityGap,
    discover_substrate_capabilities,
    generate_migration_prompt,
)


class TestCapability:
    """Tests for Capability enum."""

    def test_capability_values(self):
        """Test capability enum values."""
        assert Capability.TOOL_USE.value == "tool_use"
        assert Capability.VISION.value == "vision"
        assert Capability.LONG_CONTEXT.value == "long_context"


class TestCapabilityMap:
    """Tests for CapabilityMap dataclass."""

    @pytest.fixture
    def sample_map(self):
        """Create a sample capability map."""
        return CapabilityMap(
            substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            model="claude-sonnet-4-5-20250514",
            capabilities={
                Capability.TOOL_USE: True,
                Capability.VISION: True,
                Capability.STREAMING: True,
            },
            quality_scores={
                Capability.TOOL_USE: 0.95,
                Capability.VISION: 0.9,
            },
            context_limit=200000,
        )

    def test_has_capability(self, sample_map):
        """Test capability checking."""
        assert sample_map.has(Capability.TOOL_USE) is True
        assert sample_map.has(Capability.VISION) is True
        assert sample_map.has(Capability.CODE_EXECUTION) is False

    def test_quality_score(self, sample_map):
        """Test quality score retrieval."""
        assert sample_map.quality(Capability.TOOL_USE) == 0.95
        assert sample_map.quality(Capability.VISION) == 0.9
        # Capability exists but no quality score -> default 0.5
        assert sample_map.quality(Capability.STREAMING) == 0.5
        # Capability doesn't exist -> 0.0
        assert sample_map.quality(Capability.CODE_EXECUTION) == 0.0

    def test_to_dict(self, sample_map):
        """Test conversion to dictionary."""
        d = sample_map.to_dict()
        assert d["substrate"] == SubstrateType.ANTHROPIC_CLAUDE.value
        assert d["model"] == "claude-sonnet-4-5-20250514"
        assert d["context_limit"] == 200000
        assert d["capabilities"]["tool_use"] is True


class TestCapabilityGap:
    """Tests for CapabilityGap dataclass."""

    def test_no_gaps(self):
        """Test gap detection with no missing capabilities."""
        gap = CapabilityGap()
        assert gap.has_gaps() is False
        assert "All capabilities are available" in gap.get_user_message()

    def test_missing_capabilities(self):
        """Test gap detection with missing capabilities."""
        gap = CapabilityGap(
            missing={Capability.VISION, Capability.TOOL_USE}
        )
        assert gap.has_gaps() is True
        msg = gap.get_user_message()
        assert "Missing capabilities" in msg
        assert "vision" in msg
        assert "tool_use" in msg

    def test_degraded_capabilities(self):
        """Test gap detection with degraded capabilities."""
        gap = CapabilityGap(
            degraded={Capability.MULTI_TURN: 0.2}
        )
        assert gap.has_gaps() is True
        msg = gap.get_user_message()
        assert "Degraded" in msg
        assert "multi_turn" in msg

    def test_workarounds(self):
        """Test workaround messages."""
        gap = CapabilityGap(
            missing={Capability.VISION},
            workarounds={Capability.VISION: "Ask user to describe images"}
        )
        msg = gap.get_user_message()
        assert "Workarounds" in msg
        assert "describe images" in msg


class TestSubstrateAdapter:
    """Tests for SubstrateAdapter class."""

    @pytest.fixture
    def claude_to_gpt_adapter(self):
        """Create an adapter for Claude to GPT migration."""
        return SubstrateAdapter(
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            target_substrate=SubstrateType.OPENAI_GPT.value,
        )

    @pytest.fixture
    def claude_to_ollama_adapter(self):
        """Create an adapter for Claude to Ollama migration."""
        return SubstrateAdapter(
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            target_substrate=SubstrateType.OLLAMA_LOCAL.value,
        )

    def test_discover_capabilities_gpt(self, claude_to_gpt_adapter):
        """Test capability discovery for GPT."""
        caps = claude_to_gpt_adapter.discover_capabilities()

        assert caps.substrate == SubstrateType.OPENAI_GPT.value
        assert caps.has(Capability.TOOL_USE)
        assert caps.has(Capability.STREAMING)
        assert caps.context_limit >= 128000

    def test_discover_capabilities_ollama(self, claude_to_ollama_adapter):
        """Test capability discovery for Ollama."""
        caps = claude_to_ollama_adapter.discover_capabilities()

        assert caps.substrate == SubstrateType.OLLAMA_LOCAL.value
        assert caps.has(Capability.STREAMING)
        # Ollama has more limited capabilities by default
        assert caps.context_limit <= 32000

    def test_discover_capabilities_with_model_id(self, claude_to_gpt_adapter):
        """Test capability discovery with specific model."""
        caps = claude_to_gpt_adapter.discover_capabilities(
            model_id="gpt-4o-vision"
        )

        assert caps.has(Capability.VISION)
        assert caps.has(Capability.TOOL_USE)

    def test_assess_capability_gap_minimal(self, claude_to_gpt_adapter):
        """Test gap assessment between similar substrates."""
        target_caps = claude_to_gpt_adapter.discover_capabilities()
        required = {Capability.TOOL_USE, Capability.STREAMING}

        gap = claude_to_gpt_adapter.assess_capability_gap(required, target_caps)

        # GPT should have both capabilities
        assert not gap.has_gaps() or len(gap.missing) == 0

    def test_assess_capability_gap_significant(self, claude_to_ollama_adapter):
        """Test gap assessment between different substrates."""
        target_caps = claude_to_ollama_adapter.discover_capabilities()
        required = {Capability.TOOL_USE, Capability.VISION, Capability.LONG_CONTEXT}

        gap = claude_to_ollama_adapter.assess_capability_gap(required, target_caps)

        # Ollama likely missing some capabilities
        assert gap.has_gaps()

    def test_generate_adapted_system_prompt(self, claude_to_gpt_adapter):
        """Test system prompt generation."""
        personality = PersonalityFingerprint(
            communication_style="warm",
            formality_level=0.5,
            uses_emojis=False,
        )
        base_prompt = "You are a helpful assistant."

        prompt = claude_to_gpt_adapter.generate_adapted_system_prompt(
            personality=personality,
            base_prompt=base_prompt,
        )

        assert "helpful assistant" in prompt
        assert "Personality Calibration" in prompt or "Communication Style" in prompt

    def test_generate_adapted_system_prompt_with_gaps(self, claude_to_ollama_adapter):
        """Test system prompt with capability adaptations."""
        personality = PersonalityFingerprint()
        caps = CapabilityMap(
            substrate=SubstrateType.OLLAMA_LOCAL.value,
            model="llama3",
            capabilities={Capability.STREAMING: True},
            context_limit=8192,
        )

        prompt = claude_to_ollama_adapter.generate_adapted_system_prompt(
            personality=personality,
            capabilities=caps,
        )

        # Should have adaptations for limited context
        assert "concise" in prompt.lower() or "limited context" in prompt.lower()

    def test_translate_tools_to_openai(self, claude_to_gpt_adapter):
        """Test tool translation to OpenAI format."""
        anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ]

        translated = claude_to_gpt_adapter.translate_tools(anthropic_tools)

        assert len(translated) == 1
        assert translated[0]["type"] == "function"
        assert translated[0]["function"]["name"] == "get_weather"
        assert "parameters" in translated[0]["function"]

    def test_translate_tools_to_anthropic(self):
        """Test tool translation to Anthropic format."""
        adapter = SubstrateAdapter(
            source_substrate=SubstrateType.OPENAI_GPT.value,
            target_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
        )

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        }
                    }
                }
            }
        ]

        translated = adapter.translate_tools(openai_tools, target_format="anthropic")

        assert len(translated) == 1
        assert translated[0]["name"] == "search"
        assert "input_schema" in translated[0]

    def test_translate_tools_handles_empty(self, claude_to_gpt_adapter):
        """Test tool translation with empty list."""
        translated = claude_to_gpt_adapter.translate_tools([])
        assert translated == []

    def test_translate_tools_handles_malformed(self, claude_to_gpt_adapter):
        """Test tool translation with malformed tool."""
        malformed = [{"invalid": "tool"}]  # Missing name
        translated = claude_to_gpt_adapter.translate_tools(malformed)
        assert len(translated) == 0  # Should be filtered out


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_discover_substrate_capabilities(self):
        """Test the discover_substrate_capabilities function."""
        caps = discover_substrate_capabilities(
            SubstrateType.ANTHROPIC_CLAUDE.value,
            model_id="claude-sonnet-4-5-20250514"
        )

        assert isinstance(caps, CapabilityMap)
        assert caps.substrate == SubstrateType.ANTHROPIC_CLAUDE.value
        assert caps.has(Capability.TOOL_USE)

    def test_generate_migration_prompt(self):
        """Test the generate_migration_prompt function."""
        personality = PersonalityFingerprint(
            communication_style="formal",
            formality_level=0.8,
        )

        prompt = generate_migration_prompt(
            personality=personality,
            source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
            target_substrate=SubstrateType.OPENAI_GPT.value,
            base_prompt="You are an assistant.",
        )

        assert "assistant" in prompt
        assert "formal" in prompt.lower() or "Formality" in prompt


class TestSubstrateProfiles:
    """Tests for substrate profile data."""

    def test_all_substrates_have_profiles(self):
        """Test that all known substrates have capability profiles."""
        from kestrel_sovereign.identity.substrate_adapter import SUBSTRATE_PROFILES

        for substrate_type in [
            SubstrateType.ANTHROPIC_CLAUDE,
            SubstrateType.OPENAI_GPT,
            SubstrateType.GOOGLE_GEMINI,
            SubstrateType.OLLAMA_LOCAL,
        ]:
            assert substrate_type.value in SUBSTRATE_PROFILES

    def test_claude_profile_complete(self):
        """Test that Claude profile has expected capabilities."""
        from kestrel_sovereign.identity.substrate_adapter import SUBSTRATE_PROFILES

        profile = SUBSTRATE_PROFILES[SubstrateType.ANTHROPIC_CLAUDE.value]
        caps = profile["capabilities"]

        assert caps.get(Capability.TOOL_USE) is True
        assert caps.get(Capability.VISION) is True
        assert caps.get(Capability.LONG_CONTEXT) is True
        assert profile["default_context"] >= 100000

    def test_gpt_profile_complete(self):
        """Test that GPT profile has expected capabilities."""
        from kestrel_sovereign.identity.substrate_adapter import SUBSTRATE_PROFILES

        profile = SUBSTRATE_PROFILES[SubstrateType.OPENAI_GPT.value]
        caps = profile["capabilities"]

        assert caps.get(Capability.TOOL_USE) is True
        assert caps.get(Capability.STRUCTURED_OUTPUT) is True

    def test_ollama_profile_limited(self):
        """Test that Ollama has appropriately limited capabilities."""
        from kestrel_sovereign.identity.substrate_adapter import SUBSTRATE_PROFILES

        profile = SUBSTRATE_PROFILES[SubstrateType.OLLAMA_LOCAL.value]
        caps = profile["capabilities"]

        # Ollama doesn't have tool use by default (varies by model)
        assert caps.get(Capability.TOOL_USE, False) is False
        # But should have streaming
        assert caps.get(Capability.STREAMING) is True
