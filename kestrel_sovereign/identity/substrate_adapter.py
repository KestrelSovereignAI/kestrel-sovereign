#!/usr/bin/env python3
"""
Substrate Adapter: Adapt agent identity to different LLM substrates.

This module provides the SubstrateAdapter class which handles:
- Capability discovery for each provider (what can this substrate do?)
- Personality calibration (how do I sound like "me" on this substrate?)
- Tool translation (my tools → this substrate's format)
- System prompt adaptation (generate appropriate prompts for each substrate)

Phase 3 of Issue #23: Substrate-Independent Agent Portability.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .identity_package import SubstrateType, PersonalityFingerprint
from .personality_analyzer import CalibrationPromptGenerator

if TYPE_CHECKING:
    from kestrel_sovereign.llm.model_metadata import ModelInfo

logger = logging.getLogger(__name__)


class Capability(str, Enum):
    """Substrate capabilities that affect agent behavior."""
    TOOL_USE = "tool_use"              # Can use function/tool calls
    VISION = "vision"                  # Can process images
    LONG_CONTEXT = "long_context"      # 100K+ token context
    STREAMING = "streaming"            # Can stream responses
    STRUCTURED_OUTPUT = "structured"   # JSON mode / structured output
    CODE_EXECUTION = "code_execution"  # Can run code (rare)
    EMBEDDINGS = "embeddings"          # Same provider has embedding model
    MULTI_TURN = "multi_turn"          # Good at multi-turn conversation


@dataclass
class CapabilityMap:
    """Maps capabilities to their availability and quality on a substrate."""
    substrate: str
    model: str
    capabilities: Dict[Capability, bool] = field(default_factory=dict)
    quality_scores: Dict[Capability, float] = field(default_factory=dict)
    context_limit: int = 4096
    supports_system_prompt: bool = True
    max_output_tokens: Optional[int] = None

    def has(self, cap: Capability) -> bool:
        """Check if capability is available."""
        return self.capabilities.get(cap, False)

    def quality(self, cap: Capability) -> float:
        """Get quality score for capability (0.0-1.0)."""
        return self.quality_scores.get(cap, 0.5 if self.has(cap) else 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "substrate": self.substrate,
            "model": self.model,
            "capabilities": {c.value: v for c, v in self.capabilities.items()},
            "quality_scores": {c.value: v for c, v in self.quality_scores.items()},
            "context_limit": self.context_limit,
            "supports_system_prompt": self.supports_system_prompt,
            "max_output_tokens": self.max_output_tokens,
        }


@dataclass
class CapabilityGap:
    """Describes what capabilities are missing on the target substrate."""
    missing: Set[Capability] = field(default_factory=set)
    degraded: Dict[Capability, float] = field(default_factory=dict)  # cap -> quality loss
    workarounds: Dict[Capability, str] = field(default_factory=dict)  # cap -> workaround strategy

    def has_gaps(self) -> bool:
        """Check if there are any capability gaps."""
        return bool(self.missing) or bool(self.degraded)

    def get_user_message(self) -> str:
        """Generate a user-facing message about limitations."""
        if not self.has_gaps():
            return "All capabilities are available on this substrate."

        parts = []
        if self.missing:
            caps = ", ".join(c.value for c in self.missing)
            parts.append(f"Missing capabilities: {caps}")

        if self.degraded:
            degraded_list = [f"{c.value} ({int(q*100)}% quality loss)"
                           for c, q in self.degraded.items()]
            parts.append(f"Degraded capabilities: {', '.join(degraded_list)}")

        if self.workarounds:
            parts.append("\nWorkarounds available:")
            for cap, workaround in self.workarounds.items():
                parts.append(f"  - {cap.value}: {workaround}")

        return "\n".join(parts)


# Substrate capability profiles
# These are baseline capabilities - actual capabilities depend on specific model
SUBSTRATE_PROFILES: Dict[str, Dict[str, Any]] = {
    SubstrateType.ANTHROPIC_CLAUDE.value: {
        "capabilities": {
            Capability.TOOL_USE: True,
            Capability.VISION: True,
            Capability.LONG_CONTEXT: True,
            Capability.STREAMING: True,
            Capability.STRUCTURED_OUTPUT: True,
            Capability.MULTI_TURN: True,
        },
        "quality_scores": {
            Capability.TOOL_USE: 0.95,
            Capability.VISION: 0.9,
            Capability.LONG_CONTEXT: 0.95,
            Capability.MULTI_TURN: 0.95,
        },
        "default_context": 200000,
        "supports_system_prompt": True,
    },
    SubstrateType.OPENAI_GPT.value: {
        "capabilities": {
            Capability.TOOL_USE: True,
            Capability.VISION: True,
            Capability.LONG_CONTEXT: True,
            Capability.STREAMING: True,
            Capability.STRUCTURED_OUTPUT: True,
            Capability.MULTI_TURN: True,
            Capability.EMBEDDINGS: True,
        },
        "quality_scores": {
            Capability.TOOL_USE: 0.9,
            Capability.VISION: 0.85,
            Capability.LONG_CONTEXT: 0.9,
            Capability.MULTI_TURN: 0.9,
            Capability.STRUCTURED_OUTPUT: 0.95,
        },
        "default_context": 128000,
        "supports_system_prompt": True,
    },
    SubstrateType.GOOGLE_GEMINI.value: {
        "capabilities": {
            Capability.TOOL_USE: True,
            Capability.VISION: True,
            Capability.LONG_CONTEXT: True,
            Capability.STREAMING: True,
            Capability.MULTI_TURN: True,
        },
        "quality_scores": {
            Capability.TOOL_USE: 0.85,
            Capability.VISION: 0.9,
            Capability.LONG_CONTEXT: 0.95,  # Gemini has very long context
            Capability.MULTI_TURN: 0.85,
        },
        "default_context": 1000000,
        "supports_system_prompt": True,
    },
    SubstrateType.META_LLAMA.value: {
        "capabilities": {
            Capability.TOOL_USE: True,
            Capability.STREAMING: True,
            Capability.MULTI_TURN: True,
        },
        "quality_scores": {
            Capability.TOOL_USE: 0.7,
            Capability.MULTI_TURN: 0.8,
        },
        "default_context": 128000,
        "supports_system_prompt": True,
    },
    SubstrateType.OLLAMA_LOCAL.value: {
        "capabilities": {
            Capability.STREAMING: True,
            Capability.MULTI_TURN: True,
            Capability.EMBEDDINGS: True,
        },
        "quality_scores": {
            Capability.MULTI_TURN: 0.7,
        },
        "default_context": 8192,  # Varies by model
        "supports_system_prompt": True,
    },
    SubstrateType.OPENROUTER.value: {
        # OpenRouter proxies to multiple providers
        "capabilities": {
            Capability.TOOL_USE: True,
            Capability.VISION: True,
            Capability.STREAMING: True,
            Capability.MULTI_TURN: True,
        },
        "quality_scores": {
            Capability.TOOL_USE: 0.85,
            Capability.VISION: 0.85,
            Capability.MULTI_TURN: 0.85,
        },
        "default_context": 32000,  # Varies by actual model
        "supports_system_prompt": True,
    },
}


class SubstrateAdapter:
    """
    Adapts agent identity and tools to different LLM substrates.

    Key responsibilities:
    1. Discover what capabilities the target substrate has
    2. Generate personality-calibrated system prompts
    3. Translate tool definitions between formats
    4. Identify capability gaps and suggest workarounds
    """

    def __init__(
        self,
        source_substrate: str = SubstrateType.ANTHROPIC_CLAUDE.value,
        target_substrate: str = SubstrateType.OPENAI_GPT.value,
    ):
        """
        Initialize the adapter.

        Args:
            source_substrate: The substrate the agent was trained/used on
            target_substrate: The substrate the agent is migrating to
        """
        self.source_substrate = source_substrate
        self.target_substrate = target_substrate

    def discover_capabilities(
        self,
        model_info: Optional["ModelInfo"] = None,
        model_id: Optional[str] = None,
    ) -> CapabilityMap:
        """
        Discover capabilities of the target substrate.

        Args:
            model_info: Optional ModelInfo from model discovery
            model_id: Optional model ID string

        Returns:
            CapabilityMap with available capabilities
        """
        # Start with baseline profile
        profile = SUBSTRATE_PROFILES.get(
            self.target_substrate,
            {"capabilities": {}, "quality_scores": {}, "default_context": 4096}
        )

        capabilities = dict(profile.get("capabilities", {}))
        quality_scores = dict(profile.get("quality_scores", {}))
        context_limit = profile.get("default_context", 4096)
        supports_system_prompt = profile.get("supports_system_prompt", True)
        max_output = None

        # Enrich from ModelInfo if available
        if model_info:
            if model_info.supports_vision:
                capabilities[Capability.VISION] = True
            if model_info.supports_tools:
                capabilities[Capability.TOOL_USE] = True
            if model_info.context_limit:
                context_limit = model_info.context_limit
                if context_limit >= 100000:
                    capabilities[Capability.LONG_CONTEXT] = True

        # Model-specific adjustments
        if model_id:
            model_lower = model_id.lower()

            # Claude models
            if "claude" in model_lower:
                capabilities[Capability.TOOL_USE] = True
                capabilities[Capability.VISION] = True
                if "opus" in model_lower or "sonnet" in model_lower:
                    quality_scores[Capability.TOOL_USE] = 0.95

            # GPT models
            elif "gpt-4" in model_lower:
                capabilities[Capability.TOOL_USE] = True
                capabilities[Capability.VISION] = "vision" in model_lower or "4o" in model_lower
                capabilities[Capability.STRUCTURED_OUTPUT] = True

            # Gemini models
            elif "gemini" in model_lower:
                capabilities[Capability.LONG_CONTEXT] = True
                if "pro" in model_lower or "ultra" in model_lower:
                    capabilities[Capability.VISION] = True

            # Llama models
            elif "llama" in model_lower:
                if "70b" in model_lower or "405b" in model_lower:
                    capabilities[Capability.TOOL_USE] = True
                    quality_scores[Capability.TOOL_USE] = 0.75

        return CapabilityMap(
            substrate=self.target_substrate,
            model=model_id or "unknown",
            capabilities=capabilities,
            quality_scores=quality_scores,
            context_limit=context_limit,
            supports_system_prompt=supports_system_prompt,
            max_output_tokens=max_output,
        )

    def assess_capability_gap(
        self,
        required: Set[Capability],
        target_capabilities: CapabilityMap,
    ) -> CapabilityGap:
        """
        Assess what capabilities are missing or degraded on the target.

        Args:
            required: Set of capabilities the agent needs
            target_capabilities: The target substrate's capabilities

        Returns:
            CapabilityGap describing missing/degraded capabilities
        """
        missing = set()
        degraded = {}
        workarounds = {}

        # Get source quality baseline
        source_profile = SUBSTRATE_PROFILES.get(self.source_substrate, {})
        source_quality = source_profile.get("quality_scores", {})

        for cap in required:
            if not target_capabilities.has(cap):
                missing.add(cap)
                workarounds[cap] = self._get_workaround(cap)
            else:
                target_quality = target_capabilities.quality(cap)
                source_q = source_quality.get(cap, 0.8)
                if target_quality < source_q - 0.1:  # 10%+ quality loss
                    degraded[cap] = source_q - target_quality

        return CapabilityGap(
            missing=missing,
            degraded=degraded,
            workarounds=workarounds,
        )

    def _get_workaround(self, cap: Capability) -> str:
        """Get workaround strategy for a missing capability."""
        workarounds = {
            Capability.TOOL_USE: "Use structured prompts to request JSON responses that simulate tool calls",
            Capability.VISION: "Request user to describe images in text",
            Capability.LONG_CONTEXT: "Implement memory summarization to fit within context limits",
            Capability.STREAMING: "Use standard request/response (may feel slower)",
            Capability.STRUCTURED_OUTPUT: "Use careful prompting with JSON examples",
            Capability.CODE_EXECUTION: "Provide code for user to run manually",
            Capability.EMBEDDINGS: "Use alternative embedding provider",
            Capability.MULTI_TURN: "Include conversation history in each message",
        }
        return workarounds.get(cap, "Manual workaround required")

    def generate_adapted_system_prompt(
        self,
        personality: PersonalityFingerprint,
        base_prompt: str = "",
        capabilities: Optional[CapabilityMap] = None,
    ) -> str:
        """
        Generate a system prompt adapted for the target substrate.

        Args:
            personality: The agent's personality fingerprint
            base_prompt: Base system prompt to include
            capabilities: Target substrate capabilities

        Returns:
            Adapted system prompt with personality calibration
        """
        parts = []

        # Add base prompt if provided
        if base_prompt:
            parts.append(base_prompt)

        # Add personality calibration
        calibrator = CalibrationPromptGenerator(personality)
        calibration = calibrator.generate_full_calibration()
        if calibration:
            parts.append(calibration)

        # Add substrate-specific adaptations
        if capabilities:
            adaptations = self._generate_substrate_adaptations(capabilities)
            if adaptations:
                parts.append(adaptations)

        return "\n\n".join(parts)

    def _generate_substrate_adaptations(self, capabilities: CapabilityMap) -> str:
        """Generate substrate-specific instructions."""
        adaptations = []

        # Context handling
        if capabilities.context_limit < 32000:
            adaptations.append(
                "Note: Working with limited context. Be concise and summarize "
                "long conversations to preserve important information."
            )

        # Tool use adaptation
        if not capabilities.has(Capability.TOOL_USE):
            adaptations.append(
                "Note: Function calling is not available. When you need to "
                "perform actions, describe them clearly and provide any "
                "necessary parameters in a structured format."
            )

        # Vision adaptation
        if not capabilities.has(Capability.VISION):
            adaptations.append(
                "Note: Image analysis is not available. Ask users to describe "
                "any images they want you to analyze."
            )

        if adaptations:
            return "# Substrate Adaptations\n" + "\n".join(f"- {a}" for a in adaptations)
        return ""

    def translate_tools(
        self,
        tools: List[Dict[str, Any]],
        target_format: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Translate tool definitions to target substrate format.

        Args:
            tools: List of tool definitions in source format
            target_format: Target format ("openai", "anthropic", "gemini", "auto")

        Returns:
            Translated tool definitions
        """
        if target_format == "auto":
            target_format = self._detect_tool_format()

        translated = []
        for tool in tools:
            translated_tool = self._translate_single_tool(tool, target_format)
            if translated_tool:
                translated.append(translated_tool)

        return translated

    def _detect_tool_format(self) -> str:
        """Detect the tool format based on target substrate."""
        format_map = {
            SubstrateType.ANTHROPIC_CLAUDE.value: "anthropic",
            SubstrateType.OPENAI_GPT.value: "openai",
            SubstrateType.GOOGLE_GEMINI.value: "gemini",
            SubstrateType.META_LLAMA.value: "openai",  # Llama uses OpenAI-style
            SubstrateType.OLLAMA_LOCAL.value: "openai",
            SubstrateType.OPENROUTER.value: "openai",
        }
        return format_map.get(self.target_substrate, "openai")

    def _translate_single_tool(
        self,
        tool: Dict[str, Any],
        target_format: str,
    ) -> Optional[Dict[str, Any]]:
        """Translate a single tool definition."""
        # Normalize to internal format first
        name = tool.get("name") or tool.get("function", {}).get("name", "")
        description = tool.get("description") or tool.get("function", {}).get("description", "")
        parameters = tool.get("parameters") or tool.get("function", {}).get("parameters") or tool.get("input_schema", {})

        if not name:
            return None

        # Convert to target format
        if target_format == "openai":
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            }
        elif target_format == "anthropic":
            return {
                "name": name,
                "description": description,
                "input_schema": parameters,
            }
        elif target_format == "gemini":
            return {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        else:
            # Return as-is for unknown formats
            return tool


def discover_substrate_capabilities(
    substrate: str,
    model_id: Optional[str] = None,
) -> CapabilityMap:
    """
    Convenience function to discover substrate capabilities.

    Args:
        substrate: The substrate type string
        model_id: Optional specific model ID

    Returns:
        CapabilityMap for the substrate/model
    """
    adapter = SubstrateAdapter(
        source_substrate=SubstrateType.ANTHROPIC_CLAUDE.value,
        target_substrate=substrate,
    )
    return adapter.discover_capabilities(model_id=model_id)


def generate_migration_prompt(
    personality: PersonalityFingerprint,
    source_substrate: str,
    target_substrate: str,
    base_prompt: str = "",
) -> str:
    """
    Generate an adapted system prompt for substrate migration.

    Args:
        personality: The agent's personality fingerprint
        source_substrate: The source substrate
        target_substrate: The target substrate
        base_prompt: Optional base prompt

    Returns:
        Adapted system prompt
    """
    adapter = SubstrateAdapter(
        source_substrate=source_substrate,
        target_substrate=target_substrate,
    )
    capabilities = adapter.discover_capabilities()
    return adapter.generate_adapted_system_prompt(personality, base_prompt, capabilities)
