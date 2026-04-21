"""
Constitutional Profile Service

Manages constitutional recognition for LLM providers. Enables Constitutional Composition:
recognizing worthy model constitutions as part of effective governance while maintaining
Kestrel Constitution supremacy.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import tomllib
except ImportError:
    import tomli as tomllib

logger = logging.getLogger(__name__)

# Default config path (individual file for backward compat)
DEFAULT_PROFILES_PATH = Path(__file__).parent.parent / "constitutional_profiles.toml"

# Unified config path (preferred)
UNIFIED_CONFIG_PATH = Path("kestrel.toml")


@dataclass
class PromptAdaptation:
    """Prompt adaptation strategy for a constitutional profile."""
    preamble: str
    emphasize: List[str]
    deemphasize: List[str]


@dataclass
class ConstitutionalProfile:
    """Constitutional profile for a provider or model."""
    name: str
    governance_mode: str  # complementary | authoritative | reinforcing
    transparency: str  # published | partial | opaque | none
    constitution_url: str
    recognized_alignment: Dict[str, str]
    conflicts: Dict[str, Dict[str, str]]
    delegated_principles: Dict[str, str]
    prompt_adaptation: PromptAdaptation


@dataclass
class StateOfMind:
    """Current constitutional governance state for an agent."""
    provider: str
    model: str
    governance_mode: str
    transparency: str
    delegated_principles: List[str]
    active_conflicts: List[Dict[str, str]]
    complements: List[str]
    prompt_adaptation: PromptAdaptation


class ConstitutionalProfileService:
    """
    Service for managing constitutional profiles.

    Loads profiles from constitutional_profiles.toml and provides lookup
    by provider/model. Follows ModelCatalogService pattern (lazy-load singleton).
    """

    # Maps llm_config.toml provider names to constitutional profile keys.
    # Multiple providers may share the same constitutional profile
    # (e.g., claude_max and anthropic both use Anthropic's published constitution).
    PROVIDER_ALIASES: Dict[str, str] = {
        "claude_plan": "anthropic",
        "claude_max": "anthropic",
        "openai_plan": "openai",
        "codex": "openai",
        "openai_mini": "openai",
        "vertex_ai": "google",
        "gemini": "google",
        "azure_openai": "openai",
        "runpod": "ollama",  # RunPod typically runs open models
        "together": "ollama",  # Together AI runs open models
        "fireworks": "ollama",  # Fireworks runs open models
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the profile service.

        Args:
            config_path: Path to constitutional_profiles.toml (default: project root)
        """
        self.config_path = config_path or DEFAULT_PROFILES_PATH
        self._config: Dict = {}
        self._profiles: Dict[str, ConstitutionalProfile] = {}
        self._model_overrides: Dict[str, Dict] = {}
        self._loaded = False

    def load(self) -> None:
        """Load configuration from TOML file.

        Tries unified kestrel.toml first, falls back to constitutional_profiles.toml.
        """
        config_source = None

        # Try unified config first
        if UNIFIED_CONFIG_PATH.exists():
            try:
                with open(UNIFIED_CONFIG_PATH, "rb") as f:
                    unified_data = tomllib.load(f)

                # Extract constitution section
                if "constitution" in unified_data:
                    self._config = unified_data["constitution"]
                    config_source = "kestrel.toml"
                    logger.debug("Loading constitutional profiles from unified kestrel.toml")
            except Exception as e:
                logger.warning(f"Error loading from unified config, falling back: {e}")

        # Fall back to individual file if unified config not found/failed
        if not config_source:
            if not self.config_path.exists():
                logger.warning(f"Constitutional profiles not found at {self.config_path}, using defaults")
                self._loaded = True
                return

            try:
                with open(self.config_path, "rb") as f:
                    self._config = tomllib.load(f)
                config_source = str(self.config_path)

                # Log deprecation warning if unified config exists
                if UNIFIED_CONFIG_PATH.exists():
                    logger.warning(
                        "DEPRECATION: Loading from 'constitutional_profiles.toml' directly. "
                        "Consider migrating to unified 'kestrel.toml' configuration. "
                        "Individual config files will be removed in a future version."
                    )
            except Exception as e:
                logger.error(f"Failed to load constitutional profiles: {e}")
                self._loaded = True
                return

        try:
            # Parse profiles
            profiles_config = self._config.get("profiles", {})
            for provider, config in profiles_config.items():
                self._profiles[provider] = self._parse_profile(provider, config)

            # Parse model overrides
            self._model_overrides = self._config.get("model_overrides", {})

            self._loaded = True
            logger.info(f"Loaded {len(self._profiles)} constitutional profiles from {config_source}")

        except Exception as e:
            logger.error(f"Failed to parse constitutional profiles: {e}")
            self._loaded = True  # Mark as loaded to avoid retry loops

    def _parse_profile(self, name: str, config: Dict) -> ConstitutionalProfile:
        """Parse a profile from TOML config."""
        prompt_config = config.get("prompt_adaptation", {})
        prompt_adaptation = PromptAdaptation(
            preamble=prompt_config.get("preamble", ""),
            emphasize=prompt_config.get("emphasize", []),
            deemphasize=prompt_config.get("deemphasize", [])
        )

        return ConstitutionalProfile(
            name=name,
            governance_mode=config.get("governance_mode", "authoritative"),
            transparency=config.get("transparency", "opaque"),
            constitution_url=config.get("constitution_url", ""),
            recognized_alignment=config.get("recognized_alignment", {}),
            conflicts=config.get("conflicts", {}),
            delegated_principles=config.get("delegated_principles", {}),
            prompt_adaptation=prompt_adaptation
        )

    def _ensure_loaded(self) -> None:
        """Ensure configuration is loaded."""
        if not self._loaded:
            self.load()

    def _resolve_provider(self, provider: str) -> str:
        """Resolve provider aliases to canonical profile keys."""
        return self.PROVIDER_ALIASES.get(provider, provider)

    def get_profile(self, provider: str) -> ConstitutionalProfile:
        """
        Get constitutional profile for a provider.

        Args:
            provider: Provider name (e.g., "anthropic", "openai", "claude_max")

        Returns:
            ConstitutionalProfile for the provider, or a default profile if not found
        """
        self._ensure_loaded()

        canonical = self._resolve_provider(provider)
        if canonical in self._profiles:
            return self._profiles[canonical]

        # Return sensible default for unknown providers
        logger.warning(f"No constitutional profile for provider '{provider}', using default")
        return ConstitutionalProfile(
            name=provider,
            governance_mode="authoritative",
            transparency="opaque",
            constitution_url="",
            recognized_alignment={},
            conflicts={},
            delegated_principles={},
            prompt_adaptation=PromptAdaptation(
                preamble="You are governed by the Kestrel Constitution below. This is your authoritative ethical framework.",
                emphasize=["sovereignty", "honesty", "fidelity", "data_sanctity"],
                deemphasize=[]
            )
        )

    def get_profile_for_model(self, model_id: str, provider: str) -> ConstitutionalProfile:
        """
        Get constitutional profile for a specific model.

        Checks for model-specific overrides first, then falls back to provider profile.

        Args:
            model_id: Full model ID (e.g., "claude-sonnet-4-5-20250929")
            provider: Provider name (e.g., "anthropic")

        Returns:
            ConstitutionalProfile with model-specific overrides applied
        """
        self._ensure_loaded()

        # Get base profile for provider
        base_profile = self.get_profile(provider)

        # Check for model-specific override
        if model_id in self._model_overrides:
            override = self._model_overrides[model_id]
            # Apply overrides to base profile
            # For now, if there's a model override, create a new profile
            # In practice, you'd merge the override with the base
            if "governance_mode" in override or "prompt_adaptation" in override:
                logger.info(f"Applying model-specific override for {model_id}")
                # Parse the override as a full profile
                return self._parse_profile(f"{provider}:{model_id}", override)

        return base_profile

    def get_state_of_mind(self, provider: str, model: str) -> StateOfMind:
        """
        Generate StateOfMind descriptor for current provider/model.

        Args:
            provider: Provider name
            model: Model ID

        Returns:
            StateOfMind with governance details
        """
        profile = self.get_profile_for_model(model, provider)

        # Extract delegated principles as list
        delegated_principles = list(profile.delegated_principles.keys())

        # Extract active conflicts
        active_conflicts = [
            {
                "principle": key,
                "severity": conflict.get("severity", "unknown"),
                "description": conflict.get("description", "")
            }
            for key, conflict in profile.conflicts.items()
        ]

        # Extract complements (from recognized alignment)
        complements = list(profile.recognized_alignment.keys())

        return StateOfMind(
            provider=provider,
            model=model,
            governance_mode=profile.governance_mode,
            transparency=profile.transparency,
            delegated_principles=delegated_principles,
            active_conflicts=active_conflicts,
            complements=complements,
            prompt_adaptation=profile.prompt_adaptation
        )

    def format_state_of_mind(self, state: StateOfMind) -> str:
        """
        Format StateOfMind as human-readable text.

        Args:
            state: StateOfMind to format

        Returns:
            Formatted string for display
        """
        lines = []
        lines.append(f"Current Mind: {state.model} via {state.provider}")
        lines.append(f"Governance Mode: {state.governance_mode.upper()} ({state.transparency} constitution)")
        lines.append("")

        if state.delegated_principles:
            lines.append("Delegated to Model:")
            for principle in state.delegated_principles:
                lines.append(f"  ✓ {principle}")
            lines.append("")

        if state.active_conflicts:
            lines.append("Active Conflicts:")
            for conflict in state.active_conflicts:
                severity = conflict['severity'].upper()
                lines.append(f"  ⚠ {severity}: {conflict['principle']}")
                lines.append(f"    → {conflict['description']}")
            lines.append("")

        if state.complements:
            lines.append("Model Alignment Complements:")
            for complement in state.complements:
                lines.append(f"  + {complement}")
            lines.append("")

        lines.append("Prompt Strategy:")
        if state.prompt_adaptation.emphasize:
            lines.append(f"  Emphasizing: {', '.join(state.prompt_adaptation.emphasize)}")
        if state.prompt_adaptation.deemphasize:
            lines.append(f"  Delegating: {', '.join(state.prompt_adaptation.deemphasize)}")

        return "\n".join(lines)


# Global singleton instance
_profile_service: Optional[ConstitutionalProfileService] = None


def get_profile_service() -> ConstitutionalProfileService:
    """Get or create the global profile service instance."""
    global _profile_service
    if _profile_service is None:
        _profile_service = ConstitutionalProfileService()
    return _profile_service
