from enum import Enum
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Union


@dataclass
class PrivacyConfig:
    """
    Privacy configuration using independent flags.

    This replaces the old PrivacyMode enum with orthogonal concerns:
    - storage: How/whether data is persisted
    - llm_location: Whether cloud LLMs are allowed
    - shareable: Whether content can be exported/shared
    - computer_access: Whether the agent may touch the host machine
      (read/write files, run shell). Always defaults False; never inherited
      from a preset. The ComputerUseFeature checks this flag on every call.
    """
    storage: Literal["none", "temp", "scrubbed", "full"] = "full"
    llm_location: Literal["local", "cloud"] = "cloud"
    shareable: bool = False
    computer_access: bool = False

    def allows_cloud_llm(self) -> bool:
        """Check if cloud LLM providers are allowed."""
        return self.llm_location == "cloud"

    def allows_persistent_storage(self) -> bool:
        """Check if persistent storage is allowed."""
        return self.storage in ("scrubbed", "full")

    def requires_anonymization(self) -> bool:
        """Check if PII scrubbing is required."""
        return self.storage == "scrubbed"

    def uses_temp_storage(self) -> bool:
        """Check if using temporary session storage."""
        return self.storage == "temp"

    def is_ephemeral(self) -> bool:
        """Check if no storage at all (pure in-memory)."""
        return self.storage == "none"

    def allows_computer_access(self) -> bool:
        """Check if the agent is permitted to touch the host machine."""
        return self.computer_access


# Named presets for common privacy configurations
PRIVACY_PRESETS: Dict[str, PrivacyConfig] = {
    "ephemeral": PrivacyConfig(storage="none", llm_location="local", shareable=False),
    "isolated": PrivacyConfig(storage="temp", llm_location="local", shareable=False),
    "anonymous": PrivacyConfig(storage="scrubbed", llm_location="cloud", shareable=False),
    "normal": PrivacyConfig(storage="full", llm_location="cloud", shareable=False),
    "public": PrivacyConfig(storage="full", llm_location="cloud", shareable=True),
}


def get_privacy_preset(name: str) -> PrivacyConfig:
    """Get a privacy config preset by name."""
    if name not in PRIVACY_PRESETS:
        raise ValueError(f"Unknown privacy preset: {name}. Valid presets: {list(PRIVACY_PRESETS.keys())}")
    # Return a copy to prevent mutation of the original.
    # computer_access is intentionally NOT included in presets — it must be
    # opted into explicitly by setting the flag after preset construction.
    preset = PRIVACY_PRESETS[name]
    return PrivacyConfig(
        storage=preset.storage,
        llm_location=preset.llm_location,
        shareable=preset.shareable,
        computer_access=preset.computer_access,
    )


# === Backward Compatibility ===
# Keep PrivacyMode enum for existing code, but it now maps to PrivacyConfig

class PrivacyMode(Enum):
    """
    Privacy modes for agent conversations.
    
    DEPRECATED: Use PrivacyConfig directly for new code.
    This enum is maintained for backward compatibility.
    """
    EPHEMERAL = "ephemeral"     # Nothing stored, local LLM only
    ISOLATED = "isolated"       # Temporary session storage, local LLM only
    ANONYMOUS = "anonymous"     # Scrubbed storage (PII removed), cloud LLM allowed
    NORMAL = "normal"           # Standard persistent storage
    PUBLIC = "public"           # Shareable and exportable
    
    def to_config(self) -> PrivacyConfig:
        """Convert this mode to the equivalent PrivacyConfig."""
        return get_privacy_preset(self.value)
    
    @classmethod
    def from_config(cls, config: PrivacyConfig) -> "PrivacyMode":
        """Find the closest matching mode for a config (for backward compat).

        ``computer_access`` is intentionally excluded from the match — it is
        an orthogonal capability flag, not part of preset identity.
        """
        # Find exact match first
        for name, preset in PRIVACY_PRESETS.items():
            if (config.storage == preset.storage and
                config.llm_location == preset.llm_location and
                config.shareable == preset.shareable):
                return cls(name)
        # No exact match - return NORMAL as default
        return cls.NORMAL


def privacy_mode_to_config(mode: Union[PrivacyMode, str]) -> PrivacyConfig:
    """Convert a PrivacyMode or preset name to PrivacyConfig."""
    if isinstance(mode, PrivacyMode):
        return mode.to_config()
    elif isinstance(mode, str):
        return get_privacy_preset(mode)
    else:
        raise TypeError(f"Expected PrivacyMode or str, got {type(mode)}")