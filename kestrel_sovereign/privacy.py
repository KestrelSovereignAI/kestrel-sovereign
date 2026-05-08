"""Privacy modes — sovereign-side configuration with SDK-canonical enum.

`PrivacyMode` is re-exported from `kestrel_sdk.storage.database` so feature
packages and sovereign share **one** enum identity (no parallel copies, no
broken `isinstance` checks at the seam).

`PrivacyConfig` and the preset dict stay sovereign-private — they carry
sovereign-specific flags (`computer_access`, `llm_location`) that the SDK
deliberately doesn't model.

The `to_config` / `from_config` instance/class methods that lived on the
old enum are now module-level functions: `privacy_mode_to_config()` and
`privacy_config_to_mode()`. Same behavior, different call site.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Union

from kestrel_sdk.storage.database import PrivacyMode

__all__ = [
    "PrivacyMode",
    "PrivacyConfig",
    "PRIVACY_PRESETS",
    "get_privacy_preset",
    "privacy_mode_to_config",
    "privacy_config_to_mode",
]


@dataclass
class PrivacyConfig:
    """
    Privacy configuration using independent flags.

    Orthogonal concerns:
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


def privacy_mode_to_config(mode: Union[PrivacyMode, str]) -> PrivacyConfig:
    """Convert a `PrivacyMode` (or its string value) to `PrivacyConfig`.

    Replaces the `PrivacyMode.to_config()` instance method that lived on
    the old sovereign-local enum. The SDK enum carries values only; this
    helper handles the sovereign-specific config fan-out.
    """
    if isinstance(mode, PrivacyMode):
        return get_privacy_preset(mode.value)
    if isinstance(mode, str):
        return get_privacy_preset(mode)
    raise TypeError(f"Expected PrivacyMode or str, got {type(mode)}")


def privacy_config_to_mode(config: PrivacyConfig) -> PrivacyMode:
    """Find the closest matching mode for a `PrivacyConfig`.

    Replaces the `PrivacyMode.from_config()` classmethod. ``computer_access``
    is intentionally excluded from the match — it is an orthogonal capability
    flag, not part of preset identity. Falls back to ``NORMAL`` when no
    preset matches exactly.
    """
    for name, preset in PRIVACY_PRESETS.items():
        if (config.storage == preset.storage and
            config.llm_location == preset.llm_location and
            config.shareable == preset.shareable):
            return PrivacyMode(name)
    return PrivacyMode.NORMAL
