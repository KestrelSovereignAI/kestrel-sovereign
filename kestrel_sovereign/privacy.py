"""Privacy modes — sovereign-side presets and configuration.

The SDK has a database privacy enum for storage-engine routing. Sovereign keeps
its own chat/agent privacy enum because these presets carry sovereign-specific
policy dimensions (`processing`, `sharing`, `assurance`, `audit`,
`computer_access`) that the SDK deliberately doesn't model.

The `to_config` / `from_config` instance/class methods that lived on the
old enum are now module-level functions: `privacy_mode_to_config()` and
`privacy_config_to_mode()`. Same behavior, different call site.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Optional, Union

__all__ = [
    "PrivacyMode",
    "PrivacyConfig",
    "PRIVACY_PRESETS",
    "get_privacy_preset",
    "privacy_mode_to_config",
    "privacy_config_to_mode",
]


class PrivacyMode(Enum):
    """Named sovereign privacy presets."""

    EPHEMERAL = "ephemeral"
    ISOLATED = "isolated"
    ANONYMOUS = "anonymous"
    NORMAL = "normal"
    PUBLIC = "public"
    DEIDENTIFIED = "deidentified"


@dataclass
class PrivacyConfig:
    """
    Privacy configuration using independent flags.

    Orthogonal concerns:
    - storage: How/whether data is persisted
    - processing: Where inference/processing may happen
    - sharing: Whether content remains private or may be shared/exported
    - assurance: The privacy assurance level backing the preset
    - audit: Whether an audit/evidence artifact is required
    - computer_access: Whether the agent may touch the host machine
      (read/write files, run shell). Always defaults False; never inherited
      from a preset. The ComputerUseFeature checks this flag on every call.

    `llm_location` and `shareable` are legacy aliases retained for older
    callers. New code should use `processing` and `sharing`.
    """
    storage: Literal[
        "none", "temp", "pii_redacted", "deidentified", "full", "scrubbed"
    ] = "full"
    processing: Literal["local", "trusted", "cloud"] = "cloud"
    sharing: Literal["private", "research", "public"] = "private"
    assurance: Literal[
        "none", "pii_redacted", "safe_harbor", "expert_determination"
    ] = "none"
    audit: Literal["optional", "required"] = "optional"
    computer_access: bool = False
    llm_location: Optional[Literal["local", "cloud"]] = None
    shareable: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.storage == "scrubbed":
            self.storage = "pii_redacted"
        if self.storage == "pii_redacted" and self.assurance == "none":
            self.assurance = "pii_redacted"
        if self.llm_location is not None:
            self.processing = self.llm_location
        else:
            self.llm_location = "cloud" if self.processing == "cloud" else "local"
        if self.shareable is not None:
            self.sharing = "public" if self.shareable else "private"
        else:
            self.shareable = self.sharing == "public"
        if self.storage == "deidentified":
            self.processing = "trusted"
            self.sharing = "research"
            self.llm_location = "local"
            self.shareable = False
            if self.assurance == "none":
                self.assurance = "safe_harbor"
            if self.audit == "optional":
                self.audit = "required"

    def allows_cloud_llm(self) -> bool:
        """Check if cloud LLM providers are allowed."""
        return self.processing == "cloud"

    def allows_persistent_storage(self) -> bool:
        """Check if persistent storage is allowed."""
        return self.storage in ("pii_redacted", "deidentified", "full")

    def requires_anonymization(self) -> bool:
        """Check if PII redaction is required before persistence."""
        return self.storage == "pii_redacted"

    def requires_deidentification(self) -> bool:
        """Check if HIPAA-style de-identification is required."""
        return self.storage == "deidentified"

    def uses_temp_storage(self) -> bool:
        """Check if using temporary session storage."""
        return self.storage == "temp"

    def is_ephemeral(self) -> bool:
        """Check if no storage at all (pure in-memory)."""
        return self.storage == "none"

    def allows_computer_access(self) -> bool:
        """Check if the agent is permitted to touch the host machine."""
        return self.computer_access

    def requires_audit(self) -> bool:
        """Check if writes/exports require an audit or evidence artifact."""
        return self.audit == "required"


# Named presets for common privacy configurations
PRIVACY_PRESETS: Dict[str, PrivacyConfig] = {
    "ephemeral": PrivacyConfig(
        storage="none", processing="local", sharing="private"
    ),
    "isolated": PrivacyConfig(
        storage="temp", processing="local", sharing="private"
    ),
    "anonymous": PrivacyConfig(
        storage="pii_redacted",
        processing="local",
        sharing="private",
        assurance="pii_redacted",
    ),
    "normal": PrivacyConfig(storage="full", processing="cloud", sharing="private"),
    "public": PrivacyConfig(storage="full", processing="cloud", sharing="public"),
    "deidentified": PrivacyConfig(
        storage="deidentified",
        processing="trusted",
        sharing="research",
        assurance="safe_harbor",
        audit="required",
    ),
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
        processing=preset.processing,
        sharing=preset.sharing,
        assurance=preset.assurance,
        audit=preset.audit,
        computer_access=preset.computer_access,
    )


def privacy_mode_to_config(mode: Union[PrivacyMode, str]) -> PrivacyConfig:
    """Convert a `PrivacyMode` (or its string value) to `PrivacyConfig`.
    """
    if isinstance(mode, PrivacyMode):
        return get_privacy_preset(mode.value)
    if isinstance(mode, str):
        return get_privacy_preset(mode)
    raise TypeError(f"Expected PrivacyMode or str, got {type(mode)}")


def privacy_config_to_mode(config: PrivacyConfig) -> PrivacyMode:
    """Find the closest matching mode for a `PrivacyConfig`.

    ``computer_access`` is intentionally excluded from the match — it is an
    orthogonal capability flag, not part of preset identity. Falls back to
    ``NORMAL`` when no preset matches exactly.
    """
    for name, preset in PRIVACY_PRESETS.items():
        if (config.storage == preset.storage and
            config.processing == preset.processing and
            config.sharing == preset.sharing and
            config.assurance == preset.assurance and
            config.audit == preset.audit):
            return PrivacyMode(name)
    return PrivacyMode.NORMAL
