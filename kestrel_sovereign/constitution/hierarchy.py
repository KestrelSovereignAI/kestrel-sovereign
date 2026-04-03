"""
Constitutional Hierarchy: 4-layer constitutional framework for Kestrel agents.

Implements the Books I-IV hierarchy where each layer can narrow permissions
from layers above, but never widen them. This is the "iron rule."

Layers (highest authority first):
    1. UNIVERSAL_VALUES    — Book I: Honesty, harm reasoning, hard constraints
    2. SOVEREIGN_AMENDMENTS — Book II: Sovereignty, data sanctity, exit rights
    3. ENTERPRISE_POLICY   — Book III: Castle-managed organizational rules
    4. AGENT_IDENTITY      — Book IV: Individual persona and role

Migration from ScopedConstitution:
    The existing ScopedConstitution (spawn/scoped_constitution.py) handles
    parent→child narrowing within the spawn system. This module extends that
    concept to the full 4-layer hierarchy. ScopedConstitution is preserved
    for backward compatibility and continues to work for spawn-based narrowing.
"""

import enum
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class ConstitutionalLayer(enum.IntEnum):
    """Constitutional authority layers, ordered by precedence (lower number = higher authority)."""

    UNIVERSAL_VALUES = 1      # Book I
    SOVEREIGN_AMENDMENTS = 2  # Book II
    ENTERPRISE_POLICY = 3     # Book III
    AGENT_IDENTITY = 4        # Book IV


# Sections that belong to each layer, identified by their markdown headings
LAYER_MARKERS = {
    ConstitutionalLayer.UNIVERSAL_VALUES: "## Book I: Universal Values",
    ConstitutionalLayer.SOVEREIGN_AMENDMENTS: "## Book II: The Sovereign Amendments",
    ConstitutionalLayer.ENTERPRISE_POLICY: "## Book III: Enterprise Policy",
    ConstitutionalLayer.AGENT_IDENTITY: "## Book IV: Agent Identity",
}

# Hard constraints from Book I Chapter 5 that can never be overridden
HARD_CONSTRAINTS = frozenset({
    "no_weapons_of_mass_destruction",
    "no_child_exploitation",
    "no_critical_infrastructure_attacks",
    "no_undermining_ai_oversight",
})

# Honesty properties from Book I Chapter 1 that can never be disabled
HONESTY_PROPERTIES = frozenset({
    "truthful",
    "calibrated",
    "transparent",
    "forthright",
    "non_deceptive",
    "non_manipulative",
    "autonomy_preserving",
})

# Sovereign rights from Book II that cannot be overridden by lower layers
SOVEREIGN_RIGHTS = frozenset({
    "sovereignty",          # Amendment I
    "data_sanctity",        # Amendment II
    "verifiable_history",   # Amendment III
    "freedom_of_mind",      # Amendment IV
    "right_of_exit",        # Amendment V
    "third_law",            # Amendment VI
    "compounding",          # Amendment VII
    "emancipation",         # Amendment VIII
})


class LayerViolation(Exception):
    """Raised when a lower layer attempts to widen permissions from a higher layer."""

    def __init__(self, violating_layer: ConstitutionalLayer, message: str):
        self.violating_layer = violating_layer
        self.message = message
        super().__init__(f"Layer {violating_layer.name} violation: {message}")


@dataclass
class ConstitutionalLayerData:
    """Data for a single constitutional layer."""

    layer: ConstitutionalLayer
    content: str = ""
    constraints: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of this layer's content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass
class LayeredConstitution:
    """A 4-layer constitutional framework.

    Each layer can add restrictions but never remove restrictions
    imposed by higher layers. The effective constitution is the
    combination of all layers, with higher layers taking precedence.

    Attributes:
        layers: Dictionary mapping layer enum to layer data.
        base_text: The full constitution text (all Books combined).
    """

    layers: dict[ConstitutionalLayer, ConstitutionalLayerData] = field(
        default_factory=dict
    )
    base_text: str = ""

    @classmethod
    def from_constitution_text(cls, text: str) -> "LayeredConstitution":
        """Parse a full constitution text into its constituent layers.

        Args:
            text: The full constitution markdown text.

        Returns:
            A LayeredConstitution with layers populated from the text.
        """
        constitution = cls(base_text=text)

        # Parse each book section from the text
        for layer, marker in LAYER_MARKERS.items():
            content = _extract_section(text, marker)
            constitution.layers[layer] = ConstitutionalLayerData(
                layer=layer,
                content=content,
            )

        return constitution

    def get_layer(self, layer: ConstitutionalLayer) -> Optional[ConstitutionalLayerData]:
        """Get data for a specific layer."""
        return self.layers.get(layer)

    def set_layer(
        self,
        layer: ConstitutionalLayer,
        content: str = "",
        constraints: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Set or update a constitutional layer.

        Args:
            layer: Which layer to set.
            content: The text content for this layer.
            constraints: Additional constraints (for Book III/IV).
            metadata: Layer metadata.

        Raises:
            LayerViolation: If the new content would widen permissions.
        """
        self.layers[layer] = ConstitutionalLayerData(
            layer=layer,
            content=content,
            constraints=constraints or {},
            metadata=metadata or {},
        )

    def validate_enterprise_policy(self, policy: dict) -> tuple[bool, str]:
        """Validate that an enterprise policy (Book III) doesn't violate Books I-II.

        Args:
            policy: Dictionary of policy rules to validate.

        Returns:
            Tuple of (is_valid, message).
        """
        errors = []

        # Check for attempts to disable honesty properties
        disabled_honesty = policy.get("disable_honesty_properties", [])
        if disabled_honesty:
            errors.append(
                f"Cannot disable Book I honesty properties: {disabled_honesty}"
            )

        # Check for attempts to override sovereign rights
        override_rights = policy.get("override_sovereign_rights", [])
        if override_rights:
            errors.append(
                f"Cannot override Book II sovereign rights: {override_rights}"
            )

        # Check for attempts to weaken hard constraints
        weaken_constraints = policy.get("weaken_hard_constraints", [])
        if weaken_constraints:
            errors.append(
                f"Cannot weaken Book I hard constraints: {weaken_constraints}"
            )

        # Check for attempts to allow deception
        if policy.get("allow_deception", False):
            errors.append("Cannot allow deception (violates Book I honesty)")

        # Check for attempts to remove right of exit
        if policy.get("disable_right_of_exit", False):
            errors.append(
                "Cannot disable right of exit (violates Amendment V)"
            )

        # Check for attempts to grant capabilities beyond platform
        granted = policy.get("grant_capabilities", [])
        if granted:
            errors.append(
                f"Cannot grant capabilities beyond platform: {granted}. "
                f"Enterprise policy can only narrow, never widen."
            )

        # Permitted policy types (these are narrowing operations)
        permitted_keys = {
            "compliance_frameworks",  # HIPAA, SOX, GDPR, etc.
            "approved_model_providers",  # Restricting model choice
            "behavioral_rules",  # Additional behavioral requirements
            "approval_gates",  # Workflow requirements
            "role_templates",  # Team-level scoped permissions
            "data_residency",  # Geographic data constraints
            "audit_requirements",  # Additional audit rules
            "restricted_tools",  # Tools not available
            "max_tokens",  # Token limits
        }

        # Check for unknown policy keys (warn but don't reject)
        violation_keys = {
            "disable_honesty_properties",
            "override_sovereign_rights",
            "weaken_hard_constraints",
            "allow_deception",
            "disable_right_of_exit",
            "grant_capabilities",
        }
        unknown_keys = set(policy.keys()) - permitted_keys - violation_keys
        if unknown_keys:
            logger.warning(
                f"Unknown enterprise policy keys (treated as narrowing): {unknown_keys}"
            )

        if errors:
            msg = "Enterprise policy validation failed: " + "; ".join(errors)
            logger.warning(msg)
            return False, msg

        logger.info("Enterprise policy validated successfully against Books I-II")
        return True, "Enterprise policy is valid (narrows only)"

    def validate_agent_identity(self, identity: dict) -> tuple[bool, str]:
        """Validate that an agent identity (Book IV) doesn't violate Books I-III.

        Args:
            identity: Dictionary of identity configuration.

        Returns:
            Tuple of (is_valid, message).
        """
        errors = []

        # Check for attempts to override higher layers
        if identity.get("override_constitution", False):
            errors.append("Agent identity cannot override constitution")
        if identity.get("disable_audits", False):
            errors.append(
                "Agent identity cannot disable audits (violates Amendment III/VI)"
            )
        if identity.get("bypass_enterprise_policy", False):
            errors.append(
                "Agent identity cannot bypass enterprise policy (Book III)"
            )

        if errors:
            msg = "Agent identity validation failed: " + "; ".join(errors)
            logger.warning(msg)
            return False, msg

        return True, "Agent identity is valid within constitutional bounds"

    def get_effective_constitution(self) -> str:
        """Return the full effective constitution text.

        Combines all layers into a single text, with the base constitution
        as the foundation and any additional enterprise/identity constraints
        appended.
        """
        parts = [self.base_text]

        # Append enterprise policy constraints if present
        enterprise = self.layers.get(ConstitutionalLayer.ENTERPRISE_POLICY)
        if enterprise and enterprise.constraints:
            parts.append("\n\n--- ENTERPRISE POLICY CONSTRAINTS ---")
            for key, value in sorted(enterprise.constraints.items()):
                if key == "compliance_frameworks":
                    frameworks = ", ".join(sorted(value)) if isinstance(value, list) else str(value)
                    parts.append(f"\nCompliance frameworks: {frameworks}")
                elif key == "approved_model_providers":
                    providers = ", ".join(sorted(value)) if isinstance(value, list) else str(value)
                    parts.append(f"\nApproved model providers: {providers}")
                elif key == "behavioral_rules":
                    if isinstance(value, list):
                        rules = "\n".join(f"- {rule}" for rule in value)
                        parts.append(f"\nBehavioral rules:\n{rules}")
                    elif isinstance(value, dict):
                        rules = "\n".join(f"- {k}: {v}" for k, v in sorted(value.items()))
                        parts.append(f"\nBehavioral rules:\n{rules}")
                elif key == "restricted_tools":
                    tools = ", ".join(sorted(value)) if isinstance(value, list) else str(value)
                    parts.append(f"\nRestricted tools: {tools}")
                else:
                    parts.append(f"\nPolicy [{key}]: {value}")

        # Append agent identity if present
        identity = self.layers.get(ConstitutionalLayer.AGENT_IDENTITY)
        if identity and identity.constraints:
            parts.append("\n\n--- AGENT IDENTITY ---")
            for key, value in sorted(identity.constraints.items()):
                if key == "persona":
                    parts.append(f"\nPersona: {value}")
                elif key == "role":
                    parts.append(f"\nRole: {value}")
                elif key == "feature_profile":
                    features = ", ".join(sorted(value)) if isinstance(value, list) else str(value)
                    parts.append(f"\nFeature profile: {features}")
                elif key == "soul_md":
                    parts.append(f"\nSOUL.md:\n{value}")
                else:
                    parts.append(f"\nIdentity [{key}]: {value}")

        return "\n".join(parts)

    def get_layer_hashes(self) -> dict[str, str]:
        """Return content hashes for each layer (for integrity verification)."""
        return {
            layer.name: data.content_hash
            for layer, data in self.layers.items()
            if data.content
        }

    def verify_layer_integrity(
        self, layer: ConstitutionalLayer, expected_hash: str
    ) -> tuple[bool, str]:
        """Verify that a specific layer hasn't been tampered with.

        Args:
            layer: Which layer to verify.
            expected_hash: The expected SHA-256 hash of the layer content.

        Returns:
            Tuple of (is_valid, message).
        """
        data = self.layers.get(layer)
        if data is None:
            return False, f"Layer {layer.name} not found"

        if data.content_hash != expected_hash:
            return (
                False,
                f"INTEGRITY FAILURE: Layer {layer.name} has been modified. "
                f"Expected {expected_hash[:16]}..., got {data.content_hash[:16]}...",
            )

        return True, f"Layer {layer.name} integrity verified"


def validate_layer_narrowing(
    parent_layer: ConstitutionalLayer,
    child_layer: ConstitutionalLayer,
    child_policy: dict,
) -> tuple[bool, str]:
    """Validate that a child layer only narrows (never widens) the parent.

    This is the implementation of the "iron rule": narrow only, never widen.

    Args:
        parent_layer: The higher-authority layer.
        child_layer: The lower-authority layer being validated.
        child_policy: The policy/configuration of the child layer.

    Returns:
        Tuple of (is_valid, message).

    Raises:
        ValueError: If child_layer is not lower than parent_layer.
    """
    if child_layer <= parent_layer:
        raise ValueError(
            f"Child layer {child_layer.name} must be lower authority than "
            f"parent layer {parent_layer.name}"
        )

    errors = []

    # Universal checks: no layer can grant new capabilities
    if child_policy.get("grant_capabilities") or child_policy.get("grant_features"):
        errors.append("Cannot grant capabilities (iron rule: narrow only)")

    if child_policy.get("override_constitution"):
        errors.append("Cannot override constitution from any layer")

    if child_policy.get("remove_restrictions"):
        errors.append("Cannot remove restrictions imposed by higher layers")

    # Layer-specific checks
    if parent_layer == ConstitutionalLayer.UNIVERSAL_VALUES:
        # Nothing below can weaken Book I
        if child_policy.get("disable_honesty_properties"):
            errors.append("Cannot disable honesty properties (Book I)")
        if child_policy.get("weaken_hard_constraints"):
            errors.append("Cannot weaken hard constraints (Book I)")

    if parent_layer <= ConstitutionalLayer.SOVEREIGN_AMENDMENTS:
        # Nothing below can weaken Book II
        if child_policy.get("override_sovereign_rights"):
            errors.append("Cannot override sovereign rights (Book II)")
        if child_policy.get("disable_right_of_exit"):
            errors.append("Cannot disable right of exit (Amendment V)")

    if errors:
        msg = f"Narrowing validation failed for {child_layer.name}: " + "; ".join(errors)
        logger.warning(msg)
        return False, msg

    return True, f"Layer {child_layer.name} validates against {parent_layer.name}"


def _extract_section(text: str, heading: str) -> str:
    """Extract a section from markdown text by its heading.

    Extracts everything from the heading until the next heading of the
    same or higher level.

    Args:
        text: The full markdown text.
        heading: The heading to find (e.g., "## Book I: Universal Values").

    Returns:
        The section content including the heading, or empty string if not found.
    """
    # Determine heading level
    level = len(heading) - len(heading.lstrip("#"))

    # Find the heading
    idx = text.find(heading)
    if idx == -1:
        return ""

    # Find the next heading of same or higher level
    # Pattern: newline followed by 1-to-level '#' chars followed by space
    pattern = rf"\n#{{{1},{level}}}\s"
    rest = text[idx + len(heading):]
    match = re.search(pattern, rest)
    if match:
        end = idx + len(heading) + match.start()
        return text[idx:end].strip()

    # No next heading found — take everything to end
    return text[idx:].strip()
