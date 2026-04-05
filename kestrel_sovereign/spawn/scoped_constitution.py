"""
Scoped Constitution: Constitutional narrowing for spawned agents.

Ensures child agents can have fewer capabilities than their parent,
but never more. Wraps a base constitution with additional constraints
from a SpawnMandate.

This module implements spawn-level narrowing. For the full 4-layer
constitutional hierarchy (Books I-IV), see kestrel_sovereign.constitution.hierarchy.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Well-known constraint types that restrict capabilities
RESTRICTION_CONSTRAINTS = frozenset({
    "read-only",
    "web-search-only",
    "no-wallet",
    "no-spawn",
    "no-memory-write",
    "no-file-access",
    "no-network",
    "no-tools",
})


@dataclass
class ScopedConstitution:
    """Wraps a base constitution with additional constraints from a SpawnMandate.

    The scoped constitution enforces that child agents can only have
    restrictions applied — never grants of new capabilities beyond
    what the parent possesses.

    This class handles spawn-level narrowing (parent→child). For the full
    4-layer hierarchy (Books I-IV: Universal Values → Sovereign Amendments →
    Enterprise Policy → Agent Identity), see LayeredConstitution in
    kestrel_sovereign.constitution.hierarchy.

    Attributes:
        base_constitution: The full text of the parent's constitution.
        additional_constraints: Dict of constraint rules from the SpawnMandate.
        features_allowed: List of feature names the child is permitted to use.
        parent_features: Set of feature names the parent agent has.
        enterprise_policy: Optional enterprise policy constraints (Book III).
        agent_identity: Optional agent identity constraints (Book IV).
    """

    base_constitution: str
    additional_constraints: dict = field(default_factory=dict)
    features_allowed: list[str] = field(default_factory=list)
    parent_features: set[str] = field(default_factory=set)
    enterprise_policy: dict = field(default_factory=dict)
    agent_identity: dict = field(default_factory=dict)

    def validate_constraints(self) -> tuple[bool, str]:
        """Validate that all constraints are restrictions, not capability grants.

        Validates spawn-level constraints and, if present, enterprise policy
        and agent identity against the constitutional hierarchy (Books I-IV).

        Returns:
            Tuple of (is_valid, message).
        """
        errors = []

        # Validate features_allowed is a subset of parent features
        if self.features_allowed and self.parent_features:
            child_features = set(self.features_allowed)
            extra_features = child_features - self.parent_features
            if extra_features:
                errors.append(
                    f"Child requests features not available to parent: "
                    f"{sorted(extra_features)}"
                )

        # Validate additional_constraints don't grant new capabilities
        for key, value in self.additional_constraints.items():
            if key == "grant_features":
                errors.append(
                    f"Constraint 'grant_features' attempts to grant new "
                    f"capabilities — this is not allowed"
                )
            elif key == "override_constitution":
                errors.append(
                    f"Constraint 'override_constitution' attempts to weaken "
                    f"base constitution — this is not allowed"
                )
            elif key == "remove_restrictions":
                errors.append(
                    f"Constraint 'remove_restrictions' attempts to remove "
                    f"existing restrictions — this is not allowed"
                )
            elif key == "behavioral_rules":
                # Behavioral rules are allowed — they add restrictions
                if not isinstance(value, (list, dict)):
                    errors.append(
                        f"Constraint 'behavioral_rules' must be a list or dict, "
                        f"got {type(value).__name__}"
                    )
            elif key == "restricted_tools":
                # Restricting tools is allowed
                if not isinstance(value, list):
                    errors.append(
                        f"Constraint 'restricted_tools' must be a list, "
                        f"got {type(value).__name__}"
                    )
            elif key == "max_tokens":
                # Token limits are restrictions
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(
                        f"Constraint 'max_tokens' must be a non-negative number"
                    )

        # Validate enterprise policy against Books I-II if present
        if self.enterprise_policy:
            from kestrel_sovereign.constitution.hierarchy import LayeredConstitution
            lc = LayeredConstitution()
            is_valid, msg = lc.validate_enterprise_policy(self.enterprise_policy)
            if not is_valid:
                errors.append(msg)

        # Validate agent identity against all higher layers if present
        if self.agent_identity:
            from kestrel_sovereign.constitution.hierarchy import LayeredConstitution
            lc = LayeredConstitution()
            is_valid, msg = lc.validate_agent_identity(self.agent_identity)
            if not is_valid:
                errors.append(msg)

        if errors:
            msg = "Constraint validation failed: " + "; ".join(errors)
            logger.warning(msg)
            return False, msg

        logger.info("Scoped constitution constraints validated successfully")
        return True, "All constraints are valid restrictions"

    def get_effective_constitution(self) -> str:
        """Return the full constitution text including scoped constraints.

        The effective constitution is the base constitution plus any
        additional behavioral rules and restrictions.
        """
        parts = [self.base_constitution]

        if self.additional_constraints or self.features_allowed:
            parts.append("\n\n--- SPAWN MANDATE CONSTRAINTS ---")

        if self.features_allowed:
            parts.append(
                f"\nAllowed features: {', '.join(sorted(self.features_allowed))}"
            )

        for key, value in sorted(self.additional_constraints.items()):
            if key == "behavioral_rules":
                if isinstance(value, list):
                    rules_text = "\n".join(f"- {rule}" for rule in value)
                    parts.append(f"\nAdditional behavioral rules:\n{rules_text}")
                elif isinstance(value, dict):
                    rules_text = "\n".join(
                        f"- {k}: {v}" for k, v in sorted(value.items())
                    )
                    parts.append(f"\nAdditional behavioral rules:\n{rules_text}")
            elif key == "restricted_tools":
                parts.append(
                    f"\nRestricted tools (not available): "
                    f"{', '.join(sorted(value))}"
                )
            elif key in RESTRICTION_CONSTRAINTS:
                parts.append(f"\nRestriction active: {key}")
            elif key == "max_tokens":
                parts.append(f"\nToken limit: {value}")
            else:
                parts.append(f"\nConstraint [{key}]: {value}")

        return "\n".join(parts)

    def verify_integrity(self, current_constraints: dict) -> tuple[bool, str]:
        """Verify that the scoped constraints haven't been tampered with.

        Compares the current constraints against the original mandate constraints.

        Args:
            current_constraints: The constraints to verify against the original.

        Returns:
            Tuple of (is_valid, message).
        """
        if current_constraints != self.additional_constraints:
            return (
                False,
                "INTEGRITY FAILURE: Scoped constitution constraints have been modified"
            )
        return True, "Scoped constitution integrity verified"
