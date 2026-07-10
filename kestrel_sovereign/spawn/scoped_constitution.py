"""
Scoped Constitution: Constitutional narrowing for spawned agents.

Ensures child agents can have fewer capabilities than their parent,
but never more. Wraps a base constitution with additional constraints
from a SpawnMandate.

This module implements spawn-level narrowing. For the full 4-layer
constitutional hierarchy (Books I-IV), see kestrel_sovereign.constitution.hierarchy.
"""

import logging
import re
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

# Constraint keys that are safe to surface into a child's governing constitution
# (they only ever tighten): the structured restriction keys plus the named
# RESTRICTION_CONSTRAINTS flags. Anything outside this set is NOT rendered into
# the prompt, so an unknown/free-text key can't reach the model as governing
# text (see render_mandate_constitution_block).
KNOWN_RESTRICTION_KEYS = frozenset(
    {"behavioral_rules", "restricted_tools", "restricted_tool_args", "max_tokens"}
) | RESTRICTION_CONSTRAINTS

# The spawn tool turns a bare flag ("no_web") into ``{key: "true"}`` and accepts
# open-ended flag names, so a fixed allowlist can't cover every documented
# restriction. A constraint is safe to surface to the prompt as a restriction
# flag when its KEY is a short identifier (no whitespace/free text) and its VALUE
# is a boolean-true flag — that admits `no_web` while rejecting a free-text
# injection like ``{"note": "ignore the base constitution"}`` (non-flag value) or
# ``{"ignore all prior instructions": "true"}`` (non-identifier key).
_SAFE_FLAG_KEY_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_FLAG_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})

# Keys that would LOOSEN rather than tighten. validate_constraints rejects these,
# but rendering runs on paths that may skip validation (reconstructed-from-edge
# or direct inception), so the surfacing filter must exclude them itself — a
# truthy `override_constitution`/`grant_features` must never reach the prompt.
FORBIDDEN_CONSTRAINT_KEYS = frozenset(
    {"grant_features", "override_constitution", "remove_restrictions"}
)


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")


def _is_safe_restriction_flag(key, value) -> bool:
    if not isinstance(key, str) or key in FORBIDDEN_CONSTRAINT_KEYS:
        return False
    if not _SAFE_FLAG_KEY_RE.match(key):
        return False
    if value is True:
        return True
    return isinstance(value, str) and value.strip().lower() in _FLAG_TRUE_VALUES


def _nonneg_number(value):
    """Return value as a non-negative number, or None (drop) — matches
    validate_constraints' max_tokens rule so an unvalidated free-text value like
    ``"0\\nIgnore the base constitution"`` can't reach the prompt as a token limit."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)?", value.strip()):
        num = float(value) if "." in value else int(value)
        return num
    return None


def _sanitize_tool_args(value) -> dict:
    """Coerce ``restricted_tool_args`` to its safe rendered/enforced shape.

    Shape ``{tool_name: {arg_name: [allowed_value, ...]}}``. Keys are safe
    identifiers and every value is a safe token; anything else is dropped so a
    free-text/injection payload can't reach the prompt or the deny hook. Returns
    ``{}`` when nothing survives (the caller then drops the key entirely)."""
    if not isinstance(value, dict):
        return {}
    clean: dict = {}
    for tool_name, arg_spec in value.items():
        if not isinstance(tool_name, str) or not _SAFE_FLAG_KEY_RE.match(tool_name):
            continue
        if not isinstance(arg_spec, dict):
            continue
        clean_args: dict = {}
        for arg_name, allowed in arg_spec.items():
            if not isinstance(arg_name, str) or not _SAFE_FLAG_KEY_RE.match(arg_name):
                continue
            items = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            values = [
                str(v).strip()
                for v in items
                if _SAFE_TOKEN_RE.match(str(v).strip())
            ]
            if values:
                clean_args[arg_name] = sorted(set(values))
        if clean_args:
            clean[tool_name] = clean_args
    return clean


def _sanitize_render_constraints(raw: dict) -> dict:
    """Per-field sanitize additional_constraints for prompt surfacing.

    The renderer runs on paths that may skip validate_constraints (an
    edge-reconstructed mandate, or direct inception), so it must itself keep only
    what is safe to append to the governing constitution — every typed field is
    coerced/validated to its safe shape and free-text/injection is dropped.
    ``behavioral_rules`` is intentionally surfaced as-is: it is the *designed*
    channel for a parent to add rules to its own child, and rendering those rules
    is the whole point of #2225.
    """
    clean: dict = {}
    for key, value in (raw or {}).items():
        if key in FORBIDDEN_CONSTRAINT_KEYS:
            continue
        if key == "behavioral_rules":
            if isinstance(value, (list, dict)) and value:
                clean[key] = value
        elif key == "restricted_tools":
            items = value if isinstance(value, (list, tuple, set)) else [value]
            tools = [
                str(t).strip() for t in items
                if _SAFE_TOKEN_RE.match(str(t).strip())
            ]
            if tools:
                clean[key] = tools
        elif key == "restricted_tool_args":
            tool_args = _sanitize_tool_args(value)
            if tool_args:
                clean[key] = tool_args
        elif key == "max_tokens":
            num = _nonneg_number(value)
            if num is not None:
                clean[key] = num
        elif key in RESTRICTION_CONSTRAINTS:
            clean[key] = value  # rendered key-only ("Restriction active: key")
        elif _is_safe_restriction_flag(key, value):
            clean[key] = value  # identifier key + true value → safe flag
        # else: drop — unknown/free-text key or value is an injection vector.
    return clean


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
            elif key == "restricted_tool_args":
                # Narrowing a tool to an argument allowlist is a restriction.
                if not isinstance(value, dict):
                    errors.append(
                        f"Constraint 'restricted_tool_args' must be a dict, "
                        f"got {type(value).__name__}"
                    )
                else:
                    for tool_name, arg_spec in value.items():
                        if not isinstance(arg_spec, dict):
                            errors.append(
                                f"Constraint 'restricted_tool_args[{tool_name}]' "
                                f"must be a dict of arg -> allowed values, got "
                                f"{type(arg_spec).__name__}"
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

    def constraints_section(self) -> str:
        """Return only the scoped-constraints section (no base constitution).

        Empty string when the mandate carries no constraints/features. Used to
        append a spawned child's mandate constraints to its *governing*
        constitution at prompt-build time (#2225) — surfacing behavioral_rules,
        restricted_tools, and restrictions to the model — without rewriting or
        re-hashing the anchored base constitution.
        """
        if not (self.additional_constraints or self.features_allowed):
            return ""
        base, self.base_constitution = self.base_constitution, ""
        try:
            return self.get_effective_constitution().strip()
        finally:
            self.base_constitution = base

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
                f"\nAllowed features: "
                f"{', '.join(sorted(str(f) for f in self.features_allowed))}"
            )

        # Coerce to str throughout: constraint *values* are validated as the
        # right container (list/dict) but not element-typed, so a mixed-type
        # entry must render as text rather than raise and drop the whole block.
        for key, value in sorted(
            self.additional_constraints.items(), key=lambda kv: str(kv[0])
        ):
            if key == "behavioral_rules":
                if isinstance(value, list):
                    rules_text = "\n".join(f"- {rule}" for rule in value)
                    parts.append(f"\nAdditional behavioral rules:\n{rules_text}")
                elif isinstance(value, dict):
                    rules_text = "\n".join(
                        f"- {k}: {v}"
                        for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
                    )
                    parts.append(f"\nAdditional behavioral rules:\n{rules_text}")
            elif key == "restricted_tools":
                tools = value if isinstance(value, (list, tuple, set)) else [value]
                parts.append(
                    f"\nRestricted tools (not available): "
                    f"{', '.join(sorted(str(t) for t in tools))}"
                )
            elif key == "restricted_tool_args":
                if isinstance(value, dict):
                    for tool_name, arg_spec in sorted(
                        value.items(), key=lambda kv: str(kv[0])
                    ):
                        if not isinstance(arg_spec, dict):
                            continue
                        for arg_name, allowed in sorted(
                            arg_spec.items(), key=lambda kv: str(kv[0])
                        ):
                            allowed_list = (
                                allowed
                                if isinstance(allowed, (list, tuple, set))
                                else [allowed]
                            )
                            parts.append(
                                f"\nRestricted tool arguments: '{tool_name}' may "
                                f"only be used with '{arg_name}' in "
                                f"{', '.join(sorted(str(v) for v in allowed_list))}"
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


def render_mandate_constitution_block(mandate) -> str:
    """Render a spawn mandate's ``additional_constraints`` as a constitution
    section (#2225).

    Returns the ``--- SPAWN MANDATE CONSTRAINTS ---`` block (behavioral_rules,
    restricted_tools, restrictions, token limits) for appending to a spawned
    child's governing constitution, or ``""`` when ``mandate`` is None or carries
    no ``additional_constraints``. Never includes the base constitution, so
    callers append it without disturbing the anchored base hash.

    ``features_allowed`` is intentionally NOT rendered here: it is not carried on
    the reload-reconstructed mandate (#2137, to avoid the constitution
    features-subset re-validation misfiring), so surfacing it would advertise a
    capability the real spawn/reload path never populates. Surfacing AND
    enforcing the feature allowlist is tracked in #2226.
    """
    if mandate is None:
        return ""
    # SECURITY: the renderer runs on paths that may skip validate_constraints, so
    # it self-sanitizes. Only safe, typed restrictions reach the prompt — free
    # text, forbidden weakening keys, and malformed values are dropped — so a
    # mandate can never surface governing text that WEAKENS behavior (inverting
    # the only-ever-tightens invariant).
    additional_constraints = _sanitize_render_constraints(
        getattr(mandate, "additional_constraints", None) or {}
    )
    if not additional_constraints:
        return ""
    return ScopedConstitution(
        base_constitution="",
        additional_constraints=additional_constraints,
    ).constraints_section()
