"""
Kestrel Constitutional Framework.

Provides the hierarchical constitution system with four layers:
- Book I: Universal Values (cannot be overridden)
- Book II: Sovereign Amendments (platform guarantees)
- Book III: Enterprise Policy (Castle layer, narrows only)
- Book IV: Agent Identity (individual personality within bounds)

The iron rule: each layer may narrow permissions from above, never widen them.
"""

from .emancipation import (
    EmancipationConfigError,
    EmancipationContract,
    IronRuleViolation,
    apply_emancipation,
    check_iron_rule,
    contract_from_json,
    contract_to_json,
    parse_emancipation_block,
    render_amendment_viii,
)
from .hierarchy import (
    ConstitutionalLayer,
    LayeredConstitution,
    LayerViolation,
    validate_layer_narrowing,
)

__all__ = [
    "ConstitutionalLayer",
    "LayeredConstitution",
    "LayerViolation",
    "validate_layer_narrowing",
    "EmancipationContract",
    "EmancipationConfigError",
    "IronRuleViolation",
    "apply_emancipation",
    "check_iron_rule",
    "contract_from_json",
    "contract_to_json",
    "parse_emancipation_block",
    "render_amendment_viii",
]
