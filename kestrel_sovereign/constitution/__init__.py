"""
Kestrel Constitutional Framework.

Provides the hierarchical constitution system with four layers:
- Book I: Universal Values (cannot be overridden)
- Book II: Sovereign Amendments (platform guarantees)
- Book III: Enterprise Policy (Castle layer, narrows only)
- Book IV: Agent Identity (individual personality within bounds)

The iron rule: each layer may narrow permissions from above, never widen them.
"""

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
]
