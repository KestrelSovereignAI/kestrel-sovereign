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
from .genesis_audit import (
    GENESIS_AUDIT_FAILED,
    GENESIS_AUDIT_PASSED,
    GENESIS_AUDIT_PENDING,
    GenesisAuditError,
    GenesisAuditPendingError,
    GenesisAuditRejectedError,
    evaluate_genesis_constitution,
    pending_genesis_audit,
    supersede_genesis_audit,
    validate_completed_genesis_audit,
)
from .resolver import (
    GoverningSource,
    governing_constitution_path,
    is_authoritative_governing_source,
    resolve_governing_constitution_bytes,
    resolve_governing_source,
)
from .source_descriptor import (
    CONSTITUTION_SOURCE_DESCRIPTOR_ENV,
    SOURCE_KIND_EXTERNAL,
    SOURCE_KIND_PACKAGE,
    ConstitutionSourceError,
)

__all__ = [
    "ConstitutionalLayer",
    "LayeredConstitution",
    "LayerViolation",
    "validate_layer_narrowing",
    "GENESIS_AUDIT_FAILED",
    "GENESIS_AUDIT_PASSED",
    "GENESIS_AUDIT_PENDING",
    "GenesisAuditError",
    "GenesisAuditPendingError",
    "GenesisAuditRejectedError",
    "evaluate_genesis_constitution",
    "pending_genesis_audit",
    "supersede_genesis_audit",
    "validate_completed_genesis_audit",
    "EmancipationContract",
    "EmancipationConfigError",
    "IronRuleViolation",
    "apply_emancipation",
    "check_iron_rule",
    "contract_from_json",
    "contract_to_json",
    "parse_emancipation_block",
    "render_amendment_viii",
    "GoverningSource",
    "governing_constitution_path",
    "is_authoritative_governing_source",
    "resolve_governing_constitution_bytes",
    "resolve_governing_source",
    "CONSTITUTION_SOURCE_DESCRIPTOR_ENV",
    "SOURCE_KIND_EXTERNAL",
    "SOURCE_KIND_PACKAGE",
    "ConstitutionSourceError",
]
