"""Canonical assertion contracts and offline semantic knowledge resources.

This package is the public surface for Kestrel's dependency-free assertion
value model and its versioned, local semantic-resource registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .assertion import (
    IDENTITY_VERSION,
    IRI_PROFILE,
    LITERAL_PROFILE,
    MAPPING_SCHEMA_VERSION,
    RDF_LANG_STRING,
    XSD_BOOLEAN,
    XSD_DATE,
    XSD_DATETIME,
    XSD_DATETIME_STAMP,
    XSD_DECIMAL,
    XSD_INTEGER,
    XSD_STRING,
    XSD_TIME,
    Assertion,
    AssertionObject,
    AssertionQuery,
    AssertionResult,
    AssertionStatus,
    AssertionTerm,
    AssertionValidationError,
    BlankNode,
    DerivedLineage,
    DirectLineage,
    EpistemicState,
    IRI,
    Instant,
    Lineage,
    Literal,
    LocalIdentifier,
    OntologyRef,
    Resource,
    SourceOccurrence,
    SourceProvenance,
    TemporalInterval,
    Visibility,
    derive_assertion_id,
    identity_preimage,
    lineage_from_mapping,
    normalize_iri,
)
if TYPE_CHECKING:
    from .corpus import (
        CORPUS_SCHEMA_VERSION,
        CorpusCheckpoint,
        CorpusEligibilityDecision,
        CorpusEligibilityReason,
        CorpusValidationStatus,
        GovernedAssertionCorpusService,
        GovernedCorpusBudgetExceeded,
        GovernedCorpusDelta,
        GovernedCorpusError,
        GovernedCorpusExample,
        GovernedCorpusLimits,
        GovernedCorpusObservability,
        GovernedCorpusPolicy,
        GovernedCorpusSnapshot,
        GovernedCorpusStorage,
        GovernedCorpusTombstone,
        GovernedCorpusUnavailable,
    )
    from .rdf_codec import (
        RdfAssertionCodec,
        RdfAssertionReadAdapter,
        RdfBlankNode,
        RdfCapabilityReport,
        RdfCodecConfiguration,
        RdfCodecError,
        RdfDataset,
        RdfImportBudgetError,
        RdfImportDocument,
        RdfImportLimits,
        RdfImportOwnership,
        RdfImportSecurityError,
        RdfIri,
        RdfLiteral,
        RdfOwnershipError,
        RdfProjectionKind,
        RdfTerm,
        RdfTriple,
        RdfTripleTerm,
        RdfTypedQuery,
        Sparql11AssertionReadAdapter,
        Sparql12AssertionReadAdapter,
        UnsupportedRdfCapabilityError,
    )
    from .registry import (
        ArtifactPin,
        ExperimentalCapabilityError,
        KnowledgeRegistryError,
        ResolvedSemanticCapability,
        ResourceKind,
        ResourceRequirement,
        SemanticCapabilityContract,
        SemanticKnowledgeRegistry,
        SemanticResource,
        SemanticVersion,
        StandardsMaturity,
        VersionConstraint,
        get_knowledge_registry,
        load_knowledge_registry,
    )
    from .shacl_validation import (
        DEFAULT_SHACL_WRITE_POLICY,
        GovernedShaclValidationService,
        ShaclCapabilityUnavailable,
        ShaclSnapshotMismatch,
        ShaclValidationError,
        ShaclValidationLimits,
        ShaclValidationReport,
        ShaclWritePolicy,
        ShapeSetReference,
        ValidationFinding,
        ValidationSeverity,
        ValidationSource,
        ValidationState,
        ValidationWriteAction,
    )
    from .capabilities import (
        CapabilitySelection,
        SemanticCapabilityConfigurationError,
        SemanticRuntimeCapabilities,
        semantic_capabilities_from_config,
    )

__all__ = [
    "ArtifactPin",
    "Assertion",
    "AssertionObject",
    "AssertionQuery",
    "AssertionResult",
    "AssertionStatus",
    "AssertionTerm",
    "AssertionValidationError",
    "BlankNode",
    "DerivedLineage",
    "DirectLineage",
    "EpistemicState",
    "ExperimentalCapabilityError",
    "BoundedInferenceService",
    "CORPUS_SCHEMA_VERSION",
    "ClosureState",
    "ClosureStatus",
    "CorpusCheckpoint",
    "CorpusEligibilityDecision",
    "CorpusEligibilityReason",
    "CorpusValidationStatus",
    "DerivationExplanation",
    "IDENTITY_VERSION",
    "IRI",
    "IRI_PROFILE",
    "Instant",
    "InferenceError",
    "InferenceLimits",
    "InferenceProfile",
    "inference_limits_from_config",
    "inference_profile_from_config",
    "validate_inference_profile",
    "InferenceService",
    "KnowledgeRegistryError",
    "LITERAL_PROFILE",
    "Lineage",
    "Literal",
    "LocalIdentifier",
    "MAPPING_SCHEMA_VERSION",
    "MaterializationResult",
    "SemanticMaintenanceError",
    "SemanticMaintenanceLimits",
    "SemanticMaintenanceResult",
    "SemanticMaintenanceService",
    "SemanticMaintenanceStatus",
    "SemanticMaintenanceTrainingReadiness",
    "OntologyRef",
    "RDF_LANG_STRING",
    "RdfAssertionCodec",
    "RdfAssertionReadAdapter",
    "RdfBlankNode",
    "RdfCapabilityReport",
    "RdfCodecConfiguration",
    "RdfCodecError",
    "RdfDataset",
    "RdfImportBudgetError",
    "RdfImportDocument",
    "RdfImportLimits",
    "RdfImportOwnership",
    "RdfImportSecurityError",
    "RdfIri",
    "RdfLiteral",
    "RdfOwnershipError",
    "RdfProjectionKind",
    "RdfTerm",
    "RdfTriple",
    "RdfTripleTerm",
    "RdfTypedQuery",
    "ResolvedSemanticCapability",
    "Resource",
    "ResourceKind",
    "ResourceRequirement",
    "DEFAULT_SHACL_WRITE_POLICY",
    "GovernedShaclValidationService",
    "GovernedAssertionCorpusService",
    "GovernedCorpusBudgetExceeded",
    "GovernedCorpusDelta",
    "GovernedCorpusError",
    "GovernedCorpusExample",
    "GovernedCorpusLimits",
    "GovernedCorpusObservability",
    "GovernedCorpusPolicy",
    "GovernedCorpusSnapshot",
    "GovernedCorpusStorage",
    "GovernedCorpusTombstone",
    "GovernedCorpusUnavailable",
    "ShaclCapabilityUnavailable",
    "ShaclSnapshotMismatch",
    "ShaclValidationError",
    "ShaclValidationLimits",
    "ShaclValidationReport",
    "ShaclWritePolicy",
    "ShapeSetReference",
    "SemanticCapabilityContract",
    "CapabilitySelection",
    "SemanticCapabilityConfigurationError",
    "SemanticRuntimeCapabilities",
    "semantic_capabilities_from_config",
    "SemanticInferenceService",
    "SemanticKnowledgeRegistry",
    "SemanticResource",
    "SemanticVersion",
    "SourceOccurrence",
    "SourceProvenance",
    "Sparql11AssertionReadAdapter",
    "Sparql12AssertionReadAdapter",
    "StandardsMaturity",
    "TemporalInterval",
    "ValidationFinding",
    "ValidationSeverity",
    "ValidationSource",
    "ValidationState",
    "ValidationWriteAction",
    "VersionConstraint",
    "Visibility",
    "XSD_BOOLEAN",
    "XSD_DATE",
    "XSD_DATETIME",
    "XSD_DATETIME_STAMP",
    "XSD_DECIMAL",
    "XSD_INTEGER",
    "XSD_STRING",
    "XSD_TIME",
    "UnsupportedRdfCapabilityError",
    "derive_assertion_id",
    "get_knowledge_registry",
    "identity_preimage",
    "lineage_from_mapping",
    "load_knowledge_registry",
    "normalize_iri",
    "maintenance_limits_from_config",
    "maintenance_allows_prior_verified_snapshot",
]

_REGISTRY_EXPORTS = frozenset(
    {
        "ArtifactPin",
        "ExperimentalCapabilityError",
        "KnowledgeRegistryError",
        "ResolvedSemanticCapability",
        "ResourceKind",
        "ResourceRequirement",
        "SemanticCapabilityContract",
        "SemanticKnowledgeRegistry",
        "SemanticResource",
        "SemanticVersion",
        "StandardsMaturity",
        "VersionConstraint",
        "get_knowledge_registry",
        "load_knowledge_registry",
    }
)

_RDF_CODEC_EXPORTS = frozenset(
    {
        "RdfAssertionCodec",
        "RdfAssertionReadAdapter",
        "RdfBlankNode",
        "RdfCapabilityReport",
        "RdfCodecConfiguration",
        "RdfCodecError",
        "RdfDataset",
        "RdfImportBudgetError",
        "RdfImportDocument",
        "RdfImportLimits",
        "RdfImportOwnership",
        "RdfImportSecurityError",
        "RdfIri",
        "RdfLiteral",
        "RdfOwnershipError",
        "RdfProjectionKind",
        "RdfTerm",
        "RdfTriple",
        "RdfTripleTerm",
        "RdfTypedQuery",
        "Sparql11AssertionReadAdapter",
        "Sparql12AssertionReadAdapter",
        "UnsupportedRdfCapabilityError",
    }
)

_INFERENCE_EXPORTS = frozenset(
    {
        "BoundedInferenceService",
        "ClosureState",
        "ClosureStatus",
        "DerivationExplanation",
        "InferenceError",
        "InferenceLimits",
        "InferenceProfile",
        "inference_limits_from_config",
        "inference_profile_from_config",
        "validate_inference_profile",
        "InferenceService",
        "MaterializationResult",
        "SemanticInferenceService",
    }
)

_SHACL_VALIDATION_EXPORTS = frozenset(
    {
        "DEFAULT_SHACL_WRITE_POLICY",
        "GovernedShaclValidationService",
        "ShaclCapabilityUnavailable",
        "ShaclSnapshotMismatch",
        "ShaclValidationError",
        "ShaclValidationLimits",
        "ShaclValidationReport",
        "ShaclWritePolicy",
        "ShapeSetReference",
        "ValidationFinding",
        "ValidationSeverity",
        "ValidationSource",
        "ValidationState",
        "ValidationWriteAction",
    }
)

_MAINTENANCE_EXPORTS = frozenset(
    {
        "SemanticMaintenanceError",
        "SemanticMaintenanceLimits",
        "SemanticMaintenanceResult",
        "SemanticMaintenanceService",
        "SemanticMaintenanceStatus",
        "SemanticMaintenanceTrainingReadiness",
        "maintenance_allows_prior_verified_snapshot",
        "maintenance_limits_from_config",
    }
)

_CAPABILITY_CONFIGURATION_EXPORTS = frozenset(
    {
        "CapabilitySelection",
        "SemanticCapabilityConfigurationError",
        "SemanticRuntimeCapabilities",
        "semantic_capabilities_from_config",
    }
)

_CORPUS_EXPORTS = frozenset(
    {
        "CORPUS_SCHEMA_VERSION",
        "CorpusCheckpoint",
        "CorpusEligibilityDecision",
        "CorpusEligibilityReason",
        "CorpusValidationStatus",
        "GovernedAssertionCorpusService",
        "GovernedCorpusBudgetExceeded",
        "GovernedCorpusDelta",
        "GovernedCorpusError",
        "GovernedCorpusExample",
        "GovernedCorpusLimits",
        "GovernedCorpusObservability",
        "GovernedCorpusPolicy",
        "GovernedCorpusSnapshot",
        "GovernedCorpusStorage",
        "GovernedCorpusTombstone",
        "GovernedCorpusUnavailable",
    }
)


def __getattr__(name: str):
    """Lazily expose optional boundaries so registry CLI execution stays clean."""
    if name in _REGISTRY_EXPORTS:
        from . import registry

        return getattr(registry, name)
    if name in _RDF_CODEC_EXPORTS:
        from . import rdf_codec

        return getattr(rdf_codec, name)
    if name in _MAINTENANCE_EXPORTS:
        from . import maintenance

        return getattr(maintenance, name)
    if name in _CAPABILITY_CONFIGURATION_EXPORTS:
        from . import capabilities

        return getattr(capabilities, name)
    if name in _CORPUS_EXPORTS:
        from . import corpus

        return getattr(corpus, name)
    if name in _INFERENCE_EXPORTS:
        from . import inference

        return getattr(inference, name)
    if name in _SHACL_VALIDATION_EXPORTS:
        from . import shacl_validation

        return getattr(shacl_validation, name)
    raise AttributeError(name)
