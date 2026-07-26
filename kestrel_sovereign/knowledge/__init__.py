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
    "IDENTITY_VERSION",
    "IRI",
    "IRI_PROFILE",
    "Instant",
    "KnowledgeRegistryError",
    "LITERAL_PROFILE",
    "Lineage",
    "Literal",
    "LocalIdentifier",
    "MAPPING_SCHEMA_VERSION",
    "OntologyRef",
    "RDF_LANG_STRING",
    "ResolvedSemanticCapability",
    "Resource",
    "ResourceKind",
    "ResourceRequirement",
    "SemanticCapabilityContract",
    "SemanticKnowledgeRegistry",
    "SemanticResource",
    "SemanticVersion",
    "SourceOccurrence",
    "SourceProvenance",
    "StandardsMaturity",
    "TemporalInterval",
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
    "derive_assertion_id",
    "get_knowledge_registry",
    "identity_preimage",
    "lineage_from_mapping",
    "load_knowledge_registry",
    "normalize_iri",
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


def __getattr__(name: str):
    """Lazily expose registry exports so its CLI can run as a module."""
    if name not in _REGISTRY_EXPORTS:
        raise AttributeError(name)
    from . import registry

    return getattr(registry, name)
