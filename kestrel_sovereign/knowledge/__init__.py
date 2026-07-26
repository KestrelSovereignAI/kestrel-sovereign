"""Offline, versioned semantic knowledge contracts and package resources."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
]


def __getattr__(name: str):
    """Expose registry types without importing its module during ``-m`` startup."""
    if name not in __all__:
        raise AttributeError(name)
    from . import registry

    return getattr(registry, name)
