"""Data models for GitHub feature - re-exported from kestrel-feature-github package.

This module exists for backward compatibility. The canonical implementation
lives in kestrel_feature_github.models.
"""
from kestrel_feature_github.models import (
    FileType,
    RepoFile,
    FileContent,
    SearchResult,
    CodeDefinition,
    ComponentManifest,
)

__all__ = [
    "FileType",
    "RepoFile",
    "FileContent",
    "SearchResult",
    "CodeDefinition",
    "ComponentManifest",
]
