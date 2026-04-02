"""
Feature Registry — catalog loader and status resolver.

Loads the static feature catalog from data/feature_registry.toml and merges
it with runtime state (installed entry_points, enabled features) to produce
a unified view of all available feature packages.

Consumed by CLI, API, and Feature Store UI.
"""

import importlib.metadata
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from kestrel_sovereign.features import (
    FEATURE_ENTRY_POINT_GROUP,
    get_disabled_features,
)

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent / "data" / "feature_registry.toml"


class FeatureStatus(str, Enum):
    """Runtime status of a feature package."""

    AVAILABLE = "available"    # In registry, not installed
    INSTALLED = "installed"    # Has entry_point, not enabled for this agent
    ENABLED = "enabled"        # Loaded and active
    DISABLED = "disabled"      # Explicitly disabled via env var


@dataclass
class SkillInfo:
    """Static skill declaration from the registry."""

    name: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)


@dataclass
class FeaturePackageInfo:
    """Metadata for a feature package from the registry."""

    name: str
    package: str
    git: str
    features: List[str]
    description: str
    tags: List[str] = field(default_factory=list)
    icon: str = ""
    core: bool = False
    skills: List[SkillInfo] = field(default_factory=list)
    status: FeatureStatus = FeatureStatus.AVAILABLE


def load_registry(path: Optional[Path] = None) -> Dict[str, FeaturePackageInfo]:
    """
    Load the feature registry catalog from TOML.

    Args:
        path: Path to the registry TOML file. Defaults to the bundled catalog.

    Returns:
        Dict mapping package short name to FeaturePackageInfo.
    """
    catalog_path = path or REGISTRY_PATH

    if not catalog_path.exists():
        logger.warning(f"Feature registry not found at {catalog_path}")
        return {}

    with open(catalog_path, "rb") as f:
        data = tomllib.load(f)

    registry: Dict[str, FeaturePackageInfo] = {}

    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue

        skills = []
        skills_data = entry.get("skills", {})
        for skill_name, skill_info in skills_data.items():
            if isinstance(skill_info, dict):
                skills.append(SkillInfo(
                    name=skill_name,
                    description=skill_info.get("description", ""),
                    category=skill_info.get("category", ""),
                    tags=skill_info.get("tags", []),
                ))

        registry[name] = FeaturePackageInfo(
            name=name,
            package=entry.get("package", ""),
            git=entry.get("git", ""),
            features=entry.get("features", []),
            description=entry.get("description", ""),
            tags=entry.get("tags", []),
            icon=entry.get("icon", ""),
            core=entry.get("core", False),
            skills=skills,
        )

    return registry


def _get_installed_entrypoint_classes() -> Set[str]:
    """Get the set of Feature class names available via installed entry_points."""
    installed: Set[str] = set()
    try:
        eps = importlib.metadata.entry_points()
    except Exception:
        return installed

    if hasattr(eps, "select"):
        feature_eps = eps.select(group=FEATURE_ENTRY_POINT_GROUP)
    else:
        feature_eps = eps.get(FEATURE_ENTRY_POINT_GROUP, [])

    for ep in feature_eps:
        installed.add(ep.name)

    return installed


def resolve_status(
    registry: Dict[str, FeaturePackageInfo],
    enabled_class_names: Optional[Set[str]] = None,
) -> Dict[str, FeaturePackageInfo]:
    """
    Resolve runtime status for each package in the registry.

    Merges the static catalog with:
    - Installed entry_points (marks packages as INSTALLED)
    - Currently enabled features (marks as ENABLED)
    - Disabled features env var (marks as DISABLED)

    Args:
        registry: The loaded registry catalog.
        enabled_class_names: Set of Feature class names currently loaded
            and active. If None, no features are marked as enabled.

    Returns:
        The same registry dict with status fields updated.
    """
    disabled = get_disabled_features()
    installed_ep_classes = _get_installed_entrypoint_classes()
    enabled = enabled_class_names or set()

    for info in registry.values():
        feature_classes = set(info.features)

        # Check if any feature class is disabled
        if feature_classes & disabled:
            info.status = FeatureStatus.DISABLED
            continue

        # Check if any feature class is currently enabled
        if feature_classes & enabled:
            info.status = FeatureStatus.ENABLED
            continue

        # Check if the package is installed (core packages or entry_points)
        if info.core or (feature_classes & installed_ep_classes):
            info.status = FeatureStatus.INSTALLED
            continue

        info.status = FeatureStatus.AVAILABLE

    return registry


def get_registry(
    enabled_class_names: Optional[Set[str]] = None,
    path: Optional[Path] = None,
) -> Dict[str, FeaturePackageInfo]:
    """
    Load the feature registry and resolve status in one call.

    Args:
        enabled_class_names: Set of Feature class names currently active.
        path: Optional override for the registry TOML path.

    Returns:
        Dict mapping package short name to FeaturePackageInfo with status resolved.
    """
    registry = load_registry(path)
    return resolve_status(registry, enabled_class_names)


def get_package_for_feature(feature_class_name: str, path: Optional[Path] = None) -> Optional[FeaturePackageInfo]:
    """
    Look up which package a feature class belongs to.

    Args:
        feature_class_name: The Feature class name (e.g. "RunPodFeature").
        path: Optional override for the registry TOML path.

    Returns:
        The FeaturePackageInfo or None if not found.
    """
    registry = load_registry(path)
    for info in registry.values():
        if feature_class_name in info.features:
            return info
    return None


def get_all_skills(path: Optional[Path] = None) -> List[SkillInfo]:
    """
    Get all skill declarations across all packages.

    Args:
        path: Optional override for the registry TOML path.

    Returns:
        Flat list of all SkillInfo entries.
    """
    registry = load_registry(path)
    skills = []
    for info in registry.values():
        skills.extend(info.skills)
    return skills


def get_skills_for_package(package_name: str, path: Optional[Path] = None) -> List[SkillInfo]:
    """
    Get skill declarations for a specific package.

    Args:
        package_name: The short name (e.g. "cloud", "voice").
        path: Optional override for the registry TOML path.

    Returns:
        List of SkillInfo entries for that package.
    """
    registry = load_registry(path)
    info = registry.get(package_name)
    if info is None:
        return []
    return info.skills
