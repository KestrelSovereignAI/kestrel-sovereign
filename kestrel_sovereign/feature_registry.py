"""
Feature Registry — catalog loader and status resolver.

Loads the static capability catalog from data/feature_registry.toml and merges
it with runtime state (bundled core entries, installed entry_points, enabled
features) to produce a unified view for the Feature Store UI.

Consumed by CLI, API, and Feature Store UI.
"""

import importlib.metadata
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
    """Runtime status of a catalog entry."""

    AVAILABLE = "available"    # In registry, not installed / not bundled
    INSTALLED = "installed"    # Bundled or has entry_point, not enabled
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
    """Metadata for a feature/package/provider entry from the registry."""

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


@dataclass(frozen=True)
class InstalledFeatureRuntime:
    """Runtime metadata advertised by an installed feature distribution."""

    class_name: str
    entry_point: str
    distribution: str
    runtime: str = "in-process"
    # `service`: the runnable in the isolated venv — a console-script name (from
    # the service project's [project.scripts]) or a "module:func" callable.
    service: Optional[str] = None
    # `project`: install target for the venv (path or distribution). Defaults to
    # `distribution` when unset. Kept distinct from `service` so the runnable is
    # never mistaken for a `python -m` module / pip-install target.
    project: Optional[str] = None
    # `venv`: optional explicit venv-path override.
    venv: Optional[str] = None
    description: str = ""
    config_schema: Optional[Dict[str, Any]] = None


def load_registry(path: Optional[Path] = None) -> Dict[str, FeaturePackageInfo]:
    """
    Load the feature registry catalog from TOML.

    Args:
        path: Path to the registry TOML file. Defaults to the bundled catalog.

    Returns:
        Dict mapping catalog short name to FeaturePackageInfo.
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


def _entrypoint_class_name(ep_value: str, ep_name: str) -> str:
    if ep_value and ":" in ep_value:
        attr = ep_value.split(":", 1)[1].strip()
        if attr:
            return attr.split(".")[-1]
    return ep_name


def _read_distribution_pyproject(dist) -> Dict[str, Any]:
    """Read ``pyproject.toml`` from installed distribution files, if present."""
    candidates = ("pyproject.toml",)

    # Wheels may expose project files through the Distribution.files API.
    try:
        files = list(dist.files or [])
    except Exception:  # noqa: BLE001
        files = []

    for package_file in files:
        path_text = str(package_file)
        if Path(path_text).name not in candidates:
            continue
        try:
            path = dist.locate_file(package_file)
            if Path(path).is_file():
                with open(path, "rb") as f:
                    return tomllib.load(f)
        except Exception:  # noqa: BLE001
            continue

    # Some editable installs expose read_text for non-dist-info payload files.
    for candidate in candidates:
        try:
            text = dist.read_text(candidate)
        except Exception:  # noqa: BLE001
            text = None
        if text:
            try:
                return tomllib.loads(text)
            except Exception:  # noqa: BLE001
                return {}

    return {}


def _feature_runtime_from_pyproject(data: Dict[str, Any]) -> Dict[str, Any]:
    section = (
        data.get("tool", {})
        .get("kestrel", {})
        .get("feature", {})
    )
    return section if isinstance(section, dict) else {}


def discover_installed_feature_runtimes() -> Dict[str, InstalledFeatureRuntime]:
    """Return feature runtime metadata from entry-point distributions.

    This is intentionally metadata-only: it reads entry points and installed
    project metadata without importing feature modules. Packages that omit
    ``[tool.kestrel.feature]`` default to the historical in-process runtime.
    """
    discovered: Dict[str, InstalledFeatureRuntime] = {}
    try:
        eps = importlib.metadata.entry_points()
    except Exception:  # noqa: BLE001
        return discovered

    if hasattr(eps, "select"):
        feature_eps = eps.select(group=FEATURE_ENTRY_POINT_GROUP)
    else:
        feature_eps = eps.get(FEATURE_ENTRY_POINT_GROUP, [])

    for ep in feature_eps:
        class_name = _entrypoint_class_name(getattr(ep, "value", "") or "", ep.name)
        dist = getattr(ep, "dist", None)
        dist_name = getattr(dist, "name", "") if dist is not None else ""
        feature_meta: Dict[str, Any] = {}
        if dist is not None:
            feature_meta = _feature_runtime_from_pyproject(
                _read_distribution_pyproject(dist)
            )

        runtime = str(feature_meta.get("runtime") or "in-process").strip().lower()
        service = feature_meta.get("service")
        project = feature_meta.get("project")
        venv = feature_meta.get("venv")
        description = str(feature_meta.get("description") or "")
        config_schema = feature_meta.get("config_schema")
        if config_schema is not None and not isinstance(config_schema, dict):
            config_schema = None

        discovered[class_name] = InstalledFeatureRuntime(
            class_name=class_name,
            entry_point=getattr(ep, "value", "") or "",
            distribution=dist_name,
            runtime=runtime,
            service=str(service) if service else None,
            project=str(project) if project else None,
            venv=str(venv) if venv else None,
            description=description,
            config_schema=config_schema,
        )

    return discovered


def get_installed_feature_runtime(
    feature_class_name: str,
) -> Optional[InstalledFeatureRuntime]:
    """Look up installed runtime metadata for a feature class name."""
    return discover_installed_feature_runtimes().get(feature_class_name)


def resolve_status(
    registry: Dict[str, FeaturePackageInfo],
    enabled_class_names: Optional[Set[str]] = None,
) -> Dict[str, FeaturePackageInfo]:
    """
    Resolve runtime status for each catalog entry.

    Merges the static catalog with:
    - Bundled core entries or installed entry_points (marks as INSTALLED)
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

        # Check if the entry is bundled or installed via entry point.
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
        feature_class_name: The Feature class name (e.g. "VoiceFeature").
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
