"""
Feature Registry — catalog loader and status resolver.

Loads the static capability catalog from data/feature_registry.toml and merges
it with runtime state (bundled entries, installed extension entry points or
standalone distributions, enabled features) to produce a unified view for the
Feature Store UI.

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

# Every entry-point group that can back a registry-listed extension package.
# Keep the legacy voice group for already-published providers, while treating
# the feature-owned group as canonical for current TTS/STT packages.
EXTENSION_ENTRY_POINT_GROUPS = (
    FEATURE_ENTRY_POINT_GROUP,
    "kestrel_sovereign.host_features",
    "kestrel_sovereign.cloud_providers",
    "kestrel_feature_voice_providers",
    "kestrel_sovereign.voice_providers",
    "kestrel_sovereign.conversation_providers",
    "kestrel_sovereign.storage_providers",
)


class FeatureStatus(str, Enum):
    """Runtime status of a catalog entry."""

    AVAILABLE = "available"    # In registry, not installed / not bundled
    INSTALLED = "installed"    # Bundled/detected install, not enabled
    ENABLED = "enabled"        # Loaded and active
    DISABLED = "disabled"      # Explicitly disabled via env var


class PackageBoundary(str, Enum):
    """Ownership and runtime contract for one registry catalog row."""

    BUNDLED = "bundled"
    BUNDLED_COMPONENT = "bundled-component"
    FEATURE_PACKAGE = "feature-package"
    PROVIDER_PACKAGE = "provider-package"
    STANDALONE_TOOL = "standalone-tool"


class RegistryValidationError(ValueError):
    """The static registry contains contradictory package-boundary metadata."""


@dataclass
class SkillInfo:
    """Static skill declaration from the registry."""

    name: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)


@dataclass
class FeaturePackageInfo:
    """Metadata for a boundary-typed capability/install-target catalog row."""

    name: str
    package: str
    git: str
    features: List[str]
    description: str
    tags: List[str] = field(default_factory=list)
    icon: str = ""
    core: bool = False
    boundary: Optional[PackageBoundary] = None
    bundled_components: List[str] = field(default_factory=list)
    provider_classes: List[str] = field(default_factory=list)
    entry_point_groups: List[str] = field(default_factory=list)
    command: Optional[str] = None
    provider_for: Optional[str] = None
    companion: Optional[str] = None
    skills: List[SkillInfo] = field(default_factory=list)
    status: FeatureStatus = FeatureStatus.AVAILABLE
    # Frontend capability keys this feature's UI panels gate on (#2041). The
    # registry short name and the frontend ``hasCapability(...)`` key usually
    # match, but some diverge — ``observability`` backs the ``metrics`` panel,
    # ``security`` backs both ``audit`` and ``permissions``. When set, these keys
    # are what ``compute_feature_capabilities`` emits instead of the short name,
    # so the derived capability set lines up with ``PANEL_CAPABILITIES``. Empty
    # means "use the short name".
    ui_capabilities: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep direct construction in downstream code backward compatible.
        # Registry TOML rows themselves must declare ``boundary`` explicitly
        # and are validated by ``load_registry``.
        if self.boundary is None:
            self.boundary = (
                PackageBoundary.BUNDLED
                if self.core
                else PackageBoundary.FEATURE_PACKAGE
            )

    @property
    def runtime_components(self) -> List[str]:
        """Names advertised by this row's lifecycle/provider runtime."""
        return [
            *self.features,
            *self.bundled_components,
            *self.provider_classes,
        ]

    @property
    def installable(self) -> bool:
        """Whether ``package`` is an independent install target."""
        return self.boundary not in {
            PackageBoundary.BUNDLED,
            PackageBoundary.BUNDLED_COMPONENT,
        }


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


def validate_registry(
    registry: Dict[str, FeaturePackageInfo],
    *,
    bundled_feature_classes: Optional[Set[str]] = None,
) -> None:
    """Reject contradictory ownership metadata.

    ``bundled_feature_classes`` is optional so normal catalog reads stay
    metadata-only. Tests and inventory checks pass the actual result of
    ``discover_local_feature_class_names()`` to enforce exact agreement with
    the in-tree discovery contract.
    """
    errors: List[str] = []
    bundled_declared: Set[str] = set()

    for name, info in registry.items():
        boundary = info.boundary
        if not info.package:
            errors.append(f"{name}: package must name the owning distribution")
        expected_core = boundary in {
            PackageBoundary.BUNDLED,
            PackageBoundary.BUNDLED_COMPONENT,
        }
        if info.core is not expected_core:
            errors.append(
                f"{name}: core must be {str(expected_core).lower()} when "
                f"boundary = {boundary.value!r}"
            )

        if boundary is PackageBoundary.BUNDLED:
            if info.package != "kestrel-sovereign":
                errors.append(
                    f"{name}: bundled rows must be owned by "
                    "package = 'kestrel-sovereign'"
                )
            if not info.features:
                errors.append(f"{name}: bundled rows must declare Feature classes")
            if (
                info.bundled_components
                or info.provider_classes
                or info.entry_point_groups
                or info.command
            ):
                errors.append(
                    f"{name}: bundled rows cannot declare provider or standalone fields"
                )
            bundled_declared.update(info.features)
        elif boundary is PackageBoundary.BUNDLED_COMPONENT:
            if info.package != "kestrel-sovereign":
                errors.append(
                    f"{name}: bundled-component rows must be owned by "
                    "package = 'kestrel-sovereign'"
                )
            if info.features:
                errors.append(
                    f"{name}: non-Feature core objects belong in "
                    "'bundled_components', not 'features'"
                )
            if not info.bundled_components:
                errors.append(
                    f"{name}: bundled-component rows must declare bundled_components"
                )
            if info.provider_classes or info.entry_point_groups or info.command:
                errors.append(
                    f"{name}: bundled-component rows cannot declare provider "
                    "or standalone fields"
                )
        elif boundary is PackageBoundary.FEATURE_PACKAGE:
            if info.package == "kestrel-sovereign":
                errors.append(
                    f"{name}: feature-package rows require an external package"
                )
            if not info.features:
                errors.append(
                    f"{name}: feature-package rows must declare Feature classes"
                )
            if (
                info.bundled_components
                or info.provider_classes
                or info.entry_point_groups
                or info.command
            ):
                errors.append(
                    f"{name}: feature-package rows cannot declare provider or "
                    "standalone fields"
                )
        elif boundary is PackageBoundary.PROVIDER_PACKAGE:
            if info.package == "kestrel-sovereign":
                errors.append(
                    f"{name}: provider-package rows require an external package"
                )
            if info.features or info.bundled_components:
                errors.append(
                    f"{name}: provider implementations belong in "
                    "'provider_classes', not 'features'"
                )
            if not info.provider_classes:
                errors.append(
                    f"{name}: provider-package rows must declare provider_classes"
                )
            if not info.entry_point_groups:
                errors.append(
                    f"{name}: provider-package rows must declare entry_point_groups"
                )
            invalid_groups = sorted(
                set(info.entry_point_groups) - set(EXTENSION_ENTRY_POINT_GROUPS)
            )
            if invalid_groups:
                errors.append(
                    f"{name}: unknown provider entry-point groups: "
                    + ", ".join(invalid_groups)
                )
            if FEATURE_ENTRY_POINT_GROUP in info.entry_point_groups:
                errors.append(
                    f"{name}: provider packages cannot use the Feature lifecycle "
                    "entry-point group"
                )
            if info.command:
                errors.append(
                    f"{name}: provider-package rows cannot declare a command"
                )
        elif boundary is PackageBoundary.STANDALONE_TOOL:
            if info.package == "kestrel-sovereign":
                errors.append(
                    f"{name}: standalone-tool rows require an external package"
                )
            if (
                info.features
                or info.bundled_components
                or info.provider_classes
                or info.entry_point_groups
            ):
                errors.append(
                    f"{name}: standalone tools cannot declare lifecycle or "
                    "provider entry points"
                )
            if not info.command:
                errors.append(
                    f"{name}: standalone-tool rows must declare their command"
                )

        if info.provider_for:
            target = registry.get(info.provider_for)
            if boundary is not PackageBoundary.PROVIDER_PACKAGE:
                errors.append(
                    f"{name}: provider_for is valid only on provider-package rows"
                )
            if target is None:
                errors.append(
                    f"{name}: provider_for references unknown row "
                    f"{info.provider_for!r}"
                )
            elif target.boundary not in {
                PackageBoundary.BUNDLED,
                PackageBoundary.FEATURE_PACKAGE,
            }:
                errors.append(
                    f"{name}: provider_for must reference a Feature lifecycle row"
                )
        if info.companion:
            target = registry.get(info.companion)
            if target is None:
                errors.append(
                    f"{name}: companion references unknown row "
                    f"{info.companion!r}"
                )
            elif target.boundary is not PackageBoundary.STANDALONE_TOOL:
                errors.append(
                    f"{name}: companion must reference a standalone-tool row"
                )

    if bundled_feature_classes is not None:
        missing = sorted(bundled_feature_classes - bundled_declared)
        stale = sorted(bundled_declared - bundled_feature_classes)
        if missing:
            errors.append(
                "bundled discovery classes missing from registry: "
                + ", ".join(missing)
            )
        if stale:
            errors.append(
                "registry classes not discoverable in-tree: " + ", ".join(stale)
            )

    if errors:
        raise RegistryValidationError(
            "Invalid feature registry package boundaries:\n- "
            + "\n- ".join(errors)
        )


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
        if "boundary" not in entry:
            raise RegistryValidationError(
                f"{name}: missing required package-boundary field 'boundary'"
            )
        if "core" not in entry:
            raise RegistryValidationError(
                f"{name}: missing required compatibility field 'core'"
            )
        try:
            boundary = PackageBoundary(entry["boundary"])
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in PackageBoundary)
            raise RegistryValidationError(
                f"{name}: boundary must be one of: {choices}"
            ) from exc

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
            boundary=boundary,
            bundled_components=entry.get("bundled_components", []),
            provider_classes=entry.get("provider_classes", []),
            entry_point_groups=entry.get("entry_point_groups", []),
            command=entry.get("command"),
            provider_for=entry.get("provider_for"),
            companion=entry.get("companion"),
            skills=skills,
            ui_capabilities=entry.get("ui_capabilities", []),
        )

    validate_registry(registry)
    return registry


def _entrypoint_class_name(ep_value: str, ep_name: str) -> str:
    """Return the implementation class advertised by an entry point."""
    if ep_value and ":" in ep_value:
        attr = ep_value.split(":", 1)[1].strip()
        if attr:
            return attr.split(".")[-1]
    return ep_name


def iter_extension_entry_points():
    """Yield ``(group, entry_point)`` for every installed Kestrel extension.

    This metadata-only iterator is the shared source of truth for registry
    status and CLI upgrade/capture discovery. It intentionally does not import
    extension modules.
    """
    try:
        eps = importlib.metadata.entry_points()
    except Exception:  # noqa: BLE001
        return

    for group in EXTENSION_ENTRY_POINT_GROUPS:
        if hasattr(eps, "select"):
            group_eps = eps.select(group=group)
        else:
            group_eps = eps.get(group, [])
        for ep in group_eps:
            yield group, ep


def _get_installed_entrypoint_classes() -> Set[str]:
    """Get implementation classes exposed by installed extension packages."""
    installed: Set[str] = set()
    for _group, ep in iter_extension_entry_points():
        installed.add(
            _entrypoint_class_name(getattr(ep, "value", "") or "", ep.name)
        )

    return installed


def _get_installed_entrypoint_classes_by_group() -> Dict[str, Set[str]]:
    """Map each extension entry-point group to advertised implementation names."""
    installed: Dict[str, Set[str]] = {}
    for group, ep in iter_extension_entry_points():
        installed.setdefault(group, set()).add(
            _entrypoint_class_name(getattr(ep, "value", "") or "", ep.name)
        )
    return installed


def _is_distribution_installed(package: str) -> bool:
    """Return whether an independently installed distribution is present."""
    try:
        importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


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
    - Bundled entries, installed extension entry points, or installed
      standalone distributions (marks as INSTALLED)
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
    installed_ep_classes_by_group = (
        _get_installed_entrypoint_classes_by_group()
    )
    enabled = enabled_class_names or set()

    for info in registry.values():
        feature_classes = set(info.features)
        runtime_components = set(info.runtime_components)

        # Check if any feature class is disabled
        if feature_classes & disabled:
            info.status = FeatureStatus.DISABLED
            continue

        # Check if any feature class is currently enabled
        if feature_classes & enabled:
            info.status = FeatureStatus.ENABLED
            continue

        # Bundled lifecycle modules need no external entry point. Feature and
        # provider packages count as installed only when their declared runtime
        # component is advertised. Standalone tools have no Kestrel entry point,
        # so distribution metadata is their installation evidence.
        entrypoint_installed = bool(
            runtime_components & installed_ep_classes
        )
        if info.boundary is PackageBoundary.PROVIDER_PACKAGE:
            provider_group_classes: Set[str] = set()
            for group in info.entry_point_groups:
                provider_group_classes.update(
                    installed_ep_classes_by_group.get(group, set())
                )
            entrypoint_installed = bool(
                set(info.provider_classes) & provider_group_classes
            )

        installed = (
            info.boundary in {
                PackageBoundary.BUNDLED,
                PackageBoundary.BUNDLED_COMPONENT,
            }
            or entrypoint_installed
            or (
                info.boundary is PackageBoundary.STANDALONE_TOOL
                and _is_distribution_installed(info.package)
            )
        )
        if installed:
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
